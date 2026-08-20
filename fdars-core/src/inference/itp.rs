//! Interval Testing Procedure (ITP) family for functional hypothesis testing.
//!
//! Implements the ITP as defined in the CRAN `fdatest` 2.1.1 package
//! (`ITP1bspline`, `ITP2bspline`, `ITPlmbspline`), matching the algorithm of
//! Pini & Vantini (2016, Biometrics). The ITP projects functional observations
//! onto a finite basis, runs per-component univariate permutation tests, and
//! applies an interval-wise closure adjustment so that the adjusted p-value for
//! basis component `k` equals the maximum joint permutation p-value over all
//! contiguous intervals containing `k`.
//!
//! # References
//!
//! * Pini, A. & Vantini, S. (2016). Interval-wise testing for functional data.
//!   *Biometrics*, 73(3), 835–845. <https://doi.org/10.1111/biom.12679>
//! * CRAN `fdatest` 2.1.1 — <https://cran.r-project.org/package=fdatest>

use crate::basis::projection::{fdata_to_basis, ProjectionBasisType};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─────────────────────────────────────────────────────────────────────────────
// Public result type
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an Interval Testing Procedure (ITP) family test.
///
/// Provides per-basis-component raw and adjusted p-values. The adjusted
/// p-values implement the interval-wise closure adjustment (Pini & Vantini,
/// Biometrics 2016): `adjusted_pvalues[k]` is the maximum over all contiguous
/// intervals `[a, b]` containing `k` of the joint permutation p-value for that
/// interval. A small adjusted p-value at component `k` indicates that the
/// domain sub-interval represented by basis function `k` contributes to the
/// rejection of H₀.
///
/// **Note on `n_basis`:** For B-spline bases, the actual number of basis
/// functions used may differ from the requested `nbasis` due to knot clamping.
/// Always read `n_basis` from `ItpResult` rather than the argument passed to
/// the entry point.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ItpResult {
    /// Adjusted (interval-wise closed) p-values, one per basis component.
    pub adjusted_pvalues: Vec<f64>,
    /// Raw (point-wise) permutation p-values, one per basis component.
    /// Uses the `(n_ge + 1) / (n_perm + 1)` correction (avoids zero p-values;
    /// deliberate divergence from the R `fdatest` convention of `n_ge / B`).
    pub raw_pvalues: Vec<f64>,
    /// Basis type used for projection.
    pub basis_type: ProjectionBasisType,
    /// Number of basis functions actually used (may differ from requested for
    /// B-splines due to knot clamping).
    pub n_basis: usize,
    /// Number of permutations used.
    pub n_perm: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Private closure-adjustment helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Rank-transform the permutation statistic matrix to pseudo-p-values.
///
/// For each component `k`, ranks all `b` permutation statistics in descending
/// order. The rank-based pseudo-p for permutation `i` at component `k` is
/// `rank_desc / b`, so the largest stat gets `1/b` (smallest pseudo-p) and the
/// smallest stat gets `b/b = 1.0` (largest pseudo-p).
///
/// Returns `L` of shape `(b, p)`.
fn rank_transform(t_perm: &[Vec<f64>], p: usize, b: usize) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0f64; p]; b];
    let assignments: Vec<Vec<(usize, f64)>> = iter_maybe_parallel!(0..p)
        .map(|k| {
            let mut col: Vec<(f64, usize)> = (0..b).map(|i| (t_perm[i][k], i)).collect();
            // Sort descending: largest stat → rank 1 → smallest pseudo-p (1/b)
            col.sort_unstable_by(|a, b_| {
                b_.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            col.iter()
                .enumerate()
                .map(|(rank_zero_based, &(_, orig_idx))| {
                    (orig_idx, (rank_zero_based + 1) as f64 / b as f64)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    for (k, col_assignments) in assignments.iter().enumerate() {
        for &(orig_idx, pseudo_p) in col_assignments {
            l[orig_idx][k] = pseudo_p;
        }
    }
    l
}

/// Fisher's combining function: `-2 * Σ log(max(v, 1e-300))`.
///
/// Clamps each p-value to at least `1e-300` before taking the log, preventing
/// `-inf` / NaN when a raw p-value is exactly 0.0 (T-30-02).
#[inline]
fn fisher_cf(vals: &[f64]) -> f64 {
    -2.0 * vals.iter().map(|&v| v.max(1e-300).ln()).sum::<f64>()
}

/// Build the asymmetric interval p-value matrix via the O(p²) interval loop.
///
/// `pval_matrix[row][col]` holds the joint permutation p-value for the
/// contiguous interval whose length is `p - row`, starting at column `col`.
///
/// * Row `p-1` (R's row `p`): raw per-component p-values (length-1 intervals).
/// * Row `row_idx = p - interval_len` (R's row `i = p - interval_len`): joint
///   p-values for all length-`interval_len` contiguous intervals.
///
/// The circular "wrap-around" is implemented by doubling both the raw p-value
/// vector and the `L` matrix (circular trick from the R source).
///
/// **Raw p-value divergence:** `pval_matrix` for interval rows uses
/// `n_ge / n_perm` (no +1 correction), matching the R source for the internal
/// closure matrix. Only the top-level `raw_pvalues` field in `ItpResult` uses
/// the `(n_ge + 1) / (n_perm + 1)` correction.
fn build_pval_matrix(
    raw_pvalues: &[f64],
    l: &[Vec<f64>],
    p: usize,
    n_perm: usize,
) -> Vec<Vec<f64>> {
    let mut mat = vec![vec![1.0f64; p]; p];

    // Last row: raw p-values (length-1 intervals)
    mat[p - 1][..p].copy_from_slice(&raw_pvalues[..p]);

    // Doubled arrays for the circular wrap-around
    let pval_2x: Vec<f64> = raw_pvalues
        .iter()
        .chain(raw_pvalues.iter())
        .copied()
        .collect();
    let l_2x: Vec<Vec<f64>> = l
        .iter()
        .map(|row| row.iter().chain(row.iter()).copied().collect())
        .collect();

    // interval_len = 2..=p  (R's i from p-1 down to 1)
    for interval_len in 2..=p {
        let row_idx = p - interval_len; // R's row i = p - interval_len
        for j in 0..p {
            let inf = j; // 0-indexed start in the 2x array
            let sup = j + interval_len; // exclusive end
            let t0_temp = fisher_cf(&pval_2x[inf..sup]);
            let n_ge = l_2x
                .iter()
                .filter(|perm_row| fisher_cf(&perm_row[inf..sup]) >= t0_temp)
                .count();
            mat[row_idx][j] = n_ge as f64 / n_perm as f64;
        }
    }
    mat
}

/// Compute interval-wise closure-adjusted p-values.
///
/// For each basis component `k`, the adjusted p-value is the maximum over all
/// contiguous intervals `[a, b]` (where `a ≤ k ≤ b`) of the joint p-value for
/// that interval. This implements the `pval.correct` function from the CRAN
/// `fdatest` package (Pini & Vantini 2016 closure), matching
/// `fdatest::ITP1bspline`'s `pval.correct`.
///
/// **Implementation note:** The R source doubles and reverses the column-doubled
/// matrix (`matrice_pval_2_2x <- matrice_pval_2_2x[, (2*p):1]`) before the
/// cone walk, then reverses the output (`corrected.pval[p:1]`). The reversal
/// restores natural component order (component 0, 1, …, p-1 matching the
/// coefficient matrix columns).
///
/// **Raw-p divergence:** The `(n_ge + 1)/(n_perm + 1)` correction used in
/// `itp_one_pop`'s `raw_pvalues` avoids zero p-values and is a deliberate
/// deviation from R's `/B` convention (RESEARCH Assumption A4).
fn pval_correct(pval_matrix: &[Vec<f64>], p: usize) -> Vec<f64> {
    // `get_2x_rev(row, col)`: column `col` in the doubled+reversed (2p-wide)
    // matrix maps to original column `(2*p - 1 - col) % p`.
    //
    // Equivalent to R's:
    //   matrice_pval_2_2x <- cbind(pval.matrix, pval.matrix)
    //   matrice_pval_2_2x <- matrice_pval_2_2x[, (2*p):1]   # reverse columns
    let get_2x_rev = |row: usize, col: usize| -> f64 {
        let orig_col = (2 * p - 1).saturating_sub(col) % p;
        pval_matrix[row][orig_col]
    };

    let mut corrected = vec![0.0f64; p];
    for var in 0..p {
        // R: pval_var <- matrice_pval_2_2x[p, var]  (1-indexed row p = our row p-1)
        let mut pval_var = get_2x_rev(p - 1, var);
        let mut fine = var;
        // R: for riga in (p-1):1 → our riga_idx = p-2 down to 0
        for riga_idx in (0..p - 1).rev() {
            fine += 1;
            // R: pval_cono <- matrice_pval_2_2x[riga, inizio:fine]
            for col in var..=fine {
                let v = get_2x_rev(riga_idx, col);
                if v > pval_var {
                    pval_var = v;
                }
            }
        }
        corrected[var] = pval_var;
    }
    // R: corrected.pval <- corrected.pval[p:1]  (reverse to natural component order)
    corrected.reverse();
    corrected
}

// ─────────────────────────────────────────────────────────────────────────────
// One-population entry point helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Subtract `mu0` elementwise from each row of `data` if `Some`; otherwise
/// return a reference-compatible clone. Returns `Err` if `mu0.len() != m`.
fn center_one_pop(data: &FdMatrix, mu0: Option<&[f64]>) -> Result<FdMatrix, FdarError> {
    let (n, m) = data.shape();
    match mu0 {
        None => Ok(data.clone()),
        Some(mu) => {
            if mu.len() != m {
                return Err(FdarError::InvalidDimension {
                    parameter: "mu0",
                    expected: format!("{m} elements (matching data columns)"),
                    actual: format!("{} elements", mu.len()),
                });
            }
            let mut centered = FdMatrix::zeros(n, m);
            for j in 0..m {
                for i in 0..n {
                    centered[(i, j)] = data[(i, j)] - mu[j];
                }
            }
            Ok(centered)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point: one-population ITP
// ─────────────────────────────────────────────────────────────────────────────

/// Interval-wise one-population test (sign-flip permutation).
///
/// Tests H₀: the mean function equals `mu0` (or zero if `None`). Projects the
/// (possibly centred) functional data onto `nbasis` basis functions, then runs
/// an interval-wise closure test on the basis coefficients.
///
/// The test statistic per basis component `k` is `|colMean(coeff[:, k])|`.
/// The permutation null is sign-flip: each curve's coefficient row is
/// multiplied by an i.i.d. ±1 Bernoulli draw.
///
/// Matches `fdatest::ITP1bspline` up to the `(n_ge + 1) / (n_perm + 1)`
/// p-value correction (R uses `n_ge / B` without the +1).
///
/// # Arguments
///
/// * `data` — functional data matrix, shape `(n, m)` (column-major).
/// * `argvals` — evaluation points, length `m`.
/// * `mu0` — null mean function, length `m`; `None` = zero mean.
/// * `basis_type` — `ProjectionBasisType::Bspline` or `::Fourier`.
/// * `nbasis` — requested number of basis functions (≥ 2). For B-splines the
///   actual count may be lower due to knot clamping; use `result.n_basis`.
/// * `n_perm` — number of sign-flip permutations (≥ 1).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// * `InvalidDimension` — if `n < 2`, `argvals.len() != m`, or `mu0.len() != m`.
/// * `InvalidParameter` — if `nbasis < 2`, `n_perm == 0`, or basis projection fails.
#[must_use = "the ItpResult contains the adjusted p-values"]
pub fn itp_one_pop(
    data: &FdMatrix,
    argvals: &[f64],
    mu0: Option<&[f64]>,
    basis_type: ProjectionBasisType,
    nbasis: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ItpResult, FdarError> {
    // 1. Validate inputs
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows (observations)".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if nbasis < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: "must be >= 2".to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    // 2. Subtract mu0 (if provided), then project to basis coefficients
    let centered = center_one_pop(data, mu0)?;
    let proj = fdata_to_basis(&centered, argvals, nbasis, basis_type).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("basis projection failed (nbasis={nbasis}, m={m})"),
        }
    })?;
    let coeff = proj.coefficients; // FdMatrix shape (n, p)
    let p = proj.n_basis; // actual basis count (may differ from nbasis for B-spline)

    // 3. Observed per-component statistic: |colMean(coeff[:, k])|
    let t0: Vec<f64> = (0..p)
        .map(|k| {
            let mean_k = (0..n).map(|i| coeff[(i, k)]).sum::<f64>() / n as f64;
            mean_k.abs()
        })
        .collect();

    // 4. Sign-flip permutation loop → t_perm (n_perm, p)
    // Single sequential loop: one RNG drives all permutations in order.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut t_perm: Vec<Vec<f64>> = Vec::with_capacity(n_perm);
    for _ in 0..n_perm {
        use rand::Rng;
        let signs: Vec<f64> = (0..n)
            .map(|_| if rng.gen::<bool>() { 1.0 } else { -1.0 })
            .collect();
        let row: Vec<f64> = (0..p)
            .map(|k| {
                let mean_k = (0..n).map(|i| coeff[(i, k)] * signs[i]).sum::<f64>() / n as f64;
                mean_k.abs()
            })
            .collect();
        t_perm.push(row);
    }

    // 5. Raw per-component p-values (INF-01 convention: +1 correction)
    let raw_pvalues: Vec<f64> = (0..p)
        .map(|k| {
            let n_ge = t_perm.iter().filter(|row| row[k] >= t0[k]).count();
            (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
        })
        .collect();

    // 6. Rank-transform → L matrix (n_perm, p)
    let l = rank_transform(&t_perm, p, n_perm);

    // 7. Build O(p²) interval p-value matrix
    let pval_matrix = build_pval_matrix(&raw_pvalues, &l, p, n_perm);

    // 8. Closure max-adjustment
    let adjusted_pvalues = pval_correct(&pval_matrix, p);

    Ok(ItpResult {
        adjusted_pvalues,
        raw_pvalues,
        basis_type,
        n_basis: p,
        n_perm,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Two-population entry point helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Validate two-population inputs; return `(n_a, n_b, m)` or `FdarError`.
fn validate_two_samples_itp(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize, usize), FdarError> {
    let (n_a, m_a) = data_a.shape();
    let (n_b, m_b) = data_b.shape();
    if m_a == 0 || m_b == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column (grid points)".to_string(),
            actual: format!("data_a has {m_a} columns, data_b has {m_b} columns"),
        });
    }
    if m_a != m_b {
        return Err(FdarError::InvalidDimension {
            parameter: "data_b",
            expected: format!("{m_a} columns (matching data_a)"),
            actual: format!("{m_b} columns"),
        });
    }
    if argvals.len() != m_a {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m_a} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if n_a < 2 || n_b < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows per sample".to_string(),
            actual: format!("data_a has {n_a} rows, data_b has {n_b} rows"),
        });
    }
    Ok((n_a, n_b, m_a))
}

/// Pool two coefficient matrices (shape `(n_a, p)` and `(n_b, p)`) into one
/// `(n_a + n_b, p)` matrix. Rows 0..n_a come from `coeff_a`, rows n_a.. from `coeff_b`.
fn pool_coefficients_itp(
    coeff_a: &FdMatrix,
    coeff_b: &FdMatrix,
    n_a: usize,
    n_b: usize,
    p: usize,
) -> FdMatrix {
    let mut pooled = FdMatrix::zeros(n_a + n_b, p);
    for k in 0..p {
        for i in 0..n_a {
            pooled[(i, k)] = coeff_a[(i, k)];
        }
        for i in 0..n_b {
            pooled[(n_a + i, k)] = coeff_b[(i, k)];
        }
    }
    pooled
}

/// Fisher–Yates in-place shuffle of an index vector (7-line copy of
/// `permutation::shuffle_labels`, which is private to that module).
fn shuffle_itp(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point: two-population ITP
// ─────────────────────────────────────────────────────────────────────────────

/// Interval-wise two-population test (pool + relabel permutation).
///
/// Tests H₀: the mean functions of groups A and B are equal. Projects both
/// groups onto `nbasis` basis functions, pools the resulting coefficient
/// matrices `(n_a + n_b, p)`, and runs an interval-wise closure test on the
/// per-component mean-difference statistic.
///
/// The test statistic per basis component `k` is
/// `|colMean(coeff_a[:, k]) - colMean(coeff_b[:, k])|`.  The permutation null
/// relabels the pooled coefficient rows via Fisher–Yates (inline copy of the
/// `permutation::shuffle_labels` pattern).
///
/// Matches `fdatest::ITP2bspline` up to the `(n_ge + 1) / (n_perm + 1)`
/// p-value correction (R uses `n_ge / B` without the +1).
///
/// # Arguments
///
/// * `data_a` — functional data matrix for group A, shape `(n_a, m)`.
/// * `data_b` — functional data matrix for group B, shape `(n_b, m)`.
/// * `argvals` — evaluation points, length `m`.
/// * `basis_type` — `ProjectionBasisType::Bspline` or `::Fourier`.
/// * `nbasis` — requested number of basis functions (≥ 2).
/// * `n_perm` — number of relabel permutations (≥ 1).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// * `InvalidDimension` — if `n_a < 2 || n_b < 2`, `m_a != m_b`, or `argvals.len() != m`.
/// * `InvalidParameter` — if `nbasis < 2`, `n_perm == 0`, or basis projection fails.
#[must_use = "the ItpResult contains the adjusted p-values"]
pub fn itp_two_pop(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    basis_type: ProjectionBasisType,
    nbasis: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ItpResult, FdarError> {
    // 1. Validate inputs
    let (n_a, n_b, m) = validate_two_samples_itp(data_a, data_b, argvals)?;
    if nbasis < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: "must be >= 2".to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    // 2. Project each group to basis coefficients
    let proj_a = fdata_to_basis(data_a, argvals, nbasis, basis_type).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("basis projection failed for data_a (nbasis={nbasis}, m={m})"),
        }
    })?;
    let proj_b = fdata_to_basis(data_b, argvals, nbasis, basis_type).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("basis projection failed for data_b (nbasis={nbasis}, m={m})"),
        }
    })?;
    let p = proj_a.n_basis; // actual basis count (clamp-safe)
    let coeff_a = proj_a.coefficients; // (n_a, p)
    let coeff_b = proj_b.coefficients; // (n_b, p)

    // 3. Pool coefficient rows into (n, p)
    let pooled = pool_coefficients_itp(&coeff_a, &coeff_b, n_a, n_b, p);
    let n = n_a + n_b;

    // 4. Observed per-component statistic: |colMean(a) - colMean(b)|
    let t0: Vec<f64> = (0..p)
        .map(|k| {
            let m_a = (0..n_a).map(|i| pooled[(i, k)]).sum::<f64>() / n_a as f64;
            let m_b = (n_a..n).map(|i| pooled[(i, k)]).sum::<f64>() / n_b as f64;
            (m_a - m_b).abs()
        })
        .collect();

    // 5. Pool + relabel permutation loop → t_perm (n_perm, p)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm_idx: Vec<usize> = (0..n).collect();
    let mut t_perm: Vec<Vec<f64>> = Vec::with_capacity(n_perm);
    for _ in 0..n_perm {
        shuffle_itp(&mut perm_idx, &mut rng);
        let row: Vec<f64> = (0..p)
            .map(|k| {
                let m_a = (0..n_a).map(|r| pooled[(r, k)]).sum::<f64>() / n_a as f64;
                let m_b = (n_a..n).map(|i| pooled[(perm_idx[i], k)]).sum::<f64>() / n_b as f64;
                (m_a - m_b).abs()
            })
            .collect();
        t_perm.push(row);
    }

    // 6. Raw per-component p-values (+1 correction)
    let raw_pvalues: Vec<f64> = (0..p)
        .map(|k| {
            let n_ge = t_perm.iter().filter(|row| row[k] >= t0[k]).count();
            (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
        })
        .collect();

    // 7–9. Rank-transform → pval_matrix → closure adjustment
    let l = rank_transform(&t_perm, p, n_perm);
    let pval_matrix = build_pval_matrix(&raw_pvalues, &l, p, n_perm);
    let adjusted_pvalues = pval_correct(&pval_matrix, p);

    Ok(ItpResult {
        adjusted_pvalues,
        raw_pvalues,
        basis_type,
        n_basis: p,
        n_perm,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// FLM entry point helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Simple-regression t-statistic `|β̂_k / se_k|` for basis component `k`.
///
/// Fits OLS regression of `y` on the k-th column of `coeff`. Returns `0.0`
/// if the predictor is degenerate (`sxx < 1e-30`) or the residual variance
/// is non-positive (`se2 <= 0.0`), preventing divide-by-zero and NaN
/// propagation (T-30-04 guard).
fn component_t_stat(y: &[f64], coeff: &FdMatrix, k: usize) -> f64 {
    let n = y.len();
    let mx: f64 = (0..n).map(|i| coeff[(i, k)]).sum::<f64>() / n as f64;
    let my: f64 = y.iter().sum::<f64>() / n as f64;
    let sxx: f64 = (0..n).map(|i| (coeff[(i, k)] - mx).powi(2)).sum();
    if sxx < 1e-30 {
        return 0.0;
    }
    let sxy: f64 = (0..n).map(|i| (coeff[(i, k)] - mx) * (y[i] - my)).sum();
    let beta = sxy / sxx;
    let rss: f64 = (0..n)
        .map(|i| {
            let yhat = my + beta * (coeff[(i, k)] - mx);
            (y[i] - yhat).powi(2)
        })
        .sum();
    // se2 = rss / ((n-2) * sxx)  [standard error of beta squared]
    let se2 = rss / ((n - 2) as f64 * sxx);
    if se2 <= 0.0 {
        return 0.0;
    }
    (beta / se2.sqrt()).abs()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry point: interval-wise FLM coefficient test
// ─────────────────────────────────────────────────────────────────────────────

/// Interval-wise FLM coefficient test (response-permutation null).
///
/// Tests H₀: the response `y` is independent of the functional predictor
/// `data`. Projects `data` onto `nbasis` basis functions and, for each
/// basis component `k`, computes a simple-regression t-statistic
/// `|β̂_k / se_k|`. The permutation null shuffles the response vector `y`
/// (response permutation) and re-evaluates all per-component t-statistics.
///
/// **Assumption A2 divergence from R:** This implementation uses the
/// response-permutation simplification (shuffle `y`) rather than the
/// partial-residual method employed by `fdatest::ITPlmbspline`. The simpler
/// approach tests the global null "y is independent of the functional
/// predictor" consistently with the INF-01 permutation philosophy.
/// Per-component partial-residual permutation would require fitting
/// `n_perm × p` additional regressions and is not implemented here.
///
/// # Arguments
///
/// * `data` — functional data matrix (predictor), shape `(n, m)`.
/// * `y` — response vector, length `n`.
/// * `argvals` — evaluation points, length `m`.
/// * `basis_type` — `ProjectionBasisType::Bspline` or `::Fourier`.
/// * `nbasis` — requested number of basis functions (≥ 2).
/// * `n_perm` — number of response permutations (≥ 1).
/// * `seed` — RNG seed for reproducibility.
///
/// # Errors
///
/// * `InvalidDimension` — if `n < 2`, `y.len() != n`, or `argvals.len() != m`.
/// * `InvalidParameter` — if `nbasis < 2`, `n_perm == 0`, or basis projection fails.
#[must_use = "the ItpResult contains the adjusted p-values"]
pub fn itp_flm(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    basis_type: ProjectionBasisType,
    nbasis: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ItpResult, FdarError> {
    // 1. Validate inputs
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows (observations)".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements (matching data rows)"),
            actual: format!("{} elements", y.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if nbasis < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: "must be >= 2".to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    // 2. Project X onto basis coefficients (once)
    let proj = fdata_to_basis(data, argvals, nbasis, basis_type).ok_or_else(|| {
        FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("basis projection failed (nbasis={nbasis}, m={m})"),
        }
    })?;
    let coeff = proj.coefficients; // (n, p)
    let p = proj.n_basis;

    // 3. Observed per-component t-statistics
    let t0: Vec<f64> = (0..p).map(|k| component_t_stat(y, &coeff, k)).collect();

    // 4. Response-permutation loop → t_perm (n_perm, p)
    let mut rng = StdRng::seed_from_u64(seed);
    let mut perm_idx: Vec<usize> = (0..n).collect();
    let mut t_perm: Vec<Vec<f64>> = Vec::with_capacity(n_perm);
    let mut y_perm: Vec<f64> = vec![0.0; n];
    for _ in 0..n_perm {
        shuffle_itp(&mut perm_idx, &mut rng);
        for i in 0..n {
            y_perm[i] = y[perm_idx[i]];
        }
        let row: Vec<f64> = (0..p)
            .map(|k| component_t_stat(&y_perm, &coeff, k))
            .collect();
        t_perm.push(row);
    }

    // 5. Raw per-component p-values (+1 correction)
    let raw_pvalues: Vec<f64> = (0..p)
        .map(|k| {
            let n_ge = t_perm.iter().filter(|row| row[k] >= t0[k]).count();
            (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
        })
        .collect();

    // 6–8. Rank-transform → pval_matrix → closure adjustment
    let l = rank_transform(&t_perm, p, n_perm);
    let pval_matrix = build_pval_matrix(&raw_pvalues, &l, p, n_perm);
    let adjusted_pvalues = pval_correct(&pval_matrix, p);

    Ok(ItpResult {
        adjusted_pvalues,
        raw_pvalues,
        basis_type,
        n_basis: p,
        n_perm,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    // ─── pval_correct hand-computed unit test ────────────────────────────────

    /// Verifies the closure-adjustment index math against a hand-traced
    /// execution for p = 4.
    ///
    /// The `pval_matrix` was chosen with strictly decreasing values along rows
    /// (larger intervals have lower joint p-values) to make the cone-walk
    /// outcome easy to compute by inspection. Expected values were derived by
    /// tracing `get_2x_rev` and the cone-walk loop by hand.
    ///
    /// With this matrix:
    /// ```text
    /// row 0 (len-4):  [0.30, 0.25, 0.20, 0.15]
    /// row 1 (len-3):  [0.40, 0.35, 0.28, 0.22]
    /// row 2 (len-2):  [0.50, 0.45, 0.38, 0.32]
    /// row 3 (len-1):  [0.60, 0.55, 0.48, 0.42]
    /// ```
    ///
    /// Hand-traced result (before reverse): [0.42, 0.48, 0.55, 0.60]
    /// After `.reverse()`:                  [0.60, 0.55, 0.48, 0.42]
    #[test]
    fn pval_correct_hand_computed() {
        let p = 4;
        // pval_matrix[row][col]
        // row 0: full interval (length p)
        // row p-1: raw per-component (length 1)
        let pval_matrix = vec![
            vec![0.30, 0.25, 0.20, 0.15], // row 0: length-4 interval
            vec![0.40, 0.35, 0.28, 0.22], // row 1: length-3 intervals
            vec![0.50, 0.45, 0.38, 0.32], // row 2: length-2 intervals
            vec![0.60, 0.55, 0.48, 0.42], // row 3: raw p-values (length-1)
        ];

        let adjusted = pval_correct(&pval_matrix, p);

        // Expected values derived by hand (see docstring above):
        // get_2x_rev maps col c → original col (2*4-1-c) % 4 = (7-c) % 4
        // var=0: start get_2x_rev(3,0)=mat[3][(7)%4]=mat[3][3]=0.42; no update → 0.42
        // var=1: start get_2x_rev(3,1)=mat[3][(6)%4]=mat[3][2]=0.48; no update → 0.48
        // var=2: start get_2x_rev(3,2)=mat[3][(5)%4]=mat[3][1]=0.55; no update → 0.55
        // var=3: start get_2x_rev(3,3)=mat[3][(4)%4]=mat[3][0]=0.60; no update → 0.60
        // before reverse: [0.42, 0.48, 0.55, 0.60]
        // after  reverse: [0.60, 0.55, 0.48, 0.42]
        let expected = [0.60, 0.55, 0.48, 0.42];
        assert_eq!(adjusted.len(), p);
        for (k, (&got, &exp)) in adjusted.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-12,
                "adjusted_pvalues[{k}]: got {got}, expected {exp}"
            );
        }
    }

    /// Verifies that `fisher_cf` never produces NaN or -inf on a 0.0 p-value.
    #[test]
    fn fisher_cf_log_safe() {
        let v = fisher_cf(&[0.0, 0.5, 1.0]);
        assert!(
            v.is_finite(),
            "fisher_cf must be finite even with 0.0 input: {v}"
        );
        // Clamped: -2*(ln(1e-300) + ln(0.5) + ln(1.0))
        let expected = -2.0 * (1e-300f64.ln() + 0.5f64.ln() + 1.0f64.ln());
        assert!(
            (v - expected).abs() < 1e-10,
            "fisher_cf value mismatch: {v} vs {expected}"
        );
    }

    // ─── itp_one_pop tests ───────────────────────────────────────────────────

    /// Generates a sample of sine curves with an optional additive shift on
    /// the sub-interval `[shift_lo, shift_hi]`.
    fn make_shifted_sample(
        n: usize,
        argvals: &[f64],
        shift: f64,
        shift_lo: f64,
        shift_hi: f64,
        seed: u64,
    ) -> FdMatrix {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let m = argvals.len();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * std::f64::consts::PI;
            for (j, &t) in argvals.iter().enumerate() {
                let noise: f64 = rng.gen::<f64>() * 0.05;
                let s = if t >= shift_lo && t <= shift_hi {
                    shift
                } else {
                    0.0
                };
                data[(i, j)] = (t * 2.0 * std::f64::consts::PI + phase).sin() + noise + s;
            }
        }
        data
    }

    /// On a localized constant shift in [0.4, 0.6], at least one adjusted
    /// p-value should be small (< 0.05). This tests that the ITP correctly
    /// identifies a localized signal.
    #[test]
    fn one_population_localized() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        // Shift of 2.0 on [0.4, 0.6]
        let data = make_shifted_sample(n, &argvals, 2.0, 0.4, 0.6, 1001);
        let result = itp_one_pop(
            &data,
            &argvals,
            None,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_one_pop should succeed");
        assert_eq!(result.n_perm, 499);
        assert!(!result.adjusted_pvalues.is_empty());
        // At least one component should be significant
        let min_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_p < 0.05,
            "Expected at least one significant component, min adjusted p = {min_p}"
        );
    }

    /// On a null sample (zero shift), all adjusted p-values should be
    /// non-significant (max > 0.05, so most are not significant).
    #[test]
    fn one_population_null() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        // No shift at all — pure sine + tiny noise
        let data = make_shifted_sample(n, &argvals, 0.0, 0.0, 1.0, 2002);
        let result = itp_one_pop(
            &data,
            &argvals,
            None,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_one_pop should succeed");
        // Under the null, the max adjusted p-value should be non-significant
        let max_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_p > 0.10,
            "Expected non-significant result under null, max adjusted p = {max_p}"
        );
    }

    /// Same inputs must produce bit-identical results.
    #[test]
    fn one_population_deterministic() {
        let m = 30;
        let n = 15;
        let argvals = uniform_grid(m);
        let data = make_shifted_sample(n, &argvals, 1.0, 0.3, 0.7, 3003);
        let r1 = itp_one_pop(
            &data,
            &argvals,
            None,
            ProjectionBasisType::Bspline,
            10,
            99,
            77,
        )
        .unwrap();
        let r2 = itp_one_pop(
            &data,
            &argvals,
            None,
            ProjectionBasisType::Bspline,
            10,
            99,
            77,
        )
        .unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical ItpResult");
    }

    /// Invalid inputs must return FdarError, never panic.
    #[test]
    fn one_population_error_paths() {
        let m = 20;
        let argvals = uniform_grid(m);

        // n < 2
        let one_row = FdMatrix::zeros(1, m);
        assert!(
            matches!(
                itp_one_pop(
                    &one_row,
                    &argvals,
                    None,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "n < 2 should return InvalidDimension"
        );

        // argvals.len() != m
        let data = FdMatrix::zeros(5, m);
        let short_argvals = uniform_grid(m - 1);
        assert!(
            matches!(
                itp_one_pop(
                    &data,
                    &short_argvals,
                    None,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "argvals mismatch should return InvalidDimension"
        );

        // nbasis < 2
        assert!(
            matches!(
                itp_one_pop(
                    &data,
                    &argvals,
                    None,
                    ProjectionBasisType::Bspline,
                    1,
                    99,
                    0
                ),
                Err(FdarError::InvalidParameter { .. })
            ),
            "nbasis < 2 should return InvalidParameter"
        );

        // n_perm == 0
        assert!(
            matches!(
                itp_one_pop(&data, &argvals, None, ProjectionBasisType::Bspline, 5, 0, 0),
                Err(FdarError::InvalidParameter { .. })
            ),
            "n_perm == 0 should return InvalidParameter"
        );
    }

    // ─── itp_two_pop tests ───────────────────────────────────────────────────

    /// Build a sample of sine curves with an optional constant additive shift
    /// on `[shift_lo, shift_hi]` (used for both one-pop and two-pop fixtures).
    fn make_two_pop_sample(
        n: usize,
        argvals: &[f64],
        shift: f64,
        shift_lo: f64,
        shift_hi: f64,
        seed: u64,
    ) -> FdMatrix {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let m = argvals.len();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * std::f64::consts::PI;
            for (j, &t) in argvals.iter().enumerate() {
                let noise: f64 = rng.gen::<f64>() * 0.05;
                let s = if t >= shift_lo && t <= shift_hi {
                    shift
                } else {
                    0.0
                };
                data[(i, j)] = (t * 2.0 * std::f64::consts::PI + phase).sin() + noise + s;
            }
        }
        data
    }

    /// On a localized constant shift between groups, at least one adjusted
    /// p-value should be small (< 0.05).
    #[test]
    fn two_population_localized() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        let data_a = make_two_pop_sample(n, &argvals, 0.0, 0.4, 0.6, 1001);
        // Group B has a shift of 2.0 on [0.4, 0.6]
        let data_b = make_two_pop_sample(n, &argvals, 2.0, 0.4, 0.6, 2002);
        let result = itp_two_pop(
            &data_a,
            &data_b,
            &argvals,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_two_pop should succeed");
        assert_eq!(result.n_perm, 499);
        assert!(!result.adjusted_pvalues.is_empty());
        let min_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_p < 0.05,
            "Expected at least one significant component, min adjusted p = {min_p}"
        );
    }

    /// Under the null (both groups same distribution), all adjusted p-values
    /// should be non-significant (max > 0.10).
    #[test]
    fn two_population_null() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        let data_a = make_two_pop_sample(n, &argvals, 0.0, 0.0, 1.0, 3003);
        let data_b = make_two_pop_sample(n, &argvals, 0.0, 0.0, 1.0, 4004);
        let result = itp_two_pop(
            &data_a,
            &data_b,
            &argvals,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_two_pop should succeed");
        let max_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_p > 0.10,
            "Expected non-significant result under null, max adjusted p = {max_p}"
        );
    }

    /// Same seed must produce bit-identical results.
    #[test]
    fn two_population_deterministic() {
        let m = 30;
        let n = 15;
        let argvals = uniform_grid(m);
        let data_a = make_two_pop_sample(n, &argvals, 0.0, 0.0, 1.0, 5005);
        let data_b = make_two_pop_sample(n, &argvals, 1.0, 0.3, 0.7, 6006);
        let r1 = itp_two_pop(
            &data_a,
            &data_b,
            &argvals,
            ProjectionBasisType::Bspline,
            10,
            99,
            77,
        )
        .unwrap();
        let r2 = itp_two_pop(
            &data_a,
            &data_b,
            &argvals,
            ProjectionBasisType::Bspline,
            10,
            99,
            77,
        )
        .unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical ItpResult");
    }

    /// Invalid inputs must return FdarError, never panic.
    #[test]
    fn two_population_error_paths() {
        let m = 20;
        let argvals = uniform_grid(m);
        let good = FdMatrix::zeros(5, m);

        // n_a < 2
        let one_row = FdMatrix::zeros(1, m);
        assert!(
            matches!(
                itp_two_pop(
                    &one_row,
                    &good,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "n_a < 2 should return InvalidDimension"
        );

        // n_b < 2
        assert!(
            matches!(
                itp_two_pop(
                    &good,
                    &one_row,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "n_b < 2 should return InvalidDimension"
        );

        // m_a != m_b
        let wide = FdMatrix::zeros(5, m + 1);
        assert!(
            matches!(
                itp_two_pop(
                    &good,
                    &wide,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "m_a != m_b should return InvalidDimension"
        );

        // argvals mismatch
        let short_argvals = uniform_grid(m - 1);
        assert!(
            matches!(
                itp_two_pop(
                    &good,
                    &good,
                    &short_argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "argvals mismatch should return InvalidDimension"
        );

        // nbasis < 2
        assert!(
            matches!(
                itp_two_pop(
                    &good,
                    &good,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    1,
                    99,
                    0
                ),
                Err(FdarError::InvalidParameter { .. })
            ),
            "nbasis < 2 should return InvalidParameter"
        );

        // n_perm == 0
        assert!(
            matches!(
                itp_two_pop(
                    &good,
                    &good,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    0,
                    0
                ),
                Err(FdarError::InvalidParameter { .. })
            ),
            "n_perm == 0 should return InvalidParameter"
        );
    }

    // ─── itp_flm tests ───────────────────────────────────────────────────────

    /// Build a sample where y is a linear functional of X (localized signal).
    /// X is sine curves; y = coefficient of B-spline component `signal_comp`
    /// (extracted directly) plus noise — so there IS a true regression signal.
    fn make_flm_sample(n: usize, argvals: &[f64], seed: u64) -> (FdMatrix, Vec<f64>) {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let m = argvals.len();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * std::f64::consts::PI;
            let amp: f64 = rng.gen::<f64>() * 0.5 + 0.75; // amplitude in [0.75, 1.25]
            for (j, &t) in argvals.iter().enumerate() {
                let noise: f64 = (rng.gen::<f64>() - 0.5) * 0.05;
                data[(i, j)] = amp * (t * 2.0 * std::f64::consts::PI + phase).sin() + noise;
            }
            // y = amplitude + small noise — correlated with the overall curve level
            let y_noise: f64 = (rng.gen::<f64>() - 0.5) * 0.1;
            y[i] = amp + y_noise;
        }
        (data, y)
    }

    /// When y is correlated with the functional predictor, at least one
    /// adjusted p-value should be significant (< 0.05).
    #[test]
    fn flm_effect() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        let (data, y) = make_flm_sample(n, &argvals, 7007);
        let result = itp_flm(
            &data,
            &y,
            &argvals,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_flm should succeed");
        assert_eq!(result.n_perm, 499);
        assert!(!result.adjusted_pvalues.is_empty());
        let min_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        assert!(
            min_p < 0.05,
            "Expected at least one significant component with functional effect, min adjusted p = {min_p}"
        );
    }

    /// When y is independent of X (pure noise), all adjusted p-values should
    /// be non-significant (max > 0.10).
    #[test]
    fn flm_null() {
        use rand::Rng;
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        let mut rng = StdRng::seed_from_u64(8008);
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * std::f64::consts::PI;
            for (j, &t) in argvals.iter().enumerate() {
                data[(i, j)] = (t * 2.0 * std::f64::consts::PI + phase).sin();
            }
        }
        // y is pure independent noise — no functional signal
        let y: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        let result = itp_flm(
            &data,
            &y,
            &argvals,
            ProjectionBasisType::Bspline,
            15,
            499,
            42,
        )
        .expect("itp_flm should succeed");
        let max_p = result
            .adjusted_pvalues
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_p > 0.10,
            "Expected non-significant result under null, max adjusted p = {max_p}"
        );
    }

    /// Invalid inputs must return FdarError, never panic.
    #[test]
    fn flm_error_paths() {
        let m = 20;
        let argvals = uniform_grid(m);
        let data = FdMatrix::zeros(5, m);
        let y = vec![0.0f64; 5];

        // n < 2
        let one_row = FdMatrix::zeros(1, m);
        let y1 = vec![0.0f64; 1];
        assert!(
            matches!(
                itp_flm(
                    &one_row,
                    &y1,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "n < 2 should return InvalidDimension"
        );

        // y.len() != n
        let y_wrong = vec![0.0f64; 3];
        assert!(
            matches!(
                itp_flm(
                    &data,
                    &y_wrong,
                    &argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "y.len() != n should return InvalidDimension"
        );

        // argvals mismatch
        let short_argvals = uniform_grid(m - 1);
        assert!(
            matches!(
                itp_flm(
                    &data,
                    &y,
                    &short_argvals,
                    ProjectionBasisType::Bspline,
                    5,
                    99,
                    0
                ),
                Err(FdarError::InvalidDimension { .. })
            ),
            "argvals mismatch should return InvalidDimension"
        );

        // nbasis < 2
        assert!(
            matches!(
                itp_flm(&data, &y, &argvals, ProjectionBasisType::Bspline, 1, 99, 0),
                Err(FdarError::InvalidParameter { .. })
            ),
            "nbasis < 2 should return InvalidParameter"
        );

        // n_perm == 0
        assert!(
            matches!(
                itp_flm(&data, &y, &argvals, ProjectionBasisType::Bspline, 5, 0, 0),
                Err(FdarError::InvalidParameter { .. })
            ),
            "n_perm == 0 should return InvalidParameter"
        );
    }
}
