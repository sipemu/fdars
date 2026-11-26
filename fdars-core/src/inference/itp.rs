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
use crate::regions::{difference_regions, DifferenceRegion};
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
/// Components whose basis function vanishes on the whole grid (the two
/// outermost functions of fdars's extended-knot B-spline basis) carry no
/// information; they are excluded from the interval-wise closure and reported
/// with raw and adjusted p-value `1.0` (GH #76).
///
/// **Power note.** The adjusted p-value of a component is the maximum over
/// *all* intervals containing it, including long intervals that are mostly
/// null, so the ITP needs effects that span several basis supports (or large
/// samples). `fdatest` reports `n_ge / B`, which is `0` for a strong effect and
/// makes every interval containing that component reject (`-2 log 0 = ∞`);
/// with fdars's valid `(n_ge + 1)/(B + 1)` p-values this shortcut does not
/// exist, so fdars's adjusted p-values are larger than `fdatest`'s on strong
/// localised effects.
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
                // Both groups are relabelled (fdatest: `coeff[sample(n), ]`).
                // Before GH #76 group A kept its observed rows here, which
                // inflated the permutation statistics and cost almost all power.
                let m_a = (0..n_a).map(|r| pooled[(perm_idx[r], k)]).sum::<f64>() / n_a as f64;
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

    // 7–9. Rank-transform → pval_matrix → closure adjustment, over the
    // components whose basis function does not vanish on the grid.
    let active: Vec<usize> = (0..p)
        .filter(|&k| (0..n).any(|i| pooled[(i, k)] != 0.0))
        .collect();
    let adjusted_pvalues = closure_on_active(&t_perm, &raw_pvalues, &active, p, n_perm);

    Ok(ItpResult {
        adjusted_pvalues,
        raw_pvalues,
        basis_type,
        n_basis: p,
        n_perm,
    })
}

/// Interval-wise closure restricted to the `active` components.
///
/// fdars's extended-knot B-spline basis contains two functions that vanish on
/// the whole grid (their coefficients are identically zero). Such a component
/// carries no information: its observed statistic ties every permutation
/// (raw p = 1) while its tie-broken permutation ranks are pseudo-uniform, so
/// keeping it in the Fisher combinations only dilutes every interval that
/// contains it (including the circular wrap-around intervals). It is therefore
/// excluded from the closure (as it does not exist in `fdatest`'s clamped
/// basis) and reported with adjusted p-value `1.0`. With no vanishing
/// component (e.g. Fourier) this is exactly the full closure.
fn closure_on_active(
    t_perm: &[Vec<f64>],
    raw_pvalues: &[f64],
    active: &[usize],
    p: usize,
    n_perm: usize,
) -> Vec<f64> {
    let q = active.len();
    if q == p {
        let l = rank_transform(t_perm, p, n_perm);
        let pval_matrix = build_pval_matrix(raw_pvalues, &l, p, n_perm);
        return pval_correct(&pval_matrix, p);
    }
    let mut adjusted = vec![1.0; p];
    if q == 0 {
        return adjusted;
    }
    let t_sub: Vec<Vec<f64>> = t_perm
        .iter()
        .map(|row| active.iter().map(|&k| row[k]).collect())
        .collect();
    let raw_sub: Vec<f64> = active.iter().map(|&k| raw_pvalues[k]).collect();
    let l = rank_transform(&t_sub, q, n_perm);
    let pval_matrix = build_pval_matrix(&raw_sub, &l, q, n_perm);
    for (adj, &k) in pval_correct(&pval_matrix, q).into_iter().zip(active) {
        adjusted[k] = adj;
    }
    adjusted
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

// ─────────────────────────────────────────────────────────────────────────────
// Mapping basis-component adjusted p-values to the evaluation grid (GH #76)
// ─────────────────────────────────────────────────────────────────────────────

/// How basis-component adjusted p-values are mapped to grid points by
/// [`itp_pointwise_pvalues`].
///
/// `fdatest` reports the ITP per basis component (`corrected.pval`) and, in
/// `plot.ITP2` for B-splines, shades for each component `k` with
/// `corrected.pval[k] < α` a rectangle of width `h = (b − a)/(p − 1)` centred
/// at the `k`-th of `p` evenly spaced positions on `[a, b]` — i.e. every
/// domain point inherits the adjusted p-value of the *nearest* component
/// position. fdars's B-spline basis ([`crate::basis::bspline::bspline_basis`])
/// uses extended (not clamped) knots, so its component centres are not evenly
/// spaced over `[a, b]` (and the two outermost functions vanish on the domain);
/// [`ItpDomainMapping::Dominant`] applies the same nearest-component principle
/// to the actual basis: the component with the largest basis value at `t`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ItpDomainMapping {
    /// `p(t) = adjusted_p[k*(t)]` with `k*(t) = argmax_k B_k(t)` (ties → lowest
    /// `k`) — the `fdatest` nearest-component convention on fdars's basis.
    #[default]
    Dominant,
    /// `p(t) = max { adjusted_p[k] : B_k(t) > 0 }` — conservative: `t` is
    /// significant only if every basis function supported at `t` is.
    Support,
}

/// Pointwise adjusted p-value function of a B-spline ITP on the grid.
///
/// Maps the per-component `result.adjusted_pvalues` of [`itp_two_pop`] /
/// [`itp_one_pop`] (computed with [`ProjectionBasisType::Bspline`] on the same
/// `argvals`) to one adjusted p-value per grid point using `mapping` (see
/// [`ItpDomainMapping`] for the relation to `fdatest`). Because the ITP
/// controls the interval-wise error rate, `{t : p(t) < α}` is the set of
/// domain points where the null is rejected at level `α`.
///
/// # Errors
///
/// * [`FdarError::InvalidParameter`] for a Fourier ITP — Fourier components
///   are frequencies, not domain locations, so no time localisation exists.
/// * [`FdarError::InvalidDimension`] if `argvals` has fewer than 2 points or
///   `result.adjusted_pvalues.len() != result.n_basis`.
///
/// # Examples
///
/// ```
/// use fdars_core::basis::projection::ProjectionBasisType;
/// use fdars_core::inference::{itp_pointwise_pvalues, itp_two_pop, ItpDomainMapping};
/// use fdars_core::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let a = FdMatrix::from_column_major(
///     (0..6 * m).map(|k| ((k * 7919) % 11) as f64 / 11.0).collect(), 6, m).unwrap();
/// let b = FdMatrix::from_column_major(
///     (0..6 * m).map(|k| ((k * 104_729) % 13) as f64 / 13.0).collect(), 6, m).unwrap();
/// let itp = itp_two_pop(&a, &b, &t, ProjectionBasisType::Bspline, 8, 99, 1).unwrap();
/// let pw = itp_pointwise_pvalues(&itp, &t, ItpDomainMapping::Dominant).unwrap();
/// assert_eq!(pw.len(), m);
/// assert!(pw.iter().all(|p| itp.adjusted_pvalues.contains(p)));
/// ```
pub fn itp_pointwise_pvalues(
    result: &ItpResult,
    argvals: &[f64],
    mapping: ItpDomainMapping,
) -> Result<Vec<f64>, FdarError> {
    if result.basis_type != ProjectionBasisType::Bspline {
        return Err(FdarError::InvalidParameter {
            parameter: "result",
            message: "only a B-spline ITP can be mapped to the domain (Fourier components \
                      are frequencies)"
                .to_string(),
        });
    }
    let m = argvals.len();
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: "at least 2 points".to_string(),
            actual: format!("{m}"),
        });
    }
    let p = result.n_basis;
    if result.adjusted_pvalues.len() != p || p < 6 {
        return Err(FdarError::InvalidDimension {
            parameter: "result",
            expected: format!("n_basis >= 6 adjusted p-values (n_basis = {p})"),
            actual: format!("{}", result.adjusted_pvalues.len()),
        });
    }
    // Same basis as `fdata_to_basis(.., Bspline)`: order 4, nknots = p − 4.
    let basis = crate::basis::bspline::bspline_basis(argvals, p - 4, 4);
    let adj = &result.adjusted_pvalues;
    Ok((0..m)
        .map(|j| {
            let value = |k: usize| basis[j + k * m];
            match mapping {
                ItpDomainMapping::Dominant => {
                    let mut best = 0;
                    for k in 1..p {
                        if value(k) > value(best) {
                            best = k;
                        }
                    }
                    adj[best]
                }
                ItpDomainMapping::Support => (0..p)
                    .filter(|&k| value(k) > 1e-12)
                    .map(|k| adj[k])
                    .fold(0.0_f64, f64::max),
            }
        })
        .collect())
}

/// Configuration for [`itp_two_pop_regions`].
///
/// Construct with `ItpRegionConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ItpRegionConfig {
    /// Requested number of B-spline basis functions (default 15; ≥ 6 after
    /// clamping, see [`ItpResult::n_basis`]).
    pub nbasis: usize,
    /// Number of permutations (default 999).
    pub n_perm: usize,
    /// RNG seed (default 42).
    pub seed: u64,
    /// Interval-wise significance level (default 0.05).
    pub alpha: f64,
    /// Component → grid mapping (default [`ItpDomainMapping::Dominant`]).
    pub mapping: ItpDomainMapping,
}

impl Default for ItpRegionConfig {
    fn default() -> Self {
        Self {
            nbasis: 15,
            n_perm: 999,
            seed: 42,
            alpha: 0.05,
            mapping: ItpDomainMapping::Dominant,
        }
    }
}

/// Result of [`itp_two_pop_regions`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ItpRegionsResult {
    /// The underlying per-component ITP.
    pub itp: ItpResult,
    /// Adjusted p-value at each grid point ([`itp_pointwise_pvalues`]).
    pub pointwise_adjusted_pvalues: Vec<f64>,
    /// Pointwise sample-mean difference `mean_a(t) − mean_b(t)`, used for the
    /// region directions.
    pub mean_difference: Vec<f64>,
    /// Contiguous grid regions with `p(t) < alpha`, split where the sign of
    /// `mean_difference` changes (`Positive` ⇔ A above B);
    /// `min_adjusted_pvalue` is the smallest `p(t)` in the region.
    pub regions: Vec<DifferenceRegion>,
}

/// Two-population B-spline ITP localised on the grid: where do the two mean
/// functions differ?
///
/// Runs [`itp_two_pop`] with a B-spline basis, maps the closure-adjusted
/// component p-values to the grid with [`itp_pointwise_pvalues`] and returns
/// the contiguous regions where the adjusted p-value is below `alpha`, each
/// with the direction of the mean difference. The interval-wise error rate is
/// controlled at `alpha` (Pini & Vantini 2016).
///
/// # Errors
///
/// As [`itp_two_pop`]; additionally [`FdarError::InvalidParameter`] for
/// `alpha` outside `(0, 1)`.
///
/// # Examples
///
/// ```
/// use fdars_core::inference::{itp_two_pop_regions, ItpRegionConfig};
/// use fdars_core::FdMatrix;
///
/// let m = 50;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 25;
/// let (mut a, mut b) = (FdMatrix::zeros(n, m), FdMatrix::zeros(n, m));
/// for i in 0..n {
///     for j in 0..m {
///         let noise = |s: usize| 0.3 * (((i * 31 + j * 17 + s) % 13) as f64 / 6.0 - 1.0);
///         // B is shifted up on [0.5, 1].
///         let step = 1.0 / (1.0 + (-(t[j] - 0.5) / 0.03).exp());
///         a[(i, j)] = noise(0);
///         b[(i, j)] = step + noise(5);
///     }
/// }
/// let mut cfg = ItpRegionConfig::default();
/// cfg.n_perm = 499;
/// let res = itp_two_pop_regions(&a, &b, &t, &cfg).unwrap();
/// assert!(!res.regions.is_empty());
/// assert!(res.regions.iter().all(|r| r.t_start > 0.3));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn itp_two_pop_regions(
    data_a: &FdMatrix,
    data_b: &FdMatrix,
    argvals: &[f64],
    config: &ItpRegionConfig,
) -> Result<ItpRegionsResult, FdarError> {
    if !(config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {}", config.alpha),
        });
    }
    let itp = itp_two_pop(
        data_a,
        data_b,
        argvals,
        ProjectionBasisType::Bspline,
        config.nbasis,
        config.n_perm,
        config.seed,
    )?;
    let pointwise = itp_pointwise_pvalues(&itp, argvals, config.mapping)?;
    let mean_difference = column_mean_difference(data_a, data_b);
    let significant: Vec<bool> = pointwise.iter().map(|&p| p < config.alpha).collect();
    let regions = difference_regions(argvals, &mean_difference, &significant, Some(&pointwise))?;
    Ok(ItpRegionsResult {
        itp,
        pointwise_adjusted_pvalues: pointwise,
        mean_difference,
        regions,
    })
}

/// `colMeans(a) − colMeans(b)` (same column count assumed, validated upstream).
fn column_mean_difference(a: &FdMatrix, b: &FdMatrix) -> Vec<f64> {
    let (n_a, n_b) = (a.nrows() as f64, b.nrows() as f64);
    (0..a.ncols())
        .map(|j| a.column(j).iter().sum::<f64>() / n_a - b.column(j).iter().sum::<f64>() / n_b)
        .collect()
}

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

    /// Build a sample where y is the mean of X over [lo, hi] plus small noise.
    /// This creates a strong, localized functional regression signal: curves with
    /// larger values on [lo, hi] have larger y.
    fn make_flm_sample(
        n: usize,
        argvals: &[f64],
        lo: f64,
        hi: f64,
        seed: u64,
    ) -> (FdMatrix, Vec<f64>) {
        use rand::Rng;
        let mut rng = StdRng::seed_from_u64(seed);
        let m = argvals.len();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            // Each curve is a random linear ramp with different slope/offset
            let scale: f64 = rng.gen::<f64>() * 3.0 + 1.0; // slope in [1, 4]
            let offset: f64 = (rng.gen::<f64>() - 0.5) * 2.0; // offset in [-1, 1]
            let mut local_sum = 0.0f64;
            let mut local_cnt = 0usize;
            for (j, &t) in argvals.iter().enumerate() {
                let noise: f64 = (rng.gen::<f64>() - 0.5) * 0.02;
                let v = scale * t + offset + noise;
                data[(i, j)] = v;
                if t >= lo && t <= hi {
                    local_sum += v;
                    local_cnt += 1;
                }
            }
            // y = mean of X on [lo, hi] plus tiny noise — strong local signal
            let y_noise: f64 = (rng.gen::<f64>() - 0.5) * 0.05;
            y[i] = if local_cnt > 0 {
                local_sum / local_cnt as f64
            } else {
                0.0
            } + y_noise;
        }
        (data, y)
    }

    /// When y is the mean of X over [0.3, 0.7], at least one adjusted p-value
    /// should be significant (< 0.05) — a strong localized functional signal.
    #[test]
    fn flm_effect() {
        let m = 50;
        let n = 30;
        let argvals = uniform_grid(m);
        let (data, y) = make_flm_sample(n, &argvals, 0.3, 0.7, 7007);
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

    // ─── fdatest 2.1.1 golden (GH #76) ──────────────────────────────────────

    // Reference: R fdatest 2.1.1, ITP2bspline(d1, d2, mu = 0, order = 4,
    // nknots = 5, B = 40, paired = FALSE) after set.seed(7), on the seeded data
    // of `.planning/quick/261126-t4g-aln-localised-group-differences-scb-band/
    // golden_itp2bspline.R`. The script replicates ITP2bspline's permutation
    // loop with the same seed to export the permutation statistic matrix
    // T_coeff (B x p, row-major) and checks that it reproduces ITP2bspline's
    // `pval`. `pval`, `pval.matrix` and `corrected.pval` are ITP2bspline's own
    // outputs.
    const P_G: usize = 7;
    const B_G: usize = 40;
    const FDATEST_T_COEFF: [f64; 280] = [
        0.007881063994480835,
        0.475628258105942,
        0.029199514929517223,
        0.13095763499926305,
        0.03464089998195963,
        0.20138924595225965,
        0.44854936670515483,
        0.1851273293322671,
        0.44933583725496123,
        0.35880529451517984,
        0.12093995839940597,
        0.43685803482026553,
        0.5378490294067939,
        0.46130098771633554,
        0.08950842092716685,
        0.23969732338505512,
        0.14579282102642366,
        0.29326655845321176,
        0.7114269667142417,
        0.9689937672946379,
        0.06013306400852285,
        0.11720872484085824,
        0.06816962434395768,
        0.45378028507212664,
        0.7961739395856707,
        1.2277926641323194,
        1.168648247672096,
        0.1425367400484742,
        0.08165278672200499,
        0.263383628657741,
        0.11420526215989324,
        0.0762992030242702,
        0.42342424924859867,
        0.5027317679228018,
        0.19214202711809295,
        0.14547743258377022,
        0.3427562266977598,
        0.7367551202657958,
        0.6874068448518079,
        0.476708949862759,
        0.08696488448975565,
        0.3210571143010865,
        0.1470804364745245,
        0.19265374710560612,
        0.3213345075426368,
        0.24992940621839121,
        0.06922545426266535,
        0.10363181125269727,
        0.08636818775750957,
        0.2454266894344282,
        0.3758730058426826,
        0.627271573472181,
        0.49852295153997367,
        0.07413652192094777,
        0.22254372792743754,
        0.2692917036452958,
        0.32670600474969513,
        0.29613067361441153,
        0.06237642485677619,
        0.2857722156257585,
        0.17214082301634948,
        0.03655855507356465,
        0.37607341055348287,
        0.07734868255067977,
        0.17462141125898445,
        0.42340516539502926,
        0.1887485345822186,
        0.5849770772098024,
        0.4992705271342489,
        0.0207555651742915,
        0.11647967144163492,
        0.15826467974110348,
        0.002539430635001727,
        0.24634595164125903,
        0.016126377353146593,
        0.1608305300837516,
        0.38958291032763603,
        0.11906740227327617,
        0.3748719971009128,
        0.5686140282541647,
        0.4873320721226696,
        0.4423817774526546,
        0.17308115915331068,
        0.16486263093242884,
        0.03398995041953706,
        0.2571962311170063,
        0.4262978024580095,
        0.048218258457449065,
        0.3120072670911791,
        0.012623973487011608,
        0.503155986906876,
        0.031115631694991885,
        0.030471049504536274,
        0.23526863417740151,
        0.027554343424959132,
        0.25944038673505054,
        0.6378770337610009,
        0.021194650758259714,
        0.34730590739375533,
        0.09655284943024844,
        0.0546114390433472,
        0.21149604398241492,
        0.16655914277602402,
        0.09276068266132442,
        0.10109280034593118,
        0.2786573400220308,
        0.016941045111027786,
        0.34144211051806095,
        0.7072678662444637,
        1.2461528589226134,
        1.1660885112160324,
        0.2838879229409328,
        0.26551719543085495,
        0.24633240327578,
        0.1605704553352092,
        0.1458798104301758,
        0.7978655086209677,
        0.8076022751643271,
        0.5419537727350497,
        0.02488858518606976,
        0.17582761916747847,
        0.7923961994006683,
        0.6307848331607383,
        0.41623485180277364,
        0.6259810568553883,
        0.38049853282801993,
        0.10155679244794551,
        0.8015281755234052,
        0.7074399332318679,
        0.5508129684156813,
        0.2501095346050035,
        0.07305044358951418,
        0.5626423204375817,
        0.3119744033107274,
        0.5700524316040323,
        0.4198817862788291,
        0.004283425218033571,
        0.046277095224789955,
        0.0690253894599312,
        0.0683694407370588,
        0.1006605829780769,
        0.16385387234749615,
        0.24423444346930623,
        0.3244356620279831,
        0.1742540970234645,
        0.18865447815884373,
        0.12399651071715814,
        0.012987800220593175,
        0.29168719358700573,
        0.41511462344052474,
        0.5269644901456594,
        0.2768716743654054,
        0.5909663386283054,
        0.46212922081988717,
        0.22180021695187813,
        0.08687638376018103,
        0.0632305238002624,
        0.4435376261343204,
        0.02890364368849596,
        0.34529697259498016,
        0.23105265151568802,
        0.20751300717677112,
        0.025090182119876903,
        0.1845321560480515,
        0.13428110984468752,
        0.712572315548622,
        0.6115819188123397,
        0.38032623086087625,
        0.1911293798721203,
        0.2393826391404229,
        0.17239489657237628,
        0.24794189476036071,
        0.15102256094728195,
        0.11404165030490038,
        0.09500439461297802,
        0.22864545512347823,
        0.560579793961606,
        0.580730515212142,
        0.20819161452688,
        0.3468115304327275,
        0.5111957315279929,
        0.20812690933177422,
        0.1255236280877168,
        0.6212085222947782,
        0.961966728058606,
        1.1427931764469152,
        1.1750685248982842,
        0.6476945821444479,
        0.1898944640914467,
        0.13874757604603713,
        0.15581285051031146,
        0.18324533427846723,
        0.15457799709110742,
        0.3150000315382988,
        0.32009125933687915,
        0.04946178740066476,
        0.09398176109252439,
        0.02365965365769085,
        0.2778914113262656,
        0.5810365616342011,
        0.9538772696212459,
        0.7833488570891294,
        0.36656382289885797,
        0.1120888780875614,
        0.10775757196060243,
        0.16386620995514245,
        0.40308617435462996,
        0.8115573368811172,
        0.5187721137538369,
        0.21679169209349433,
        0.16650089664385886,
        0.09434167705299024,
        0.46342695088187713,
        0.6020331567619368,
        0.20978577597813586,
        0.07615873687990893,
        0.6227554325092157,
        0.14097122763147857,
        0.10415108098537795,
        0.09973434916246382,
        0.2653364456183485,
        0.02423339041808048,
        0.08952661035598658,
        0.08900966254778934,
        0.1055530309929416,
        0.3918396938269359,
        0.37216664425964296,
        0.24399761525545016,
        0.4466546492883794,
        0.08955881390837755,
        0.21419569408407377,
        0.006136731752165874,
        0.18405188513122106,
        0.5739197017877313,
        0.3694304027478663,
        0.3221343228917386,
        0.6560239140888917,
        0.059487987279487925,
        0.14392955222678833,
        0.6337886256271035,
        0.22720118070991302,
        0.18439213607659127,
        0.04548903541580329,
        0.26572060519728774,
        0.2077977063785096,
        0.04103032494395765,
        0.28911095254583297,
        0.766495005065398,
        0.7317648498329973,
        0.6038262514655843,
        0.16303233127831723,
        0.1512684184533296,
        0.23848988284317948,
        0.019484130992971083,
        0.36859422551626575,
        0.7690795724862095,
        0.42114658771221625,
        0.14344529723220223,
        0.16906069673702365,
        0.04546803451605727,
        0.4269718554996321,
        0.6103706865365062,
        0.2408040451512443,
        0.5120152683851364,
        0.46216429863558883,
        0.6372924973724445,
        0.20953229840079446,
        0.17974140734779187,
        0.5344331028132208,
        0.19839796398025344,
        0.6614455355652022,
        0.6295710840759438,
        0.29959443821182097,
        0.1808930968112289,
        0.12445826155522932,
        0.09355862921731806,
        0.48677495780936464,
        1.0639355239776804,
        0.981389780910036,
        0.3080081494557859,
    ];
    const FDATEST_PVAL: [f64; 7] = [0.75, 0.275, 0.025, 0.025, 0.0, 0.0, 0.075];
    const FDATEST_PVAL_MATRIX: [f64; 49] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.025, 0.1, 0.025, 0.0, 0.0, 0.0, 0.0,
        0.225, 0.5, 0.05, 0.025, 0.0, 0.0, 0.0, 0.175, 0.75, 0.275, 0.025, 0.025, 0.0, 0.0, 0.075,
    ];
    const FDATEST_CORRECTED: [f64; 7] = [0.75, 0.5, 0.1, 0.025, 0.0, 0.0, 0.225];

    /// Given fdatest's permutation statistics and raw p-values, the fdars
    /// closure pipeline (rank transform → interval p-value matrix →
    /// `pval_correct`) reproduces fdatest's `pval.matrix` and `corrected.pval`.
    #[test]
    fn closure_matches_fdatest_itp2bspline() {
        let t_perm: Vec<Vec<f64>> = FDATEST_T_COEFF.chunks(P_G).map(<[f64]>::to_vec).collect();
        assert_eq!(t_perm.len(), B_G);
        let l = rank_transform(&t_perm, P_G, B_G);
        let mat = build_pval_matrix(&FDATEST_PVAL, &l, P_G, B_G);
        for r in 0..P_G {
            for c in 0..P_G {
                let expected = FDATEST_PVAL_MATRIX[r * P_G + c];
                assert!(
                    (mat[r][c] - expected).abs() < 1e-12,
                    "pval.matrix[{r}][{c}]: fdars {} vs fdatest {expected}",
                    mat[r][c]
                );
            }
        }
        let corrected = pval_correct(&mat, P_G);
        for (k, (&a, &e)) in corrected.iter().zip(FDATEST_CORRECTED.iter()).enumerate() {
            assert!((a - e).abs() < 1e-12, "corrected[{k}]: {a} vs {e}");
        }
    }

    fn fake_bspline_result(adjusted: Vec<f64>) -> ItpResult {
        let p = adjusted.len();
        ItpResult {
            raw_pvalues: adjusted.clone(),
            adjusted_pvalues: adjusted,
            basis_type: ProjectionBasisType::Bspline,
            n_basis: p,
            n_perm: 1,
        }
    }

    /// Dominant mapping = nearest basis-function centre. For fdars's extended
    /// cubic basis with `nknots` knots on `[0, 1]` (spacing `h`), component `k`
    /// is centred at `(k − 2)·h`; hence grid point `t` maps to
    /// `k = 2 + round(t / h)`.
    #[test]
    fn pointwise_dominant_is_nearest_centre() {
        let p = 9; // nknots = 5, h = 0.25
        let adjusted: Vec<f64> = (0..p).map(|k| 0.01 * (k + 1) as f64).collect();
        let res = fake_bspline_result(adjusted.clone());
        let t = uniform_grid(41);
        let pw = itp_pointwise_pvalues(&res, &t, ItpDomainMapping::Dominant).unwrap();
        for (j, &tj) in t.iter().enumerate() {
            let x = tj / 0.25;
            if (x - x.floor() - 0.5).abs() < 1e-9 {
                continue; // equidistant from two centres
            }
            let k = 2 + x.round() as usize;
            assert_eq!(pw[j], adjusted[k], "t = {tj}");
        }
        // Outermost (vanishing) components never dominate.
        assert!(pw.iter().all(|&v| v != adjusted[0] && v != adjusted[p - 1]));
    }

    #[test]
    fn pointwise_support_is_conservative() {
        let adjusted = vec![1.0, 0.9, 0.01, 0.01, 0.01, 0.6, 0.01, 0.8, 1.0];
        let res = fake_bspline_result(adjusted);
        let t = uniform_grid(41);
        let dom = itp_pointwise_pvalues(&res, &t, ItpDomainMapping::Dominant).unwrap();
        let sup = itp_pointwise_pvalues(&res, &t, ItpDomainMapping::Support).unwrap();
        assert!(dom.iter().zip(&sup).all(|(d, s)| s >= d));
        // t = 0.25 is a knot: only components 2, 3, 4 are non-zero (all 0.01).
        assert_eq!(sup[10], 0.01);
        // Fourier results cannot be mapped.
        let mut f = fake_bspline_result(vec![0.5; 7]);
        f.basis_type = ProjectionBasisType::Fourier;
        assert!(itp_pointwise_pvalues(&f, &t, ItpDomainMapping::Dominant).is_err());
    }

    /// Localised mean difference (a smooth step on [0.55, 1]; the ITP needs
    /// effects spanning several basis supports): regions sit on the step only.
    #[test]
    fn itp_two_pop_regions_localises_bump() {
        use rand::Rng;
        let m = 60;
        let t = uniform_grid(m);
        let mut rng = StdRng::seed_from_u64(76);
        let mut gen = |bump: f64| {
            let mut d = FdMatrix::zeros(25, m);
            for i in 0..25 {
                let phase: f64 = rng.gen_range(0.0..1.0);
                for j in 0..m {
                    let e: f64 = rng.gen_range(-0.4..0.4);
                    let smooth = 0.1 * (2.0 * std::f64::consts::PI * (t[j] + phase)).sin();
                    d[(i, j)] = smooth + bump / (1.0 + (-(t[j] - 0.55) / 0.03).exp()) + e;
                }
            }
            d
        };
        let a = gen(0.0);
        let b = gen(1.0);
        let cfg = ItpRegionConfig {
            n_perm: 499,
            ..ItpRegionConfig::default()
        };
        let res = itp_two_pop_regions(&a, &b, &t, &cfg).unwrap();
        assert!(!res.regions.is_empty(), "bump should be detected");
        for r in &res.regions {
            assert_eq!(r.direction, crate::regions::SignificanceDirection::Negative);
            assert!(r.t_start >= 0.4, "{r:?}");
            assert!(r.min_adjusted_pvalue.unwrap() < 0.05);
        }
        assert!(res
            .regions
            .iter()
            .any(|r| r.t_start <= 0.7 && r.t_end >= 0.7));
        let mut bad = cfg.clone();
        bad.alpha = 1.5;
        assert!(itp_two_pop_regions(&a, &b, &t, &bad).is_err());
    }

    /// The two vanishing B-spline components are excluded and reported as 1.
    #[test]
    fn vanishing_bspline_components_reported_as_one() {
        let m = 30;
        let t = uniform_grid(m);
        let a = make_two_pop_sample(8, &t, 0.0, 0.0, 1.0, 11);
        let b = make_two_pop_sample(8, &t, 1.0, 0.3, 0.7, 12);
        let r = itp_two_pop(&a, &b, &t, ProjectionBasisType::Bspline, 10, 99, 3).unwrap();
        let p = r.n_basis;
        assert_eq!(r.adjusted_pvalues[0], 1.0);
        assert_eq!(r.adjusted_pvalues[p - 1], 1.0);
        // Adjusted >= raw for every component (closure is a max over intervals).
        for k in 1..p - 1 {
            assert!(r.adjusted_pvalues[k] >= r.raw_pvalues[k] - 1e-12);
        }
    }
}
