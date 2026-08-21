//! Functional autocorrelation, partial autocorrelation, and white-noise bands.
//!
//! Implements the L2-norm functional ACF following the `fdaACF` convention
//! (Mestre et al. 2021) and the scalar Durbin-Levinson fPACF. The Monte-Carlo
//! strong-white-noise confidence band is the sole band method provided.
//!
//! # Algorithm references
//!
//! Mestre et al. (2021), "Functional autocorrelation function for functional
//! time series", *Computational Statistics & Data Analysis*.
//! <https://github.com/GMestreM/fdaACF>

use super::FacfResult;
use crate::error::FdarError;
use crate::helpers::{simpsons_weights, trapz, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;

// ─── Input validation ────────────────────────────────────────────────────────

/// Validate that `data` is non-empty and `argvals` length matches data columns.
///
/// Returns `(n, m)` on success.
fn validate_fts_input(data: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    Ok((n, m))
}

// ─── Internal numeric helpers ─────────────────────────────────────────────────

/// Compute the sample mean curve: `xbar[j] = (1/n) Σ_i data[(i,j)]`.
fn mean_curve(data: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut xbar = vec![0.0f64; m];
    let inv_n = 1.0 / n as f64;
    for j in 0..m {
        let mut s = 0.0;
        for i in 0..n {
            s += data[(i, j)];
        }
        xbar[j] = s * inv_n;
    }
    xbar
}

/// Compute the lag-h sample autocovariance matrix.
///
/// Returns a flat m×m Vec in column-major order:
/// `c_h[j1 + j2 * m] = (1/n) Σ_{i=0}^{n-h-1} (x_{i,j1} - xbar[j1]) * (x_{i+h,j2} - xbar[j2])`
///
/// Normalised by `1/n` (not `1/(n-h)`) following the `fdaACF` / `ftsa` convention.
/// Access via `c_h[j1 + j2 * m]` where `j1` is the row index and `j2` is the
/// column index (column-major stride = m).
///
/// # Note for reusers (plan 34-03)
///
/// This function is the shared spine used by the long-run covariance estimator.
/// The `h = 0` case returns the sample covariance operator C_0.
pub(crate) fn autocovariance_matrix(
    data: &FdMatrix,
    xbar: &[f64],
    h: usize,
    n: usize,
    m: usize,
) -> Vec<f64> {
    let mut c_h = vec![0.0f64; m * m];
    let inv_n = 1.0 / n as f64;
    for i in 0..(n - h) {
        for j1 in 0..m {
            let xi1 = data[(i, j1)] - xbar[j1];
            for j2 in 0..m {
                let xi2 = data[(i + h, j2)] - xbar[j2];
                c_h[j1 + j2 * m] += xi1 * xi2;
            }
        }
    }
    for x in &mut c_h {
        *x *= inv_n;
    }
    c_h
}

/// Hilbert-Schmidt squared L2 norm of an m×m matrix (column-major).
///
/// `‖C_h‖²_HS = Σ_{j1,j2} c_h[j1+j2*m]² * weights[j1] * weights[j2]`
fn hs_norm_sq(c_h: &[f64], m: usize, weights: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for j1 in 0..m {
        let w1 = weights[j1];
        for j2 in 0..m {
            let val = c_h[j1 + j2 * m];
            sum += val * val * w1 * weights[j2];
        }
    }
    sum
}

/// Compute the fACF normalization denominator.
///
/// `normalization = ∫_T C_0(t,t) dt` — trapezoidal integral of the diagonal of C_0.
/// Returns `Err(ComputationFailed)` when the diagonal integral is below `NUMERICAL_EPS`
/// (degenerate / zero-variance input).
fn acf_normalization(
    c0: &[f64],
    m: usize,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let diag: Vec<f64> = (0..m).map(|j| c0[j + j * m]).collect();
    let norm = trapz(&diag, argvals);
    if norm.abs() < NUMERICAL_EPS {
        return Err(FdarError::ComputationFailed {
            operation: "functional_acf",
            detail: "lag-0 covariance diagonal integrates to near zero (degenerate data)"
                .to_string(),
        });
    }
    Ok(norm)
}

// ─── Monte-Carlo white-noise band ─────────────────────────────────────────────

/// Compute the (chi²-mixture) MC white-noise band threshold.
///
/// Under the strong-white-noise null the scaled statistic `N * ‖Ĉ_h‖²_HS` converges
/// to `Q ~ Σ_{j,k} λ_j λ_k χ²_1(j,k)` where λ_1,…,λ_K are the truncated eigenvalues
/// of C_0.  This function returns the `ci`-quantile of the distribution of `Q / N`
/// via Monte-Carlo simulation using `n_sim` realisations.
///
/// `eigenvalues` must already be sorted descending and truncated to those with
/// `λ_j / λ_max > 1e-4`.
fn mc_band_threshold(
    eigenvalues: &[f64],
    n: usize,
    n_sim: usize,
    ci: f64,
    seed: u64,
) -> f64 {
    use rand_distr::{ChiSquared, Distribution};
    let mut rng = StdRng::seed_from_u64(seed);
    let chi2 = ChiSquared::new(1.0).expect("df=1 is always valid");
    let mut realizations = Vec::with_capacity(n_sim);
    for _ in 0..n_sim {
        let mut q = 0.0f64;
        for &lj in eigenvalues {
            for &lk in eigenvalues {
                q += lj * lk * chi2.sample(&mut rng);
            }
        }
        realizations.push(q / n as f64);
    }
    realizations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((ci * n_sim as f64) as usize).min(n_sim - 1);
    realizations[idx]
}

// ─── Durbin-Levinson fPACF ────────────────────────────────────────────────────

/// Scalar Durbin-Levinson recursion for the partial autocorrelation function.
///
/// `rho[k]` = ρ_{k+1} (0-indexed; rho[0] = ρ_1, rho[1] = ρ_2, …).
///
/// Returns a Vec of length `rho.len()` where `pacf[k]` = ϕ_{k+1,k+1}.
///
/// # Numerical stability
///
/// The denominator `1 - Σ phi[k-1][j] * rho[j-1]` can approach zero for
/// near-unit-root series. When `|denominator| < 1e-12` the remaining PACF
/// values are set to `0.0` and the recursion stops early.
fn durbin_levinson_pacf(rho: &[f64]) -> Vec<f64> {
    let p = rho.len();
    if p == 0 {
        return vec![];
    }
    // phi[k][j] uses 1-based indices; allocate on heap to avoid stack blowup.
    let mut phi = vec![vec![0.0f64; p + 1]; p + 1];
    let mut pacf = vec![0.0f64; p];

    phi[1][1] = rho[0];
    pacf[0] = rho[0];

    for k in 2..=p {
        // numerator = rho[k-1] - Σ_{j=1}^{k-1} phi[k-1][j] * rho[k-1-j]
        let num = rho[k - 1]
            - (1..k)
                .map(|j| phi[k - 1][j] * rho[k - 1 - j])
                .sum::<f64>();
        // denominator = 1 - Σ_{j=1}^{k-1} phi[k-1][j] * rho[j-1]
        let den = 1.0
            - (1..k)
                .map(|j| phi[k - 1][j] * rho[j - 1])
                .sum::<f64>();
        if den.abs() < 1e-12 {
            // Denominator collapse — set remaining PACF to 0.
            break;
        }
        phi[k][k] = num / den;
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
        pacf[k - 1] = phi[k][k];
    }
    pacf
}

// ─── Public entry points ──────────────────────────────────────────────────────

/// Functional autocorrelation and partial autocorrelation of a curve series.
///
/// Computes the L2-norm functional ACF at lags `1..=max_lag` following the
/// `fdaACF` convention (Mestre et al. 2021), the scalar Durbin-Levinson fPACF,
/// and Monte-Carlo strong-white-noise confidence bands.
///
/// # Arguments
///
/// * `data` — Time-ordered functional observations (`N × m`, column-major).
///   Rows are curves ordered from earliest to latest.
/// * `argvals` — Evaluation points on the common grid (length `m`).
/// * `max_lag` — Maximum lag to compute.  `None` uses `min(20, N/4)`.
/// * `n_sim` — Monte-Carlo replications for the white-noise band (default 999).
/// * `ci` — Confidence level for the upper band (default 0.95).
/// * `seed` — Deterministic RNG seed for the MC band.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] — `data` is empty, `argvals.len() != m`,
///   or the requested `max_lag + 1 > N` (too few curves).
/// * [`FdarError::InvalidParameter`] — `max_lag == 0` (must be ≥ 1).
/// * [`FdarError::ComputationFailed`] — the lag-0 covariance diagonal
///   integrates to near zero (degenerate / constant-curve input).
///
/// # Algorithm
///
/// 1. Compute the sample mean curve and Simpson quadrature weights.
/// 2. Compute the m×m lag-0 sample autocovariance operator C_0 (normalised by 1/N).
/// 3. For each h = 1..=max_lag, compute C_h and `ρ_h = sqrt(‖C_h‖²_HS) / normalization`
///    where `normalization = ∫ C_0(t,t) dt` (trace integral, trapezoidal).
/// 4. Eigendecompose C_0 via `nalgebra::SymmetricEigen`; truncate eigenvalues
///    with `λ_j / λ_max < 1e-4`.
/// 5. Run `n_sim` MC draws of `Q = Σ_{j,k} λ_j λ_k χ²_1(j,k)` to obtain the
///    `ci`-quantile; `upper_band[h] = sqrt(q_ci) / normalization`.
/// 6. Apply scalar Durbin-Levinson to `acf` to obtain `pacf`.
///
/// # Divergence from R fdaACF
///
/// **White-noise band:** `fdaACF` offers both an exact Imhof band (via the
/// `CompQuadForm` R package) and a Monte-Carlo path. This implementation
/// provides the **Monte-Carlo approximation only** — no pure-Rust `Imhof`
/// equivalent exists without adding a new crate dependency. The MC path
/// converges as `n_sim → ∞`; the default `n_sim = 999` matches `fdars`'
/// permutation-test convention (use 10 000 for publication-quality bands).
///
/// **fPACF:** see [`functional_pacf`] for the Durbin-Levinson divergence note.
#[must_use = "returns functional ACF result; result should be examined"]
pub fn functional_acf(
    data: &FdMatrix,
    argvals: &[f64],
    max_lag: Option<usize>,
    n_sim: usize,
    ci: f64,
    seed: u64,
) -> Result<FacfResult, FdarError> {
    use nalgebra::DMatrix;

    let (n, m) = validate_fts_input(data, argvals)?;

    // Resolve max_lag default: min(20, N/4), floored at 1.
    let ml = match max_lag {
        Some(0) => {
            return Err(FdarError::InvalidParameter {
                parameter: "max_lag",
                message: "must be >= 1".to_string(),
            });
        }
        Some(v) => v,
        None => 20usize.min(n / 4).max(1),
    };

    // Ensure we have at least ml+1 curves for the lag-ml autocovariance.
    if ml + 1 > n {
        return Err(FdarError::InvalidDimension {
            parameter: "max_lag",
            expected: format!("<= {}", n - 1),
            actual: format!("{ml}"),
        });
    }

    let weights = simpsons_weights(argvals);
    let xbar = mean_curve(data, n, m);

    // C_0 — needed for normalization, eigendecomposition, and the band.
    let c0 = autocovariance_matrix(data, &xbar, 0, n, m);
    let normalization = acf_normalization(&c0, m, argvals)?;

    // Compute fACF values at each lag.
    let mut lags = Vec::with_capacity(ml);
    let mut acf_vals = Vec::with_capacity(ml);
    for h in 1..=ml {
        let c_h = autocovariance_matrix(data, &xbar, h, n, m);
        let norm_sq = hs_norm_sq(&c_h, m, &weights);
        let rho_h = norm_sq.sqrt() / normalization;
        lags.push(h as u32);
        acf_vals.push(rho_h);
    }

    // Eigendecompose the weight-scaled C_0 for the MC white-noise band.
    //
    // The HS norm uses `Σ_{j1,j2} c_h[j1+j2*m]² * w[j1] * w[j2]` (L2 integral
    // approximation with quadrature weights). The limiting distribution of the
    // band statistic therefore involves eigenvalues of the weight-scaled
    // covariance operator: `C_0_scaled[j1,j2] = c0[j1+j2*m] * sqrt(w[j1]) * sqrt(w[j2])`.
    // These are the eigenvalues of W^{1/2} C_0 W^{1/2} and are in the same unit
    // as the HS norm. (Without weight-scaling, the raw eigenvalues of the m×m
    // matrix C_0 are O(n * variance) rather than O(variance), producing an
    // over-inflated band.)
    let mut c0_scaled = vec![0.0f64; m * m];
    for j1 in 0..m {
        let sw1 = weights[j1].sqrt();
        for j2 in 0..m {
            c0_scaled[j1 + j2 * m] = c0[j1 + j2 * m] * sw1 * weights[j2].sqrt();
        }
    }
    // Symmetrise defensively (should already be symmetric up to fp noise).
    let mut c0_mat = DMatrix::from_column_slice(m, m, &c0_scaled);
    for j1 in 0..m {
        for j2 in (j1 + 1)..m {
            let avg = 0.5 * (c0_mat[(j1, j2)] + c0_mat[(j2, j1)]);
            c0_mat[(j1, j2)] = avg;
            c0_mat[(j2, j1)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(c0_mat);
    let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
    // Sort descending.
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    // Truncate to eigenvalues with lambda_j / lambda_max > 1e-4.
    let lambda_max = eigenvalues.first().copied().unwrap_or(0.0);
    let truncated: Vec<f64> = eigenvalues
        .into_iter()
        .filter(|&lj| lj > 0.0 && lambda_max > 0.0 && lj / lambda_max > 1e-4)
        .collect();

    // Compute MC band threshold (same for every lag under the white-noise null).
    let band = if truncated.is_empty() {
        0.0
    } else {
        let q = mc_band_threshold(&truncated, n, n_sim, ci, seed);
        q.sqrt() / normalization
    };

    let upper_band = vec![band; ml];

    // Durbin-Levinson fPACF over the acf sequence.
    let pacf = durbin_levinson_pacf(&acf_vals);

    Ok(FacfResult {
        lags,
        acf: acf_vals,
        pacf,
        upper_band,
    })
}

/// Functional partial autocorrelation of a curve series.
///
/// A thin wrapper around [`functional_acf`] that returns the same fully-populated
/// [`FacfResult`] (acf, pacf, upper_band all present). Calling `functional_pacf` is
/// equivalent to calling `functional_acf` — both return the fACF and fPACF together
/// because they share the same estimation pass.
///
/// # Arguments
///
/// Same as [`functional_acf`].
///
/// # Errors
///
/// Same as [`functional_acf`].
///
/// # Divergence from R fdaACF
///
/// The `fdaACF` package computes fPACF via a residual-cross-covariance approach:
/// for each order p it fits an ARH(p-1) model forward and backward using FPCA,
/// then computes the L2 norm of the cross-covariance of the residuals.
///
/// This implementation uses the **classical scalar Durbin-Levinson recursion**
/// applied to the sequence ρ_1, ρ_2, …, ρ_{max_lag}. This is a simpler, valid
/// approximation that gives the PACF of the scalar ACF sequence rather than the
/// operator-valued PACF. It is suitable for diagnosing AR(p) vs MA(q) structure
/// (cutoff-after-order-p pattern visible in fPACF, cutoff-after-q in fACF).
#[must_use = "returns functional PACF result; result should be examined"]
pub fn functional_pacf(
    data: &FdMatrix,
    argvals: &[f64],
    max_lag: Option<usize>,
    n_sim: usize,
    ci: f64,
    seed: u64,
) -> Result<FacfResult, FdarError> {
    functional_acf(data, argvals, max_lag, n_sim, ci, seed)
}

/// Functional first-difference operator.
///
/// Computes the first-order difference of a time-ordered curve series, mirroring
/// `ftsa::diff.fts` with `lag = 1`. For a series of N curves on an m-point grid,
/// the output is an `(N-1) × m` matrix where:
///
/// ```text
/// D[i, j] = data[(i+1, j)] - data[(i, j)]    for i in 0..=(N-2), j in 0..=(m-1)
/// ```
///
/// # Round-trip tolerance
///
/// The original series is recoverable from `D` and the first curve via a running
/// cumulative sum:
///
/// ```text
/// reconstructed[0, j] = data[(0, j)]
/// reconstructed[i, j] = reconstructed[i-1, j] + D[i-1, j]    for i >= 1
/// ```
///
/// This round-trips within machine precision (|reconstructed[i,j] - data[i,j]| < 1e-10
/// for typical f64 inputs). The tolerance is tight because first-differencing and
/// cumulative-summation are exact inverse operations in floating-point arithmetic
/// (no approximation is involved, only floating-point rounding).
///
/// # Higher-order differencing
///
/// Only order-1 (lag-1) differencing is provided. Higher-order or lag-d
/// differencing can be achieved by applying `functional_difference` repeatedly:
/// `functional_difference(&functional_difference(&data)?)?`. Convenience wrappers
/// for `order` and `lag` parameters are a deferred extension.
///
/// # Arguments
///
/// * `data` — Time-ordered functional observations (`N × m`, column-major). Rows
///   are curves ordered from earliest to latest.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] — `data` has fewer than 2 rows (N < 2).
///   Differencing a single curve or empty matrix is undefined.
///
/// # Examples
///
/// ```rust
/// use fdars_core::{FdMatrix, functional_difference};
///
/// // Three curves on a 5-point grid.
/// let mut data = FdMatrix::zeros(3, 5);
/// for i in 0..3 {
///     for j in 0..5 {
///         data[(i, j)] = (i as f64) * (j as f64 + 1.0);
///     }
/// }
/// let diff = functional_difference(&data).unwrap();
/// assert_eq!(diff.shape(), (2, 5)); // N-1 rows
/// ```
#[must_use = "returns first-difference curve series; result should be examined"]
pub fn functional_difference(data: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">= 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    let mut out = FdMatrix::zeros(n - 1, m);
    for i in 0..(n - 1) {
        for j in 0..m {
            out[(i, j)] = data[(i + 1, j)] - data[(i, j)];
        }
    }
    Ok(out)
}

/// Functional stationarity test (KPSS-style partial-sum statistic with Monte-Carlo p-value).
///
/// Tests H₀: the functional time series is (second-order) stationary. The test
/// statistic is a KPSS-style partial-sum functional norm:
///
/// ```text
/// T = (1/N²) Σ_{k=1}^{N} ‖S_k‖²_L2
/// ```
///
/// where `S_k[j] = Σ_{i=0}^{k-1} (x_i[j] - x̄[j])` are the partial sums of the
/// centered curves and `‖·‖²_L2 = Σ_j · · w[j]` uses Simpson quadrature weights.
/// A large T indicates non-stationarity (growing partial sums signal a trend).
///
/// # p-value computation
///
/// The Monte-Carlo p-value is computed by randomly permuting the row (curve) order
/// `n_perm` times using a seeded Fisher-Yates shuffle, recomputing T for each
/// permutation, and counting `n_ge` permutations where `perm_T >= observed_T`:
///
/// ```text
/// p_value = (n_ge + 1) / (n_perm + 1)
/// ```
///
/// This is a valid permutation p-value regardless of any long-run-variance
/// normalisation (see DIVERGENCE note below).
///
/// # Reproducibility
///
/// A single `StdRng::seed_from_u64(seed)` instance is used for all `n_perm`
/// shuffles. The same `(data, argvals, n_perm, seed)` tuple always produces a
/// bit-identical `StationarityResult`.
///
/// # Arguments
///
/// * `data` — Time-ordered functional observations (`N × m`, column-major).
/// * `argvals` — Evaluation points on the common grid (length `m`).
/// * `n_perm` — Number of row permutations for the Monte-Carlo p-value (≥ 1).
/// * `seed` — Deterministic RNG seed.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] — `data` is empty or `argvals.len() != m`.
/// * [`FdarError::InvalidParameter`] — `n_perm == 0`.
///
/// # DIVERGENCE / ASSUMED: normalization constant
///
/// The `ftsa::T_stationary` implementation (Horváth, Kokoszka, Rice 2014,
/// *Journal of Econometrics* 179:66–82) includes a long-run-variance normalization
/// factor: the statistic is scaled by the inverse of the estimated long-run
/// covariance operator norm, which controls the null distribution. This
/// normalization is not pinned from the publicly available `ftsa` documentation
/// alone (it requires reading the HKR 2014 paper or `ftsa` source code directly).
///
/// This implementation uses the **unnormalized KPSS-style partial-sum statistic**
/// with a **pure seeded-permutation p-value**. The permutation p-value is valid
/// regardless of the normalization constant: permuting the row order destroys
/// temporal dependence and simulates the null distribution of T for this specific
/// dataset. The trade-off is that the raw statistic value is not directly comparable
/// across datasets of different variance. Implementing the exact HKR 2014
/// long-run-variance normalization is a documented future-precision item.
#[must_use = "returns stationarity test result; result should be examined"]
pub fn stationarity_test(
    data: &FdMatrix,
    argvals: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<super::StationarityResult, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "must be >= 1".to_string(),
        });
    }

    let weights = simpsons_weights(argvals);
    let xbar = mean_curve(data, n, m);

    // Compute centered curves as a flat row-major buffer for efficient permutation.
    // centered[i * m + j] = data[(i,j)] - xbar[j]
    let mut centered = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            centered[i * m + j] = data[(i, j)] - xbar[j];
        }
    }

    // Compute the KPSS-style partial-sum statistic T for a given row order.
    let stationarity_statistic = |row_order: &[usize]| -> f64 {
        // S_k[j] = Σ_{i=0}^{k-1} centered[row_order[i]][j]
        // T = (1/N²) Σ_{k=1}^{N} Σ_j S_k[j]² * w[j]
        let mut partial_sum = vec![0.0f64; m];
        let mut t = 0.0f64;
        let inv_n2 = 1.0 / (n * n) as f64;
        for k in 0..n {
            let row = row_order[k];
            for j in 0..m {
                partial_sum[j] += centered[row * m + j];
            }
            // Add ‖S_{k+1}‖²_L2 to T.
            let mut norm_sq = 0.0f64;
            for j in 0..m {
                norm_sq += partial_sum[j] * partial_sum[j] * weights[j];
            }
            t += norm_sq;
        }
        t * inv_n2
    };

    // Natural order for observed statistic.
    let natural_order: Vec<usize> = (0..n).collect();
    let observed_t = stationarity_statistic(&natural_order);

    // Permutation loop: single shared RNG seeded once.
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut row_indices: Vec<usize> = (0..n).collect();
    let mut n_ge = 0usize;
    for _ in 0..n_perm {
        // Fisher-Yates in-place shuffle.
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            row_indices.swap(i, j);
        }
        let perm_t = stationarity_statistic(&row_indices);
        if perm_t >= observed_t {
            n_ge += 1;
        }
    }

    let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
    Ok(super::StationarityResult {
        statistic: observed_t,
        p_value,
        n_perm,
    })
}

/// Bartlett kernel-sandwich long-run covariance estimator.
///
/// Estimates the m×m long-run covariance operator of a functional time series
/// using the Bartlett (triangular) kernel:
///
/// ```text
/// Ĉ_LRC(s,t) = Ĉ_0(s,t)  +  Σ_{h=1}^{b-1}  (1 - h/b) * (Ĉ_h(s,t) + Ĉ_h^T(s,t))
/// ```
///
/// where `Ĉ_h` is the lag-h sample autocovariance operator and `b` is the bandwidth.
///
/// # Arguments
///
/// * `data` — Time-ordered functional observations (`N × m`, column-major).
///   Rows are curves ordered from earliest to latest.
/// * `argvals` — Evaluation points on the common grid (length `m`).
/// * `bandwidth` — Number of lags to include (exclusive upper bound in the Bartlett
///   sum). `None` uses the default `⌊N^{1/3}⌋` (standard HAC cube-root rule).
///   `Some(0)` reduces the estimator to the lag-0 sample covariance operator C_0.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] — `data` is empty or `argvals.len() != m`.
///
/// # Algorithm notes
///
/// * **Bartlett kernel only.** Flat-top (Andrews) and Parzen kernels are deferred.
/// * **Default bandwidth `⌊N^{1/3}⌋`** — the standard HAC rule of thumb (see ftsa
///   `long_run_covariance_estimation`). For small N the default may be 0 or 1,
///   which is mathematically correct (reduces to C_0 or C_0 + C_1-terms).
/// * **bandwidth 0 → C_0.** `long_run_covariance(data, argvals, Some(0))` is
///   element-wise identical to the lag-0 sample covariance operator, enabling the
///   plan-34-01 `autocovariance_matrix` helper to be the sole computation spine.
/// * **Reuses `autocovariance_matrix`.** This function calls the same `pub(crate)`
///   helper used by `functional_acf` for every lag, adding no new subsystem.
///   The loop guard `h < bandwidth && h < n` (T-34-06) prevents out-of-bounds lag.
/// * **Symmetry.** Because `Ĉ_{-h} = Ĉ_h^T` for a stationary series, the
///   accumulator adds both `w_h * Ĉ_h` and `w_h * Ĉ_h^T`, producing a symmetric
///   m×m output matrix.
///
/// # R baseline divergence
///
/// `ftsa::long_run_covariance_estimation` supports multiple kernel types (Bartlett,
/// Parzen) and an adaptive bandwidth selector. This implementation provides the
/// Bartlett kernel only and the fixed `⌊N^{1/3}⌋` default. The returned matrix is
/// directly comparable; only the bandwidth selection rule may differ for small N.
#[must_use = "returns long-run covariance result; result should be examined"]
pub fn long_run_covariance(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: Option<usize>,
) -> Result<super::LongRunCovResult, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;

    // Resolve bandwidth: None → ⌊N^{1/3}⌋; Some(b) → b (clamped to n-1 silently).
    let resolved_bandwidth = match bandwidth {
        None => (n as f64).cbrt().floor() as usize,
        Some(b) => b,
    };

    let xbar = mean_curve(data, n, m);

    // C_0 is always the base of the accumulator.
    let c0 = autocovariance_matrix(data, &xbar, 0, n, m);

    if resolved_bandwidth == 0 {
        // Bandwidth 0 → return C_0 unchanged (locked CONTEXT.md decision).
        return Ok(super::LongRunCovResult {
            cov_matrix: c0,
            m,
            bandwidth: 0,
            n_curves: n,
        });
    }

    // Accumulate: start with C_0, then add w_h * (C_h + C_h^T) for h = 1..bandwidth.
    let mut acc = c0;
    // h = bandwidth gives Bartlett weight 0 (Common Pitfalls §5); loop is exclusive.
    // Also guard h < n so autocovariance_matrix never receives h >= n.
    let max_h = resolved_bandwidth.min(n - 1);
    for h in 1..max_h {
        let w_h = 1.0 - (h as f64) / (resolved_bandwidth as f64);
        let c_h = autocovariance_matrix(data, &xbar, h, n, m);
        // Add w_h * C_h and w_h * C_h^T into the accumulator.
        for j2 in 0..m {
            for j1 in 0..m {
                let val = w_h * c_h[j1 + j2 * m];
                // C_h term: acc[j1, j2] += w_h * c_h[j1, j2]
                acc[j1 + j2 * m] += val;
                // C_h^T term: acc[j2, j1] += w_h * c_h[j1, j2]  (i.e. c_h^T[j2,j1] = c_h[j1,j2])
                acc[j2 + j1 * m] += val;
            }
        }
    }

    Ok(super::LongRunCovResult {
        cov_matrix: acc,
        m,
        bandwidth: resolved_bandwidth,
        n_curves: n,
    })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::{generate_gaussian_process, CovKernel};
    use crate::test_helpers::uniform_grid;

    // ── Test data helpers ──────────────────────────────────────────────────

    /// Generate `n` i.i.d. white-noise functional curves on a uniform grid of
    /// `m` points using `CovKernel::WhiteNoise { variance: 1.0 }`.
    fn make_whitenoise_curves(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let kernel = CovKernel::WhiteNoise { variance: 1.0 };
        let gp = generate_gaussian_process(n, &kernel, &argvals, None, Some(seed)).unwrap();
        (gp.samples, argvals)
    }

    /// Generate a functional AR(1) series: `X_i = 0.8 * X_{i-1} + eps_i`
    /// where each `eps_i` is a smooth GP sample.
    fn make_ar1_curves(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let kernel = CovKernel::Gaussian {
            length_scale: 0.3,
            variance: 1.0,
        };
        // Generate n innovation curves.
        let eps = generate_gaussian_process(n, &kernel, &argvals, None, Some(seed))
            .unwrap()
            .samples;
        let mut data = FdMatrix::zeros(n, m);
        // Initialise with the first innovation.
        for j in 0..m {
            data[(0, j)] = eps[(0, j)];
        }
        for i in 1..n {
            for j in 0..m {
                data[(i, j)] = 0.8 * data[(i - 1, j)] + eps[(i, j)];
            }
        }
        (data, argvals)
    }

    // ── Task 1 tests: fACF skeleton ────────────────────────────────────────

    /// fACF lags start at 1 (lag 0 not included) and acf is finite + non-negative.
    #[test]
    fn facf_lags_start_at_one_and_finite() {
        let (data, argvals) = make_whitenoise_curves(60, 20, 1);
        let result = functional_acf(&data, &argvals, None, 200, 0.95, 42).unwrap();
        assert!(!result.lags.is_empty(), "lags must be non-empty");
        assert_eq!(result.lags[0], 1, "first lag must be 1");
        assert_eq!(result.acf.len(), result.lags.len());
        for &rho in &result.acf {
            assert!(rho.is_finite(), "all fACF values must be finite");
            assert!(rho >= 0.0, "all fACF values must be non-negative (L2 norm)");
        }
    }

    /// Autocovariance matrix at h=0 is symmetric.
    #[test]
    fn autocovariance_c0_is_symmetric() {
        let (data, _argvals) = make_whitenoise_curves(40, 10, 2);
        let (n, m) = data.shape();
        let xbar = mean_curve(&data, n, m);
        let c0 = autocovariance_matrix(&data, &xbar, 0, n, m);
        for j1 in 0..m {
            for j2 in 0..m {
                let c_j1j2 = c0[j1 + j2 * m];
                let c_j2j1 = c0[j2 + j1 * m];
                assert!(
                    (c_j1j2 - c_j2j1).abs() < 1e-12,
                    "C_0[{j1},{j2}] = {c_j1j2} != C_0[{j2},{j1}] = {c_j2j1}"
                );
            }
        }
    }

    /// Error on empty data.
    #[test]
    fn error_empty_data() {
        let argvals = uniform_grid(20);
        let empty = FdMatrix::zeros(0, 20);
        assert!(matches!(
            functional_acf(&empty, &argvals, None, 99, 0.95, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    /// Error when argvals length mismatches data columns.
    #[test]
    fn error_argvals_mismatch() {
        let (data, _) = make_whitenoise_curves(30, 20, 3);
        let bad_argvals = uniform_grid(15); // wrong length
        assert!(matches!(
            functional_acf(&data, &bad_argvals, None, 99, 0.95, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    /// Error when max_lag >= n (too few curves).
    #[test]
    fn error_too_few_curves() {
        let (data, argvals) = make_whitenoise_curves(5, 10, 4);
        // max_lag = 5 requires at least 6 curves.
        assert!(matches!(
            functional_acf(&data, &argvals, Some(5), 99, 0.95, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }

    /// Identical seeds produce bit-identical results.
    #[test]
    fn deterministic_seed() {
        let (data, argvals) = make_whitenoise_curves(50, 20, 1);
        let r1 = functional_acf(&data, &argvals, None, 200, 0.95, 42).unwrap();
        let r2 = functional_acf(&data, &argvals, None, 200, 0.95, 42).unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical FacfResult");
    }

    // ── Task 2 tests: MC white-noise band ──────────────────────────────────

    /// On i.i.d. white-noise curves all nonzero-lag fACF values are inside the band.
    #[test]
    fn facf_whitenoise_inside_band() {
        // Use n=80 to make white-noise property robust.
        let (data, argvals) = make_whitenoise_curves(80, 20, 7);
        let result = functional_acf(&data, &argvals, Some(10), 1000, 0.95, 99).unwrap();
        assert_eq!(result.upper_band.len(), result.lags.len());
        let mut all_inside = true;
        for (h, (&rho, &band)) in result.acf.iter().zip(result.upper_band.iter()).enumerate() {
            assert!(band.is_finite() && band > 0.0, "band must be positive at lag {}", h + 1);
            if rho > band {
                all_inside = false;
            }
        }
        assert!(
            all_inside,
            "on i.i.d. white-noise all fACF lags should be inside the 95% band"
        );
    }

    /// On a functional AR(1) series the lag-1 fACF exceeds the white-noise band.
    #[test]
    fn facf_ar1_exceeds_band() {
        // Larger n for a clearer AR(1) signal.
        let (data, argvals) = make_ar1_curves(120, 20, 13);
        let result = functional_acf(&data, &argvals, Some(5), 1000, 0.95, 77).unwrap();
        let lag1_acf = result.acf[0];
        let band = result.upper_band[0];
        assert!(
            lag1_acf > band,
            "lag-1 fACF ({lag1_acf:.4}) must exceed the 95% band ({band:.4}) for AR(1) data"
        );
    }

    // ── Task 3 tests: Durbin-Levinson fPACF ───────────────────────────────

    /// Durbin-Levinson: pacf[0] == rho[0] for a single-element input.
    #[test]
    fn dl_pacf_single_rho() {
        let rho = [0.6];
        let pacf = durbin_levinson_pacf(&rho);
        assert_eq!(pacf.len(), 1);
        assert!((pacf[0] - 0.6).abs() < 1e-12, "pacf[1] must equal rho[1]");
    }

    /// On a functional AR(1) series the fPACF shows a large lag-1 value and
    /// near-zero values at higher lags (cutoff-after-order-1 shape).
    #[test]
    fn fpacf_ar1_cutoff() {
        let (data, argvals) = make_ar1_curves(120, 20, 17);
        let result = functional_pacf(&data, &argvals, Some(5), 1000, 0.95, 55).unwrap();
        assert_eq!(result.pacf.len(), result.lags.len());
        let lag1_pacf = result.pacf[0].abs();
        // Lag-2 onward should be materially smaller than lag-1.
        for (k, &v) in result.pacf.iter().enumerate().skip(1) {
            assert!(
                lag1_pacf > v.abs() * 1.5,
                "AR(1) fPACF: lag-1 |pacf| ({lag1_pacf:.4}) should exceed lag-{} |pacf| ({:.4}) by 1.5x",
                k + 1,
                v.abs()
            );
        }
    }

    /// functional_pacf returns populated pacf (not all zeros) on AR(1) data.
    #[test]
    fn fpacf_returns_populated_pacf() {
        let (data, argvals) = make_ar1_curves(80, 20, 19);
        let result = functional_pacf(&data, &argvals, Some(4), 500, 0.95, 33).unwrap();
        assert_eq!(result.pacf.len(), result.lags.len());
        assert!(
            result.pacf.iter().any(|&v| v.abs() > 0.05),
            "fPACF should have at least one nonzero entry on AR(1) data"
        );
    }

    // ── Task 1 tests: functional_difference ───────────────────────────────

    /// Differencing an N×m matrix produces an (N-1)×m matrix that round-trips
    /// via running cumulative sum within 1e-10.
    #[test]
    fn diff_roundtrip() {
        let m = 15usize;
        let n = 8usize;
        let argvals = uniform_grid(m);
        // Deterministic analytic data: data[(i,j)] = sin(i + argvals[j]).
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                data[(i, j)] = (i as f64 + t).sin();
            }
        }
        let diff = functional_difference(&data).expect("functional_difference should succeed");
        assert_eq!(diff.shape(), (n - 1, m), "output shape must be (N-1) x m");

        // Reconstruct via cumulative sum from row 0.
        let mut recon = FdMatrix::zeros(n, m);
        for j in 0..m {
            recon[(0, j)] = data[(0, j)];
        }
        for i in 1..n {
            for j in 0..m {
                recon[(i, j)] = recon[(i - 1, j)] + diff[(i - 1, j)];
            }
        }
        // Verify round-trip within 1e-10.
        for i in 0..n {
            for j in 0..m {
                let err = (recon[(i, j)] - data[(i, j)]).abs();
                assert!(
                    err < 1e-10,
                    "round-trip error at ({i},{j}): {err} exceeds 1e-10"
                );
            }
        }
    }

    /// functional_difference errors with InvalidDimension when N < 2.
    #[test]
    fn diff_too_few_rows() {
        let m = 10usize;
        // 1-row matrix.
        let one_row = FdMatrix::zeros(1, m);
        assert!(
            matches!(
                functional_difference(&one_row),
                Err(FdarError::InvalidDimension { parameter: "data", .. })
            ),
            "1-row matrix should return InvalidDimension"
        );
        // 0-row matrix (edge case — hits m==0 check in validate_fts_input if used,
        // but functional_difference checks n directly before touching argvals).
        let zero_row = FdMatrix::zeros(0, m);
        assert!(
            matches!(
                functional_difference(&zero_row),
                Err(FdarError::InvalidDimension { parameter: "data", .. })
            ),
            "0-row matrix should return InvalidDimension"
        );
    }

    // ── LRC tests (plan 34-03): long_run_covariance ───────────────────────

    /// bandwidth Some(0) must return exactly the lag-0 sample covariance C_0
    /// (element-wise within 1e-12).
    #[test]
    fn lrc_bandwidth_zero() {
        let (data, argvals) = make_whitenoise_curves(40, 10, 55);
        let (n, m) = data.shape();
        let xbar = mean_curve(&data, n, m);
        let c0 = autocovariance_matrix(&data, &xbar, 0, n, m);
        let result =
            super::super::acf::long_run_covariance(&data, &argvals, Some(0)).unwrap();
        assert_eq!(result.bandwidth, 0, "bandwidth field must be 0");
        assert_eq!(result.m, m, "m field must match data columns");
        assert_eq!(result.n_curves, n, "n_curves must match data rows");
        assert_eq!(result.cov_matrix.len(), m * m, "cov_matrix must be m×m");
        for (idx, (&lrc_val, &c0_val)) in result.cov_matrix.iter().zip(c0.iter()).enumerate() {
            assert!(
                (lrc_val - c0_val).abs() < 1e-12,
                "LRC at index {idx}: {lrc_val} != C_0 {c0_val} (bandwidth=0 must equal C_0)"
            );
        }
    }

    /// The returned cov_matrix is symmetric within 1e-10.
    #[test]
    fn lrc_symmetric() {
        let (data, argvals) = make_ar1_curves(60, 10, 66);
        let result =
            super::super::acf::long_run_covariance(&data, &argvals, None).unwrap();
        let m = result.m;
        for j1 in 0..m {
            for j2 in 0..m {
                let upper = result.cov_matrix[j1 + j2 * m];
                let lower = result.cov_matrix[j2 + j1 * m];
                assert!(
                    (upper - lower).abs() < 1e-10,
                    "LRC[{j1},{j2}]={upper} != LRC[{j2},{j1}]={lower} (must be symmetric)"
                );
            }
        }
    }

    /// bandwidth None returns a finite m×m matrix with the correct default bandwidth.
    #[test]
    fn lrc_default_bandwidth() {
        let n = 50usize;
        let m = 10usize;
        let (data, argvals) = make_whitenoise_curves(n, m, 77);
        let result =
            super::super::acf::long_run_covariance(&data, &argvals, None).unwrap();
        let expected_bw = (n as f64).cbrt().floor() as usize;
        assert_eq!(
            result.bandwidth, expected_bw,
            "default bandwidth must be ⌊N^{{1/3}}⌋ = {expected_bw}"
        );
        assert_eq!(result.m, m);
        assert_eq!(result.n_curves, n);
        assert_eq!(result.cov_matrix.len(), m * m);
        for &v in &result.cov_matrix {
            assert!(v.is_finite(), "all cov_matrix entries must be finite");
        }
    }

    // ── Task 2 tests: stationarity_test ──────────────────────────────────

    /// Stationary series (i.i.d. white-noise GP, no trend) should NOT be rejected
    /// at significance level 0.05 (p-value > 0.05) with a seeded permutation test.
    #[test]
    fn stat_test_stationary() {
        // Use n=60, m=20, 499 permutations, fixed seed for reproducibility.
        let (data, argvals) = make_whitenoise_curves(60, 20, 101);
        let result = stationarity_test(&data, &argvals, 499, 42).unwrap();
        assert!(
            result.p_value > 0.05,
            "stationary series should NOT be rejected at 0.05 (p = {:.4})",
            result.p_value
        );
        assert!(result.statistic.is_finite(), "statistic must be finite");
        assert_eq!(result.n_perm, 499);
    }

    /// Trended (non-stationary) series X_i(t) = i*t + GP_sample should be rejected
    /// at significance level 0.05 (p-value <= 0.05).
    #[test]
    fn stat_test_nonstationary() {
        let n = 50usize;
        let m = 20usize;
        let (gp_data, argvals) = make_whitenoise_curves(n, m, 202);
        // Add linear trend: X_i(t) = i * t + GP_sample.
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for (j, &t) in argvals.iter().enumerate() {
                data[(i, j)] = (i as f64) * t + gp_data[(i, j)];
            }
        }
        let result = stationarity_test(&data, &argvals, 499, 77).unwrap();
        assert!(
            result.p_value <= 0.05,
            "trended series should be rejected at 0.05 (p = {:.4})",
            result.p_value
        );
    }

    /// stationarity_test: bit-identical results for identical seed + inputs.
    #[test]
    fn stat_test_deterministic() {
        let (data, argvals) = make_whitenoise_curves(40, 15, 303);
        let r1 = stationarity_test(&data, &argvals, 199, 123).unwrap();
        let r2 = stationarity_test(&data, &argvals, 199, 123).unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical StationarityResult");
    }

    /// stationarity_test: error paths for n_perm==0 and empty input.
    #[test]
    fn stat_test_invalid() {
        let (data, argvals) = make_whitenoise_curves(30, 15, 1);
        // n_perm == 0 must return InvalidParameter.
        assert!(
            matches!(
                stationarity_test(&data, &argvals, 0, 1),
                Err(FdarError::InvalidParameter { parameter: "n_perm", .. })
            ),
            "n_perm == 0 must return InvalidParameter"
        );
        // Empty matrix must return InvalidDimension.
        let empty = FdMatrix::zeros(0, 15);
        assert!(
            matches!(
                stationarity_test(&empty, &argvals, 99, 1),
                Err(FdarError::InvalidDimension { .. })
            ),
            "empty matrix must return InvalidDimension"
        );
        // Argvals length mismatch must return InvalidDimension.
        let bad_argvals = uniform_grid(10);
        assert!(
            matches!(
                stationarity_test(&data, &bad_argvals, 99, 1),
                Err(FdarError::InvalidDimension { parameter: "argvals", .. })
            ),
            "argvals mismatch must return InvalidDimension"
        );
    }
}
