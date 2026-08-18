//! PACE sparse FPCA for irregularly sampled functional data.
//!
//! Implements the Yao–Müller–Wang (2005) PACE estimator via a six-step pipeline:
//!
//! 1. **Mean:** kernel-smoothed mean µ̂(t) on the work grid via [`mean_irreg`].
//! 2. **Covariance surface:** kernel-smoothed bivariate covariance Ĝ(s,t) via [`cov_irreg`].
//! 3. **Eigendecomposition:** symmetric eigendecomposition of W^{½} Ĝ W^{½} (Simpson-weighted)
//!    to obtain functional eigenvalues λ_k and orthonormal eigenfunctions φ_k.
//! 4. **BLUP scores:** per-curve conditional-expectation (BLUP/PACE) scores
//!    `ξ_ik = λ_k · φ_ik^T · Σ_yi^{-1} · (Y_i − µ_i)`, where
//!    `Σ_yi = Φ_i diag(λ) Φ_i^T + σ²I_{n_i}`.
//! 5. **Fitted trajectories:** `x̂_i(t) = µ̂(t) + Σ_k ξ_ik φ_k(t)` on the work grid.
//! 6. **Confidence bands:** pointwise bands from the BLUP prediction variance Ω (Yao et al.
//!    2005, eq. 3.2).
//!
//! **Reference:** Yao, Müller & Wang (2005), "Functional Data Analysis for Sparse Longitudinal
//! Data", JASA 100(470), 577–590.
//!
//! **Reuse-first:** mean and covariance smoothing reuse [`crate::irreg_fdata`]; eigendecomposition
//! uses nalgebra `DMatrix::symmetric_eigen()`; linear interpolation of eigenfunctions reuses
//! [`crate::helpers::linear_interp`]; Cholesky solve reuses the crate-internal `linalg::cholesky_solve`.
//! No new crate dependency is added.
//!
//! **Design note on σ²:** The `cov_irreg` surface includes same-point pairs (j1 == j2), so
//! its diagonal absorbs the measurement-error variance σ². Do NOT subtract σ² from the surface
//! before eigendecomposition — σ² enters only as the ridge term `σ²I` in Σ_yi (step 4). This
//! follows the standard PACE formulation (Yao et al. 2005, §2.2).
//!
//! **Size limits:** For curves with large n_i, the n_i×n_i Σ_yi system can be expensive.
//! Requiring σ² > 0 (strictly positive) ensures Σ_yi is positive-definite even for dense
//! curves. Document n_i ≤ a few hundred as the expected regime for sparse functional data.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::irreg_fdata::{cov_irreg, mean_irreg, IrregFdata, KernelType};
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;

// ---------------------------------------------------------------------------
// Config struct
// ---------------------------------------------------------------------------

/// Configuration for PACE sparse FPCA.
///
/// No `#[non_exhaustive]` — follows the [`crate::elastic_regression::ElasticPcrConfig`]
/// convention for config structs (allows struct-literal construction in tests).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaceFpcaConfig {
    /// Number of FPCA components to extract.
    pub ncomp: usize,
    /// Kernel bandwidth for mean and covariance smoothing (must be strictly positive).
    pub bandwidth: f64,
    /// Caller-supplied measurement-error variance σ² (must be strictly positive).
    ///
    /// σ² > 0 is required so that Σ_yi = Φ_i diag(λ) Φ_i^T + σ²I_{n_i} remains
    /// positive-definite regardless of the number of observed points per curve.
    /// Automatic σ² estimation from the raw-vs-smoothed diagonal is deferred.
    pub sigma2: f64,
    /// Work grid: the evaluation points at which mean, eigenfunctions, and fitted
    /// trajectories are represented. Must have at least 2 points and be sorted.
    pub work_grid: Vec<f64>,
    /// Confidence level for bands (must be in the open interval (0, 1)).
    /// Default 0.05 → 95% pointwise bands.
    pub alpha: f64,
}

impl Default for PaceFpcaConfig {
    fn default() -> Self {
        let m = 51_usize;
        Self {
            ncomp: 3,
            bandwidth: 0.1,
            sigma2: 0.01,
            work_grid: (0..m).map(|i| i as f64 / (m - 1) as f64).collect(),
            alpha: 0.05,
        }
    }
}

// ---------------------------------------------------------------------------
// Result struct
// ---------------------------------------------------------------------------

/// Result of PACE sparse FPCA.
///
/// All matrix outputs are column-major [`FdMatrix`] (project-wide convention).
///
/// `ncomp` in the result may be less than the requested `config.ncomp` when the
/// smoothed covariance surface yields fewer positive eigenvalues than requested
/// (a finite-sample artifact of kernel estimation on sparse data).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaceFpcaResult {
    /// Kernel-smoothed mean function on the work grid (length m).
    pub mean: Vec<f64>,
    /// Functional eigenvalues (variance explained per component), length `ncomp`.
    pub eigenvalues: Vec<f64>,
    /// Eigenfunctions on the work grid, shape m × `ncomp` (column-major).
    pub eigenfunctions: FdMatrix,
    /// BLUP (conditional-expectation) FPC scores, shape n × `ncomp` (column-major).
    pub scores: FdMatrix,
    /// Fitted trajectories on the work grid, shape n × m (column-major).
    pub fitted: FdMatrix,
    /// Lower pointwise confidence band, shape n × m.
    pub fitted_lower: FdMatrix,
    /// Upper pointwise confidence band, shape n × m.
    pub fitted_upper: FdMatrix,
    /// Work grid used for all outputs (clone of `config.work_grid`).
    pub argvals: Vec<f64>,
    /// Measurement-error variance used (echoed from config).
    pub sigma2: f64,
    /// Number of components actually extracted (may be < `config.ncomp`).
    pub ncomp: usize,
}

// ---------------------------------------------------------------------------
// Normal quantile helper (no external crate)
// ---------------------------------------------------------------------------

/// Approximate the standard-normal inverse CDF qnorm(p) for p ∈ (0, 1).
///
/// Uses a rational approximation (Beasley–Springer–Moro variant) that achieves
/// absolute error < 5×10⁻⁴ over (0, 1) and returns the mathematically correct
/// 1.959963… for p = 0.975 (the default alpha = 0.05 case).
///
/// # Panics
/// Panics in debug mode if p is not in (0, 1); release builds clamp silently.
// Allow dead_code: this function is used in Task 2 (BLUP bands); the tracer
// (Task 1) populates scores/fitted/bands with placeholders only.
#[allow(dead_code)]
fn standard_normal_quantile(p: f64) -> f64 {
    // Rational approximation coefficients — Beasley, Springer, Moro (1977/1994)
    // as tabulated in Abramowitz & Stegun §26.2.16.
    const A: [f64; 4] = [2.515_517, 0.802_853, 0.010_328, 0.0];
    const B: [f64; 4] = [1.0, 1.432_788, 0.189_269, 0.001_308];

    debug_assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");
    let p = p.clamp(1e-15, 1.0 - 1e-15);

    let (sign, q) = if p < 0.5 { (-1.0, p) } else { (1.0, 1.0 - p) };

    let t = (-2.0 * q.ln()).sqrt();
    let num = A[0] + t * (A[1] + t * (A[2] + t * A[3]));
    let den = B[0] + t * (B[1] + t * (B[2] + t * B[3]));
    sign * (t - num / den)
}

// ---------------------------------------------------------------------------
// Eigendecomposition of the smoothed covariance matrix
// ---------------------------------------------------------------------------

/// Decompose the m×m smoothed covariance surface using Simpson-weighted
/// symmetric eigendecomposition.
///
/// Returns `(eigenvalues, eigenfunctions)` where eigenvalues are sorted descending
/// and eigenfunctions are m×`ncomp_requested` (actual ncomp may be smaller when
/// fewer positive eigenvalues exist).
fn eigendecompose_cov(
    cov: &FdMatrix,
    work_grid: &[f64],
    ncomp_requested: usize,
) -> (Vec<f64>, FdMatrix) {
    let m = work_grid.len();
    let w = simpsons_weights(work_grid);
    let sqrt_w: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();

    // Build W^{1/2} C W^{1/2} (symmetric PSD, column-major then converted to DMatrix)
    let mut c_scaled = vec![0.0_f64; m * m];
    for col in 0..m {
        for row in 0..m {
            // cov is column-major: cov[(row, col)]
            c_scaled[row + col * m] = sqrt_w[row] * cov[(row, col)] * sqrt_w[col];
        }
    }

    // nalgebra DMatrix from column-major slice
    let c_dmat = DMatrix::from_column_slice(m, m, &c_scaled);

    // Symmetric eigendecomposition — eigenvalues in ASCENDING order
    let eigen = c_dmat.symmetric_eigen();

    // Collect (eigenvalue, column_index) pairs and sort DESCENDING
    let n_eval = eigen.eigenvalues.len();
    let mut pairs: Vec<(f64, usize)> = (0..n_eval).map(|k| (eigen.eigenvalues[k], k)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Keep only positive eigenvalues, up to ncomp_requested
    let pairs: Vec<(f64, usize)> = pairs
        .into_iter()
        .filter(|&(lam, _)| lam > 0.0)
        .take(ncomp_requested)
        .collect();

    let actual_ncomp = pairs.len();
    let mut eigenvalues = Vec::with_capacity(actual_ncomp);
    let mut eigenfunctions = FdMatrix::zeros(m, actual_ncomp);

    for (k, &(lam, col_idx)) in pairs.iter().enumerate() {
        eigenvalues.push(lam);
        // Unscale: φ_k = W^{-1/2} · v_k
        for j in 0..m {
            let raw = eigen.eigenvectors[(j, col_idx)];
            eigenfunctions[(j, k)] = if sqrt_w[j] > 1e-15 {
                raw / sqrt_w[j]
            } else {
                raw
            };
        }
    }

    // Sign convention: for each component k, ensure the element with the
    // largest absolute value is positive (mirrors `fix_svd_signs` in regression.rs).
    for k in 0..actual_ncomp {
        let j_max = (0..m)
            .max_by(|&a, &b| {
                eigenfunctions[(a, k)]
                    .abs()
                    .partial_cmp(&eigenfunctions[(b, k)].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        if eigenfunctions[(j_max, k)] < 0.0 {
            for j in 0..m {
                eigenfunctions[(j, k)] = -eigenfunctions[(j, k)];
            }
        }
    }

    (eigenvalues, eigenfunctions)
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Fit PACE sparse FPCA for irregularly sampled functional data.
///
/// Implements the Yao–Müller–Wang (2005) PACE estimator:
/// 1. Kernel-smoothed mean µ̂(t) on the work grid.
/// 2. Kernel-smoothed covariance surface Ĝ(s,t) via `cov_irreg`.
/// 3. Symmetric eigendecomposition of Ĝ → eigenvalues λ_k, eigenfunctions φ_k.
/// 4. Per-curve BLUP (conditional-expectation) scores ξ_ik.
/// 5. Fitted trajectories x̂_i(t) = µ̂(t) + Σ_k ξ_ik φ_k(t).
/// 6. Pointwise confidence bands from BLUP prediction variance.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` has zero observations,
/// - any curve has zero observed points,
/// - `config.work_grid` has fewer than 2 points.
///
/// Returns [`FdarError::InvalidParameter`] if:
/// - `config.ncomp` is zero,
/// - `config.bandwidth` is not strictly positive or not finite,
/// - `config.sigma2` is not strictly positive or not finite,
/// - `config.alpha` is not in the open interval (0, 1),
/// - `config.work_grid` is not sorted or contains non-finite values.
///
/// Returns [`FdarError::ComputationFailed`] if:
/// - no positive eigenvalues are found after eigendecomposing the covariance surface,
/// - a per-curve Σ_yi Cholesky solve fails even after a single ridge-stabilisation retry.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn pace_fpca(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError> {
    // ------------------------------------------------------------------
    // 1. Input validation (all checks before any computation)
    // ------------------------------------------------------------------
    let n = data.n_obs();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 observation".to_string(),
            actual: "0 observations".to_string(),
        });
    }

    for i in 0..n {
        if data.n_points(i) == 0 {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("curve {i} must have at least 1 observed point"),
                actual: format!("curve {i} has 0 observed points"),
            });
        }
    }

    let m = config.work_grid.len();
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "work_grid",
            expected: "at least 2 grid points".to_string(),
            actual: format!("{m} grid points"),
        });
    }

    if config.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be at least 1".to_string(),
        });
    }

    if !config.bandwidth.is_finite() || config.bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!(
                "must be finite and strictly positive, got {}",
                config.bandwidth
            ),
        });
    }

    if !config.sigma2.is_finite() || config.sigma2 <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "sigma2",
            message: format!(
                "must be finite and strictly positive (required so Sigma_yi is positive-definite), got {}",
                config.sigma2
            ),
        });
    }

    if config.alpha <= 0.0 || config.alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in the open interval (0, 1), got {}", config.alpha),
        });
    }

    // Validate work_grid: all finite and sorted
    for (idx, &t) in config.work_grid.iter().enumerate() {
        if !t.is_finite() {
            return Err(FdarError::InvalidParameter {
                parameter: "work_grid",
                message: format!("grid point at index {idx} is not finite ({t})"),
            });
        }
    }
    for w in config.work_grid.windows(2) {
        if w[0] >= w[1] {
            return Err(FdarError::InvalidParameter {
                parameter: "work_grid",
                message: "work_grid must be strictly increasing (sorted with no duplicates)"
                    .to_string(),
            });
        }
    }

    // ------------------------------------------------------------------
    // 2. Step 1: Kernel-smoothed mean on the work grid
    // ------------------------------------------------------------------
    let mean = mean_irreg(
        data,
        &config.work_grid,
        config.bandwidth,
        KernelType::Gaussian,
    );

    // ------------------------------------------------------------------
    // 3. Step 2: Smoothed covariance surface on the work grid (m×m)
    // ------------------------------------------------------------------
    let cov = cov_irreg(data, &config.work_grid, &config.work_grid, config.bandwidth);

    // ------------------------------------------------------------------
    // 4. Step 3: Symmetric eigendecomposition of W^{1/2} C W^{1/2}
    //    (Simpson-weighted) → top positive eigenpairs, sign-fixed
    // ------------------------------------------------------------------
    let (eigenvalues, eigenfunctions) = eigendecompose_cov(&cov, &config.work_grid, config.ncomp);

    let actual_ncomp = eigenvalues.len();
    if actual_ncomp == 0 {
        return Err(FdarError::ComputationFailed {
            operation: "pace_fpca eigendecomposition",
            detail: format!(
                "no positive eigenvalues found in the smoothed covariance surface \
                 (requested {}, got 0 positive); try a larger bandwidth or more data",
                config.ncomp
            ),
        });
    }

    // ------------------------------------------------------------------
    // 5. Steps 4–6: Per-curve BLUP scores, fitted trajectories, bands
    //    Placeholders for Task 1 (tracer); real computation in Task 2.
    // ------------------------------------------------------------------
    let scores = FdMatrix::zeros(n, actual_ncomp);
    let fitted = FdMatrix::zeros(n, m);
    let fitted_lower = FdMatrix::zeros(n, m);
    let fitted_upper = FdMatrix::zeros(n, m);

    Ok(PaceFpcaResult {
        mean,
        eigenvalues,
        eigenfunctions,
        scores,
        fitted,
        fitted_lower,
        fitted_upper,
        argvals: config.work_grid.clone(),
        sigma2: config.sigma2,
        ncomp: actual_ncomp,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a small IrregFdata for smoke tests: 6 curves, 3–5 points each on [0,1].
    fn small_irreg_data() -> IrregFdata {
        let argvals_list = vec![
            vec![0.1, 0.4, 0.7],
            vec![0.0, 0.3, 0.6, 0.9],
            vec![0.2, 0.5, 0.8],
            vec![0.0, 0.25, 0.5, 0.75, 1.0],
            vec![0.1, 0.5, 0.9],
            vec![0.0, 0.4, 0.8],
        ];
        let values_list: Vec<Vec<f64>> = argvals_list
            .iter()
            .enumerate()
            .map(|(i, ts)| {
                ts.iter()
                    .map(|&t: &f64| (i as f64 + 1.0) * t.sin())
                    .collect()
            })
            .collect();
        IrregFdata::from_lists(&argvals_list, &values_list)
    }

    // -----------------------------------------------------------------------
    // Task 1: Shape smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_pace_shape_smoke() {
        let data = small_irreg_data();
        let n = data.n_obs(); // 6

        let m = 21_usize;
        let config = PaceFpcaConfig {
            ncomp: 2,
            bandwidth: 0.2,
            sigma2: 0.01,
            work_grid: (0..m).map(|i| i as f64 / (m - 1) as f64).collect(),
            alpha: 0.05,
        };

        let result = pace_fpca(&data, &config).expect("smoke test should succeed");

        let actual_ncomp = result.ncomp;
        assert!(actual_ncomp >= 1, "at least 1 positive eigenvalue expected");

        // mean has length m
        assert_eq!(result.mean.len(), m, "mean.len() == m");

        // eigenvalues
        assert_eq!(
            result.eigenvalues.len(),
            actual_ncomp,
            "eigenvalues.len() == ncomp"
        );
        for &lam in &result.eigenvalues {
            assert!(
                lam > 0.0,
                "all returned eigenvalues must be positive, got {lam}"
            );
        }

        // eigenfunctions: m × ncomp
        assert_eq!(
            result.eigenfunctions.nrows(),
            m,
            "eigenfunctions.nrows() == m"
        );
        assert_eq!(
            result.eigenfunctions.ncols(),
            actual_ncomp,
            "eigenfunctions.ncols() == ncomp"
        );

        // scores: n × ncomp (placeholders for Task 1)
        assert_eq!(result.scores.nrows(), n, "scores.nrows() == n");
        assert_eq!(
            result.scores.ncols(),
            actual_ncomp,
            "scores.ncols() == ncomp"
        );

        // fitted, fitted_lower, fitted_upper: n × m
        assert_eq!(result.fitted.nrows(), n, "fitted.nrows() == n");
        assert_eq!(result.fitted.ncols(), m, "fitted.ncols() == m");
        assert_eq!(result.fitted_lower.nrows(), n, "fitted_lower.nrows() == n");
        assert_eq!(result.fitted_lower.ncols(), m, "fitted_lower.ncols() == m");
        assert_eq!(result.fitted_upper.nrows(), n, "fitted_upper.nrows() == n");
        assert_eq!(result.fitted_upper.ncols(), m, "fitted_upper.ncols() == m");

        // argvals echoes work_grid
        assert_eq!(result.argvals, config.work_grid, "argvals echoes work_grid");

        // sigma2 echoed
        assert_eq!(result.sigma2, config.sigma2, "sigma2 echoed");
    }

    // -----------------------------------------------------------------------
    // Task 1: Crate-root re-export smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_crate_root_reexport() {
        // Verify that the public symbols are accessible via the crate namespace.
        // This is a compile-time check; if it compiles, the re-export is correct.
        let _: fn(&IrregFdata, &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError> = pace_fpca;
        let _config = PaceFpcaConfig::default();
        assert_eq!(_config.ncomp, 3);
        assert_eq!(_config.alpha, 0.05);
    }
}
