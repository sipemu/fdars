//! Function-on-function regression.
//!
//! Model: `Y(s) = α(s) + ∫ β(s,t) X(t) dt + ε(s)`
//!
//! Uses double FPCA: decompose both response Y and predictor X into
//! FPC scores, regress Y-scores on X-scores, then reconstruct β(s,t)
//! and fitted curves.
//!
//! # References
//!
//! - Ramsay, J. O. & Silverman, B. W. (2005). *Functional Data Analysis*, Ch. 16-17.
//! - Yao, F., Müller, H.-G. & Wang, J.-L. (2005). Functional linear regression
//!   analysis for longitudinal data. *Annals of Statistics*, 33(6), 2873--2903.
//! - Ivanescu, A. E., Staicu, A.-M., Scheipl, F. & Greven, S. (2015).
//!   Penalized function-on-function regression. *Computational Statistics*,
//!   30(2), 539--568.

use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Result of function-on-function regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FofResult {
    /// Intercept function α(s) (length m_y)
    pub intercept: Vec<f64>,
    /// Coefficient surface β(s,t) stored as (m_y x m_x) matrix
    pub beta_surface: FdMatrix,
    /// Fitted response curves (n x m_y)
    pub fitted: FdMatrix,
    /// Residual curves (n x m_y)
    pub residuals: FdMatrix,
    /// R-squared per response grid point (length m_y)
    pub r_squared_t: Vec<f64>,
    /// Overall R-squared (mean of pointwise R-squared)
    pub r_squared: f64,
    /// Number of predictor FPC components used
    pub ncomp_x: usize,
    /// Number of response FPC components used
    pub ncomp_y: usize,
    /// FPCA of predictor (for projection)
    pub fpca_x: FpcaResult,
    /// FPCA of response (for reconstruction)
    pub fpca_y: FpcaResult,
    /// Coefficient matrix B: Y-scores = X-scores * B (ncomp_x x ncomp_y)
    pub coef_matrix: FdMatrix,
}

// ---------------------------------------------------------------------------
// Main function
// ---------------------------------------------------------------------------

/// Function-on-function regression via double FPCA.
///
/// Decomposes both predictor and response via FPCA, regresses Y-scores
/// on X-scores using OLS, then reconstructs the coefficient surface
/// β(s,t) and fitted curves.
///
/// # Arguments
/// * `x_data` - Functional predictor (n x m_x)
/// * `y_data` - Functional response (n x m_y)
/// * `x_argvals` - Predictor grid (length m_x)
/// * `y_argvals` - Response grid (length m_y)
/// * `ncomp_x` - Number of predictor FPC components
/// * `ncomp_y` - Number of response FPC components
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `x_data` and `y_data` have
/// different row counts, or argvals lengths do not match column counts.
/// Returns [`FdarError::InvalidParameter`] if `ncomp_x` or `ncomp_y` is zero.
/// Returns [`FdarError::ComputationFailed`] if FPCA or OLS fails.
///
/// # References
///
/// Ramsay, J. O. & Silverman, B. W. (2005). *Functional Data Analysis*, Ch. 16-17.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fof_regression::fof_regression;
///
/// let (n, mx, my) = (25, 30, 20);
/// let x = FdMatrix::from_column_major(
///     (0..n * mx).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, mx,
/// ).unwrap();
/// let y = FdMatrix::from_column_major(
///     (0..n * my).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         0.5 * ((i + 1.0) * j * 0.15).cos()
///     }).collect(), n, my,
/// ).unwrap();
/// let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1) as f64).collect();
/// let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1) as f64).collect();
///
/// let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();
/// assert_eq!(fit.fitted.shape(), (n, my));
/// assert_eq!(fit.beta_surface.shape(), (my, mx));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fof_regression(
    x_data: &FdMatrix,
    y_data: &FdMatrix,
    x_argvals: &[f64],
    y_argvals: &[f64],
    ncomp_x: usize,
    ncomp_y: usize,
) -> Result<FofResult, FdarError> {
    let (n_x, m_x) = x_data.shape();
    let (n_y, m_y) = y_data.shape();

    if n_x != n_y {
        return Err(FdarError::InvalidDimension {
            parameter: "y_data",
            expected: format!("{n_x} rows (matching x_data)"),
            actual: format!("{n_y} rows"),
        });
    }
    let n = n_x;

    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "x_data",
            expected: "at least 3 observations".to_string(),
            actual: format!("{n}"),
        });
    }
    if x_argvals.len() != m_x {
        return Err(FdarError::InvalidDimension {
            parameter: "x_argvals",
            expected: format!("{m_x} elements"),
            actual: format!("{} elements", x_argvals.len()),
        });
    }
    if y_argvals.len() != m_y {
        return Err(FdarError::InvalidDimension {
            parameter: "y_argvals",
            expected: format!("{m_y} elements"),
            actual: format!("{} elements", y_argvals.len()),
        });
    }
    if ncomp_x == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_x",
            message: "must be >= 1".to_string(),
        });
    }
    if ncomp_y == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_y",
            message: "must be >= 1".to_string(),
        });
    }

    let ncomp_x = ncomp_x.min(n - 1).min(m_x);
    let ncomp_y = ncomp_y.min(n - 1).min(m_y);

    // --- FPCA on X and Y ---
    let fpca_x = fdata_to_pc_1d(x_data, ncomp_x, x_argvals)?;
    let fpca_y = fdata_to_pc_1d(y_data, ncomp_y, y_argvals)?;

    // --- Multivariate OLS: Y_scores = X_scores * B ---
    // Use projected scores (weighted inner product with eigenfunctions) rather
    // than SVD-derived scores so that training and prediction follow the same
    // computational path, guaranteeing exact agreement on training data.
    let x_scores = fpca_x.project(x_data)?;
    let y_scores = fpca_y.project(y_data)?;

    let mut xtx = compute_xtx(&x_scores);
    // Small ridge regularization for numerical stability (standard in
    // double-FPCA regression; see Ivanescu et al. 2015).
    let ridge = 1e-8 * (0..ncomp_x).map(|k| xtx[k * ncomp_x + k]).sum::<f64>() / ncomp_x as f64;
    for k in 0..ncomp_x {
        xtx[k * ncomp_x + k] += ridge.max(1e-12);
    }
    let l = cholesky_factor(&xtx, ncomp_x)?;

    // Solve for each column of Y_scores separately
    let mut coef_matrix = FdMatrix::zeros(ncomp_x, ncomp_y);
    for l_col in 0..ncomp_y {
        // X' * y_scores[:,l_col]
        let mut xty = vec![0.0; ncomp_x];
        for k in 0..ncomp_x {
            let mut s = 0.0;
            for i in 0..n {
                s += x_scores[(i, k)] * y_scores[(i, l_col)];
            }
            xty[k] = s;
        }
        let b_col = cholesky_forward_back(&l, &xty, ncomp_x);
        for k in 0..ncomp_x {
            coef_matrix[(k, l_col)] = b_col[k];
        }
    }

    // --- Reconstruct coefficient surface β(s,t) ---
    // β(s_i, t_j) = Σ_k Σ_l B_{kl} * φ_x^k(t_j) * φ_y^l(s_i)
    let mut beta_surface = FdMatrix::zeros(m_y, m_x);
    for si in 0..m_y {
        for tj in 0..m_x {
            let mut val = 0.0;
            for k in 0..ncomp_x {
                for l_col in 0..ncomp_y {
                    val += coef_matrix[(k, l_col)]
                        * fpca_x.rotation[(tj, k)]
                        * fpca_y.rotation[(si, l_col)];
                }
            }
            beta_surface[(si, tj)] = val;
        }
    }

    // --- Fitted Y-scores and reconstruction ---
    // Ŷ_scores = X_scores * B (n x ncomp_y)
    let mut fitted_scores = FdMatrix::zeros(n, ncomp_y);
    for i in 0..n {
        for l_col in 0..ncomp_y {
            let mut s = 0.0;
            for k in 0..ncomp_x {
                s += x_scores[(i, k)] * coef_matrix[(k, l_col)];
            }
            fitted_scores[(i, l_col)] = s;
        }
    }

    // Reconstruct fitted curves: Ŷ(s) = mean_y(s) + Σ_l fitted_score_l * φ_y^l(s)
    let mut fitted = FdMatrix::zeros(n, m_y);
    for i in 0..n {
        for j in 0..m_y {
            let mut val = fpca_y.mean[j];
            for l_col in 0..ncomp_y {
                val += fitted_scores[(i, l_col)] * fpca_y.rotation[(j, l_col)];
            }
            fitted[(i, j)] = val;
        }
    }

    // --- Residuals ---
    let mut residuals = FdMatrix::zeros(n, m_y);
    for i in 0..n {
        for j in 0..m_y {
            residuals[(i, j)] = y_data[(i, j)] - fitted[(i, j)];
        }
    }

    // --- Intercept: α(s) = mean_y(s) (since we centered Y via FPCA) ---
    let intercept = fpca_y.mean.clone();

    // --- Pointwise R² ---
    let mut r_squared_t = vec![0.0; m_y];
    for j in 0..m_y {
        let y_mean_j = fpca_y.mean[j];
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for i in 0..n {
            ss_tot += (y_data[(i, j)] - y_mean_j).powi(2);
            ss_res += residuals[(i, j)].powi(2);
        }
        r_squared_t[j] = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
    }

    let r_squared = r_squared_t.iter().sum::<f64>() / m_y as f64;

    Ok(FofResult {
        intercept,
        beta_surface,
        fitted,
        residuals,
        r_squared_t,
        r_squared,
        ncomp_x,
        ncomp_y,
        fpca_x,
        fpca_y,
        coef_matrix,
    })
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// Predict functional responses from new functional predictors.
///
/// Projects `new_x` onto the predictor FPCA, computes predicted Y-scores
/// via the fitted coefficient matrix, and reconstructs response curves.
///
/// # Arguments
/// * `fit` - A fitted [`FofResult`]
/// * `new_x` - New functional predictor data (n_new x m_x)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the column count of `new_x`
/// does not match the predictor grid used during fitting.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fof_regression::{fof_regression, predict_fof};
///
/// let (n, mx, my) = (25, 30, 20);
/// let x = FdMatrix::from_column_major(
///     (0..n * mx).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, mx,
/// ).unwrap();
/// let y = FdMatrix::from_column_major(
///     (0..n * my).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         0.5 * ((i + 1.0) * j * 0.15).cos()
///     }).collect(), n, my,
/// ).unwrap();
/// let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1) as f64).collect();
/// let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1) as f64).collect();
///
/// let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();
/// let predicted = predict_fof(&fit, &x).unwrap();
/// assert_eq!(predicted.shape(), (n, my));
/// ```
pub fn predict_fof(fit: &FofResult, new_x: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let (n_new, _m_x) = new_x.shape();

    // Project onto predictor FPCA
    let x_scores = fit.fpca_x.project(new_x)?;

    let ncomp_x = fit.ncomp_x;
    let ncomp_y = fit.ncomp_y;
    let m_y = fit.fpca_y.mean.len();

    // Compute predicted Y-scores: Ŷ_scores = X_scores * B
    let mut pred_scores = FdMatrix::zeros(n_new, ncomp_y);
    for i in 0..n_new {
        for l_col in 0..ncomp_y {
            let mut s = 0.0;
            for k in 0..ncomp_x {
                s += x_scores[(i, k)] * fit.coef_matrix[(k, l_col)];
            }
            pred_scores[(i, l_col)] = s;
        }
    }

    // Reconstruct: Ŷ(s) = mean_y(s) + Σ_l score_l * φ_y^l(s)
    let mut predicted = FdMatrix::zeros(n_new, m_y);
    for i in 0..n_new {
        for j in 0..m_y {
            let mut val = fit.fpca_y.mean[j];
            for l_col in 0..ncomp_y {
                val += pred_scores[(i, l_col)] * fit.fpca_y.rotation[(j, l_col)];
            }
            predicted[(i, j)] = val;
        }
    }

    Ok(predicted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
// Cross-validation
// ---------------------------------------------------------------------------

/// Result of function-on-function cross-validation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FofCvResult {
    /// (ncomp_x, ncomp_y) candidates tested.
    pub candidates: Vec<(usize, usize)>,
    /// Integrated CV-MSE for each candidate.
    pub cv_errors: Vec<f64>,
    /// Optimal (ncomp_x, ncomp_y).
    pub optimal: (usize, usize),
    /// Minimum integrated CV-MSE.
    pub min_cv_mse: f64,
}

/// K-fold cross-validation for function-on-function regression.
///
/// Searches over a grid of (ncomp_x, ncomp_y) values and selects the
/// combination minimizing integrated mean squared error (IMSE).
///
/// # Arguments
/// * `x_data` - Functional predictor (n × m_x)
/// * `y_data` - Functional response (n × m_y)
/// * `x_argvals` - Predictor grid (length m_x)
/// * `y_argvals` - Response grid (length m_y)
/// * `ncomp_x_max` - Maximum predictor components to try
/// * `ncomp_y_max` - Maximum response components to try
/// * `n_folds` - Number of CV folds
/// * `seed` - Random seed for fold assignment
///
/// # References
///
/// Ivanescu, A. E., Staicu, A.-M., Scheipl, F. & Greven, S. (2015).
/// Penalized function-on-function regression. *Computational Statistics*,
/// 30(2), 539--568.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fof_cv(
    x_data: &FdMatrix,
    y_data: &FdMatrix,
    x_argvals: &[f64],
    y_argvals: &[f64],
    ncomp_x_max: usize,
    ncomp_y_max: usize,
    n_folds: usize,
    seed: u64,
) -> Result<FofCvResult, FdarError> {
    let n = x_data.nrows();
    if n < n_folds {
        return Err(FdarError::InvalidDimension {
            parameter: "x_data",
            expected: format!("at least {n_folds} rows"),
            actual: format!("{n}"),
        });
    }

    let folds = crate::cv::create_folds(n, n_folds, seed);
    let ncomp_x_max = ncomp_x_max.min(n - 2);
    let ncomp_y_max = ncomp_y_max.min(n - 2);
    let m_y = y_data.ncols();

    // Integration weights for IMSE
    let y_weights = crate::helpers::simpsons_weights(y_argvals);

    let mut candidates = Vec::new();
    let mut cv_errors = Vec::new();
    let mut best = (1, 1);
    let mut best_mse = f64::INFINITY;

    for ncx in 1..=ncomp_x_max {
        for ncy in 1..=ncomp_y_max {
            let mut total_imse = 0.0;
            let mut count = 0;

            for fold in 0..n_folds {
                let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
                let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
                let n_test = test_idx.len();
                if n_test == 0 || train_idx.len() < ncx.max(ncy) + 2 {
                    continue;
                }

                let train_x = x_data.select_rows(&train_idx);
                let train_y = y_data.select_rows(&train_idx);
                let test_x = x_data.select_rows(&test_idx);
                let test_y = y_data.select_rows(&test_idx);

                let Ok(fit) = fof_regression(&train_x, &train_y, x_argvals, y_argvals, ncx, ncy)
                else {
                    continue;
                };

                let Ok(predicted) = predict_fof(&fit, &test_x) else {
                    continue;
                };

                // Integrated MSE per test curve
                for ti in 0..n_test {
                    let imse: f64 = (0..m_y)
                        .map(|j| (test_y[(ti, j)] - predicted[(ti, j)]).powi(2) * y_weights[j])
                        .sum();
                    total_imse += imse;
                    count += 1;
                }
            }

            let mse = if count > 0 {
                total_imse / count as f64
            } else {
                f64::INFINITY
            };

            candidates.push((ncx, ncy));
            cv_errors.push(mse);

            if mse < best_mse {
                best_mse = mse;
                best = (ncx, ncy);
            }
        }
    }

    if candidates.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "fof_cv",
            detail: "no valid (ncomp_x, ncomp_y) produced CV errors".into(),
        });
    }

    Ok(FofCvResult {
        candidates,
        cv_errors,
        optimal: best,
        min_cv_mse: best_mse,
    })
}

// ---------------------------------------------------------------------------
// Random-effects function-on-function regression
// ---------------------------------------------------------------------------

/// Configuration for [`fof_re_regression`].
///
/// No `#[non_exhaustive]` — callers may construct with struct literals.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FofReConfig {
    /// Number of predictor FPC components (default: 3)
    pub ncomp_x: usize,
    /// Number of response FPC components (default: 3)
    pub ncomp_y: usize,
    /// Maximum REML EM iterations per Y-score component (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance for variance components (default: 1e-10)
    pub tol: f64,
}

impl Default for FofReConfig {
    fn default() -> Self {
        Self {
            ncomp_x: 3,
            ncomp_y: 3,
            max_iter: 50,
            tol: 1e-10,
        }
    }
}

/// Result of a random-effects function-on-function regression.
///
/// Extends [`FofResult`] with subject-level random intercepts on Y-score
/// components, enabling prediction for grouped/longitudinal functional data.
///
/// # Divergence from `refund::pffr`
///
/// R's `pffr(y ~ pcre(x))` uses a penalized-spline GAMM (mgcv) backend with
/// functional random effects represented in a spline basis. `fof_re_regression`
/// instead uses the FPC-score parametrization: both X and Y are decomposed into
/// FPC scores (`fdata_to_pc_1d`), and for each Y-score component a scalar linear
/// mixed model (REML EM, reusing `famm::fit_scalar_mixed_model`) is fitted with
/// the X-scores as fixed-effect covariates and subject IDs as the grouping factor.
/// No basis penalties are applied — this matches the locked design decision for
/// Phase 32.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FofReResult {
    /// Intercept function α(s) = mean_y(s) (length m_y)
    pub intercept: Vec<f64>,
    /// Coefficient surface β(s,t) stored as (m_y × m_x) matrix
    pub beta_surface: FdMatrix,
    /// Fitted response curves (n × m_y), including fixed + random effects
    pub fitted: FdMatrix,
    /// Residual curves (n × m_y)
    pub residuals: FdMatrix,
    /// R-squared per response grid point (length m_y)
    pub r_squared_t: Vec<f64>,
    /// Overall R-squared (mean of pointwise R-squared)
    pub r_squared: f64,
    /// Number of predictor FPC components used
    pub ncomp_x: usize,
    /// Number of response FPC components used
    pub ncomp_y: usize,
    /// FPCA of predictor (carried for prediction)
    pub fpca_x: FpcaResult,
    /// FPCA of response (carried for reconstruction)
    pub fpca_y: FpcaResult,
    /// Fixed-effect coefficient matrix B (ncomp_x × ncomp_y):
    /// Y-scores ≈ X-scores * B
    pub coef_matrix: FdMatrix,
    /// Subject-level random intercept functions (n_subjects × m_y)
    pub random_effects: FdMatrix,
    /// Per-Y-component random-intercept variance (length ncomp_y)
    pub sigma2_u: Vec<f64>,
    /// Mean residual variance across Y-score components
    pub sigma2_eps: f64,
    /// Number of unique subjects
    pub n_subjects: usize,
}

/// Flexible random-effects function-on-function regression via double FPCA.
///
/// Extends [`fof_regression`] by fitting a scalar linear mixed model (REML EM)
/// for each Y-score component, adding subject-level random intercepts. This
/// captures within-subject correlation in grouped/longitudinal functional data.
///
/// # Algorithm
///
/// 1. **Double FPCA** (same as `fof_regression`): decompose X and Y into FPC
///    scores via `fdata_to_pc_1d`.
/// 2. **Per-Y-score mixed model** (new): for each Y-score component `l`,
///    fit `y_scores[:,l] = x_scores * γ_l + u_{subj(i),l} + ε_il` using
///    `famm::fit_scalar_mixed_model`. X-scores are passed directly as
///    covariates without rescaling (Pitfall 2: projection already carries
///    the L² weighting — do NOT re-apply `h.sqrt()` normalization).
/// 3. **Reconstruction** (same structure as `fof_regression`): build
///    `beta_surface` from `coef_matrix`; fitted curves add mean_y + fixed
///    X-score contribution + subject random-effect contribution;
///    `random_effects` reconstructed via `famm::recover_random_effects`.
///
/// # Arguments
/// * `x_data` - Functional predictor (n × m_x)
/// * `y_data` - Functional response (n × m_y)
/// * `subject_ids` - Subject identifier for each observation (length n)
/// * `x_argvals` - Predictor evaluation grid (length m_x)
/// * `y_argvals` - Response evaluation grid (length m_y)
/// * `config` - [`FofReConfig`] with component counts and convergence settings
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `x_data` and `y_data` have different row counts
/// - fewer than 3 observations
/// - argvals lengths do not match column counts
/// - `subject_ids.len()` does not equal `n`
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp_x` or `ncomp_y` is zero.
/// Returns [`FdarError::ComputationFailed`] if FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fof_regression::{fof_re_regression, FofReConfig};
///
/// let (n, mx, my) = (20, 25, 15);
/// let n_subjects = 5;
/// let x = FdMatrix::from_column_major(
///     (0..n * mx).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, mx,
/// ).unwrap();
/// let y = FdMatrix::from_column_major(
///     (0..n * my).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         0.5 * ((i + 1.0) * j * 0.15).cos()
///     }).collect(), n, my,
/// ).unwrap();
/// let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1) as f64).collect();
/// let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1) as f64).collect();
/// // 4 visits per subject
/// let subject_ids: Vec<usize> = (0..n).map(|i| i / 4).collect();
///
/// let config = FofReConfig::default();
/// let fit = fof_re_regression(&x, &y, &subject_ids, &tx, &ty, &config).unwrap();
/// assert_eq!(fit.fitted.shape(), (n, my));
/// assert_eq!(fit.beta_surface.shape(), (my, mx));
/// assert_eq!(fit.random_effects.nrows(), n_subjects);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fof_re_regression(
    x_data: &FdMatrix,
    y_data: &FdMatrix,
    subject_ids: &[usize],
    x_argvals: &[f64],
    y_argvals: &[f64],
    config: &FofReConfig,
) -> Result<FofReResult, FdarError> {
    let (n_x, m_x) = x_data.shape();
    let (n_y, m_y) = y_data.shape();

    // --- Input validation ---
    if n_x != n_y {
        return Err(FdarError::InvalidDimension {
            parameter: "y_data",
            expected: format!("{n_x} rows (matching x_data)"),
            actual: format!("{n_y} rows"),
        });
    }
    let n = n_x;

    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "x_data",
            expected: "at least 3 observations".to_string(),
            actual: format!("{n}"),
        });
    }
    if x_argvals.len() != m_x {
        return Err(FdarError::InvalidDimension {
            parameter: "x_argvals",
            expected: format!("{m_x} elements"),
            actual: format!("{} elements", x_argvals.len()),
        });
    }
    if y_argvals.len() != m_y {
        return Err(FdarError::InvalidDimension {
            parameter: "y_argvals",
            expected: format!("{m_y} elements"),
            actual: format!("{} elements", y_argvals.len()),
        });
    }
    if subject_ids.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "subject_ids",
            expected: format!("length {n}"),
            actual: format!("length {}", subject_ids.len()),
        });
    }
    if config.ncomp_x == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_x",
            message: "must be >= 1".to_string(),
        });
    }
    if config.ncomp_y == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_y",
            message: "must be >= 1".to_string(),
        });
    }

    let ncomp_x = config.ncomp_x.min(n - 1).min(m_x);
    let ncomp_y = config.ncomp_y.min(n - 1).min(m_y);

    // --- Step 1: Double FPCA ---
    let fpca_x = fdata_to_pc_1d(x_data, ncomp_x, x_argvals)?;
    let fpca_y = fdata_to_pc_1d(y_data, ncomp_y, y_argvals)?;

    // Project to score space using L²-weighted inner product
    let x_scores = fpca_x.project(x_data)?;
    let y_scores = fpca_y.project(y_data)?;

    // Build subject structure (non-contiguous IDs handled by build_subject_map)
    let (subject_map, n_subjects) = crate::famm::build_subject_map(subject_ids);

    // Wrap x_scores as covariates for fit_scalar_mixed_model.
    // INTENTIONAL: x_scores are passed directly WITHOUT re-applying h.sqrt() normalization.
    // The L² weighting is already embedded in fpca_x.project(); re-scaling would double-count
    // it (Pitfall 2, per Phase 32 RESEARCH.md). This diverges from fit_all_components in
    // famm.rs which applies score_scale = h.sqrt() before calling fit_scalar_mixed_model.
    let p = ncomp_x; // number of fixed-effect covariates per Y-score model

    // --- Step 2: Per-Y-score mixed model ---
    let mut coef_matrix = FdMatrix::zeros(ncomp_x, ncomp_y);
    // u_hat_per_component[l] = Vec of length n_subjects (random intercepts for Y-score l)
    let mut u_hat_per_component: Vec<Vec<f64>> = Vec::with_capacity(ncomp_y);
    let mut sigma2_u = vec![0.0; ncomp_y];
    let mut sigma2_eps_total = 0.0_f64;

    for l in 0..ncomp_y {
        // Extract y_scores column l into a Vec<f64>
        let y_scores_l: Vec<f64> = (0..n).map(|i| y_scores[(i, l)]).collect();

        let result = crate::famm::fit_scalar_mixed_model(
            &y_scores_l,
            &subject_map,
            n_subjects,
            Some(&x_scores),
            p,
        );

        // gamma_l: fixed-effect coefficients (length ncomp_x)
        for k in 0..ncomp_x {
            if k < result.gamma.len() {
                coef_matrix[(k, l)] = result.gamma[k];
            }
        }
        u_hat_per_component.push(result.u_hat);
        sigma2_u[l] = result.sigma2_u;
        sigma2_eps_total += result.sigma2_eps;
    }
    let sigma2_eps = sigma2_eps_total / ncomp_y as f64;

    // --- Step 3: Reconstruction ---

    // Reconstruct β(s,t) surface
    let mut beta_surface = FdMatrix::zeros(m_y, m_x);
    for si in 0..m_y {
        for tj in 0..m_x {
            let mut val = 0.0;
            for k in 0..ncomp_x {
                for l in 0..ncomp_y {
                    val +=
                        coef_matrix[(k, l)] * fpca_x.rotation[(tj, k)] * fpca_y.rotation[(si, l)];
                }
            }
            beta_surface[(si, tj)] = val;
        }
    }

    // recover_random_effects expects u_hat[subject][component].
    // u_hat_per_component is organized as [component][subject], so transpose.
    let mut u_hat_by_subject: Vec<Vec<f64>> = vec![vec![0.0; ncomp_y]; n_subjects];
    for l in 0..ncomp_y {
        for s in 0..n_subjects {
            u_hat_by_subject[s][l] = u_hat_per_component[l][s];
        }
    }
    // Reconstruct random effects: n_subjects × m_y
    // random_effects[s, j] = Σ_l u_hat_by_subject[s][l] * phi_y^l(j)
    let random_effects = crate::famm::recover_random_effects(
        &u_hat_by_subject,
        &fpca_y.rotation,
        n_subjects,
        m_y,
        ncomp_y,
    );

    // Compute fitted curves: mean_y + fixed-effect contribution + subject random effect
    let mut fitted = FdMatrix::zeros(n, m_y);
    for i in 0..n {
        let s = subject_map[i];
        for j in 0..m_y {
            let mut val = fpca_y.mean[j];
            // Fixed-effect contribution: Σ_l (Σ_k x_scores[i,k] * gamma[k,l]) * phi_y^l(j)
            for l in 0..ncomp_y {
                let mut score_l = 0.0;
                for k in 0..ncomp_x {
                    score_l += x_scores[(i, k)] * coef_matrix[(k, l)];
                }
                val += score_l * fpca_y.rotation[(j, l)];
            }
            // Random-effect contribution: b_s(j) = Σ_l u_hat[l][s] * phi_y^l(j)
            val += random_effects[(s, j)];
            fitted[(i, j)] = val;
        }
    }

    // Residuals
    let mut residuals = FdMatrix::zeros(n, m_y);
    for i in 0..n {
        for j in 0..m_y {
            residuals[(i, j)] = y_data[(i, j)] - fitted[(i, j)];
        }
    }

    // Intercept: α(s) = mean_y(s)
    let intercept = fpca_y.mean.clone();

    // Pointwise R²
    let mut r_squared_t = vec![0.0; m_y];
    for j in 0..m_y {
        let y_mean_j = fpca_y.mean[j];
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;
        for i in 0..n {
            ss_tot += (y_data[(i, j)] - y_mean_j).powi(2);
            ss_res += residuals[(i, j)].powi(2);
        }
        r_squared_t[j] = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
    }
    let r_squared = r_squared_t.iter().sum::<f64>() / m_y as f64;

    Ok(FofReResult {
        intercept,
        beta_surface,
        fitted,
        residuals,
        r_squared_t,
        r_squared,
        ncomp_x,
        ncomp_y,
        fpca_x,
        fpca_y,
        coef_matrix,
        random_effects,
        sigma2_u,
        sigma2_eps,
        n_subjects,
    })
}

// ---------------------------------------------------------------------------
// Prediction for random-effects FoF model
// ---------------------------------------------------------------------------

/// Predict functional responses from new functional predictors using a fitted
/// [`FofReResult`].
///
/// Prediction is **fixed-effect only** — no random effects are added for
/// unseen subjects. This mirrors the `fmm_predict` / `predict_fof` convention:
/// new subjects receive only the population-level fixed-effect prediction.
///
/// # Arguments
/// * `fit` - A fitted [`FofReResult`]
/// * `new_x` - New functional predictor data (n_new × m_x)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the column count of `new_x`
/// does not match the predictor grid used during fitting.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::fof_regression::{fof_re_regression, predict_fof_re, FofReConfig};
///
/// let (n, mx, my) = (20, 25, 15);
/// let x = FdMatrix::from_column_major(
///     (0..n * mx).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, mx,
/// ).unwrap();
/// let y = FdMatrix::from_column_major(
///     (0..n * my).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         0.5 * ((i + 1.0) * j * 0.15).cos()
///     }).collect(), n, my,
/// ).unwrap();
/// let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1) as f64).collect();
/// let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1) as f64).collect();
/// let subject_ids: Vec<usize> = (0..n).map(|i| i / 4).collect();
///
/// let config = FofReConfig::default();
/// let fit = fof_re_regression(&x, &y, &subject_ids, &tx, &ty, &config).unwrap();
/// let predicted = predict_fof_re(&fit, &x).unwrap();
/// assert_eq!(predicted.shape(), (n, my));
/// ```
#[must_use = "prediction result should not be discarded"]
pub fn predict_fof_re(fit: &FofReResult, new_x: &FdMatrix) -> Result<FdMatrix, FdarError> {
    let (n_new, _m_x) = new_x.shape();

    // Project onto predictor FPCA
    let x_scores = fit.fpca_x.project(new_x)?;

    let ncomp_x = fit.ncomp_x;
    let ncomp_y = fit.ncomp_y;
    let m_y = fit.fpca_y.mean.len();

    // Compute predicted Y-scores: Ŷ_scores = X_scores * coef_matrix (fixed effect only)
    let mut pred_scores = FdMatrix::zeros(n_new, ncomp_y);
    for i in 0..n_new {
        for l in 0..ncomp_y {
            let mut s = 0.0;
            for k in 0..ncomp_x {
                s += x_scores[(i, k)] * fit.coef_matrix[(k, l)];
            }
            pred_scores[(i, l)] = s;
        }
    }

    // Reconstruct: Ŷ(s) = mean_y(s) + Σ_l score_l * φ_y^l(s)
    // No random effects for new (unseen) subjects — fixed-effect only prediction.
    let mut predicted = FdMatrix::zeros(n_new, m_y);
    for i in 0..n_new {
        for j in 0..m_y {
            let mut val = fit.fpca_y.mean[j];
            for l in 0..ncomp_y {
                val += pred_scores[(i, l)] * fit.fpca_y.rotation[(j, l)];
            }
            predicted[(i, j)] = val;
        }
    }

    Ok(predicted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate test data with multiple independent modes of variation so
    /// that requesting several FPC components produces a well-conditioned
    /// score matrix.
    fn make_fof_data(
        n: usize,
        mx: usize,
        my: usize,
        seed: u64,
    ) -> (FdMatrix, FdMatrix, Vec<f64>, Vec<f64>) {
        let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1).max(1) as f64).collect();
        let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1).max(1) as f64).collect();

        let mut x = FdMatrix::zeros(n, mx);
        let mut y = FdMatrix::zeros(n, my);

        for i in 0..n {
            // Multiple independent per-observation loadings for X
            let a =
                ((seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 500.0) - 1.0;
            let b =
                ((seed.wrapping_mul(7).wrapping_add(i as u64 * 53) % 1000) as f64 / 500.0) - 1.0;
            let c =
                ((seed.wrapping_mul(3).wrapping_add(i as u64 * 79) % 1000) as f64 / 500.0) - 1.0;
            for j in 0..mx {
                x[(i, j)] = a * (2.0 * PI * tx[j]).sin() + b * (4.0 * PI * tx[j]).cos() + c * tx[j];
            }

            // Y depends on X via integral-like coupling with distinct modes
            for j in 0..my {
                y[(i, j)] = 1.5 * a * (2.0 * PI * ty[j]).cos() - 0.8 * b * (3.0 * PI * ty[j]).sin()
                    + 0.5 * c * ty[j].powi(2)
                    + 0.01 * (seed.wrapping_add(i as u64 + j as u64) % 10) as f64;
            }
        }
        (x, y, tx, ty)
    }

    #[test]
    fn test_fof_regression_dimensions() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();

        assert_eq!(fit.fitted.shape(), (30, 25));
        assert_eq!(fit.residuals.shape(), (30, 25));
        assert_eq!(fit.beta_surface.shape(), (25, 40));
        assert_eq!(fit.intercept.len(), 25);
        assert_eq!(fit.r_squared_t.len(), 25);
        assert_eq!(fit.coef_matrix.shape(), (3, 3));
        assert_eq!(fit.ncomp_x, 3);
        assert_eq!(fit.ncomp_y, 3);
    }

    #[test]
    fn test_fof_regression_r_squared_positive() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();

        // For correlated data, overall R² should be positive
        assert!(
            fit.r_squared > 0.0,
            "R² should be positive for correlated data, got {}",
            fit.r_squared
        );
    }

    #[test]
    fn test_predict_fof_training_matches_fitted() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();
        let predicted = predict_fof(&fit, &x).unwrap();

        assert_eq!(predicted.shape(), fit.fitted.shape());
        let (n, my) = predicted.shape();
        for i in 0..n {
            for j in 0..my {
                assert!(
                    (predicted[(i, j)] - fit.fitted[(i, j)]).abs() < 1e-6,
                    "predicted should match fitted at ({i}, {j}): {} vs {}",
                    predicted[(i, j)],
                    fit.fitted[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_predict_fof_new_data_finite() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();

        // Create slightly different new data
        let n_new = 10;
        let mx = 40;
        let mut new_x = FdMatrix::zeros(n_new, mx);
        for i in 0..n_new {
            let p = (i as f64 + 0.5) * PI / n_new as f64;
            for j in 0..mx {
                new_x[(i, j)] = (2.0 * PI * tx[j] + p).cos();
            }
        }

        let predicted = predict_fof(&fit, &new_x).unwrap();
        assert_eq!(predicted.shape(), (n_new, 25));
        for i in 0..n_new {
            for j in 0..25 {
                assert!(
                    predicted[(i, j)].is_finite(),
                    "prediction should be finite at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_fof_regression_mismatched_n() {
        let (x, _y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let y_bad = FdMatrix::zeros(20, 25);
        let result = fof_regression(&x, &y_bad, &tx, &ty, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_fof_regression_bad_argvals() {
        let (x, y, _tx, ty) = make_fof_data(30, 40, 25, 42);
        let bad_tx: Vec<f64> = (0..10).map(|j| j as f64).collect(); // wrong length
        let result = fof_regression(&x, &y, &bad_tx, &ty, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_fof_regression_zero_ncomp() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        assert!(fof_regression(&x, &y, &tx, &ty, 0, 3).is_err());
        assert!(fof_regression(&x, &y, &tx, &ty, 3, 0).is_err());
    }

    #[test]
    fn test_fof_cv() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let cv = fof_cv(&x, &y, &tx, &ty, 4, 4, 5, 42).unwrap();
        assert!(!cv.candidates.is_empty());
        assert!(cv.optimal.0 >= 1);
        assert!(cv.optimal.1 >= 1);
        assert!(cv.min_cv_mse.is_finite());
    }

    #[test]
    fn test_fof_regression_residuals_consistent() {
        let (x, y, tx, ty) = make_fof_data(30, 40, 25, 42);
        let fit = fof_regression(&x, &y, &tx, &ty, 3, 3).unwrap();

        let (n, my) = y.shape();
        for i in 0..n {
            for j in 0..my {
                let expected_resid = y[(i, j)] - fit.fitted[(i, j)];
                assert!(
                    (fit.residuals[(i, j)] - expected_resid).abs() < 1e-10,
                    "residual mismatch at ({i}, {j})"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Helper: generate FoF data with subject IDs for RE tests
    // -----------------------------------------------------------------------

    /// Generate test data with a grouped structure (repeated visits per subject).
    ///
    /// Each subject contributes `n_visits` curves.  Subject-level random shifts
    /// are added to Y so that random effects should be detectable.
    fn make_fof_re_data(
        n_subjects: usize,
        n_visits: usize,
        mx: usize,
        my: usize,
        seed: u64,
    ) -> (FdMatrix, FdMatrix, Vec<usize>, Vec<f64>, Vec<f64>) {
        let n = n_subjects * n_visits;
        let tx: Vec<f64> = (0..mx).map(|j| j as f64 / (mx - 1).max(1) as f64).collect();
        let ty: Vec<f64> = (0..my).map(|j| j as f64 / (my - 1).max(1) as f64).collect();

        let mut x = FdMatrix::zeros(n, mx);
        let mut y = FdMatrix::zeros(n, my);
        let mut subject_ids = Vec::with_capacity(n);

        for s in 0..n_subjects {
            // Subject-level random shift to Y (so RE should be non-zero)
            let shift =
                ((seed.wrapping_mul(13).wrapping_add(s as u64 * 97) % 1000) as f64 / 500.0) - 1.0;

            for v in 0..n_visits {
                let i = s * n_visits + v;
                subject_ids.push(s);

                let a = ((seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 500.0)
                    - 1.0;
                let b = ((seed.wrapping_mul(7).wrapping_add(i as u64 * 53) % 1000) as f64 / 500.0)
                    - 1.0;

                for j in 0..mx {
                    x[(i, j)] = a * (2.0 * PI * tx[j]).sin() + b * (4.0 * PI * tx[j]).cos();
                }
                for j in 0..my {
                    // Fixed effect: depends on x-loadings
                    // Random shift: subject-level
                    y[(i, j)] = 1.5 * a * (2.0 * PI * ty[j]).cos()
                        - 0.8 * b * (3.0 * PI * ty[j]).sin()
                        + shift  // subject-level random intercept in curve-space
                        + 0.01 * (seed.wrapping_add(i as u64 + j as u64) % 10) as f64;
                }
            }
        }

        (x, y, subject_ids, tx, ty)
    }

    // -----------------------------------------------------------------------
    // Task 1: fof_re_regression tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fof_re_regression_dims() {
        let (x, y, ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let n = x.nrows();
        let mx = x.ncols();
        let my = y.ncols();
        let n_subjects = 5;
        let config = FofReConfig {
            ncomp_x: 3,
            ncomp_y: 3,
            ..FofReConfig::default()
        };
        let fit = fof_re_regression(&x, &y, &ids, &tx, &ty, &config).unwrap();

        assert_eq!(fit.beta_surface.shape(), (my, mx), "beta_surface shape");
        assert_eq!(fit.fitted.shape(), (n, my), "fitted shape");
        assert_eq!(fit.residuals.shape(), (n, my), "residuals shape");
        assert_eq!(
            fit.random_effects.nrows(),
            n_subjects,
            "random_effects rows"
        );
        assert_eq!(fit.random_effects.ncols(), my, "random_effects cols");
        assert_eq!(fit.coef_matrix.shape(), (3, 3), "coef_matrix shape");
        assert_eq!(fit.intercept.len(), my, "intercept length");
        assert_eq!(fit.sigma2_u.len(), 3, "sigma2_u length");
        assert_eq!(fit.n_subjects, n_subjects, "n_subjects");
        assert_eq!(fit.ncomp_x, 3, "ncomp_x");
        assert_eq!(fit.ncomp_y, 3, "ncomp_y");
    }

    #[test]
    fn test_fof_re_regression_invariant() {
        let (x, y, ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let config = FofReConfig::default();
        let fit = fof_re_regression(&x, &y, &ids, &tx, &ty, &config).unwrap();

        let (n, my) = y.shape();
        for i in 0..n {
            for j in 0..my {
                let reconstructed = fit.fitted[(i, j)] + fit.residuals[(i, j)];
                assert!(
                    (reconstructed - y[(i, j)]).abs() < 1e-6,
                    "fitted+residuals != y at ({i},{j}): reconstructed={reconstructed}, y={}",
                    y[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_fof_re_regression_re_nonzero() {
        // With distinct subject-level shifts, random_effects L2 norm must be > 0
        let (x, y, ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let config = FofReConfig::default();
        let fit = fof_re_regression(&x, &y, &ids, &tx, &ty, &config).unwrap();

        let n_subjects = fit.random_effects.nrows();
        let my = fit.random_effects.ncols();
        let mut l2_sq = 0.0_f64;
        for s in 0..n_subjects {
            for j in 0..my {
                l2_sq += fit.random_effects[(s, j)].powi(2);
            }
        }
        let l2 = l2_sq.sqrt();
        assert!(
            l2 > 0.0,
            "random_effects L2 norm should be > 0 for grouped data, got {l2}"
        );
    }

    #[test]
    fn test_fof_re_regression_ids_mismatch() {
        let (x, y, _ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let bad_ids: Vec<usize> = vec![0, 1, 2]; // wrong length
        let config = FofReConfig::default();
        let err = fof_re_regression(&x, &y, &bad_ids, &tx, &ty, &config).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "subject_ids");
            }
            other => panic!("Expected InvalidDimension for subject_ids, got {other:?}"),
        }
    }

    #[test]
    fn test_fof_re_reexport() {
        // Smoke test: fof_re_regression is accessible in scope via super::*
        let _ = fof_re_regression;
    }

    // -----------------------------------------------------------------------
    // Task 2: predict_fof_re tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_predict_fof_re_shape() {
        let (x, y, ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let my = y.ncols();
        let config = FofReConfig::default();
        let fit = fof_re_regression(&x, &y, &ids, &tx, &ty, &config).unwrap();

        let n_new = 7;
        let mx = x.ncols();
        let new_x = FdMatrix::from_column_major(
            (0..n_new * mx)
                .map(|k| {
                    let i = (k % n_new) as f64;
                    let j = (k / n_new) as f64;
                    (i * j * 0.1).sin()
                })
                .collect(),
            n_new,
            mx,
        )
        .unwrap();

        let predicted = predict_fof_re(&fit, &new_x).unwrap();
        assert_eq!(predicted.shape(), (n_new, my), "shape mismatch");

        // All entries must be finite
        for i in 0..n_new {
            for j in 0..my {
                assert!(
                    predicted[(i, j)].is_finite(),
                    "non-finite prediction at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_predict_fof_re_training_matches_fixed() {
        // On training data, predict_fof_re returns fixed-effect-only fitted values.
        // These should equal: mean_y + sum_l (x_scores[i,l] * coef_matrix[:,l]) * phi_y^l
        // i.e., fitted minus the random-effect contribution.
        let (x, y, ids, tx, ty) = make_fof_re_data(5, 4, 30, 20, 42);
        let my = y.ncols();
        let config = FofReConfig::default();
        let fit = fof_re_regression(&x, &y, &ids, &tx, &ty, &config).unwrap();

        let fixed_only = predict_fof_re(&fit, &x).unwrap();
        assert_eq!(fixed_only.shape(), (x.nrows(), my));

        // Compute expected fixed-effect part manually: mean_y + X_scores * coef_matrix reconstructed
        let x_scores = fit.fpca_x.project(&x).unwrap();
        let ncomp_x = fit.ncomp_x;
        let ncomp_y = fit.ncomp_y;
        let n = x.nrows();

        for i in 0..n {
            for j in 0..my {
                let mut expected = fit.fpca_y.mean[j];
                for l in 0..ncomp_y {
                    let mut sc = 0.0;
                    for k in 0..ncomp_x {
                        sc += x_scores[(i, k)] * fit.coef_matrix[(k, l)];
                    }
                    expected += sc * fit.fpca_y.rotation[(j, l)];
                }
                assert!(
                    (fixed_only[(i, j)] - expected).abs() < 1e-6,
                    "fixed-only prediction mismatch at ({i},{j}): got {}, expected {expected}",
                    fixed_only[(i, j)]
                );
            }
        }
    }
}
