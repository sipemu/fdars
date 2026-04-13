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
}
