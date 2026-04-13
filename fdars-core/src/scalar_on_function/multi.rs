//! Multi-predictor scalar-on-function regression.
//!
//! Extends [`fregre_lm`](super::fregre_lm::fregre_lm) to accept multiple
//! functional predictors, each with its own FPCA decomposition and number
//! of components.
//!
//! Model: `y = α + Σ_k ∫ β_k(t) X_k(t) dt + γᵀz + ε`
//!
//! # References
//!
//! - Ramsay, J. O. & Silverman, B. W. (2005). *Functional Data Analysis*, Ch. 15.
//! - Febrero-Bande, M. & Oviedo de la Fuente, M. (2012). Statistical Computing
//!   in Functional Data Analysis: The R Package fda.usc. *Journal of Statistical
//!   Software*, 51(4), 1--28.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

use super::{compute_fitted, compute_r_squared, ols_solve, MultiFregreLmResult};

/// Scalar-on-function regression with multiple functional predictors.
///
/// Runs FPCA on each functional predictor separately, concatenates score
/// matrices, and fits OLS. Each predictor can have a different grid and
/// number of components.
///
/// # Arguments
/// * `predictors` - Slice of `(data, argvals, ncomp)` tuples for each functional predictor
/// * `y` - Scalar response (length n)
/// * `scalar_covariates` - Optional scalar covariates (n x p)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if predictors have differing row counts,
/// `y.len() != n`, or `scalar_covariates` row count differs from `n`.
/// Returns [`FdarError::InvalidParameter`] if `predictors` is empty.
/// Returns [`FdarError::ComputationFailed`] if any FPCA or OLS step fails.
///
/// # References
///
/// Ramsay, J. O. & Silverman, B. W. (2005). *Functional Data Analysis*, Ch. 15.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_lm_multi;
///
/// let (n, m1, m2) = (25, 30, 20);
/// let x1 = FdMatrix::from_column_major(
///     (0..n * m1).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, m1,
/// ).unwrap();
/// let x2 = FdMatrix::from_column_major(
///     (0..n * m2).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.3).cos()
///     }).collect(), n, m2,
/// ).unwrap();
/// let t1: Vec<f64> = (0..m1).map(|j| j as f64 / (m1 - 1) as f64).collect();
/// let t2: Vec<f64> = (0..m2).map(|j| j as f64 / (m2 - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
///
/// let fit = fregre_lm_multi(
///     &[(&x1, t1.as_slice(), 3), (&x2, t2.as_slice(), 2)],
///     &y, None,
/// ).unwrap();
/// assert_eq!(fit.fitted_values.len(), n);
/// assert_eq!(fit.beta_t.len(), 2);
/// assert_eq!(fit.ncomp, vec![3, 2]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_lm_multi(
    predictors: &[(&FdMatrix, &[f64], usize)],
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Result<MultiFregreLmResult, FdarError> {
    // --- Validate inputs ---
    if predictors.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "predictors",
            message: "at least one functional predictor is required".to_string(),
        });
    }

    let n = predictors[0].0.nrows();
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 3 observations".to_string(),
            actual: format!("{n}"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }

    for (k, &(data_k, argvals_k, _)) in predictors.iter().enumerate() {
        if data_k.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "predictors",
                expected: format!("{n} rows for predictor {k}"),
                actual: format!("{} rows", data_k.nrows()),
            });
        }
        if argvals_k.len() != data_k.ncols() {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals",
                expected: format!("{} elements for predictor {k}", data_k.ncols()),
                actual: format!("{} elements", argvals_k.len()),
            });
        }
    }

    if let Some(sc) = scalar_covariates {
        if sc.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }

    // --- Run FPCA on each predictor ---
    let k_preds = predictors.len();
    let mut fpcas: Vec<FpcaResult> = Vec::with_capacity(k_preds);
    let mut ncomp_vec: Vec<usize> = Vec::with_capacity(k_preds);

    for &(data_k, argvals_k, ncomp_k) in predictors {
        let nc = ncomp_k.max(1).min(n - 1).min(data_k.ncols());
        let fpca = fdata_to_pc_1d(data_k, nc, argvals_k)?;
        ncomp_vec.push(nc);
        fpcas.push(fpca);
    }

    let total_scores: usize = ncomp_vec.iter().sum();
    let p_scalar = scalar_covariates.map_or(0, FdMatrix::ncols);
    let p_total = 1 + total_scores + p_scalar;

    // --- Build concatenated design matrix: [1, scores_1, ..., scores_K, scalars] ---
    // Use projected scores (weighted inner product with eigenfunctions) rather
    // than SVD-derived scores so that training and prediction follow the same
    // computational path, guaranteeing exact agreement on training data.
    let mut design = FdMatrix::zeros(n, p_total);
    for i in 0..n {
        design[(i, 0)] = 1.0;
    }

    let mut col_offset = 1;
    for (idx, fpca) in fpcas.iter().enumerate() {
        let nc = ncomp_vec[idx];
        let m_k = fpca.mean.len();
        let data_k = predictors[idx].0;
        for i in 0..n {
            for k in 0..nc {
                let mut s = 0.0;
                for j in 0..m_k {
                    s += (data_k[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)] * fpca.weights[j];
                }
                design[(i, col_offset + k)] = s;
            }
        }
        col_offset += nc;
    }

    if let Some(sc) = scalar_covariates {
        for i in 0..n {
            for j in 0..p_scalar {
                design[(i, col_offset + j)] = sc[(i, j)];
            }
        }
    }

    // --- OLS ---
    let (coeffs, _hat_diag) = ols_solve(&design, y)?;

    let fitted_values = compute_fitted(&design, &coeffs);
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, r_squared_adj) = compute_r_squared(y, &residuals, p_total);

    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let df_resid = (n as f64 - p_total as f64).max(1.0);
    let residual_se = (ss_res / df_resid).sqrt();

    let nf = n as f64;
    let aic = nf * (ss_res / nf).ln() + 2.0 * p_total as f64;
    let bic = nf * (ss_res / nf).ln() + nf.ln() * p_total as f64;

    // --- Recover β_k(t) for each predictor ---
    let mut beta_t: Vec<Vec<f64>> = Vec::with_capacity(k_preds);
    let mut coeff_offset = 1;
    for (idx, fpca) in fpcas.iter().enumerate() {
        let nc = ncomp_vec[idx];
        let m_k = fpca.rotation.nrows();
        let fpc_coeffs = &coeffs[coeff_offset..coeff_offset + nc];
        let mut beta_k = vec![0.0; m_k];
        for k in 0..nc {
            for j in 0..m_k {
                beta_k[j] += fpc_coeffs[k] * fpca.rotation[(j, k)];
            }
        }
        beta_t.push(beta_k);
        coeff_offset += nc;
    }

    // Scalar coefficients
    let gamma: Vec<f64> = coeffs[1 + total_scores..].to_vec();

    Ok(MultiFregreLmResult {
        intercept: coeffs[0],
        beta_t,
        gamma,
        fitted_values,
        residuals,
        r_squared,
        r_squared_adj,
        ncomp: ncomp_vec,
        fpcas,
        coefficients: coeffs,
        residual_se,
        aic,
        bic,
    })
}

/// Predict from a multi-predictor functional linear model.
///
/// Projects each new predictor onto its stored FPCA, concatenates scores,
/// and multiplies by the fitted coefficients.
///
/// # Arguments
/// * `fit` - A fitted [`MultiFregreLmResult`]
/// * `new_predictors` - Slice of new functional predictor matrices (one per predictor, each n_new x m_k)
/// * `new_scalar` - Optional new scalar covariates (n_new x p)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the number of new predictors
/// does not match the number used in fitting, or if row counts differ.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::{fregre_lm_multi, predict_fregre_lm_multi};
///
/// let (n, m1, m2) = (25, 30, 20);
/// let x1 = FdMatrix::from_column_major(
///     (0..n * m1).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(), n, m1,
/// ).unwrap();
/// let x2 = FdMatrix::from_column_major(
///     (0..n * m2).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.3).cos()
///     }).collect(), n, m2,
/// ).unwrap();
/// let t1: Vec<f64> = (0..m1).map(|j| j as f64 / (m1 - 1) as f64).collect();
/// let t2: Vec<f64> = (0..m2).map(|j| j as f64 / (m2 - 1) as f64).collect();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
///
/// let fit = fregre_lm_multi(
///     &[(&x1, t1.as_slice(), 3), (&x2, t2.as_slice(), 2)],
///     &y, None,
/// ).unwrap();
/// let preds = predict_fregre_lm_multi(&fit, &[&x1, &x2], None).unwrap();
/// assert_eq!(preds.len(), n);
/// ```
pub fn predict_fregre_lm_multi(
    fit: &MultiFregreLmResult,
    new_predictors: &[&FdMatrix],
    new_scalar: Option<&FdMatrix>,
) -> Result<Vec<f64>, FdarError> {
    let k_preds = fit.fpcas.len();
    if new_predictors.len() != k_preds {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{k_preds} predictors"),
            actual: format!("{} predictors", new_predictors.len()),
        });
    }

    let n_new = new_predictors[0].nrows();
    for (idx, &pred) in new_predictors.iter().enumerate() {
        if pred.nrows() != n_new {
            return Err(FdarError::InvalidDimension {
                parameter: "new_predictors",
                expected: format!("{n_new} rows for predictor {idx}"),
                actual: format!("{} rows", pred.nrows()),
            });
        }
    }

    if let Some(sc) = new_scalar {
        if sc.nrows() != n_new {
            return Err(FdarError::InvalidDimension {
                parameter: "new_scalar",
                expected: format!("{n_new} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }

    // Compute predictions directly via manual projection (matching the
    // approach used by predict_fregre_lm to ensure numerical consistency
    // between fitted values and training-data predictions).
    let p_scalar = fit.gamma.len();

    let mut predictions = vec![0.0; n_new];
    for i in 0..n_new {
        let mut yhat = fit.intercept;

        // Score-coefficient contributions from each functional predictor
        let mut coeff_offset = 1;
        for (idx, &pred) in new_predictors.iter().enumerate() {
            let fpca = &fit.fpcas[idx];
            let nc = fit.ncomp[idx];
            let m_k = fpca.mean.len();
            for k in 0..nc {
                let mut s = 0.0;
                for j in 0..m_k {
                    s += (pred[(i, j)] - fpca.mean[j]) * fpca.rotation[(j, k)] * fpca.weights[j];
                }
                yhat += fit.coefficients[coeff_offset + k] * s;
            }
            coeff_offset += nc;
        }

        // Scalar covariate contributions
        if let Some(sc) = new_scalar {
            for j in 0..p_scalar {
                yhat += fit.gamma[j] * sc[(i, j)];
            }
        }

        predictions[i] = yhat;
    }

    Ok(predictions)
}

/// K-fold cross-validation for multi-predictor functional regression.
///
/// Searches over a grid of per-predictor ncomp values (all predictors share
/// the same ncomp) and selects the one minimizing CV-MSE.
///
/// # Arguments
/// * `predictors` - Slice of `(data, argvals)` for each functional predictor
/// * `y` - Scalar response (length n)
/// * `scalar_covariates` - Optional scalar covariates (n × p)
/// * `ncomp_max` - Maximum ncomp to try (shared across all predictors)
/// * `n_folds` - Number of CV folds
/// * `seed` - Random seed for fold assignment
///
/// # References
///
/// Febrero-Bande, M. & Oviedo de la Fuente, M. (2012). Statistical Computing
/// in Functional Data Analysis: The R Package fda.usc. *Journal of Statistical
/// Software*, 51(4), 1--28.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_lm_multi_cv(
    predictors: &[(&FdMatrix, &[f64])],
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp_max: usize,
    n_folds: usize,
    seed: u64,
) -> Result<MultiCvResult, FdarError> {
    if predictors.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "predictors",
            message: "need at least one functional predictor".into(),
        });
    }
    let n = predictors[0].0.nrows();
    if n < n_folds {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("at least {n_folds} rows"),
            actual: format!("{n}"),
        });
    }

    let folds = crate::cv::create_folds(n, n_folds, seed);
    let ncomp_max = ncomp_max.min(n - 2);
    let k_preds = predictors.len();

    let mut candidates = Vec::new();
    let mut cv_errors = Vec::new();
    let mut best_ncomp = 1;
    let mut best_mse = f64::INFINITY;
    let mut best_oof = vec![f64::NAN; n];

    for nc in 1..=ncomp_max {
        let mut oof_preds = vec![f64::NAN; n];
        let mut total_se = 0.0;
        let mut count = 0;

        for fold in 0..n_folds {
            let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
            let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
            let n_train = train_idx.len();
            let n_test = test_idx.len();
            if n_test == 0 || n_train < nc + 2 {
                continue;
            }

            // Build train/test splits for each predictor
            let mut train_preds: Vec<(FdMatrix, Vec<f64>, usize)> = Vec::with_capacity(k_preds);
            let mut test_preds: Vec<FdMatrix> = Vec::with_capacity(k_preds);

            for &(data_k, argvals_k) in predictors {
                let train_k = data_k.select_rows(&train_idx);
                let test_k = data_k.select_rows(&test_idx);
                train_preds.push((train_k, argvals_k.to_vec(), nc));
                test_preds.push(test_k);
            }

            let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
            let train_sc = scalar_covariates.map(|sc| sc.select_rows(&train_idx));
            let test_sc = scalar_covariates.map(|sc| sc.select_rows(&test_idx));

            let pred_refs: Vec<(&FdMatrix, &[f64], usize)> = train_preds
                .iter()
                .map(|(d, a, c)| (d, a.as_slice(), *c))
                .collect();

            let Ok(fit) = fregre_lm_multi(&pred_refs, &train_y, train_sc.as_ref()) else {
                continue;
            };

            let test_pred_refs: Vec<&FdMatrix> = test_preds.iter().collect();
            let Ok(preds) = predict_fregre_lm_multi(&fit, &test_pred_refs, test_sc.as_ref()) else {
                continue;
            };

            for (ti, &i) in test_idx.iter().enumerate() {
                oof_preds[i] = preds[ti];
                total_se += (y[i] - preds[ti]).powi(2);
                count += 1;
            }
        }

        let mse = if count > 0 {
            total_se / count as f64
        } else {
            f64::INFINITY
        };

        candidates.push(nc);
        cv_errors.push(mse);

        if mse < best_mse {
            best_mse = mse;
            best_ncomp = nc;
            best_oof = oof_preds;
        }
    }

    if candidates.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "fregre_lm_multi_cv",
            detail: "no valid ncomp produced CV errors".into(),
        });
    }

    Ok(MultiCvResult {
        candidates,
        cv_errors,
        optimal_ncomp: best_ncomp,
        min_cv_mse: best_mse,
        oof_predictions: best_oof,
    })
}

/// Result of multi-predictor cross-validation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MultiCvResult {
    /// Candidate ncomp values tested.
    pub candidates: Vec<usize>,
    /// CV-MSE for each candidate.
    pub cv_errors: Vec<f64>,
    /// Optimal ncomp (shared across all predictors).
    pub optimal_ncomp: usize,
    /// Minimum CV-MSE.
    pub min_cv_mse: f64,
    /// Out-of-fold predictions at optimal ncomp (length n).
    pub oof_predictions: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate test data with multiple independent modes of variation per
    /// predictor so that requesting >1 FPC component produces a well-
    /// conditioned design matrix.
    fn make_multi_data(
        n: usize,
        m1: usize,
        m2: usize,
        seed: u64,
    ) -> (FdMatrix, FdMatrix, Vec<f64>, Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..m1).map(|j| j as f64 / (m1 - 1).max(1) as f64).collect();
        let t2: Vec<f64> = (0..m2).map(|j| j as f64 / (m2 - 1).max(1) as f64).collect();
        let mut x1 = FdMatrix::zeros(n, m1);
        let mut x2 = FdMatrix::zeros(n, m2);
        let mut y = vec![0.0; n];

        for i in 0..n {
            // Multiple independent per-observation "loadings"
            let a1 =
                ((seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 500.0) - 1.0;
            let b1 =
                ((seed.wrapping_mul(7).wrapping_add(i as u64 * 53) % 1000) as f64 / 500.0) - 1.0;
            let c1 =
                ((seed.wrapping_mul(3).wrapping_add(i as u64 * 79) % 1000) as f64 / 500.0) - 1.0;

            let a2 =
                ((seed.wrapping_mul(11).wrapping_add(i as u64 * 43) % 1000) as f64 / 500.0) - 1.0;
            let b2 =
                ((seed.wrapping_mul(23).wrapping_add(i as u64 * 67) % 1000) as f64 / 500.0) - 1.0;

            // X1: three modes of variation
            for j in 0..m1 {
                x1[(i, j)] =
                    a1 * (2.0 * PI * t1[j]).sin() + b1 * (4.0 * PI * t1[j]).cos() + c1 * t1[j];
            }
            // X2: two modes of variation (different frequencies)
            for j in 0..m2 {
                x2[(i, j)] = a2 * (2.0 * PI * t2[j]).cos() + b2 * (6.0 * PI * t2[j]).sin();
            }
            y[i] = 2.0 * a1 - 1.5 * b1
                + 0.8 * a2
                + 0.3 * b2
                + 0.05 * (seed.wrapping_add(i as u64) % 10) as f64;
        }
        (x1, x2, y, t1, t2)
    }

    #[test]
    fn test_fregre_lm_multi_two_predictors() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let fit = fregre_lm_multi(&[(&x1, &t1, 3), (&x2, &t2, 2)], &y, None).unwrap();

        assert_eq!(fit.fitted_values.len(), 30);
        assert_eq!(fit.residuals.len(), 30);
        assert_eq!(fit.beta_t.len(), 2);
        assert_eq!(fit.beta_t[0].len(), 40);
        assert_eq!(fit.beta_t[1].len(), 25);
        assert_eq!(fit.ncomp, vec![3, 2]);
        assert!(fit.r_squared >= 0.0);
        assert!(fit.r_squared <= 1.0 + 1e-10);
    }

    #[test]
    fn test_fregre_lm_multi_with_scalar() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let mut sc = FdMatrix::zeros(30, 2);
        for i in 0..30 {
            sc[(i, 0)] = i as f64 / 30.0;
            sc[(i, 1)] = (i as f64 * 0.7).sin();
        }
        let fit = fregre_lm_multi(&[(&x1, &t1, 3), (&x2, &t2, 2)], &y, Some(&sc)).unwrap();

        assert_eq!(fit.gamma.len(), 2);
        assert!(fit.r_squared >= 0.0);
    }

    #[test]
    fn test_predict_multi_on_training() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let fit = fregre_lm_multi(&[(&x1, &t1, 3), (&x2, &t2, 2)], &y, None).unwrap();

        let preds = predict_fregre_lm_multi(&fit, &[&x1, &x2], None).unwrap();
        assert_eq!(preds.len(), 30);
        for i in 0..30 {
            assert!(
                (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
                "prediction on training data should match fitted values at index {i}: got {}, expected {}",
                preds[i], fit.fitted_values[i]
            );
        }
    }

    #[test]
    fn test_predict_multi_new_data_finite() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let fit = fregre_lm_multi(&[(&x1, &t1, 3), (&x2, &t2, 2)], &y, None).unwrap();

        // Create slightly different "new" data
        let n_new = 10;
        let mut new_x1 = FdMatrix::zeros(n_new, 40);
        let mut new_x2 = FdMatrix::zeros(n_new, 25);
        for i in 0..n_new {
            let p = (i as f64 + 0.5) * PI / n_new as f64;
            for j in 0..40 {
                new_x1[(i, j)] = (2.0 * PI * t1[j] + p).sin() + 0.1;
            }
            for j in 0..25 {
                new_x2[(i, j)] = (2.0 * PI * t2[j] + p).cos() - 0.1;
            }
        }

        let preds = predict_fregre_lm_multi(&fit, &[&new_x1, &new_x2], None).unwrap();
        assert_eq!(preds.len(), n_new);
        for &p in &preds {
            assert!(p.is_finite(), "predictions should be finite, got {p}");
        }
    }

    #[test]
    fn test_fregre_lm_multi_mismatched_n() {
        let (x1, _x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let x2_bad = FdMatrix::zeros(20, 25);
        let result = fregre_lm_multi(&[(&x1, &t1, 3), (&x2_bad, &t2, 2)], &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fregre_lm_multi_single_predictor_matches_fregre_lm() {
        let n = 30;
        let m = 50;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];
        for i in 0..n {
            // Generate data with multiple modes of variation (matching the
            // pattern used in the existing fregre_lm tests).
            let a =
                ((42_u64.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 500.0) - 1.0;
            let b =
                ((42_u64.wrapping_mul(7).wrapping_add(i as u64 * 53) % 1000) as f64 / 500.0) - 1.0;
            let c =
                ((42_u64.wrapping_mul(3).wrapping_add(i as u64 * 79) % 1000) as f64 / 500.0) - 1.0;
            for j in 0..m {
                data[(i, j)] = a * (2.0 * PI * t[j]).sin() + b * (4.0 * PI * t[j]).cos() + c * t[j];
            }
            y[i] = 2.0 * a + 3.0 * b + 0.05 * (42_u64.wrapping_add(i as u64) % 10) as f64;
        }

        let single = crate::scalar_on_function::fregre_lm(&data, &y, None, 3).unwrap();
        let multi = fregre_lm_multi(&[(&data, &t, 3)], &y, None).unwrap();

        // R-squared should be very close
        assert!(
            (single.r_squared - multi.r_squared).abs() < 1e-8,
            "single R²={} vs multi R²={}",
            single.r_squared,
            multi.r_squared
        );

        // Fitted values should match
        for i in 0..n {
            assert!(
                (single.fitted_values[i] - multi.fitted_values[i]).abs() < 1e-8,
                "fitted values differ at {i}"
            );
        }
    }

    #[test]
    fn test_fregre_lm_multi_empty_predictors() {
        let y = vec![1.0, 2.0, 3.0];
        let result = fregre_lm_multi(&[], &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_fregre_lm_multi_cv() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let cv = fregre_lm_multi_cv(&[(&x1, &t1), (&x2, &t2)], &y, None, 5, 5, 42).unwrap();
        assert!(!cv.candidates.is_empty());
        assert!(cv.optimal_ncomp >= 1);
        assert!(cv.min_cv_mse.is_finite());
        assert_eq!(cv.oof_predictions.len(), 30);
        // At least some OOF predictions should be finite
        let n_finite = cv.oof_predictions.iter().filter(|x| x.is_finite()).count();
        assert!(n_finite > 20, "most OOF predictions should be finite");
    }

    #[test]
    fn test_fregre_lm_multi_residuals_sum_near_zero() {
        let (x1, x2, y, t1, t2) = make_multi_data(30, 40, 25, 42);
        let fit = fregre_lm_multi(&[(&x1, &t1, 3), (&x2, &t2, 2)], &y, None).unwrap();

        let resid_sum: f64 = fit.residuals.iter().sum();
        assert!(
            resid_sum.abs() < 1e-8,
            "residuals should sum to ~0 with intercept, got {resid_sum}"
        );
    }
}
