use super::{
    build_design_matrix, cholesky_factor, compute_beta_se, compute_fitted, compute_ols_std_errors,
    compute_r_squared, compute_xtx, ols_solve, recover_beta_t, resolve_ncomp,
    validate_fregre_inputs, FregreCvResult, FregreLmResult, ModelSelectionResult,
    SelectionCriterion,
};
use crate::cv::create_folds;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

// ---------------------------------------------------------------------------
// fregre_lm: FPC-based functional linear model
// ---------------------------------------------------------------------------

/// Functional linear model with optional scalar covariates.
///
/// Fits the model: `y = α + Σ_k γ_k ξ_k + γ_z' z + ε`
/// where ξ_k are FPC scores of the functional predictor X(t) and z are scalar covariates.
/// The functional coefficient is recovered as `β(t) = Σ_k γ_k φ_k(t)`.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix (n × p), `None` for pure functional model
/// * `ncomp` - Number of FPC components (if 0, selected by GCV)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, `y.len() != n`, or `scalar_covariates` row count differs from `n`.
/// Returns [`FdarError::InvalidParameter`] if auto-selected `ncomp` via
/// [`fregre_cv`] fails due to invalid range, or if `ncomp` passed to FPCA is zero.
/// Returns [`FdarError::ComputationFailed`] if the SVD inside FPCA fails to
/// extract components, or if the OLS normal equations (X'X) are singular.
///
/// # Returns
/// [`FregreLmResult`] with estimated coefficients, fitted values, and diagnostics
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_lm;
///
/// // 20 curves at 30 evaluation points (each curve has a different frequency)
/// let (n, m) = (20, 30);
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let fit = fregre_lm(&data, &y, None, 3).unwrap();
/// assert_eq!(fit.fitted_values.len(), 20);
/// assert_eq!(fit.beta_t.len(), 30);
/// assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<FregreLmResult, FdarError> {
    let (n, m) = data.shape();
    validate_fregre_inputs(n, m, y, scalar_covariates)?;

    let ncomp = resolve_ncomp(ncomp, data, y, scalar_covariates, n, m)?;

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);
    let p_total = design.ncols();
    let (coeffs, hat_diag) = ols_solve(&design, y)?;

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
    let sigma2 = ss_res / df_resid;

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p_total).unwrap_or_else(|_| vec![1.0; p_total * p_total]);
    let std_errors = compute_ols_std_errors(&l, p_total, sigma2);

    let gcv = hat_diag
        .iter()
        .zip(&residuals)
        .map(|(&h, &r)| (r / (1.0 - h).max(1e-10)).powi(2))
        .sum::<f64>()
        / n as f64;

    let beta_t = recover_beta_t(&coeffs[1..=ncomp], &fpca.rotation, m);
    let beta_se = compute_beta_se(&std_errors[1..=ncomp], &fpca.rotation, m);
    let gamma: Vec<f64> = coeffs[1 + ncomp..].to_vec();

    let nf = n as f64;
    let rss = ss_res;
    let aic = nf * (rss / nf).ln() + 2.0 * p_total as f64;
    let bic = nf * (rss / nf).ln() + (nf).ln() * p_total as f64;

    Ok(FregreLmResult {
        intercept: coeffs[0],
        beta_t,
        beta_se,
        gamma,
        fitted_values,
        residuals,
        r_squared,
        r_squared_adj,
        std_errors,
        ncomp,
        fpca,
        coefficients: coeffs,
        residual_se,
        gcv,
        aic,
        bic,
    })
}

// ---------------------------------------------------------------------------
// fregre_cv: Cross-validation for K selection
// ---------------------------------------------------------------------------

/// Copy a subset of rows from src into dst.
fn copy_rows(dst: &mut FdMatrix, src: &FdMatrix, src_rows: &[usize]) {
    let ncols = src.ncols();
    for (dst_i, &src_i) in src_rows.iter().enumerate() {
        for j in 0..ncols {
            dst[(dst_i, j)] = src[(src_i, j)];
        }
    }
}

/// Compute CV error for a single K across all folds (randomized assignment).
fn cv_error_for_k(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    k: usize,
    n_folds: usize,
    folds: &[usize],
) -> Result<f64, FdarError> {
    let n = data.nrows();
    let ncols = data.ncols();
    let mut total_error = 0.0;
    let mut count = 0;

    for fold in 0..n_folds {
        let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
        let n_test = test_idx.len();
        if n_test == 0 || train_idx.len() < k + 2 {
            continue;
        }

        let mut train_data = FdMatrix::zeros(train_idx.len(), ncols);
        let mut test_data = FdMatrix::zeros(n_test, ncols);
        copy_rows(&mut train_data, data, &train_idx);
        copy_rows(&mut test_data, data, &test_idx);

        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let test_y: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

        let train_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(train_idx.len(), sc.ncols());
            copy_rows(&mut m, sc, &train_idx);
            m
        });
        let test_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(n_test, sc.ncols());
            copy_rows(&mut m, sc, &test_idx);
            m
        });

        let Ok(fit) = fregre_lm(&train_data, &train_y, train_sc.as_ref(), k) else {
            continue;
        };

        let predictions = predict_fregre_lm(&fit, &test_data, test_sc.as_ref());
        let fold_mse: f64 = predictions
            .iter()
            .zip(&test_y)
            .map(|(&yhat, &yi)| (yhat - yi).powi(2))
            .sum::<f64>()
            / n_test as f64;

        total_error += fold_mse * n_test as f64;
        count += n_test;
    }

    if count > 0 {
        Ok(total_error / count as f64)
    } else {
        Err(FdarError::ComputationFailed {
            operation: "CV error computation",
            detail: "no valid folds produced predictions; try reducing ncomp or increasing the number of observations".to_string(),
        })
    }
}

/// K-fold cross-validation for selecting the number of FPC components.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix
/// * `k_min` - Minimum number of components to test
/// * `k_max` - Maximum number of components to test
/// * `n_folds` - Number of CV folds
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer rows than `n_folds`.
/// Returns [`FdarError::InvalidParameter`] if `k_min < 1` or `k_min > k_max`.
/// Returns [`FdarError::ComputationFailed`] if no candidate K value produces a
/// valid CV error across all folds.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_cv(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    k_min: usize,
    k_max: usize,
    n_folds: usize,
) -> Result<FregreCvResult, FdarError> {
    let n = data.nrows();
    if n < n_folds {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("at least {n_folds} rows (n_folds)"),
            actual: format!("{n}"),
        });
    }
    if k_min < 1 || k_min > k_max {
        return Err(FdarError::InvalidParameter {
            parameter: "k_min/k_max",
            message: format!("k_min={k_min} must be >= 1 and <= k_max={k_max}"),
        });
    }

    let k_max = k_max.min(n - 2).min(data.ncols());

    // Use randomized fold assignment (consistent seed for reproducibility)
    let folds = create_folds(n, n_folds, 42);

    let mut k_values = Vec::new();
    let mut cv_errors = Vec::new();

    for k in k_min..=k_max {
        if let Ok(err) = cv_error_for_k(data, y, scalar_covariates, k, n_folds, &folds) {
            k_values.push(k);
            cv_errors.push(err);
        }
    }

    if k_values.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "fregre_cv",
            detail: "no valid K values produced CV errors; all candidate ncomp values failed — check data for zero-variance columns or increase n".to_string(),
        });
    }

    let (optimal_idx, &min_cv_error) = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .expect("checked: cv_errors is non-empty");

    let optimal_k = k_values[optimal_idx];

    // Collect OOF predictions at optimal K
    let ncols = data.ncols();
    let mut oof_predictions = vec![f64::NAN; n];
    let mut fold_errors = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
        let n_test = test_idx.len();
        if n_test == 0 || train_idx.len() < optimal_k + 2 {
            fold_errors.push(f64::NAN);
            continue;
        }

        let mut train_data = FdMatrix::zeros(train_idx.len(), ncols);
        let mut test_data = FdMatrix::zeros(n_test, ncols);
        copy_rows(&mut train_data, data, &train_idx);
        copy_rows(&mut test_data, data, &test_idx);

        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();

        let train_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(train_idx.len(), sc.ncols());
            copy_rows(&mut m, sc, &train_idx);
            m
        });
        let test_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(n_test, sc.ncols());
            copy_rows(&mut m, sc, &test_idx);
            m
        });

        if let Ok(fit) = fregre_lm(&train_data, &train_y, train_sc.as_ref(), optimal_k) {
            let preds = predict_fregre_lm(&fit, &test_data, test_sc.as_ref());
            let mut fold_mse = 0.0;
            for (ti, &i) in test_idx.iter().enumerate() {
                oof_predictions[i] = preds[ti];
                fold_mse += (preds[ti] - y[i]).powi(2);
            }
            fold_errors.push(fold_mse / n_test as f64);
        } else {
            fold_errors.push(f64::NAN);
        }
    }

    Ok(FregreCvResult {
        k_values: k_values.clone(),
        cv_errors,
        optimal_k,
        min_cv_error,
        oof_predictions,
        fold_assignments: folds,
        fold_errors,
    })
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

/// Select optimal ncomp for `fregre_lm` using AIC, BIC, or GCV.
///
/// Fits models for `ncomp = 1..=max_ncomp`, collects information criteria,
/// and returns the best ncomp under the chosen criterion.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar responses (length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `max_ncomp` — Maximum number of FPC components to try
/// * `criterion` — Which criterion to minimise
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `max_ncomp` is zero.
/// Returns [`FdarError::ComputationFailed`] if no candidate ncomp produces a
/// valid fitted model (all calls to [`fregre_lm`] fail).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn model_selection_ncomp(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    max_ncomp: usize,
    criterion: SelectionCriterion,
) -> Result<ModelSelectionResult, FdarError> {
    if max_ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_ncomp",
            message: "must be >= 1".to_string(),
        });
    }

    let mut criteria = Vec::with_capacity(max_ncomp);

    for k in 1..=max_ncomp {
        if let Ok(fit) = fregre_lm(data, y, scalar_covariates, k) {
            criteria.push((k, fit.aic, fit.bic, fit.gcv));
        }
    }

    if criteria.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "model_selection_ncomp",
            detail: "no valid models could be fitted; all candidate ncomp values failed — check data for degeneracy or reduce the range".to_string(),
        });
    }

    let best_idx = match criterion {
        SelectionCriterion::Aic => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i),
        SelectionCriterion::Bic => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i),
        SelectionCriterion::Gcv => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i),
    };

    Ok(ModelSelectionResult {
        best_ncomp: criteria[best_idx].0,
        criteria,
    })
}

/// Predict new responses using a fitted functional linear model.
///
/// # Arguments
/// * `fit` - A fitted [`FregreLmResult`]
/// * `new_data` - New functional predictor matrix (n_new × m)
/// * `new_scalar` - Optional new scalar covariates (n_new × p)
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::{fregre_lm, predict_fregre_lm};
///
/// let (n, m) = (20, 30);
/// let data = FdMatrix::from_column_major(
///     (0..n * m).map(|k| {
///         let i = (k % n) as f64;
///         let j = (k / n) as f64;
///         ((i + 1.0) * j * 0.2).sin()
///     }).collect(),
///     n, m,
/// ).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| (i as f64 * 0.5).sin()).collect();
/// let fit = fregre_lm(&data, &y, None, 3).unwrap();
///
/// // Predict on same data
/// let predictions = predict_fregre_lm(&fit, &data, None);
/// assert_eq!(predictions.len(), 20);
/// ```
pub fn predict_fregre_lm(
    fit: &FregreLmResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();

    let mut predictions = vec![0.0; n_new];
    for i in 0..n_new {
        let mut yhat = fit.intercept;
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (new_data[(i, j)] - fit.fpca.mean[j])
                    * fit.fpca.rotation[(j, k)]
                    * fit.fpca.weights[j];
            }
            yhat += fit.coefficients[1 + k] * s;
        }
        if let Some(sc) = new_scalar {
            for j in 0..p_scalar {
                yhat += fit.gamma[j] * sc[(i, j)];
            }
        }
        predictions[i] = yhat;
    }
    predictions
}
