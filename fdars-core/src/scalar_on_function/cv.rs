use super::nonparametric::{
    compute_pairwise_distances, compute_scalar_distances, gaussian_kernel, select_bandwidth_loo,
};
use super::*;

// ---------------------------------------------------------------------------
// Basis regression CV (R's fregre.basis.cv)
// ---------------------------------------------------------------------------

/// Compute penalized OLS coefficients: (X'X + lam*P) \ X'y.
fn compute_penalized_ols(
    train_coefs: &FdMatrix,
    train_y: &[f64],
    penalty: &[f64],
    lam: f64,
    k: usize,
) -> Vec<f64> {
    let n_train = train_y.len();
    let mut xtx = vec![0.0; k * k];
    let mut xty_vec = vec![0.0; k];
    for i in 0..n_train {
        for j in 0..k {
            xty_vec[j] += train_coefs[(i, j)] * train_y[i];
            for l in 0..k {
                xtx[j * k + l] += train_coefs[(i, j)] * train_coefs[(i, l)];
            }
        }
    }
    for j in 0..k {
        for l in 0..k {
            xtx[j * k + l] += lam * penalty[j * k + l];
        }
    }
    crate::smoothing::solve_gaussian_pub(&mut xtx, &mut xty_vec, k)
}

/// Evaluate test-set MSE given OLS coefficients and test data.
fn evaluate_fold_error(beta: &[f64], test_coefs: &FdMatrix, test_y: &[f64], k: usize) -> f64 {
    let n_test = test_y.len();
    let mut mse = 0.0;
    for i in 0..n_test {
        let mut yhat = 0.0;
        for j in 0..k {
            yhat += test_coefs[(i, j)] * beta[j];
        }
        mse += (test_y[i] - yhat).powi(2);
    }
    mse / n_test as f64
}

/// Compute CV mean, SE, and best index from per-fold error vectors.
fn compute_cv_statistics(
    cv_fold_errors: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>, Option<(usize, f64)>) {
    let n_params = cv_fold_errors.len();
    let mut cv_errors = vec![0.0; n_params];
    let mut cv_se = vec![0.0; n_params];
    for li in 0..n_params {
        let errs = &cv_fold_errors[li];
        if errs.is_empty() {
            cv_errors[li] = f64::INFINITY;
            continue;
        }
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        cv_errors[li] = mean;
        if errs.len() > 1 {
            let var =
                errs.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (errs.len() - 1) as f64;
            cv_se[li] = (var / errs.len() as f64).sqrt();
        }
    }
    let best = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, &val)| (idx, val));
    (cv_errors, cv_se, best)
}

/// Validate inputs for [`fregre_basis_cv`] and resolve the lambda grid.
///
/// Returns `(n, m, lambdas)` on success.
fn validate_basis_cv_inputs(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    lambda_range: Option<&[f64]>,
    nbasis: usize,
) -> Result<(usize, usize, Vec<f64>), FdarError> {
    let (n, m) = data.shape();
    if n < n_folds {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("at least {n_folds} rows (n_folds)"),
            actual: format!("{n}"),
        });
    }
    if m == 0 || y.len() != n || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y/argvals",
            expected: format!("m > 0, y.len() == n={n}, argvals.len() == m={m}"),
            actual: format!(
                "m={}, y.len()={}, argvals.len()={}",
                m,
                y.len(),
                argvals.len()
            ),
        });
    }
    if nbasis < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("must be >= 2, got {nbasis}"),
        });
    }

    let lambdas: Vec<f64> = if let Some(lr) = lambda_range {
        if lr.is_empty() {
            return Err(FdarError::InvalidParameter {
                parameter: "lambda_range",
                message: "must not be empty".to_string(),
            });
        }
        lr.to_vec()
    } else {
        (0..20)
            .map(|i| {
                let log_lam = -4.0 + 8.0 * f64::from(i) / 19.0;
                10.0_f64.powf(log_lam)
            })
            .collect()
    };

    Ok((n, m, lambdas))
}

/// K-fold CV for selecting the regularization parameter lambda
/// in basis-regression (R's `fregre.basis.cv`).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_basis_cv(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    lambda_range: Option<&[f64]>,
    nbasis: usize,
    basis_type: &crate::smooth_basis::BasisType,
) -> Result<FregreBasisCvResult, FdarError> {
    use crate::smooth_basis::{smooth_basis, BasisType, FdPar};

    let (n, _m, lambdas) =
        validate_basis_cv_inputs(data, y, argvals, n_folds, lambda_range, nbasis)?;

    let penalty = match basis_type {
        BasisType::Bspline { order } => {
            crate::smooth_basis::bspline_penalty_matrix(argvals, nbasis, *order, 2)
        }
        BasisType::Fourier { period } => {
            crate::smooth_basis::fourier_penalty_matrix(nbasis, *period, 2)
        }
    };

    let folds = crate::cv::create_folds(n, n_folds, 42);
    let mut cv_fold_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(n_folds); lambdas.len()];

    for fold in 0..n_folds {
        let (train_idx, test_idx) = crate::cv::fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let train_data = crate::cv::subset_rows(data, &train_idx);
        let train_y = crate::cv::subset_vec(y, &train_idx);
        let test_data = crate::cv::subset_rows(data, &test_idx);
        let test_y = crate::cv::subset_vec(y, &test_idx);

        for (li, &lam) in lambdas.iter().enumerate() {
            let fdpar = FdPar {
                basis_type: basis_type.clone(),
                nbasis,
                lambda: lam,
                lfd_order: 2,
                penalty_matrix: penalty.clone(),
            };

            let train_result = match smooth_basis(&train_data, argvals, &fdpar) {
                Ok(r) => r,
                Err(_) => {
                    cv_fold_errors[li].push(f64::INFINITY);
                    continue;
                }
            };
            let test_result = match smooth_basis(&test_data, argvals, &fdpar) {
                Ok(r) => r,
                Err(_) => {
                    cv_fold_errors[li].push(f64::INFINITY);
                    continue;
                }
            };

            let k = train_result.coefficients.ncols();
            let beta =
                compute_penalized_ols(&train_result.coefficients, &train_y, &penalty, lam, k);
            let fold_mse = evaluate_fold_error(&beta, &test_result.coefficients, &test_y, k);
            cv_fold_errors[li].push(fold_mse);
        }
    }

    let (cv_errors, cv_se, best) = compute_cv_statistics(&cv_fold_errors);
    let (best_idx, min_cv_error) = best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "fregre_basis_cv",
        detail: "no valid lambda values produced CV errors".to_string(),
    })?;

    Ok(FregreBasisCvResult {
        optimal_lambda: lambdas[best_idx],
        cv_errors,
        cv_se,
        lambda_values: lambdas,
        min_cv_error,
    })
}

// ---------------------------------------------------------------------------
// Nonparametric regression bandwidth CV (R's fregre.np.cv)
// ---------------------------------------------------------------------------

/// Build default bandwidth grid from 20 quantiles of pairwise distances.
fn select_default_bandwidth_grid(func_dists: &[f64], n: usize) -> Result<Vec<f64>, FdarError> {
    let mut nonzero: Vec<f64> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = func_dists[i * n + j];
            if d > 0.0 {
                nonzero.push(d);
            }
        }
    }
    crate::helpers::sort_nan_safe(&mut nonzero);
    if nonzero.is_empty() {
        return Err(FdarError::ComputationFailed {
            operation: "select_default_bandwidth_grid",
            detail: "all pairwise distances are zero".to_string(),
        });
    }
    Ok((1..=20)
        .map(|qi| {
            let q = 0.05 + 0.90 * f64::from(qi - 1) / 19.0;
            let idx = ((nonzero.len() as f64 * q) as usize).min(nonzero.len() - 1);
            nonzero[idx].max(1e-10)
        })
        .collect())
}

/// Nadaraya-Watson prediction for a single test observation.
fn compute_nw_prediction(
    train_idx: &[usize],
    ti: usize,
    func_dists: &[f64],
    scalar_dists: &[f64],
    train_y: &[f64],
    n: usize,
    h: f64,
    h_scalar: f64,
    has_scalar: bool,
    fallback_y: f64,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for (local_j, &tj) in train_idx.iter().enumerate() {
        let kf = gaussian_kernel(func_dists[ti * n + tj], h);
        let ks = if has_scalar {
            gaussian_kernel(scalar_dists[ti * n + tj], h_scalar)
        } else {
            1.0
        };
        let w = kf * ks;
        num += w * train_y[local_j];
        den += w;
    }
    if den > 1e-15 {
        num / den
    } else {
        fallback_y
    }
}

/// Validate inputs for [`fregre_np_cv`] and resolve the bandwidth grid.
///
/// Returns `(n, func_dists, scalar_dists, has_scalar, h_values)` on success.
fn validate_np_cv_inputs(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    h_range: Option<&[f64]>,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<(usize, Vec<f64>, Vec<f64>, bool, Vec<f64>), FdarError> {
    let (n, m) = data.shape();
    if n < n_folds || n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("at least {} rows (max(n_folds, 3))", n_folds.max(3)),
            actual: format!("{n}"),
        });
    }
    if m == 0 || y.len() != n || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y/argvals",
            expected: format!("m > 0, y.len() == n={n}, argvals.len() == m={m}"),
            actual: format!(
                "m={}, y.len()={}, argvals.len()={}",
                m,
                y.len(),
                argvals.len()
            ),
        });
    }

    let func_dists = compute_pairwise_distances(data, argvals);
    let has_scalar = scalar_covariates.is_some();
    let scalar_dists = scalar_covariates
        .map(compute_scalar_distances)
        .unwrap_or_default();

    let h_values: Vec<f64> = if let Some(hr) = h_range {
        if hr.is_empty() {
            return Err(FdarError::InvalidParameter {
                parameter: "h_range",
                message: "must not be empty".to_string(),
            });
        }
        hr.to_vec()
    } else {
        select_default_bandwidth_grid(&func_dists, n)?
    };

    Ok((n, func_dists, scalar_dists, has_scalar, h_values))
}

/// K-fold CV for selecting the bandwidth in functional nonparametric
/// regression (R's `fregre.np.cv`).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_np_cv(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    h_range: Option<&[f64]>,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<FregreNpCvResult, FdarError> {
    let (n, func_dists, scalar_dists, has_scalar, h_values) =
        validate_np_cv_inputs(data, y, argvals, n_folds, h_range, scalar_covariates)?;

    let folds = crate::cv::create_folds(n, n_folds, 42);
    let mut cv_fold_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(n_folds); h_values.len()];

    let h_scalar = if has_scalar {
        select_bandwidth_loo(&scalar_dists, y, n, Some(&func_dists))
    } else {
        1.0
    };

    for fold in 0..n_folds {
        let (train_idx, test_idx) = crate::cv::fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }
        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();

        for (hi, &h) in h_values.iter().enumerate() {
            let mut fold_mse = 0.0;
            for &ti in &test_idx {
                let y_hat = compute_nw_prediction(
                    &train_idx,
                    ti,
                    &func_dists,
                    &scalar_dists,
                    &train_y,
                    n,
                    h,
                    h_scalar,
                    has_scalar,
                    y[ti],
                );
                fold_mse += (y[ti] - y_hat).powi(2);
            }
            fold_mse /= test_idx.len() as f64;
            cv_fold_errors[hi].push(fold_mse);
        }
    }

    let (cv_errors, cv_se, best) = compute_cv_statistics(&cv_fold_errors);
    let (best_idx, min_cv_error) = best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "fregre_np_cv",
        detail: "no valid bandwidth values produced CV errors".to_string(),
    })?;

    Ok(FregreNpCvResult {
        optimal_h: h_values[best_idx],
        cv_errors,
        cv_se,
        h_values,
        min_cv_error,
    })
}
