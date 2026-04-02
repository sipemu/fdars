//! PLS-based scalar-on-function regression.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pls_1d;

use super::{
    build_design_matrix, compute_fitted, compute_r_squared, ols_solve, PlsRegressionResult,
};

/// Fit scalar-on-function regression using PLS components.
///
/// Extracts PLS components from the functional predictor, then regresses the
/// scalar response on the PLS scores via OLS. This is useful when the predictor
/// has many correlated features, as PLS selects components that maximize
/// covariance with the response (unlike FPCA, which maximizes variance).
///
/// # Arguments
/// * `data` - Functional predictor matrix (n x m)
/// * `y` - Scalar response vector (length n)
/// * `argvals` - Evaluation grid points (length m)
/// * `ncomp` - Number of PLS components to extract
/// * `scalar_covariates` - Optional scalar covariates (n x p)
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_pls;
///
/// let n = 30;
/// let m = 50;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let vals: Vec<f64> = (0..n).flat_map(|i| {
///     t.iter().map(move |&tj| (2.0 * std::f64::consts::PI * tj).sin() + 0.1 * i as f64)
/// }).collect();
/// let data = FdMatrix::from_column_major(vals, n, m).unwrap();
/// let y: Vec<f64> = (0..n).map(|i| 2.0 + 0.5 * i as f64).collect();
///
/// let fit = fregre_pls(&data, &y, &t, 3, None).unwrap();
/// assert!(fit.r_squared >= 0.0);
/// assert_eq!(fit.fitted_values.len(), n);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_pls(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<PlsRegressionResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows, m > 0 columns".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements"),
            actual: format!("{} elements", y.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
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

    let ncomp = ncomp.min(n).min(m);
    let pls = fdata_to_pls_1d(data, y, ncomp, argvals)?;

    // Build design matrix: [1, pls_scores, scalar_covariates]
    let design = build_design_matrix(&pls.scores, ncomp, scalar_covariates, n);
    let p_total = design.ncols();
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

    // Recover beta(t) = sum_k coeff_k * w_k(t)  (PLS weight functions)
    let beta_t: Vec<f64> = (0..m)
        .map(|j| {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += coeffs[1 + k] * pls.weights[(j, k)];
            }
            val
        })
        .collect();

    let gamma: Vec<f64> = coeffs[1 + ncomp..].to_vec();

    let nf = n as f64;
    let aic = nf * (ss_res / nf).ln() + 2.0 * p_total as f64;
    let bic = nf * (ss_res / nf).ln() + nf.ln() * p_total as f64;

    Ok(PlsRegressionResult {
        intercept: coeffs[0],
        beta_t,
        gamma,
        fitted_values,
        residuals,
        r_squared,
        r_squared_adj,
        ncomp,
        pls,
        coefficients: coeffs,
        residual_se,
        aic,
        bic,
    })
}

/// Predict scalar responses for new functional data using a fitted PLS regression.
///
/// # Arguments
/// * `fit` - Fitted PLS regression result
/// * `new_data` - New functional observations (n_new x m)
/// * `new_scalar` - Optional new scalar covariates (n_new x p)
pub fn predict_fregre_pls(
    fit: &PlsRegressionResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Result<Vec<f64>, FdarError> {
    let (n_new, _m) = new_data.shape();
    let scores = fit.pls.project(new_data)?;
    let ncomp = fit.ncomp;

    let mut predictions = vec![fit.intercept; n_new];
    for i in 0..n_new {
        for k in 0..ncomp {
            predictions[i] += scores[(i, k)] * fit.coefficients[1 + k];
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
        let p_scalar = sc.ncols();
        for i in 0..n_new {
            for j in 0..p_scalar {
                predictions[i] += sc[(i, j)] * fit.gamma[j];
            }
        }
    }

    Ok(predictions)
}
