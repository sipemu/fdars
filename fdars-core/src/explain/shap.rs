//! SHAP values and Friedman H-statistic.

use super::helpers::{
    accumulate_kernel_shap_sample, build_coalition_scores, compute_column_means, compute_h_squared,
    compute_mean_scalar, get_obs_scalar, logistic_pdp_mean, make_grid, project_scores,
    sample_random_coalition, shapley_kernel_weight, solve_kernel_shap_obs,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};
use rand::prelude::*;

// ===========================================================================
// SHAP Values (FPC-level)
// ===========================================================================

/// FPC-level SHAP values for model interpretability.
#[derive(Debug, Clone, PartialEq)]
pub struct FpcShapValues {
    /// SHAP values (n x ncomp).
    pub values: FdMatrix,
    /// Base value (mean prediction).
    pub base_value: f64,
    /// Mean FPC scores (length ncomp).
    pub mean_scores: Vec<f64>,
}

/// Exact SHAP values for a linear functional regression model.
///
/// For linear models, SHAP values are exact: `values[(i,k)] = coef[1+k] * (score_i_k - mean_k)`.
/// The efficiency property holds: `base_value + sum_k values[(i,k)] ~ fitted_values[i]`
/// (with scalar covariate effects absorbed into the base value).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
/// Returns [`FdarError::InvalidParameter`] if `fit.ncomp` is zero.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_lm;
/// use fdars_core::explain::fpc_shap_values;
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
/// let shap = fpc_shap_values(&fit, &data, None).unwrap();
/// assert_eq!(shap.values.shape(), (20, 3));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fpc_shap_values(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<FpcShapValues, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);

    let mut base_value = fit.intercept;
    for k in 0..ncomp {
        base_value += fit.coefficients[1 + k] * mean_scores[k];
    }
    let p_scalar = fit.gamma.len();
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);
    for j in 0..p_scalar {
        base_value += fit.gamma[j] * mean_z[j];
    }

    let mut values = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            values[(i, k)] = fit.coefficients[1 + k] * (scores[(i, k)] - mean_scores[k]);
        }
    }

    Ok(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}

/// Kernel SHAP values for a functional logistic regression model.
///
/// Uses sampling-based Kernel SHAP approximation since the logistic link is nonlinear.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or `fit.ncomp`
/// is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fpc_shap_values_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<FpcShapValues, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "must be > 0".into(),
        });
    }
    let ncomp = fit.ncomp;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    let p_scalar = fit.gamma.len();
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let mean_scores = compute_column_means(&scores, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let predict_proba = |obs_scores: &[f64], obs_z: &[f64]| -> f64 {
        let mut eta = fit.intercept;
        for k in 0..ncomp {
            eta += fit.coefficients[1 + k] * obs_scores[k];
        }
        for j in 0..p_scalar {
            eta += fit.gamma[j] * obs_z[j];
        }
        sigmoid(eta)
    };

    let base_value = predict_proba(&mean_scores, &mean_z);
    let mut values = FdMatrix::zeros(n, ncomp);
    let mut rng = StdRng::seed_from_u64(seed);

    for i in 0..n {
        let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
        let obs_z = get_obs_scalar(scalar_covariates, i, p_scalar, &mean_z);

        let mut ata = vec![0.0; ncomp * ncomp];
        let mut atb = vec![0.0; ncomp];

        for _ in 0..n_samples {
            let (coalition, s_size) = sample_random_coalition(&mut rng, ncomp);
            let weight = shapley_kernel_weight(ncomp, s_size);
            let coal_scores = build_coalition_scores(&coalition, &obs_scores, &mean_scores);

            let f_coal = predict_proba(&coal_scores, &obs_z);
            let f_base = predict_proba(&mean_scores, &obs_z);
            let y_val = f_coal - f_base;

            accumulate_kernel_shap_sample(&mut ata, &mut atb, &coalition, weight, y_val, ncomp);
        }

        solve_kernel_shap_obs(&mut ata, &atb, ncomp, &mut values, i);
    }

    Ok(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}

// ===========================================================================
// Friedman H-statistic
// ===========================================================================

/// Result of the Friedman H-statistic for interaction between two FPC components.
#[derive(Debug, Clone, PartialEq)]
pub struct FriedmanHResult {
    /// First component index.
    pub component_j: usize,
    /// Second component index.
    pub component_k: usize,
    /// Interaction strength H^2.
    pub h_squared: f64,
    /// Grid values for component j.
    pub grid_j: Vec<f64>,
    /// Grid values for component k.
    pub grid_k: Vec<f64>,
    /// 2D partial dependence surface (n_grid x n_grid).
    pub pdp_2d: FdMatrix,
}

/// Friedman H-statistic for interaction between two FPC components (linear model).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `component_j == component_k`,
/// `n_grid < 2`, or either component index is `>= fit.ncomp`.
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn friedman_h_statistic(
    fit: &FregreLmResult,
    data: &FdMatrix,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Result<FriedmanHResult, FdarError> {
    if component_j == component_k {
        return Err(FdarError::InvalidParameter {
            parameter: "component_j/component_k",
            message: "must be different".into(),
        });
    }
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be >= 2".into(),
        });
    }
    if component_j >= fit.ncomp || component_k >= fit.ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!(
                "component_j={} or component_k={} >= ncomp={}",
                component_j, component_k, fit.ncomp
            ),
        });
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);
    let coefs = &fit.coefficients;

    let pdp_j = pdp_1d_linear(&scores, coefs, ncomp, component_j, &grid_j, n);
    let pdp_k = pdp_1d_linear(&scores, coefs, ncomp, component_k, &grid_k, n);
    let pdp_2d = pdp_2d_linear(
        &scores,
        coefs,
        ncomp,
        component_j,
        component_k,
        &grid_j,
        &grid_k,
        n,
        n_grid,
    );

    let f_bar: f64 = fit.fitted_values.iter().sum::<f64>() / n as f64;
    let h_squared = compute_h_squared(&pdp_2d, &pdp_j, &pdp_k, f_bar, n_grid);

    Ok(FriedmanHResult {
        component_j,
        component_k,
        h_squared,
        grid_j,
        grid_k,
        pdp_2d,
    })
}

/// Friedman H-statistic for interaction between two FPC components (logistic model).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `component_j == component_k`,
/// `n_grid < 2`, either component index is `>= fit.ncomp`, or
/// `scalar_covariates` is `None` when the model has scalar covariates.
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn friedman_h_statistic_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Result<FriedmanHResult, FdarError> {
    let (n, m) = data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    if component_j == component_k {
        return Err(FdarError::InvalidParameter {
            parameter: "component_j/component_k",
            message: "must be different".into(),
        });
    }
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be >= 2".into(),
        });
    }
    if component_j >= ncomp || component_k >= ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!(
                "component_j={component_j} or component_k={component_k} >= ncomp={ncomp}"
            ),
        });
    }
    if p_scalar > 0 && scalar_covariates.is_none() {
        return Err(FdarError::InvalidParameter {
            parameter: "scalar_covariates",
            message: "required when model has scalar covariates".into(),
        });
    }
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);

    let pm = |replacements: &[(usize, f64)]| {
        logistic_pdp_mean(
            &scores,
            fit.intercept,
            &fit.coefficients,
            &fit.gamma,
            scalar_covariates,
            n,
            ncomp,
            replacements,
        )
    };

    let pdp_j: Vec<f64> = grid_j.iter().map(|&gj| pm(&[(component_j, gj)])).collect();
    let pdp_k: Vec<f64> = grid_k.iter().map(|&gk| pm(&[(component_k, gk)])).collect();

    let pdp_2d = logistic_pdp_2d(
        &scores,
        fit.intercept,
        &fit.coefficients,
        &fit.gamma,
        scalar_covariates,
        n,
        ncomp,
        component_j,
        component_k,
        &grid_j,
        &grid_k,
        n_grid,
    );

    let f_bar: f64 = fit.probabilities.iter().sum::<f64>() / n as f64;
    let h_squared = compute_h_squared(&pdp_2d, &pdp_j, &pdp_k, f_bar, n_grid);

    Ok(FriedmanHResult {
        component_j,
        component_k,
        h_squared,
        grid_j,
        grid_k,
        pdp_2d,
    })
}

// ---------------------------------------------------------------------------
// Private H-statistic helpers
// ---------------------------------------------------------------------------

/// Compute 1D PDP for a linear model along one component.
fn pdp_1d_linear(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    component: usize,
    grid: &[f64],
    n: usize,
) -> Vec<f64> {
    grid.iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut yhat = coefs[0];
                for c in 0..ncomp {
                    let s = if c == component { gval } else { scores[(i, c)] };
                    yhat += coefs[1 + c] * s;
                }
                sum += yhat;
            }
            sum / n as f64
        })
        .collect()
}

/// Compute 2D PDP for a linear model along two components.
fn pdp_2d_linear(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    comp_j: usize,
    comp_k: usize,
    grid_j: &[f64],
    grid_k: &[f64],
    n: usize,
    n_grid: usize,
) -> FdMatrix {
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            let replacements = [(comp_j, gj), (comp_k, gk)];
            let mut sum = 0.0;
            for i in 0..n {
                sum += linear_predict_replaced(scores, coefs, ncomp, i, &replacements);
            }
            pdp_2d[(gj_idx, gk_idx)] = sum / n as f64;
        }
    }
    pdp_2d
}

/// Compute linear prediction with optional component replacements.
fn linear_predict_replaced(
    scores: &FdMatrix,
    coefs: &[f64],
    ncomp: usize,
    i: usize,
    replacements: &[(usize, f64)],
) -> f64 {
    let mut yhat = coefs[0];
    for c in 0..ncomp {
        let s = replacements
            .iter()
            .find(|&&(comp, _)| comp == c)
            .map_or(scores[(i, c)], |&(_, val)| val);
        yhat += coefs[1 + c] * s;
    }
    yhat
}

/// Compute 2D logistic PDP on a grid using logistic_pdp_mean.
fn logistic_pdp_2d(
    scores: &FdMatrix,
    intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    comp_j: usize,
    comp_k: usize,
    grid_j: &[f64],
    grid_k: &[f64],
    n_grid: usize,
) -> FdMatrix {
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            pdp_2d[(gj_idx, gk_idx)] = logistic_pdp_mean(
                scores,
                intercept,
                coefficients,
                gamma,
                scalar_covariates,
                n,
                ncomp,
                &[(comp_j, gj), (comp_k, gk)],
            );
        }
    }
    pdp_2d
}
