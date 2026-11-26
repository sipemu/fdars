//! PDP/ICE and beta decomposition.

use super::helpers::{ice_to_pdp, logistic_eta_base, make_grid, project_scores};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};

/// Result of a functional partial dependence plot.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FunctionalPdpResult {
    /// FPC score grid values (length n_grid).
    pub grid_values: Vec<f64>,
    /// Average prediction across observations at each grid point (length n_grid).
    pub pdp_curve: Vec<f64>,
    /// Individual conditional expectation curves (n x n_grid).
    pub ice_curves: FdMatrix,
    /// Which FPC component was varied.
    pub component: usize,
}

/// Functional PDP/ICE for a linear functional regression model.
///
/// Varies the FPC score for `component` across a grid while keeping other scores
/// fixed, producing ICE curves and their average (PDP).
///
/// For a linear model, ICE curves are parallel lines (same slope, different intercepts).
///
/// # Arguments
/// * `fit` -- A fitted [`FregreLmResult`]
/// * `data` -- Original functional predictor matrix (n x m)
/// * `scalar_covariates` -- Optional scalar covariates (n x p)
/// * `component` -- Which FPC component to vary (0-indexed, must be < fit.ncomp)
/// * `n_grid` -- Number of grid points (must be >= 2)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its column
/// count does not match `fit.fpca.mean`, or its row count does not match
/// `fit.fitted_values`.
/// Returns [`FdarError::InvalidParameter`] if `component >= fit.ncomp` or
/// `n_grid < 2`.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::fregre_lm;
/// use fdars_core::explain::functional_pdp;
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
/// let pdp = functional_pdp(&fit, &data, None, 0, 10).unwrap();
/// assert_eq!(pdp.pdp_curve.len(), 10);
/// assert_eq!(pdp.ice_curves.shape(), (20, 10));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_pdp(
    fit: &FregreLmResult,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Result<FunctionalPdpResult, FdarError> {
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
    if n != fit.fitted_values.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} rows matching fitted_values", fit.fitted_values.len()),
            actual: format!("{n}"),
        });
    }
    if component >= fit.ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, fit.ncomp),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be >= 2".into(),
        });
    }

    let ncomp = fit.ncomp;
    let scores = project_scores(
        data,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        ncomp,
        &fit.fpca.weights,
    );
    let grid_values = make_grid(&scores, component, n_grid);

    let coef_c = fit.coefficients[1 + component];
    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    for i in 0..n {
        let base = fit.fitted_values[i] - coef_c * scores[(i, component)];
        for g in 0..n_grid {
            ice_curves[(i, g)] = base + coef_c * grid_values[g];
        }
    }

    let pdp_curve = ice_to_pdp(&ice_curves, n, n_grid);

    Ok(FunctionalPdpResult {
        grid_values,
        pdp_curve,
        ice_curves,
        component,
    })
}

/// Functional PDP/ICE for a functional logistic regression model.
///
/// Predictions pass through sigmoid, so ICE curves are non-parallel.
///
/// # Arguments
/// * `fit` -- A fitted [`FunctionalLogisticResult`]
/// * `data` -- Original functional predictor matrix (n x m)
/// * `scalar_covariates` -- Optional scalar covariates (n x p)
/// * `component` -- Which FPC component to vary (0-indexed, must be < fit.ncomp)
/// * `n_grid` -- Number of grid points (must be >= 2)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its column
/// count does not match `fit.fpca.mean`.
/// Returns [`FdarError::InvalidParameter`] if `component >= fit.ncomp`,
/// `n_grid < 2`, or `scalar_covariates` is `None` when the model has scalar
/// covariates.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_pdp_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Result<FunctionalPdpResult, FdarError> {
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
    if component >= fit.ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, fit.ncomp),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be >= 2".into(),
        });
    }

    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    if p_scalar > 0 && scalar_covariates.is_none() {
        return Err(FdarError::InvalidParameter {
            parameter: "scalar_covariates",
            message: "required when model has scalar covariates".into(),
        });
    }

    let scores = project_scores(
        data,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        ncomp,
        &fit.fpca.weights,
    );
    let grid_values = make_grid(&scores, component, n_grid);

    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    let coef_c = fit.coefficients[1 + component];
    for i in 0..n {
        let eta_base = logistic_eta_base(
            fit.intercept,
            &fit.coefficients,
            &fit.gamma,
            &scores,
            scalar_covariates,
            i,
            ncomp,
            component,
        );
        for g in 0..n_grid {
            ice_curves[(i, g)] = sigmoid(eta_base + coef_c * grid_values[g]);
        }
    }

    let pdp_curve = ice_to_pdp(&ice_curves, n, n_grid);

    Ok(FunctionalPdpResult {
        grid_values,
        pdp_curve,
        ice_curves,
        component,
    })
}

// ---------------------------------------------------------------------------
// Beta decomposition
// ---------------------------------------------------------------------------

/// Per-FPC decomposition of the functional coefficient beta(t).
#[derive(Debug, Clone, PartialEq)]
pub struct BetaDecomposition {
    /// `components[k]` = coef_k * phi_k(t), each of length m.
    pub components: Vec<Vec<f64>>,
    /// FPC regression coefficients (length ncomp).
    pub coefficients: Vec<f64>,
    /// Proportion of ||beta(t)||^2 explained by each component.
    pub variance_proportion: Vec<f64>,
}

/// Decompose beta(t) = sum_k coef_k * phi_k(t) for a linear functional regression.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `fit.ncomp` is zero.
/// Returns [`FdarError::InvalidDimension`] if `fit.fpca.mean` has zero length.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn beta_decomposition(fit: &FregreLmResult) -> Result<BetaDecomposition, FdarError> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca.mean",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    Ok(decompose_beta(
        &fit.coefficients,
        &fit.fpca.rotation,
        ncomp,
        m,
    ))
}

/// Decompose beta(t) for a functional logistic regression.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `fit.ncomp` is zero.
/// Returns [`FdarError::InvalidDimension`] if `fit.fpca.mean` has zero length.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn beta_decomposition_logistic(
    fit: &FunctionalLogisticResult,
) -> Result<BetaDecomposition, FdarError> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".into(),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "fpca.mean",
            expected: ">0 length".into(),
            actual: "0".into(),
        });
    }
    Ok(decompose_beta(
        &fit.coefficients,
        &fit.fpca.rotation,
        ncomp,
        m,
    ))
}

fn decompose_beta(
    coefficients: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> BetaDecomposition {
    let mut components = Vec::with_capacity(ncomp);
    let mut coefs = Vec::with_capacity(ncomp);
    let mut norms_sq = Vec::with_capacity(ncomp);

    for k in 0..ncomp {
        let ck = coefficients[1 + k];
        coefs.push(ck);
        let comp: Vec<f64> = (0..m).map(|j| ck * rotation[(j, k)]).collect();
        let nsq: f64 = comp.iter().map(|v| v * v).sum();
        norms_sq.push(nsq);
        components.push(comp);
    }

    let total_sq: f64 = norms_sq.iter().sum();
    let variance_proportion = if total_sq > 0.0 {
        norms_sq.iter().map(|&s| s / total_sq).collect()
    } else {
        vec![0.0; ncomp]
    };

    BetaDecomposition {
        components,
        coefficients: coefs,
        variance_proportion,
    }
}

// ---------------------------------------------------------------------------
// Significant regions (moved to `crate::regions`; re-exported here so the
// historical `explain::` paths keep working)
// ---------------------------------------------------------------------------

pub use crate::regions::{
    significant_regions, significant_regions_from_se, SignificanceDirection, SignificantRegion,
};
