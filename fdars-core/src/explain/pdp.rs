//! PDP/ICE and beta decomposition.

use super::helpers::*;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};

/// Result of a functional partial dependence plot.
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
pub fn functional_pdp(
    fit: &FregreLmResult,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Option<FunctionalPdpResult> {
    let (n, m) = data.shape();
    if component >= fit.ncomp
        || n_grid < 2
        || n == 0
        || m != fit.fpca.mean.len()
        || n != fit.fitted_values.len()
    {
        return None;
    }

    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
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

    Some(FunctionalPdpResult {
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
pub fn functional_pdp_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Option<FunctionalPdpResult> {
    let (n, m) = data.shape();
    if component >= fit.ncomp || n_grid < 2 || n == 0 || m != fit.fpca.mean.len() {
        return None;
    }

    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    if p_scalar > 0 && scalar_covariates.is_none() {
        return None;
    }

    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
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

    Some(FunctionalPdpResult {
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
pub struct BetaDecomposition {
    /// `components[k]` = coef_k * phi_k(t), each of length m.
    pub components: Vec<Vec<f64>>,
    /// FPC regression coefficients (length ncomp).
    pub coefficients: Vec<f64>,
    /// Proportion of ||beta(t)||^2 explained by each component.
    pub variance_proportion: Vec<f64>,
}

/// Decompose beta(t) = sum_k coef_k * phi_k(t) for a linear functional regression.
pub fn beta_decomposition(fit: &FregreLmResult) -> Option<BetaDecomposition> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 || m == 0 {
        return None;
    }
    decompose_beta(&fit.coefficients, &fit.fpca.rotation, ncomp, m)
}

/// Decompose beta(t) for a functional logistic regression.
pub fn beta_decomposition_logistic(fit: &FunctionalLogisticResult) -> Option<BetaDecomposition> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.mean.len();
    if ncomp == 0 || m == 0 {
        return None;
    }
    decompose_beta(&fit.coefficients, &fit.fpca.rotation, ncomp, m)
}

fn decompose_beta(
    coefficients: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> Option<BetaDecomposition> {
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

    Some(BetaDecomposition {
        components,
        coefficients: coefs,
        variance_proportion,
    })
}

// ---------------------------------------------------------------------------
// Significant regions
// ---------------------------------------------------------------------------

/// Direction of a significant region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignificanceDirection {
    Positive,
    Negative,
}

/// A contiguous interval where the confidence band excludes zero.
#[derive(Debug, Clone)]
pub struct SignificantRegion {
    /// Start index (inclusive).
    pub start_idx: usize,
    /// End index (inclusive).
    pub end_idx: usize,
    /// Direction of the effect.
    pub direction: SignificanceDirection,
}

/// Identify contiguous regions where the CI `[lower, upper]` excludes zero.
pub fn significant_regions(lower: &[f64], upper: &[f64]) -> Option<Vec<SignificantRegion>> {
    if lower.len() != upper.len() || lower.is_empty() {
        return None;
    }
    let n = lower.len();
    let mut regions = Vec::new();
    let mut i = 0;
    while i < n {
        if let Some(d) = detect_direction(lower[i], upper[i]) {
            let start = i;
            i += 1;
            while i < n && detect_direction(lower[i], upper[i]) == Some(d) {
                i += 1;
            }
            regions.push(SignificantRegion {
                start_idx: start,
                end_idx: i - 1,
                direction: d,
            });
        } else {
            i += 1;
        }
    }
    Some(regions)
}

/// Build CI from beta(t) +/- z * SE, then find significant regions.
pub fn significant_regions_from_se(
    beta_t: &[f64],
    beta_se: &[f64],
    z_alpha: f64,
) -> Option<Vec<SignificantRegion>> {
    if beta_t.len() != beta_se.len() || beta_t.is_empty() {
        return None;
    }
    let lower: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b - z_alpha * s)
        .collect();
    let upper: Vec<f64> = beta_t
        .iter()
        .zip(beta_se)
        .map(|(b, s)| b + z_alpha * s)
        .collect();
    significant_regions(&lower, &upper)
}

/// Detect significance direction at a single point from CI bounds.
fn detect_direction(lower: f64, upper: f64) -> Option<SignificanceDirection> {
    if lower > 0.0 {
        Some(SignificanceDirection::Positive)
    } else if upper < 0.0 {
        Some(SignificanceDirection::Negative)
    } else {
        None
    }
}
