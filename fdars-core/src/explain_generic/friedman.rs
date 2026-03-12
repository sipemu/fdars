use crate::error::FdarError;
use crate::explain::{make_grid, FriedmanHResult};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Compute the 2D partial dependence surface over a grid of two components.
///
/// For each pair `(grid_j[gj], grid_k[gk])`, fixes components `component_j`
/// and `component_k` to those grid values and averages the model prediction
/// over all `n` observations.
fn compute_pdp_grid_2d(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    grid_j: &[f64],
    grid_k: &[f64],
    component_j: usize,
    component_k: usize,
    ncomp: usize,
    n: usize,
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n_grid: usize,
) -> FdMatrix {
    let mut pdp_2d = FdMatrix::zeros(n_grid, n_grid);
    for (gj_idx, &gj) in grid_j.iter().enumerate() {
        for (gk_idx, &gk) in grid_k.iter().enumerate() {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_j] = gj;
                s[component_k] = gk;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            pdp_2d[(gj_idx, gk_idx)] = sum / n as f64;
        }
    }
    pdp_2d
}

/// Compute the H-squared statistic from marginal and joint PDP surfaces.
///
/// First computes the mean prediction `f_bar` (averaging over all observations),
/// then delegates to [`crate::explain::compute_h_squared`] for the actual
/// interaction / total-variance ratio.
fn compute_h_squared_from_pdps(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    pdp_2d: &FdMatrix,
    pdp_j: &[f64],
    pdp_k: &[f64],
    ncomp: usize,
    n: usize,
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n_grid: usize,
) -> f64 {
    let f_bar: f64 = (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
            let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
            } else {
                None
            };
            model.predict_from_scores(&s, obs_z.as_deref())
        })
        .sum::<f64>()
        / n as f64;
    crate::explain::compute_h_squared(pdp_2d, pdp_j, pdp_k, f_bar, n_grid)
}

/// Generic Friedman H-statistic for interaction between two FPC components.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `component_j == component_k`,
/// `n_grid < 2`, or either component index is `>= ncomp`.
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_friedman_h(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component_j: usize,
    component_k: usize,
    n_grid: usize,
) -> Result<FriedmanHResult, FdarError> {
    if component_j == component_k {
        return Err(FdarError::InvalidParameter {
            parameter: "component_j/component_k",
            message: "component_j and component_k must differ".into(),
        });
    }
    let (n, m) = data.shape();
    let ncomp = model.ncomp();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: format!("n_grid must be >= 2, got {n_grid}"),
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

    let scores = model.project(data);
    let grid_j = make_grid(&scores, component_j, n_grid);
    let grid_k = make_grid(&scores, component_k, n_grid);
    let p_scalar = scalar_covariates.map_or(0, crate::matrix::FdMatrix::ncols);

    // Compute 1D PDPs via generic predict
    let pdp_j: Vec<f64> = grid_j
        .iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_j] = gval;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            sum / n as f64
        })
        .collect();

    let pdp_k: Vec<f64> = grid_k
        .iter()
        .map(|&gval| {
            let mut sum = 0.0;
            for i in 0..n {
                let mut s: Vec<f64> = (0..ncomp).map(|c| scores[(i, c)]).collect();
                s[component_k] = gval;
                let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
                    scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
                } else {
                    None
                };
                sum += model.predict_from_scores(&s, obs_z.as_deref());
            }
            sum / n as f64
        })
        .collect();

    let pdp_2d = compute_pdp_grid_2d(
        model,
        &scores,
        &grid_j,
        &grid_k,
        component_j,
        component_k,
        ncomp,
        n,
        scalar_covariates,
        p_scalar,
        n_grid,
    );

    let h_squared = compute_h_squared_from_pdps(
        model,
        &scores,
        &pdp_2d,
        &pdp_j,
        &pdp_k,
        ncomp,
        n,
        scalar_covariates,
        p_scalar,
        n_grid,
    );

    Ok(FriedmanHResult {
        component_j,
        component_k,
        h_squared,
        grid_j,
        grid_k,
        pdp_2d,
    })
}
