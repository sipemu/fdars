use crate::error::FdarError;
use crate::explain::{ice_to_pdp, make_grid, FunctionalPdpResult};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Generic partial dependence plot / ICE curves for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model's FPCA mean length.
/// Returns [`FdarError::InvalidParameter`] if `component >= ncomp` or
/// `n_grid < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_pdp(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_grid: usize,
) -> Result<FunctionalPdpResult, FdarError> {
    let (n, m) = data.shape();
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
    if component >= model.ncomp() {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, model.ncomp()),
        });
    }
    if n_grid < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: format!("n_grid must be >= 2, got {n_grid}"),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let grid_values = make_grid(&scores, component, n_grid);

    let p_scalar = scalar_covariates.map_or(0, crate::matrix::FdMatrix::ncols);
    let mut ice_curves = FdMatrix::zeros(n, n_grid);
    // Pre-allocate buffers outside the observation loop and reuse them
    let mut obs_scores = vec![0.0; ncomp];
    let mut obs_z = vec![0.0; p_scalar];
    for i in 0..n {
        for k in 0..ncomp {
            obs_scores[k] = scores[(i, k)];
        }
        let obs_z_slice: Option<&[f64]> = if p_scalar > 0 {
            if let Some(sc) = scalar_covariates {
                for j in 0..p_scalar {
                    obs_z[j] = sc[(i, j)];
                }
                Some(&obs_z)
            } else {
                None
            }
        } else {
            None
        };
        for g in 0..n_grid {
            obs_scores[component] = grid_values[g];
            ice_curves[(i, g)] = model.predict_from_scores(&obs_scores, obs_z_slice);
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
