use crate::error::FdarError;
use crate::explain::{compute_ale, AleResult};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Generic ALE plot for an FPC component in any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows
/// or its column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_bins` is zero or
/// `component >= ncomp`.
/// Returns [`FdarError::ComputationFailed`] if the internal ALE computation
/// fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_ale(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    component: usize,
    n_bins: usize,
) -> Result<AleResult, FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 2".into(),
            actual: format!("{n} rows"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_bins == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: "n_bins must be > 0".into(),
        });
    }
    if component >= model.ncomp() {
        return Err(FdarError::InvalidParameter {
            parameter: "component",
            message: format!("component {} >= ncomp {}", component, model.ncomp()),
        });
    }
    let ncomp = model.ncomp();
    let p_scalar = scalar_covariates.map_or(0, crate::matrix::FdMatrix::ncols);
    let scores = model.project(data);

    let predict = |obs_scores: &[f64], obs_scalar: Option<&[f64]>| -> f64 {
        model.predict_from_scores(obs_scores, obs_scalar)
    };

    compute_ale(
        &scores,
        scalar_covariates,
        n,
        ncomp,
        p_scalar,
        component,
        n_bins,
        &predict,
    )
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_ale",
        detail: "compute_ale returned None".into(),
    })
}
