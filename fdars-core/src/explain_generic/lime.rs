use crate::error::FdarError;
use crate::explain::{compute_lime, LimeResult};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Generic LIME explanation for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `observation >= n`,
/// `n_samples` is zero, `kernel_width <= 0`, or the model has zero
/// components.
/// Returns [`FdarError::InvalidDimension`] if `data` columns do not match
/// the model.
/// Returns [`FdarError::ComputationFailed`] if the internal LIME
/// computation fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_lime(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    _scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
) -> Result<LimeResult, FdarError> {
    let (n, m) = data.shape();
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".into(),
        });
    }
    if kernel_width <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "kernel_width",
            message: format!("kernel_width must be > 0, got {kernel_width}"),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    let scores = model.project(data);
    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();

    let mut score_sd = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_sd[k] = (ss / (n - 1).max(1) as f64).sqrt().max(1e-10);
    }

    let predict = |s: &[f64]| -> f64 { model.predict_from_scores(s, None) };

    compute_lime(
        &obs_scores,
        &score_sd,
        ncomp,
        n_samples,
        kernel_width,
        seed,
        observation,
        &predict,
    )
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_lime",
        detail: "compute_lime returned None".into(),
    })
}
