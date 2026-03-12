use crate::error::FdarError;
use crate::explain::{anchor_beam_search, AnchorResult};
use crate::matrix::FdMatrix;

use super::{FpcPredictor, TaskType};

/// Generic anchor explanation for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `observation >= n` or
/// `n_bins < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_anchor(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
) -> Result<AnchorResult, FdarError> {
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
    if observation >= n {
        return Err(FdarError::InvalidParameter {
            parameter: "observation",
            message: format!("observation {observation} >= n {n}"),
        });
    }
    if n_bins < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: format!("n_bins must be >= 2, got {n_bins}"),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let p_scalar = scalar_covariates.map_or(0, crate::matrix::FdMatrix::ncols);

    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(observation, k)]).collect();
    let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
        scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(observation, j)]).collect())
    } else {
        None
    };
    let obs_pred = model.predict_from_scores(&obs_scores, obs_z.as_deref());

    // Pre-compute all predictions for the same_pred closure
    let all_preds: Vec<f64> = (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let iz: Option<Vec<f64>> = if p_scalar > 0 {
                scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
            } else {
                None
            };
            model.predict_from_scores(&s, iz.as_deref())
        })
        .collect();

    let same_pred: Box<dyn Fn(usize) -> bool> = match model.task_type() {
        TaskType::Regression => {
            let pred_mean = all_preds.iter().sum::<f64>() / n as f64;
            let pred_std = (all_preds
                .iter()
                .map(|&p| (p - pred_mean).powi(2))
                .sum::<f64>()
                / (n - 1).max(1) as f64)
                .sqrt();
            let tol = pred_std.max(1e-10);
            Box::new(move |i: usize| (all_preds[i] - obs_pred).abs() <= tol)
        }
        TaskType::BinaryClassification => {
            let obs_class: f64 = if obs_pred >= 0.5 { 1.0 } else { 0.0 };
            Box::new(move |i: usize| {
                let class_i: f64 = if all_preds[i] >= 0.5 { 1.0 } else { 0.0 };
                (class_i - obs_class).abs() < 1e-10
            })
        }
        TaskType::MulticlassClassification(_) => {
            let obs_class = obs_pred.round();
            Box::new(move |i: usize| (all_preds[i].round() - obs_class).abs() < 1e-10)
        }
    };

    let (rule, _) = anchor_beam_search(
        &scores,
        ncomp,
        n,
        observation,
        precision_threshold,
        n_bins,
        &*same_pred,
    );

    Ok(AnchorResult {
        rule,
        observation,
        predicted_value: obs_pred,
    })
}
