use crate::error::FdarError;
use crate::explain::{
    build_stability_result, compute_vif_from_scores, subsample_rows, StabilityAnalysisResult,
    VifResult,
};
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{fregre_lm, functional_logistic};
use rand::prelude::*;

use super::{FpcPredictor, TaskType};

/// Generic explanation stability via bootstrap resampling.
///
/// Refits the model on bootstrap samples and measures variability of
/// coefficients, β(t), and metric (R² or accuracy).
///
/// Note: This only works for regression and logistic models since it requires
/// refitting. For classification models, bootstrap refitting is not yet supported.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 rows,
/// zero columns, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_boot < 2`, `ncomp` is
/// zero, or `task_type` is `MulticlassClassification`.
/// Returns [`FdarError::ComputationFailed`] if not enough bootstrap refits
/// succeed.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_stability(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    seed: u64,
    task_type: TaskType,
) -> Result<StabilityAnalysisResult, FdarError> {
    let (n, m) = data.shape();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 4".into(),
            actual: format!("{n} rows"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0".into(),
            actual: "0 columns".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if n_boot < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_boot",
            message: format!("n_boot must be >= 2, got {n_boot}"),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be > 0".into(),
        });
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_beta_t: Vec<Vec<f64>> = Vec::new();
    let mut all_coefs: Vec<Vec<f64>> = Vec::new();
    let mut all_metrics: Vec<f64> = Vec::new();
    let mut all_abs_coefs: Vec<Vec<f64>> = Vec::new();

    match task_type {
        TaskType::Regression => {
            for _ in 0..n_boot {
                let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                let boot_data = subsample_rows(data, &idx);
                let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
                let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
                if let Ok(refit) = fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp) {
                    all_beta_t.push(refit.beta_t.clone());
                    let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
                    all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
                    all_coefs.push(coefs);
                    all_metrics.push(refit.r_squared);
                }
            }
        }
        TaskType::BinaryClassification => {
            for _ in 0..n_boot {
                let idx: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
                let boot_data = subsample_rows(data, &idx);
                let boot_y: Vec<f64> = idx.iter().map(|&i| y[i]).collect();
                let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &idx));
                let has_both = boot_y.iter().any(|&v| v < 0.5) && boot_y.iter().any(|&v| v >= 0.5);
                if !has_both {
                    continue;
                }
                if let Ok(refit) =
                    functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, 25, 1e-6)
                {
                    all_beta_t.push(refit.beta_t.clone());
                    let coefs: Vec<f64> = (0..ncomp).map(|k| refit.coefficients[1 + k]).collect();
                    all_abs_coefs.push(coefs.iter().map(|c| c.abs()).collect());
                    all_coefs.push(coefs);
                    all_metrics.push(refit.accuracy);
                }
            }
        }
        TaskType::MulticlassClassification(_) => {
            return Err(FdarError::InvalidParameter {
                parameter: "task_type",
                message: "stability analysis not supported for multiclass".into(),
            });
        }
    }

    build_stability_result(
        &all_beta_t,
        &all_coefs,
        &all_abs_coefs,
        &all_metrics,
        m,
        ncomp,
    )
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "generic_stability",
        detail: "not enough successful bootstrap refits; try increasing n_boot or check that the model fits reliably on subsampled data".into(),
    })
}

/// Generic VIF for any FPC-based model (only depends on score matrix).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::ComputationFailed`] if the internal VIF computation
/// fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_vif(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<VifResult, FdarError> {
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
    let ncomp = model.ncomp();
    let scores = model.project(data);
    compute_vif_from_scores(&scores, ncomp, scalar_covariates, n)
}
