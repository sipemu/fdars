use crate::error::FdarError;
use crate::explain::{
    compute_mean_scalar, compute_sobol_component, generate_sobol_matrices, SobolIndicesResult,
};
use crate::matrix::FdMatrix;
use rand::prelude::*;

use super::FpcPredictor;

/// Generic Sobol sensitivity indices for any FPC-based model (Saltelli MC).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 2 rows
/// or its column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or the
/// model has zero components.
/// Returns [`FdarError::ComputationFailed`] if the variance of model output
/// is near zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_sobol_indices(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<SobolIndicesResult, FdarError> {
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
    if n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    let p_scalar = scalar_covariates.map_or(0, crate::matrix::FdMatrix::ncols);
    let scores = model.project(data);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let eval_model = |s: &[f64]| -> f64 {
        let sc = if mean_z.is_empty() {
            None
        } else {
            Some(mean_z.as_slice())
        };
        model.predict_from_scores(s, sc)
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let (mat_a, mat_b) = generate_sobol_matrices(&scores, n, ncomp, n_samples, &mut rng);

    let f_a: Vec<f64> = mat_a.iter().map(|s| eval_model(s)).collect();
    let f_b: Vec<f64> = mat_b.iter().map(|s| eval_model(s)).collect();

    let mean_fa = f_a.iter().sum::<f64>() / n_samples as f64;
    // Monte Carlo estimate, population variance
    let var_fa = f_a.iter().map(|&v| (v - mean_fa).powi(2)).sum::<f64>() / n_samples as f64;

    if var_fa < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "generic_sobol_indices",
            detail: "variance of model output is near zero".into(),
        });
    }

    let mut first_order = vec![0.0; ncomp];
    let mut total_order = vec![0.0; ncomp];
    let mut component_variance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let (s_k, st_k) = compute_sobol_component(
            &mat_a,
            &mat_b,
            &f_a,
            &f_b,
            var_fa,
            k,
            n_samples,
            &eval_model,
        );
        first_order[k] = s_k;
        total_order[k] = st_k;
        component_variance[k] = s_k * var_fa;
    }

    Ok(SobolIndicesResult {
        first_order,
        total_order,
        var_y: var_fa,
        component_variance,
    })
}
