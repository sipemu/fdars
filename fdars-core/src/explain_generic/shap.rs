use crate::error::FdarError;
use crate::explain::{
    accumulate_kernel_shap_sample, compute_column_means, compute_mean_scalar, get_obs_scalar,
    sample_random_coalition, shapley_kernel_weight, solve_kernel_shap_obs, FpcShapValues,
};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;

use super::FpcPredictor;

/// Generic Kernel SHAP values for any FPC-based model.
///
/// For nonlinear models uses sampling-based Kernel SHAP; linear models get
/// the same approximation (which converges to exact with enough samples).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if `n_samples` is zero or the
/// model has zero components.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_shap_values(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n_samples: usize,
    seed: u64,
) -> Result<FpcShapValues, FdarError> {
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

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
    let mean_scores = compute_column_means(&scores, ncomp);
    let mean_z = compute_mean_scalar(scalar_covariates, p_scalar, n);

    let base_value = model.predict_from_scores(
        &mean_scores,
        if mean_z.is_empty() {
            None
        } else {
            Some(&mean_z)
        },
    );

    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut rng_i = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
            let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let obs_z = get_obs_scalar(scalar_covariates, i, p_scalar, &mean_z);

            let mut ata = vec![0.0; ncomp * ncomp];
            let mut atb = vec![0.0; ncomp];
            // Pre-allocate coalition scores buffer outside the inner loop
            let mut coal_scores = vec![0.0; ncomp];

            // Pre-compute f_base once (it is constant across all coalitions)
            let f_base = model.predict_from_scores(
                &mean_scores,
                if obs_z.is_empty() { None } else { Some(&obs_z) },
            );

            for _ in 0..n_samples {
                let (coalition, s_size) = sample_random_coalition(&mut rng_i, ncomp);
                let weight = shapley_kernel_weight(ncomp, s_size);
                // Reuse pre-allocated buffer instead of allocating a new Vec each iteration
                for (k, &in_coal) in coalition.iter().enumerate() {
                    coal_scores[k] = if in_coal {
                        obs_scores[k]
                    } else {
                        mean_scores[k]
                    };
                }

                let f_coal = model.predict_from_scores(
                    &coal_scores,
                    if obs_z.is_empty() { None } else { Some(&obs_z) },
                );
                let y_val = f_coal - f_base;

                accumulate_kernel_shap_sample(&mut ata, &mut atb, &coalition, weight, y_val, ncomp);
            }

            // Solve locally and return row
            let mut local_values = FdMatrix::zeros(1, ncomp);
            solve_kernel_shap_obs(&mut ata, &atb, ncomp, &mut local_values, 0);
            (0..ncomp).map(|k| local_values[(0, k)]).collect()
        })
        .collect();

    let mut values = FdMatrix::zeros(n, ncomp);
    for (i, row) in rows.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            values[(i, k)] = v;
        }
    }

    Ok(FpcShapValues {
        values,
        base_value,
        mean_scores,
    })
}
