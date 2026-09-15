use crate::error::FdarError;
use crate::explain::{
    accumulate_kernel_shap_sample, compute_column_means, compute_mean_scalar, get_obs_scalar,
    sample_random_coalition, shapley_kernel_weight, solve_kernel_shap_obs, FpcShapValues,
};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Compute Kernel SHAP row for a single observation.
#[allow(clippy::too_many_arguments)]
fn shap_row_for_obs(
    model: &dyn FpcPredictor,
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    mean_scores: &[f64],
    mean_z: &[f64],
    p_scalar: usize,
    ncomp: usize,
    n_samples: usize,
    seed: u64,
    i: usize,
) -> Vec<f64> {
    let mut rng_i = crate::helpers::seed_for_thread(seed, i);
    let obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
    let obs_z = get_obs_scalar(scalar_covariates, i, p_scalar, mean_z);

    let mut ata = vec![0.0; ncomp * ncomp];
    let mut atb = vec![0.0; ncomp];
    let mut coal_scores = vec![0.0; ncomp];

    let f_base = model.predict_from_scores(
        mean_scores,
        if obs_z.is_empty() { None } else { Some(&obs_z) },
    );

    accumulate_shap_samples(
        model,
        mean_scores,
        &obs_scores,
        &obs_z,
        ncomp,
        n_samples,
        &mut rng_i,
        f_base,
        &mut coal_scores,
        &mut ata,
        &mut atb,
    );

    let mut local_values = FdMatrix::zeros(1, ncomp);
    solve_kernel_shap_obs(&mut ata, &atb, ncomp, &mut local_values, 0);
    (0..ncomp).map(|k| local_values[(0, k)]).collect()
}

/// Accumulate kernel SHAP samples for one observation.
#[allow(clippy::too_many_arguments)]
fn accumulate_shap_samples(
    model: &dyn FpcPredictor,
    mean_scores: &[f64],
    obs_scores: &[f64],
    obs_z: &[f64],
    ncomp: usize,
    n_samples: usize,
    rng_i: &mut rand::prelude::StdRng,
    f_base: f64,
    coal_scores: &mut [f64],
    ata: &mut [f64],
    atb: &mut [f64],
) {
    for _ in 0..n_samples {
        let (coalition, s_size) = sample_random_coalition(rng_i, ncomp);
        let weight = shapley_kernel_weight(ncomp, s_size);
        for (k, &in_coal) in coalition.iter().enumerate() {
            coal_scores[k] = if in_coal {
                obs_scores[k]
            } else {
                mean_scores[k]
            };
        }
        let f_coal = model.predict_from_scores(
            coal_scores,
            if obs_z.is_empty() { None } else { Some(obs_z) },
        );
        let y_val = f_coal - f_base;
        accumulate_kernel_shap_sample(ata, atb, &coalition, weight, y_val, ncomp);
    }
}

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
            shap_row_for_obs(
                model,
                &scores,
                scalar_covariates,
                &mean_scores,
                &mean_z,
                p_scalar,
                ncomp,
                n_samples,
                seed,
                i,
            )
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
