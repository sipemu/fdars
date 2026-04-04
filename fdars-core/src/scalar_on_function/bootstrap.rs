use super::{fregre_lm, functional_logistic, BootstrapCiResult};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;

#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Bootstrap CIs for β(t)
// ---------------------------------------------------------------------------

/// Gather rows from `src` by index (with replacement), returning a new matrix.
fn subsample_rows(src: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = src.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (dst_i, &src_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(dst_i, j)] = src[(src_i, j)];
        }
    }
    out
}

use crate::helpers::quantile_sorted as quantile;

/// Bootstrap confidence intervals for β(t) from a functional linear model.
///
/// Uses cases bootstrap (resampling observation indices with replacement) to
/// build pointwise and simultaneous confidence bands for the functional
/// coefficient β(t).
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar response vector (length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
/// * `n_boot` — Number of bootstrap replicates
/// * `alpha` — Significance level (e.g., 0.05 for 95% CI)
/// * `seed` — RNG seed for reproducibility
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_boot` is zero or `alpha` is
/// not in the open interval (0, 1).
/// Returns [`FdarError::ComputationFailed`] if fewer than 3 bootstrap
/// replicates converge, or if the original [`fregre_lm`] fit fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bootstrap_ci_fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    alpha: f64,
    seed: u64,
) -> Result<BootstrapCiResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y",
            expected: format!("n >= 3, m > 0, y.len() == n (n={n}, m={m})"),
            actual: format!("n={}, m={}, y.len()={}", n, m, y.len()),
        });
    }
    if n_boot == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_boot",
            message: "must be >= 1".to_string(),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
    }

    // Fit original model
    let original = fregre_lm(data, y, scalar_covariates, ncomp)?;
    let center = original.beta_t.clone();

    // Bootstrap replicates
    let boot_betas: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_boot)
        .filter_map(|b| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let boot_data = subsample_rows(data, &indices);
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &indices));

            fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp)
                .ok()
                .map(|fit| fit.beta_t)
        })
        .collect();

    let n_boot_success = boot_betas.len();
    if n_boot_success < 3 {
        return Err(FdarError::ComputationFailed {
            operation: "bootstrap_ci_fregre_lm",
            detail: format!(
                "only {n_boot_success} of {n_boot} bootstrap replicates converged (need >= 3)"
            ),
        });
    }

    // Pointwise bands: sort each grid point across replicates
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let mut lower = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut boot_se = vec![0.0; m];

    for j in 0..m {
        let mut vals: Vec<f64> = boot_betas.iter().map(|b| b[j]).collect();
        crate::helpers::sort_nan_safe(&mut vals);
        lower[j] = quantile(&vals, lo_q);
        upper[j] = quantile(&vals, hi_q);

        // Bootstrap SE at this grid point
        let mean_j: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var_j: f64 =
            vals.iter().map(|&v| (v - mean_j).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        boot_se[j] = var_j.sqrt().max(1e-15);
    }

    // Simultaneous bands: sup-norm bootstrap
    let mut t_stats: Vec<f64> = boot_betas
        .iter()
        .map(|b| {
            (0..m)
                .map(|j| ((b[j] - center[j]) / boot_se[j]).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    crate::helpers::sort_nan_safe(&mut t_stats);

    let c_alpha = quantile(&t_stats, 1.0 - alpha);
    let sim_lower: Vec<f64> = (0..m).map(|j| center[j] - c_alpha * boot_se[j]).collect();
    let sim_upper: Vec<f64> = (0..m).map(|j| center[j] + c_alpha * boot_se[j]).collect();

    Ok(BootstrapCiResult {
        lower,
        upper,
        center,
        sim_lower,
        sim_upper,
        n_boot_success,
    })
}

/// Bootstrap confidence intervals for β(t) from a functional logistic model.
///
/// Same algorithm as [`bootstrap_ci_fregre_lm`] but using [`functional_logistic`]
/// as the inner estimator. Degenerate resamples (all-0 or all-1 y) fail naturally.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Binary response vector (0.0 or 1.0, length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
/// * `n_boot` — Number of bootstrap replicates
/// * `alpha` — Significance level
/// * `seed` — RNG seed
/// * `max_iter` — Maximum IRLS iterations per replicate
/// * `tol` — IRLS convergence tolerance
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bootstrap_ci_functional_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    alpha: f64,
    seed: u64,
    max_iter: usize,
    tol: f64,
) -> Result<BootstrapCiResult, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y",
            expected: format!("n >= 3, m > 0, y.len() == n (n={n}, m={m})"),
            actual: format!("n={}, m={}, y.len()={}", n, m, y.len()),
        });
    }
    if n_boot == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_boot",
            message: "must be >= 1".to_string(),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
    }

    // Fit original model
    let original = functional_logistic(data, y, scalar_covariates, ncomp, max_iter, tol)?;
    let center = original.beta_t.clone();

    // Bootstrap replicates
    let boot_betas: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_boot)
        .filter_map(|b| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let boot_data = subsample_rows(data, &indices);
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &indices));

            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, max_iter, tol)
                .ok()
                .map(|fit| fit.beta_t)
        })
        .collect();

    let n_boot_success = boot_betas.len();
    if n_boot_success < 3 {
        return Err(FdarError::ComputationFailed {
            operation: "bootstrap_ci_functional_logistic",
            detail: format!(
                "only {n_boot_success} of {n_boot} bootstrap replicates converged (need >= 3)"
            ),
        });
    }

    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let mut lower = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut boot_se = vec![0.0; m];

    for j in 0..m {
        let mut vals: Vec<f64> = boot_betas.iter().map(|b| b[j]).collect();
        crate::helpers::sort_nan_safe(&mut vals);
        lower[j] = quantile(&vals, lo_q);
        upper[j] = quantile(&vals, hi_q);

        let mean_j: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var_j: f64 =
            vals.iter().map(|&v| (v - mean_j).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        boot_se[j] = var_j.sqrt().max(1e-15);
    }

    let mut t_stats: Vec<f64> = boot_betas
        .iter()
        .map(|b| {
            (0..m)
                .map(|j| ((b[j] - center[j]) / boot_se[j]).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    crate::helpers::sort_nan_safe(&mut t_stats);

    let c_alpha = quantile(&t_stats, 1.0 - alpha);
    let sim_lower: Vec<f64> = (0..m).map(|j| center[j] - c_alpha * boot_se[j]).collect();
    let sim_upper: Vec<f64> = (0..m).map(|j| center[j] + c_alpha * boot_se[j]).collect();

    Ok(BootstrapCiResult {
        lower,
        upper,
        center,
        sim_lower,
        sim_upper,
        n_boot_success,
    })
}
