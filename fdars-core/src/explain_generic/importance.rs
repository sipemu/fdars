use crate::error::FdarError;
use crate::explain::{
    clone_scores_matrix, compute_conditioning_bins, permute_component,
    ConditionalPermutationImportanceResult, FpcPermutationImportance,
};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;

use super::{compute_baseline_metric, compute_metric_from_score_matrix, FpcPredictor};

/// Generic permutation importance for any FPC-based model.
///
/// Uses R² for regression, accuracy for classification.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its
/// column count does not match the model, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_perm` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Result<FpcPermutationImportance, FdarError> {
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
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "n_perm must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);
    let baseline = compute_baseline_metric(model, &scores, y, n);

    let results: Vec<(f64, f64)> = iter_maybe_parallel!(0..ncomp)
        .map(|k| {
            let mut rng_k = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
            let mut sum_metric = 0.0;
            for _ in 0..n_perm {
                let mut perm_scores = clone_scores_matrix(&scores, n, ncomp);
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng_k);
                for i in 0..n {
                    perm_scores[(i, k)] = scores[(idx[i], k)];
                }
                sum_metric += compute_metric_from_score_matrix(model, &perm_scores, y, n);
            }
            let mean_perm = sum_metric / n_perm as f64;
            (baseline - mean_perm, mean_perm)
        })
        .collect();

    let importance: Vec<f64> = results.iter().map(|&(imp, _)| imp).collect();
    let permuted_metric: Vec<f64> = results.iter().map(|&(_, pm)| pm).collect();

    Ok(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

/// Generic conditional permutation importance for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its
/// column count does not match the model, or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `n_perm` or `n_bins` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_conditional_permutation_importance(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    _scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ConditionalPermutationImportanceResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    if n_perm == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_perm",
            message: "n_perm must be > 0".into(),
        });
    }
    if n_bins == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bins",
            message: "n_bins must be > 0".into(),
        });
    }
    let ncomp = model.ncomp();
    let scores = model.project(data);

    let baseline = compute_baseline_metric(model, &scores, y, n);

    let metric_fn =
        |score_mat: &FdMatrix| -> f64 { compute_metric_from_score_matrix(model, score_mat, y, n) };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &metric_fn);
        permuted_metric[k] = mean_cond;
        importance[k] = baseline - mean_cond;
        unconditional_importance[k] = baseline - mean_uncond;
    }

    Ok(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}
