//! Permutation importance, pointwise importance, and conditional permutation importance.

use super::helpers::*;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{sigmoid, FregreLmResult, FunctionalLogisticResult};
use rand::prelude::*;

// ===========================================================================
// FPC Permutation Importance
// ===========================================================================

/// Result of FPC permutation importance.
pub struct FpcPermutationImportance {
    /// R^2 (or accuracy) drop per component (length ncomp).
    pub importance: Vec<f64>,
    /// Baseline metric (R^2 or accuracy).
    pub baseline_metric: f64,
    /// Mean metric after permuting each component.
    pub permuted_metric: Vec<f64>,
}

/// Permutation importance for a linear functional regression (metric = R^2).
pub fn fpc_permutation_importance(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Option<FpcPermutationImportance> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    // Baseline R^2 -- compute from same FPC-only prediction used in permuted path
    // to ensure consistent comparison (gamma terms are constant across permutations)
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return None;
    }
    let identity_idx: Vec<usize> = (0..n).collect();
    let ss_res_base = permuted_ss_res_linear(
        &scores,
        &fit.coefficients,
        y,
        n,
        ncomp,
        ncomp,
        &identity_idx,
    );
    let baseline = 1.0 - ss_res_base / ss_tot;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_r2 = 0.0;
        for _ in 0..n_perm {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            let ss_res_perm =
                permuted_ss_res_linear(&scores, &fit.coefficients, y, n, ncomp, k, &idx);
            sum_r2 += 1.0 - ss_res_perm / ss_tot;
        }
        let mean_perm = sum_r2 / n_perm as f64;
        permuted_metric[k] = mean_perm;
        importance[k] = baseline - mean_perm;
    }

    Some(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

/// Permutation importance for functional logistic regression (metric = accuracy).
pub fn fpc_permutation_importance_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    n_perm: usize,
    seed: u64,
) -> Option<FpcPermutationImportance> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 {
        return None;
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let baseline: f64 = (0..n)
        .filter(|&i| {
            let pred = if fit.probabilities[i] >= 0.5 {
                1.0
            } else {
                0.0
            };
            (pred - y[i]).abs() < 1e-10
        })
        .count() as f64
        / n as f64;

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];

    for k in 0..ncomp {
        let mut sum_acc = 0.0;
        for _ in 0..n_perm {
            let mut perm_scores = clone_scores_matrix(&scores, n, ncomp);
            shuffle_global(&mut perm_scores, &scores, k, n, &mut rng);
            sum_acc += logistic_accuracy_from_scores(
                &perm_scores,
                fit.intercept,
                &fit.coefficients,
                y,
                n,
                ncomp,
            );
        }
        let mean_acc = sum_acc / n_perm as f64;
        permuted_metric[k] = mean_acc;
        importance[k] = baseline - mean_acc;
    }

    Some(FpcPermutationImportance {
        importance,
        baseline_metric: baseline,
        permuted_metric,
    })
}

/// Compute SS_res with component k shuffled by given index permutation.
fn permuted_ss_res_linear(
    scores: &FdMatrix,
    coefficients: &[f64],
    y: &[f64],
    n: usize,
    ncomp: usize,
    k: usize,
    perm_idx: &[usize],
) -> f64 {
    (0..n)
        .map(|i| {
            let mut yhat = coefficients[0];
            for c in 0..ncomp {
                let s = if c == k {
                    scores[(perm_idx[i], c)]
                } else {
                    scores[(i, c)]
                };
                yhat += coefficients[1 + c] * s;
            }
            (y[i] - yhat).powi(2)
        })
        .sum()
}

// ===========================================================================
// Pointwise Variable Importance
// ===========================================================================

/// Result of pointwise variable importance analysis.
pub struct PointwiseImportanceResult {
    /// Importance at each grid point (length m).
    pub importance: Vec<f64>,
    /// Normalized importance summing to 1 (length m).
    pub importance_normalized: Vec<f64>,
    /// Per-component importance (ncomp x m).
    pub component_importance: FdMatrix,
    /// Variance of each FPC score (length ncomp).
    pub score_variance: Vec<f64>,
}

/// Pointwise variable importance for a linear functional regression model.
///
/// Measures how much X(t_j) contributes to prediction variance via the FPC decomposition.
pub fn pointwise_importance(fit: &FregreLmResult) -> Option<PointwiseImportanceResult> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.rotation.nrows();
    let n = fit.fpca.scores.nrows();
    if ncomp == 0 || m == 0 || n < 2 {
        return None;
    }

    let score_variance = compute_score_variance(&fit.fpca.scores, n, ncomp);
    let (component_importance, importance, importance_normalized) =
        compute_pointwise_importance_core(
            &fit.coefficients,
            &fit.fpca.rotation,
            &score_variance,
            ncomp,
            m,
        );

    Some(PointwiseImportanceResult {
        importance,
        importance_normalized,
        component_importance,
        score_variance,
    })
}

/// Pointwise variable importance for a functional logistic regression model.
pub fn pointwise_importance_logistic(
    fit: &FunctionalLogisticResult,
) -> Option<PointwiseImportanceResult> {
    let ncomp = fit.ncomp;
    let m = fit.fpca.rotation.nrows();
    let n = fit.fpca.scores.nrows();
    if ncomp == 0 || m == 0 || n < 2 {
        return None;
    }

    let score_variance = compute_score_variance(&fit.fpca.scores, n, ncomp);
    let (component_importance, importance, importance_normalized) =
        compute_pointwise_importance_core(
            &fit.coefficients,
            &fit.fpca.rotation,
            &score_variance,
            ncomp,
            m,
        );

    Some(PointwiseImportanceResult {
        importance,
        importance_normalized,
        component_importance,
        score_variance,
    })
}

/// Compute component importance matrix and aggregated importance.
fn compute_pointwise_importance_core(
    coefficients: &[f64],
    rotation: &FdMatrix,
    score_variance: &[f64],
    ncomp: usize,
    m: usize,
) -> (FdMatrix, Vec<f64>, Vec<f64>) {
    let mut component_importance = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let ck = coefficients[1 + k];
        for j in 0..m {
            component_importance[(k, j)] = (ck * rotation[(j, k)]).powi(2) * score_variance[k];
        }
    }

    let mut importance = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            importance[j] += component_importance[(k, j)];
        }
    }

    let total: f64 = importance.iter().sum();
    let importance_normalized = if total > 0.0 {
        importance.iter().map(|&v| v / total).collect()
    } else {
        vec![0.0; m]
    };

    (component_importance, importance, importance_normalized)
}

// ===========================================================================
// Conditional Permutation Importance
// ===========================================================================

/// Result of conditional permutation importance.
pub struct ConditionalPermutationImportanceResult {
    /// Conditional importance per FPC component, length ncomp.
    pub importance: Vec<f64>,
    /// Baseline metric (R^2 or accuracy).
    pub baseline_metric: f64,
    /// Mean metric after conditional permutation, length ncomp.
    pub permuted_metric: Vec<f64>,
    /// Unconditional (standard) permutation importance for comparison, length ncomp.
    pub unconditional_importance: Vec<f64>,
}

/// Conditional permutation importance for a linear functional regression model.
pub fn conditional_permutation_importance(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Option<ConditionalPermutationImportanceResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 || n_bins == 0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if ss_tot == 0.0 {
        return None;
    }
    let ss_res_base: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let baseline = 1.0 - ss_res_base / ss_tot;

    let predict_r2 = |score_mat: &FdMatrix| -> f64 {
        let ss_res: f64 = (0..n)
            .map(|i| {
                let mut yhat = fit.coefficients[0];
                for c in 0..ncomp {
                    yhat += fit.coefficients[1 + c] * score_mat[(i, c)];
                }
                (y[i] - yhat).powi(2)
            })
            .sum();
        1.0 - ss_res / ss_tot
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &predict_r2);
        permuted_metric[k] = mean_cond;
        importance[k] = baseline - mean_cond;
        unconditional_importance[k] = baseline - mean_uncond;
    }

    Some(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}

/// Conditional permutation importance for a functional logistic regression model.
pub fn conditional_permutation_importance_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n_bins: usize,
    n_perm: usize,
    seed: u64,
) -> Option<ConditionalPermutationImportanceResult> {
    let (n, m) = data.shape();
    if n == 0 || n != y.len() || m != fit.fpca.mean.len() || n_perm == 0 || n_bins == 0 {
        return None;
    }
    let _ = scalar_covariates;
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let baseline: f64 = (0..n)
        .filter(|&i| {
            let pred = if fit.probabilities[i] >= 0.5 {
                1.0
            } else {
                0.0
            };
            (pred - y[i]).abs() < 1e-10
        })
        .count() as f64
        / n as f64;

    let predict_acc = |score_mat: &FdMatrix| -> f64 {
        let correct: usize = (0..n)
            .filter(|&i| {
                let mut eta = fit.intercept;
                for c in 0..ncomp {
                    eta += fit.coefficients[1 + c] * score_mat[(i, c)];
                }
                let pred = if sigmoid(eta) >= 0.5 { 1.0 } else { 0.0 };
                (pred - y[i]).abs() < 1e-10
            })
            .count();
        correct as f64 / n as f64
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut importance = vec![0.0; ncomp];
    let mut permuted_metric = vec![0.0; ncomp];
    let mut unconditional_importance = vec![0.0; ncomp];

    for k in 0..ncomp {
        let bins = compute_conditioning_bins(&scores, ncomp, k, n, n_bins);
        let (mean_cond, mean_uncond) =
            permute_component(&scores, &bins, k, n, ncomp, n_perm, &mut rng, &predict_acc);
        permuted_metric[k] = mean_cond;
        importance[k] = baseline - mean_cond;
        unconditional_importance[k] = baseline - mean_uncond;
    }

    Some(ConditionalPermutationImportanceResult {
        importance,
        baseline_metric: baseline,
        permuted_metric,
        unconditional_importance,
    })
}
