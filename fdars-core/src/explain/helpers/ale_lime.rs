//! ALE (Accumulated Local Effects) and LIME helpers.

use crate::matrix::FdMatrix;
use crate::scalar_on_function::{cholesky_factor, cholesky_forward_back};
use rand::prelude::*;

/// ALE computation shared between linear and logistic models.
pub(crate) fn compute_ale(
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    p_scalar: usize,
    component: usize,
    n_bins: usize,
    predict: &dyn Fn(&[f64], Option<&[f64]>) -> f64,
) -> super::super::ale_lime::AleResult {
    use super::super::ale_lime::AleResult;

    let mut col: Vec<(f64, usize)> = (0..n).map(|i| (scores[(i, component)], i)).collect();
    col.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let bin_edges = compute_ale_bin_edges(&col, n, n_bins);
    let n_bins_actual = bin_edges.len() - 1;
    let bin_assignments = assign_ale_bins(&col, &bin_edges, n, n_bins_actual);

    let mut deltas = vec![0.0; n_bins_actual];
    let mut bin_counts = vec![0usize; n_bins_actual];

    for i in 0..n {
        let b = bin_assignments[i];
        bin_counts[b] += 1;

        let mut obs_scores: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
        let obs_z: Option<Vec<f64>> = if p_scalar > 0 {
            scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i, j)]).collect())
        } else {
            None
        };
        let z_ref = obs_z.as_deref();

        obs_scores[component] = bin_edges[b + 1];
        let f_upper = predict(&obs_scores, z_ref);
        obs_scores[component] = bin_edges[b];
        let f_lower = predict(&obs_scores, z_ref);

        deltas[b] += f_upper - f_lower;
    }

    for b in 0..n_bins_actual {
        if bin_counts[b] > 0 {
            deltas[b] /= bin_counts[b] as f64;
        }
    }

    let mut ale_values = vec![0.0; n_bins_actual];
    ale_values[0] = deltas[0];
    for b in 1..n_bins_actual {
        ale_values[b] = ale_values[b - 1] + deltas[b];
    }

    let total_n: usize = bin_counts.iter().sum();
    if total_n > 0 {
        let weighted_mean: f64 = ale_values
            .iter()
            .zip(&bin_counts)
            .map(|(&a, &c)| a * c as f64)
            .sum::<f64>()
            / total_n as f64;
        for v in &mut ale_values {
            *v -= weighted_mean;
        }
    }

    let bin_midpoints: Vec<f64> = (0..n_bins_actual)
        .map(|b| (bin_edges[b] + bin_edges[b + 1]) / 2.0)
        .collect();

    AleResult {
        bin_midpoints,
        ale_values,
        bin_edges,
        bin_counts,
        component,
    }
}

/// Compute quantile-based ALE bin edges from sorted component values.
fn compute_ale_bin_edges(sorted_col: &[(f64, usize)], n: usize, n_bins: usize) -> Vec<f64> {
    let actual_bins = n_bins.min(n);
    let mut bin_edges = Vec::with_capacity(actual_bins + 1);
    bin_edges.push(sorted_col[0].0);
    for b in 1..actual_bins {
        let idx = crate::utility::f64_to_usize_clamped(b as f64 / actual_bins as f64 * n as f64);
        let idx = idx.min(n - 1);
        let val = sorted_col[idx].0;
        if (val - *bin_edges.last().expect("non-empty collection")).abs() > 1e-15 {
            bin_edges.push(val);
        }
    }
    let last_val = sorted_col[n - 1].0;
    if (last_val - *bin_edges.last().expect("non-empty collection")).abs() > 1e-15 {
        bin_edges.push(last_val);
    }
    if bin_edges.len() < 2 {
        bin_edges.push(bin_edges[0] + 1.0);
    }
    bin_edges
}

/// Assign observations to ALE bins.
fn assign_ale_bins(
    sorted_col: &[(f64, usize)],
    bin_edges: &[f64],
    n: usize,
    n_bins_actual: usize,
) -> Vec<usize> {
    let mut bin_assignments = vec![0usize; n];
    for &(val, orig_idx) in sorted_col {
        let mut b = n_bins_actual - 1;
        for bb in 0..n_bins_actual - 1 {
            if val < bin_edges[bb + 1] {
                b = bb;
                break;
            }
        }
        bin_assignments[orig_idx] = b;
    }
    bin_assignments
}

/// LIME computation shared between linear and logistic models.
pub(crate) fn compute_lime(
    obs_scores: &[f64],
    score_sd: &[f64],
    ncomp: usize,
    n_samples: usize,
    kernel_width: f64,
    seed: u64,
    observation: usize,
    predict: &dyn Fn(&[f64]) -> f64,
) -> Option<super::super::ale_lime::LimeResult> {
    use super::super::ale_lime::LimeResult;

    let mut rng = StdRng::seed_from_u64(seed);

    let (perturbed, predictions, weights) = sample_lime_perturbations(
        obs_scores,
        score_sd,
        ncomp,
        n_samples,
        kernel_width,
        &mut rng,
        predict,
    )?;

    // Weighted OLS: fit y = intercept + sum beta_k (z_k - obs_k)
    let p = ncomp + 1;
    let mut ata = vec![0.0; p * p];
    let mut atb = vec![0.0; p];

    for i in 0..n_samples {
        let w = weights[i];
        let mut x = vec![0.0; p];
        x[0] = 1.0;
        for k in 0..ncomp {
            x[1 + k] = perturbed[i][k] - obs_scores[k];
        }
        for j1 in 0..p {
            for j2 in 0..p {
                ata[j1 * p + j2] += w * x[j1] * x[j2];
            }
            atb[j1] += w * x[j1] * predictions[i];
        }
    }

    for j in 0..p {
        ata[j * p + j] += 1e-10;
    }

    let l = cholesky_factor(&ata, p).ok()?;
    let beta = cholesky_forward_back(&l, &atb, p);

    let local_intercept = beta[0];
    let attributions: Vec<f64> = beta[1..].to_vec();
    let local_r_squared = weighted_r_squared(
        &predictions,
        &beta,
        &perturbed,
        obs_scores,
        &weights,
        ncomp,
        n_samples,
    );

    Some(LimeResult {
        observation,
        attributions,
        local_intercept,
        local_r_squared,
        kernel_width,
    })
}

/// Sample LIME perturbations, compute predictions and kernel weights.
/// Returns None if Normal distribution creation fails.
fn sample_lime_perturbations(
    obs_scores: &[f64],
    score_sd: &[f64],
    ncomp: usize,
    n_samples: usize,
    kernel_width: f64,
    rng: &mut StdRng,
    predict: &dyn Fn(&[f64]) -> f64,
) -> Option<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> {
    use rand_distr::Normal;

    let mut perturbed = vec![vec![0.0; ncomp]; n_samples];
    let mut predictions = vec![0.0; n_samples];
    let mut weights = vec![0.0; n_samples];

    for i in 0..n_samples {
        let mut dist_sq = 0.0;
        for k in 0..ncomp {
            let normal = Normal::new(obs_scores[k], score_sd[k]).ok()?;
            perturbed[i][k] = rng.sample(normal);
            let d = perturbed[i][k] - obs_scores[k];
            dist_sq += d * d;
        }
        predictions[i] = predict(&perturbed[i]);
        weights[i] = (-dist_sq / (2.0 * kernel_width * kernel_width)).exp();
    }
    Some((perturbed, predictions, weights))
}

/// Weighted R^2 from predictions, fitted values, and weights.
fn weighted_r_squared(
    predictions: &[f64],
    beta: &[f64],
    perturbed: &[Vec<f64>],
    obs_scores: &[f64],
    weights: &[f64],
    ncomp: usize,
    n_samples: usize,
) -> f64 {
    let w_sum: f64 = weights.iter().sum();
    let w_mean_y: f64 = weights
        .iter()
        .zip(predictions)
        .map(|(&w, &y)| w * y)
        .sum::<f64>()
        / w_sum;

    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for i in 0..n_samples {
        let mut yhat = beta[0];
        for k in 0..ncomp {
            yhat += beta[1 + k] * (perturbed[i][k] - obs_scores[k]);
        }
        ss_tot += weights[i] * (predictions[i] - w_mean_y).powi(2);
        ss_res += weights[i] * (predictions[i] - yhat).powi(2);
    }

    if ss_tot > 0.0 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    }
}
