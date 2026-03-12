//! Importance, permutation, and Sobol sensitivity helpers.

use crate::matrix::FdMatrix;
use rand::prelude::*;

use super::projection::clone_scores_matrix;

/// Shuffle component k globally (unconditional).
pub(crate) fn shuffle_global(
    perm_scores: &mut FdMatrix,
    scores: &FdMatrix,
    k: usize,
    n: usize,
    rng: &mut StdRng,
) {
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(rng);
    for i in 0..n {
        perm_scores[(i, k)] = scores[(idx[i], k)];
    }
}

/// Shuffle component k within conditional bins.
pub(crate) fn shuffle_within_bins(
    perm_scores: &mut FdMatrix,
    scores: &FdMatrix,
    bins: &[Vec<usize>],
    k: usize,
    rng: &mut StdRng,
) {
    for bin in bins {
        if bin.len() <= 1 {
            continue;
        }
        let mut bin_indices = bin.clone();
        bin_indices.shuffle(rng);
        for (rank, &orig_idx) in bin.iter().enumerate() {
            perm_scores[(orig_idx, k)] = scores[(bin_indices[rank], k)];
        }
    }
}

/// Compute conditioning bins for conditional permutation importance.
pub(crate) fn compute_conditioning_bins(
    scores: &FdMatrix,
    ncomp: usize,
    target_k: usize,
    n: usize,
    n_bins: usize,
) -> Vec<Vec<usize>> {
    let mut cond_var: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        for c in 0..ncomp {
            if c != target_k {
                cond_var[i] += scores[(i, c)].abs();
            }
        }
    }

    let mut sorted_cond: Vec<(f64, usize)> =
        cond_var.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    sorted_cond.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let actual_bins = n_bins.min(n);
    let mut bin_assignment = vec![0usize; n];
    for (rank, &(_, idx)) in sorted_cond.iter().enumerate() {
        bin_assignment[idx] = (rank * actual_bins / n).min(actual_bins - 1);
    }

    let mut bins: Vec<Vec<usize>> = vec![vec![]; actual_bins];
    for i in 0..n {
        bins[bin_assignment[i]].push(i);
    }
    bins
}

/// Run conditional + unconditional permutations for one component and return mean metrics.
pub(crate) fn permute_component<F: Fn(&FdMatrix) -> f64>(
    scores: &FdMatrix,
    bins: &[Vec<usize>],
    k: usize,
    n: usize,
    ncomp: usize,
    n_perm: usize,
    rng: &mut StdRng,
    metric_fn: &F,
) -> (f64, f64) {
    let mut sum_cond = 0.0;
    let mut sum_uncond = 0.0;
    for _ in 0..n_perm {
        let mut perm_cond = clone_scores_matrix(scores, n, ncomp);
        let mut perm_uncond = clone_scores_matrix(scores, n, ncomp);
        shuffle_within_bins(&mut perm_cond, scores, bins, k, rng);
        shuffle_global(&mut perm_uncond, scores, k, n, rng);
        sum_cond += metric_fn(&perm_cond);
        sum_uncond += metric_fn(&perm_uncond);
    }
    (sum_cond / n_perm as f64, sum_uncond / n_perm as f64)
}

/// Generate Sobol A and B matrices by resampling from FPC scores.
pub(crate) fn generate_sobol_matrices(
    scores: &FdMatrix,
    n: usize,
    ncomp: usize,
    n_samples: usize,
    rng: &mut StdRng,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut mat_a = vec![vec![0.0; ncomp]; n_samples];
    let mut mat_b = vec![vec![0.0; ncomp]; n_samples];
    for i in 0..n_samples {
        let ia = rng.gen_range(0..n);
        let ib = rng.gen_range(0..n);
        for k in 0..ncomp {
            mat_a[i][k] = scores[(ia, k)];
            mat_b[i][k] = scores[(ib, k)];
        }
    }
    (mat_a, mat_b)
}

/// Compute first-order and total-order Sobol indices for one component.
pub(crate) fn compute_sobol_component(
    mat_a: &[Vec<f64>],
    mat_b: &[Vec<f64>],
    f_a: &[f64],
    f_b: &[f64],
    var_fa: f64,
    k: usize,
    n_samples: usize,
    eval_model: &dyn Fn(&[f64]) -> f64,
) -> (f64, f64) {
    let f_ab_k: Vec<f64> = (0..n_samples)
        .map(|i| {
            let mut s = mat_a[i].clone();
            s[k] = mat_b[i][k];
            eval_model(&s)
        })
        .collect();

    let s_k: f64 = (0..n_samples)
        .map(|i| f_b[i] * (f_ab_k[i] - f_a[i]))
        .sum::<f64>()
        / n_samples as f64
        / var_fa;

    let st_k: f64 = (0..n_samples)
        .map(|i| (f_a[i] - f_ab_k[i]).powi(2))
        .sum::<f64>()
        / (2.0 * n_samples as f64 * var_fa);

    (s_k, st_k)
}
