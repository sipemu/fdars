//! Kernel matrix, prototype/criticism selection, and depth helpers.

use crate::depth;
use crate::matrix::FdMatrix;

/// Compute pairwise Gaussian kernel matrix from FPC scores.
pub(crate) fn gaussian_kernel_matrix(scores: &FdMatrix, ncomp: usize, bandwidth: f64) -> Vec<f64> {
    let n = scores.nrows();
    let mut k = vec![0.0; n * n];
    let bw2 = 2.0 * bandwidth * bandwidth;
    for i in 0..n {
        k[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let mut dist_sq = 0.0;
            for c in 0..ncomp {
                let d = scores[(i, c)] - scores[(j, c)];
                dist_sq += d * d;
            }
            let val = (-dist_sq / bw2).exp();
            k[i * n + j] = val;
            k[j * n + i] = val;
        }
    }
    k
}

/// Compute median pairwise distance from FPC scores (bandwidth heuristic).
pub(crate) fn median_bandwidth(scores: &FdMatrix, n: usize, ncomp: usize) -> f64 {
    let mut dists: Vec<f64> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for c in 0..ncomp {
                let d = scores[(i, c)] - scores[(j, c)];
                d2 += d * d;
            }
            dists.push(d2.sqrt());
        }
    }
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if dists.is_empty() {
        1.0
    } else {
        dists[dists.len() / 2].max(1e-10)
    }
}

/// Compute kernel mean: mu_data[i] = (1/n) sum_j K(i,j).
pub(crate) fn compute_kernel_mean(kernel: &[f64], n: usize) -> Vec<f64> {
    let mut mu_data = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            mu_data[i] += kernel[i * n + j];
        }
        mu_data[i] /= n as f64;
    }
    mu_data
}

/// Greedy MMD-based prototype selection.
pub(crate) fn greedy_prototype_selection(
    mu_data: &[f64],
    kernel: &[f64],
    n: usize,
    n_prototypes: usize,
) -> (Vec<usize>, Vec<bool>) {
    let mut selected: Vec<usize> = Vec::with_capacity(n_prototypes);
    let mut is_selected = vec![false; n];

    for _ in 0..n_prototypes {
        let best_idx = find_best_prototype(mu_data, kernel, n, &is_selected, &selected);
        selected.push(best_idx);
        is_selected[best_idx] = true;
    }
    (selected, is_selected)
}

/// Compute witness function values.
pub(crate) fn compute_witness(
    kernel: &[f64],
    mu_data: &[f64],
    selected: &[usize],
    n: usize,
) -> Vec<f64> {
    let mut witness = vec![0.0; n];
    for i in 0..n {
        let mean_k_selected: f64 =
            selected.iter().map(|&j| kernel[i * n + j]).sum::<f64>() / selected.len() as f64;
        witness[i] = mu_data[i] - mean_k_selected;
    }
    witness
}

/// Find the best unselected prototype candidate.
fn find_best_prototype(
    mu_data: &[f64],
    kernel: &[f64],
    n: usize,
    is_selected: &[bool],
    selected: &[usize],
) -> usize {
    let mut best_idx = 0;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..n {
        if is_selected[i] {
            continue;
        }
        let mut score = 2.0 * mu_data[i];
        if !selected.is_empty() {
            let mean_k: f64 =
                selected.iter().map(|&j| kernel[i * n + j]).sum::<f64>() / selected.len() as f64;
            score -= mean_k;
        }
        if score > best_val {
            best_val = score;
            best_idx = i;
        }
    }
    best_idx
}

/// Compute depth of scores using the specified depth type.
pub(crate) fn compute_score_depths(
    scores: &FdMatrix,
    depth_type: super::super::advanced::DepthType,
) -> Vec<f64> {
    match depth_type {
        super::super::advanced::DepthType::FraimanMuniz => {
            depth::fraiman_muniz_1d(scores, scores, false)
        }
        super::super::advanced::DepthType::ModifiedBand => {
            depth::modified_band_1d(scores, scores)
        }
        super::super::advanced::DepthType::FunctionalSpatial => {
            depth::functional_spatial_1d(scores, scores, None)
        }
    }
}

/// Compute beta depth from bootstrap coefficient vectors.
pub(crate) fn beta_depth_from_bootstrap(
    boot_coefs: &[Vec<f64>],
    orig_coefs: &[f64],
    ncomp: usize,
    depth_type: super::super::advanced::DepthType,
) -> f64 {
    if boot_coefs.len() < 2 {
        return 0.0;
    }
    let mut boot_mat = FdMatrix::zeros(boot_coefs.len(), ncomp);
    for (i, coefs) in boot_coefs.iter().enumerate() {
        for k in 0..ncomp {
            boot_mat[(i, k)] = coefs[k];
        }
    }
    let mut orig_mat = FdMatrix::zeros(1, ncomp);
    for k in 0..ncomp {
        orig_mat[(0, k)] = orig_coefs[k];
    }
    compute_single_depth(&orig_mat, &boot_mat, depth_type)
}

/// Compute depth of a single row among a reference matrix using the specified depth type.
fn compute_single_depth(
    row: &FdMatrix,
    reference: &FdMatrix,
    depth_type: super::super::advanced::DepthType,
) -> f64 {
    let depths = match depth_type {
        super::super::advanced::DepthType::FraimanMuniz => {
            depth::fraiman_muniz_1d(row, reference, false)
        }
        super::super::advanced::DepthType::ModifiedBand => {
            depth::modified_band_1d(row, reference)
        }
        super::super::advanced::DepthType::FunctionalSpatial => {
            depth::functional_spatial_1d(row, reference, None)
        }
    };
    if depths.is_empty() {
        0.0
    } else {
        depths[0]
    }
}
