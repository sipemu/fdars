//! Internal helper functions shared across explainability submodules.

use crate::depth;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{cholesky_factor, cholesky_forward_back, sigmoid};
use rand::prelude::*;

// ---------------------------------------------------------------------------
// Score projection and subsetting
// ---------------------------------------------------------------------------

/// Project data → FPC scores.
pub(crate) fn project_scores(
    data: &FdMatrix,
    mean: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
) -> FdMatrix {
    let (n, m) = data.shape();
    let mut scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (data[(i, j)] - mean[j]) * rotation[(j, k)];
            }
            scores[(i, k)] = s;
        }
    }
    scores
}

/// Subsample rows from an FdMatrix.
pub(crate) fn subsample_rows(data: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = data.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (new_i, &orig_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(new_i, j)] = data[(orig_i, j)];
        }
    }
    out
}

/// Clone an FdMatrix of scores.
pub(crate) fn clone_scores_matrix(scores: &FdMatrix, n: usize, ncomp: usize) -> FdMatrix {
    let mut perm = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for c in 0..ncomp {
            perm[(i, c)] = scores[(i, c)];
        }
    }
    perm
}

// ---------------------------------------------------------------------------
// Score statistics
// ---------------------------------------------------------------------------

/// Compute the grid for a single FPC score column.
pub(crate) fn make_grid(scores: &FdMatrix, component: usize, n_grid: usize) -> Vec<f64> {
    let n = scores.nrows();
    let mut mn = f64::INFINITY;
    let mut mx = f64::NEG_INFINITY;
    for i in 0..n {
        let v = scores[(i, component)];
        if v < mn {
            mn = v;
        }
        if v > mx {
            mx = v;
        }
    }
    if (mx - mn).abs() < 1e-15 {
        mx = mn + 1.0;
    }
    (0..n_grid)
        .map(|g| mn + (mx - mn) * g as f64 / (n_grid - 1) as f64)
        .collect()
}

/// Compute column means of an FdMatrix.
pub(crate) fn compute_column_means(mat: &FdMatrix, ncols: usize) -> Vec<f64> {
    let n = mat.nrows();
    let mut means = vec![0.0; ncols];
    for k in 0..ncols {
        for i in 0..n {
            means[k] += mat[(i, k)];
        }
        means[k] /= n as f64;
    }
    means
}

/// Compute mean scalar covariates from an optional FdMatrix.
pub(crate) fn compute_mean_scalar(
    scalar_covariates: Option<&FdMatrix>,
    p_scalar: usize,
    n: usize,
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar)
            .map(|j| {
                let mut s = 0.0;
                for i in 0..n {
                    s += sc[(i, j)];
                }
                s / n as f64
            })
            .collect()
    } else {
        vec![0.0; p_scalar]
    }
}

/// Compute score variance for each component (mean-zero scores from FPCA).
pub(crate) fn compute_score_variance(scores: &FdMatrix, n: usize, ncomp: usize) -> Vec<f64> {
    let mut score_variance = vec![0.0; ncomp];
    for k in 0..ncomp {
        let mut ss = 0.0;
        for i in 0..n {
            let s = scores[(i, k)];
            ss += s * s;
        }
        score_variance[k] = ss / (n - 1) as f64;
    }
    score_variance
}

/// Compute column means of ICE curves -> PDP.
pub(crate) fn ice_to_pdp(ice_curves: &FdMatrix, n: usize, n_grid: usize) -> Vec<f64> {
    let mut pdp = vec![0.0; n_grid];
    for g in 0..n_grid {
        for i in 0..n {
            pdp[g] += ice_curves[(i, g)];
        }
        pdp[g] /= n as f64;
    }
    pdp
}

// ---------------------------------------------------------------------------
// SHAP helpers
// ---------------------------------------------------------------------------

/// Binomial coefficient C(n, k).
fn binom_coeff(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k);
    let mut result: usize = 1;
    for i in 0..k {
        result = result.saturating_mul(n - i) / (i + 1);
    }
    result
}

/// Compute Shapley kernel weight for a coalition of given size.
pub(crate) fn shapley_kernel_weight(ncomp: usize, s_size: usize) -> f64 {
    if s_size == 0 || s_size == ncomp {
        1e6
    } else {
        let binom = binom_coeff(ncomp, s_size) as f64;
        if binom > 0.0 {
            (ncomp - 1) as f64 / (binom * s_size as f64 * (ncomp - s_size) as f64)
        } else {
            1.0
        }
    }
}

/// Sample a random coalition of FPC components via Fisher-Yates partial shuffle.
pub(crate) fn sample_random_coalition(rng: &mut StdRng, ncomp: usize) -> (Vec<bool>, usize) {
    let s_size = if ncomp <= 1 {
        rng.gen_range(0..=1usize)
    } else {
        rng.gen_range(1..ncomp)
    };
    let mut coalition = vec![false; ncomp];
    let mut indices: Vec<usize> = (0..ncomp).collect();
    for j in 0..s_size.min(ncomp) {
        let swap = rng.gen_range(j..ncomp);
        indices.swap(j, swap);
    }
    for j in 0..s_size {
        coalition[indices[j]] = true;
    }
    (coalition, s_size)
}

/// Build coalition scores: use observation value if in coalition, mean otherwise.
pub(crate) fn build_coalition_scores(
    coalition: &[bool],
    obs_scores: &[f64],
    mean_scores: &[f64],
) -> Vec<f64> {
    coalition
        .iter()
        .enumerate()
        .map(|(k, &in_coal)| {
            if in_coal {
                obs_scores[k]
            } else {
                mean_scores[k]
            }
        })
        .collect()
}

/// Get observation's scalar covariates, or use mean if unavailable.
pub(crate) fn get_obs_scalar(
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    p_scalar: usize,
    mean_z: &[f64],
) -> Vec<f64> {
    if p_scalar == 0 {
        return vec![];
    }
    if let Some(sc) = scalar_covariates {
        (0..p_scalar).map(|j| sc[(i, j)]).collect()
    } else {
        mean_z.to_vec()
    }
}

/// Accumulate one WLS sample for Kernel SHAP: A'A += w z z', A'b += w z y.
pub(crate) fn accumulate_kernel_shap_sample(
    ata: &mut [f64],
    atb: &mut [f64],
    coalition: &[bool],
    weight: f64,
    y_val: f64,
    ncomp: usize,
) {
    for k1 in 0..ncomp {
        let z1 = if coalition[k1] { 1.0 } else { 0.0 };
        for k2 in 0..ncomp {
            let z2 = if coalition[k2] { 1.0 } else { 0.0 };
            ata[k1 * ncomp + k2] += weight * z1 * z2;
        }
        atb[k1] += weight * z1 * y_val;
    }
}

/// Solve Kernel SHAP for one observation: regularize ATA, Cholesky solve, store in values matrix.
pub(crate) fn solve_kernel_shap_obs(
    ata: &mut [f64],
    atb: &[f64],
    ncomp: usize,
    values: &mut FdMatrix,
    i: usize,
) {
    for k in 0..ncomp {
        ata[k * ncomp + k] += 1e-10;
    }
    if let Some(l) = cholesky_factor(ata, ncomp) {
        let phi = cholesky_forward_back(&l, atb, ncomp);
        for k in 0..ncomp {
            values[(i, k)] = phi[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Friedman H-statistic helpers
// ---------------------------------------------------------------------------

/// Compute H^2 statistic from 1D and 2D PDPs.
pub(crate) fn compute_h_squared(
    pdp_2d: &FdMatrix,
    pdp_j: &[f64],
    pdp_k: &[f64],
    f_bar: f64,
    n_grid: usize,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for gj in 0..n_grid {
        for gk in 0..n_grid {
            let f2 = pdp_2d[(gj, gk)];
            let interaction = f2 - pdp_j[gj] - pdp_k[gk] + f_bar;
            num += interaction * interaction;
            let centered = f2 - f_bar;
            den += centered * centered;
        }
    }
    if den > 0.0 {
        num / den
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Importance / permutation helpers
// ---------------------------------------------------------------------------

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
    sorted_cond.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
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

// ---------------------------------------------------------------------------
// Prototype / criticism helpers
// ---------------------------------------------------------------------------

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
    dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

// ---------------------------------------------------------------------------
// Saliency / domain helpers
// ---------------------------------------------------------------------------

/// Compute saliency map: saliency[(i,j)] = sum_k weight_k * (scores[(i,k)] - mean_k) * rotation[(j,k)].
pub(crate) fn compute_saliency_map(
    scores: &FdMatrix,
    mean_scores: &[f64],
    weights: &[f64],
    rotation: &FdMatrix,
    n: usize,
    m: usize,
    ncomp: usize,
) -> FdMatrix {
    let mut saliency_map = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += weights[k] * (scores[(i, k)] - mean_scores[k]) * rotation[(j, k)];
            }
            saliency_map[(i, j)] = val;
        }
    }
    saliency_map
}

/// Mean absolute value per column of an n x m matrix.
pub(crate) fn mean_absolute_column(mat: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut result = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            result[j] += mat[(i, j)].abs();
        }
        result[j] /= n as f64;
    }
    result
}

/// Reconstruct delta function from delta scores and rotation matrix.
pub(crate) fn reconstruct_delta_function(
    delta_scores: &[f64],
    rotation: &FdMatrix,
    ncomp: usize,
    m: usize,
) -> Vec<f64> {
    let mut delta_function = vec![0.0; m];
    for j in 0..m {
        for k in 0..ncomp {
            delta_function[j] += delta_scores[k] * rotation[(j, k)];
        }
    }
    delta_function
}

// ---------------------------------------------------------------------------
// Domain selection helpers
// ---------------------------------------------------------------------------

/// Compute domain selection from beta_t.
pub(crate) fn compute_domain_selection(
    beta_t: &[f64],
    window_width: usize,
    threshold: f64,
) -> Option<super::sensitivity::DomainSelectionResult> {
    use super::sensitivity::DomainSelectionResult;

    let m = beta_t.len();
    if m == 0 || window_width == 0 || window_width > m || threshold <= 0.0 {
        return None;
    }

    let pointwise_importance: Vec<f64> = beta_t.iter().map(|&b| b * b).collect();
    let total_imp: f64 = pointwise_importance.iter().sum();
    if total_imp == 0.0 {
        return Some(DomainSelectionResult {
            pointwise_importance,
            intervals: vec![],
            window_width,
            threshold,
        });
    }

    // Sliding window with running sum
    let mut window_sum: f64 = pointwise_importance[..window_width].iter().sum();
    let mut raw_intervals: Vec<(usize, usize, f64)> = Vec::new();
    if window_sum / total_imp >= threshold {
        raw_intervals.push((0, window_width - 1, window_sum));
    }
    for start in 1..=(m - window_width) {
        window_sum -= pointwise_importance[start - 1];
        window_sum += pointwise_importance[start + window_width - 1];
        if window_sum / total_imp >= threshold {
            raw_intervals.push((start, start + window_width - 1, window_sum));
        }
    }

    let mut intervals = merge_overlapping_intervals(raw_intervals);
    intervals.sort_by(|a, b| b.importance.partial_cmp(&a.importance).unwrap());

    Some(DomainSelectionResult {
        pointwise_importance,
        intervals,
        window_width,
        threshold,
    })
}

/// Merge overlapping intervals, accumulating importance.
fn merge_overlapping_intervals(
    raw: Vec<(usize, usize, f64)>,
) -> Vec<super::sensitivity::ImportantInterval> {
    use super::sensitivity::ImportantInterval;

    let mut intervals: Vec<ImportantInterval> = Vec::new();
    for (s, e, imp) in raw {
        if let Some(last) = intervals.last_mut() {
            if s <= last.end_idx + 1 {
                last.end_idx = e;
                last.importance += imp;
                continue;
            }
        }
        intervals.push(ImportantInterval {
            start_idx: s,
            end_idx: e,
            importance: imp,
        });
    }
    intervals
}

// ---------------------------------------------------------------------------
// ALE helpers
// ---------------------------------------------------------------------------

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
) -> Option<super::ale_lime::AleResult> {
    use super::ale_lime::AleResult;

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

    Some(AleResult {
        bin_midpoints,
        ale_values,
        bin_edges,
        bin_counts,
        component,
    })
}

/// Compute quantile-based ALE bin edges from sorted component values.
fn compute_ale_bin_edges(sorted_col: &[(f64, usize)], n: usize, n_bins: usize) -> Vec<f64> {
    let actual_bins = n_bins.min(n);
    let mut bin_edges = Vec::with_capacity(actual_bins + 1);
    bin_edges.push(sorted_col[0].0);
    for b in 1..actual_bins {
        let idx = (b as f64 / actual_bins as f64 * n as f64) as usize;
        let idx = idx.min(n - 1);
        let val = sorted_col[idx].0;
        if (val - *bin_edges.last().unwrap()).abs() > 1e-15 {
            bin_edges.push(val);
        }
    }
    let last_val = sorted_col[n - 1].0;
    if (last_val - *bin_edges.last().unwrap()).abs() > 1e-15 {
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

// ---------------------------------------------------------------------------
// LIME helpers
// ---------------------------------------------------------------------------

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
) -> Option<super::ale_lime::LimeResult> {
    use super::ale_lime::LimeResult;

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

    let l = cholesky_factor(&ata, p)?;
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

// ---------------------------------------------------------------------------
// Anchor helpers
// ---------------------------------------------------------------------------

/// Beam search for anchor rules in FPC score space.
pub(crate) fn anchor_beam_search(
    scores: &FdMatrix,
    ncomp: usize,
    n: usize,
    observation: usize,
    precision_threshold: f64,
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> (super::advanced::AnchorRule, Vec<bool>) {
    let bin_edges: Vec<Vec<f64>> = (0..ncomp)
        .map(|k| compute_bin_edges(scores, k, n, n_bins))
        .collect();

    let obs_bins: Vec<usize> = (0..ncomp)
        .map(|k| find_bin(scores[(observation, k)], &bin_edges[k], n_bins))
        .collect();

    let beam_width = 3;
    let mut best_conditions: Vec<usize> = Vec::new();
    let mut best_precision = 0.0;
    let mut best_matching = vec![true; n];
    let mut used = vec![false; ncomp];

    for _iter in 0..ncomp {
        let mut candidates = beam_search_candidates(
            scores,
            ncomp,
            &used,
            &obs_bins,
            &bin_edges,
            n_bins,
            &best_conditions,
            &best_matching,
            same_pred,
            beam_width,
        );

        if candidates.is_empty() {
            break;
        }

        let (new_conds, prec, matching) = candidates.remove(0);
        used[*new_conds.last().unwrap()] = true;
        best_conditions = new_conds;
        best_precision = prec;
        best_matching = matching;

        if best_precision >= precision_threshold {
            break;
        }
    }

    let rule = build_anchor_rule(
        &best_conditions,
        &bin_edges,
        &obs_bins,
        best_precision,
        &best_matching,
        n,
    );
    (rule, best_matching)
}

/// Compute quantile bin edges for a column of scores.
fn compute_bin_edges(scores: &FdMatrix, component: usize, n: usize, n_bins: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = (0..n).map(|i| scores[(i, component)]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(f64::NEG_INFINITY);
    for b in 1..n_bins {
        edges.push(quantile_sorted(&vals, b as f64 / n_bins as f64));
    }
    edges.push(f64::INFINITY);
    edges
}

/// Find which bin a value falls into given bin edges.
fn find_bin(value: f64, edges: &[f64], n_bins: usize) -> usize {
    for bi in 0..n_bins {
        if value >= edges[bi] && value < edges[bi + 1] {
            return bi;
        }
    }
    n_bins - 1
}

/// Compute which observations match a bin constraint on a component.
fn apply_bin_filter(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
) -> Vec<bool> {
    let lo = edges[bin];
    let hi = edges[bin + 1];
    let is_last = bin == n_bins - 1;
    (0..current_matching.len())
        .map(|i| {
            current_matching[i]
                && scores[(i, component)] >= lo
                && (is_last || scores[(i, component)] < hi)
        })
        .collect()
}

/// Evaluate a candidate condition: add component to current matching and compute precision.
fn evaluate_anchor_candidate(
    current_matching: &[bool],
    scores: &FdMatrix,
    component: usize,
    bin: usize,
    edges: &[f64],
    n_bins: usize,
    same_pred: &dyn Fn(usize) -> bool,
) -> Option<(f64, Vec<bool>)> {
    let new_matching = apply_bin_filter(current_matching, scores, component, bin, edges, n_bins);
    let n_match = new_matching.iter().filter(|&&v| v).count();
    if n_match == 0 {
        return None;
    }
    let n_same = (0..new_matching.len())
        .filter(|&i| new_matching[i] && same_pred(i))
        .count();
    Some((n_same as f64 / n_match as f64, new_matching))
}

/// Evaluate all unused components in beam search and return sorted candidates.
fn beam_search_candidates(
    scores: &FdMatrix,
    ncomp: usize,
    used: &[bool],
    obs_bins: &[usize],
    bin_edges: &[Vec<f64>],
    n_bins: usize,
    best_conditions: &[usize],
    best_matching: &[bool],
    same_pred: &dyn Fn(usize) -> bool,
    beam_width: usize,
) -> Vec<(Vec<usize>, f64, Vec<bool>)> {
    let mut candidates: Vec<(Vec<usize>, f64, Vec<bool>)> = Vec::new();

    for k in 0..ncomp {
        if used[k] {
            continue;
        }
        if let Some((precision, matching)) = evaluate_anchor_candidate(
            best_matching,
            scores,
            k,
            obs_bins[k],
            &bin_edges[k],
            n_bins,
            same_pred,
        ) {
            let mut conds = best_conditions.to_vec();
            conds.push(k);
            candidates.push((conds, precision, matching));
        }
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.truncate(beam_width);
    candidates
}

/// Build an AnchorRule from selected components, bin edges, and observation bins.
fn build_anchor_rule(
    components: &[usize],
    bin_edges: &[Vec<f64>],
    obs_bins: &[usize],
    precision: f64,
    matching: &[bool],
    n: usize,
) -> super::advanced::AnchorRule {
    use super::advanced::{AnchorCondition, AnchorRule};

    let conditions: Vec<AnchorCondition> = components
        .iter()
        .map(|&k| AnchorCondition {
            component: k,
            lower_bound: bin_edges[k][obs_bins[k]],
            upper_bound: bin_edges[k][obs_bins[k] + 1],
        })
        .collect();
    let n_match = matching.iter().filter(|&&v| v).count();
    AnchorRule {
        conditions,
        precision,
        coverage: n_match as f64 / n as f64,
        n_matching: n_match,
    }
}

// ---------------------------------------------------------------------------
// Sobol helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Depth helpers
// ---------------------------------------------------------------------------

/// Compute depth of scores using the specified depth type.
pub(crate) fn compute_score_depths(
    scores: &FdMatrix,
    depth_type: super::advanced::DepthType,
) -> Vec<f64> {
    match depth_type {
        super::advanced::DepthType::FraimanMuniz => depth::fraiman_muniz_1d(scores, scores, false),
        super::advanced::DepthType::ModifiedBand => depth::modified_band_1d(scores, scores),
        super::advanced::DepthType::FunctionalSpatial => {
            depth::functional_spatial_1d(scores, scores, None)
        }
    }
}

/// Compute beta depth from bootstrap coefficient vectors.
pub(crate) fn beta_depth_from_bootstrap(
    boot_coefs: &[Vec<f64>],
    orig_coefs: &[f64],
    ncomp: usize,
    depth_type: super::advanced::DepthType,
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
    depth_type: super::advanced::DepthType,
) -> f64 {
    let depths = match depth_type {
        super::advanced::DepthType::FraimanMuniz => depth::fraiman_muniz_1d(row, reference, false),
        super::advanced::DepthType::ModifiedBand => depth::modified_band_1d(row, reference),
        super::advanced::DepthType::FunctionalSpatial => {
            depth::functional_spatial_1d(row, reference, None)
        }
    };
    if depths.is_empty() {
        0.0
    } else {
        depths[0]
    }
}

// ---------------------------------------------------------------------------
// Stability helpers
// ---------------------------------------------------------------------------

/// Quantile of a pre-sorted slice using linear interpolation.
pub(crate) fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let lo = lo.min(n - 1);
    let hi = hi.min(n - 1);
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Sample standard deviation of a slice.
fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    // Bessel's correction for sample std
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Compute average ranks of a slice (1-based, average ranks for ties).
fn compute_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        values[a]
            .partial_cmp(&values[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && (values[idx[j]] - values[idx[i]]).abs() < 1e-15 {
            j += 1;
        }
        let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based average
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman rank correlation between two equal-length slices.
fn spearman_rank_correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = compute_ranks(a);
    let rb = compute_ranks(b);
    let mean_a: f64 = ra.iter().sum::<f64>() / n as f64;
    let mean_b: f64 = rb.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut da2 = 0.0;
    let mut db2 = 0.0;
    for i in 0..n {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        num += da * db;
        da2 += da * da;
        db2 += db * db;
    }
    let denom = (da2 * db2).sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        num / denom
    }
}

/// Mean pairwise Spearman rank correlation across a set of vectors.
fn mean_pairwise_spearman(vectors: &[Vec<f64>]) -> f64 {
    let n = vectors.len();
    if n < 2 {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += spearman_rank_correlation(&vectors[i], &vectors[j]);
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// Compute pointwise mean, std, and coefficient of variation from bootstrap samples.
fn pointwise_mean_std_cv(samples: &[Vec<f64>], length: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = samples.len();
    let mut mean = vec![0.0; length];
    let mut std = vec![0.0; length];
    for j in 0..length {
        let vals: Vec<f64> = samples.iter().map(|s| s[j]).collect();
        mean[j] = vals.iter().sum::<f64>() / n as f64;
        let var = vals.iter().map(|&v| (v - mean[j]).powi(2)).sum::<f64>() / (n - 1) as f64;
        std[j] = var.sqrt();
    }
    let eps = 1e-15;
    let cv: Vec<f64> = (0..length)
        .map(|j| {
            if mean[j].abs() > eps {
                std[j] / mean[j].abs()
            } else {
                0.0
            }
        })
        .collect();
    (mean, std, cv)
}

/// Compute per-component std from bootstrap coefficient vectors.
fn coefficient_std_from_bootstrap(all_coefs: &[Vec<f64>], ncomp: usize) -> Vec<f64> {
    (0..ncomp)
        .map(|k| {
            let vals: Vec<f64> = all_coefs.iter().map(|c| c[k]).collect();
            sample_std(&vals)
        })
        .collect()
}

/// Build stability result from collected bootstrap data.
pub(crate) fn build_stability_result(
    all_beta_t: &[Vec<f64>],
    all_coefs: &[Vec<f64>],
    all_abs_coefs: &[Vec<f64>],
    all_metrics: &[f64],
    m: usize,
    ncomp: usize,
) -> Option<super::advanced::StabilityAnalysisResult> {
    use super::advanced::StabilityAnalysisResult;

    let n_success = all_beta_t.len();
    if n_success < 2 {
        return None;
    }
    let (_mean, beta_t_std, beta_t_cv) = pointwise_mean_std_cv(all_beta_t, m);
    let coefficient_std = coefficient_std_from_bootstrap(all_coefs, ncomp);
    let metric_std = sample_std(all_metrics);
    let importance_stability = mean_pairwise_spearman(all_abs_coefs);

    Some(StabilityAnalysisResult {
        beta_t_std,
        coefficient_std,
        metric_std,
        beta_t_cv,
        importance_stability,
        n_boot_success: n_success,
    })
}

// ---------------------------------------------------------------------------
// Logistic helpers shared by PDP, permutation importance, etc.
// ---------------------------------------------------------------------------

/// Compute base logistic eta for one observation, excluding a given component.
pub(super) fn logistic_eta_base(
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    ncomp: usize,
    exclude_component: usize,
) -> f64 {
    let mut eta = fit_intercept;
    for k in 0..ncomp {
        if k != exclude_component {
            eta += coefficients[1 + k] * scores[(i, k)];
        }
    }
    if let Some(sc) = scalar_covariates {
        for j in 0..gamma.len() {
            eta += gamma[j] * sc[(i, j)];
        }
    }
    eta
}

/// Compute logistic accuracy from a score matrix.
pub(super) fn logistic_accuracy_from_scores(
    score_mat: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    y: &[f64],
    n: usize,
    ncomp: usize,
) -> f64 {
    let correct: usize = (0..n)
        .filter(|&i| {
            let mut eta = fit_intercept;
            for c in 0..ncomp {
                eta += coefficients[1 + c] * score_mat[(i, c)];
            }
            let pred = if sigmoid(eta) >= 0.5 { 1.0 } else { 0.0 };
            (pred - y[i]).abs() < 1e-10
        })
        .count();
    correct as f64 / n as f64
}

/// Compute mean logistic prediction with optional component replacements.
pub(super) fn logistic_pdp_mean(
    scores: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    replacements: &[(usize, f64)],
) -> f64 {
    let p_scalar = gamma.len();
    let mut sum = 0.0;
    for i in 0..n {
        let mut eta = fit_intercept;
        for c in 0..ncomp {
            let s = replacements
                .iter()
                .find(|&&(comp, _)| comp == c)
                .map(|&(_, val)| val)
                .unwrap_or(scores[(i, c)]);
            eta += coefficients[1 + c] * s;
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                eta += gamma[j] * sc[(i, j)];
            }
        }
        sum += sigmoid(eta);
    }
    sum / n as f64
}

/// Predict from FPC scores + scalar covariates using linear model coefficients.
pub(super) fn predict_from_scores(
    scores: &FdMatrix,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Vec<f64> {
    let n = scores.nrows();
    let mut preds = vec![0.0; n];
    for i in 0..n {
        let mut yhat = coefficients[0];
        for k in 0..ncomp {
            yhat += coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..gamma.len() {
                yhat += gamma[j] * sc[(i, j)];
            }
        }
        preds[i] = yhat;
    }
    preds
}

/// Validate inputs for conformal prediction. Returns (n_cal, n_proper) on success.
pub(super) fn validate_conformal_inputs(
    n: usize,
    m: usize,
    n_test: usize,
    m_test: usize,
    train_y_len: usize,
    ncomp: usize,
    cal_fraction: f64,
    alpha: f64,
) -> Option<(usize, usize)> {
    let shapes_ok = n >= 4 && n == train_y_len && m > 0 && n_test > 0 && m_test == m;
    let params_ok = cal_fraction > 0.0 && cal_fraction < 1.0 && alpha > 0.0 && alpha < 1.0;
    if !(shapes_ok && params_ok) {
        return None;
    }
    let n_cal = ((n as f64 * cal_fraction).round() as usize).max(2);
    let n_proper = n - n_cal;
    (n_proper >= ncomp + 2).then_some((n_cal, n_proper))
}

/// Compute conformal calibration quantile and coverage from absolute residuals.
pub(super) fn conformal_quantile_and_coverage(
    calibration_scores: &[f64],
    cal_n: usize,
    alpha: f64,
) -> (f64, f64) {
    let q_level = (((cal_n + 1) as f64 * (1.0 - alpha)).ceil() / cal_n as f64).min(1.0);
    let mut sorted_scores = calibration_scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let residual_quantile = quantile_sorted(&sorted_scores, q_level);

    let coverage = calibration_scores
        .iter()
        .filter(|&&s| s <= residual_quantile)
        .count() as f64
        / cal_n as f64;

    (residual_quantile, coverage)
}

/// Weighted calibration gap for a group of sorted indices.
pub(super) fn calibration_gap_weighted(
    indices: &[usize],
    y: &[f64],
    probabilities: &[f64],
    total_n: usize,
) -> f64 {
    let cnt = indices.len();
    if cnt == 0 {
        return 0.0;
    }
    let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();
    let sum_p: f64 = indices.iter().map(|&i| probabilities[i]).sum();
    let gap = (sum_y / cnt as f64 - sum_p / cnt as f64).abs();
    cnt as f64 / total_n as f64 * gap
}
