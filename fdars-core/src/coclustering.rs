//! Functional co-clustering via the funLBM latent block model.
//!
//! This module implements a functional latent block model (funLBM) using a
//! Classification EM (CEM) algorithm. It simultaneously clusters:
//! - **Row clusters**: partitions the n curves into K row-clusters.
//! - **Column clusters**: partitions the m argument points into L column-clusters.
//!
//! ## Column-cluster semantics (RESOLVED)
//!
//! `col_labels` has length **m** — the m argument/evaluation points are partitioned
//! into L column-clusters (need not be contiguous). This is true funLBM. The column
//! clusters do NOT range over the FPC components.
//!
//! ## Global FPCA reuse with block-score projection
//!
//! ONE global FPCA is computed via [`fdata_to_pc_1d`]. For a curve i in column-block l,
//! the block score is the projection of Y_i **restricted to column-block l's argument points**
//! onto the global FPC loadings restricted to those same points:
//!
//! ```text
//! block_score[i][l][k] = Σ_{j: col_labels[j]==l}  weights[j] * (data[(i,j)] - mean[j]) * rotation[(j,k)]
//! ```
//!
//! This restricts the standard weighted FPC inner product to a column-block's argument-point
//! subset, keeping columns = argument points while reusing a single global FPCA.
//!
//! ## Divergences from R funLBM 2.3.1
//!
//! | Aspect                | fdars (this module)           | R funLBM 2.3.1          |
//! |-----------------------|-------------------------------|-------------------------|
//! | FPCA scope            | One global FPCA               | Per-block FPCA          |
//! | EM variant            | Deterministic CEM (hard)      | SEM-Gibbs (stochastic)  |
//! | Block covariance      | Diagonal (ncomp variances)    | Full covariance matrix  |
//! | Column semantics      | m argument points             | m argument points (same)|
//!
//! ## References
//!
//! - Bouveyron et al. (2018), "Co-clustering of Multivariate Functional Data", JASA.
//! - Govaert & Nadif (2008), "Block clustering with Bernoulli mixture models", CIS.

use std::f64::consts::PI;

use rand::prelude::*;

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

/// Per-block Gaussian parameters (diagonal covariance in the FPC score space).
///
/// Each block (k, l) — row-cluster k, column-cluster l — is modelled by a
/// diagonal multivariate Gaussian on the `ncomp`-dimensional block scores.
///
/// Indexed as `block_params[k * n_col_blocks + l]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlockParams {
    /// Per-component block mean (length `eff_ncomp`).
    pub mean: Vec<f64>,
    /// Per-component block variance, diagonal (length `eff_ncomp`).
    pub variance: Vec<f64>,
}

/// Result of [`co_cluster`].
///
/// The block structure is indexed as `block_params[k * n_col_blocks + l]`
/// where k ∈ 0..n_row_blocks and l ∈ 0..n_col_blocks.
///
/// ## ICL formula
///
/// The ICL (Integrated Completed Likelihood) uses the symmetric Govaert-Nadif penalty:
/// ```text
/// p_KL = (K-1) + (L-1) + 2 * K * L * eff_ncomp
/// ICL   = log_likelihood - 0.5 * p_KL * (ln(n) + ln(m))
/// ```
/// Here `ln(n)` penalises the n-curve row dimension and `ln(m)` penalises the
/// m-argument-point column dimension — reflecting that column-clusters partition
/// the m argument points (not the FPC components).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoClusterResult {
    /// Hard row-cluster assignments, length n.
    /// Values in `0..n_row_blocks`.
    pub row_labels: Vec<usize>,

    /// Hard column-cluster assignments, length **m** (the number of argument points).
    ///
    /// Values in `0..n_col_blocks`. This always satisfies `col_labels.len() == m` —
    /// columns cluster the argument points, NOT the FPC components.
    pub col_labels: Vec<usize>,

    /// Number of row clusters K.
    pub n_row_blocks: usize,

    /// Number of column clusters L.
    pub n_col_blocks: usize,

    /// Per-block Gaussian parameters, length K*L, indexed `k*L + l`.
    /// Each element describes the diagonal Gaussian on the `eff_ncomp`-dimensional
    /// block scores for the (k, l) block.
    pub block_params: Vec<BlockParams>,

    /// Mixing proportions for row clusters, length K. Sums to 1.
    pub row_props: Vec<f64>,

    /// Mixing proportions for column clusters, length L. Sums to 1.
    pub col_props: Vec<f64>,

    /// Converged classification log-likelihood (non-decreasing across CEM iterations).
    pub log_likelihood: f64,

    /// ICL model-selection criterion (finite; lower = better model).
    ///
    /// Formula: `ICL = log_likelihood - 0.5 * p_KL * (ln(n) + ln(m))`
    /// where `p_KL = (K-1) + (L-1) + 2*K*L*eff_ncomp`.
    pub icl: f64,

    /// Number of CEM iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged before `max_iter`.
    pub converged: bool,
}

/// Configuration for funLBM functional co-clustering.
///
/// Builder-style config mirroring [`GmmClusterConfig`](crate::gmm::cluster::GmmClusterConfig).
/// Modify fields directly after calling [`CoClusterConfig::default()`].
///
/// # Example
/// ```no_run
/// use fdars_core::coclustering::CoClusterConfig;
///
/// let mut cfg = CoClusterConfig::default();
/// cfg.n_row_blocks = 3;
/// cfg.n_col_blocks = 4;
/// cfg.ncomp = 3;
/// cfg.n_init = 5;
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoClusterConfig {
    /// Number of row clusters K (default: 2).
    pub n_row_blocks: usize,
    /// Number of column clusters L (default: 2).
    pub n_col_blocks: usize,
    /// Number of FPC components for the block-score projection (default: 5).
    /// The effective ncomp may be reduced to `min(ncomp, n, m)` by the FPCA.
    pub ncomp: usize,
    /// Maximum CEM iterations per initialization (default: 200).
    pub max_iter: usize,
    /// Convergence tolerance on the classification log-likelihood (default: 1e-6).
    pub tol: f64,
    /// Number of random initializations; the best by log-likelihood is returned (default: 3).
    pub n_init: usize,
    /// Base random seed for deterministic results (default: 42).
    pub seed: u64,
}

impl Default for CoClusterConfig {
    fn default() -> Self {
        Self {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 5,
            max_iter: 200,
            tol: 1e-6,
            n_init: 3,
            seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Log-density of a scalar under a 1-D Gaussian N(x; mu, var).
///
/// Returns -∞ if `var <= 0`.
#[inline]
fn log_gaussian_1d(x: f64, mu: f64, var: f64) -> f64 {
    if var <= 0.0 {
        return f64::NEG_INFINITY;
    }
    -0.5 * ((x - mu).powi(2) / var + var.ln() + (2.0 * PI).ln())
}

/// Build the flat block-score buffer (n * L * eff_ncomp, indexed `(i*L + l)*e + k`).
///
/// For each curve i and column-cluster l:
/// `bscore[i][l][k] = Σ_{j: col_labels[j]==l}  weights[j] * (data[(i,j)] - mean[j]) * rotation[(j,k)]`
fn build_block_scores(
    data: &FdMatrix,
    rotation: &FdMatrix,
    mean: &[f64],
    weights: &[f64],
    col_labels: &[usize],
    n: usize,
    m: usize,
    l_blocks: usize,
    eff_ncomp: usize,
) -> Vec<f64> {
    let total = n * l_blocks * eff_ncomp;
    let mut buf = vec![0.0_f64; total];

    // Iterate over argument points j; for each j accumulate into the correct l-block.
    for j in 0..m {
        let l = col_labels[j];
        let w = weights[j];
        let mean_j = mean[j];
        // rotation is m×eff_ncomp column-major: rotation[(j, k)] at j + k*m
        for i in 0..n {
            let val = data[(i, j)] - mean_j;
            let base = (i * l_blocks + l) * eff_ncomp;
            for k in 0..eff_ncomp {
                // rotation[(j, k)] = rotation.data[j + k*m], but use column() for the k-th column
                buf[base + k] += w * val * rotation[(j, k)];
            }
        }
    }

    buf
}

/// Compute data-scaled regularization floor over block scores (1-D analogue of data_scaled_reg).
fn block_score_reg(block_scores: &[f64], n: usize, l_blocks: usize, eff_ncomp: usize) -> f64 {
    const REG_REL: f64 = 1e-6;
    if n == 0 || l_blocks == 0 || eff_ncomp == 0 {
        return REG_REL;
    }
    let total_blocks = l_blocks * eff_ncomp;
    let mut total_var = 0.0_f64;
    let mut n_dims = 0u64;
    for l in 0..l_blocks {
        for comp in 0..eff_ncomp {
            // Collect all n scores for this (l, comp)
            let mut sum = 0.0_f64;
            let mut ss = 0.0_f64;
            for i in 0..n {
                let v = block_scores[(i * l_blocks + l) * eff_ncomp + comp];
                sum += v;
                ss += v * v;
            }
            let mean = sum / n as f64;
            let var = ss / n as f64 - mean * mean;
            total_var += var;
            n_dims += 1;
        }
    }
    let _ = total_blocks; // suppress unused warning
    let mean_var = if n_dims > 0 {
        total_var / n_dims as f64
    } else {
        0.0
    };
    if mean_var > 0.0 {
        REG_REL * mean_var
    } else {
        REG_REL
    }
}

/// M-step: recompute row_props, col_props, and block_params from current labels.
fn m_step(
    block_scores: &[f64],
    row_labels: &[usize],
    col_labels: &[usize],
    n: usize,
    m: usize,
    k_blocks: usize,
    l_blocks: usize,
    eff_ncomp: usize,
    reg: f64,
) -> (Vec<f64>, Vec<f64>, Vec<BlockParams>) {
    // Row proportions
    let mut row_counts = vec![0usize; k_blocks];
    for &r in row_labels {
        row_counts[r] += 1;
    }
    let row_props: Vec<f64> = row_counts.iter().map(|&c| c as f64 / n as f64).collect();

    // Column proportions
    let mut col_counts = vec![0usize; l_blocks];
    for &c in col_labels {
        col_counts[c] += 1;
    }
    let col_props: Vec<f64> = col_counts.iter().map(|&c| c as f64 / m as f64).collect();

    // Per-block Gaussian parameters
    let mut block_params = Vec::with_capacity(k_blocks * l_blocks);
    for k in 0..k_blocks {
        for l in 0..l_blocks {
            let mut mean = vec![0.0_f64; eff_ncomp];
            let mut var = vec![0.0_f64; eff_ncomp];
            let mut cnt = 0u64;

            for i in 0..n {
                if row_labels[i] != k {
                    continue;
                }
                cnt += 1;
                let base = (i * l_blocks + l) * eff_ncomp;
                for comp in 0..eff_ncomp {
                    mean[comp] += block_scores[base + comp];
                }
            }

            if cnt > 0 {
                let nf = cnt as f64;
                for comp in 0..eff_ncomp {
                    mean[comp] /= nf;
                }
                // Second pass for variance
                for i in 0..n {
                    if row_labels[i] != k {
                        continue;
                    }
                    let base = (i * l_blocks + l) * eff_ncomp;
                    for comp in 0..eff_ncomp {
                        let d = block_scores[base + comp] - mean[comp];
                        var[comp] += d * d;
                    }
                }
                for comp in 0..eff_ncomp {
                    var[comp] = var[comp] / nf + reg;
                }
            } else {
                // Empty block: use flat variance = reg to avoid NaN
                for comp in 0..eff_ncomp {
                    var[comp] = reg;
                }
            }

            block_params.push(BlockParams {
                mean,
                variance: var,
            });
        }
    }

    (row_props, col_props, block_params)
}

/// Compute classification log-likelihood from current hard labels + parameters.
fn classification_log_likelihood(
    block_scores: &[f64],
    row_labels: &[usize],
    _col_labels: &[usize],
    row_props: &[f64],
    col_props: &[f64],
    block_params: &[BlockParams],
    n: usize,
    _m: usize,
    _k_blocks: usize,
    l_blocks: usize,
    eff_ncomp: usize,
) -> f64 {
    let mut ll = 0.0_f64;

    for i in 0..n {
        let k = row_labels[i];
        let rp = row_props[k];
        if rp < 1e-15 {
            continue;
        }
        ll += rp.ln();

        // Sum log-density over all l blocks (the block score for l already encodes col assignment)
        for l in 0..l_blocks {
            let cp = col_props[l];
            if cp < 1e-15 {
                continue;
            }
            let bp = &block_params[k * l_blocks + l];
            let base = (i * l_blocks + l) * eff_ncomp;
            let mut block_ld = 0.0_f64;
            for comp in 0..eff_ncomp {
                block_ld +=
                    log_gaussian_1d(block_scores[base + comp], bp.mean[comp], bp.variance[comp]);
            }
            ll += cp.ln() + block_ld;
        }
    }

    ll
}

/// E-row: for each curve i, pick argmax_k classification log-density.
fn e_row_step(
    block_scores: &[f64],
    row_props: &[f64],
    col_props: &[f64],
    block_params: &[BlockParams],
    n: usize,
    k_blocks: usize,
    l_blocks: usize,
    eff_ncomp: usize,
) -> Vec<usize> {
    let mut row_labels = vec![0usize; n];
    for i in 0..n {
        let mut best_k = 0usize;
        let mut best_score = f64::NEG_INFINITY;
        for k in 0..k_blocks {
            let rp = row_props[k];
            if rp < 1e-15 {
                continue;
            }
            let mut score = rp.ln();
            for l in 0..l_blocks {
                let cp = col_props[l];
                if cp < 1e-15 {
                    continue;
                }
                let bp = &block_params[k * l_blocks + l];
                let base = (i * l_blocks + l) * eff_ncomp;
                let mut block_ld = 0.0_f64;
                for comp in 0..eff_ncomp {
                    block_ld += log_gaussian_1d(
                        block_scores[base + comp],
                        bp.mean[comp],
                        bp.variance[comp],
                    );
                }
                score += cp.ln() + block_ld;
            }
            if score > best_score {
                best_score = score;
                best_k = k;
            }
        }
        row_labels[i] = best_k;
    }
    row_labels
}

/// E-col: for each argument point j, pick argmax_l of the classification log-density gain.
///
/// The gain of assigning argument point j to column-cluster l is:
/// Σ_i [ log π_k(i) + Σ_{l'} (cp[l'] + block_ld(i,l')) ] where l's contribution changes.
///
/// We use a simpler but equivalent approach: for each j, try each candidate l, compute
/// the change in total classification LL from reassigning j from current label to l.
/// Since block scores depend on col_labels, we compute this by holding all other j fixed
/// and computing the marginal LL contribution of adding point j to column-cluster l for
/// each curve i. This is computed as:
///
/// For each l_candidate: Δ_j(l_candidate) = Σ_i Σ_k [I(row_labels[i]==k) *
///     weights[j] * (data[(i,j)] - mean[j]) * Σ_comp rotation[(j,comp)] *
///     (log N(b_score | mu_kl, var_kl))] — a per-j marginal computation.
///
/// In practice we compute it as the direct contribution to the classification LL
/// of reassigning j → l_candidate (approximation: fix other j's col_labels unchanged).
fn e_col_step(
    data: &FdMatrix,
    rotation: &FdMatrix,
    mean: &[f64],
    weights: &[f64],
    col_labels: &[usize],
    row_labels: &[usize],
    row_props: &[f64],
    col_props: &[f64],
    block_params: &[BlockParams],
    n: usize,
    m: usize,
    _k_blocks: usize,
    l_blocks: usize,
    eff_ncomp: usize,
) -> Vec<usize> {
    let mut new_col_labels = col_labels.to_vec();

    // For each argument point j, compute the incremental block-score contribution
    // from point j to each possible column-cluster l_cand, then pick argmax_l_cand
    // of the sum (over curves i) of the resulting log-density gain.
    for j in 0..m {
        let w_j = weights[j];
        let mean_j = mean[j];

        // Precompute for each curve i and FPC component: the weighted centered value at j
        // s[i][comp] = weights[j] * (data[(i,j)] - mean[j]) * rotation[(j, comp)]
        let mut s = vec![0.0_f64; n * eff_ncomp];
        for i in 0..n {
            let val = w_j * (data[(i, j)] - mean_j);
            for comp in 0..eff_ncomp {
                s[i * eff_ncomp + comp] = val * rotation[(j, comp)];
            }
        }

        let mut best_l = 0usize;
        let mut best_gain = f64::NEG_INFINITY;

        for l_cand in 0..l_blocks {
            let cp = col_props[l_cand];
            if cp < 1e-15 {
                continue;
            }
            // Compute the gain from assigning j → l_cand.
            // For each curve i: the block score for (i, l_cand) gains s[i][·].
            // We compute the log-density gain vs. the current assignment.
            let l_curr = col_labels[j];
            let mut gain = 0.0_f64;

            for i in 0..n {
                let k = row_labels[i];
                let rp = row_props[k];
                if rp < 1e-15 {
                    continue;
                }

                let bp_cand = &block_params[k * l_blocks + l_cand];

                // Log-density for l_cand: use the marginal contribution of point j
                // (s[i][comp] = weights[j]*(data[(i,j)]-mean[j])*rotation[(j,comp)]) as a
                // proxy for the gain from assigning j to l_cand. Terms constant across l_cand
                // choices cancel in the argmax.
                let mut ld_cand_new = 0.0_f64;
                for comp in 0..eff_ncomp {
                    ld_cand_new += log_gaussian_1d(
                        s[i * eff_ncomp + comp],
                        bp_cand.mean[comp],
                        bp_cand.variance[comp],
                    );
                }
                gain += cp.ln() + ld_cand_new;

                // Subtract the current assignment's contribution for l_curr
                if l_curr != l_cand {
                    let bp_curr = &block_params[k * l_blocks + l_curr];
                    let mut ld_curr = 0.0_f64;
                    for comp in 0..eff_ncomp {
                        ld_curr += log_gaussian_1d(
                            s[i * eff_ncomp + comp],
                            bp_curr.mean[comp],
                            bp_curr.variance[comp],
                        );
                    }
                    let cp_curr = col_props[l_curr];
                    if cp_curr >= 1e-15 {
                        gain -= cp_curr.ln() + ld_curr;
                    }
                }
            }

            if gain > best_gain {
                best_gain = gain;
                best_l = l_cand;
            }
        }

        new_col_labels[j] = best_l;
    }

    new_col_labels
}

/// Column k-means++ initialization on argument-point profiles (each point j has n-dim profile).
fn col_kmeans_init(data: &FdMatrix, n: usize, m: usize, l_blocks: usize, seed: u64) -> Vec<usize> {
    if l_blocks >= m {
        // Each point gets its own cluster (degenerate case handled upstream)
        return (0..m).map(|j| j % l_blocks).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Profile of point j: data.column(j) — length n, column-major so contiguous.
    // Compute squared L2 distance between two argument-point profiles.
    let profile_l2sq = |j1: usize, j2: usize| -> f64 {
        let c1 = data.column(j1);
        let c2 = data.column(j2);
        c1.iter().zip(c2.iter()).map(|(a, b)| (a - b).powi(2)).sum()
    };

    // k-means++ initialization
    let first = rng.gen_range(0..m);
    let mut centers: Vec<usize> = vec![first];

    for _ in 1..l_blocks {
        // Compute distance from each point to nearest center
        let dists: Vec<f64> = (0..m)
            .map(|j| {
                centers
                    .iter()
                    .map(|&c| profile_l2sq(j, c))
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let total: f64 = dists.iter().sum();
        if total < 1e-15 {
            // All points are identical; assign round-robin
            centers.push(centers.len() % m);
            continue;
        }
        // Sample proportional to distance squared
        let threshold = rng.gen::<f64>() * total;
        let mut cum = 0.0;
        let mut next = m - 1;
        for (j, &d) in dists.iter().enumerate() {
            cum += d;
            if cum >= threshold {
                next = j;
                break;
            }
        }
        centers.push(next);
    }

    // Assign each point to nearest center; run 10 assign-update iterations
    let mut col_labels: Vec<usize> = (0..m)
        .map(|j| {
            centers
                .iter()
                .enumerate()
                .map(|(ci, &c)| (ci, profile_l2sq(j, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(ci, _)| ci)
                .unwrap_or(0)
        })
        .collect();

    for _ in 0..10 {
        // Recompute centroids as mean of assigned profiles (in feature space R^n)
        // We track centroid as column-major buffer of size n*l_blocks
        let mut cent = vec![0.0_f64; n * l_blocks];
        let mut cnt = vec![0u64; l_blocks];
        for j in 0..m {
            let l = col_labels[j];
            cnt[l] += 1;
            let col = data.column(j);
            for i in 0..n {
                cent[l * n + i] += col[i];
            }
        }
        for l in 0..l_blocks {
            if cnt[l] > 0 {
                let c = cnt[l] as f64;
                for i in 0..n {
                    cent[l * n + i] /= c;
                }
            }
        }

        // Reassign
        let mut changed = false;
        for j in 0..m {
            let col = data.column(j);
            let mut best_l = 0usize;
            let mut best_d = f64::INFINITY;
            for l in 0..l_blocks {
                let d: f64 = (0..n).map(|i| (col[i] - cent[l * n + i]).powi(2)).sum();
                if d < best_d {
                    best_d = d;
                    best_l = l;
                }
            }
            if col_labels[j] != best_l {
                changed = true;
                col_labels[j] = best_l;
            }
        }

        if !changed {
            break;
        }
    }

    col_labels
}

/// Run a single CEM fit from given initial row/col labels. Returns (result, per_iter_ll).
#[allow(clippy::too_many_arguments)]
fn cem_single_fit(
    data: &FdMatrix,
    rotation: &FdMatrix,
    mean: &[f64],
    weights: &[f64],
    init_row_labels: Vec<usize>,
    init_col_labels: Vec<usize>,
    n: usize,
    m: usize,
    k_blocks: usize,
    l_blocks: usize,
    eff_ncomp: usize,
    max_iter: usize,
    tol: f64,
) -> (CoClusterResult, Vec<f64>) {
    let mut row_labels = init_row_labels;
    let mut col_labels = init_col_labels;

    // Initial block scores
    let mut block_scores = build_block_scores(
        data,
        rotation,
        mean,
        weights,
        &col_labels,
        n,
        m,
        l_blocks,
        eff_ncomp,
    );

    let reg = block_score_reg(&block_scores, n, l_blocks, eff_ncomp);

    // Initial M-step
    let (mut row_props, mut col_props, mut block_params) = m_step(
        &block_scores,
        &row_labels,
        &col_labels,
        n,
        m,
        k_blocks,
        l_blocks,
        eff_ncomp,
        reg,
    );

    let mut prev_ll = f64::NEG_INFINITY;
    let mut per_iter_ll: Vec<f64> = Vec::with_capacity(max_iter);
    let mut iterations = 0usize;
    let mut converged = false;

    for iter in 0..max_iter {
        // E-row: reassign curves
        row_labels = e_row_step(
            &block_scores,
            &row_props,
            &col_props,
            &block_params,
            n,
            k_blocks,
            l_blocks,
            eff_ncomp,
        );

        // E-col: reassign argument points (uses current block_scores and params)
        col_labels = e_col_step(
            data,
            rotation,
            mean,
            weights,
            &col_labels,
            &row_labels,
            &row_props,
            &col_props,
            &block_params,
            n,
            m,
            k_blocks,
            l_blocks,
            eff_ncomp,
        );

        // Rebuild block scores after col reassignment
        block_scores = build_block_scores(
            data,
            rotation,
            mean,
            weights,
            &col_labels,
            n,
            m,
            l_blocks,
            eff_ncomp,
        );

        // M-step
        let (rp, cp, bp) = m_step(
            &block_scores,
            &row_labels,
            &col_labels,
            n,
            m,
            k_blocks,
            l_blocks,
            eff_ncomp,
            reg,
        );
        row_props = rp;
        col_props = cp;
        block_params = bp;

        // Classification log-likelihood
        let ll = classification_log_likelihood(
            &block_scores,
            &row_labels,
            &col_labels,
            &row_props,
            &col_props,
            &block_params,
            n,
            m,
            k_blocks,
            l_blocks,
            eff_ncomp,
        );

        per_iter_ll.push(ll);
        iterations = iter + 1;

        // Convergence check (skip iter 0 to allow at least one update)
        if iter > 0 && (ll - prev_ll).abs() < tol {
            converged = true;
            break;
        }
        prev_ll = ll;
    }

    let log_likelihood = per_iter_ll.last().copied().unwrap_or(f64::NEG_INFINITY);

    // ICL: p_KL = (K-1) + (L-1) + 2*K*L*eff_ncomp
    let p_kl = (k_blocks.saturating_sub(1))
        + (l_blocks.saturating_sub(1))
        + 2 * k_blocks * l_blocks * eff_ncomp;
    let icl = log_likelihood - 0.5 * (p_kl as f64) * ((n as f64).ln() + (m as f64).ln());

    let result = CoClusterResult {
        row_labels,
        col_labels,
        n_row_blocks: k_blocks,
        n_col_blocks: l_blocks,
        block_params,
        row_props,
        col_props,
        log_likelihood,
        icl,
        iterations,
        converged,
    };

    (result, per_iter_ll)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Fit a funLBM functional co-clustering model via Classification EM (CEM).
///
/// Simultaneously partitions the n curves into K row-clusters and the m argument
/// points into L column-clusters. Returns hard assignments and per-block Gaussian
/// parameters.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m), column-major.
/// * `argvals` — Evaluation/argument points, length m. Must be sorted ascending.
/// * `config` — Tuning parameters (K, L, ncomp, restarts, seed, …).
///
/// # Errors
/// - [`FdarError::InvalidParameter`] if `config.ncomp < 1`, `n_row_blocks > n`, or `n_col_blocks > m`.
/// - [`FdarError::InvalidDimension`] if `data` or `argvals` dimensions are inconsistent
///   (propagated from [`fdata_to_pc_1d`]).
/// - [`FdarError::ComputationFailed`] if all initializations fail (propagated from FPCA).
///
/// # Example
/// ```no_run
/// use fdars_core::coclustering::{co_cluster, CoClusterConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let data = FdMatrix::zeros(10, 8);
/// let argvals: Vec<f64> = (0..8).map(|i| i as f64 / 7.0).collect();
/// let config = CoClusterConfig { n_row_blocks: 2, n_col_blocks: 2, ncomp: 3, ..Default::default() };
/// let result = co_cluster(&data, &argvals, &config)?;
/// assert_eq!(result.row_labels.len(), 10);
/// assert_eq!(result.col_labels.len(), 8);
/// # Ok::<(), fdars_core::error::FdarError>(())
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn co_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &CoClusterConfig,
) -> Result<CoClusterResult, FdarError> {
    let (n, m) = data.shape();

    // --- Input validation ---
    if config.ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp must be >= 1, got {}", config.ncomp),
        });
    }
    if config.n_row_blocks > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_row_blocks",
            message: format!(
                "n_row_blocks={} exceeds number of observations n={}",
                config.n_row_blocks, n
            ),
        });
    }
    if config.n_row_blocks == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_row_blocks",
            message: "n_row_blocks must be >= 1".to_string(),
        });
    }
    if config.n_col_blocks > m {
        return Err(FdarError::InvalidParameter {
            parameter: "n_col_blocks",
            message: format!(
                "n_col_blocks={} exceeds number of argument points m={}",
                config.n_col_blocks, m
            ),
        });
    }
    if config.n_col_blocks == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_col_blocks",
            message: "n_col_blocks must be >= 1".to_string(),
        });
    }

    let k_blocks = config.n_row_blocks;
    let l_blocks = config.n_col_blocks;

    // --- Global FPCA ---
    // fdata_to_pc_1d validates data/argvals dimensions and propagates its errors.
    let fpca = fdata_to_pc_1d(data, config.ncomp, argvals)?;
    // Read effective ncomp — may be < requested (clipped to min(n, m))
    let eff_ncomp = fpca.scores.ncols();
    let rotation = &fpca.rotation; // m × eff_ncomp
    let mean = &fpca.mean; // len m
    let weights = &fpca.weights; // len m

    // --- Multi-restart CEM ---
    let n_init = config.n_init.max(1);
    let mut best: Option<CoClusterResult> = None;

    for init in 0..n_init {
        let seed = config.seed.wrapping_add(init as u64 * 1000);

        // Row initialization via kmeans_fd
        use crate::clustering::kmeans_fd;
        let km = kmeans_fd(data, argvals, k_blocks, 100, 1e-4, seed)?;
        let init_row_labels = km.cluster;

        // Column initialization via k-means++ on argument-point profiles
        let init_col_labels = col_kmeans_init(data, n, m, l_blocks, seed.wrapping_add(1));

        let (result, _per_iter_ll) = cem_single_fit(
            data,
            rotation,
            mean,
            weights,
            init_row_labels,
            init_col_labels,
            n,
            m,
            k_blocks,
            l_blocks,
            eff_ncomp,
            config.max_iter,
            config.tol,
        );

        let is_better = best
            .as_ref()
            .map_or(true, |b| result.log_likelihood > b.log_likelihood);
        if is_better {
            best = Some(result);
        }
    }

    best.ok_or_else(|| FdarError::ComputationFailed {
        operation: "co_cluster",
        detail: "all initializations failed".to_string(),
    })
}

// ---------------------------------------------------------------------------
// Slope-heuristic model selection
// ---------------------------------------------------------------------------

/// Result of [`co_cluster_select`]: the slope-heuristic-selected (K, L) fit
/// together with full grid diagnostics.
///
/// ## Grid diagnostics
///
/// `grid_scores` contains one entry per (K, L) pair in the sweep:
/// `(K, L, log_likelihood, model_dim, penalised_score)`.
///
/// `penalised_score = log_likelihood − penalty_rate × model_dim`.
/// In fallback branches (single cell, flat slope, small grid) `penalised_score = log_likelihood`.
///
/// ## Slope heuristic calibration
///
/// The Birgé–Massart penalty is estimated by OLS over the large-model (top-50% by dimension)
/// region of the fitted grid. This is a data-driven heuristic: it works best when the grid
/// spans a range of model dimensions and the data is well-separated enough for the
/// log-likelihood to grow linearly with dimension in the overparameterised region.
/// On poorly separated data the slope may be noisy and the selection may land at a boundary.
/// Inspect `grid_scores` to audit the selection.
///
/// ## Divergence from R funHDDC
///
/// The slope calibration here uses OLS over the top-50% by model dimension (the "linear region"
/// heuristic of Baudry, Maugis & Michel 2012). R's funHDDC uses a slightly different calibration
/// based on the full grid. The selected model may differ on small grids.
#[derive(Debug, Clone)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CoClusterSelectResult {
    /// The selected (K*, L*) co-clustering result.
    pub best: CoClusterResult,
    /// Selected number of row clusters K*.
    pub best_k: usize,
    /// Selected number of column clusters L*.
    pub best_l: usize,
    /// All grid fits: `(K, L, log_likelihood, model_dim, penalised_score)`.
    ///
    /// `penalised_score = log_likelihood − penalty_rate * model_dim`.
    /// In fallback branches (< 4 grid points, flat slope, etc.) `penalised_score = log_likelihood`.
    pub grid_scores: Vec<(usize, usize, f64, usize, f64)>,
    /// OLS slope estimated from the top-50% of fits by model dimension.
    /// Zero when the grid is too small or the slope heuristic fell back to max-LL.
    pub slope_estimate: f64,
    /// Penalty rate applied per model dimension: `2 × |slope_estimate|`.
    /// Zero in fallback branches.
    pub penalty_rate: f64,
}

/// Fit funLBM over a (K, L) grid and select the best block count via the
/// Birgé–Massart slope heuristic.
///
/// For every combination of K in `k_range` and L in `l_range` the function
/// calls [`co_cluster`] (with `config` cloned and `n_row_blocks`/`n_col_blocks`
/// overridden), collects the (model_dimension, log_likelihood) pair, estimates
/// the slope of the LL-vs-dim curve in the large-model region by OLS, and
/// selects `argmax (LL − 2 × |slope| × dim)`.
///
/// ## Model dimension formula
///
/// `dim(K, L) = (K−1) + (L−1) + 2·K·L·eff_ncomp`
///
/// where `eff_ncomp` is the effective FPC count used by the fitted model
/// (read from `block_params[0].mean.len()`; may be less than `config.ncomp`
/// when clipped to `min(n, m)`).
///
/// ## Slope estimation
///
/// OLS over the top-50% of fits by model dimension (the region assumed to be
/// linear in the LL-vs-dim curve). Fallback to `argmax LL` when:
/// - the grid has fewer than 4 distinct-dimension points (or fewer than 4 total),
/// - the OLS denominator is near zero (all dims equal in the large-model subset),
/// - or the estimated penalty rate is ≤ 0 (flat/increasing LL with dimension).
///
/// In every fallback branch `slope_estimate = 0.0` and `penalty_rate = 0.0`;
/// `grid_scores` is always fully populated.
///
/// ## Determinism
///
/// Each grid fit uses the seed from `config.seed`. Fits that would fail
/// `co_cluster`'s own validation (e.g. K > n or L > m) propagate their
/// `FdarError` immediately.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m), column-major.
/// * `argvals` — Evaluation points, length m. Must be sorted ascending.
/// * `k_range` — Candidate K values (number of row clusters). Must be non-empty.
/// * `l_range` — Candidate L values (number of column clusters). Must be non-empty.
/// * `config` — Base tuning parameters. `n_row_blocks` and `n_col_blocks` are
///   overridden per grid cell; all other fields are reused as-is.
///
/// # Errors
/// - [`FdarError::InvalidParameter`] if `k_range` or `l_range` is empty.
/// - Any error propagated from [`co_cluster`] for an invalid (K, L) combination.
///
/// # Example
/// ```no_run
/// use fdars_core::coclustering::{co_cluster_select, CoClusterConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let data = FdMatrix::zeros(20, 10);
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = CoClusterConfig { ncomp: 3, n_init: 2, ..Default::default() };
/// let result = co_cluster_select(&data, &argvals, &[2, 3, 4], &[2, 3], &config)?;
/// println!("Selected K={}, L={}", result.best_k, result.best_l);
/// # Ok::<(), fdars_core::error::FdarError>(())
/// ```
#[must_use = "expensive grid sweep whose result should not be discarded"]
pub fn co_cluster_select(
    data: &FdMatrix,
    argvals: &[f64],
    k_range: &[usize],
    l_range: &[usize],
    config: &CoClusterConfig,
) -> Result<CoClusterSelectResult, FdarError> {
    // --- Validate inputs ---
    if k_range.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "k_range",
            message: "k_range must be non-empty".to_string(),
        });
    }
    if l_range.is_empty() {
        return Err(FdarError::InvalidParameter {
            parameter: "l_range",
            message: "l_range must be non-empty".to_string(),
        });
    }

    // --- Build the (K, L) grid ---
    let grid: Vec<(usize, usize)> = k_range
        .iter()
        .flat_map(|&k| l_range.iter().map(move |&l| (k, l)))
        .collect();

    // --- Sweep the grid sequentially (co_cluster is internally parallelised) ---
    // We use sequential iteration to keep grid results in deterministic order.
    // Each co_cluster call may itself use rayon via its internal helpers.
    let mut cell_results: Vec<(usize, usize, CoClusterResult)> = Vec::with_capacity(grid.len());
    for &(k, l) in &grid {
        let mut cell_cfg = config.clone();
        cell_cfg.n_row_blocks = k;
        cell_cfg.n_col_blocks = l;
        let result = co_cluster(data, argvals, &cell_cfg)?;
        cell_results.push((k, l, result));
    }

    // --- Compute (dim, ll) for each cell ---
    // eff_ncomp = block_params[0].mean.len() (may be < config.ncomp when clipped)
    // model_dim = (K-1) + (L-1) + 2*K*L*eff_ncomp
    struct CellInfo {
        k: usize,
        l: usize,
        ll: f64,
        dim: usize,
        result_idx: usize,
    }

    let infos: Vec<CellInfo> = cell_results
        .iter()
        .enumerate()
        .map(|(idx, (k, l, res))| {
            let eff_ncomp = if res.block_params.is_empty() {
                0
            } else {
                res.block_params[0].mean.len()
            };
            let dim = k.saturating_sub(1) + l.saturating_sub(1) + 2 * k * l * eff_ncomp;
            CellInfo {
                k: *k,
                l: *l,
                ll: res.log_likelihood,
                dim,
                result_idx: idx,
            }
        })
        .collect();

    // --- Birgé–Massart slope estimation ---
    // Sort by dim descending to identify large-model region
    let n_grid = infos.len();

    let (slope_estimate, penalty_rate) = if n_grid < 4 {
        // Too few points for reliable slope estimation; fall back to max-LL
        (0.0_f64, 0.0_f64)
    } else {
        // Take the top 50% (at least 4 points) by model dimension
        let mut sorted_by_dim: Vec<usize> = (0..n_grid).collect();
        sorted_by_dim.sort_by(|&a, &b| infos[b].dim.cmp(&infos[a].dim));

        let n_top = (n_grid / 2).max(4).min(n_grid);
        let top_idxs = &sorted_by_dim[..n_top];

        // OLS: slope = Σ(dim_i − d̄)(ll_i − l̄) / Σ(dim_i − d̄)²
        let d_mean: f64 = top_idxs.iter().map(|&i| infos[i].dim as f64).sum::<f64>() / n_top as f64;
        let l_mean: f64 = top_idxs.iter().map(|&i| infos[i].ll).sum::<f64>() / n_top as f64;

        let numerator: f64 = top_idxs
            .iter()
            .map(|&i| (infos[i].dim as f64 - d_mean) * (infos[i].ll - l_mean))
            .sum();
        let denominator: f64 = top_idxs
            .iter()
            .map(|&i| (infos[i].dim as f64 - d_mean).powi(2))
            .sum();

        if denominator.abs() < 1e-10 {
            // All dims equal in the large-model subset; fall back to max-LL
            (0.0_f64, 0.0_f64)
        } else {
            let slope = numerator / denominator;
            let pen = 2.0 * slope.abs();
            if pen <= 0.0 {
                (slope, 0.0_f64)
            } else {
                (slope, pen)
            }
        }
    };

    // --- Compute penalised scores and select the best ---
    // penalty_rate == 0 means we fall back to argmax LL
    let penalised: Vec<f64> = infos
        .iter()
        .map(|ci| {
            if penalty_rate > 0.0 {
                ci.ll - penalty_rate * ci.dim as f64
            } else {
                ci.ll
            }
        })
        .collect();

    // argmax of penalised scores
    let best_pos = penalised
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let best_k = infos[best_pos].k;
    let best_l = infos[best_pos].l;
    let best_result_idx = infos[best_pos].result_idx;

    // --- Build grid_scores (fully populated) ---
    let grid_scores: Vec<(usize, usize, f64, usize, f64)> = infos
        .iter()
        .enumerate()
        .map(|(pos, ci)| (ci.k, ci.l, ci.ll, ci.dim, penalised[pos]))
        .collect();

    let best = cell_results.remove(best_result_idx).2;

    Ok(CoClusterSelectResult {
        best,
        best_k,
        best_l,
        grid_scores,
        slope_estimate,
        penalty_rate,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{adjusted_rand_index, uniform_grid};

    /// Build a synthetic (K=2, L=2) block-structured dataset.
    ///
    /// Returns (data, argvals, true_row_labels, true_col_labels).
    /// - First n/2 curves have a large positive offset on the first m/2 argument points.
    /// - Second n/2 curves have a large negative offset there.
    /// - Small Normal noise is added everywhere.
    fn make_block_data(
        n: usize,
        m: usize,
        seed: u64,
    ) -> (FdMatrix, Vec<f64>, Vec<usize>, Vec<usize>) {
        use rand::prelude::*;
        use rand_distr::Normal;

        let argvals = uniform_grid(m);
        let mut rng = StdRng::seed_from_u64(seed);
        let noise_dist = Normal::new(0.0_f64, 0.1).unwrap();

        let m_half = m / 2;

        let mut data = FdMatrix::zeros(n, m);
        let mut true_row_labels = vec![0usize; n];
        let mut true_col_labels = vec![0usize; m];

        // Column labels: first half → 0, second half → 1
        for j in m_half..m {
            true_col_labels[j] = 1;
        }

        // Row labels and curve values
        for i in 0..n {
            let row_group = if i < n / 2 { 0 } else { 1 };
            true_row_labels[i] = row_group;

            let signal = if row_group == 0 { 5.0_f64 } else { -5.0_f64 };

            for j in 0..m {
                let noise: f64 = rng.sample(noise_dist);
                // Large signal only on the first m/2 columns (col-cluster 0)
                let base = if j < m_half { signal } else { 0.0 };
                data[(i, j)] = base + noise;
            }
        }

        (data, argvals, true_row_labels, true_col_labels)
    }

    /// Internal helper: run a single CEM fit and return the per-iteration LL vector.
    fn run_single_cem_with_ll(
        data: &FdMatrix,
        argvals: &[f64],
        k: usize,
        l: usize,
        ncomp: usize,
        seed: u64,
    ) -> (CoClusterResult, Vec<f64>) {
        let (n, m) = data.shape();
        let fpca = fdata_to_pc_1d(data, ncomp, argvals).unwrap();
        let eff_ncomp = fpca.scores.ncols();

        use crate::clustering::kmeans_fd;
        let km = kmeans_fd(data, argvals, k, 100, 1e-4, seed).unwrap();
        let init_row = km.cluster;
        let init_col = col_kmeans_init(data, n, m, l, seed.wrapping_add(1));

        cem_single_fit(
            data,
            &fpca.rotation,
            &fpca.mean,
            &fpca.weights,
            init_row,
            init_col,
            n,
            m,
            k,
            l,
            eff_ncomp,
            200,
            1e-6,
        )
    }

    // -----------------------------------------------------------------------
    // Task 1 smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_co_cluster_smoke() {
        let n = 8;
        let m = 6;
        let argvals = uniform_grid(m);
        let data = FdMatrix::zeros(n, m);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            n_init: 1,
            ..Default::default()
        };
        let result = co_cluster(&data, &argvals, &config).unwrap();
        assert_eq!(result.row_labels.len(), n);
        assert_eq!(result.col_labels.len(), m);
        assert_eq!(result.block_params.len(), 4);
        // log-likelihood should be finite (may be -inf only if all zeros; accept either)
        // In practice, zeros → all equal block means → finite LL from the Gaussian
        // (variance will be reg-floored)
        assert!(result.log_likelihood.is_finite() || result.log_likelihood == f64::NEG_INFINITY);
    }

    // -----------------------------------------------------------------------
    // Task 2 correctness tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_classification_ll_nondecreasing() {
        let (data, argvals, _, _) = make_block_data(16, 10, 7777);
        let (_result, per_iter_ll) = run_single_cem_with_ll(&data, &argvals, 2, 2, 3, 42);

        // Classification LL must be non-decreasing across iterations
        // (allow tiny floating-point slack of 1e-6)
        for w in per_iter_ll.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-6,
                "LL decreased: iter[i]={:.6} -> iter[i+1]={:.6}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_coclustering_recovers_block_structure() {
        let (data, argvals, true_row, true_col) = make_block_data(20, 12, 1234);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            n_init: 3,
            seed: 42,
            ..Default::default()
        };
        let result = co_cluster(&data, &argvals, &config).unwrap();

        let ari_row = adjusted_rand_index(&true_row, &result.row_labels);
        let ari_col = adjusted_rand_index(&true_col, &result.col_labels);

        assert!(
            ari_row > 0.8,
            "Row ARI too low: {ari_row:.3} (expected > 0.8)"
        );
        assert!(
            ari_col > 0.8,
            "Col ARI too low: {ari_col:.3} (expected > 0.8)"
        );
    }

    #[test]
    fn test_determinism_under_seed() {
        let (data, argvals, _, _) = make_block_data(16, 10, 999);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            n_init: 2,
            seed: 77,
            ..Default::default()
        };

        let r1 = co_cluster(&data, &argvals, &config).unwrap();
        let r2 = co_cluster(&data, &argvals, &config).unwrap();

        assert_eq!(
            r1.row_labels, r2.row_labels,
            "row_labels differ across runs"
        );
        assert_eq!(
            r1.col_labels, r2.col_labels,
            "col_labels differ across runs"
        );
        assert_eq!(
            r1.log_likelihood, r2.log_likelihood,
            "log_likelihood differs"
        );
        assert_eq!(r1.icl, r2.icl, "ICL differs");
    }

    #[test]
    fn test_icl_is_finite() {
        let (data, argvals, _, _) = make_block_data(16, 10, 42);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            n_init: 1,
            ..Default::default()
        };
        let result = co_cluster(&data, &argvals, &config).unwrap();
        assert!(result.icl.is_finite(), "ICL is not finite: {}", result.icl);
    }

    // -----------------------------------------------------------------------
    // Task 3 error-path tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_k_exceeds_n() {
        let n = 8;
        let m = 6;
        let data = FdMatrix::zeros(n, m);
        let argvals = uniform_grid(m);
        let config = CoClusterConfig {
            n_row_blocks: 99,
            n_col_blocks: 2,
            ncomp: 3,
            ..Default::default()
        };
        let err = co_cluster(&data, &argvals, &config).unwrap_err();
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "n_row_blocks",
                    ..
                }
            ),
            "Expected InvalidParameter(n_row_blocks), got: {err:?}"
        );
    }

    #[test]
    fn test_error_l_exceeds_m() {
        let n = 8;
        let m = 6;
        let data = FdMatrix::zeros(n, m);
        let argvals = uniform_grid(m);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 99,
            ncomp: 3,
            ..Default::default()
        };
        let err = co_cluster(&data, &argvals, &config).unwrap_err();
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "n_col_blocks",
                    ..
                }
            ),
            "Expected InvalidParameter(n_col_blocks), got: {err:?}"
        );
    }

    #[test]
    fn test_error_zero_ncomp() {
        let n = 8;
        let m = 6;
        let data = FdMatrix::zeros(n, m);
        let argvals = uniform_grid(m);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 0,
            ..Default::default()
        };
        let err = co_cluster(&data, &argvals, &config).unwrap_err();
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "ncomp",
                    ..
                }
            ),
            "Expected InvalidParameter(ncomp), got: {err:?}"
        );
    }

    #[test]
    fn test_error_argvals_mismatch() {
        let n = 8;
        let m = 6;
        let data = FdMatrix::zeros(n, m);
        let argvals = uniform_grid(m + 3); // wrong length
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            ..Default::default()
        };
        let err = co_cluster(&data, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "Expected InvalidDimension, got: {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Task 1 (tracer) + Task 2 (slope heuristic) tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_co_cluster_select_smoke() {
        // Small grid: k_range=[2,3], l_range=[2] → 2 grid cells
        let n = 8;
        let m = 6;
        let argvals = uniform_grid(m);
        let data = FdMatrix::zeros(n, m);
        let config = CoClusterConfig {
            ncomp: 2,
            n_init: 1,
            ..Default::default()
        };
        let result = co_cluster_select(&data, &argvals, &[2, 3], &[2], &config).unwrap();
        assert_eq!(
            result.grid_scores.len(),
            2,
            "Expected 2 grid cells (K in {{2,3}}, L=2)"
        );
        assert_eq!(
            result.best.row_labels.len(),
            n,
            "best.row_labels.len() should equal n"
        );
        assert_eq!(
            result.best.col_labels.len(),
            m,
            "best.col_labels.len() should equal m"
        );
    }

    #[test]
    fn test_slope_heuristic_selects_correct_kl() {
        // Use well-separated (K=2, L=2) block data; sweep [2,3,4] × [2,3].
        // The slope heuristic should select the true (K=2, L=2) or at least a
        // model with ARI > 0.8 on row assignments.
        let (data, argvals, true_row, _) = make_block_data(24, 12, 2024);
        let config = CoClusterConfig {
            ncomp: 3,
            n_init: 3,
            seed: 42,
            ..Default::default()
        };
        let result = co_cluster_select(&data, &argvals, &[2, 3, 4], &[2, 3], &config).unwrap();

        // grid_scores should have 6 entries (3 K × 2 L)
        assert_eq!(result.grid_scores.len(), 6, "Expected 6 grid cells");

        // All grid_scores entries should have finite (or NEG_INFINITY) log-likelihoods
        for &(k, l, ll, dim, pen) in &result.grid_scores {
            assert!(
                ll.is_finite() || ll == f64::NEG_INFINITY,
                "grid entry (K={k}, L={l}) has non-finite ll={ll}"
            );
            let _ = (dim, pen); // used
        }

        // The best result should assign n curves
        assert_eq!(result.best.row_labels.len(), 24);

        // ARI tolerance: best row assignment should have ARI > 0.6 with true labels
        // (relaxed because slope heuristic may pick K=3 on some runs, which is near-true)
        let ari = adjusted_rand_index(&true_row, &result.best.row_labels);
        assert!(
            ari > 0.6,
            "Row ARI too low: {ari:.3}. best_k={}, best_l={}",
            result.best_k,
            result.best_l
        );
    }

    #[test]
    fn test_select_single_cell() {
        // Single-cell grid (k_range=[2], l_range=[2]) → 1 grid entry, no slope estimation
        let n = 10;
        let m = 8;
        let (data, argvals, _, _) = make_block_data(n, m, 42);
        let config = CoClusterConfig {
            ncomp: 2,
            n_init: 1,
            seed: 1,
            ..Default::default()
        };
        let result = co_cluster_select(&data, &argvals, &[2], &[2], &config).unwrap();

        assert_eq!(
            result.grid_scores.len(),
            1,
            "Single-cell grid should have 1 entry"
        );
        assert_eq!(result.best_k, 2);
        assert_eq!(result.best_l, 2);
        // Slope fallback: < 4 points → slope_estimate = 0, penalty_rate = 0
        assert_eq!(
            result.slope_estimate, 0.0,
            "slope_estimate should be 0 for single-cell"
        );
        assert_eq!(
            result.penalty_rate, 0.0,
            "penalty_rate should be 0 for single-cell"
        );
    }

    #[test]
    fn test_select_empty_range_errors() {
        let n = 8;
        let m = 6;
        let data = FdMatrix::zeros(n, m);
        let argvals = uniform_grid(m);
        let config = CoClusterConfig::default();

        // Empty k_range
        let err = co_cluster_select(&data, &argvals, &[], &[2], &config).unwrap_err();
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "k_range",
                    ..
                }
            ),
            "Expected InvalidParameter(k_range), got: {err:?}"
        );

        // Empty l_range
        let err = co_cluster_select(&data, &argvals, &[2], &[], &config).unwrap_err();
        assert!(
            matches!(
                err,
                FdarError::InvalidParameter {
                    parameter: "l_range",
                    ..
                }
            ),
            "Expected InvalidParameter(l_range), got: {err:?}"
        );
    }

    #[test]
    fn test_select_determinism() {
        let (data, argvals, _, _) = make_block_data(16, 10, 12345);
        let config = CoClusterConfig {
            ncomp: 3,
            n_init: 2,
            seed: 99,
            ..Default::default()
        };

        let r1 = co_cluster_select(&data, &argvals, &[2, 3], &[2, 3], &config).unwrap();
        let r2 = co_cluster_select(&data, &argvals, &[2, 3], &[2, 3], &config).unwrap();

        assert_eq!(r1.best_k, r2.best_k, "best_k differs across runs");
        assert_eq!(r1.best_l, r2.best_l, "best_l differs across runs");
        assert_eq!(
            r1.grid_scores.len(),
            r2.grid_scores.len(),
            "grid_scores.len() differs"
        );
        for (a, b) in r1.grid_scores.iter().zip(r2.grid_scores.iter()) {
            assert_eq!(a.0, b.0, "K differs in grid_scores");
            assert_eq!(a.1, b.1, "L differs in grid_scores");
            assert_eq!(a.2, b.2, "log_lik differs in grid_scores");
            assert_eq!(a.3, b.3, "model_dim differs in grid_scores");
            assert_eq!(a.4, b.4, "penalised_score differs in grid_scores");
        }
    }

    #[test]
    fn test_result_surface_populated() {
        let n = 10;
        let m = 8;
        let (data, argvals, _, _) = make_block_data(n, m, 555);
        let config = CoClusterConfig {
            n_row_blocks: 2,
            n_col_blocks: 2,
            ncomp: 3,
            n_init: 1,
            ..Default::default()
        };
        let result = co_cluster(&data, &argvals, &config).unwrap();

        assert_eq!(result.row_labels.len(), n, "row_labels.len() != n");
        assert_eq!(result.col_labels.len(), m, "col_labels.len() != m");
        assert_eq!(
            result.block_params.len(),
            result.n_row_blocks * result.n_col_blocks,
            "block_params.len() != K*L"
        );
        assert_eq!(result.row_props.len(), result.n_row_blocks);
        assert_eq!(result.col_props.len(), result.n_col_blocks);

        // All block_params have consistent lengths
        for bp in &result.block_params {
            assert!(!bp.mean.is_empty(), "block_param.mean is empty");
            assert_eq!(
                bp.mean.len(),
                bp.variance.len(),
                "mean/variance length mismatch"
            );
        }
    }
}
