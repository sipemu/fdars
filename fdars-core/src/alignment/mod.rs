//! Elastic alignment and SRSF (Square-Root Slope Function) transforms.
//!
//! This module provides phase-amplitude separation for functional data via
//! the elastic framework. Key capabilities:
//!
//! - [`srsf_transform`] / [`srsf_inverse`] — SRSF representation and reconstruction
//! - [`elastic_align_pair`] — Pairwise curve alignment via dynamic programming
//! - [`elastic_distance`] — Elastic (Fisher-Rao) distance between curves
//! - [`align_to_target`] — Align a set of curves to a common target
//! - [`karcher_mean`] — Karcher (Fréchet) mean in the elastic metric
//! - [`elastic_self_distance_matrix`] / [`elastic_cross_distance_matrix`] — Distance matrices
//! - [`reparameterize_curve`] / [`compose_warps`] — Warping utilities

mod clustering;
mod constrained;
mod diagnostics;
mod karcher;
mod lambda_cv;
mod nd;
mod pairwise;
mod phase_boxplot;
mod quality;
mod set;
mod shape;
mod srsf;
mod tsrvf;
mod warp_stats;

#[cfg(test)]
mod tests;

// Re-export all public items so that `crate::alignment::X` continues to work.
pub use clustering::{
    cut_dendrogram, elastic_hierarchical, elastic_kmeans, ElasticClusterConfig,
    ElasticClusterMethod, ElasticClusterResult, ElasticDendrogram,
};
pub use constrained::{
    elastic_align_pair_constrained, elastic_align_pair_with_landmarks, ConstrainedAlignmentResult,
};
pub use diagnostics::{
    diagnose_alignment, diagnose_pairwise, AlignmentDiagnostic, AlignmentDiagnosticSummary,
    DiagnosticConfig,
};
pub use karcher::karcher_mean;
pub use lambda_cv::{lambda_cv, LambdaCvConfig, LambdaCvResult};
pub use nd::{
    elastic_align_pair_nd, elastic_distance_nd, srsf_inverse_nd, srsf_transform_nd,
    AlignmentResultNd,
};
pub use pairwise::{
    amplitude_distance, amplitude_self_distance_matrix, elastic_align_pair,
    elastic_align_pair_penalized, elastic_cross_distance_matrix, elastic_distance,
    elastic_self_distance_matrix, phase_distance_pair, phase_self_distance_matrix, WarpPenaltyType,
};
pub use phase_boxplot::{phase_boxplot, PhaseBoxplot};
pub use quality::{
    alignment_quality, pairwise_consistency, warp_complexity, warp_smoothness, AlignmentQuality,
};
pub use set::{align_to_target, elastic_decomposition, DecompositionResult};
pub use shape::{
    orbit_representative, shape_distance, shape_mean, shape_self_distance_matrix,
    OrbitRepresentative, ShapeDistanceResult, ShapeMeanResult, ShapeQuotient,
};
pub use srsf::{
    compose_warps, invert_warp, reparameterize_curve, srsf_inverse, srsf_transform,
    warp_inverse_error,
};
pub use tsrvf::{
    tsrvf_from_alignment, tsrvf_from_alignment_with_method, tsrvf_inverse, tsrvf_transform,
    tsrvf_transform_with_method, TransportMethod, TsrvfResult,
};
pub use warp_stats::{warp_statistics, WarpStatistics};

// Re-export pub(crate) items so other crate modules can use them.
pub(crate) use karcher::sqrt_mean_inverse;

use crate::helpers::linear_interp;
use crate::matrix::FdMatrix;
use crate::warping::normalize_warp;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of aligning one curve to another.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentResult {
    /// Warping function γ mapping the domain to itself.
    pub gamma: Vec<f64>,
    /// The aligned (reparameterized) curve.
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
}

/// Result of aligning a set of curves to a common target.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentSetResult {
    /// Warping functions (n × m).
    pub gammas: FdMatrix,
    /// Aligned curves (n × m).
    pub aligned_data: FdMatrix,
    /// Elastic distances for each curve.
    pub distances: Vec<f64>,
}

/// Result of the Karcher mean computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KarcherMeanResult {
    /// Karcher mean curve.
    pub mean: Vec<f64>,
    /// SRSF of the Karcher mean.
    pub mean_srsf: Vec<f64>,
    /// Final warping functions (n × m).
    pub gammas: FdMatrix,
    /// Curves aligned to the mean (n × m).
    pub aligned_data: FdMatrix,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
    /// Pre-computed SRSFs of aligned curves (n × m), if available.
    /// When set, FPCA functions use these instead of recomputing from `aligned_data`.
    pub aligned_srsfs: Option<FdMatrix>,
}

impl KarcherMeanResult {
    /// Create a new `KarcherMeanResult`.
    pub fn new(
        mean: Vec<f64>,
        mean_srsf: Vec<f64>,
        gammas: FdMatrix,
        aligned_data: FdMatrix,
        n_iter: usize,
        converged: bool,
        aligned_srsfs: Option<FdMatrix>,
    ) -> Self {
        Self {
            mean,
            mean_srsf,
            gammas,
            aligned_data,
            n_iter,
            converged,
            aligned_srsfs,
        }
    }
}

// ─── Dynamic Programming Alignment ──────────────────────────────────────────
// Faithful port of fdasrvf's DP algorithm (dp_grid.cpp / dp_nbhd.cpp).

/// Pre-computed coprime neighborhood for nbhd_dim=7 (fdasrvf default).
/// All (dr, dc) with 1 ≤ dr, dc ≤ 7 and gcd(dr, dc) = 1.
/// dr = row delta (q2 direction), dc = column delta (q1 direction).
#[rustfmt::skip]
const COPRIME_NBHD_7: [(usize, usize); 35] = [
    (1,1),(1,2),(1,3),(1,4),(1,5),(1,6),(1,7),
    (2,1),      (2,3),      (2,5),      (2,7),
    (3,1),(3,2),      (3,4),(3,5),      (3,7),
    (4,1),      (4,3),      (4,5),      (4,7),
    (5,1),(5,2),(5,3),(5,4),      (5,6),(5,7),
    (6,1),                  (6,5),      (6,7),
    (7,1),(7,2),(7,3),(7,4),(7,5),(7,6),
];

/// Compute the edge weight for a move from grid point (sr, sc) to (tr, tc).
///
/// Port of fdasrvf's `dp_edge_weight` for 1-D curves on a shared uniform grid.
/// - Rows = q2 indices, columns = q1 indices (matching fdasrvf convention).
/// - `slope = (argvals[tr] - argvals[sr]) / (argvals[tc] - argvals[sc])` = γ'
/// - Walks through sub-intervals synchronized at both curves' breakpoints,
///   accumulating `(q1[idx1] - √slope · q2[idx2])² · dt`.
#[inline]
pub(super) fn dp_edge_weight(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    sc: usize,
    tc: usize,
    sr: usize,
    tr: usize,
) -> f64 {
    let n1 = tc - sc;
    let n2 = tr - sr;
    if n1 == 0 || n2 == 0 {
        return f64::INFINITY;
    }

    let slope = (argvals[tr] - argvals[sr]) / (argvals[tc] - argvals[sc]);
    let rslope = slope.sqrt();

    // Walk through sub-intervals synchronized at breakpoints of both curves
    let mut weight = 0.0;
    let mut i1 = 0usize; // sub-interval index in q1 direction
    let mut i2 = 0usize; // sub-interval index in q2 direction

    while i1 < n1 && i2 < n2 {
        // Current sub-interval boundaries as fractions of the total span
        let left1 = i1 as f64 / n1 as f64;
        let right1 = (i1 + 1) as f64 / n1 as f64;
        let left2 = i2 as f64 / n2 as f64;
        let right2 = (i2 + 1) as f64 / n2 as f64;

        let left = left1.max(left2);
        let right = right1.min(right2);
        let dt = right - left;

        if dt > 0.0 {
            let diff = q1[sc + i1] - rslope * q2[sr + i2];
            weight += diff * diff * dt;
        }

        // Advance whichever sub-interval ends first
        if right1 < right2 {
            i1 += 1;
        } else if right2 < right1 {
            i2 += 1;
        } else {
            i1 += 1;
            i2 += 1;
        }
    }

    // Scale by the span in q1 direction
    weight * (argvals[tc] - argvals[sc])
}

/// Compute the λ·(slope−1)²·dt penalty for a DP edge.
#[inline]
pub(super) fn dp_lambda_penalty(
    argvals: &[f64],
    sc: usize,
    tc: usize,
    sr: usize,
    tr: usize,
    lambda: f64,
) -> f64 {
    if lambda > 0.0 {
        let dt = argvals[tc] - argvals[sc];
        let slope = (argvals[tr] - argvals[sr]) / dt;
        lambda * (slope - 1.0).powi(2) * dt
    } else {
        0.0
    }
}

/// Traceback a parent-pointer array from bottom-right to top-left.
///
/// Returns the path as `(row, col)` pairs from `(0,0)` to `(nrows-1, ncols-1)`.
fn dp_traceback(parent: &[u32], nrows: usize, ncols: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::with_capacity(nrows + ncols);
    let mut cur = (nrows - 1) * ncols + (ncols - 1);
    loop {
        path.push((cur / ncols, cur % ncols));
        if cur == 0 || parent[cur] == u32::MAX {
            break;
        }
        cur = parent[cur] as usize;
    }
    path.reverse();
    path
}

/// Try to relax cell `(tr, tc)` from each coprime neighbor, updating cost and parent.
#[inline]
fn dp_relax_cell<F>(
    e: &mut [f64],
    parent: &mut [u32],
    ncols: usize,
    tr: usize,
    tc: usize,
    edge_cost: &F,
) where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    let idx = tr * ncols + tc;
    for &(dr, dc) in &COPRIME_NBHD_7 {
        if dr > tr || dc > tc {
            continue;
        }
        let sr = tr - dr;
        let sc = tc - dc;
        let src_idx = sr * ncols + sc;
        if e[src_idx] == f64::INFINITY {
            continue;
        }
        let cost = e[src_idx] + edge_cost(sr, sc, tr, tc);
        if cost < e[idx] {
            e[idx] = cost;
            parent[idx] = src_idx as u32;
        }
    }
}

/// Shared DP grid fill + traceback using the coprime neighborhood.
///
/// `edge_cost(sr, sc, tr, tc)` returns the combined edge weight + penalty for
/// a move from local (sr, sc) to local (tr, tc). Returns the raw local-index
/// path from (0,0) to (nrows-1, ncols-1).
pub(super) fn dp_grid_solve<F>(nrows: usize, ncols: usize, edge_cost: F) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    let mut e = vec![f64::INFINITY; nrows * ncols];
    let mut parent = vec![u32::MAX; nrows * ncols];
    e[0] = 0.0;

    for tr in 0..nrows {
        for tc in 0..ncols {
            if tr == 0 && tc == 0 {
                continue;
            }
            dp_relax_cell(&mut e, &mut parent, ncols, tr, tc, &edge_cost);
        }
    }

    dp_traceback(&parent, nrows, ncols)
}

/// Convert a DP path (local row,col indices) to an interpolated+normalized gamma warp.
pub(super) fn dp_path_to_gamma(path: &[(usize, usize)], argvals: &[f64]) -> Vec<f64> {
    let path_tc: Vec<f64> = path.iter().map(|&(_, c)| argvals[c]).collect();
    let path_tr: Vec<f64> = path.iter().map(|&(r, _)| argvals[r]).collect();
    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&path_tc, &path_tr, t))
        .collect();
    normalize_warp(&mut gamma, argvals);
    gamma
}

/// Core DP alignment between two SRSFs on a grid.
///
/// Finds the optimal warping γ minimizing ‖q₁ - (q₂∘γ)√γ'‖².
/// Uses fdasrvf's coprime neighborhood (nbhd_dim=7 → 35 move directions).
/// SRSFs are L2-normalized before alignment (matching fdasrvf's `optimum.reparam`).
pub(crate) fn dp_alignment_core(q1: &[f64], q2: &[f64], argvals: &[f64], lambda: f64) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return argvals.to_vec();
    }

    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    let path = dp_grid_solve(m, m, |sr, sc, tr, tc| {
        dp_edge_weight(&q1n, &q2n, argvals, sc, tc, sr, tr)
            + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
    });

    dp_path_to_gamma(&path, argvals)
}

/// Greatest common divisor (Euclidean algorithm). Used only in tests.
#[cfg(test)]
pub(super) fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Generate coprime neighborhood: all (i,j) with 1 ≤ i,j ≤ nbhd_dim, gcd(i,j) = 1.
/// With nbhd_dim=7 this produces 35 pairs, matching fdasrvf's default.
#[cfg(test)]
pub(super) fn generate_coprime_nbhd(nbhd_dim: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for i in 1..=nbhd_dim {
        for j in 1..=nbhd_dim {
            if gcd(i, j) == 1 {
                pairs.push((i, j));
            }
        }
    }
    pairs
}
