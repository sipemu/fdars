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

mod bayesian;
mod closed;
mod clustering;
mod constrained;
mod diagnostics;
mod elastic_depth;
mod fpns;
mod generative;
mod geodesic;
mod karcher;
mod lambda_cv;
mod multires;
mod nd;
mod outlier;
mod pairwise;
mod partial_match;
mod persistence;
mod phase_boxplot;
mod quality;
mod robust_karcher;
mod set;
mod shape;
mod shape_ci;
mod shift;
mod srsf;
mod transfer;
mod tsrvf;
mod warp_stats;

#[cfg(test)]
mod tests;

// Re-export all public items so that `crate::alignment::X` continues to work.
pub use bayesian::{bayesian_align_pair, BayesianAlignConfig, BayesianAlignmentResult};
pub use closed::{
    elastic_align_pair_closed, elastic_distance_closed, karcher_mean_closed, ClosedAlignmentResult,
    ClosedKarcherMeanResult,
};
pub use clustering::{
    cut_dendrogram, hierarchical_from_distances, kmedoids_from_distances, Dendrogram,
    KMedoidsConfig, KMedoidsResult, Linkage,
};
pub use constrained::{
    elastic_align_pair_constrained, elastic_align_pair_with_landmarks, ConstrainedAlignmentResult,
};
pub use diagnostics::{
    diagnose_alignment, diagnose_pairwise, AlignmentDiagnostic, AlignmentDiagnosticSummary,
    DiagnosticConfig,
};
pub use elastic_depth::{elastic_depth, ElasticDepthResult};
pub use fpns::{horiz_fpns, FpnsResult};
pub use generative::{gauss_model, joint_gauss_model, GenerativeModelResult};
pub use geodesic::{curve_geodesic, curve_geodesic_nd, GeodesicPath, GeodesicPathNd};
pub use karcher::{karcher_mean, karcher_mean_banded, karcher_mean_with_band};
pub use lambda_cv::{lambda_cv, LambdaCvConfig, LambdaCvResult};
pub use multires::{elastic_align_pair_multires, MultiresConfig};
pub use nd::{
    elastic_align_pair_nd, elastic_distance_nd, srsf_inverse_nd, srsf_transform_nd,
    AlignmentResultNd,
};
pub use nd::{karcher_covariance_nd, karcher_mean_nd, pca_nd, KarcherMeanResultNd, PcaNdResult};
pub use outlier::{elastic_outlier_detection, ElasticOutlierConfig, ElasticOutlierResult};
pub use pairwise::{
    amplitude_distance, amplitude_self_distance_matrix, elastic_align_pair,
    elastic_align_pair_banded, elastic_align_pair_penalized, elastic_cross_distance_matrix,
    elastic_cross_distance_matrix_banded, elastic_cross_distance_matrix_with_band,
    elastic_distance, elastic_distance_banded, elastic_self_distance_matrix,
    elastic_self_distance_matrix_banded, elastic_self_distance_matrix_with_band,
    phase_distance_pair, phase_self_distance_matrix, WarpPenaltyType,
};
pub use partial_match::{elastic_partial_match, PartialMatchConfig, PartialMatchResult};
pub use persistence::{peak_persistence, PersistenceDiagramResult};
pub use phase_boxplot::{phase_boxplot, PhaseBoxplot};
pub use quality::{
    alignment_quality, least_squares_score, pairwise_consistency, pairwise_correlation_score,
    sobolev_least_squares_score, warp_complexity, warp_smoothness, AlignmentQuality,
};
pub use robust_karcher::{
    karcher_median, robust_karcher_mean, RobustKarcherConfig, RobustKarcherResult,
};
pub use set::{align_to_target, elastic_decomposition, DecompositionResult};
pub use shape::{
    orbit_representative, shape_distance, shape_mean, shape_self_distance_matrix,
    OrbitRepresentative, ShapeDistanceResult, ShapeMeanResult, ShapeQuotient,
};
pub use shape_ci::{shape_confidence_interval, ShapeCiConfig, ShapeCiResult};
pub use shift::{
    least_squares_shift_registration, ShiftRegistrationResult, DEFAULT_MAX_SHIFT_FRACTION,
};
pub use srsf::{
    compose_warps, invert_warp, reparameterize_curve, srsf_inverse, srsf_transform,
    warp_inverse_error,
};
pub use transfer::{transfer_alignment, TransferAlignConfig, TransferAlignResult};
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
use std::cell::RefCell;

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

// ─── Trait: AlignmentOutput ─────────────────────────────────────────────────

/// Common interface for alignment results, enabling interchangeable
/// alignment methods in downstream analysis (elastic FPCA, regression, etc.).
pub trait AlignmentOutput {
    /// The estimated mean/template curve (length m).
    fn mean(&self) -> &[f64];
    /// The mean SRSF (length m).
    fn mean_srsf(&self) -> &[f64];
    /// The aligned curves (n × m).
    fn aligned_data(&self) -> &FdMatrix;
    /// The warping functions (n × m).
    fn gammas(&self) -> &FdMatrix;
    /// Whether the algorithm converged.
    fn converged(&self) -> bool;
    /// Number of iterations performed.
    fn n_iter(&self) -> usize;
}

impl AlignmentOutput for KarcherMeanResult {
    fn mean(&self) -> &[f64] {
        &self.mean
    }
    fn mean_srsf(&self) -> &[f64] {
        &self.mean_srsf
    }
    fn aligned_data(&self) -> &FdMatrix {
        &self.aligned_data
    }
    fn gammas(&self) -> &FdMatrix {
        &self.gammas
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn n_iter(&self) -> usize {
        self.n_iter
    }
}

impl AlignmentOutput for RobustKarcherResult {
    fn mean(&self) -> &[f64] {
        &self.mean
    }
    fn mean_srsf(&self) -> &[f64] {
        &self.mean_srsf
    }
    fn aligned_data(&self) -> &FdMatrix {
        &self.aligned_data
    }
    fn gammas(&self) -> &FdMatrix {
        &self.gammas
    }
    fn converged(&self) -> bool {
        self.converged
    }
    fn n_iter(&self) -> usize {
        self.n_iter
    }
}

// ─── Conversions ───────────────────────────────────────────────────────────

impl From<RobustKarcherResult> for KarcherMeanResult {
    fn from(r: RobustKarcherResult) -> Self {
        Self {
            mean: r.mean,
            mean_srsf: r.mean_srsf,
            gammas: r.gammas,
            aligned_data: r.aligned_data,
            n_iter: r.n_iter,
            converged: r.converged,
            aligned_srsfs: None,
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
    if tc == sc || tr == sr {
        return f64::INFINITY;
    }
    // General (possibly non-uniform grid) path: slope and span come from argvals.
    let rslope = ((argvals[tr] - argvals[sr]) / (argvals[tc] - argvals[sc])).sqrt();
    let span = argvals[tc] - argvals[sc];
    dp_edge_weight_core(q1, q2, sc, tc, sr, tr, rslope, span)
}

/// Core edge-weight integer walk, shared by the general and uniform-grid paths.
///
/// `rslope = √γ'` and `span` (the q1-direction interval length) are supplied by
/// the caller so the uniform-grid path can look `rslope` up from a precomputed
/// `√(dr/dc)` table (avoiding a `sqrt` + division on every one of the `m²·35`
/// edge evaluations) and derive `span` as `dc·h`.
///
/// Walks the merged sub-interval breakpoints of both curves in integer units of
/// `1/(n1·n2)`: q1's breakpoints land on multiples of n2, q2's on multiples of
/// n1 (both n1, n2 ≤ 7 from the coprime neighborhood). Integer arithmetic keeps
/// the loop free of per-step float divisions and hoists the single `/(n1·n2)`
/// divisor out. Control flow (and tie handling) is exact.
#[inline]
pub(super) fn dp_edge_weight_core(
    q1: &[f64],
    q2: &[f64],
    sc: usize,
    tc: usize,
    sr: usize,
    tr: usize,
    rslope: f64,
    span: f64,
) -> f64 {
    let n1 = tc - sc;
    let n2 = tr - sr;
    if n1 == 0 || n2 == 0 {
        return f64::INFINITY;
    }

    let mut weight_scaled = 0.0;
    let mut i1 = 0usize; // sub-interval index in q1 direction
    let mut i2 = 0usize; // sub-interval index in q2 direction

    while i1 < n1 && i2 < n2 {
        let left = (i1 * n2).max(i2 * n1);
        let right1 = (i1 + 1) * n2;
        let right2 = (i2 + 1) * n1;
        let right = right1.min(right2);
        let dt = right - left;

        if dt > 0 {
            let diff = q1[sc + i1] - rslope * q2[sr + i2];
            weight_scaled += diff * diff * dt as f64;
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

    // Undo the 1/(n1·n2) integer scaling, then scale by the span in q1 direction.
    weight_scaled / (n1 * n2) as f64 * span
}

/// Return the uniform grid spacing `h` if `argvals` is (numerically) uniform.
///
/// Used to enable the fast edge-weight path in [`dp_alignment_core`]: on a
/// uniform grid the DP slope is exactly `dr/dc` and the span is `dc·h`, so both
/// become precomputable and independent of absolute position.
fn uniform_spacing(argvals: &[f64]) -> Option<f64> {
    let m = argvals.len();
    if m < 2 {
        return None;
    }
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    if h.is_nan() || h <= 0.0 {
        return None;
    }
    let tol = h * 1e-9;
    for k in 1..m {
        if (argvals[k] - argvals[k - 1] - h).abs() > tol {
            return None;
        }
    }
    Some(h)
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

thread_local! {
    /// Per-thread reusable DP scratch: `(cost grid, parent pointers)`.
    ///
    /// The grid fill in [`dp_grid_solve`] allocates two `nrows·ncols` buffers on
    /// every alignment; in Karcher-mean / distance-matrix routines that is
    /// `n·iters` (or `n²`) allocations of buffers that are cleared and rewritten
    /// anyway. Keeping them in thread-local storage removes the allocator
    /// round-trip while staying compatible with the `iter_maybe_parallel!` outer
    /// loops (each rayon worker gets its own scratch).
    static DP_SCRATCH: RefCell<(Vec<f64>, Vec<u32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
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
    dp_grid_solve_banded(nrows, ncols, None, edge_cost)
}

/// Shared DP grid fill + traceback, optionally restricted to a Sakoe–Chiba band.
///
/// When `band = Some(r)`, only cells with `|tr − tc| ≤ r` are filled; the rest
/// stay at `+∞` and are skipped as predecessors, so the optimal warp is confined
/// to a diagonal corridor of radius `r`. This turns the O(nrows·ncols) grid fill
/// into O(min(nrows,ncols)·r), the main algorithmic speedup for near-diagonal
/// warps. `band = None` reproduces the full unbanded search exactly. A feasible
/// path always exists for any `r ≥ 0` because the `(1,1)` diagonal move keeps
/// `tr = tc`.
pub(super) fn dp_grid_solve_banded<F>(
    nrows: usize,
    ncols: usize,
    band: Option<usize>,
    edge_cost: F,
) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    DP_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        let (e, parent) = &mut *scratch;
        let size = nrows * ncols;
        e.clear();
        e.resize(size, f64::INFINITY);
        parent.clear();
        parent.resize(size, u32::MAX);
        e[0] = 0.0;

        for tr in 0..nrows {
            for tc in 0..ncols {
                if tr == 0 && tc == 0 {
                    continue;
                }
                if let Some(r) = band {
                    if tr.abs_diff(tc) > r {
                        continue;
                    }
                }
                dp_relax_cell(e, parent, ncols, tr, tc, &edge_cost);
            }
        }

        dp_traceback(parent, nrows, ncols)
    })
}

/// Convert a band width expressed as a fraction of the domain into an index
/// radius for [`dp_grid_solve_banded`].
///
/// `band_frac` is the largest allowed warp displacement `|γ(t) − t|` as a
/// fraction of the number of grid points. Returns `None` (unbounded search)
/// unless `0 < band_frac < 1`; a positive fraction yields a radius of at least
/// 1 so that some warping is always permitted.
pub(super) fn band_radius(band_frac: f64, m: usize) -> Option<usize> {
    if band_frac > 0.0 && band_frac < 1.0 {
        Some(((band_frac * m as f64).ceil() as usize).max(1))
    } else {
        None
    }
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
    dp_alignment_core_banded(q1, q2, argvals, lambda, None)
}

/// Core DP alignment, optionally restricted to a Sakoe–Chiba band of `band`
/// grid-index radius (see [`dp_grid_solve_banded`] / [`band_radius`]).
///
/// `band = None` is identical to [`dp_alignment_core`]. A finite band confines
/// the warp to a diagonal corridor, trading unbounded warps for a large speedup.
pub(crate) fn dp_alignment_core_banded(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    lambda: f64,
    band: Option<usize>,
) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return argvals.to_vec();
    }

    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    let path = if let Some(h) = uniform_spacing(argvals) {
        // Uniform-grid fast path: the DP slope is exactly dr/dc, so precompute
        // the 35 possible √(dr/dc) values (indices 1..=7) once instead of a
        // sqrt + division on every one of the m²·35 edge evaluations. The span
        // is dc·h. Both are independent of absolute position.
        let mut rslope_tab = [[0.0f64; 8]; 8];
        for (dr, row) in rslope_tab.iter_mut().enumerate().skip(1) {
            for (dc, cell) in row.iter_mut().enumerate().skip(1) {
                *cell = (dr as f64 / dc as f64).sqrt();
            }
        }
        dp_grid_solve_banded(m, m, band, |sr, sc, tr, tc| {
            let dr = tr - sr;
            let dc = tc - sc;
            dp_edge_weight_core(
                &q1n,
                &q2n,
                sc,
                tc,
                sr,
                tr,
                rslope_tab[dr][dc],
                dc as f64 * h,
            ) + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
        })
    } else {
        dp_grid_solve_banded(m, m, band, |sr, sc, tr, tc| {
            dp_edge_weight(&q1n, &q2n, argvals, sc, tc, sr, tr)
                + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
        })
    };

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
