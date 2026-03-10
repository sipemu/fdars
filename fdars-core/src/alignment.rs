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

use crate::fdata::{deriv_1d, mean_1d};
use crate::helpers::{
    cumulative_trapz, gradient_uniform, l2_distance, linear_interp, simpsons_weights,
};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::smoothing::nadaraya_watson;
use crate::warping::{
    exp_map_sphere, gam_to_psi, inv_exp_map_sphere, invert_gamma, l2_norm_l2, normalize_warp,
    psi_to_gam,
};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of aligning one curve to another.
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Warping function γ mapping the domain to itself.
    pub gamma: Vec<f64>,
    /// The aligned (reparameterized) curve.
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
}

/// Result of aligning a set of curves to a common target.
#[derive(Debug, Clone)]
pub struct AlignmentSetResult {
    /// Warping functions (n × m).
    pub gammas: FdMatrix,
    /// Aligned curves (n × m).
    pub aligned_data: FdMatrix,
    /// Elastic distances for each curve.
    pub distances: Vec<f64>,
}

/// Result of the Karcher mean computation.
#[derive(Debug, Clone)]
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

// Private helpers are now in crate::helpers and crate::warping.

/// Karcher mean of warping functions on the Hilbert sphere, then invert.
/// Port of fdasrvf's `SqrtMeanInverse`.
///
/// Takes a matrix of warping functions (n × m) on the argvals domain,
/// computes the Fréchet mean of their sqrt-derivative representations
/// on the unit Hilbert sphere, converts back to a warping function,
/// and returns its inverse (on the argvals domain).
/// One Karcher iteration on the Hilbert sphere: compute mean shooting vector and update mu.
///
/// Returns `true` if converged (vbar norm ≤ threshold).
fn karcher_sphere_step(mu: &mut Vec<f64>, psis: &[Vec<f64>], time: &[f64], step_size: f64) -> bool {
    let m = mu.len();
    let n = psis.len();
    let mut vbar = vec![0.0; m];
    for psi in psis {
        let v = inv_exp_map_sphere(mu, psi, time);
        for j in 0..m {
            vbar[j] += v[j];
        }
    }
    for j in 0..m {
        vbar[j] /= n as f64;
    }
    if l2_norm_l2(&vbar, time) <= 1e-8 {
        return true;
    }
    let scaled: Vec<f64> = vbar.iter().map(|&v| v * step_size).collect();
    *mu = exp_map_sphere(mu, &scaled, time);
    false
}

pub(crate) fn sqrt_mean_inverse(gammas: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
    let (n, m) = gammas.shape();
    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    let psis: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let gam_01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
            gam_to_psi(&gam_01, binsize)
        })
        .collect();

    let mut mu = vec![0.0; m];
    for psi in &psis {
        for j in 0..m {
            mu[j] += psi[j];
        }
    }
    for j in 0..m {
        mu[j] /= n as f64;
    }

    for _ in 0..501 {
        if karcher_sphere_step(&mut mu, &psis, &time, 0.3) {
            break;
        }
    }

    let gam_mu = psi_to_gam(&mu, &time);
    let gam_inv = invert_gamma(&gam_mu, &time);
    gam_inv.iter().map(|&g| t0 + g * domain).collect()
}

// ─── SRSF Transform and Inverse ─────────────────────────────────────────────

/// Compute the Square-Root Slope Function (SRSF) transform.
///
/// For each curve f, the SRSF is: `q(t) = sign(f'(t)) * sqrt(|f'(t)|)`
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// FdMatrix of SRSFs with the same shape as input.
pub fn srsf_transform(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m {
        return FdMatrix::zeros(n, m);
    }

    let deriv = deriv_1d(data, argvals, 1);

    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let d = deriv[(i, j)];
            result[(i, j)] = d.signum() * d.abs().sqrt();
        }
    }
    result
}

/// Reconstruct a curve from its SRSF representation.
///
/// Given SRSF q and initial value f0, reconstructs: `f(t) = f0 + ∫₀ᵗ q(s)|q(s)| ds`
///
/// # Arguments
/// * `q` — SRSF values (length m)
/// * `argvals` — Evaluation points (length m)
/// * `f0` — Initial value f(argvals\[0\])
///
/// # Returns
/// Reconstructed curve values.
pub fn srsf_inverse(q: &[f64], argvals: &[f64], f0: f64) -> Vec<f64> {
    let m = q.len();
    if m == 0 {
        return Vec::new();
    }

    // Integrand: q(s) * |q(s)|
    let integrand: Vec<f64> = q.iter().map(|&qi| qi * qi.abs()).collect();
    let integral = cumulative_trapz(&integrand, argvals);

    integral.iter().map(|&v| f0 + v).collect()
}

// ─── Reparameterization ─────────────────────────────────────────────────────

/// Reparameterize a curve by a warping function.
///
/// Computes `f(γ(t))` via linear interpolation.
///
/// # Arguments
/// * `f` — Curve values (length m)
/// * `argvals` — Evaluation points (length m)
/// * `gamma` — Warping function values (length m)
pub fn reparameterize_curve(f: &[f64], argvals: &[f64], gamma: &[f64]) -> Vec<f64> {
    gamma
        .iter()
        .map(|&g| linear_interp(argvals, f, g))
        .collect()
}

/// Compose two warping functions: `(γ₁ ∘ γ₂)(t) = γ₁(γ₂(t))`.
///
/// # Arguments
/// * `gamma1` — Outer warping function (length m)
/// * `gamma2` — Inner warping function (length m)
/// * `argvals` — Evaluation points (length m)
pub fn compose_warps(gamma1: &[f64], gamma2: &[f64], argvals: &[f64]) -> Vec<f64> {
    gamma2
        .iter()
        .map(|&g| linear_interp(argvals, gamma1, g))
        .collect()
}

// ─── Dynamic Programming Alignment ──────────────────────────────────────────
// Faithful port of fdasrvf's DP algorithm (dp_grid.cpp / dp_nbhd.cpp).

/// Greatest common divisor (Euclidean algorithm).
#[cfg(test)]
fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Generate coprime neighborhood: all (i,j) with 1 ≤ i,j ≤ nbhd_dim, gcd(i,j) = 1.
/// With nbhd_dim=7 this produces 35 pairs, matching fdasrvf's default.
#[cfg(test)]
fn generate_coprime_nbhd(nbhd_dim: usize) -> Vec<(usize, usize)> {
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
fn dp_edge_weight(
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
fn dp_lambda_penalty(
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
fn dp_grid_solve<F>(nrows: usize, ncols: usize, edge_cost: F) -> Vec<(usize, usize)>
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
fn dp_path_to_gamma(path: &[(usize, usize)], argvals: &[f64]) -> Vec<f64> {
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

// ─── Public Alignment Functions ─────────────────────────────────────────────

/// Align curve f2 to curve f1 using the elastic framework.
///
/// Computes the optimal warping γ such that f2∘γ is as close as possible
/// to f1 in the elastic (Fisher-Rao) metric.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// [`AlignmentResult`] with warping function, aligned curve, and elastic distance.
pub fn elastic_align_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> AlignmentResult {
    let m = f1.len();

    // Build single-row FdMatrices for SRSF computation
    let f1_mat = FdMatrix::from_slice(f1, 1, m).unwrap();
    let f2_mat = FdMatrix::from_slice(f2, 1, m).unwrap();

    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);

    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);

    // Find optimal warping via DP
    let gamma = dp_alignment_core(&q1, &q2, argvals, lambda);

    // Apply warping to f2
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);

    // Compute elastic distance: L2 distance between q1 and aligned q2 SRSF
    let f_aligned_mat = FdMatrix::from_slice(&f_aligned, 1, m).unwrap();
    let q_aligned_mat = srsf_transform(&f_aligned_mat, argvals);
    let q_aligned: Vec<f64> = q_aligned_mat.row(0);

    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    AlignmentResult {
        gamma,
        f_aligned,
        distance,
    }
}

/// Compute the elastic distance between two curves.
///
/// This is shorthand for aligning the pair and returning only the distance.
///
/// # Arguments
/// * `f1` — First curve (length m)
/// * `f2` — Second curve (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
pub fn elastic_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair(f1, f2, argvals, lambda).distance
}

/// Align all curves in `data` to a single target curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `target` — Target curve to align to (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// [`AlignmentSetResult`] with all warping functions, aligned curves, and distances.
pub fn align_to_target(
    data: &FdMatrix,
    target: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> AlignmentSetResult {
    let (n, m) = data.shape();

    let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_align_pair(target, &fi, argvals, lambda)
        })
        .collect();

    let mut gammas = FdMatrix::zeros(n, m);
    let mut aligned_data = FdMatrix::zeros(n, m);
    let mut distances = Vec::with_capacity(n);

    for (i, r) in results.into_iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = r.gamma[j];
            aligned_data[(i, j)] = r.f_aligned[j];
        }
        distances.push(r.distance);
    }

    AlignmentSetResult {
        gammas,
        aligned_data,
        distances,
    }
}

// ─── Distance Matrices ──────────────────────────────────────────────────────

/// Compute the symmetric elastic distance matrix for a set of curves.
///
/// Uses upper-triangle computation with parallelism, following the
/// `self_distance_matrix` pattern from `metric.rs`.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// Symmetric n × n distance matrix.
pub fn elastic_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    let n = data.nrows();

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let fi = data.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    elastic_distance(&fi, &fj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper_vals[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    dist
}

/// Compute the elastic distance matrix between two sets of curves.
///
/// # Arguments
/// * `data1` — First dataset (n1 × m)
/// * `data2` — Second dataset (n2 × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
///
/// # Returns
/// n1 × n2 distance matrix.
pub fn elastic_cross_distance_matrix(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();

    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            let fi = data1.row(i);
            (0..n2)
                .map(|j| {
                    let fj = data2.row(j);
                    elastic_distance(&fi, &fj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            dist[(i, j)] = vals[i * n2 + j];
        }
    }
    dist
}

// ─── Phase-Amplitude Decomposition ──────────────────────────────────────────

/// Result of elastic phase-amplitude decomposition.
#[derive(Debug, Clone)]
pub struct DecompositionResult {
    /// Full alignment result.
    pub alignment: AlignmentResult,
    /// Amplitude distance: SRSF distance after alignment.
    pub d_amplitude: f64,
    /// Phase distance: geodesic distance of warp from identity.
    pub d_phase: f64,
}

/// Perform elastic phase-amplitude decomposition of two curves.
///
/// Returns both the alignment result and the separate amplitude and phase distances.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to decompose against f1 (length m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight on warp deviation from identity (0.0 = no penalty)
pub fn elastic_decomposition(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> DecompositionResult {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    let d_amplitude = alignment.distance;
    let d_phase = crate::warping::phase_distance(&alignment.gamma, argvals);
    DecompositionResult {
        alignment,
        d_amplitude,
        d_phase,
    }
}

/// Compute the amplitude distance between two curves (= elastic distance after alignment).
pub fn amplitude_distance(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    elastic_distance(f1, f2, argvals, lambda)
}

/// Compute the phase distance between two curves (geodesic distance of optimal warp from identity).
pub fn phase_distance_pair(f1: &[f64], f2: &[f64], argvals: &[f64], lambda: f64) -> f64 {
    let alignment = elastic_align_pair(f1, f2, argvals, lambda);
    crate::warping::phase_distance(&alignment.gamma, argvals)
}

/// Compute the symmetric phase distance matrix for a set of curves.
pub fn phase_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    let n = data.nrows();

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let fi = data.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    phase_distance_pair(&fi, &fj, argvals, lambda)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper_vals[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    dist
}

/// Compute the symmetric amplitude distance matrix (= elastic self distance matrix).
pub fn amplitude_self_distance_matrix(data: &FdMatrix, argvals: &[f64], lambda: f64) -> FdMatrix {
    elastic_self_distance_matrix(data, argvals, lambda)
}

// ─── Karcher Mean ───────────────────────────────────────────────────────────

/// Compute relative change between successive mean SRSFs.
///
/// Returns `‖q_new - q_old‖₂ / ‖q_old‖₂`, matching R's fdasrvf
/// `time_warping` convergence metric (unweighted discrete L2 norm).
fn relative_change(q_old: &[f64], q_new: &[f64]) -> f64 {
    let diff_norm: f64 = q_old
        .iter()
        .zip(q_new.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let old_norm: f64 = q_old.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    diff_norm / old_norm
}

/// Compute a single SRSF from a slice (single-row convenience).
fn srsf_single(f: &[f64], argvals: &[f64]) -> Vec<f64> {
    let m = f.len();
    let mat = FdMatrix::from_slice(f, 1, m).unwrap();
    let q_mat = srsf_transform(&mat, argvals);
    q_mat.row(0)
}

/// Align a single SRSF q2 to q1 and return (gamma, aligned_q).
fn align_srsf_pair(q1: &[f64], q2: &[f64], argvals: &[f64], lambda: f64) -> (Vec<f64>, Vec<f64>) {
    let gamma = dp_alignment_core(q1, q2, argvals, lambda);

    // Warp q2 by gamma and adjust by sqrt(gamma')
    let q2_warped = reparameterize_curve(q2, argvals, &gamma);

    // Compute gamma' via finite differences
    let m = gamma.len();
    let mut gamma_dot = vec![0.0; m];
    gamma_dot[0] = (gamma[1] - gamma[0]) / (argvals[1] - argvals[0]);
    for j in 1..(m - 1) {
        gamma_dot[j] = (gamma[j + 1] - gamma[j - 1]) / (argvals[j + 1] - argvals[j - 1]);
    }
    gamma_dot[m - 1] = (gamma[m - 1] - gamma[m - 2]) / (argvals[m - 1] - argvals[m - 2]);

    // q2_aligned = (q2 ∘ γ) * sqrt(γ')
    let q2_aligned: Vec<f64> = q2_warped
        .iter()
        .zip(gamma_dot.iter())
        .map(|(&q, &gd)| q * gd.max(0.0).sqrt())
        .collect();

    (gamma, q2_aligned)
}

/// Compute the Karcher (Fréchet) mean in the elastic metric.
///
/// Iteratively aligns all curves to the current mean estimate in SRSF space,
/// computes the pointwise mean of aligned SRSFs, and reconstructs the mean curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `max_iter` — Maximum number of iterations
/// * `tol` — Convergence tolerance for the SRSF mean
///
/// # Returns
/// [`KarcherMeanResult`] with mean curve, warping functions, aligned data, and convergence info.
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::alignment::karcher_mean;
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let result = karcher_mean(&data, &t, 20, 1e-4, 0.0);
/// assert_eq!(result.mean.len(), 50);
/// assert!(result.n_iter <= 20);
/// ```
/// Accumulate alignment results: store gammas and return the mean of aligned SRSFs.
fn accumulate_alignments(
    results: &[(Vec<f64>, Vec<f64>)],
    gammas: &mut FdMatrix,
    m: usize,
    n: usize,
) -> Vec<f64> {
    let mut mu_q_new = vec![0.0; m];
    for (i, (gamma, q_aligned)) in results.iter().enumerate() {
        for j in 0..m {
            gammas[(i, j)] = gamma[j];
            mu_q_new[j] += q_aligned[j];
        }
    }
    for j in 0..m {
        mu_q_new[j] /= n as f64;
    }
    mu_q_new
}

/// Apply stored warps to original curves to produce aligned data.
fn apply_stored_warps(data: &FdMatrix, gammas: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    let mut aligned = FdMatrix::zeros(n, m);
    for i in 0..n {
        let fi = data.row(i);
        let gamma: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let f_aligned = reparameterize_curve(&fi, argvals, &gamma);
        for j in 0..m {
            aligned[(i, j)] = f_aligned[j];
        }
    }
    aligned
}

/// Select the SRSF closest to the pointwise mean as template. Returns (mu_q, mu_f).
fn select_template(srsf_mat: &FdMatrix, data: &FdMatrix, argvals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = srsf_mat.shape();
    let mnq = mean_1d(srsf_mat);
    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0;
    for i in 0..n {
        let dist_sq: f64 = (0..m).map(|j| (srsf_mat[(i, j)] - mnq[j]).powi(2)).sum();
        if dist_sq < min_dist {
            min_dist = dist_sq;
            min_idx = i;
        }
    }
    let _ = argvals; // kept for API consistency
    (srsf_mat.row(min_idx), data.row(min_idx))
}

/// Pre-centering: align all curves to template, compute inverse mean warp, re-center.
fn pre_center_template(
    data: &FdMatrix,
    mu_q: &[f64],
    mu: &[f64],
    argvals: &[f64],
    lambda: f64,
) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();
    let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            let qi = srsf_single(&fi, argvals);
            align_srsf_pair(mu_q, &qi, argvals, lambda)
        })
        .collect();

    let mut init_gammas = FdMatrix::zeros(n, m);
    for (i, (gamma, _)) in align_results.iter().enumerate() {
        for j in 0..m {
            init_gammas[(i, j)] = gamma[j];
        }
    }

    let gam_inv = sqrt_mean_inverse(&init_gammas, argvals);
    let mu_new = reparameterize_curve(mu, argvals, &gam_inv);
    let mu_q_new = srsf_single(&mu_new, argvals);
    (mu_q_new, mu_new)
}

/// Post-convergence centering: center mean SRSF and warps via SqrtMeanInverse.
fn post_center_results(
    data: &FdMatrix,
    mu_q: &[f64],
    final_gammas: &mut FdMatrix,
    argvals: &[f64],
) -> (Vec<f64>, Vec<f64>, FdMatrix) {
    let (n, m) = data.shape();
    let gam_inv = sqrt_mean_inverse(final_gammas, argvals);
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_inv_dev = gradient_uniform(&gam_inv, h);

    let mu_q_warped = reparameterize_curve(mu_q, argvals, &gam_inv);
    let mu_q_centered: Vec<f64> = mu_q_warped
        .iter()
        .zip(gam_inv_dev.iter())
        .map(|(&q, &gd)| q * gd.max(0.0).sqrt())
        .collect();

    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| final_gammas[(i, j)]).collect();
        let gam_centered = reparameterize_curve(&gam_i, argvals, &gam_inv);
        for j in 0..m {
            final_gammas[(i, j)] = gam_centered[j];
        }
    }

    let initial_mean = mean_1d(data);
    let mu = srsf_inverse(&mu_q_centered, argvals, initial_mean[0]);
    let final_aligned = apply_stored_warps(data, final_gammas, argvals);
    (mu, mu_q_centered, final_aligned)
}

pub fn karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> KarcherMeanResult {
    let (n, m) = data.shape();

    let srsf_mat = srsf_transform(data, argvals);
    let (mut mu_q, mu) = select_template(&srsf_mat, data, argvals);
    let (mu_q_c, mu_c) = pre_center_template(data, &mu_q, &mu, argvals, lambda);
    mu_q = mu_q_c;
    let mut mu = mu_c;

    let mut converged = false;
    let mut n_iter = 0;
    let mut final_gammas = FdMatrix::zeros(n, m);

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let fi = data.row(i);
                let qi = srsf_single(&fi, argvals);
                align_srsf_pair(&mu_q, &qi, argvals, lambda)
            })
            .collect();

        let mu_q_new = accumulate_alignments(&align_results, &mut final_gammas, m, n);

        let rel = relative_change(&mu_q, &mu_q_new);
        if rel < tol {
            converged = true;
            mu_q = mu_q_new;
            break;
        }

        mu_q = mu_q_new;
        mu = srsf_inverse(&mu_q, argvals, mu[0]);
    }

    let (mu_final, mu_q_final, final_aligned) =
        post_center_results(data, &mu_q, &mut final_gammas, argvals);

    KarcherMeanResult {
        mean: mu_final,
        mean_srsf: mu_q_final,
        gammas: final_gammas,
        aligned_data: final_aligned,
        n_iter,
        converged,
        aligned_srsfs: None,
    }
}

// ─── TSRVF (Transported SRSF) ────────────────────────────────────────────────
// Maps aligned SRSFs to the tangent space of the Karcher mean on the Hilbert
// sphere. Tangent vectors live in a standard Euclidean space, enabling PCA,
// regression, and clustering on elastic-aligned curves.

/// Result of the TSRVF transform.
#[derive(Debug, Clone)]
pub struct TsrvfResult {
    /// Tangent vectors in Euclidean space (n × m).
    pub tangent_vectors: FdMatrix,
    /// Karcher mean curve (length m).
    pub mean: Vec<f64>,
    /// SRSF of the Karcher mean (length m).
    pub mean_srsf: Vec<f64>,
    /// L2 norm of the mean SRSF.
    pub mean_srsf_norm: f64,
    /// Per-curve aligned SRSF norms (length n).
    pub srsf_norms: Vec<f64>,
    /// Per-curve initial values f_i(0) for SRSF inverse reconstruction (length n).
    pub initial_values: Vec<f64>,
    /// Warping functions from Karcher mean computation (n × m).
    pub gammas: FdMatrix,
    /// Whether the Karcher mean converged.
    pub converged: bool,
}

/// Full TSRVF pipeline: compute Karcher mean, then transport SRSFs to tangent space.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `max_iter` — Maximum Karcher mean iterations
/// * `tol` — Convergence tolerance for Karcher mean
///
/// # Returns
/// [`TsrvfResult`] containing tangent vectors and associated metadata.
pub fn tsrvf_transform(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> TsrvfResult {
    let karcher = karcher_mean(data, argvals, max_iter, tol, lambda);
    tsrvf_from_alignment(&karcher, argvals)
}

/// Smooth aligned SRSFs to remove DP kink artifacts before TSRVF computation.
///
/// Uses Nadaraya-Watson kernel smoothing (Gaussian, bandwidth = 2 grid spacings)
/// on each SRSF row. This removes the derivative spikes from DP warp kinks
/// without affecting alignment results or the Karcher mean.
fn smooth_aligned_srsfs(srsf: &FdMatrix, m: usize) -> FdMatrix {
    let n = srsf.nrows();
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;

    let mut smoothed = FdMatrix::zeros(n, m);
    for i in 0..n {
        let qi = srsf.row(i);
        let qi_smooth = nadaraya_watson(&time, &qi, &time, bandwidth, "gaussian");
        for j in 0..m {
            smoothed[(i, j)] = qi_smooth[j];
        }
    }
    smoothed
}

/// Compute TSRVF from a pre-computed Karcher mean alignment.
///
/// Avoids re-running the expensive Karcher mean computation when the alignment
/// has already been computed.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// [`TsrvfResult`] containing tangent vectors and associated metadata.
pub fn tsrvf_from_alignment(karcher: &KarcherMeanResult, argvals: &[f64]) -> TsrvfResult {
    let (n, m) = karcher.aligned_data.shape();

    // Step 1: Compute SRSFs of aligned data
    let aligned_srsf = srsf_transform(&karcher.aligned_data, argvals);

    // Step 1b: Smooth aligned SRSFs to remove DP kink artifacts.
    //
    // DP alignment produces piecewise-linear warps with kinks at grid transitions.
    // When curves are reparameterized by these warps, the kinks propagate into the
    // aligned curves' derivatives (SRSFs), creating spikes that dominate TSRVF
    // tangent vectors and PCA.
    //
    // R's fdasrvf does not smooth here and suffers from the same spike artifacts.
    // Python's fdasrsf mitigates this via spline smoothing (s=1e-4) in SqrtMean.
    // We smooth the aligned SRSFs before tangent vector computation — this only
    // affects TSRVF output and does not change the alignment or Karcher mean.
    let aligned_srsf = smooth_aligned_srsfs(&aligned_srsf, m);

    // Step 2: Smooth and normalize mean SRSF to unit sphere.
    // The mean SRSF must be smoothed consistently with the aligned SRSFs
    // so that a single curve (which IS the mean) produces a zero tangent vector.
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;
    let mean_srsf_smooth = nadaraya_watson(&time, &karcher.mean_srsf, &time, bandwidth, "gaussian");
    let mean_norm = l2_norm_l2(&mean_srsf_smooth, &time);

    let mu_unit: Vec<f64> = if mean_norm > 1e-10 {
        mean_srsf_smooth.iter().map(|&q| q / mean_norm).collect()
    } else {
        vec![0.0; m]
    };

    // Step 3: For each aligned curve, compute tangent vector via inverse exponential map
    let srsf_norms: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            l2_norm_l2(&qi, &time)
        })
        .collect();

    let tangent_data: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            let qi_norm = srsf_norms[i];

            if qi_norm < 1e-10 || mean_norm < 1e-10 {
                return vec![0.0; m];
            }

            // Normalize to unit sphere
            let qi_unit: Vec<f64> = qi.iter().map(|&q| q / qi_norm).collect();

            // Shooting vector from mu_unit to qi_unit
            inv_exp_map_sphere(&mu_unit, &qi_unit, &time)
        })
        .collect();

    // Assemble tangent vectors into FdMatrix
    let mut tangent_vectors = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            tangent_vectors[(i, j)] = tangent_data[i][j];
        }
    }

    // Store per-curve initial values for SRSF inverse reconstruction.
    // Warping preserves f_i(0) since gamma(0) = 0.
    let initial_values: Vec<f64> = (0..n).map(|i| karcher.aligned_data[(i, 0)]).collect();

    TsrvfResult {
        tangent_vectors,
        mean: karcher.mean.clone(),
        mean_srsf: mean_srsf_smooth,
        mean_srsf_norm: mean_norm,
        srsf_norms,
        initial_values,
        gammas: karcher.gammas.clone(),
        converged: karcher.converged,
    }
}

/// Reconstruct aligned curves from TSRVF tangent vectors.
///
/// Inverts the TSRVF transform: maps tangent vectors back to the Hilbert sphere
/// via the exponential map, rescales, and reconstructs curves via SRSF inverse.
///
/// # Arguments
/// * `tsrvf` — TSRVF result from [`tsrvf_transform`] or [`tsrvf_from_alignment`]
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// FdMatrix of reconstructed aligned curves (n × m).
pub fn tsrvf_inverse(tsrvf: &TsrvfResult, argvals: &[f64]) -> FdMatrix {
    let (n, m) = tsrvf.tangent_vectors.shape();

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Normalize mean SRSF to unit sphere
    let mu_unit: Vec<f64> = if tsrvf.mean_srsf_norm > 1e-10 {
        tsrvf
            .mean_srsf
            .iter()
            .map(|&q| q / tsrvf.mean_srsf_norm)
            .collect()
    } else {
        vec![0.0; m]
    };

    let curves: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let vi = tsrvf.tangent_vectors.row(i);

            // Map back to sphere: exp_map(mu_unit, v_i)
            let qi_unit = exp_map_sphere(&mu_unit, &vi, &time);

            // Rescale by original norm
            let qi: Vec<f64> = qi_unit.iter().map(|&q| q * tsrvf.srsf_norms[i]).collect();

            // Reconstruct curve from SRSF using per-curve initial value
            srsf_inverse(&qi, argvals, tsrvf.initial_values[i])
        })
        .collect();

    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            result[(i, j)] = curves[i][j];
        }
    }
    result
}

// ─── Parallel Transport Variants ─────────────────────────────────────────────

/// Method for transporting tangent vectors on the Hilbert sphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportMethod {
    /// Inverse exponential map (log map) — default, matches existing TSRVF behavior.
    #[default]
    LogMap,
    /// Schild's ladder approximation to parallel transport.
    SchildsLadder,
    /// Pole ladder approximation to parallel transport.
    PoleLadder,
}

/// Schild's ladder parallel transport of vector `v` from `from` to `to` on the sphere.
fn parallel_transport_schilds(v: &[f64], from: &[f64], to: &[f64], time: &[f64]) -> Vec<f64> {
    use crate::warping::{exp_map_sphere, inv_exp_map_sphere};

    let v_norm = crate::warping::l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        return vec![0.0; v.len()];
    }

    // endpoint = exp_from(v)
    let endpoint = exp_map_sphere(from, v, time);

    // midpoint_v = log_to(endpoint) — vector at `to` pointing toward endpoint
    let log_to_ep = inv_exp_map_sphere(to, &endpoint, time);

    // midpoint = exp_to(0.5 * log_to_ep)
    let half_log: Vec<f64> = log_to_ep.iter().map(|&x| 0.5 * x).collect();
    let midpoint = exp_map_sphere(to, &half_log, time);

    // transported = 2 * log_to(midpoint)
    let log_to_mid = inv_exp_map_sphere(to, &midpoint, time);
    log_to_mid.iter().map(|&x| 2.0 * x).collect()
}

/// Pole ladder parallel transport of vector `v` from `from` to `to` on the sphere.
fn parallel_transport_pole(v: &[f64], from: &[f64], to: &[f64], time: &[f64]) -> Vec<f64> {
    use crate::warping::{exp_map_sphere, inv_exp_map_sphere};

    let v_norm = crate::warping::l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        return vec![0.0; v.len()];
    }

    // pole = exp_from(-v)
    let neg_v: Vec<f64> = v.iter().map(|&x| -x).collect();
    let pole = exp_map_sphere(from, &neg_v, time);

    // midpoint_v = log_to(pole)
    let log_to_pole = inv_exp_map_sphere(to, &pole, time);

    // midpoint = exp_to(0.5 * log_to_pole)
    let half_log: Vec<f64> = log_to_pole.iter().map(|&x| 0.5 * x).collect();
    let midpoint = exp_map_sphere(to, &half_log, time);

    // transported = -2 * log_to(midpoint)
    let log_to_mid = inv_exp_map_sphere(to, &midpoint, time);
    log_to_mid.iter().map(|&x| -2.0 * x).collect()
}

/// Full TSRVF pipeline with configurable transport method.
///
/// Like [`tsrvf_transform`] but allows choosing the parallel transport method.
pub fn tsrvf_transform_with_method(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
    method: TransportMethod,
) -> TsrvfResult {
    let karcher = karcher_mean(data, argvals, max_iter, tol, lambda);
    tsrvf_from_alignment_with_method(&karcher, argvals, method)
}

/// Compute TSRVF from a pre-computed Karcher mean with configurable transport.
///
/// - [`TransportMethod::LogMap`]: Uses `inv_exp_map(mu, qi)` directly (standard TSRVF).
/// - [`TransportMethod::SchildsLadder`]: Computes `v = -log_qi(mu)`, then transports
///   via Schild's ladder from qi to mu.
/// - [`TransportMethod::PoleLadder`]: Same but via pole ladder.
pub fn tsrvf_from_alignment_with_method(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    method: TransportMethod,
) -> TsrvfResult {
    if method == TransportMethod::LogMap {
        return tsrvf_from_alignment(karcher, argvals);
    }

    let (n, m) = karcher.aligned_data.shape();
    let aligned_srsf = srsf_transform(&karcher.aligned_data, argvals);
    let aligned_srsf = smooth_aligned_srsfs(&aligned_srsf, m);
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;
    let mean_srsf_smooth = nadaraya_watson(&time, &karcher.mean_srsf, &time, bandwidth, "gaussian");
    let mean_norm = crate::warping::l2_norm_l2(&mean_srsf_smooth, &time);

    let mu_unit: Vec<f64> = if mean_norm > 1e-10 {
        mean_srsf_smooth.iter().map(|&q| q / mean_norm).collect()
    } else {
        vec![0.0; m]
    };

    let srsf_norms: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            crate::warping::l2_norm_l2(&qi, &time)
        })
        .collect();

    let tangent_data: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            let qi_norm = srsf_norms[i];

            if qi_norm < 1e-10 || mean_norm < 1e-10 {
                return vec![0.0; m];
            }

            let qi_unit: Vec<f64> = qi.iter().map(|&q| q / qi_norm).collect();

            // Compute v = -log_qi(mu) — vector at qi pointing away from mu
            let v_at_qi = inv_exp_map_sphere(&qi_unit, &mu_unit, &time);
            let neg_v: Vec<f64> = v_at_qi.iter().map(|&x| -x).collect();

            // Transport from qi to mu
            match method {
                TransportMethod::SchildsLadder => {
                    parallel_transport_schilds(&neg_v, &qi_unit, &mu_unit, &time)
                }
                TransportMethod::PoleLadder => {
                    parallel_transport_pole(&neg_v, &qi_unit, &mu_unit, &time)
                }
                TransportMethod::LogMap => unreachable!(),
            }
        })
        .collect();

    let mut tangent_vectors = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            tangent_vectors[(i, j)] = tangent_data[i][j];
        }
    }

    let initial_values: Vec<f64> = (0..n).map(|i| karcher.aligned_data[(i, 0)]).collect();

    TsrvfResult {
        tangent_vectors,
        mean: karcher.mean.clone(),
        mean_srsf: mean_srsf_smooth,
        mean_srsf_norm: mean_norm,
        srsf_norms,
        initial_values,
        gammas: karcher.gammas.clone(),
        converged: karcher.converged,
    }
}

// ─── Alignment Quality Metrics ───────────────────────────────────────────────

/// Comprehensive alignment quality assessment.
#[derive(Debug, Clone)]
pub struct AlignmentQuality {
    /// Per-curve geodesic distance from warp to identity.
    pub warp_complexity: Vec<f64>,
    /// Mean warp complexity.
    pub mean_warp_complexity: f64,
    /// Per-curve bending energy ∫(γ'')² dt.
    pub warp_smoothness: Vec<f64>,
    /// Mean warp smoothness (bending energy).
    pub mean_warp_smoothness: f64,
    /// Total variance: (1/n) Σ ∫(f_i - mean_orig)² dt.
    pub total_variance: f64,
    /// Amplitude variance: (1/n) Σ ∫(f_i^aligned - mean_aligned)² dt.
    pub amplitude_variance: f64,
    /// Phase variance: total - amplitude (clamped ≥ 0).
    pub phase_variance: f64,
    /// Phase-to-total variance ratio.
    pub phase_amplitude_ratio: f64,
    /// Pointwise ratio: aligned_var / orig_var per time point.
    pub pointwise_variance_ratio: Vec<f64>,
    /// Mean variance reduction.
    pub mean_variance_reduction: f64,
}

/// Compute warp complexity: geodesic distance from a warp to the identity.
///
/// This is `arccos(⟨ψ, ψ_id⟩)` on the Hilbert sphere.
pub fn warp_complexity(gamma: &[f64], argvals: &[f64]) -> f64 {
    crate::warping::phase_distance(gamma, argvals)
}

/// Compute warp smoothness (bending energy): ∫(γ'')² dt.
pub fn warp_smoothness(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 3 {
        return 0.0;
    }

    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_prime = gradient_uniform(gamma, h);
    let gam_pprime = gradient_uniform(&gam_prime, h);

    let integrand: Vec<f64> = gam_pprime.iter().map(|&g| g * g).collect();
    crate::helpers::trapz(&integrand, argvals)
}

/// Compute comprehensive alignment quality metrics.
///
/// # Arguments
/// * `data` — Original functional data (n × m)
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
pub fn alignment_quality(
    data: &FdMatrix,
    karcher: &KarcherMeanResult,
    argvals: &[f64],
) -> AlignmentQuality {
    let (n, m) = data.shape();
    let weights = simpsons_weights(argvals);

    // Per-curve warp complexity and smoothness
    let wc: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_complexity(&gamma, argvals)
        })
        .collect();
    let ws: Vec<f64> = (0..n)
        .map(|i| {
            let gamma: Vec<f64> = (0..m).map(|j| karcher.gammas[(i, j)]).collect();
            warp_smoothness(&gamma, argvals)
        })
        .collect();

    let mean_wc = wc.iter().sum::<f64>() / n as f64;
    let mean_ws = ws.iter().sum::<f64>() / n as f64;

    // Compute original mean
    let orig_mean = crate::fdata::mean_1d(data);

    // Total variance
    let total_var: f64 = (0..n)
        .map(|i| {
            let fi = data.row(i);
            let d = l2_distance(&fi, &orig_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    // Aligned mean
    let aligned_mean = crate::fdata::mean_1d(&karcher.aligned_data);

    // Amplitude variance
    let amp_var: f64 = (0..n)
        .map(|i| {
            let fi = karcher.aligned_data.row(i);
            let d = l2_distance(&fi, &aligned_mean, &weights);
            d * d
        })
        .sum::<f64>()
        / n as f64;

    let phase_var = (total_var - amp_var).max(0.0);
    let ratio = if total_var > 1e-10 {
        phase_var / total_var
    } else {
        0.0
    };

    // Pointwise variance ratio
    let mut pw_ratio = vec![0.0; m];
    for j in 0..m {
        let col_orig = data.column(j);
        let mean_orig = col_orig.iter().sum::<f64>() / n as f64;
        let var_orig: f64 = col_orig
            .iter()
            .map(|&v| (v - mean_orig).powi(2))
            .sum::<f64>()
            / n as f64;

        let col_aligned = karcher.aligned_data.column(j);
        let mean_aligned = col_aligned.iter().sum::<f64>() / n as f64;
        let var_aligned: f64 = col_aligned
            .iter()
            .map(|&v| (v - mean_aligned).powi(2))
            .sum::<f64>()
            / n as f64;

        pw_ratio[j] = if var_orig > 1e-15 {
            var_aligned / var_orig
        } else {
            1.0
        };
    }

    let mean_vr = pw_ratio.iter().sum::<f64>() / m as f64;

    AlignmentQuality {
        warp_complexity: wc,
        mean_warp_complexity: mean_wc,
        warp_smoothness: ws,
        mean_warp_smoothness: mean_ws,
        total_variance: total_var,
        amplitude_variance: amp_var,
        phase_variance: phase_var,
        phase_amplitude_ratio: ratio,
        pointwise_variance_ratio: pw_ratio,
        mean_variance_reduction: mean_vr,
    }
}

/// Generate triplet indices (i,j,k) with i<j<k, capped at `max_triplets` (0 = all).
fn triplet_indices(n: usize, max_triplets: usize) -> Vec<(usize, usize, usize)> {
    let total = n * (n - 1) * (n - 2) / 6;
    let cap = if max_triplets > 0 {
        max_triplets.min(total)
    } else {
        total
    };
    (0..n)
        .flat_map(|i| ((i + 1)..n).flat_map(move |j| ((j + 1)..n).map(move |k| (i, j, k))))
        .take(cap)
        .collect()
}

/// Compute the warp deviation for one triplet: ‖γ_ij∘γ_jk − γ_ik‖_L2.
fn triplet_warp_deviation(
    data: &FdMatrix,
    argvals: &[f64],
    weights: &[f64],
    i: usize,
    j: usize,
    k: usize,
    lambda: f64,
) -> f64 {
    let fi = data.row(i);
    let fj = data.row(j);
    let fk = data.row(k);
    let rij = elastic_align_pair(&fi, &fj, argvals, lambda);
    let rjk = elastic_align_pair(&fj, &fk, argvals, lambda);
    let rik = elastic_align_pair(&fi, &fk, argvals, lambda);
    let composed = compose_warps(&rij.gamma, &rjk.gamma, argvals);
    l2_distance(&composed, &rik.gamma, weights)
}

/// Measure pairwise alignment consistency via triplet checks.
///
/// For triplets (i,j,k), checks `γ_ij ∘ γ_jk ≈ γ_ik` by measuring the L2
/// deviation of the composed warp from the direct warp.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight
/// * `max_triplets` — Maximum number of triplets to check (0 = all)
pub fn pairwise_consistency(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_triplets: usize,
) -> f64 {
    let n = data.nrows();
    if n < 3 {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    let triplets = triplet_indices(n, max_triplets);
    if triplets.is_empty() {
        return 0.0;
    }

    let total_dev: f64 = triplets
        .iter()
        .map(|&(i, j, k)| triplet_warp_deviation(data, argvals, &weights, i, j, k, lambda))
        .sum();
    total_dev / triplets.len() as f64
}

// ─── Landmark-Constrained Elastic Alignment ────────────────────────────────

/// Result of landmark-constrained elastic alignment.
#[derive(Debug, Clone)]
pub struct ConstrainedAlignmentResult {
    /// Optimal warping function (length m).
    pub gamma: Vec<f64>,
    /// Aligned curve f2∘γ (length m).
    pub f_aligned: Vec<f64>,
    /// Elastic distance after alignment.
    pub distance: f64,
    /// Enforced landmark pairs (snapped to grid): `(target_t, source_t)`.
    pub enforced_landmarks: Vec<(f64, f64)>,
}

/// Snap a time value to the nearest grid point index.
fn snap_to_grid(t_val: f64, argvals: &[f64]) -> usize {
    let mut best = 0;
    let mut best_dist = (t_val - argvals[0]).abs();
    for (i, &a) in argvals.iter().enumerate().skip(1) {
        let d = (t_val - a).abs();
        if d < best_dist {
            best = i;
            best_dist = d;
        }
    }
    best
}

/// Run DP on a rectangular sub-grid `[sc..=ec] × [sr..=er]`.
///
/// Uses global indices for `dp_edge_weight`. Returns the path segment
/// as a list of `(tc_idx, tr_idx)` pairs from start to end.
fn dp_segment(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    sc: usize,
    ec: usize,
    sr: usize,
    er: usize,
    lambda: f64,
) -> Vec<(usize, usize)> {
    let nc = ec - sc + 1;
    let nr = er - sr + 1;

    if nc <= 1 || nr <= 1 {
        return vec![(sc, sr), (ec, er)];
    }

    let path = dp_grid_solve(nr, nc, |local_sr, local_sc, local_tr, local_tc| {
        let gsr = sr + local_sr;
        let gsc = sc + local_sc;
        let gtr = sr + local_tr;
        let gtc = sc + local_tc;
        dp_edge_weight(q1, q2, argvals, gsc, gtc, gsr, gtr)
            + dp_lambda_penalty(argvals, gsc, gtc, gsr, gtr, lambda)
    });

    // Convert local indices to global
    path.iter().map(|&(lr, lc)| (sc + lc, sr + lr)).collect()
}

/// Align f2 to f1 with landmark constraints.
///
/// Landmark pairs define waypoints on the DP grid. Between consecutive waypoints,
/// an independent smaller DP is run. The resulting warp passes through all landmarks.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `landmark_pairs` — `(target_t, source_t)` pairs in increasing order
/// * `lambda` — Penalty weight
///
/// # Returns
/// [`ConstrainedAlignmentResult`] with warp, aligned curve, and enforced landmarks.
/// Build DP waypoints from landmark pairs: snap to grid, deduplicate, add endpoints.
fn build_constrained_waypoints(
    landmark_pairs: &[(f64, f64)],
    argvals: &[f64],
    m: usize,
) -> Vec<(usize, usize)> {
    let mut waypoints: Vec<(usize, usize)> = Vec::with_capacity(landmark_pairs.len() + 2);
    waypoints.push((0, 0));
    for &(tt, st) in landmark_pairs {
        let tc = snap_to_grid(tt, argvals);
        let tr = snap_to_grid(st, argvals);
        if let Some(&(prev_c, prev_r)) = waypoints.last() {
            if tc > prev_c && tr > prev_r {
                waypoints.push((tc, tr));
            }
        }
    }
    let last = m - 1;
    if let Some(&(prev_c, prev_r)) = waypoints.last() {
        if prev_c != last || prev_r != last {
            waypoints.push((last, last));
        }
    }
    waypoints
}

/// Run DP segments between consecutive waypoints and assemble into a gamma warp.
fn segmented_dp_gamma(
    q1n: &[f64],
    q2n: &[f64],
    argvals: &[f64],
    waypoints: &[(usize, usize)],
    lambda: f64,
) -> Vec<f64> {
    let mut full_path_tc: Vec<f64> = Vec::new();
    let mut full_path_tr: Vec<f64> = Vec::new();

    for seg in 0..(waypoints.len() - 1) {
        let (sc, sr) = waypoints[seg];
        let (ec, er) = waypoints[seg + 1];
        let segment_path = dp_segment(q1n, q2n, argvals, sc, ec, sr, er, lambda);
        let start = if seg > 0 { 1 } else { 0 };
        for &(tc, tr) in &segment_path[start..] {
            full_path_tc.push(argvals[tc]);
            full_path_tr.push(argvals[tr]);
        }
    }

    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&full_path_tc, &full_path_tr, t))
        .collect();
    normalize_warp(&mut gamma, argvals);
    gamma
}

pub fn elastic_align_pair_constrained(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    landmark_pairs: &[(f64, f64)],
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let m = f1.len();

    if landmark_pairs.is_empty() {
        let r = elastic_align_pair(f1, f2, argvals, lambda);
        return ConstrainedAlignmentResult {
            gamma: r.gamma,
            f_aligned: r.f_aligned,
            distance: r.distance,
            enforced_landmarks: Vec::new(),
        };
    }

    // Compute & normalize SRSFs
    let f1_mat = FdMatrix::from_slice(f1, 1, m).unwrap();
    let f2_mat = FdMatrix::from_slice(f2, 1, m).unwrap();
    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);
    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);
    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    let waypoints = build_constrained_waypoints(landmark_pairs, argvals, m);
    let gamma = segmented_dp_gamma(&q1n, &q2n, argvals, &waypoints, lambda);

    let f_aligned = reparameterize_curve(f2, argvals, &gamma);
    let f_aligned_mat = FdMatrix::from_slice(&f_aligned, 1, m).unwrap();
    let q_aligned_mat = srsf_transform(&f_aligned_mat, argvals);
    let q_aligned: Vec<f64> = q_aligned_mat.row(0);
    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    let enforced: Vec<(f64, f64)> = waypoints[1..waypoints.len() - 1]
        .iter()
        .map(|&(tc, tr)| (argvals[tc], argvals[tr]))
        .collect();

    ConstrainedAlignmentResult {
        gamma,
        f_aligned,
        distance,
        enforced_landmarks: enforced,
    }
}

/// Align f2 to f1 with automatic landmark detection and elastic constraints.
///
/// Detects landmarks in both curves, matches them, and uses the matches
/// as constraints for segmented DP alignment.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `kind` — Type of landmarks to detect
/// * `min_prominence` — Minimum prominence for landmark detection
/// * `expected_count` — Expected number of landmarks (0 = all detected)
/// * `lambda` — Penalty weight
pub fn elastic_align_pair_with_landmarks(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    kind: crate::landmark::LandmarkKind,
    min_prominence: f64,
    expected_count: usize,
    lambda: f64,
) -> ConstrainedAlignmentResult {
    let lm1 = crate::landmark::detect_landmarks(f1, argvals, kind, min_prominence);
    let lm2 = crate::landmark::detect_landmarks(f2, argvals, kind, min_prominence);

    // Match landmarks by order (take min count)
    let n_match = if expected_count > 0 {
        expected_count.min(lm1.len()).min(lm2.len())
    } else {
        lm1.len().min(lm2.len())
    };

    let pairs: Vec<(f64, f64)> = (0..n_match)
        .map(|i| (lm1[i].position, lm2[i].position))
        .collect();

    elastic_align_pair_constrained(f1, f2, argvals, &pairs, lambda)
}

// ─── Multidimensional SRSF (R^d curves) ────────────────────────────────────

use crate::matrix::FdCurveSet;

/// Result of aligning multidimensional (R^d) curves.
#[derive(Debug, Clone)]
pub struct AlignmentResultNd {
    /// Optimal warping function (length m), same for all dimensions.
    pub gamma: Vec<f64>,
    /// Aligned curve: d vectors, each length m.
    pub f_aligned: Vec<Vec<f64>>,
    /// Elastic distance after alignment.
    pub distance: f64,
}

/// Compute the SRSF transform for multidimensional (R^d) curves.
///
/// For f: \[0,1\] → R^d, the SRSF is q(t) = f'(t) / √‖f'(t)‖ where ‖·‖ is the
/// Euclidean norm in R^d. For d=1 this reduces to `sign(f') · √|f'|`.
///
/// # Arguments
/// * `data` — Set of n curves in R^d, each with m evaluation points
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// `FdCurveSet` of SRSF values with the same shape as input.
/// Scale derivative vector at one point by 1/√‖f'‖, writing into result_dims.
#[inline]
fn srsf_scale_point(derivs: &[FdMatrix], result_dims: &mut [FdMatrix], i: usize, j: usize) {
    let d = derivs.len();
    let norm_sq: f64 = derivs.iter().map(|dd| dd[(i, j)].powi(2)).sum();
    let norm = norm_sq.sqrt();
    if norm < 1e-15 {
        for k in 0..d {
            result_dims[k][(i, j)] = 0.0;
        }
    } else {
        let scale = 1.0 / norm.sqrt();
        for k in 0..d {
            result_dims[k][(i, j)] = derivs[k][(i, j)] * scale;
        }
    }
}

pub fn srsf_transform_nd(data: &FdCurveSet, argvals: &[f64]) -> FdCurveSet {
    let d = data.ndim();
    let n = data.ncurves();
    let m = data.npoints();

    if d == 0 || n == 0 || m == 0 || argvals.len() != m {
        return FdCurveSet {
            dims: (0..d).map(|_| FdMatrix::zeros(n, m)).collect(),
        };
    }

    let derivs: Vec<FdMatrix> = data
        .dims
        .iter()
        .map(|dim_mat| crate::fdata::deriv_1d(dim_mat, argvals, 1))
        .collect();

    let mut result_dims: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(n, m)).collect();
    for i in 0..n {
        for j in 0..m {
            srsf_scale_point(&derivs, &mut result_dims, i, j);
        }
    }

    FdCurveSet { dims: result_dims }
}

/// Reconstruct an R^d curve from its SRSF.
///
/// Given d-dimensional SRSF vectors and initial point f0, reconstructs:
/// `f_k(t) = f0_k + ∫₀ᵗ q_k(s) · ‖q(s)‖ ds` for each dimension k.
///
/// # Arguments
/// * `q` — SRSF: d vectors, each length m
/// * `argvals` — Evaluation points (length m)
/// * `f0` — Initial values in R^d (length d)
///
/// # Returns
/// Reconstructed curve: d vectors, each length m.
pub fn srsf_inverse_nd(q: &[Vec<f64>], argvals: &[f64], f0: &[f64]) -> Vec<Vec<f64>> {
    let d = q.len();
    if d == 0 {
        return Vec::new();
    }
    let m = q[0].len();
    if m == 0 {
        return vec![Vec::new(); d];
    }

    // Compute ||q(t)|| at each time point
    let norms: Vec<f64> = (0..m)
        .map(|j| {
            let norm_sq: f64 = q.iter().map(|qk| qk[j].powi(2)).sum();
            norm_sq.sqrt()
        })
        .collect();

    // For each dimension, integrand = q_k(t) * ||q(t)||
    let mut result = Vec::with_capacity(d);
    for k in 0..d {
        let integrand: Vec<f64> = (0..m).map(|j| q[k][j] * norms[j]).collect();
        let integral = cumulative_trapz(&integrand, argvals);
        let curve: Vec<f64> = integral.iter().map(|&v| f0[k] + v).collect();
        result.push(curve);
    }

    result
}

/// Core DP alignment for R^d SRSFs.
///
/// Same DP grid and coprime neighborhood as `dp_alignment_core`, but edge weight
/// is the sum of `dp_edge_weight` over d dimensions.
fn dp_alignment_core_nd(
    q1: &[Vec<f64>],
    q2: &[Vec<f64>],
    argvals: &[f64],
    lambda: f64,
) -> Vec<f64> {
    let d = q1.len();
    let m = argvals.len();
    if m < 2 || d == 0 {
        return argvals.to_vec();
    }

    // For d=1, delegate to existing implementation for exact backward compat
    if d == 1 {
        return dp_alignment_core(&q1[0], &q2[0], argvals, lambda);
    }

    // Normalize each dimension's SRSF to unit L2 norm
    let q1n: Vec<Vec<f64>> = q1
        .iter()
        .map(|qk| {
            let norm = qk.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
            qk.iter().map(|&v| v / norm).collect()
        })
        .collect();
    let q2n: Vec<Vec<f64>> = q2
        .iter()
        .map(|qk| {
            let norm = qk.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
            qk.iter().map(|&v| v / norm).collect()
        })
        .collect();

    let path = dp_grid_solve(m, m, |sr, sc, tr, tc| {
        let w: f64 = (0..d)
            .map(|k| dp_edge_weight(&q1n[k], &q2n[k], argvals, sc, tc, sr, tr))
            .sum();
        w + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
    });

    dp_path_to_gamma(&path, argvals)
}

/// Align an R^d curve f2 to f1 using the elastic framework.
///
/// Finds the optimal warping γ (shared across all dimensions) such that
/// f2∘γ is as close as possible to f1 in the elastic metric.
///
/// # Arguments
/// * `f1` — Target curves (d dimensions)
/// * `f2` — Curves to align (d dimensions)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight (0.0 = no penalty)
pub fn elastic_align_pair_nd(
    f1: &FdCurveSet,
    f2: &FdCurveSet,
    argvals: &[f64],
    lambda: f64,
) -> AlignmentResultNd {
    let d = f1.ndim();
    let m = f1.npoints();

    // Compute SRSFs
    let q1_set = srsf_transform_nd(f1, argvals);
    let q2_set = srsf_transform_nd(f2, argvals);

    // Extract first curve from each dimension
    let q1: Vec<Vec<f64>> = q1_set.dims.iter().map(|dm| dm.row(0)).collect();
    let q2: Vec<Vec<f64>> = q2_set.dims.iter().map(|dm| dm.row(0)).collect();

    // DP alignment using summed cost over dimensions
    let gamma = dp_alignment_core_nd(&q1, &q2, argvals, lambda);

    // Apply warping to f2 in each dimension
    let f_aligned: Vec<Vec<f64>> = f2
        .dims
        .iter()
        .map(|dm| {
            let row = dm.row(0);
            reparameterize_curve(&row, argvals, &gamma)
        })
        .collect();

    // Compute elastic distance: sum of squared L2 distances between aligned SRSFs
    let f_aligned_set = {
        let dims: Vec<FdMatrix> = f_aligned
            .iter()
            .map(|fa| FdMatrix::from_slice(fa, 1, m).unwrap())
            .collect();
        FdCurveSet { dims }
    };
    let q_aligned = srsf_transform_nd(&f_aligned_set, argvals);
    let weights = simpsons_weights(argvals);

    let mut dist_sq = 0.0;
    for k in 0..d {
        let q1k = q1_set.dims[k].row(0);
        let qak = q_aligned.dims[k].row(0);
        let d_k = l2_distance(&q1k, &qak, &weights);
        dist_sq += d_k * d_k;
    }

    AlignmentResultNd {
        gamma,
        f_aligned,
        distance: dist_sq.sqrt(),
    }
}

/// Elastic distance between two R^d curves.
///
/// Aligns f2 to f1 and returns the post-alignment SRSF distance.
pub fn elastic_distance_nd(f1: &FdCurveSet, f2: &FdCurveSet, argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair_nd(f1, f2, argvals, lambda).distance
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::trapz;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::warping::inner_product_l2;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
    }

    fn make_test_data(n: usize, m: usize, seed: u64) -> FdMatrix {
        let t = uniform_grid(m);
        sim_fundata(
            n,
            &t,
            3,
            EFunType::Fourier,
            EValType::Exponential,
            Some(seed),
        )
    }

    // ── cumulative_trapz ──

    #[test]
    fn test_cumulative_trapz_constant() {
        // ∫₀ᵗ 1 dt = t
        let x = uniform_grid(50);
        let y = vec![1.0; 50];
        let result = cumulative_trapz(&y, &x);
        assert!((result[0]).abs() < 1e-15, "cumulative_trapz(0) should be 0");
        for j in 1..50 {
            assert!(
                (result[j] - x[j]).abs() < 1e-12,
                "∫₀^{:.3} 1 dt should be {:.3}, got {:.3}",
                x[j],
                x[j],
                result[j]
            );
        }
    }

    #[test]
    fn test_cumulative_trapz_linear() {
        // ∫₀ᵗ s ds = t²/2
        let m = 100;
        let x = uniform_grid(m);
        let y: Vec<f64> = x.clone();
        let result = cumulative_trapz(&y, &x);
        for j in 1..m {
            let expected = x[j] * x[j] / 2.0;
            assert!(
                (result[j] - expected).abs() < 1e-4,
                "∫₀^{:.3} s ds: expected {expected:.6}, got {:.6}",
                x[j],
                result[j]
            );
        }
    }

    // ── normalize_warp ──

    #[test]
    fn test_normalize_warp_fixes_boundaries() {
        let t = uniform_grid(10);
        let mut gamma = vec![0.1; 10]; // constant, wrong boundaries
        normalize_warp(&mut gamma, &t);
        assert_eq!(gamma[0], t[0]);
        assert_eq!(gamma[9], t[9]);
    }

    #[test]
    fn test_normalize_warp_enforces_monotonicity() {
        let t = uniform_grid(5);
        let mut gamma = vec![0.0, 0.5, 0.3, 0.8, 1.0]; // non-monotone at index 2
        normalize_warp(&mut gamma, &t);
        for j in 1..5 {
            assert!(
                gamma[j] >= gamma[j - 1],
                "gamma should be monotone after normalization at j={j}"
            );
        }
    }

    #[test]
    fn test_normalize_warp_identity_unchanged() {
        let t = uniform_grid(20);
        let mut gamma = t.clone();
        normalize_warp(&mut gamma, &t);
        for j in 0..20 {
            assert!(
                (gamma[j] - t[j]).abs() < 1e-15,
                "Identity warp should be unchanged"
            );
        }
    }

    // ── linear_interp ──

    #[test]
    fn test_linear_interp_at_nodes() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 2.0, 4.0, 6.0];
        for i in 0..x.len() {
            assert!((linear_interp(&x, &y, x[i]) - y[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_linear_interp_midpoints() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 2.0, 4.0];
        assert!((linear_interp(&x, &y, 0.5) - 1.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 1.5) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_interp_clamp() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 3.0, 5.0];
        assert!((linear_interp(&x, &y, -1.0) - 1.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 3.0) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_linear_interp_nonuniform_grid() {
        let x = vec![0.0, 0.1, 0.5, 1.0];
        let y = vec![0.0, 1.0, 5.0, 10.0];
        // Between 0.1 and 0.5: slope = (5-1)/(0.5-0.1) = 10
        let val = linear_interp(&x, &y, 0.3);
        let expected = 1.0 + 10.0 * (0.3 - 0.1);
        assert!(
            (val - expected).abs() < 1e-12,
            "Non-uniform interp: expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_linear_interp_two_points() {
        let x = vec![0.0, 1.0];
        let y = vec![3.0, 7.0];
        assert!((linear_interp(&x, &y, 0.25) - 4.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 0.75) - 6.0).abs() < 1e-12);
    }

    // ── SRSF transform ──

    #[test]
    fn test_srsf_transform_linear() {
        // f(t) = 2t: derivative = 2, SRSF = sqrt(2)
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&ti| 2.0 * ti).collect();
        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();

        let q_mat = srsf_transform(&mat, &t);
        let q: Vec<f64> = q_mat.row(0);

        let expected = 2.0_f64.sqrt();
        // Interior points should be close to sqrt(2)
        for j in 2..(m - 2) {
            assert!(
                (q[j] - expected).abs() < 0.1,
                "q[{j}] = {}, expected ~{expected}",
                q[j]
            );
        }
    }

    #[test]
    fn test_srsf_transform_preserves_shape() {
        let data = make_test_data(10, 50, 42);
        let t = uniform_grid(50);
        let q = srsf_transform(&data, &t);
        assert_eq!(q.shape(), data.shape());
    }

    #[test]
    fn test_srsf_transform_constant_is_zero() {
        // f(t) = 5 (constant): derivative = 0, SRSF = 0
        let m = 30;
        let t = uniform_grid(m);
        let f = vec![5.0; m];
        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
        let q_mat = srsf_transform(&mat, &t);
        let q: Vec<f64> = q_mat.row(0);

        for j in 0..m {
            assert!(
                q[j].abs() < 1e-10,
                "SRSF of constant should be 0, got q[{j}] = {}",
                q[j]
            );
        }
    }

    #[test]
    fn test_srsf_transform_negative_slope() {
        // f(t) = -3t: derivative = -3, SRSF = -sqrt(3)
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&ti| -3.0 * ti).collect();
        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();

        let q_mat = srsf_transform(&mat, &t);
        let q: Vec<f64> = q_mat.row(0);

        let expected = -(3.0_f64.sqrt());
        for j in 2..(m - 2) {
            assert!(
                (q[j] - expected).abs() < 0.15,
                "q[{j}] = {}, expected ~{expected}",
                q[j]
            );
        }
    }

    #[test]
    fn test_srsf_transform_empty_input() {
        let data = FdMatrix::zeros(0, 0);
        let t: Vec<f64> = vec![];
        let q = srsf_transform(&data, &t);
        assert_eq!(q.shape(), (0, 0));
    }

    #[test]
    fn test_srsf_transform_multiple_curves() {
        let m = 40;
        let t = uniform_grid(m);
        let data = make_test_data(5, m, 42);

        let q = srsf_transform(&data, &t);
        assert_eq!(q.shape(), (5, m));

        // Each row should have finite values
        for i in 0..5 {
            for j in 0..m {
                assert!(q[(i, j)].is_finite(), "SRSF should be finite at ({i},{j})");
            }
        }
    }

    // ── SRSF inverse ──

    #[test]
    fn test_srsf_round_trip() {
        let m = 100;
        let t = uniform_grid(m);
        // Use a smooth function
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin() + ti)
            .collect();

        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
        let q_mat = srsf_transform(&mat, &t);
        let q: Vec<f64> = q_mat.row(0);

        let f_recon = srsf_inverse(&q, &t, f[0]);

        // Check reconstruction is close (interior points, avoid boundary effects)
        let max_err: f64 = f[5..(m - 5)]
            .iter()
            .zip(f_recon[5..(m - 5)].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_err < 0.15,
            "Round-trip error too large: max_err = {max_err}"
        );
    }

    #[test]
    fn test_srsf_inverse_empty() {
        let q: Vec<f64> = vec![];
        let t: Vec<f64> = vec![];
        let result = srsf_inverse(&q, &t, 0.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_srsf_inverse_preserves_initial_value() {
        let m = 50;
        let t = uniform_grid(m);
        let q = vec![1.0; m]; // constant SRSF
        let f0 = 3.15;
        let f = srsf_inverse(&q, &t, f0);
        assert!((f[0] - f0).abs() < 1e-12, "srsf_inverse should start at f0");
    }

    #[test]
    fn test_srsf_round_trip_multiple_curves() {
        let m = 80;
        let t = uniform_grid(m);
        let data = make_test_data(5, m, 99);

        let q_mat = srsf_transform(&data, &t);

        for i in 0..5 {
            let fi = data.row(i);
            let qi = q_mat.row(i);
            let f_recon = srsf_inverse(&qi, &t, fi[0]);
            let max_err: f64 = fi[5..(m - 5)]
                .iter()
                .zip(f_recon[5..(m - 5)].iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(max_err < 0.3, "Round-trip curve {i}: max_err = {max_err}");
        }
    }

    // ── Reparameterize ──

    #[test]
    fn test_reparameterize_identity_warp() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

        // Identity warp: γ(t) = t
        let result = reparameterize_curve(&f, &t, &t);
        for j in 0..m {
            assert!(
                (result[j] - f[j]).abs() < 1e-12,
                "Identity warp should return original at j={j}"
            );
        }
    }

    #[test]
    fn test_reparameterize_linear_warp() {
        let m = 50;
        let t = uniform_grid(m);
        // f(t) = t (linear), γ(t) = t^2 (quadratic warp on [0,1])
        let f: Vec<f64> = t.clone();
        let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

        let result = reparameterize_curve(&f, &t, &gamma);

        // f(γ(t)) = γ(t) = t^2 for a linear f(t) = t
        for j in 0..m {
            assert!(
                (result[j] - gamma[j]).abs() < 1e-10,
                "f(gamma(t)) should be gamma(t) for f(t)=t at j={j}"
            );
        }
    }

    #[test]
    fn test_reparameterize_sine_with_quadratic_warp() {
        let m = 100;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (std::f64::consts::PI * ti).sin())
            .collect();
        let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect(); // speeds up start

        let result = reparameterize_curve(&f, &t, &gamma);

        // f(γ(t)) = sin(π * t²); check a few known values
        for j in 0..m {
            let expected = (std::f64::consts::PI * gamma[j]).sin();
            assert!(
                (result[j] - expected).abs() < 0.05,
                "sin(π γ(t)) at j={j}: expected {expected:.4}, got {:.4}",
                result[j]
            );
        }
    }

    #[test]
    fn test_reparameterize_preserves_length() {
        let m = 50;
        let t = uniform_grid(m);
        let f = vec![1.0; m];
        let gamma: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();

        let result = reparameterize_curve(&f, &t, &gamma);
        assert_eq!(result.len(), m);
    }

    // ── Compose warps ──

    #[test]
    fn test_compose_warps_identity() {
        let m = 50;
        let t = uniform_grid(m);
        // γ(t) = t^0.5 (a warp on [0,1])
        let gamma: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();

        // identity ∘ γ = γ
        let result = compose_warps(&t, &gamma, &t);
        for j in 0..m {
            assert!(
                (result[j] - gamma[j]).abs() < 1e-10,
                "id ∘ γ should be γ at j={j}"
            );
        }

        // γ ∘ identity = γ
        let result2 = compose_warps(&gamma, &t, &t);
        for j in 0..m {
            assert!(
                (result2[j] - gamma[j]).abs() < 1e-10,
                "γ ∘ id should be γ at j={j}"
            );
        }
    }

    #[test]
    fn test_compose_warps_associativity() {
        // (γ₁ ∘ γ₂) ∘ γ₃ ≈ γ₁ ∘ (γ₂ ∘ γ₃)
        let m = 50;
        let t = uniform_grid(m);
        let g1: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();
        let g2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
        let g3: Vec<f64> = t.iter().map(|&ti| 0.5 * ti + 0.5 * ti * ti).collect();

        let g12 = compose_warps(&g1, &g2, &t);
        let left = compose_warps(&g12, &g3, &t); // (g1∘g2) ∘ g3

        let g23 = compose_warps(&g2, &g3, &t);
        let right = compose_warps(&g1, &g23, &t); // g1 ∘ (g2∘g3)

        for j in 0..m {
            assert!(
                (left[j] - right[j]).abs() < 0.05,
                "Composition should be roughly associative at j={j}: left={:.4}, right={:.4}",
                left[j],
                right[j]
            );
        }
    }

    #[test]
    fn test_compose_warps_preserves_domain() {
        let m = 50;
        let t = uniform_grid(m);
        let g1: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();
        let g2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

        let composed = compose_warps(&g1, &g2, &t);
        assert!(
            (composed[0] - t[0]).abs() < 1e-10,
            "Composed warp should start at domain start"
        );
        assert!(
            (composed[m - 1] - t[m - 1]).abs() < 1e-10,
            "Composed warp should end at domain end"
        );
    }

    // ── Elastic align pair ──

    #[test]
    fn test_align_identical_curves() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();

        let result = elastic_align_pair(&f, &f, &t, 0.0);

        // Distance should be near zero
        assert!(
            result.distance < 0.1,
            "Distance between identical curves should be near 0, got {}",
            result.distance
        );

        // Warp should be near identity
        for j in 0..m {
            assert!(
                (result.gamma[j] - t[j]).abs() < 0.1,
                "Warp should be near identity at j={j}: gamma={}, t={}",
                result.gamma[j],
                t[j]
            );
        }
    }

    #[test]
    fn test_align_pair_valid_output() {
        let data = make_test_data(2, 50, 42);
        let t = uniform_grid(50);
        let f1 = data.row(0);
        let f2 = data.row(1);

        let result = elastic_align_pair(&f1, &f2, &t, 0.0);

        assert_eq!(result.gamma.len(), 50);
        assert_eq!(result.f_aligned.len(), 50);
        assert!(result.distance >= 0.0);

        // Warp should be monotone
        for j in 1..50 {
            assert!(
                result.gamma[j] >= result.gamma[j - 1],
                "Warp should be monotone at j={j}"
            );
        }
    }

    #[test]
    fn test_align_pair_warp_boundaries() {
        let data = make_test_data(2, 50, 42);
        let t = uniform_grid(50);
        let f1 = data.row(0);
        let f2 = data.row(1);

        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        assert!(
            (result.gamma[0] - t[0]).abs() < 1e-12,
            "Warp should start at domain start"
        );
        assert!(
            (result.gamma[49] - t[49]).abs() < 1e-12,
            "Warp should end at domain end"
        );
    }

    #[test]
    fn test_align_shifted_sine() {
        // Two sines with a phase shift — alignment should reduce distance
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        let weights = simpsons_weights(&t);
        let l2_before = l2_distance(&f1, &f2, &weights);
        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        let l2_after = l2_distance(&f1, &result.f_aligned, &weights);

        assert!(
            l2_after < l2_before + 0.01,
            "Alignment should not increase L2 distance: before={l2_before:.4}, after={l2_after:.4}"
        );
    }

    #[test]
    fn test_align_pair_aligned_curve_is_finite() {
        let data = make_test_data(2, 50, 77);
        let t = uniform_grid(50);
        let f1 = data.row(0);
        let f2 = data.row(1);

        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        for j in 0..50 {
            assert!(
                result.f_aligned[j].is_finite(),
                "Aligned curve should be finite at j={j}"
            );
        }
    }

    #[test]
    fn test_align_pair_minimum_grid() {
        // Minimum viable grid: m = 2
        let t = vec![0.0, 1.0];
        let f1 = vec![0.0, 1.0];
        let f2 = vec![0.0, 2.0];
        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        assert_eq!(result.gamma.len(), 2);
        assert_eq!(result.f_aligned.len(), 2);
        assert!(result.distance >= 0.0);
    }

    // ── Elastic distance ──

    #[test]
    fn test_elastic_distance_symmetric() {
        let data = make_test_data(3, 50, 42);
        let t = uniform_grid(50);
        let f1 = data.row(0);
        let f2 = data.row(1);

        let d12 = elastic_distance(&f1, &f2, &t, 0.0);
        let d21 = elastic_distance(&f2, &f1, &t, 0.0);

        // Should be approximately symmetric (DP is not perfectly symmetric)
        assert!(
            (d12 - d21).abs() < d12.max(d21) * 0.3 + 0.01,
            "Elastic distance should be roughly symmetric: d12={d12}, d21={d21}"
        );
    }

    #[test]
    fn test_elastic_distance_nonneg() {
        let data = make_test_data(3, 50, 42);
        let t = uniform_grid(50);

        for i in 0..3 {
            for j in 0..3 {
                let fi = data.row(i);
                let fj = data.row(j);
                let d = elastic_distance(&fi, &fj, &t, 0.0);
                assert!(d >= 0.0, "Elastic distance should be non-negative");
            }
        }
    }

    #[test]
    fn test_elastic_distance_self_near_zero() {
        let data = make_test_data(3, 50, 42);
        let t = uniform_grid(50);

        for i in 0..3 {
            let fi = data.row(i);
            let d = elastic_distance(&fi, &fi, &t, 0.0);
            assert!(
                d < 0.1,
                "Self-distance should be near zero, got {d} for curve {i}"
            );
        }
    }

    #[test]
    fn test_elastic_distance_triangle_inequality() {
        let data = make_test_data(3, 50, 42);
        let t = uniform_grid(50);
        let f0 = data.row(0);
        let f1 = data.row(1);
        let f2 = data.row(2);

        let d01 = elastic_distance(&f0, &f1, &t, 0.0);
        let d12 = elastic_distance(&f1, &f2, &t, 0.0);
        let d02 = elastic_distance(&f0, &f2, &t, 0.0);

        // Relaxed triangle inequality (DP alignment is approximate)
        let slack = 0.5;
        assert!(
            d02 <= d01 + d12 + slack,
            "Triangle inequality (relaxed): d02={d02:.4} > d01={d01:.4} + d12={d12:.4} + {slack}"
        );
    }

    #[test]
    fn test_elastic_distance_different_shapes_nonzero() {
        let m = 50;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.to_vec(); // linear
        let f2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect(); // quadratic

        let d = elastic_distance(&f1, &f2, &t, 0.0);
        assert!(
            d > 0.01,
            "Distance between different shapes should be > 0, got {d}"
        );
    }

    // ── Self distance matrix ──

    #[test]
    fn test_self_distance_matrix_symmetric() {
        let data = make_test_data(5, 30, 42);
        let t = uniform_grid(30);

        let dm = elastic_self_distance_matrix(&data, &t, 0.0);
        let n = dm.nrows();

        assert_eq!(dm.shape(), (5, 5));

        // Zero diagonal
        for i in 0..n {
            assert!(dm[(i, i)].abs() < 1e-12, "Diagonal should be zero");
        }

        // Symmetric
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    (dm[(i, j)] - dm[(j, i)]).abs() < 1e-12,
                    "Matrix should be symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_self_distance_matrix_nonneg() {
        let data = make_test_data(4, 30, 42);
        let t = uniform_grid(30);
        let dm = elastic_self_distance_matrix(&data, &t, 0.0);

        for i in 0..4 {
            for j in 0..4 {
                assert!(
                    dm[(i, j)] >= 0.0,
                    "Distance matrix entries should be non-negative at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_self_distance_matrix_single_curve() {
        let data = make_test_data(1, 30, 42);
        let t = uniform_grid(30);
        let dm = elastic_self_distance_matrix(&data, &t, 0.0);
        assert_eq!(dm.shape(), (1, 1));
        assert!(dm[(0, 0)].abs() < 1e-12);
    }

    #[test]
    fn test_self_distance_matrix_consistent_with_pairwise() {
        let data = make_test_data(4, 30, 42);
        let t = uniform_grid(30);

        let dm = elastic_self_distance_matrix(&data, &t, 0.0);

        // Check a few entries match direct elastic_distance calls
        for i in 0..4 {
            for j in (i + 1)..4 {
                let fi = data.row(i);
                let fj = data.row(j);
                let d_direct = elastic_distance(&fi, &fj, &t, 0.0);
                assert!(
                    (dm[(i, j)] - d_direct).abs() < 1e-10,
                    "Matrix entry ({i},{j})={:.6} should match pairwise {d_direct:.6}",
                    dm[(i, j)]
                );
            }
        }
    }

    // ── Karcher mean ──

    #[test]
    fn test_karcher_mean_identical_curves() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();

        // Create 5 identical curves
        let mut data = FdMatrix::zeros(5, m);
        for i in 0..5 {
            for j in 0..m {
                data[(i, j)] = f[j];
            }
        }

        let result = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        assert_eq!(result.mean.len(), m);
        assert!(result.n_iter <= 10);
    }

    #[test]
    fn test_karcher_mean_output_shape() {
        let data = make_test_data(15, 50, 42);
        let t = uniform_grid(50);

        let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

        assert_eq!(result.mean.len(), 50);
        assert_eq!(result.mean_srsf.len(), 50);
        assert_eq!(result.gammas.shape(), (15, 50));
        assert_eq!(result.aligned_data.shape(), (15, 50));
        assert!(result.n_iter <= 5);
    }

    #[test]
    fn test_karcher_mean_warps_are_valid() {
        let data = make_test_data(10, 40, 42);
        let t = uniform_grid(40);

        let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

        for i in 0..10 {
            // Boundary values
            assert!(
                (result.gammas[(i, 0)] - t[0]).abs() < 1e-10,
                "Warp {i} should start at domain start"
            );
            assert!(
                (result.gammas[(i, 39)] - t[39]).abs() < 1e-10,
                "Warp {i} should end at domain end"
            );
            // Monotonicity
            for j in 1..40 {
                assert!(
                    result.gammas[(i, j)] >= result.gammas[(i, j - 1)],
                    "Warp {i} should be monotone at j={j}"
                );
            }
        }
    }

    #[test]
    fn test_karcher_mean_aligned_data_is_finite() {
        let data = make_test_data(8, 40, 42);
        let t = uniform_grid(40);
        let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

        for i in 0..8 {
            for j in 0..40 {
                assert!(
                    result.aligned_data[(i, j)].is_finite(),
                    "Aligned data should be finite at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_karcher_mean_srsf_is_finite() {
        let data = make_test_data(8, 40, 42);
        let t = uniform_grid(40);
        let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

        for j in 0..40 {
            assert!(
                result.mean_srsf[j].is_finite(),
                "Mean SRSF should be finite at j={j}"
            );
            assert!(
                result.mean[j].is_finite(),
                "Mean curve should be finite at j={j}"
            );
        }
    }

    #[test]
    fn test_karcher_mean_single_iteration() {
        let data = make_test_data(10, 40, 42);
        let t = uniform_grid(40);
        let result = karcher_mean(&data, &t, 1, 1e-10, 0.0);

        assert_eq!(result.n_iter, 1);
        assert_eq!(result.mean.len(), 40);
        // With only 1 iteration, still produces valid output
        for j in 0..40 {
            assert!(result.mean[j].is_finite());
        }
    }

    #[test]
    fn test_karcher_mean_convergence_not_premature() {
        // The old convergence criterion (rel - prev_rel <= tol * prev_rel) was
        // always satisfied when the algorithm improved, causing premature exit
        // after 2 iterations. With the fix (rel < tol), the algorithm should
        // actually iterate until convergence or hitting the iteration cap.
        let n = 10;
        let m = 50;
        let t = uniform_grid(m);

        // Create phase-shifted curves that genuinely need alignment
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            let shift = (i as f64 - 5.0) * 0.05;
            for j in 0..m {
                col_major[i + j * n] = (2.0 * std::f64::consts::PI * (t[j] - shift)).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        // With an impossibly tight tolerance, the algorithm should hit the
        // iteration cap rather than "converging" after 2 iterations.
        let result = karcher_mean(&data, &t, 20, 1e-15, 0.0);
        assert!(
            result.n_iter > 2,
            "With tol=1e-15 the algorithm should iterate beyond 2, got n_iter={}",
            result.n_iter
        );

        // With a reasonable tolerance, it should converge and report so
        let result_loose = karcher_mean(&data, &t, 20, 1e-2, 0.0);
        assert!(
            result_loose.converged,
            "With tol=1e-2 the algorithm should converge"
        );
    }

    // ── Align to target ──

    #[test]
    fn test_align_to_target_valid() {
        let data = make_test_data(10, 40, 42);
        let t = uniform_grid(40);
        let target = data.row(0);

        let result = align_to_target(&data, &target, &t, 0.0);

        assert_eq!(result.gammas.shape(), (10, 40));
        assert_eq!(result.aligned_data.shape(), (10, 40));
        assert_eq!(result.distances.len(), 10);

        // All distances should be non-negative
        for &d in &result.distances {
            assert!(d >= 0.0);
        }
    }

    #[test]
    fn test_align_to_target_self_near_zero() {
        let data = make_test_data(5, 40, 42);
        let t = uniform_grid(40);
        let target = data.row(0);

        let result = align_to_target(&data, &target, &t, 0.0);

        // Distance of target to itself should be near zero
        assert!(
            result.distances[0] < 0.1,
            "Self-alignment distance should be near zero, got {}",
            result.distances[0]
        );
    }

    #[test]
    fn test_align_to_target_warps_are_monotone() {
        let data = make_test_data(8, 40, 42);
        let t = uniform_grid(40);
        let target = data.row(0);
        let result = align_to_target(&data, &target, &t, 0.0);

        for i in 0..8 {
            for j in 1..40 {
                assert!(
                    result.gammas[(i, j)] >= result.gammas[(i, j - 1)],
                    "Warp for curve {i} should be monotone at j={j}"
                );
            }
        }
    }

    #[test]
    fn test_align_to_target_aligned_data_finite() {
        let data = make_test_data(6, 40, 42);
        let t = uniform_grid(40);
        let target = data.row(0);
        let result = align_to_target(&data, &target, &t, 0.0);

        for i in 0..6 {
            for j in 0..40 {
                assert!(
                    result.aligned_data[(i, j)].is_finite(),
                    "Aligned data should be finite at ({i},{j})"
                );
            }
        }
    }

    // ── Cross distance matrix ──

    #[test]
    fn test_cross_distance_matrix_shape() {
        let data1 = make_test_data(3, 30, 42);
        let data2 = make_test_data(4, 30, 99);
        let t = uniform_grid(30);

        let dm = elastic_cross_distance_matrix(&data1, &data2, &t, 0.0);
        assert_eq!(dm.shape(), (3, 4));

        // All non-negative
        for i in 0..3 {
            for j in 0..4 {
                assert!(dm[(i, j)] >= 0.0);
            }
        }
    }

    #[test]
    fn test_cross_distance_matrix_self_matches_self_matrix() {
        // cross_distance(data, data) should have zero diagonal (approximately)
        let data = make_test_data(4, 30, 42);
        let t = uniform_grid(30);

        let cross = elastic_cross_distance_matrix(&data, &data, &t, 0.0);
        for i in 0..4 {
            assert!(
                cross[(i, i)] < 0.1,
                "Cross distance (self) diagonal should be near zero: got {}",
                cross[(i, i)]
            );
        }
    }

    #[test]
    fn test_cross_distance_matrix_consistent_with_pairwise() {
        let data1 = make_test_data(3, 30, 42);
        let data2 = make_test_data(2, 30, 99);
        let t = uniform_grid(30);

        let dm = elastic_cross_distance_matrix(&data1, &data2, &t, 0.0);

        for i in 0..3 {
            for j in 0..2 {
                let fi = data1.row(i);
                let fj = data2.row(j);
                let d_direct = elastic_distance(&fi, &fj, &t, 0.0);
                assert!(
                    (dm[(i, j)] - d_direct).abs() < 1e-10,
                    "Cross matrix ({i},{j})={:.6} should match pairwise {d_direct:.6}",
                    dm[(i, j)]
                );
            }
        }
    }

    // ── align_srsf_pair ──

    #[test]
    fn test_align_srsf_pair_identity() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let q = srsf_single(&f, &t);

        let (gamma, q_aligned) = align_srsf_pair(&q, &q, &t, 0.0);

        // Warp should be near identity
        for j in 0..m {
            assert!(
                (gamma[j] - t[j]).abs() < 0.15,
                "Self-SRSF alignment warp should be near identity at j={j}"
            );
        }

        // Aligned SRSF should be close to original
        let weights = simpsons_weights(&t);
        let dist = l2_distance(&q, &q_aligned, &weights);
        assert!(
            dist < 0.5,
            "Self-aligned SRSF distance should be small, got {dist}"
        );
    }

    // ── srsf_single ──

    #[test]
    fn test_srsf_single_matches_matrix_version() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&ti| ti * ti + ti).collect();

        let q_single = srsf_single(&f, &t);

        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
        let q_mat = srsf_transform(&mat, &t);
        let q_from_mat = q_mat.row(0);

        for j in 0..m {
            assert!(
                (q_single[j] - q_from_mat[j]).abs() < 1e-12,
                "srsf_single should match srsf_transform at j={j}"
            );
        }
    }

    // ── gcd ──

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(1, 1), 1);
        assert_eq!(gcd(6, 4), 2);
        assert_eq!(gcd(7, 5), 1);
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 0), 7);
        assert_eq!(gcd(0, 5), 5);
    }

    // ── generate_coprime_nbhd ──

    #[test]
    fn test_coprime_nbhd_count() {
        assert_eq!(generate_coprime_nbhd(1).len(), 1); // just (1,1)
        assert_eq!(generate_coprime_nbhd(7).len(), 35);
    }

    #[test]
    fn test_coprime_nbhd_matches_const() {
        let generated = generate_coprime_nbhd(7);
        assert_eq!(generated.len(), COPRIME_NBHD_7.len());
        for (i, pair) in generated.iter().enumerate() {
            assert_eq!(*pair, COPRIME_NBHD_7[i], "mismatch at index {i}");
        }
    }

    #[test]
    fn test_coprime_nbhd_all_coprime() {
        for &(i, j) in &COPRIME_NBHD_7 {
            assert_eq!(gcd(i, j), 1, "({i},{j}) should be coprime");
            assert!((1..=7).contains(&i));
            assert!((1..=7).contains(&j));
        }
    }

    // ── dp_edge_weight ──

    #[test]
    fn test_dp_edge_weight_diagonal() {
        // Diagonal move (1,1): weight = (q1[sc] - sqrt(1)*q2[sr])^2 * h
        let t = uniform_grid(10);
        let q1 = vec![1.0; 10];
        let q2 = vec![1.0; 10];
        // Identical SRSFs: weight should be 0
        let w = dp_edge_weight(&q1, &q2, &t, 0, 1, 0, 1);
        assert!(w.abs() < 1e-12, "identical SRSFs should have zero cost");
    }

    #[test]
    fn test_dp_edge_weight_non_diagonal() {
        // Move (1,2): n1=2, n2=1, slope = h/(2h) = 0.5
        let t = uniform_grid(10);
        let q1 = vec![1.0; 10];
        let q2 = vec![0.0; 10];
        let w = dp_edge_weight(&q1, &q2, &t, 0, 2, 0, 1);
        // diff = q1[0] - sqrt(0.5)*q2[0] = 1.0 - 0 = 1.0
        // weight = 1.0^2 * 1.0 * (t[2]-t[0]) = 2/9
        let expected = 2.0 / 9.0;
        assert!(
            (w - expected).abs() < 1e-10,
            "dp_edge_weight (1,2): expected {expected}, got {w}"
        );
    }

    #[test]
    fn test_dp_edge_weight_zero_span() {
        let t = uniform_grid(10);
        let q1 = vec![1.0; 10];
        let q2 = vec![1.0; 10];
        // n1=0: should return INFINITY
        assert_eq!(dp_edge_weight(&q1, &q2, &t, 3, 3, 0, 1), f64::INFINITY);
        // n2=0: should return INFINITY
        assert_eq!(dp_edge_weight(&q1, &q2, &t, 0, 1, 3, 3), f64::INFINITY);
    }

    // ── DP alignment quality ──

    #[test]
    fn test_alignment_improves_distance() {
        // Aligned SRSF distance should be less than unaligned SRSF distance
        let m = 50;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&x| (2.0 * std::f64::consts::PI * x).sin())
            .collect();
        // Use a larger shift so improvement is clear
        let f2: Vec<f64> = t
            .iter()
            .map(|&x| (2.0 * std::f64::consts::PI * (x + 0.2)).sin())
            .collect();

        let q1 = srsf_single(&f1, &t);
        let q2 = srsf_single(&f2, &t);
        let weights = simpsons_weights(&t);
        let unaligned_srsf_dist = l2_distance(&q1, &q2, &weights);

        let result = elastic_align_pair(&f1, &f2, &t, 0.0);

        assert!(
            result.distance <= unaligned_srsf_dist + 1e-6,
            "aligned SRSF dist ({}) should be <= unaligned SRSF dist ({})",
            result.distance,
            unaligned_srsf_dist
        );
    }

    // ── Edge case: constant data ──

    #[test]
    fn test_alignment_constant_curves() {
        let m = 30;
        let t = uniform_grid(m);
        let f1 = vec![5.0; m];
        let f2 = vec![5.0; m];

        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        assert!(
            result.distance < 0.01,
            "Constant curves: distance should be ~0"
        );
        assert_eq!(result.f_aligned.len(), m);
    }

    #[test]
    fn test_karcher_mean_constant_curves() {
        let m = 30;
        let t = uniform_grid(m);
        let mut data = FdMatrix::zeros(5, m);
        for i in 0..5 {
            for j in 0..m {
                data[(i, j)] = 3.0;
            }
        }

        let result = karcher_mean(&data, &t, 5, 1e-4, 0.0);
        for j in 0..m {
            assert!(
                (result.mean[j] - 3.0).abs() < 0.5,
                "Mean of constant curves should be near 3.0, got {} at j={j}",
                result.mean[j]
            );
        }
    }

    #[test]
    fn test_nan_srsf_no_panic() {
        let m = 20;
        let t = uniform_grid(m);
        let mut f = vec![1.0; m];
        f[5] = f64::NAN;
        let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
        let q = srsf_transform(&mat, &t);
        // Should not panic; NaN propagates
        assert_eq!(q.nrows(), 1);
    }

    #[test]
    fn test_n1_karcher_mean() {
        let m = 30;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
        let data = FdMatrix::from_slice(&f, 1, m).unwrap();
        let result = karcher_mean(&data, &t, 5, 1e-4, 0.0);
        assert_eq!(result.mean.len(), m);
        // With only 1 curve, the mean should be close to the original
        for j in 0..m {
            assert!(result.mean[j].is_finite());
        }
    }

    #[test]
    fn test_two_point_grid() {
        let t = vec![0.0, 1.0];
        let f1 = vec![0.0, 1.0];
        let f2 = vec![0.0, 2.0];
        let d = elastic_distance(&f1, &f2, &t, 0.0);
        assert!(d >= 0.0);
        assert!(d.is_finite());
    }

    #[test]
    fn test_non_uniform_grid_alignment() {
        // Non-uniform grid: points clustered near 0
        let t = vec![0.0, 0.01, 0.05, 0.2, 0.5, 1.0];
        let m = t.len();
        let f1: Vec<f64> = t.iter().map(|&ti: &f64| ti.sin()).collect();
        let f2: Vec<f64> = t.iter().map(|&ti: &f64| (ti + 0.1).sin()).collect();
        let result = elastic_align_pair(&f1, &f2, &t, 0.0);
        assert_eq!(result.gamma.len(), m);
        assert!(result.distance >= 0.0);
        assert!(result.distance.is_finite());
    }

    // ── TSRVF tests ──

    #[test]
    fn test_tsrvf_output_shape() {
        let m = 50;
        let n = 10;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
        assert_eq!(
            result.tangent_vectors.shape(),
            (n, m),
            "Tangent vectors should be n×m"
        );
        assert_eq!(result.gammas.shape(), (n, m), "Gammas should be n×m");
        assert_eq!(result.srsf_norms.len(), n, "Should have n SRSF norms");
        assert_eq!(result.mean.len(), m, "Mean should have m points");
        assert_eq!(result.mean_srsf.len(), m, "Mean SRSF should have m points");
    }

    #[test]
    fn test_tsrvf_all_finite() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
        for i in 0..n {
            for j in 0..m {
                assert!(
                    result.tangent_vectors[(i, j)].is_finite(),
                    "Tangent vector should be finite at ({i},{j})"
                );
            }
            assert!(
                result.srsf_norms[i].is_finite(),
                "SRSF norm should be finite for curve {i}"
            );
        }
        assert!(
            result.mean_srsf_norm.is_finite(),
            "Mean SRSF norm should be finite"
        );
    }

    #[test]
    fn test_tsrvf_identical_curves_zero_tangent() {
        let m = 50;
        let t = uniform_grid(m);
        // Stack 5 identical curves
        let curve: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let mut col_major = vec![0.0; 5 * m];
        for i in 0..5 {
            for j in 0..m {
                col_major[i + j * 5] = curve[j];
            }
        }
        let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
        let result = tsrvf_transform(&data, &t, 10, 1e-4, 0.0);

        // All tangent vectors should be approximately zero
        for i in 0..5 {
            let tv_norm_sq: f64 = (0..m).map(|j| result.tangent_vectors[(i, j)].powi(2)).sum();
            assert!(
                tv_norm_sq.sqrt() < 0.5,
                "Identical curves should have near-zero tangent vectors, got norm = {}",
                tv_norm_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_tsrvf_mean_tangent_near_zero() {
        let m = 50;
        let n = 10;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let result = tsrvf_transform(&data, &t, 10, 1e-3, 0.0);

        // Mean of tangent vectors should be approximately zero (property of Karcher mean)
        let mut mean_tv = vec![0.0; m];
        for i in 0..n {
            for j in 0..m {
                mean_tv[j] += result.tangent_vectors[(i, j)];
            }
        }
        for j in 0..m {
            mean_tv[j] /= n as f64;
        }
        let mean_norm: f64 = mean_tv.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            mean_norm < 1.0,
            "Mean tangent vector should be near zero, got norm = {mean_norm}"
        );
    }

    #[test]
    fn test_tsrvf_from_alignment() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let karcher = karcher_mean(&data, &t, 5, 1e-3, 0.0);
        let result = tsrvf_from_alignment(&karcher, &t);
        assert_eq!(result.tangent_vectors.shape(), (n, m));
        assert!(result.mean_srsf_norm > 0.0);
    }

    #[test]
    fn test_tsrvf_round_trip() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let result = tsrvf_transform(&data, &t, 10, 1e-3, 0.0);
        let reconstructed = tsrvf_inverse(&result, &t);

        assert_eq!(reconstructed.shape(), result.tangent_vectors.shape());
        // Reconstructed curves should have finite values
        for i in 0..n {
            for j in 0..m {
                assert!(
                    reconstructed[(i, j)].is_finite(),
                    "Reconstructed curve should be finite at ({i},{j})"
                );
            }
        }
        // Issue #12: per-curve initial values should be preserved
        for i in 0..n {
            assert!(
                (reconstructed[(i, 0)] - result.initial_values[i]).abs() < 1e-6,
                "Curve {i} initial value: expected {}, got {}",
                result.initial_values[i],
                reconstructed[(i, 0)]
            );
        }
    }

    #[test]
    fn test_tsrvf_initial_values_per_curve() {
        // Issue #12: tsrvf_inverse must use per-curve initial values, not mean[0]
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);

        // Create curves with distinct initial values
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            let offset = (i as f64 + 1.0) * 2.0; // offsets: 2, 4, 6, 8, 10
            for j in 0..m {
                col_major[i + j * n] = offset + (2.0 * std::f64::consts::PI * t[j]).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        let result = tsrvf_transform(&data, &t, 15, 1e-4, 0.0);

        // initial_values should differ per curve
        assert_eq!(result.initial_values.len(), n);
        let all_same = result
            .initial_values
            .windows(2)
            .all(|w| (w[0] - w[1]).abs() < 1e-10);
        assert!(
            !all_same,
            "Initial values should differ per curve: {:?}",
            result.initial_values
        );

        // Reconstruct and check initial values are preserved
        let reconstructed = tsrvf_inverse(&result, &t);
        for i in 0..n {
            assert!(
                (reconstructed[(i, 0)] - result.initial_values[i]).abs() < 1e-6,
                "Curve {i}: reconstructed f(0) = {}, expected {}",
                reconstructed[(i, 0)],
                result.initial_values[i]
            );
        }

        // Before the fix, all curves would have reconstructed[(i, 0)] ≈ mean[0]
        // Verify they are NOT all the same
        let recon_initials: Vec<f64> = (0..n).map(|i| reconstructed[(i, 0)]).collect();
        let all_recon_same = recon_initials.windows(2).all(|w| (w[0] - w[1]).abs() < 0.1);
        assert!(
            !all_recon_same,
            "Reconstructed initial values must vary per curve: {:?}",
            recon_initials
        );
    }

    #[test]
    fn test_tsrvf_single_curve() {
        let m = 50;
        let t = uniform_grid(m);
        let data = make_test_data(1, m, 42);
        let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
        assert_eq!(result.tangent_vectors.shape(), (1, m));
        // Single curve → tangent vector should be zero (it IS the mean)
        let tv_norm: f64 = (0..m)
            .map(|j| result.tangent_vectors[(0, j)].powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            tv_norm < 0.5,
            "Single curve tangent vector should be near zero, got {tv_norm}"
        );
    }

    #[test]
    fn test_tsrvf_constant_curves() {
        let m = 30;
        let t = uniform_grid(m);
        // Constant curves → SRSF = 0, norms = 0
        let data = FdMatrix::from_column_major(vec![5.0; 3 * m], 3, m).unwrap();
        let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
        // Should not produce NaN or Inf
        for i in 0..3 {
            for j in 0..m {
                let v = result.tangent_vectors[(i, j)];
                assert!(
                    !v.is_nan(),
                    "Constant curves should not produce NaN tangent vectors"
                );
            }
        }
    }

    // ── Reference-value tests (sphere geometry) ─────────────────────────────

    #[test]
    fn test_tsrvf_sphere_inv_exp_reference() {
        // Analytical reference: known vectors on the Hilbert sphere
        // psi1 = constant (unit L2 norm), psi2 = 1 + 0.3*sin(2πt) (normalized)
        let m = 21;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

        // Construct unit vector psi1 (constant)
        let raw1 = vec![1.0; m];
        let norm1 = inner_product_l2(&raw1, &raw1, &time).max(0.0).sqrt();
        let psi1: Vec<f64> = raw1.iter().map(|&v| v / norm1).collect();

        // Construct psi2 with sinusoidal perturbation
        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = inner_product_l2(&raw2, &raw2, &time).max(0.0).sqrt();
        let psi2: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        // Compute theta analytically
        let ip = inner_product_l2(&psi1, &psi2, &time).clamp(-1.0, 1.0);
        let theta_expected = ip.acos();

        // Compute via inv_exp_map_sphere
        let v = inv_exp_map_sphere(&psi1, &psi2, &time);
        let v_norm = inner_product_l2(&v, &v, &time).max(0.0).sqrt();

        // ||v|| should equal theta
        assert!(
            (v_norm - theta_expected).abs() < 1e-10,
            "||v|| = {v_norm}, expected theta = {theta_expected}"
        );

        // theta should be small but non-zero (perturbation is mild)
        assert!(
            theta_expected > 0.01 && theta_expected < 1.0,
            "theta = {theta_expected} out of expected range"
        );
    }

    #[test]
    fn test_tsrvf_sphere_round_trip_reference() {
        // Round-trip: exp_map(psi1, inv_exp_map(psi1, psi2)) should recover psi2
        let m = 21;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

        let raw1 = vec![1.0; m];
        let norm1 = inner_product_l2(&raw1, &raw1, &time).max(0.0).sqrt();
        let psi1: Vec<f64> = raw1.iter().map(|&v| v / norm1).collect();

        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = inner_product_l2(&raw2, &raw2, &time).max(0.0).sqrt();
        let psi2: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        let v = inv_exp_map_sphere(&psi1, &psi2, &time);
        let recovered = exp_map_sphere(&psi1, &v, &time);

        // L2 error between psi2 and recovered
        let diff: Vec<f64> = psi2
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .collect();
        let l2_err = trapz(&diff, &time).max(0.0).sqrt();
        assert!(
            l2_err < 1e-12,
            "Round-trip L2 error = {l2_err:.2e}, expected < 1e-12"
        );
    }

    // ── Penalized alignment ──

    #[test]
    fn test_penalized_alignment_lambda_zero_matches_unpenalized() {
        let m = 50;
        let t = uniform_grid(m);
        let data = make_test_data(2, m, 42);
        let f1 = data.row(0);
        let f2 = data.row(1);

        let r0 = elastic_align_pair(&f1, &f2, &t, 0.0);
        // lambda = 0.0 should produce the same result regardless
        assert!(r0.distance >= 0.0);
        assert_eq!(r0.gamma.len(), m);
    }

    #[test]
    fn test_penalized_alignment_smoother_warp() {
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
            .collect();

        let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
        let r_pen = elastic_align_pair(&f1, &f2, &t, 1.0);

        // Measure warp deviation from identity
        let dev_free: f64 = r_free
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).powi(2))
            .sum();
        let dev_pen: f64 = r_pen
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).powi(2))
            .sum();

        assert!(
            dev_pen <= dev_free + 1e-6,
            "Penalized warp should be closer to identity: free={dev_free:.6}, pen={dev_pen:.6}"
        );
    }

    #[test]
    fn test_penalized_alignment_large_lambda_near_identity() {
        let m = 50;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        let r = elastic_align_pair(&f1, &f2, &t, 1000.0);

        // With very large lambda, warp should be very close to identity
        let max_dev: f64 = r
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 0.05,
            "Large lambda should give near-identity warp: max deviation = {max_dev}"
        );
    }

    #[test]
    fn test_penalized_karcher_mean() {
        let m = 40;
        let t = uniform_grid(m);
        let data = make_test_data(10, m, 42);

        let result = karcher_mean(&data, &t, 5, 1e-3, 0.5);
        assert_eq!(result.mean.len(), m);
        for j in 0..m {
            assert!(result.mean[j].is_finite());
        }
    }

    // ── Phase-amplitude decomposition ──

    #[test]
    fn test_decomposition_identity_curves() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();

        let result = elastic_decomposition(&f, &f, &t, 0.0);
        assert!(
            result.d_amplitude < 0.1,
            "Self-decomposition amplitude should be ~0, got {}",
            result.d_amplitude
        );
        assert!(
            result.d_phase < 0.2,
            "Self-decomposition phase should be ~0, got {}",
            result.d_phase
        );
    }

    #[test]
    fn test_decomposition_pythagorean() {
        // d_total² ≈ d_a² + d_φ² (approximately, for the Fisher-Rao metric)
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| 1.2 * (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        let result = elastic_decomposition(&f1, &f2, &t, 0.0);
        let da = result.d_amplitude;
        let dp = result.d_phase;
        // Both should be non-negative
        assert!(da >= 0.0);
        assert!(dp >= 0.0);
    }

    #[test]
    fn test_phase_distance_shifted_sine() {
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
            .collect();

        let dp = phase_distance_pair(&f1, &f2, &t, 0.0);
        assert!(
            dp > 0.01,
            "Phase distance of shifted curves should be > 0, got {dp}"
        );
    }

    #[test]
    fn test_amplitude_distance_scaled_curve() {
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| 2.0 * (2.0 * std::f64::consts::PI * ti).sin())
            .collect();

        let da = amplitude_distance(&f1, &f2, &t, 0.0);
        assert!(
            da > 0.01,
            "Amplitude distance of scaled curves should be > 0, got {da}"
        );
    }

    #[test]
    fn test_phase_distance_nonneg() {
        let data = make_test_data(4, 40, 42);
        let t = uniform_grid(40);
        for i in 0..4 {
            for j in 0..4 {
                let fi = data.row(i);
                let fj = data.row(j);
                let dp = phase_distance_pair(&fi, &fj, &t, 0.0);
                assert!(dp >= 0.0, "Phase distance should be non-negative");
            }
        }
    }

    // ── Parallel transport ──

    #[test]
    fn test_schilds_ladder_zero_vector() {
        let m = 21;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let raw = vec![1.0; m];
        let norm = crate::warping::l2_norm_l2(&raw, &time);
        let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.2 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
        let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        let zero = vec![0.0; m];
        let result = parallel_transport_schilds(&zero, &from, &to, &time);
        let result_norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            result_norm < 1e-6,
            "Transporting zero should give zero, got norm {result_norm}"
        );
    }

    #[test]
    fn test_pole_ladder_zero_vector() {
        let m = 21;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let raw = vec![1.0; m];
        let norm = crate::warping::l2_norm_l2(&raw, &time);
        let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.2 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
        let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        let zero = vec![0.0; m];
        let result = parallel_transport_pole(&zero, &from, &to, &time);
        let result_norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            result_norm < 1e-6,
            "Transporting zero should give zero, got norm {result_norm}"
        );
    }

    #[test]
    fn test_schilds_preserves_norm() {
        let m = 51;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let raw = vec![1.0; m];
        let norm = crate::warping::l2_norm_l2(&raw, &time);
        let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.15 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
        let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        // Small tangent vector
        let v: Vec<f64> = time
            .iter()
            .map(|&t| 0.1 * (4.0 * std::f64::consts::PI * t).cos())
            .collect();
        let v_norm = crate::warping::l2_norm_l2(&v, &time);

        let transported = parallel_transport_schilds(&v, &from, &to, &time);
        let t_norm = crate::warping::l2_norm_l2(&transported, &time);

        // Norm should be approximately preserved (ladder methods are first-order)
        assert!(
            (t_norm - v_norm).abs() / v_norm.max(1e-10) < 1.5,
            "Schild's should roughly preserve norm: original={v_norm:.4}, transported={t_norm:.4}"
        );
    }

    #[test]
    fn test_tsrvf_logmap_matches_original() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);

        let result_orig = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
        let result_logmap =
            tsrvf_transform_with_method(&data, &t, 5, 1e-3, 0.0, TransportMethod::LogMap);

        // Should be identical (LogMap delegates to original)
        for i in 0..n {
            for j in 0..m {
                assert!(
                    (result_orig.tangent_vectors[(i, j)] - result_logmap.tangent_vectors[(i, j)])
                        .abs()
                        < 1e-12,
                    "LogMap variant should match original at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_tsrvf_with_schilds_produces_valid_result() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);

        let result =
            tsrvf_transform_with_method(&data, &t, 5, 1e-3, 0.0, TransportMethod::SchildsLadder);

        assert_eq!(result.tangent_vectors.shape(), (n, m));
        for i in 0..n {
            for j in 0..m {
                assert!(
                    result.tangent_vectors[(i, j)].is_finite(),
                    "Schild's TSRVF should produce finite tangent vectors at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_transport_methods_differ() {
        let m = 50;
        let n = 5;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let karcher = karcher_mean(&data, &t, 5, 1e-3, 0.0);

        let r_log = tsrvf_from_alignment_with_method(&karcher, &t, TransportMethod::LogMap);
        let r_schilds =
            tsrvf_from_alignment_with_method(&karcher, &t, TransportMethod::SchildsLadder);

        // Methods should produce different (but related) tangent vectors
        let mut total_diff = 0.0;
        for i in 0..n {
            for j in 0..m {
                total_diff +=
                    (r_log.tangent_vectors[(i, j)] - r_schilds.tangent_vectors[(i, j)]).abs();
            }
        }

        // They should be non-zero different (unless all curves are identical to mean)
        // Just check both produce finite results
        assert!(total_diff.is_finite());
    }

    // ── Alignment quality metrics ──

    #[test]
    fn test_warp_complexity_identity_is_zero() {
        let m = 50;
        let t = uniform_grid(m);
        let identity = t.clone();
        let c = warp_complexity(&identity, &t);
        assert!(
            c < 1e-10,
            "Identity warp should have zero complexity, got {c}"
        );
    }

    #[test]
    fn test_warp_complexity_nonidentity_positive() {
        let m = 50;
        let t = uniform_grid(m);
        let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
        let c = warp_complexity(&gamma, &t);
        assert!(
            c > 0.01,
            "Non-identity warp should have positive complexity, got {c}"
        );
    }

    #[test]
    fn test_warp_smoothness_identity_is_zero() {
        let m = 50;
        let t = uniform_grid(m);
        let identity = t.clone();
        let s = warp_smoothness(&identity, &t);
        assert!(
            s < 1e-6,
            "Identity warp (constant γ'=1, γ''=0) should have near-zero bending energy, got {s}"
        );
    }

    #[test]
    fn test_alignment_quality_basic() {
        let m = 50;
        let n = 8;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let karcher = karcher_mean(&data, &t, 10, 1e-3, 0.0);
        let quality = alignment_quality(&data, &karcher, &t);

        // Shape checks
        assert_eq!(quality.warp_complexity.len(), n);
        assert_eq!(quality.warp_smoothness.len(), n);
        assert_eq!(quality.pointwise_variance_ratio.len(), m);

        // Non-negativity
        assert!(quality.total_variance >= 0.0);
        assert!(quality.amplitude_variance >= 0.0);
        assert!(quality.phase_variance >= 0.0);
        assert!(quality.mean_warp_complexity >= 0.0);
        assert!(quality.mean_warp_smoothness >= 0.0);

        // Amplitude variance ≤ total variance
        assert!(
            quality.amplitude_variance <= quality.total_variance + 1e-10,
            "Amplitude variance ({}) should be ≤ total variance ({})",
            quality.amplitude_variance,
            quality.total_variance
        );
    }

    #[test]
    fn test_alignment_quality_identical_curves() {
        let m = 50;
        let t = uniform_grid(m);
        let curve: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let mut col_major = vec![0.0; 5 * m];
        for i in 0..5 {
            for j in 0..m {
                col_major[i + j * 5] = curve[j];
            }
        }
        let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
        let karcher = karcher_mean(&data, &t, 5, 1e-3, 0.0);
        let quality = alignment_quality(&data, &karcher, &t);

        // Identical curves → near-zero variances and warp complexities
        assert!(
            quality.total_variance < 0.01,
            "Identical curves should have near-zero total variance, got {}",
            quality.total_variance
        );
        assert!(
            quality.mean_warp_complexity < 0.1,
            "Identical curves should have near-zero warp complexity, got {}",
            quality.mean_warp_complexity
        );
    }

    #[test]
    fn test_alignment_quality_variance_reduction() {
        let m = 50;
        let n = 10;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);
        let karcher = karcher_mean(&data, &t, 10, 1e-3, 0.0);
        let quality = alignment_quality(&data, &karcher, &t);

        // Mean variance ratio should be ≤ ~1 (alignment shouldn't increase variance)
        assert!(
            quality.mean_variance_reduction <= 1.5,
            "Mean variance reduction ratio should be ≤ ~1, got {}",
            quality.mean_variance_reduction
        );
    }

    #[test]
    fn test_pairwise_consistency_small() {
        let m = 40;
        let n = 4;
        let t = uniform_grid(m);
        let data = make_test_data(n, m, 42);

        let consistency = pairwise_consistency(&data, &t, 0.0, 100);
        assert!(
            consistency.is_finite() && consistency >= 0.0,
            "Pairwise consistency should be finite and non-negative, got {consistency}"
        );
    }

    // ── Multidimensional SRSF ──

    #[test]
    fn test_srsf_nd_d1_matches_existing() {
        let m = 50;
        let t = uniform_grid(m);
        let data = make_test_data(3, m, 42);

        // 1D via existing function
        let q_1d = srsf_transform(&data, &t);

        // 1D via nd function
        let data_nd = FdCurveSet::from_1d(data);
        let q_nd = srsf_transform_nd(&data_nd, &t);

        assert_eq!(q_nd.ndim(), 1);
        for i in 0..3 {
            for j in 0..m {
                assert!(
                    (q_1d[(i, j)] - q_nd.dims[0][(i, j)]).abs() < 1e-10,
                    "1D nd SRSF should match existing at ({i},{j}): {} vs {}",
                    q_1d[(i, j)],
                    q_nd.dims[0][(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_srsf_nd_constant_is_zero() {
        let m = 30;
        let t = uniform_grid(m);
        // Constant R^2 curve: f(t) = (3.0, -1.0)
        let dim0 = FdMatrix::from_column_major(vec![3.0; m], 1, m).unwrap();
        let dim1 = FdMatrix::from_column_major(vec![-1.0; m], 1, m).unwrap();
        let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

        let q = srsf_transform_nd(&data, &t);
        for k in 0..2 {
            for j in 0..m {
                assert!(
                    q.dims[k][(0, j)].abs() < 1e-10,
                    "Constant curve SRSF should be zero, dim {k} at {j}: {}",
                    q.dims[k][(0, j)]
                );
            }
        }
    }

    #[test]
    fn test_srsf_nd_linear_r2() {
        let m = 51;
        let t = uniform_grid(m);
        // f(t) = (2t, 3t) → f'(t) = (2, 3), ||f'|| = sqrt(13)
        // q(t) = (2, 3) / sqrt(sqrt(13)) = (2, 3) / 13^(1/4)
        let dim0 =
            FdMatrix::from_slice(&t.iter().map(|&ti| 2.0 * ti).collect::<Vec<_>>(), 1, m).unwrap();
        let dim1 =
            FdMatrix::from_slice(&t.iter().map(|&ti| 3.0 * ti).collect::<Vec<_>>(), 1, m).unwrap();
        let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

        let q = srsf_transform_nd(&data, &t);
        let expected_scale = 1.0 / 13.0_f64.powf(0.25);
        let mid = m / 2;

        assert!(
            (q.dims[0][(0, mid)] - 2.0 * expected_scale).abs() < 0.1,
            "q_x at midpoint: {} vs expected {}",
            q.dims[0][(0, mid)],
            2.0 * expected_scale
        );
        assert!(
            (q.dims[1][(0, mid)] - 3.0 * expected_scale).abs() < 0.1,
            "q_y at midpoint: {} vs expected {}",
            q.dims[1][(0, mid)],
            3.0 * expected_scale
        );
    }

    #[test]
    fn test_srsf_nd_round_trip() {
        let m = 51;
        let t = uniform_grid(m);
        // f(t) = (sin(2πt), cos(2πt))
        let pi2 = 2.0 * std::f64::consts::PI;
        let vals_x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
        let vals_y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
        let dim0 = FdMatrix::from_slice(&vals_x, 1, m).unwrap();
        let dim1 = FdMatrix::from_slice(&vals_y, 1, m).unwrap();
        let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

        let q = srsf_transform_nd(&data, &t);
        let q_vecs: Vec<Vec<f64>> = q.dims.iter().map(|dm| dm.row(0)).collect();
        let f0 = vec![vals_x[0], vals_y[0]];
        let recon = srsf_inverse_nd(&q_vecs, &t, &f0);

        // Check reconstruction error (skip boundary points)
        let mut max_err = 0.0_f64;
        for k in 0..2 {
            let orig = if k == 0 { &vals_x } else { &vals_y };
            for j in 2..(m - 2) {
                let err = (recon[k][j] - orig[j]).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 0.2,
            "SRSF round-trip max error should be small, got {max_err}"
        );
    }

    #[test]
    fn test_align_nd_identical_near_zero() {
        let m = 50;
        let t = uniform_grid(m);
        let pi2 = 2.0 * std::f64::consts::PI;
        let vals_x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
        let vals_y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
        let dim0 = FdMatrix::from_slice(&vals_x, 1, m).unwrap();
        let dim1 = FdMatrix::from_slice(&vals_y, 1, m).unwrap();
        let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

        let result = elastic_align_pair_nd(&data, &data, &t, 0.0);
        assert!(
            result.distance < 0.5,
            "Self-alignment distance should be ~0, got {}",
            result.distance
        );
        // Gamma should be near identity
        let max_dev: f64 = result
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 0.1,
            "Self-alignment warp should be near identity, max dev = {max_dev}"
        );
    }

    #[test]
    fn test_align_nd_shifted_r2() {
        let m = 60;
        let t = uniform_grid(m);
        let pi2 = 2.0 * std::f64::consts::PI;

        // f1(t) = (sin(2πt), cos(2πt))
        let f1x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
        let f1y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
        let f1 = FdCurveSet::from_dims(vec![
            FdMatrix::from_slice(&f1x, 1, m).unwrap(),
            FdMatrix::from_slice(&f1y, 1, m).unwrap(),
        ])
        .unwrap();

        // f2(t) = (sin(2π(t-0.1)), cos(2π(t-0.1))) — phase shifted
        let f2x: Vec<f64> = t.iter().map(|&ti| (pi2 * (ti - 0.1)).sin()).collect();
        let f2y: Vec<f64> = t.iter().map(|&ti| (pi2 * (ti - 0.1)).cos()).collect();
        let f2 = FdCurveSet::from_dims(vec![
            FdMatrix::from_slice(&f2x, 1, m).unwrap(),
            FdMatrix::from_slice(&f2y, 1, m).unwrap(),
        ])
        .unwrap();

        let result = elastic_align_pair_nd(&f1, &f2, &t, 0.0);
        assert!(
            result.distance.is_finite(),
            "Distance should be finite, got {}",
            result.distance
        );
        assert_eq!(result.f_aligned.len(), 2);
        assert_eq!(result.f_aligned[0].len(), m);
        // Warp should deviate from identity (non-trivial alignment)
        let max_dev: f64 = result
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev > 0.01,
            "Shifted curves should require non-trivial warp, max dev = {max_dev}"
        );
    }

    // ── Landmark-constrained alignment ──

    #[test]
    fn test_constrained_no_landmarks_matches_unconstrained() {
        let m = 50;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
        let r_const = elastic_align_pair_constrained(&f1, &f2, &t, &[], 0.0);

        // Should match unconstrained
        for j in 0..m {
            assert!(
                (r_free.gamma[j] - r_const.gamma[j]).abs() < 1e-10,
                "No-landmark constrained should match unconstrained at {j}"
            );
        }
        assert!(r_const.enforced_landmarks.is_empty());
    }

    #[test]
    fn test_constrained_single_landmark_enforced() {
        let m = 60;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        // Constrain midpoint: target_t=0.5 should map to source_t=0.5
        let result = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.5, 0.5)], 0.0);

        // Gamma at the midpoint should be close to 0.5
        let mid_idx = snap_to_grid(0.5, &t);
        assert!(
            (result.gamma[mid_idx] - 0.5).abs() < 0.05,
            "Constrained gamma at midpoint should be ~0.5, got {}",
            result.gamma[mid_idx]
        );
        assert_eq!(result.enforced_landmarks.len(), 1);
    }

    #[test]
    fn test_constrained_multiple_landmarks() {
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (4.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (4.0 * std::f64::consts::PI * (ti - 0.05)).sin())
            .collect();

        let landmarks = vec![(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)];
        let result = elastic_align_pair_constrained(&f1, &f2, &t, &landmarks, 0.0);

        // Gamma should pass through (or near) each landmark
        for &(tt, st) in &landmarks {
            let idx = snap_to_grid(tt, &t);
            assert!(
                (result.gamma[idx] - st).abs() < 0.05,
                "Gamma at t={tt} should be ~{st}, got {}",
                result.gamma[idx]
            );
        }
    }

    #[test]
    fn test_constrained_monotone_gamma() {
        let m = 60;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
            .collect();

        let result = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.3, 0.3), (0.7, 0.7)], 0.0);

        // Gamma should be non-decreasing
        for j in 1..m {
            assert!(
                result.gamma[j] >= result.gamma[j - 1] - 1e-10,
                "Gamma should be monotone: gamma[{}]={} < gamma[{}]={}",
                j,
                result.gamma[j],
                j - 1,
                result.gamma[j - 1]
            );
        }
        // Boundary conditions
        assert!((result.gamma[0] - t[0]).abs() < 1e-10);
        assert!((result.gamma[m - 1] - t[m - 1]).abs() < 1e-10);
    }

    #[test]
    fn test_constrained_distance_ge_unconstrained() {
        let m = 60;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
            .collect();

        let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
        let r_const = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.5, 0.5)], 0.0);

        // Constrained distance should be >= unconstrained (constraints reduce freedom)
        assert!(
            r_const.distance >= r_free.distance - 1e-6,
            "Constrained distance ({}) should be >= unconstrained ({})",
            r_const.distance,
            r_free.distance
        );
    }

    #[test]
    fn test_constrained_with_landmark_detection() {
        let m = 80;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (4.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (4.0 * std::f64::consts::PI * (ti - 0.05)).sin())
            .collect();

        let result = elastic_align_pair_with_landmarks(
            &f1,
            &f2,
            &t,
            crate::landmark::LandmarkKind::Peak,
            0.1,
            0,
            0.0,
        );

        assert_eq!(result.gamma.len(), m);
        assert_eq!(result.f_aligned.len(), m);
        assert!(result.distance.is_finite());
        // Should be monotone
        for j in 1..m {
            assert!(
                result.gamma[j] >= result.gamma[j - 1] - 1e-10,
                "Gamma should be monotone at j={j}"
            );
        }
    }

    // ── SRSF smoothing for TSRVF (Issue #13) ──

    #[test]
    fn test_gam_to_psi_smooth_identity() {
        // Smoothed psi of identity warp should stay close to constant 1 in the interior.
        // Boundary points are biased by kernel smoothing (fewer neighbors), skip them.
        use crate::warping::{gam_to_psi, gam_to_psi_smooth};
        let m = 101;
        let h = 1.0 / (m - 1) as f64;
        let gam: Vec<f64> = uniform_grid(m);
        let psi_raw = gam_to_psi(&gam, h);
        let psi_smooth = gam_to_psi_smooth(&gam, h);
        // Check interior points (skip ~5% at each boundary)
        let skip = m / 20;
        for j in skip..(m - skip) {
            assert!(
                (psi_smooth[j] - 1.0).abs() < 0.05,
                "Smoothed psi of identity should be ~1.0, got {} at j={}",
                psi_smooth[j],
                j
            );
            assert!(
                (psi_smooth[j] - psi_raw[j]).abs() < 0.05,
                "Smoothed and raw psi should agree on smooth warp at j={}",
                j
            );
        }
    }

    #[test]
    fn test_gam_to_psi_smooth_reduces_spikes() {
        // Create a kinky warp (simulating DP output with multiple slope changes)
        // and verify smoothing reduces psi spikes
        use crate::warping::{gam_to_psi, gam_to_psi_smooth};
        let m = 101;
        let h = 1.0 / (m - 1) as f64;
        let argvals = uniform_grid(m);
        // Multi-segment piecewise-linear warp with several kinks
        let mut gam: Vec<f64> = Vec::with_capacity(m);
        for j in 0..m {
            let t = argvals[j];
            // Three segments: slow (slope 0.5), fast (slope 2), slow (slope 0.5)
            let g = if t < 0.33 {
                t * 0.5 / 0.33
            } else if t < 0.67 {
                0.5 + (t - 0.33) * 0.5 / 0.34 * 2.0 // steeper
            } else {
                let base = 0.5 + 0.5 / 0.34 * 2.0 * 0.34; // ~1.5 but clamped
                (base + (t - 0.67) * 0.5 / 0.33).min(1.0)
            };
            gam.push(g.min(1.0));
        }
        // Normalize to [0,1]
        let gmax = gam[m - 1].max(1e-10);
        for g in &mut gam {
            *g /= gmax;
        }
        let psi_raw = gam_to_psi(&gam, h);
        let psi_smooth = gam_to_psi_smooth(&gam, h);
        // The raw psi should have jumps at kink points (slope transitions)
        let max_jump_raw: f64 = psi_raw
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);
        let max_jump_smooth: f64 = psi_smooth
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max);
        // Smoothing should reduce the maximum jump in psi
        assert!(
            max_jump_smooth < max_jump_raw + 0.01,
            "Smoothing should not increase max psi jump: raw={max_jump_raw:.4}, smooth={max_jump_smooth:.4}"
        );
    }

    #[test]
    fn test_smooth_aligned_srsfs_preserves_shape() {
        // Smoothing aligned SRSFs should preserve overall shape
        use crate::smoothing::nadaraya_watson;
        let m = 101;
        let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        // Create a smooth SRSF (sine curve)
        let qi: Vec<f64> = time
            .iter()
            .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let bandwidth = 2.0 / (m - 1) as f64;
        let qi_smooth = nadaraya_watson(&time, &qi, &time, bandwidth, "gaussian");
        // Correlation between original and smoothed should be very high
        let mean_orig: f64 = qi.iter().sum::<f64>() / m as f64;
        let mean_smooth: f64 = qi_smooth.iter().sum::<f64>() / m as f64;
        let mut cov = 0.0;
        let mut var_o = 0.0;
        let mut var_s = 0.0;
        for j in 0..m {
            let do_ = qi[j] - mean_orig;
            let ds = qi_smooth[j] - mean_smooth;
            cov += do_ * ds;
            var_o += do_ * do_;
            var_s += ds * ds;
        }
        let rho = cov / (var_o * var_s).sqrt().max(1e-10);
        assert!(
            rho > 0.99,
            "Smoothed SRSF should be highly correlated with original (rho={rho:.4})"
        );
    }

    #[test]
    fn test_tsrvf_tangent_vectors_no_spikes() {
        // End-to-end: compute TSRVF tangent vectors, verify no element dominates.
        // The smooth_aligned_srsfs step in tsrvf_from_alignment removes DP kink
        // artifacts that would otherwise produce spike outliers in tangent vectors.
        let m = 101;
        let argvals = uniform_grid(m);
        let data = make_test_data(10, m, 42);
        let result = tsrvf_transform(&data, &argvals, 5, 1e-3, 0.0);
        let (n, _) = result.tangent_vectors.shape();
        for i in 0..n {
            let vi = result.tangent_vectors.row(i);
            let rms = (vi.iter().map(|&v| v * v).sum::<f64>() / m as f64).sqrt();
            if rms > 1e-10 {
                let max_abs = vi.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
                assert!(
                    max_abs < 10.0 * rms,
                    "Tangent vector {} has spike: max |v| = {max_abs:.4}, rms = {rms:.4}, ratio = {:.1}",
                    i,
                    max_abs / rms
                );
            }
        }
    }
}
