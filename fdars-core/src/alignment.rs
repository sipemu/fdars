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
use crate::helpers::{l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
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
}

// ─── Private helpers ────────────────────────────────────────────────────────

/// Linear interpolation at point `t` using binary search.
fn linear_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    if t <= x[0] {
        return y[0];
    }
    let last = x.len() - 1;
    if t >= x[last] {
        return y[last];
    }

    // Binary search for the interval containing t
    let idx = match x.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(i) => return y[i],
        Err(i) => i,
    };

    let t0 = x[idx - 1];
    let t1 = x[idx];
    let y0 = y[idx - 1];
    let y1 = y[idx];
    y0 + (y1 - y0) * (t - t0) / (t1 - t0)
}

/// Cumulative trapezoidal integration.
fn cumulative_trapz(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    for k in 1..n {
        out[k] = out[k - 1] + 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }
    out
}

/// Ensure γ is a valid warping: monotone non-decreasing, with correct boundary values.
fn normalize_warp(gamma: &mut [f64], argvals: &[f64]) {
    let n = gamma.len();
    if n == 0 {
        return;
    }

    // Fix boundaries
    gamma[0] = argvals[0];
    gamma[n - 1] = argvals[n - 1];

    // Enforce monotonicity
    for i in 1..n {
        if gamma[i] < gamma[i - 1] {
            gamma[i] = gamma[i - 1];
        }
    }
}

// ─── Sphere Geometry for Warping Functions ──────────────────────────────────
// Implements the Hilbert sphere representation of warping functions used by
// fdasrvf's `SqrtMeanInverse`: psi(t) = sqrt(gamma'(t)).

/// Trapezoidal integration of `y` over `x`.
fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for k in 1..y.len() {
        sum += 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }
    sum
}

/// Numerical gradient with uniform spacing (forward/central/backward differences).
fn gradient_uniform(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    g[0] = (y[1] - y[0]) / h;
    for i in 1..(n - 1) {
        g[i] = (y[i + 1] - y[i - 1]) / (2.0 * h);
    }
    g[n - 1] = (y[n - 1] - y[n - 2]) / h;
    g
}

/// Convert warping function to Hilbert sphere representation: psi = sqrt(gamma').
fn gam_to_psi(gam: &[f64], h: f64) -> Vec<f64> {
    gradient_uniform(gam, h)
        .iter()
        .map(|&g| g.max(0.0).sqrt())
        .collect()
}

/// Convert psi back to warping function: gamma = cumtrapz(psi^2), normalized to [0,1].
fn psi_to_gam(psi: &[f64], time: &[f64]) -> Vec<f64> {
    let psi_sq: Vec<f64> = psi.iter().map(|&p| p * p).collect();
    let gam = cumulative_trapz(&psi_sq, time);
    let min_val = gam.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = gam.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-10);
    gam.iter().map(|&v| (v - min_val) / range).collect()
}

/// L2 inner product: integral(psi1 * psi2 dt) via trapezoidal rule.
fn inner_product_l2(psi1: &[f64], psi2: &[f64], time: &[f64]) -> f64 {
    let prod: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
    trapz(&prod, time)
}

/// L2 norm: sqrt(integral(psi^2 dt)).
fn l2_norm_l2(psi: &[f64], time: &[f64]) -> f64 {
    inner_product_l2(psi, psi, time).max(0.0).sqrt()
}

/// Inverse exponential (log) map on the Hilbert sphere.
/// Returns tangent vector at `mu` pointing toward `psi`.
fn inv_exp_map_sphere(mu: &[f64], psi: &[f64], time: &[f64]) -> Vec<f64> {
    let ip = inner_product_l2(mu, psi, time).clamp(-1.0, 1.0);
    let theta = ip.acos();
    if theta < 1e-10 {
        vec![0.0; mu.len()]
    } else {
        let coeff = theta / theta.sin();
        let cos_theta = theta.cos();
        mu.iter()
            .zip(psi.iter())
            .map(|(&m, &p)| coeff * (p - cos_theta * m))
            .collect()
    }
}

/// Exponential map on the Hilbert sphere.
/// Moves from `psi` along tangent vector `v`.
fn exp_map_sphere(psi: &[f64], v: &[f64], time: &[f64]) -> Vec<f64> {
    let v_norm = l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        psi.to_vec()
    } else {
        let cos_n = v_norm.cos();
        let sin_n = v_norm.sin();
        psi.iter()
            .zip(v.iter())
            .map(|(&p, &vi)| cos_n * p + sin_n * vi / v_norm)
            .collect()
    }
}

/// Invert a warping function: find gamma_inv such that gamma_inv(gamma(t)) = t.
/// `gam` and `time` are both on [0,1].
fn invert_gamma(gam: &[f64], time: &[f64]) -> Vec<f64> {
    let n = time.len();
    // Interpolate (gam -> time) at query points time
    // i.e., for each t in time, find s such that gam(s) = t, return s
    let mut gam_inv: Vec<f64> = time.iter().map(|&t| linear_interp(gam, time, t)).collect();
    gam_inv[0] = time[0];
    gam_inv[n - 1] = time[n - 1];
    gam_inv
}

/// Karcher mean of warping functions on the Hilbert sphere, then invert.
/// Port of fdasrvf's `SqrtMeanInverse`.
///
/// Takes a matrix of warping functions (n × m) on the argvals domain,
/// computes the Fréchet mean of their sqrt-derivative representations
/// on the unit Hilbert sphere, converts back to a warping function,
/// and returns its inverse (on the argvals domain).
fn sqrt_mean_inverse(gammas: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
    let (n, m) = gammas.shape();
    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;

    // Work on [0,1] internally
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    // Convert each gamma to psi = sqrt(gamma') on the unit sphere
    let mut psis: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let gam_01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
        psis.push(gam_to_psi(&gam_01, binsize));
    }

    // Initialize mu as pointwise mean of psis
    let mut mu = vec![0.0; m];
    for psi in &psis {
        for j in 0..m {
            mu[j] += psi[j];
        }
    }
    for j in 0..m {
        mu[j] /= n as f64;
    }

    // Karcher mean iteration on the Hilbert sphere
    let step_size = 0.3;
    let max_iter = 501;

    for _ in 0..max_iter {
        // Compute mean shooting vector (Karcher gradient)
        let mut vbar = vec![0.0; m];
        for psi in &psis {
            let v = inv_exp_map_sphere(&mu, psi, &time);
            for j in 0..m {
                vbar[j] += v[j];
            }
        }
        for j in 0..m {
            vbar[j] /= n as f64;
        }

        // Convergence check
        if l2_norm_l2(&vbar, &time) <= 1e-8 {
            break;
        }

        // Move mu along geodesic
        let scaled: Vec<f64> = vbar.iter().map(|&v| v * step_size).collect();
        mu = exp_map_sphere(&mu, &scaled, &time);
    }

    // Convert mean psi back to warping function, then invert
    let gam_mu = psi_to_gam(&mu, &time);
    let gam_inv = invert_gamma(&gam_mu, &time);

    // Scale back to argvals domain
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

/// Core DP alignment between two SRSFs on a grid.
///
/// Finds the optimal warping γ minimizing ‖q₁ - (q₂∘γ)√γ'‖².
/// Uses fdasrvf's coprime neighborhood (nbhd_dim=7 → 35 move directions).
/// SRSFs are L2-normalized before alignment (matching fdasrvf's `optimum.reparam`).
fn dp_alignment_core(q1: &[f64], q2: &[f64], argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return argvals.to_vec();
    }

    // Normalize SRSFs to unit L2 norm (matching fdasrvf's optimum.reparam)
    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    // Full m×m cost table and parent pointers
    // Rows = q2 index, Columns = q1 index (matching fdasrvf)
    let mut e = vec![f64::INFINITY; m * m];
    let mut parent = vec![u32::MAX; m * m];
    e[0] = 0.0;

    for tr in 1..m {
        for tc in 1..m {
            let idx = tr * m + tc;
            for &(dr, dc) in &COPRIME_NBHD_7 {
                if dr > tr || dc > tc {
                    continue;
                }
                let sr = tr - dr;
                let sc = tc - dc;
                let src_idx = sr * m + sc;
                if e[src_idx] == f64::INFINITY {
                    continue;
                }
                let w = dp_edge_weight(&q1n, &q2n, argvals, sc, tc, sr, tr);
                let cost = e[src_idx] + w;
                if cost < e[idx] {
                    e[idx] = cost;
                    parent[idx] = src_idx as u32;
                }
            }
        }
    }

    // Traceback from (m-1, m-1) to (0, 0) using parent pointers
    let mut path_tc = Vec::with_capacity(2 * m);
    let mut path_tr = Vec::with_capacity(2 * m);
    let mut cur = (m - 1) * m + (m - 1);
    loop {
        let tr = cur / m;
        let tc = cur % m;
        path_tc.push(argvals[tc]);
        path_tr.push(argvals[tr]);
        if cur == 0 {
            break;
        }
        if parent[cur] == u32::MAX {
            break;
        }
        cur = parent[cur] as usize;
    }

    // Reverse to forward order
    path_tc.reverse();
    path_tr.reverse();

    // Re-interpolate gamma onto the argvals grid
    // path_tc = t values (q1 side), path_tr = gamma values (q2 side)
    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&path_tc, &path_tr, t))
        .collect();

    normalize_warp(&mut gamma, argvals);
    gamma
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
///
/// # Returns
/// [`AlignmentResult`] with warping function, aligned curve, and elastic distance.
pub fn elastic_align_pair(f1: &[f64], f2: &[f64], argvals: &[f64]) -> AlignmentResult {
    let m = f1.len();

    // Build single-row FdMatrices for SRSF computation
    let f1_mat = FdMatrix::from_slice(f1, 1, m).unwrap();
    let f2_mat = FdMatrix::from_slice(f2, 1, m).unwrap();

    let q1_mat = srsf_transform(&f1_mat, argvals);
    let q2_mat = srsf_transform(&f2_mat, argvals);

    let q1: Vec<f64> = q1_mat.row(0);
    let q2: Vec<f64> = q2_mat.row(0);

    // Find optimal warping via DP
    let gamma = dp_alignment_core(&q1, &q2, argvals);

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
pub fn elastic_distance(f1: &[f64], f2: &[f64], argvals: &[f64]) -> f64 {
    elastic_align_pair(f1, f2, argvals).distance
}

/// Align all curves in `data` to a single target curve.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `target` — Target curve to align to (length m)
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// [`AlignmentSetResult`] with all warping functions, aligned curves, and distances.
pub fn align_to_target(data: &FdMatrix, target: &[f64], argvals: &[f64]) -> AlignmentSetResult {
    let (n, m) = data.shape();

    let results: Vec<AlignmentResult> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_align_pair(target, &fi, argvals)
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
///
/// # Returns
/// Symmetric n × n distance matrix.
pub fn elastic_self_distance_matrix(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let n = data.nrows();

    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let fi = data.row(i);
            ((i + 1)..n)
                .map(|j| {
                    let fj = data.row(j);
                    elastic_distance(&fi, &fj, argvals)
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
///
/// # Returns
/// n1 × n2 distance matrix.
pub fn elastic_cross_distance_matrix(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();

    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            let fi = data1.row(i);
            (0..n2)
                .map(|j| {
                    let fj = data2.row(j);
                    elastic_distance(&fi, &fj, argvals)
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
fn align_srsf_pair(q1: &[f64], q2: &[f64], argvals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let gamma = dp_alignment_core(q1, q2, argvals);

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
/// let result = karcher_mean(&data, &t, 20, 1e-4);
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

pub fn karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
) -> KarcherMeanResult {
    let (n, m) = data.shape();

    // Step 1: Compute SRSFs and select closest observed SRSF to the mean as template
    let srsf_mat = srsf_transform(data, argvals);
    let mnq = mean_1d(&srsf_mat);
    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0;
    for i in 0..n {
        let dist_sq: f64 = (0..m).map(|j| (srsf_mat[(i, j)] - mnq[j]).powi(2)).sum();
        if dist_sq < min_dist {
            min_dist = dist_sq;
            min_idx = i;
        }
    }
    let mut mu_q = srsf_mat.row(min_idx);
    let mut mu = data.row(min_idx);

    // Step 2: Pre-iteration centering with SqrtMeanInverse
    // Align all curves to the selected template, then center the template
    {
        let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let fi = data.row(i);
                let qi = srsf_single(&fi, argvals);
                align_srsf_pair(&mu_q, &qi, argvals)
            })
            .collect();

        let mut init_gammas = FdMatrix::zeros(n, m);
        for (i, (gamma, _)) in align_results.iter().enumerate() {
            for j in 0..m {
                init_gammas[(i, j)] = gamma[j];
            }
        }

        // Center: compute inverse mean warp, apply to template
        let gam_inv = sqrt_mean_inverse(&init_gammas, argvals);
        mu = reparameterize_curve(&mu, argvals, &gam_inv);
        mu_q = srsf_single(&mu, argvals);
    }

    // Step 3: Main Karcher mean iteration
    let mut converged = false;
    let mut n_iter = 0;
    let mut final_gammas = FdMatrix::zeros(n, m);
    let mut prev_rel = 0.0_f64;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let align_results: Vec<(Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let fi = data.row(i);
                let qi = srsf_single(&fi, argvals);
                align_srsf_pair(&mu_q, &qi, argvals)
            })
            .collect();

        let mu_q_new = accumulate_alignments(&align_results, &mut final_gammas, m, n);

        let rel = relative_change(&mu_q, &mu_q_new);
        if rel < f64::EPSILON || (iter > 0 && rel - prev_rel <= tol * prev_rel) {
            converged = true;
            mu_q = mu_q_new;
            break;
        }
        prev_rel = rel;

        mu_q = mu_q_new;
        mu = srsf_inverse(&mu_q, argvals, mu[0]);
    }

    // Step 4: Post-convergence centering with SqrtMeanInverse
    let gam_inv = sqrt_mean_inverse(&final_gammas, argvals);
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let gam_inv_dev = gradient_uniform(&gam_inv, h);

    // Center the mean SRSF: (mu_q ∘ gamI) * sqrt(gamI')
    let mu_q_warped = reparameterize_curve(&mu_q, argvals, &gam_inv);
    mu_q = mu_q_warped
        .iter()
        .zip(gam_inv_dev.iter())
        .map(|(&q, &gd)| q * gd.max(0.0).sqrt())
        .collect();

    // Center each curve's warp: gamma_centered = gamma ∘ gamI
    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| final_gammas[(i, j)]).collect();
        let gam_centered = reparameterize_curve(&gam_i, argvals, &gam_inv);
        for j in 0..m {
            final_gammas[(i, j)] = gam_centered[j];
        }
    }

    // Reconstruct mean curve from centered SRSF
    let initial_mean = mean_1d(data);
    mu = srsf_inverse(&mu_q, argvals, initial_mean[0]);
    let final_aligned = apply_stored_warps(data, &final_gammas, argvals);

    KarcherMeanResult {
        mean: mu,
        mean_srsf: mu_q,
        gammas: final_gammas,
        aligned_data: final_aligned,
        n_iter,
        converged,
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};

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

        let result = elastic_align_pair(&f, &f, &t);

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

        let result = elastic_align_pair(&f1, &f2, &t);

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

        let result = elastic_align_pair(&f1, &f2, &t);
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
        let result = elastic_align_pair(&f1, &f2, &t);
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

        let result = elastic_align_pair(&f1, &f2, &t);
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
        let result = elastic_align_pair(&f1, &f2, &t);
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

        let d12 = elastic_distance(&f1, &f2, &t);
        let d21 = elastic_distance(&f2, &f1, &t);

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
                let d = elastic_distance(&fi, &fj, &t);
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
            let d = elastic_distance(&fi, &fi, &t);
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

        let d01 = elastic_distance(&f0, &f1, &t);
        let d12 = elastic_distance(&f1, &f2, &t);
        let d02 = elastic_distance(&f0, &f2, &t);

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

        let d = elastic_distance(&f1, &f2, &t);
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

        let dm = elastic_self_distance_matrix(&data, &t);
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
        let dm = elastic_self_distance_matrix(&data, &t);

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
        let dm = elastic_self_distance_matrix(&data, &t);
        assert_eq!(dm.shape(), (1, 1));
        assert!(dm[(0, 0)].abs() < 1e-12);
    }

    #[test]
    fn test_self_distance_matrix_consistent_with_pairwise() {
        let data = make_test_data(4, 30, 42);
        let t = uniform_grid(30);

        let dm = elastic_self_distance_matrix(&data, &t);

        // Check a few entries match direct elastic_distance calls
        for i in 0..4 {
            for j in (i + 1)..4 {
                let fi = data.row(i);
                let fj = data.row(j);
                let d_direct = elastic_distance(&fi, &fj, &t);
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

        let result = karcher_mean(&data, &t, 10, 1e-4);

        assert_eq!(result.mean.len(), m);
        assert!(result.n_iter <= 10);
    }

    #[test]
    fn test_karcher_mean_output_shape() {
        let data = make_test_data(15, 50, 42);
        let t = uniform_grid(50);

        let result = karcher_mean(&data, &t, 5, 1e-3);

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

        let result = karcher_mean(&data, &t, 5, 1e-3);

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
        let result = karcher_mean(&data, &t, 5, 1e-3);

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
        let result = karcher_mean(&data, &t, 5, 1e-3);

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
        let result = karcher_mean(&data, &t, 1, 1e-10);

        assert_eq!(result.n_iter, 1);
        assert_eq!(result.mean.len(), 40);
        // With only 1 iteration, still produces valid output
        for j in 0..40 {
            assert!(result.mean[j].is_finite());
        }
    }

    // ── Align to target ──

    #[test]
    fn test_align_to_target_valid() {
        let data = make_test_data(10, 40, 42);
        let t = uniform_grid(40);
        let target = data.row(0);

        let result = align_to_target(&data, &target, &t);

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

        let result = align_to_target(&data, &target, &t);

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
        let result = align_to_target(&data, &target, &t);

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
        let result = align_to_target(&data, &target, &t);

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

        let dm = elastic_cross_distance_matrix(&data1, &data2, &t);
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

        let cross = elastic_cross_distance_matrix(&data, &data, &t);
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

        let dm = elastic_cross_distance_matrix(&data1, &data2, &t);

        for i in 0..3 {
            for j in 0..2 {
                let fi = data1.row(i);
                let fj = data2.row(j);
                let d_direct = elastic_distance(&fi, &fj, &t);
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

        let (gamma, q_aligned) = align_srsf_pair(&q, &q, &t);

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

        let result = elastic_align_pair(&f1, &f2, &t);

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

        let result = elastic_align_pair(&f1, &f2, &t);
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

        let result = karcher_mean(&data, &t, 5, 1e-4);
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
        let result = karcher_mean(&data, &t, 5, 1e-4);
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
        let d = elastic_distance(&f1, &f2, &t);
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
        let result = elastic_align_pair(&f1, &f2, &t);
        assert_eq!(result.gamma.len(), m);
        assert!(result.distance >= 0.0);
        assert!(result.distance.is_finite());
    }
}
