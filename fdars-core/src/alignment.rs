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

/// Convert a DP traceback path into a warping function sampled at argvals.
fn path_to_gamma(path: &[(usize, usize)], argvals: &[f64], grid: &[f64]) -> Vec<f64> {
    if path.is_empty() {
        return argvals.to_vec();
    }

    // Extract the warping from the path: γ maps grid[path[k].0] -> grid[path[k].1]
    let path_t: Vec<f64> = path.iter().map(|&(i, _)| grid[i]).collect();
    let path_g: Vec<f64> = path.iter().map(|&(_, j)| grid[j]).collect();

    // Interpolate to get γ at each argval
    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&path_t, &path_g, t))
        .collect();

    normalize_warp(&mut gamma, argvals);
    gamma
}

/// Pick the minimum-cost move (diagonal / horizontal / vertical) and write into `curr_row` and `trace`.
#[inline]
fn dp_pick_best(
    prev_row: &[f64],
    curr_row: &mut [f64],
    trace: &mut [u8],
    q1_i: f64,
    q2_j: f64,
    dt_i: f64,
    dt_j: f64,
    j: usize,
    trace_off: usize,
) {
    // Diagonal: (i-1,j-1) → (i,j) with slope correction
    let sqrt_slope = (dt_j / dt_i).sqrt();
    let vd = q1_i - q2_j * sqrt_slope;
    let cost_diag = prev_row[j - 1] + vd * vd * dt_i;

    // Horizontal: (i, j-1) → (i, j)
    let vh = q1_i - q2_j;
    let cost_horiz = curr_row[j - 1] + vh * vh * dt_j;

    // Vertical: (i-1, j) → (i, j)
    let cost_vert = prev_row[j] + vh * vh * dt_i;

    if cost_diag <= cost_horiz && cost_diag <= cost_vert {
        curr_row[j] = cost_diag;
        trace[trace_off + j] = 0;
    } else if cost_horiz <= cost_vert {
        curr_row[j] = cost_horiz;
        trace[trace_off + j] = 1;
    } else {
        curr_row[j] = cost_vert;
        trace[trace_off + j] = 2;
    }
}

/// Traceback through the DP trace matrix to recover the optimal path.
fn dp_traceback(trace: &[u8], m: usize) -> Vec<(usize, usize)> {
    let mut path = Vec::with_capacity(2 * m);
    let (mut i, mut j) = (m - 1, m - 1);
    path.push((i, j));

    while i > 0 || j > 0 {
        match trace[i * m + j] {
            0 => {
                i -= 1;
                j -= 1;
            }
            1 => j -= 1,
            _ => i -= 1,
        }
        path.push((i, j));
    }

    path.reverse();
    path
}

/// Core DP alignment between two SRSFs on a grid.
///
/// Finds the optimal warping γ minimizing ‖q₁ - (q₂∘γ)√γ'‖².
fn dp_alignment_core(q1: &[f64], q2: &[f64], argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return argvals.to_vec();
    }

    let grid = argvals;
    let mut prev_row = vec![f64::MAX; m];
    let mut curr_row = vec![f64::MAX; m];
    // 0 = diagonal, 1 = horizontal, 2 = vertical
    let mut trace = vec![0u8; m * m];

    // First row: can only come from left
    prev_row[0] = 0.0;
    for j in 1..m {
        let dt = grid[j] - grid[j - 1];
        let val = q1[0] - q2[j];
        prev_row[j] = prev_row[j - 1] + val * val * dt;
        trace[j] = 1;
    }

    // Fill remaining rows
    for i in 1..m {
        let dt_i = grid[i] - grid[i - 1];
        let val = q1[i] - q2[0];
        curr_row[0] = prev_row[0] + val * val * dt_i;
        trace[i * m] = 2;

        let trace_off = i * m;
        for j in 1..m {
            let dt_j = grid[j] - grid[j - 1];
            dp_pick_best(
                &prev_row,
                &mut curr_row,
                &mut trace,
                q1[i],
                q2[j],
                dt_i,
                dt_j,
                j,
                trace_off,
            );
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    let path = dp_traceback(&trace, m);
    path_to_gamma(&path, argvals, grid)
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

/// Check convergence of the Karcher mean iteration.
fn mean_has_converged(q_old: &[f64], q_new: &[f64], weights: &[f64], tol: f64) -> bool {
    let dist = l2_distance(q_old, q_new, weights);
    dist < tol
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
    let weights = simpsons_weights(argvals);

    let mut mu = mean_1d(data);
    let mut mu_q = srsf_single(&mu, argvals);

    let mut converged = false;
    let mut n_iter = 0;
    let mut final_gammas = FdMatrix::zeros(n, m);

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

        if mean_has_converged(&mu_q, &mu_q_new, &weights, tol) {
            converged = true;
            mu_q = mu_q_new;
            break;
        }

        mu_q = mu_q_new;
        mu = srsf_inverse(&mu_q, argvals, mu[0]);
    }

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

    // ── dp_traceback ──

    #[test]
    fn test_dp_traceback_all_diagonal() {
        // Trace matrix with all diagonal (0) moves
        let m = 5;
        let trace = vec![0u8; m * m];
        let path = dp_traceback(&trace, m);
        assert_eq!(path.first(), Some(&(0, 0)));
        assert_eq!(path.last(), Some(&(m - 1, m - 1)));
        assert_eq!(path.len(), m);
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
}
