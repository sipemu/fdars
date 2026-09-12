//! Soft-DTW: differentiable relaxation of DTW using log-sum-exp softmin.
//!
//! Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for
//! Time-Series" (ICML 2017).

use crate::autodiff::Scalar;
use crate::dim::Dim;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::collections::VecDeque;

use super::{cross_distance_matrix, self_distance_matrix};

/// Result of the Soft-DTW barycenter computation.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SoftDtwBarycenterResult {
    /// The barycenter time series.
    pub barycenter: Vec<f64>,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Soft minimum of three values using log-sum-exp trick for numerical stability.
///
/// As gamma->0 this approaches hard min; as gamma->inf it approaches the mean.
#[inline]
pub(super) fn softmin3(a: f64, b: f64, c: f64, gamma: f64) -> f64 {
    let min_val = a.min(b).min(c);
    if !min_val.is_finite() {
        return min_val;
    }
    let neg_inv_gamma = -1.0 / gamma;
    let ea = ((a - min_val) * neg_inv_gamma).exp();
    let eb = ((b - min_val) * neg_inv_gamma).exp();
    let ec = ((c - min_val) * neg_inv_gamma).exp();
    min_val - gamma * (ea + eb + ec).ln()
}

/// Generic soft-minimum of three [`Scalar`] values via the log-sum-exp trick.
///
/// Mirrors [`softmin3`] but is generic over `S: Scalar`, so it composes with the
/// forward-mode [`Dual`](crate::autodiff::Dual) substrate for automatic
/// differentiation. Instantiated at `f64` it is a zero-cost passthrough that
/// reproduces [`softmin3`] bit-for-bit for NaN-free inputs (the only difference
/// is the 3-way `min` uses `PartialOrd` branching instead of `f64::min`, which
/// agree for all non-NaN values).
///
/// The infinity guard uses `min_val >= S::infinity()` rather than
/// `is_finite()` — the [`Scalar`] trait deliberately exposes no `is_finite`
/// (Phase 75 is closed); [`S::infinity`](Scalar::infinity) plus `PartialOrd`
/// covers the DP sentinel case (all three inputs `+inf`).
#[inline]
fn softmin3_generic<S: Scalar>(a: S, b: S, c: S, gamma: f64) -> S {
    // 3-way min via PartialOrd (Scalar has no `min`; Dual's ordering is
    // value-only, which is the correct forward-mode branch semantics).
    let min_val = if a <= b {
        if a <= c {
            a
        } else {
            c
        }
    } else if b <= c {
        b
    } else {
        c
    };
    // Sentinel guard: if the min is +infinity every input is +infinity; return
    // it directly (matches the `!is_finite()` early-return in the f64 version
    // for the DP-initialization case).
    if min_val >= S::infinity() {
        return min_val;
    }
    let neg_inv_gamma = S::from_f64(-1.0 / gamma);
    // Per-term guard: a `+inf` sentinel input contributes `exp(-inf) = 0` in the
    // f64 path. Computing `(inf - min_val) * neg_inv_gamma` directly would work
    // for the primal value but produces a `inf * 0 = NaN` tangent under `Dual`'s
    // product rule. Substitute an exact `S::zero()` for those terms — identical
    // to the f64 numerics, and NaN-free for the tangent.
    let term = |v: S| -> S {
        if v >= S::infinity() {
            S::zero()
        } else {
            S::exp((v - min_val) * neg_inv_gamma)
        }
    };
    let ea = term(a);
    let eb = term(b);
    let ec = term(c);
    min_val - S::from_f64(gamma) * S::ln(ea + eb + ec)
}

/// Generic soft-DTW distance core over `S: Scalar`.
///
/// Shared kernel for both the `f64` public API ([`soft_dtw_distance`]) and the
/// differentiable generic entry point ([`soft_dtw_distance_generic`]). The DP
/// recurrence, cost, and softmin are all expressed in `S`, so instantiating at
/// [`Dual`](crate::autodiff::Dual) propagates exact forward-mode tangents.
fn soft_dtw_distance_inner<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return S::zero();
    }

    let mut prev = vec![S::infinity(); m + 1];
    let mut curr = vec![S::infinity(); m + 1];
    prev[0] = S::zero();

    for i in 1..=n {
        for v in curr.iter_mut() {
            *v = S::infinity();
        }
        for j in 1..=m {
            let d = x[i - 1] - y[j - 1];
            let cost = d * d;
            curr[j] = cost + softmin3_generic(prev[j], curr[j - 1], prev[j - 1], gamma);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Compute Soft-DTW distance between two 1D time series.
///
/// Uses squared Euclidean cost and 2-row DP with O(m) memory.
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series
/// * `gamma` - Smoothing parameter (> 0). Smaller = closer to hard DTW.
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    soft_dtw_distance_inner(x, y, gamma)
}

/// Differentiable soft-DTW distance, generic over the [`Scalar`] substrate.
///
/// Companion to [`soft_dtw_distance`]: identical numerics when instantiated at
/// `f64`, and exact forward-mode gradients when instantiated at
/// [`Dual`](crate::autodiff::Dual). To obtain
/// `d(soft_dtw)/d(x[k])`, seed `x[k]` via [`Dual::seed`](crate::autodiff::Dual::seed),
/// set every other `x`/`y` value via
/// [`Dual::constant`](crate::autodiff::Dual::constant), run this function, and
/// read the tangent of the result.
///
/// The soft-DTW DP recurrence and log-sum-exp softmin are smooth in the series
/// values, so the forward-mode gradient is exact (validated against the crate's
/// own hand-written soft-DTW gradient oracle and central finite differences).
///
/// # Arguments
/// * `x` - First time series (`&[S]`)
/// * `y` - Second time series (`&[S]`)
/// * `gamma` - Smoothing parameter (> 0). Smaller = closer to hard DTW.
pub fn soft_dtw_distance_generic<S: Scalar>(x: &[S], y: &[S], gamma: f64) -> S {
    soft_dtw_distance_inner(x, y, gamma)
}

/// Compute Soft-DTW divergence: `sdtw(x,y) - 0.5*(sdtw(x,x) + sdtw(y,y))`.
///
/// The divergence is non-negative and equals zero when x == y, making it
/// a proper discrepancy measure (unlike raw Soft-DTW which can be negative).
pub fn soft_dtw_divergence(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let xy = soft_dtw_distance(x, y, gamma);
    let xx = soft_dtw_distance(x, x, gamma);
    let yy = soft_dtw_distance(y, y, gamma);
    xy - 0.5 * (xx + yy)
}

/// Compute Soft-DTW self-distance matrix (symmetric n x n).
///
/// Note: unlike true metrics, `sdtw(x, x) != 0` for finite gamma,
/// so the diagonal is computed explicitly.
pub(crate) fn soft_dtw_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
    let n = data.nrows();
    if n == 0 || data.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let mut dist = self_distance_matrix(n, |i, j| soft_dtw_distance(&rows[i], &rows[j], gamma));
    // Fill diagonal: sdtw(x, x) is typically negative for finite gamma
    for i in 0..n {
        dist[(i, i)] = soft_dtw_distance(&rows[i], &rows[i], gamma);
    }
    dist
}

/// Compute Soft-DTW cross-distance matrix (n1 x n2).
pub(crate) fn soft_dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    if n1 == 0 || n2 == 0 || data1.ncols() == 0 || data2.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows1: Vec<Vec<f64>> = (0..n1).map(|i| data1.row(i)).collect();
    let rows2: Vec<Vec<f64>> = (0..n2).map(|i| data2.row(i)).collect();
    cross_distance_matrix(n1, n2, |i, j| {
        soft_dtw_distance(&rows1[i], &rows2[j], gamma)
    })
}

/// Compute Soft-DTW divergence self-distance matrix (symmetric n x n).
pub(crate) fn soft_dtw_div_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
    let n = data.nrows();
    if n == 0 || data.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    // Pre-compute self-distances for divergence
    let self_dists: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| soft_dtw_distance(&rows[i], &rows[i], gamma))
        .collect();
    self_distance_matrix(n, |i, j| {
        let xy = soft_dtw_distance(&rows[i], &rows[j], gamma);
        xy - 0.5 * (self_dists[i] + self_dists[j])
    })
}

/// Compute Soft-DTW divergence cross-distance matrix (n1 x n2).
pub(crate) fn soft_dtw_div_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    if n1 == 0 || n2 == 0 || data1.ncols() == 0 || data2.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows1: Vec<Vec<f64>> = (0..n1).map(|i| data1.row(i)).collect();
    let rows2: Vec<Vec<f64>> = (0..n2).map(|i| data2.row(i)).collect();
    let self1: Vec<f64> = iter_maybe_parallel!(0..n1)
        .map(|i| soft_dtw_distance(&rows1[i], &rows1[i], gamma))
        .collect();
    let self2: Vec<f64> = iter_maybe_parallel!(0..n2)
        .map(|j| soft_dtw_distance(&rows2[j], &rows2[j], gamma))
        .collect();
    cross_distance_matrix(n1, n2, |i, j| {
        let xy = soft_dtw_distance(&rows1[i], &rows2[j], gamma);
        xy - 0.5 * (self1[i] + self2[j])
    })
}

/// Soft-DTW self-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to [`soft_dtw_self_1d`]; `dim` makes intent explicit.
pub fn soft_dtw_self(data: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => soft_dtw_self_1d(data, gamma),
    }
}

/// Soft-DTW cross-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to [`soft_dtw_cross_1d`]; `dim` makes intent explicit.
pub fn soft_dtw_cross(data1: &FdMatrix, data2: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => soft_dtw_cross_1d(data1, data2, gamma),
    }
}

/// Soft-DTW divergence self-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to [`soft_dtw_div_self_1d`]; `dim` makes intent explicit.
pub fn soft_dtw_div_self(data: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => soft_dtw_div_self_1d(data, gamma),
    }
}

/// Soft-DTW divergence cross-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to [`soft_dtw_div_cross_1d`]; `dim` makes intent explicit.
pub fn soft_dtw_div_cross(data1: &FdMatrix, data2: &FdMatrix, gamma: f64, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => soft_dtw_div_cross_1d(data1, data2, gamma),
    }
}

/// Full forward pass: returns the (n+1) x (m+1) R table needed for the backward pass.
fn soft_dtw_forward(x: &[f64], y: &[f64], gamma: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = y.len();
    let mut r = vec![vec![f64::INFINITY; m + 1]; n + 1];
    r[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let d = x[i - 1] - y[j - 1];
            let cost = d * d;
            r[i][j] = cost + softmin3(r[i - 1][j], r[i][j - 1], r[i - 1][j - 1], gamma);
        }
    }
    r
}

/// Backward pass: compute E matrix (soft alignment weights) from the R table.
///
/// E[i][j] represents the contribution of alignment (i,j) to the gradient.
fn soft_dtw_backward(x: &[f64], y: &[f64], r: &[Vec<f64>], gamma: f64) -> Vec<Vec<f64>> {
    let n = x.len();
    let m = y.len();
    let mut e = vec![vec![0.0; m + 2]; n + 2];
    // Boundary: E[n][m] = 1 (the endpoint contributes fully)
    e[n][m] = 1.0;

    // Also set up sentinel R values: R[n+1][*] = R[*][m+1] = INF
    // We'll handle this via bounds checking

    for i in (1..=n).rev() {
        for j in (1..=m).rev() {
            // Preserve the E[n][m] = 1.0 endpoint seed: at (n, m) all three
            // neighbour contributions (a, b, c) are gated off, so the
            // unconditional write `e[i][j] = a + b + c` would overwrite the
            // seed with 0.0 and zero the entire backward pass.
            if i == n && j == m {
                continue;
            }
            // Contribution from (i+1, j): R[i+1][j] used R[i][j] via the "up" move
            let a = if i < n {
                e[i + 1][j]
                    * (-(r[i][j] - r[i + 1][j] + r[i + 1][j] - softmin3_val(r, i + 1, j, gamma))
                        / gamma)
                        .exp()
            } else {
                0.0
            };
            // Contribution from (i, j+1): R[i][j+1] used R[i][j] via the "right" move
            let b = if j < m {
                e[i][j + 1]
                    * (-(r[i][j] - r[i][j + 1] + r[i][j + 1] - softmin3_val(r, i, j + 1, gamma))
                        / gamma)
                        .exp()
            } else {
                0.0
            };
            // Contribution from (i+1, j+1): R[i+1][j+1] used R[i][j] via the "diagonal" move
            let c = if i < n && j < m {
                e[i + 1][j + 1]
                    * (-(r[i][j] - r[i + 1][j + 1] + r[i + 1][j + 1]
                        - softmin3_val(r, i + 1, j + 1, gamma))
                        / gamma)
                        .exp()
            } else {
                0.0
            };
            e[i][j] = a + b + c;
        }
    }
    e
}

/// Helper: extract softmin3 value at position (i,j) in the R table.
#[inline]
fn softmin3_val(r: &[Vec<f64>], i: usize, j: usize, gamma: f64) -> f64 {
    softmin3(
        if i > 0 { r[i - 1][j] } else { f64::INFINITY },
        if j > 0 { r[i][j - 1] } else { f64::INFINITY },
        if i > 0 && j > 0 {
            r[i - 1][j - 1]
        } else {
            f64::INFINITY
        },
        gamma,
    )
}

/// Accumulate the Soft-DTW gradient for one series into `grad`.
///
/// Performs forward pass, backward pass, and double-loop gradient accumulation.
/// Accumulates both the soft-DTW gradient and the per-point soft-alignment weight
/// `W[k] = sum_j E[k][j]` (total alignment mass landing on barycenter point `k`).
///
/// The weight is the diagonal curvature of the (quadratic-cost) soft-DTW objective:
/// `d grad[k] / d bary[k] = 2 * W[k]`. It is preserved in the signature for
/// forward compatibility; the current L-BFGS optimizer passes a scratch buffer
/// and does not use the weight output. See CORR-01 / Phase 78 for gradient history.
fn soft_dtw_accumulate_gradient_and_weight(
    bary: &[f64],
    xi: &[f64],
    gamma: f64,
    grad: &mut [f64],
    weight: &mut [f64],
) {
    let m = bary.len();
    let r = soft_dtw_forward(bary, xi, gamma);
    let e = soft_dtw_backward(bary, xi, &r, gamma);
    for k in 1..=m {
        let mut g = 0.0;
        let mut w = 0.0;
        for j in 1..=xi.len() {
            g += e[k][j] * 2.0 * (bary[k - 1] - xi[j - 1]);
            w += e[k][j];
        }
        grad[k - 1] += g;
        weight[k - 1] += w;
    }
}

// ---- L-BFGS internals --------------------------------------------------------

/// Dot product of two equal-length slices.
#[inline]
fn lbfgs_dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// In-place axpy: `y[i] += alpha * x[i]` for all i.
#[inline]
fn lbfgs_axpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi += alpha * xi;
    }
}

/// Compute total soft-DTW objective and gradient at `bary`.
///
/// Zeros `g` and `w_scratch` first (the accumulate fn does `+=`).
/// `w_scratch` is the diagonal-curvature weight buffer required by the
/// accumulate fn signature; it is not used by the L-BFGS optimizer.
fn eval_obj_and_grad(
    bary: &[f64],
    rows: &[Vec<f64>],
    gamma: f64,
    g: &mut [f64],
    w_scratch: &mut [f64],
) -> f64 {
    for v in g.iter_mut() {
        *v = 0.0;
    }
    for v in w_scratch.iter_mut() {
        *v = 0.0;
    }
    let mut f = 0.0_f64;
    for row in rows {
        f += soft_dtw_distance(bary, row, gamma);
        soft_dtw_accumulate_gradient_and_weight(bary, row, gamma, g, w_scratch);
    }
    f
}

/// Two-loop L-BFGS recursion (Nocedal & Wright, Algorithm 7.4).
///
/// Produces the descent direction `d = -H_k * g` in `d`.
/// `q_buf` is a reusable work buffer (will be resized to `g.len()`).
/// History is stored oldest-first in `s_hist`/`y_hist`.
fn two_loop_lbfgs(
    g: &[f64],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
    gamma_k: f64,
    q_buf: &mut Vec<f64>,
    d: &mut Vec<f64>,
) {
    let n = g.len();
    let m_actual = s_hist.len();

    // q ← g
    q_buf.resize(n, 0.0);
    q_buf.copy_from_slice(g);

    let mut alphas = vec![0.0_f64; m_actual];
    // Store rho_i once (Nocedal & Wright Algorithm 7.4) — reused in both loops to
    // avoid recomputing lbfgs_dot(y, s) twice per history entry and to keep both
    // loops consistent if the denominator formula ever changes.
    let mut rhos = vec![0.0_f64; m_actual];

    // Loop 1: newest to oldest
    for i in (0..m_actual).rev() {
        let s = &s_hist[i];
        let y = &y_hist[i];
        rhos[i] = 1.0 / lbfgs_dot(y, s); // ρᵢ = 1 / (yᵢᵀsᵢ); sᵀy > 1e-20 by guard
        alphas[i] = rhos[i] * lbfgs_dot(s, q_buf);
        lbfgs_axpy(-alphas[i], y, q_buf); // q ← q - αᵢ yᵢ
    }

    // Apply initial Hessian scale: r ← γ_k · q
    d.resize(n, 0.0);
    for j in 0..n {
        d[j] = gamma_k * q_buf[j];
    }

    // Loop 2: oldest to newest
    for i in 0..m_actual {
        let s = &s_hist[i];
        let y = &y_hist[i];
        let beta = rhos[i] * lbfgs_dot(y, d);
        lbfgs_axpy(alphas[i] - beta, s, d); // r ← r + (αᵢ - βᵢ) sᵢ
    }

    // d = -r  (descent direction)
    for v in d.iter_mut() {
        *v = -*v;
    }
}

/// Backtracking Armijo line search.
///
/// Returns `(alpha, f_new)`. If no step satisfies the Armijo condition within
/// `max_bt` backtracks, returns the safeguarded minimum step `beta^max_bt` so
/// the iteration cannot diverge (the barycenter still moves by a bounded amount).
fn backtrack_armijo(
    bary: &[f64],
    d: &[f64],
    f0: f64,
    g0: &[f64],
    rows: &[Vec<f64>],
    gamma_sdtw: f64,
    c1: f64,
    beta: f64,
    max_bt: usize,
) -> (f64, f64) {
    let slope = lbfgs_dot(g0, d); // must be < 0 (caller ensures descent direction)
    let mut alpha = 1.0_f64;
    let alpha_min = beta.powi(max_bt as i32);

    for _ in 0..max_bt {
        let x_new: Vec<f64> = bary
            .iter()
            .zip(d.iter())
            .map(|(b, di)| b + alpha * di)
            .collect();
        let f_new: f64 = rows
            .iter()
            .map(|r| soft_dtw_distance(&x_new, r, gamma_sdtw))
            .sum();
        if f_new <= f0 + c1 * alpha * slope {
            return (alpha, f_new);
        }
        alpha *= beta;
    }

    // Safeguard: use alpha_min — the update is tiny but avoids stall/divergence
    let x_new: Vec<f64> = bary
        .iter()
        .zip(d.iter())
        .map(|(b, di)| b + alpha_min * di)
        .collect();
    let f_new: f64 = rows
        .iter()
        .map(|r| soft_dtw_distance(&x_new, r, gamma_sdtw))
        .sum();
    (alpha_min, f_new)
}

/// Initialize the barycenter as the pointwise mean of all series.
fn init_barycenter_mean(rows: &[Vec<f64>]) -> Vec<f64> {
    let n = rows.len();
    let m = rows[0].len();
    let mut bary = vec![0.0; m];
    for row in rows {
        for (j, val) in row.iter().enumerate() {
            bary[j] += val;
        }
    }
    for v in &mut bary {
        *v /= n as f64;
    }
    bary
}

/// Run one L-BFGS descent from a given initial barycenter.
///
/// Private helper shared by `soft_dtw_barycenter` (single-init path) and
/// `soft_dtw_barycenter_multirestart` (called once per restart). The caller
/// is responsible for providing a valid `gamma > 0` and a non-empty `rows`.
///
/// Carries all Phase 101 safeguards:
/// - curvature guard `sᵀy > 1e-20` before history push
/// - non-descent direction fallback `d = -g` + history clear
/// - Armijo safeguarded minimum step (`beta^max_bt`)
fn lbfgs_refine_from_init(
    rows: &[Vec<f64>],
    init: Vec<f64>,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult {
    let m = init.len();
    let mut bary = init;

    // L-BFGS hyperparameters (locked per CONTEXT.md design decisions)
    let m_lbfgs: usize = 10;
    let c1 = 1e-4_f64;
    let beta_ls = 0.5_f64;
    let max_bt: usize = 20;

    // Allocate reusable buffers
    let mut g = vec![0.0_f64; m];
    let mut w_scratch = vec![0.0_f64; m]; // required by accumulate fn signature; unused by L-BFGS
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::with_capacity(m_lbfgs);
    let mut q_buf = vec![0.0_f64; m];
    let mut d = vec![0.0_f64; m];

    // Initial objective and gradient
    let mut f = eval_obj_and_grad(&bary, rows, gamma, &mut g, &mut w_scratch);
    let mut converged = false;
    let mut n_iter = 0_usize;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Stopping check 1: gradient ∞-norm
        let g_inf = g.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        if g_inf < tol {
            converged = true;
            break;
        }

        // Initial Hessian scale γ_k = (sᵀy)/(yᵀy); fallback to 1.0 when empty or near-zero
        let gamma_k = if s_hist.is_empty() {
            1.0
        } else {
            let s_last = s_hist.back().unwrap();
            let y_last = y_hist.back().unwrap();
            let yy = lbfgs_dot(y_last, y_last);
            if yy < 1e-20 {
                1.0
            } else {
                lbfgs_dot(s_last, y_last) / yy
            }
        };

        // Compute descent direction d = -H_k * g via two-loop recursion
        two_loop_lbfgs(&g, &s_hist, &y_hist, gamma_k, &mut q_buf, &mut d);

        // Non-descent direction safeguard (Pitfall 5): if dot(g, d) >= 0, reset to steepest descent
        if lbfgs_dot(&g, &d) >= 0.0 {
            // d = -g
            d.iter_mut().zip(g.iter()).for_each(|(di, gi)| *di = -*gi);
            // Clear history to avoid repeating the ill-conditioned direction
            s_hist.clear();
            y_hist.clear();
        }

        // Backtracking Armijo line search
        let (alpha, f_new) = backtrack_armijo(&bary, &d, f, &g, rows, gamma, c1, beta_ls, max_bt);

        // Stopping check 2: relative cost change
        let rel_change = (f - f_new).abs() / f.abs().max(1.0);
        if rel_change < tol {
            converged = true;
        }

        // Compute s = alpha * d; update bary
        let s_vec: Vec<f64> = d.iter().map(|v| alpha * v).collect();
        for (b, si) in bary.iter_mut().zip(s_vec.iter()) {
            *b += si;
        }

        // Recompute gradient at new bary (and update f for next iteration's rel_change check)
        let g_old = g.clone();
        f = eval_obj_and_grad(&bary, rows, gamma, &mut g, &mut w_scratch);

        // Compute y = g_new - g_old
        let y_vec: Vec<f64> = g.iter().zip(g_old.iter()).map(|(gn, go)| gn - go).collect();

        // Curvature safeguard (Pitfall 1): only push when sᵀy > 1e-20
        if lbfgs_dot(&s_vec, &y_vec) > 1e-20 {
            if s_hist.len() == m_lbfgs {
                s_hist.pop_front();
                y_hist.pop_front();
            }
            s_hist.push_back(s_vec);
            y_hist.push_back(y_vec);
        }

        if converged {
            break;
        }
    }

    SoftDtwBarycenterResult {
        barycenter: bary,
        n_iter,
        converged,
    }
}

/// Compute the per-evaluation-point (column) population standard deviation of the input rows.
///
/// For each column `j` (0..m), computes the population std of `{rows[i][j]}` across `n` rows.
/// A column with zero spread (or n <= 1) returns 0.0 — those coordinates will not be perturbed
/// in `perturbed_init` (restart 0 already covers the unperturbed mean).
#[allow(dead_code)]
fn per_col_std(rows: &[Vec<f64>], m: usize) -> Vec<f64> {
    let n = rows.len();
    if n <= 1 {
        return vec![0.0; m];
    }
    let n_f = n as f64;
    let mut stds = vec![0.0_f64; m];
    for j in 0..m {
        let mean = rows.iter().map(|r| r[j]).sum::<f64>() / n_f;
        let var = rows
            .iter()
            .map(|r| (r[j] - mean) * (r[j] - mean))
            .sum::<f64>()
            / n_f;
        stds[j] = var.sqrt();
    }
    stds
}

/// Build a seeded Gaussian-perturbed init for restart `k`.
///
/// Draws additive noise from `N(0, col_std[j])` for each coordinate `j`.
/// RNG is seeded `StdRng::seed_from_u64(seed + k)` per project convention.
/// Noise is drawn in a single left-to-right pass over `j` so the RNG consumption
/// order is fixed — determinism depends on this exact order.
#[allow(dead_code)]
fn perturbed_init(mean: &[f64], col_std: &[f64], seed: u64, k: u64) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed + k);
    let normal = Normal::new(0.0_f64, 1.0_f64).expect("Normal::new(0,1) always succeeds");
    mean.iter()
        .zip(col_std.iter())
        .map(|(&mu, &std)| mu + std * rng.sample(normal))
        .collect()
}

/// Deterministic multi-restart soft-DTW barycenter driver (SDTW-02, internal).
///
/// Runs the L-BFGS refiner from `n_restarts.max(1)` initializations:
/// - Restart 0: unperturbed pointwise mean (`init_barycenter_mean`).
/// - Restart k ≥ 1: mean + additive Gaussian noise scaled to per-evaluation-point std,
///   seeded with `StdRng::seed_from_u64(seed + k)`.
///
/// Returns the barycenter with the lowest total soft-DTW cost
/// `Σᵢ soft_dtw_distance(barycenter, row_i, gamma)`, with ties broken by the lowest
/// restart index (strictly-less comparison — never replaces on equal cost).
///
/// Caller note: the public config entry point (Phase 103) will expose `n_restarts` and
/// `seed`; this function is private until that API is designed.
#[allow(dead_code)]
fn soft_dtw_barycenter_multirestart(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
    n_restarts: usize,
    seed: u64,
) -> SoftDtwBarycenterResult {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return SoftDtwBarycenterResult {
            barycenter: Vec::new(),
            n_iter: 0,
            converged: true,
        };
    }

    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();

    // Guard: gamma must be strictly positive and finite — same fallback as the 4-arg entry.
    if !gamma.is_finite() || gamma <= 0.0 {
        return SoftDtwBarycenterResult {
            barycenter: init_barycenter_mean(&rows),
            n_iter: 0,
            converged: false,
        };
    }

    let mean = init_barycenter_mean(&rows);
    let col_std = per_col_std(&rows, m);
    let n_runs = n_restarts.max(1);

    let mut best_result: Option<SoftDtwBarycenterResult> = None;
    let mut best_total = f64::INFINITY;

    for k in 0..n_runs {
        let init = if k == 0 {
            mean.clone()
        } else {
            perturbed_init(&mean, &col_std, seed, k as u64)
        };

        let result = lbfgs_refine_from_init(&rows, init, gamma, max_iter, tol);

        // Compute total soft-DTW cost for this restart's barycenter
        let total: f64 = rows
            .iter()
            .map(|r| soft_dtw_distance(&result.barycenter, r, gamma))
            .sum();

        // Strictly-less comparison: ties go to the earlier (lower-index) restart
        if total < best_total {
            best_total = total;
            best_result = Some(result);
        }
    }

    // SAFETY: n_runs >= 1, so best_result is always Some after the loop.
    best_result.unwrap()
}

/// Compute the Soft-DTW barycenter of a set of time series using L-BFGS optimization.
///
/// Minimizes the total soft-DTW cost `Σᵢ soft_dtw_distance(bary, xᵢ)` with a
/// hand-rolled limited-memory L-BFGS optimizer (two-loop recursion, m=10 history,
/// backtracking Armijo line search) driven by the exact soft-DTW gradient.
/// Initialized from the pointwise mean of the input series.
///
/// # Arguments
/// * `data` - Input time series as FdMatrix (n rows x m columns)
/// * `gamma` - Soft-DTW smoothing parameter (> 0)
/// * `max_iter` - Maximum number of L-BFGS iterations
/// * `tol` - Convergence tolerance: stops when gradient ∞-norm < `tol` OR
///   relative cost change < `tol`
///
/// # Returns
/// [`SoftDtwBarycenterResult`] containing the barycenter, iteration count, and convergence flag.
///
/// # Notes
/// `gamma` is clamped defensively: non-positive or non-finite values return the pointwise-mean
/// barycenter immediately (n_iter=0, converged=false) to avoid NaN propagation. Strict
/// `InvalidParameter` validation for gamma is deferred to the Phase 103 `Result`-returning
/// config entry point (SDTW-03).
#[must_use]
pub fn soft_dtw_barycenter(
    data: &FdMatrix,
    gamma: f64,
    max_iter: usize,
    tol: f64,
) -> SoftDtwBarycenterResult {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return SoftDtwBarycenterResult {
            barycenter: Vec::new(),
            n_iter: 0,
            converged: true,
        };
    }

    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();

    // Guard: gamma must be strictly positive and finite. NaN/inf/zero silently corrupt
    // the softmin3 computation (neg_inv_gamma = -inf or NaN → all-NaN DP table →
    // false converged=true with all-NaN barycenter). Return the pointwise mean as a
    // safe fallback. Strict InvalidParameter error is deferred to the Phase 103
    // Result-returning config entry point (SDTW-03).
    if !gamma.is_finite() || gamma <= 0.0 {
        return SoftDtwBarycenterResult {
            barycenter: init_barycenter_mean(&rows),
            n_iter: 0,
            converged: false,
        };
    }

    // Delegate to the reusable single-init refiner; restart 0 of the multi-restart
    // driver calls the same path with an identical init, ensuring the public 4-arg entry
    // and multi-restart restart-0 produce identical descents.
    lbfgs_refine_from_init(&rows, init_barycenter_mean(&rows), gamma, max_iter, tol)
}

// ---- MM baseline helper (used only in tests) ---------------------------------
// Reconstructed for the `lbfgs_cost_le_mm` test so SC#1 can be verified in-crate
// after `update_barycenter` is removed from production code.

/// One MM inverse-curvature (diagonal-Newton) update step: `bary[k] -= g[k]/(2*w[k])`.
///
/// This is the removed `update_barycenter` logic, reconstructed here so that
/// `lbfgs_cost_le_mm` can compare L-BFGS cost against the MM baseline for the
/// same iteration budget (SDTW-01 / Phase 101 SC#1). NOT called from production code.
#[cfg(test)]
fn mm_step(bary: &mut [f64], rows: &[Vec<f64>], gamma: f64) {
    let m = bary.len();
    let mut grad = vec![0.0_f64; m];
    let mut weight = vec![0.0_f64; m];
    for row in rows {
        soft_dtw_accumulate_gradient_and_weight(bary, row, gamma, &mut grad, &mut weight);
    }
    for ((b, g), w) in bary.iter_mut().zip(grad.iter()).zip(weight.iter()) {
        if *w > 1e-12 {
            *b -= g / (2.0 * w);
        }
    }
}

#[cfg(test)]
mod differentiable_tests {
    use super::*;
    use crate::autodiff::Dual;

    // Spanning, non-degenerate 5-point series (multiple DP paths exercised).
    const X: [f64; 5] = [0.1, 0.4, 0.9, 1.2, 0.7];
    const Y: [f64; 5] = [0.2, 0.3, 1.0, 1.1, 0.6];

    /// Dual gradient w.r.t. x[k] via a single seeded forward pass.
    fn dual_grad(x: &[f64], y: &[f64], gamma: f64) -> Vec<f64> {
        let y_dual: Vec<Dual> = y.iter().map(|&v| Dual::constant(v)).collect();
        (0..x.len())
            .map(|k| {
                let x_dual: Vec<Dual> = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| {
                        if i == k {
                            Dual::seed(v)
                        } else {
                            Dual::constant(v)
                        }
                    })
                    .collect();
                soft_dtw_distance_generic(&x_dual, &y_dual, gamma)
                    .extract()
                    .1
            })
            .collect()
    }

    /// Corrected soft-DTW gradient oracle (forward + backward + accumulate),
    /// retained as an INDEPENDENT cross-check reference for the fixed
    /// `soft_dtw_backward`.
    ///
    /// Phase 78 (CORR-01) fixed the endpoint-seed bug in the shipped
    /// `soft_dtw_backward` by inserting the same `if i == n && j == m {
    /// continue; }` guard that this helper uses.  The two implementations are
    /// now equivalent, but this function is kept as a structurally-independent
    /// reimplementation: it is never called from production code, and
    /// `soft_dtw_backward` does not delegate to it, so it remains a valid
    /// independent cross-check alongside the Dual forward-mode path and
    /// central finite differences (SC #3).
    fn corrected_oracle_gradient(bary: &[f64], xi: &[f64], gamma: f64) -> Vec<f64> {
        let n = bary.len();
        let m = xi.len();
        // Forward pass (identical recurrence to soft_dtw_distance).
        let mut r = vec![vec![f64::INFINITY; m + 1]; n + 1];
        r[0][0] = 0.0;
        for i in 1..=n {
            for j in 1..=m {
                let d = bary[i - 1] - xi[j - 1];
                let cost = d * d;
                r[i][j] = cost + softmin3(r[i - 1][j], r[i][j - 1], r[i - 1][j - 1], gamma);
            }
        }
        // Backward pass with the endpoint boundary preserved.
        let mut e = vec![vec![0.0; m + 2]; n + 2];
        e[n][m] = 1.0;
        for i in (1..=n).rev() {
            for j in (1..=m).rev() {
                if i == n && j == m {
                    continue; // preserve the E[n][m] = 1.0 seed
                }
                let a = if i < n {
                    e[i + 1][j] * (-(r[i][j] - softmin3_val(&r, i + 1, j, gamma)) / gamma).exp()
                } else {
                    0.0
                };
                let b = if j < m {
                    e[i][j + 1] * (-(r[i][j] - softmin3_val(&r, i, j + 1, gamma)) / gamma).exp()
                } else {
                    0.0
                };
                let c = if i < n && j < m {
                    e[i + 1][j + 1]
                        * (-(r[i][j] - softmin3_val(&r, i + 1, j + 1, gamma)) / gamma).exp()
                } else {
                    0.0
                };
                e[i][j] = a + b + c;
            }
        }
        // Accumulate d(sdtw)/d(bary[k]).
        let mut grad = vec![0.0; n];
        for k in 1..=n {
            let mut g = 0.0;
            for j in 1..=m {
                g += e[k][j] * 2.0 * (bary[k - 1] - xi[j - 1]);
            }
            grad[k - 1] = g;
        }
        grad
    }

    #[test]
    fn dual_gradient_vs_oracle() {
        // SC #1: forward-mode Dual gradient must match the (corrected) hand-
        // written soft-DTW gradient oracle (bary=x, xi=y) to <=1e-9.
        let gamma = 1.0;
        let dual = dual_grad(&X, &Y, gamma);
        let oracle = corrected_oracle_gradient(&X, &Y, gamma);
        for k in 0..X.len() {
            assert!(
                (dual[k] - oracle[k]).abs() <= 1e-9,
                "k={k}: dual {} vs oracle {}",
                dual[k],
                oracle[k]
            );
        }
    }

    #[test]
    fn f64_parity() {
        // SC #2: generic core at f64 reproduces the original soft_dtw_distance.
        for &gamma in &[0.1, 1.0, 10.0] {
            let generic = soft_dtw_distance_generic::<f64>(&X, &Y, gamma);
            let original = soft_dtw_distance(&X, &Y, gamma);
            assert!(
                (generic - original).abs() <= 1e-12,
                "gamma={gamma}: generic {generic} vs original {original}"
            );
            // Bit-identical is expected (delegation): assert exact equality too.
            assert_eq!(generic, original, "gamma={gamma}: not bit-identical");
        }
        // Longer series still parity.
        let xl: Vec<f64> = (0..20).map(|i| (i as f64 * 0.31).sin()).collect();
        let yl: Vec<f64> = (0..20).map(|i| (i as f64 * 0.27).cos()).collect();
        assert_eq!(
            soft_dtw_distance_generic::<f64>(&xl, &yl, 1.0),
            soft_dtw_distance(&xl, &yl, 1.0)
        );
    }

    /// SC#1(a)+(c): prove the fixed `soft_dtw_backward` returns a non-zero E matrix
    /// and that the shipped gradient matches both `corrected_oracle_gradient` (the
    /// independent reimplementation) and the Dual path within ~1e-6 relative
    /// tolerance.  This test FAILS on the buggy code (endpoint seed zeroed) and
    /// PASSES after the CORR-01 endpoint-skip guard is applied.
    #[test]
    fn soft_dtw_backward_nonzero_and_matches_oracle() {
        let gamma = 1.0;

        // SC#1(a): E matrix must be non-zero for non-identical input.
        let r = soft_dtw_forward(&X, &Y, gamma);
        let e = soft_dtw_backward(&X, &Y, &r, gamma);
        let e_nonzero = e.iter().flatten().any(|&v| v.abs() > 1e-12);
        assert!(
            e_nonzero,
            "E matrix must be non-zero after CORR-01 fix (all-zero indicates endpoint seed was zeroed)"
        );

        // SC#1(c): shipped accumulated gradient must match oracle and Dual within ~1e-6 relative tol.
        let n = X.len();
        let mut shipped_grad = vec![0.0; n];
        let mut scratch_weight = vec![0.0; n];
        soft_dtw_accumulate_gradient_and_weight(
            &X,
            &Y,
            gamma,
            &mut shipped_grad,
            &mut scratch_weight,
        );

        let oracle = corrected_oracle_gradient(&X, &Y, gamma);
        let dual = dual_grad(&X, &Y, gamma);

        for k in 0..n {
            // Relative tolerance with denominator guard.
            let rel_oracle = (shipped_grad[k] - oracle[k]).abs() / oracle[k].abs().max(1e-12);
            assert!(
                rel_oracle <= 1e-6,
                "k={k}: shipped grad {:.8} vs oracle {:.8} (rel err {:.2e} > 1e-6)",
                shipped_grad[k],
                oracle[k],
                rel_oracle
            );

            let rel_dual = (shipped_grad[k] - dual[k]).abs() / dual[k].abs().max(1e-12);
            assert!(
                rel_dual <= 1e-6,
                "k={k}: shipped grad {:.8} vs Dual {:.8} (rel err {:.2e} > 1e-6)",
                shipped_grad[k],
                dual[k],
                rel_dual
            );
        }
    }

    #[test]
    fn dual_gradient_vs_fd() {
        // SC #3: Dual gradient vs central finite differences, multiple gammas.
        let h = 1e-8;
        for &gamma in &[0.5, 1.0, 2.0] {
            let dual = dual_grad(&X, &Y, gamma);
            for k in 0..X.len() {
                let mut xp = X.to_vec();
                let mut xm = X.to_vec();
                xp[k] += h;
                xm[k] -= h;
                let fd = (soft_dtw_distance_generic::<f64>(&xp, &Y, gamma)
                    - soft_dtw_distance_generic::<f64>(&xm, &Y, gamma))
                    / (2.0 * h);
                assert!(
                    (dual[k] - fd).abs() <= 1e-6,
                    "gamma={gamma} k={k}: dual {} vs fd {}",
                    dual[k],
                    fd
                );
            }
        }
    }
}

/// SDTW-01 unit tests for the in-crate L-BFGS barycenter optimizer.
///
/// Covers Phase 101 success criteria:
/// - cost_monotone: the Armijo line search guarantees cost non-increase at each step
/// - lbfgs_boundedness_convex_hull: barycenter stays within data's convex hull
/// - lbfgs_max_iter_and_tol_stop: max_iter and tol stopping rules are honored
/// - lbfgs_deterministic: identical inputs produce bit-identical results (pure f64)
/// - lbfgs_cost_le_mm: L-BFGS cost ≤ MM baseline for the same budget (Phase 101 SC#1)
#[cfg(test)]
mod barycenter_lbfgs_tests {
    use super::*;
    use crate::matrix::FdMatrix;

    /// Build a small FdMatrix from a list of row vecs.
    fn mat_from_rows(rows: &[Vec<f64>]) -> FdMatrix {
        let n = rows.len();
        let m = rows[0].len();
        let mut col_major = vec![0.0_f64; n * m];
        for j in 0..m {
            for i in 0..n {
                col_major[i + j * n] = rows[i][j];
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    /// 5-row sinusoidal / quadratic curved dataset for L-BFGS tests.
    fn curved_data_5rows(m: usize) -> FdMatrix {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
        let rows: Vec<Vec<f64>> = (0..5)
            .map(|i| {
                let phase = i as f64 * 0.3;
                t.iter()
                    .map(|&s| (2.0 * std::f64::consts::PI * s + phase).sin())
                    .collect()
            })
            .collect();
        mat_from_rows(&rows)
    }

    /// SDTW-01 SC: the L-BFGS Armijo line search guarantees that each accepted step
    /// satisfies f(k+1) ≤ f(k) + c1*alpha*slope (sufficient decrease condition),
    /// so the cost sequence produced by separate runs with increasing budgets must
    /// be non-increasing.  We verify this by running for 1, 2, …, 10 iterations and
    /// checking that cost(k+1) ≤ cost(k) + small fp tolerance.
    #[test]
    fn cost_monotone() {
        let m = 30_usize;
        let data = curved_data_5rows(m);
        let gamma = 1.0_f64;
        let tol = 1e-12; // tiny tol so we only stop on max_iter

        // Each call is independent (restarts from the same pointwise-mean init).
        // Because the optimizer is deterministic, budget=k+1 reproduces all k steps of
        // budget=k and then takes one additional Armijo-guaranteed descent step.
        // So costs across runs are non-increasing iff (a) determinism holds and (b)
        // Armijo is satisfied — both properties we want to verify.
        // Note: this validates cross-run monotonicity across independent increasing-budget
        // restarts, not intra-run step descent within a single call.
        let mut prev_cost = f64::INFINITY;
        for budget in 1..=10_usize {
            let result = soft_dtw_barycenter(&data, gamma, budget, tol);
            let rows: Vec<Vec<f64>> = (0..data.nrows()).map(|i| data.row(i)).collect();
            let cost: f64 = rows
                .iter()
                .map(|r| soft_dtw_distance(&result.barycenter, r, gamma))
                .sum();
            assert!(
                cost <= prev_cost + 1e-8,
                "cost increased from budget {} to {}: {:.10} -> {:.10}",
                budget - 1,
                budget,
                prev_cost,
                cost
            );
            prev_cost = cost.min(prev_cost);
        }
    }

    /// SDTW-01 SC: the returned barycenter must lie within the data's convex hull
    /// (coordinate-wise bounding box) and contain no NaN or inf values.
    #[test]
    fn lbfgs_boundedness_convex_hull() {
        let m = 30_usize;
        let data = curved_data_5rows(m);
        let gamma = 1.0_f64;
        let result = soft_dtw_barycenter(&data, gamma, 20, 1e-6);

        assert!(
            result.barycenter.len() == m,
            "barycenter length mismatch: {} vs {m}",
            result.barycenter.len()
        );

        // Compute column-wise min/max across all input rows
        let rows: Vec<Vec<f64>> = (0..data.nrows()).map(|i| data.row(i)).collect();
        for j in 0..m {
            let col_min = rows.iter().map(|r| r[j]).fold(f64::INFINITY, f64::min);
            let col_max = rows.iter().map(|r| r[j]).fold(f64::NEG_INFINITY, f64::max);
            let bj = result.barycenter[j];
            assert!(
                bj.is_finite(),
                "barycenter[{j}] = {bj} is not finite (NaN or inf)"
            );
            // Soft-DTW barycenter may lie slightly outside the bounding box due to
            // the elastic alignment, but must not diverge — allow 10% slack.
            let range = (col_max - col_min).max(1e-6);
            assert!(
                bj >= col_min - 0.1 * range && bj <= col_max + 0.1 * range,
                "barycenter[{j}] = {bj:.6} outside 10%-slack convex hull [{:.6}, {:.6}]",
                col_min - 0.1 * range,
                col_max + 0.1 * range
            );
        }
    }

    /// SDTW-01 SC: max_iter and tol stopping rules are honored.
    /// - With max_iter=3 on a non-trivial problem, n_iter <= 3 and converged == false.
    /// - With large max_iter and loose tol, converged == true and n_iter <= max_iter.
    #[test]
    fn lbfgs_max_iter_and_tol_stop() {
        let m = 30_usize;
        let data = curved_data_5rows(m);
        let gamma = 1.0_f64;

        // Case 1: tight budget — must stop at max_iter, not converge
        let result_tight = soft_dtw_barycenter(&data, gamma, 3, 1e-15);
        assert!(
            result_tight.n_iter <= 3,
            "n_iter ({}) exceeded max_iter=3",
            result_tight.n_iter
        );
        assert!(
            !result_tight.converged,
            "should not converge in 3 iterations with tol=1e-15"
        );

        // Case 2: large budget + loose tol — must converge
        let result_loose = soft_dtw_barycenter(&data, gamma, 500, 1e-2);
        assert!(
            result_loose.converged,
            "should converge with max_iter=500 and loose tol=1e-2 (n_iter={})",
            result_loose.n_iter
        );
        assert!(
            result_loose.n_iter <= 500,
            "n_iter ({}) exceeded max_iter=500",
            result_loose.n_iter
        );
    }

    /// SDTW-01 SC: two identical calls produce bit-identical barycenter vectors.
    /// The L-BFGS optimizer is pure f64 with no randomness — determinism is required.
    #[test]
    fn lbfgs_deterministic() {
        let m = 30_usize;
        let data = curved_data_5rows(m);
        let gamma = 1.0_f64;

        let result1 = soft_dtw_barycenter(&data, gamma, 20, 1e-6);
        let result2 = soft_dtw_barycenter(&data, gamma, 20, 1e-6);

        assert_eq!(
            result1.n_iter, result2.n_iter,
            "n_iter differs between two identical calls: {} vs {}",
            result1.n_iter, result2.n_iter
        );
        assert_eq!(
            result1.converged, result2.converged,
            "converged flag differs: {} vs {}",
            result1.converged, result2.converged
        );
        for (j, (a, b)) in result1
            .barycenter
            .iter()
            .zip(result2.barycenter.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "barycenter[{j}] not bit-identical: {a} vs {b}"
            );
        }
    }

    /// SDTW-01 / Phase 101 SC#1: L-BFGS reaches total soft-DTW cost ≤ MM baseline
    /// for the same iteration budget, and strictly lower on a curved objective.
    ///
    /// The MM baseline is reconstructed in-test using the removed inverse-curvature
    /// step logic (g/(2w) per coordinate) so the comparison is self-contained.
    /// This is the unit-level cost-comparison required by 101-VALIDATION.md.
    /// The fuller SDTW-04 escape-from-local-minimum demonstration is deferred to Phase 104.
    #[test]
    fn lbfgs_cost_le_mm() {
        // Use a curved (sinusoidal, multi-row) dataset where the MM step behaves
        // differently from L-BFGS (both start from the same pointwise-mean init).
        let m = 30_usize;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        // 5 rows with distinct phases and amplitudes to create a non-trivial curved objective
        let rows: Vec<Vec<f64>> = vec![
            t.iter()
                .map(|&s| (2.0 * std::f64::consts::PI * s).sin())
                .collect(),
            t.iter()
                .map(|&s| (2.0 * std::f64::consts::PI * s + 0.5).sin())
                .collect(),
            t.iter()
                .map(|&s| 0.8 * (2.0 * std::f64::consts::PI * s + 1.0).sin())
                .collect(),
            t.iter()
                .map(|&s| 1.2 * (2.0 * std::f64::consts::PI * s - 0.4).sin())
                .collect(),
            t.iter()
                .map(|&s| (2.0 * std::f64::consts::PI * s + 1.5).sin())
                .collect(),
        ];
        let data = mat_from_rows(&rows);
        let gamma = 1.0_f64;
        let budget = 10_usize; // same number of gradient steps for both

        // --- L-BFGS cost ---
        let lbfgs_result = soft_dtw_barycenter(&data, gamma, budget, 1e-15);
        let lbfgs_cost: f64 = rows
            .iter()
            .map(|r| soft_dtw_distance(&lbfgs_result.barycenter, r, gamma))
            .sum();

        // --- MM baseline cost (reconstructed inverse-curvature step) ---
        let mut mm_bary = init_barycenter_mean(&rows);
        for _ in 0..budget {
            mm_step(&mut mm_bary, &rows, gamma);
        }
        let mm_cost: f64 = rows
            .iter()
            .map(|r| soft_dtw_distance(&mm_bary, r, gamma))
            .sum();

        // SC#1a: L-BFGS cost must be no worse than MM (with tiny eps for fp noise)
        let tiny_eps = 1e-6 * mm_cost.abs().max(1.0);
        assert!(
            lbfgs_cost <= mm_cost + tiny_eps,
            "L-BFGS cost ({lbfgs_cost:.8}) > MM cost ({mm_cost:.8}) + eps ({tiny_eps:.2e}) — \
             L-BFGS should be no worse than MM for the same budget"
        );

        // SC#1b: L-BFGS must be strictly lower on this curved case
        // (L-BFGS uses curvature information; MM is a first-order diagonal step)
        assert!(
            lbfgs_cost < mm_cost,
            "L-BFGS cost ({lbfgs_cost:.8}) is not strictly lower than MM cost ({mm_cost:.8}) \
             on the curved sinusoidal case — Phase 101 SC#1 requires strictly lower on curved objectives"
        );
    }
}

/// SDTW-02 unit tests for the deterministic multi-restart barycenter driver.
///
/// Covers Phase 102 success criteria:
/// - lowest_cost_is_argmin: returned barycenter's total cost equals the minimum over all restarts
/// - winner_le_each_restart: winner's cost <= restart-0 cost (and by construction <= all restarts)
/// - never_worse_than_single_init: multi-restart total cost <= 4-arg single-init total cost (1e-9 slack)
/// - deterministic: bit-identical barycenters for two calls with identical inputs + seed
#[cfg(test)]
mod barycenter_multirestart_tests {
    use super::*;
    use crate::matrix::FdMatrix;

    /// Build a small FdMatrix from a list of row vecs (mirrors barycenter_lbfgs_tests helper).
    fn mat_from_rows(rows: &[Vec<f64>]) -> FdMatrix {
        let n = rows.len();
        let m = rows[0].len();
        let mut col_major = vec![0.0_f64; n * m];
        for j in 0..m {
            for i in 0..n {
                col_major[i + j * n] = rows[i][j];
            }
        }
        FdMatrix::from_column_major(col_major, n, m).unwrap()
    }

    /// Asymmetric shifted-sinusoid dataset with nonzero per-column std so restarts
    /// perturb meaningfully.  Uses distinct amplitudes and phases to make the
    /// soft-DTW objective truly non-symmetric.
    fn shifted_sinusoid_data(m: usize) -> (FdMatrix, Vec<Vec<f64>>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
        // 5 rows with distinct amplitudes and phases
        let rows: Vec<Vec<f64>> = vec![
            t.iter()
                .map(|&s| (2.0 * std::f64::consts::PI * s).sin())
                .collect(),
            t.iter()
                .map(|&s| 0.9 * (2.0 * std::f64::consts::PI * s + 0.4).sin())
                .collect(),
            t.iter()
                .map(|&s| 1.1 * (2.0 * std::f64::consts::PI * s - 0.3).sin())
                .collect(),
            t.iter()
                .map(|&s| 0.7 * (2.0 * std::f64::consts::PI * s + 0.8).sin())
                .collect(),
            t.iter()
                .map(|&s| 1.2 * (2.0 * std::f64::consts::PI * s - 0.6).sin())
                .collect(),
        ];
        let mat = mat_from_rows(&rows);
        (mat, rows)
    }

    /// Compute total soft-DTW cost of `bary` against all rows.
    fn total_cost(bary: &[f64], rows: &[Vec<f64>], gamma: f64) -> f64 {
        rows.iter().map(|r| soft_dtw_distance(bary, r, gamma)).sum()
    }

    /// SDTW-02: the returned barycenter's total cost equals the true argmin over all restarts.
    ///
    /// Recomputes each restart's total cost independently (using the same seeds/inits as
    /// the driver) and asserts the driver's answer matches the manual minimum.
    #[test]
    fn lowest_cost_is_argmin() {
        let m = 30_usize;
        let (data, rows) = shifted_sinusoid_data(m);
        let gamma = 0.1_f64;
        let max_iter = 50_usize;
        let tol = 1e-6_f64;
        let n_restarts = 4_usize;
        let seed = 42_u64;

        // Run the driver
        let driver_result =
            soft_dtw_barycenter_multirestart(&data, gamma, max_iter, tol, n_restarts, seed);
        let driver_cost = total_cost(&driver_result.barycenter, &rows, gamma);

        // Independently recompute each restart to find the manual argmin
        let mean = init_barycenter_mean(&rows);
        let col_std = per_col_std(&rows, m);
        let mut manual_best_cost = f64::INFINITY;

        for k in 0..n_restarts {
            let init = if k == 0 {
                mean.clone()
            } else {
                perturbed_init(&mean, &col_std, seed, k as u64)
            };
            let result = lbfgs_refine_from_init(&rows, init, gamma, max_iter, tol);
            let cost = total_cost(&result.barycenter, &rows, gamma);
            if cost < manual_best_cost {
                manual_best_cost = cost;
            }
        }

        // Driver cost must equal the manually computed minimum (within fp rounding)
        assert!(
            (driver_cost - manual_best_cost).abs() <= 1e-9 * manual_best_cost.abs().max(1.0),
            "driver cost {driver_cost:.10} != manual argmin {manual_best_cost:.10}"
        );
    }

    /// SDTW-02: the winner's total cost is <= restart-0's (unperturbed mean) total cost.
    ///
    /// Restart 0 is the deterministic pointwise-mean init; the multi-restart result can only
    /// be as good or better (never worse).
    #[test]
    fn winner_le_each_restart() {
        let m = 30_usize;
        let (data, rows) = shifted_sinusoid_data(m);
        let gamma = 0.1_f64;
        let max_iter = 50_usize;
        let tol = 1e-6_f64;
        let n_restarts = 4_usize;
        let seed = 42_u64;

        // Run the driver
        let winner =
            soft_dtw_barycenter_multirestart(&data, gamma, max_iter, tol, n_restarts, seed);
        let winner_cost = total_cost(&winner.barycenter, &rows, gamma);

        // Compute every restart's cost independently and verify winner is cheapest
        let mean = init_barycenter_mean(&rows);
        let col_std = per_col_std(&rows, m);

        for k in 0..n_restarts {
            let init = if k == 0 {
                mean.clone()
            } else {
                perturbed_init(&mean, &col_std, seed, k as u64)
            };
            let result = lbfgs_refine_from_init(&rows, init, gamma, max_iter, tol);
            let restart_cost = total_cost(&result.barycenter, &rows, gamma);
            assert!(
                winner_cost <= restart_cost + 1e-9,
                "winner cost {winner_cost:.10} > restart {k} cost {restart_cost:.10}"
            );
        }
    }

    /// SDTW-02: multi-restart total cost <= single-init (4-arg) total cost (with 1e-9 slack).
    ///
    /// Because restart 0 is the same unperturbed mean as the 4-arg entry, the multi-restart
    /// result is never worse than the 4-arg result.
    #[test]
    fn never_worse_than_single_init() {
        let m = 30_usize;
        let (data, rows) = shifted_sinusoid_data(m);
        let gamma = 0.1_f64;
        let max_iter = 50_usize;
        let tol = 1e-6_f64;
        let n_restarts = 3_usize;
        let seed = 99_u64;

        // Multi-restart result
        let multi = soft_dtw_barycenter_multirestart(&data, gamma, max_iter, tol, n_restarts, seed);
        let cost_multi = total_cost(&multi.barycenter, &rows, gamma);

        // Single-init (4-arg) result — identical to restart 0
        let single = soft_dtw_barycenter(&data, gamma, max_iter, tol);
        let cost_single = total_cost(&single.barycenter, &rows, gamma);

        assert!(
            cost_multi <= cost_single + 1e-9,
            "multi-restart cost {cost_multi:.10} > single-init cost {cost_single:.10} + 1e-9"
        );
    }

    /// SDTW-02: two calls with identical (data, gamma, max_iter, tol, n_restarts, seed) produce
    /// bit-identical barycenters.
    #[test]
    fn deterministic() {
        let m = 30_usize;
        let (data, _rows) = shifted_sinusoid_data(m);
        let gamma = 0.1_f64;
        let max_iter = 50_usize;
        let tol = 1e-6_f64;
        let n_restarts = 4_usize;
        let seed = 42_u64;

        let result1 =
            soft_dtw_barycenter_multirestart(&data, gamma, max_iter, tol, n_restarts, seed);
        let result2 =
            soft_dtw_barycenter_multirestart(&data, gamma, max_iter, tol, n_restarts, seed);

        assert_eq!(
            result1.barycenter.len(),
            result2.barycenter.len(),
            "barycenter length mismatch between two identical calls"
        );
        for (j, (a, b)) in result1
            .barycenter
            .iter()
            .zip(result2.barycenter.iter())
            .enumerate()
        {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "barycenter[{j}] not bit-identical between two calls: {a} vs {b}"
            );
        }
    }
}
