//! Soft-DTW: differentiable relaxation of DTW using log-sum-exp softmin.
//!
//! Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for
//! Time-Series" (ICML 2017).

use crate::autodiff::Scalar;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

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
pub fn soft_dtw_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
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
pub fn soft_dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
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
pub fn soft_dtw_div_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
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
pub fn soft_dtw_div_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
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
fn soft_dtw_accumulate_gradient(bary: &[f64], xi: &[f64], gamma: f64, grad: &mut [f64]) {
    let m = bary.len();
    let r = soft_dtw_forward(bary, xi, gamma);
    let e = soft_dtw_backward(bary, xi, &r, gamma);
    for k in 1..=m {
        let mut g = 0.0;
        for j in 1..=xi.len() {
            g += e[k][j] * 2.0 * (bary[k - 1] - xi[j - 1]);
        }
        grad[k - 1] += g;
    }
}

/// Apply one gradient descent step and check convergence.
///
/// Returns `true` if the relative change is below `tol`.
fn update_barycenter(bary: &mut [f64], grad: &[f64], lr: f64, tol: f64) -> bool {
    let mut max_change = 0.0_f64;
    let mut max_val = 0.0_f64;
    for (b, &g) in bary.iter_mut().zip(grad.iter()) {
        let update = lr * g;
        *b -= update;
        max_change = max_change.max(update.abs());
        max_val = max_val.max(b.abs());
    }
    max_val > 0.0 && max_change / max_val < tol
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

/// Compute the Soft-DTW barycenter of a set of time series using gradient descent.
///
/// # Arguments
/// * `data` - Input time series as FdMatrix (n rows x m columns)
/// * `gamma` - Soft-DTW smoothing parameter
/// * `max_iter` - Maximum number of gradient descent iterations
/// * `tol` - Convergence tolerance (relative change in barycenter)
///
/// # Returns
/// [`SoftDtwBarycenterResult`] containing the barycenter, iteration count, and convergence flag.
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
    let mut bary = init_barycenter_mean(&rows);
    let lr = 1.0 / n as f64;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut grad = vec![0.0; m];
        for row in &rows {
            soft_dtw_accumulate_gradient(&bary, row, gamma, &mut grad);
        }

        if update_barycenter(&mut bary, &grad, lr, tol) {
            converged = true;
            break;
        }
    }

    SoftDtwBarycenterResult {
        barycenter: bary,
        n_iter,
        converged,
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
        soft_dtw_accumulate_gradient(&X, &Y, gamma, &mut shipped_grad);

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
