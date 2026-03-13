//! Soft-DTW: differentiable relaxation of DTW using log-sum-exp softmin.
//!
//! Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for
//! Time-Series" (ICML 2017).

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::{cross_distance_matrix, self_distance_matrix};

/// Result of the Soft-DTW barycenter computation.
#[derive(Debug, Clone)]
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

/// Compute Soft-DTW distance between two 1D time series.
///
/// Uses squared Euclidean cost and 2-row DP with O(m) memory.
///
/// # Arguments
/// * `x` - First time series
/// * `y` - Second time series
/// * `gamma` - Smoothing parameter (> 0). Smaller = closer to hard DTW.
pub fn soft_dtw_distance(x: &[f64], y: &[f64], gamma: f64) -> f64 {
    let n = x.len();
    let m = y.len();
    if n == 0 || m == 0 {
        return 0.0;
    }

    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;

    for i in 1..=n {
        curr.fill(f64::INFINITY);
        for j in 1..=m {
            let d = x[i - 1] - y[j - 1];
            let cost = d * d;
            curr[j] = cost + softmin3(prev[j], curr[j - 1], prev[j - 1], gamma);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
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
