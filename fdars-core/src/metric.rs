//! Distance metrics and semimetrics for functional data.
//!
//! This module provides various distance measures including:
//! - Lp distances (L1, L2, L∞)
//! - Hausdorff distance
//! - Dynamic Time Warping (DTW)
//! - Fourier-based semimetric
//! - Horizontal shift semimetric
//! - Soft-DTW (differentiable DTW relaxation)
//! - Kullback-Leibler divergence

use crate::helpers::{simpsons_weights, simpsons_weights_2d};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Compute weighted Lp distance between two rows of an FdMatrix, branching on p.
///
/// Avoids `powf(p)` in the inner loop for p=1 and p=2, which are ~50-100× faster.
#[inline(always)]
fn lp_weighted_distance(
    data1: &FdMatrix,
    i: usize,
    data2: &FdMatrix,
    j: usize,
    weights: &[f64],
    n_points: usize,
    p: f64,
) -> f64 {
    if (p - 2.0).abs() < 1e-14 {
        let mut sum = 0.0;
        for k in 0..n_points {
            let diff = data1[(i, k)] - data2[(j, k)];
            sum += diff * diff * weights[k];
        }
        sum.sqrt()
    } else if (p - 1.0).abs() < 1e-14 {
        let mut sum = 0.0;
        for k in 0..n_points {
            sum += (data1[(i, k)] - data2[(j, k)]).abs() * weights[k];
        }
        sum
    } else {
        let mut sum = 0.0;
        for k in 0..n_points {
            sum += (data1[(i, k)] - data2[(j, k)]).abs().powf(p) * weights[k];
        }
        sum.powf(1.0 / p)
    }
}

/// Merge base integration weights with optional user weights.
fn merge_weights(base: Vec<f64>, user_weights: &[f64]) -> Vec<f64> {
    if user_weights.len() == base.len() {
        base.iter()
            .zip(user_weights.iter())
            .map(|(b, u)| b * u)
            .collect()
    } else {
        base
    }
}

/// Build a symmetric distance matrix by computing upper-triangle entries in parallel.
#[inline]
fn self_distance_matrix(n: usize, compute: impl Fn(usize, usize) -> f64 + Sync) -> FdMatrix {
    // Pre-allocate flat buffer for upper triangle: n*(n-1)/2 entries
    let tri_len = n * (n - 1) / 2;
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| ((i + 1)..n).map(|j| compute(i, j)).collect::<Vec<_>>())
        .collect();
    debug_assert_eq!(upper_vals.len(), tri_len);
    // Scatter into symmetric matrix
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

/// Build an n1×n2 distance matrix by computing all pairs in parallel.
#[inline]
fn cross_distance_matrix(
    n1: usize,
    n2: usize,
    compute: impl Fn(usize, usize) -> f64 + Sync,
) -> FdMatrix {
    // Pre-allocate flat buffer for all n1*n2 entries (row-major order)
    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| (0..n2).map(|j| compute(i, j)).collect::<Vec<_>>())
        .collect();
    // Scatter row-major vals into column-major FdMatrix
    let mut dist = FdMatrix::zeros(n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            dist[(i, j)] = vals[i * n2 + j];
        }
    }
    dist
}

/// Compute Lp distance matrix between two sets of functional data.
///
/// # Arguments
/// * `data1` - First dataset matrix (n1 rows x n_points columns)
/// * `data2` - Second dataset matrix (n2 rows x n_points columns)
/// * `argvals` - Evaluation points for integration
/// * `p` - Order of the norm
/// * `user_weights` - Optional user weights (empty slice for none)
///
/// # Returns
/// Distance matrix (n1 rows x n2 columns)
pub fn lp_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    p: f64,
    user_weights: &[f64],
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let n_points = data1.ncols();

    if n1 == 0 || n2 == 0 || n_points == 0 || argvals.len() != n_points || data2.ncols() != n_points
    {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights(argvals), user_weights);
    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| lp_weighted_distance(data1, i, data2, j, &weights, n_points, p))
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

/// Compute Lp distance matrix for self-distances (symmetric).
///
/// Returns symmetric distance matrix (n rows x n columns).
pub fn lp_self_1d(data: &FdMatrix, argvals: &[f64], p: f64, user_weights: &[f64]) -> FdMatrix {
    let n = data.nrows();
    let n_points = data.ncols();

    if n == 0 || n_points == 0 || argvals.len() != n_points {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights(argvals), user_weights);
    // Inline the self-distance pattern rather than going through self_distance_matrix
    // to ensure LLVM can fully optimize the tight p=2 inner loop.
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| lp_weighted_distance(data, i, data, j, &weights, n_points, p))
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

/// Compute Lp distance for 2D functional data (surfaces).
pub fn lp_cross_2d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
    p: f64,
    user_weights: &[f64],
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let n_points = argvals_s.len() * argvals_t.len();
    if n1 == 0 || n2 == 0 || n_points == 0 || data1.ncols() != n_points || data2.ncols() != n_points
    {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights_2d(argvals_s, argvals_t), user_weights);
    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| lp_weighted_distance(data1, i, data2, j, &weights, n_points, p))
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

/// Compute Lp self-distance matrix for 2D functional data (symmetric).
pub fn lp_self_2d(
    data: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
    p: f64,
    user_weights: &[f64],
) -> FdMatrix {
    let n = data.nrows();
    let n_points = argvals_s.len() * argvals_t.len();
    if n == 0 || n_points == 0 || data.ncols() != n_points {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights_2d(argvals_s, argvals_t), user_weights);
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| lp_weighted_distance(data, i, data, j, &weights, n_points, p))
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

/// Precompute squared time differences for Hausdorff distance.
fn precompute_mtt(argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    let mut result = Vec::with_capacity(m * m);
    for s in 0..m {
        for t in 0..m {
            let diff = argvals[s] - argvals[t];
            result.push(diff * diff);
        }
    }
    result
}

/// Compute directed Hausdorff distance squared from one curve to another.
fn directed_hausdorff_sq(
    data_i: &FdMatrix,
    row_i: usize,
    data_j: &FdMatrix,
    row_j: usize,
    m: usize,
    mtt: &[f64],
) -> f64 {
    (0..m)
        .map(|s| {
            let x_s = data_i[(row_i, s)];
            (0..m)
                .map(|t| {
                    let y_t = data_j[(row_j, t)];
                    let val_diff = x_s - y_t;
                    val_diff * val_diff + mtt[s * m + t]
                })
                .fold(f64::INFINITY, |a, b| a.min(b))
        })
        .fold(f64::NEG_INFINITY, |a, b| a.max(b))
}

/// Compute Hausdorff distance matrix for self-distances (symmetric).
///
/// The Hausdorff distance treats curves as sets of points (t, f(t)) in 2D space.
pub fn hausdorff_self_1d(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m == 0 || argvals.len() != m {
        return FdMatrix::zeros(0, 0);
    }

    let mtt = precompute_mtt(argvals);
    self_distance_matrix(n, |i, j| {
        let d_ij = directed_hausdorff_sq(data, i, data, j, m, &mtt);
        let d_ji = directed_hausdorff_sq(data, j, data, i, m, &mtt);
        d_ij.max(d_ji).sqrt()
    })
}

/// Compute Hausdorff cross-distances for 1D functional data.
pub fn hausdorff_cross_1d(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();

    if n1 == 0 || n2 == 0 || m == 0 || argvals.len() != m || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }

    let mtt = precompute_mtt(argvals);
    cross_distance_matrix(n1, n2, |i, j| {
        let d_ij = directed_hausdorff_sq(data1, i, data2, j, m, &mtt);
        let d_ji = directed_hausdorff_sq(data2, j, data1, i, m, &mtt);
        d_ij.max(d_ji).sqrt()
    })
}

/// Run the two-row DTW dynamic programming loop with a given cost function.
#[inline]
fn dtw_dp_loop(x: &[f64], y: &[f64], w: usize, cost_fn: impl Fn(f64, f64) -> f64) -> f64 {
    let n = x.len();
    let m = y.len();
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;

    // Precompute band bounds as usize to avoid isize casts in inner loop
    let bounds: Vec<(usize, usize)> = (1..=n)
        .map(|i| {
            let r_i = i + 1;
            let j_start = (r_i as isize - w as isize).max(1) as usize;
            let j_end = (r_i + w - 1).min(m);
            (j_start, j_end)
        })
        .collect();

    for (i, &(j_start, j_end)) in bounds.iter().enumerate() {
        curr.fill(f64::INFINITY);
        for j in j_start..=j_end {
            let cost = cost_fn(x[i], y[j - 1]);
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Compute DTW distance between two time series using two-row DP.
///
/// Uses O(m) memory instead of O(nm) by keeping only two rows of the DP table.
pub fn dtw_distance(x: &[f64], y: &[f64], p: f64, w: usize) -> f64 {
    if (p - 2.0).abs() < 1e-14 {
        dtw_dp_loop(x, y, w, |a, b| {
            let d = a - b;
            d * d
        })
    } else if (p - 1.0).abs() < 1e-14 {
        dtw_dp_loop(x, y, w, |a, b| (a - b).abs())
    } else {
        dtw_dp_loop(x, y, w, |a, b| (a - b).abs().powf(p))
    }
}

/// Compute DTW distance matrix for self-distances (symmetric).
pub fn dtw_self_1d(data: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm = data.to_row_major();
    self_distance_matrix(n, |i, j| {
        dtw_distance(&rm[i * m..(i + 1) * m], &rm[j * m..(j + 1) * m], p, w)
    })
}

/// Compute DTW cross-distance matrix.
pub fn dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m1 = data1.ncols();
    let m2 = data2.ncols();
    if n1 == 0 || n2 == 0 || m1 == 0 || m2 == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm1 = data1.to_row_major();
    let rm2 = data2.to_row_major();
    cross_distance_matrix(n1, n2, |i, j| {
        dtw_distance(&rm1[i * m1..(i + 1) * m1], &rm2[j * m2..(j + 1) * m2], p, w)
    })
}

// ─── Soft-DTW ────────────────────────────────────────────────────────────────
// Differentiable relaxation of DTW using log-sum-exp softmin.
// Reference: Cuturi & Blondel, "Soft-DTW: a Differentiable Loss Function for
// Time-Series" (ICML 2017).

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
/// As γ→0 this approaches hard min; as γ→∞ it approaches the mean.
#[inline]
fn softmin3(a: f64, b: f64, c: f64, gamma: f64) -> f64 {
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
/// * `x` — First time series
/// * `y` — Second time series
/// * `gamma` — Smoothing parameter (> 0). Smaller = closer to hard DTW.
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

/// Compute Soft-DTW self-distance matrix (symmetric n×n).
pub fn soft_dtw_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm = data.to_row_major();
    self_distance_matrix(n, |i, j| {
        soft_dtw_distance(&rm[i * m..(i + 1) * m], &rm[j * m..(j + 1) * m], gamma)
    })
}

/// Compute Soft-DTW cross-distance matrix (n1×n2).
pub fn soft_dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m1 = data1.ncols();
    let m2 = data2.ncols();
    if n1 == 0 || n2 == 0 || m1 == 0 || m2 == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm1 = data1.to_row_major();
    let rm2 = data2.to_row_major();
    cross_distance_matrix(n1, n2, |i, j| {
        soft_dtw_distance(
            &rm1[i * m1..(i + 1) * m1],
            &rm2[j * m2..(j + 1) * m2],
            gamma,
        )
    })
}

/// Compute Soft-DTW divergence self-distance matrix (symmetric n×n).
pub fn soft_dtw_div_self_1d(data: &FdMatrix, gamma: f64) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm = data.to_row_major();
    // Pre-compute self-distances for divergence
    let self_dists: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| soft_dtw_distance(&rm[i * m..(i + 1) * m], &rm[i * m..(i + 1) * m], gamma))
        .collect();
    self_distance_matrix(n, |i, j| {
        let xy = soft_dtw_distance(&rm[i * m..(i + 1) * m], &rm[j * m..(j + 1) * m], gamma);
        xy - 0.5 * (self_dists[i] + self_dists[j])
    })
}

/// Compute Soft-DTW divergence cross-distance matrix (n1×n2).
pub fn soft_dtw_div_cross_1d(data1: &FdMatrix, data2: &FdMatrix, gamma: f64) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m1 = data1.ncols();
    let m2 = data2.ncols();
    if n1 == 0 || n2 == 0 || m1 == 0 || m2 == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rm1 = data1.to_row_major();
    let rm2 = data2.to_row_major();
    let self1: Vec<f64> = iter_maybe_parallel!(0..n1)
        .map(|i| {
            soft_dtw_distance(
                &rm1[i * m1..(i + 1) * m1],
                &rm1[i * m1..(i + 1) * m1],
                gamma,
            )
        })
        .collect();
    let self2: Vec<f64> = iter_maybe_parallel!(0..n2)
        .map(|j| {
            soft_dtw_distance(
                &rm2[j * m2..(j + 1) * m2],
                &rm2[j * m2..(j + 1) * m2],
                gamma,
            )
        })
        .collect();
    cross_distance_matrix(n1, n2, |i, j| {
        let xy = soft_dtw_distance(
            &rm1[i * m1..(i + 1) * m1],
            &rm2[j * m2..(j + 1) * m2],
            gamma,
        );
        xy - 0.5 * (self1[i] + self2[j])
    })
}

/// Full forward pass: returns the (n+1)×(m+1) R table needed for the backward pass.
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

/// Compute the Soft-DTW barycenter of a set of time series using gradient descent.
///
/// # Arguments
/// * `data` — Input time series as FdMatrix (n rows × m columns)
/// * `gamma` — Soft-DTW smoothing parameter
/// * `max_iter` — Maximum number of gradient descent iterations
/// * `tol` — Convergence tolerance (relative change in barycenter)
///
/// # Returns
/// [`SoftDtwBarycenterResult`] containing the barycenter, iteration count, and convergence flag.
/// Initialize the barycenter as the pointwise mean of all series.
fn init_barycenter_mean(rm: &[f64], n: usize, m: usize) -> Vec<f64> {
    let mut bary = vec![0.0; m];
    for i in 0..n {
        for j in 0..m {
            bary[j] += rm[i * m + j];
        }
    }
    for j in 0..m {
        bary[j] /= n as f64;
    }
    bary
}

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

    let rm = data.to_row_major();
    let mut bary = init_barycenter_mean(&rm, n, m);
    let lr = 1.0 / n as f64;
    let mut converged = false;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;

        let mut grad = vec![0.0; m];
        for i in 0..n {
            let xi = &rm[i * m..(i + 1) * m];
            soft_dtw_accumulate_gradient(&bary, xi, gamma, &mut grad);
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

/// Compute Fourier coefficients for a curve using a pre-planned FFT.
fn fft_coefficients_with_plan(data: &[f64], nfreq: usize, fft: &dyn rustfft::Fft<f64>) -> Vec<f64> {
    let n = data.len();
    let nfreq = nfreq.min(n / 2);
    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    buffer
        .iter()
        .take(nfreq + 1)
        .map(|c| c.norm() / n as f64)
        .collect()
}

/// Compute semimetric based on Fourier coefficients for self-distances.
pub fn fourier_self_1d(data: &FdMatrix, nfreq: usize) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let rm = data.to_row_major();
    let coeffs: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| fft_coefficients_with_plan(&rm[i * m..(i + 1) * m], nfreq, fft.as_ref()))
        .collect();
    self_distance_matrix(n, |i, j| {
        coeffs[i]
            .iter()
            .zip(coeffs[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    })
}

/// Compute semimetric based on Fourier coefficients for cross-distances.
pub fn fourier_cross_1d(data1: &FdMatrix, data2: &FdMatrix, nfreq: usize) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();
    if n1 == 0 || n2 == 0 || m == 0 || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let rm1 = data1.to_row_major();
    let rm2 = data2.to_row_major();
    let coeffs1: Vec<Vec<f64>> = iter_maybe_parallel!(0..n1)
        .map(|i| fft_coefficients_with_plan(&rm1[i * m..(i + 1) * m], nfreq, fft.as_ref()))
        .collect();
    let coeffs2: Vec<Vec<f64>> = iter_maybe_parallel!(0..n2)
        .map(|i| fft_coefficients_with_plan(&rm2[i * m..(i + 1) * m], nfreq, fft.as_ref()))
        .collect();
    cross_distance_matrix(n1, n2, |i, j| {
        coeffs1[i]
            .iter()
            .zip(coeffs2[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    })
}

/// Compute weighted L2 distance at a given horizontal shift.
fn shifted_l2_distance(x: &[f64], y: &[f64], weights: &[f64], shift: i32) -> Option<f64> {
    let n = x.len();
    let mut sum = 0.0;
    let mut valid_points = 0;
    for i in 0..n {
        let j = i as i32 + shift;
        if j >= 0 && (j as usize) < n {
            let diff = x[i] - y[j as usize];
            sum += weights[i] * diff * diff;
            valid_points += 1;
        }
    }
    if valid_points >= n / 2 {
        Some(sum.sqrt())
    } else {
        None
    }
}

/// Compute minimum L2 distance after horizontal shift between two curves.
fn hshift_distance(x: &[f64], y: &[f64], weights: &[f64], max_shift: usize) -> f64 {
    if x.is_empty() || y.len() != x.len() || weights.len() != x.len() {
        return f64::INFINITY;
    }
    let mut min_dist = f64::INFINITY;
    for shift in -(max_shift as i32)..=(max_shift as i32) {
        if let Some(dist) = shifted_l2_distance(x, y, weights, shift) {
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }
    min_dist
}

/// Compute semimetric based on horizontal shift for self-distances.
pub fn hshift_self_1d(data: &FdMatrix, argvals: &[f64], max_shift: usize) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || argvals.len() != m {
        return FdMatrix::zeros(0, 0);
    }
    let weights = simpsons_weights(argvals);
    let rm = data.to_row_major();
    self_distance_matrix(n, |i, j| {
        hshift_distance(
            &rm[i * m..(i + 1) * m],
            &rm[j * m..(j + 1) * m],
            &weights,
            max_shift,
        )
    })
}

/// Compute semimetric based on horizontal shift for cross-distances.
pub fn hshift_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    max_shift: usize,
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();
    if n1 == 0 || n2 == 0 || m == 0 || argvals.len() != m || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }
    let weights = simpsons_weights(argvals);
    let rm1 = data1.to_row_major();
    let rm2 = data2.to_row_major();
    cross_distance_matrix(n1, n2, |i, j| {
        hshift_distance(
            &rm1[i * m..(i + 1) * m],
            &rm2[j * m..(j + 1) * m],
            &weights,
            max_shift,
        )
    })
}

/// Compute Hausdorff distance between two 3D point clouds.
pub fn hausdorff_3d(points1: &[(f64, f64, f64)], points2: &[(f64, f64, f64)]) -> f64 {
    let h12 = points1
        .iter()
        .map(|p1| {
            points2
                .iter()
                .map(|p2| {
                    let ds = p1.0 - p2.0;
                    let dt = p1.1 - p2.1;
                    let df = p1.2 - p2.2;
                    (ds * ds + dt * dt + df * df).sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max);
    let h21 = points2
        .iter()
        .map(|p2| {
            points1
                .iter()
                .map(|p1| {
                    let ds = p1.0 - p2.0;
                    let dt = p1.1 - p2.1;
                    let df = p1.2 - p2.2;
                    (ds * ds + dt * dt + df * df).sqrt()
                })
                .fold(f64::INFINITY, f64::min)
        })
        .fold(0.0_f64, f64::max);
    h12.max(h21)
}

/// Extract 2D surface points from a matrix for Hausdorff distance computation.
fn extract_surfaces(
    data: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
) -> Vec<Vec<(f64, f64, f64)>> {
    let n = data.nrows();
    let m1 = argvals_s.len();
    let m2 = argvals_t.len();
    (0..n)
        .map(|curve| {
            let mut points = Vec::with_capacity(m1 * m2);
            for i in 0..m1 {
                for j in 0..m2 {
                    let k = i * m2 + j;
                    points.push((argvals_s[i], argvals_t[j], data[(curve, k)]));
                }
            }
            points
        })
        .collect()
}

/// Compute Hausdorff self-distance for 2D surfaces.
pub fn hausdorff_self_2d(data: &FdMatrix, argvals_s: &[f64], argvals_t: &[f64]) -> FdMatrix {
    let n = data.nrows();
    let n_points = argvals_s.len() * argvals_t.len();
    if n == 0 || n_points == 0 || data.ncols() != n_points {
        return FdMatrix::zeros(0, 0);
    }
    let surfaces = extract_surfaces(data, argvals_s, argvals_t);
    self_distance_matrix(n, |i, j| hausdorff_3d(&surfaces[i], &surfaces[j]))
}

/// Compute Hausdorff cross-distance for 2D surfaces.
pub fn hausdorff_cross_2d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals_s: &[f64],
    argvals_t: &[f64],
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let n_points = argvals_s.len() * argvals_t.len();
    if n1 == 0 || n2 == 0 || n_points == 0 || data1.ncols() != n_points || data2.ncols() != n_points
    {
        return FdMatrix::zeros(0, 0);
    }
    let surfaces1 = extract_surfaces(data1, argvals_s, argvals_t);
    let surfaces2 = extract_surfaces(data2, argvals_s, argvals_t);
    cross_distance_matrix(n1, n2, |i, j| hausdorff_3d(&surfaces1[i], &surfaces2[j]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn test_lp_self_distance() {
        let data = FdMatrix::from_column_major(vec![0.0, 1.0, 1.0, 2.0], 2, 2).unwrap();
        let argvals = vec![0.0, 1.0];
        let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
        assert!((dist[(0, 1)] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_lp_self_symmetric() {
        let n = 5;
        let m = 20;
        let argvals = uniform_grid(m);
        let mut flat = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                flat[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
            }
        }
        let data = FdMatrix::from_column_major(flat, n, m).unwrap();
        let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Distance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_lp_self_diagonal_zero() {
        let n = 4;
        let m = 15;
        let argvals = uniform_grid(m);
        let flat: Vec<f64> = (0..(n * m)).map(|i| i as f64 * 0.1).collect();
        let data = FdMatrix::from_column_major(flat, n, m).unwrap();
        let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
        for i in 0..n {
            assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
        }
    }

    #[test]
    fn test_lp_cross_shape() {
        let n1 = 3;
        let n2 = 4;
        let m = 20;
        let argvals = uniform_grid(m);
        let data1 =
            FdMatrix::from_column_major((0..(n1 * m)).map(|i| i as f64 * 0.1).collect(), n1, m)
                .unwrap();
        let data2 =
            FdMatrix::from_column_major((0..(n2 * m)).map(|i| i as f64 * 0.2).collect(), n2, m)
                .unwrap();
        let dist = lp_cross_1d(&data1, &data2, &argvals, 2.0, &[]);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
    }

    #[test]
    fn test_lp_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(lp_self_1d(&empty, &[], 2.0, &[]).is_empty());
        assert!(lp_cross_1d(&empty, &empty, &[], 2.0, &[]).is_empty());
    }

    #[test]
    fn test_dtw_distance() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let dist = dtw_distance(&x, &y, 2.0, 10);
        assert!((dist - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_dtw_distance_different() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 3.0, 4.0];
        let dist = dtw_distance(&x, &y, 1.0, 10);
        assert!(
            dist > 0.0,
            "Different curves should have positive DTW distance"
        );
    }

    #[test]
    fn test_dtw_self_symmetric() {
        let n = 4;
        let m = 15;
        let data =
            FdMatrix::from_column_major((0..(n * m)).map(|i| i as f64 * 0.1).collect(), n, m)
                .unwrap();
        let dist = dtw_self_1d(&data, 2.0, 5);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "DTW matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_dtw_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(dtw_self_1d(&empty, 2.0, 5).is_empty());
    }

    #[test]
    fn test_hausdorff_self_symmetric() {
        let n = 4;
        let m = 15;
        let argvals = uniform_grid(m);
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let dist = hausdorff_self_1d(&data, &argvals);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Hausdorff matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hausdorff_self_diagonal_zero() {
        let n = 3;
        let m = 10;
        let argvals = uniform_grid(m);
        let data =
            FdMatrix::from_column_major((0..(n * m)).map(|i| i as f64 * 0.1).collect(), n, m)
                .unwrap();
        let dist = hausdorff_self_1d(&data, &argvals);
        for i in 0..n {
            assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
        }
    }

    #[test]
    fn test_hausdorff_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(hausdorff_self_1d(&empty, &[]).is_empty());
    }

    #[test]
    fn test_fourier_self_symmetric() {
        let n = 4;
        let m = 32;
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let dist = fourier_self_1d(&data, 5);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Fourier distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_fourier_self_diagonal_zero() {
        let n = 3;
        let m = 32;
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
            n,
            m,
        )
        .unwrap();
        let dist = fourier_self_1d(&data, 8);
        for i in 0..n {
            assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
        }
    }

    #[test]
    fn test_fourier_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(fourier_self_1d(&empty, 5).is_empty());
    }

    #[test]
    fn test_hshift_self_symmetric() {
        let n = 4;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let dist = hshift_self_1d(&data, &argvals, 3);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Hshift distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hshift_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(hshift_self_1d(&empty, &[], 3).is_empty());
    }

    #[test]
    fn test_hausdorff_3d_identical() {
        let points1 = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        let points2 = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
        let dist = hausdorff_3d(&points1, &points2);
        assert!(
            dist.abs() < 1e-10,
            "Identical point sets should have zero distance"
        );
    }

    #[test]
    fn test_hausdorff_3d_different() {
        let points1 = vec![(0.0, 0.0, 0.0)];
        let points2 = vec![(1.0, 1.0, 1.0)];
        let dist = hausdorff_3d(&points1, &points2);
        let expected = (3.0_f64).sqrt();
        assert!(
            (dist - expected).abs() < 1e-10,
            "Expected {}, got {}",
            expected,
            dist
        );
    }

    #[test]
    fn test_lp_2d_symmetric() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data = FdMatrix::from_column_major(
            (0..(n * n_points)).map(|i| i as f64 * 0.1).collect(),
            n,
            n_points,
        )
        .unwrap();
        let dist = lp_self_2d(&data, &argvals_s, &argvals_t, 2.0, &[]);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "2D Lp distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_lp_2d_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(lp_self_2d(&empty, &[], &[], 2.0, &[]).is_empty());
    }

    #[test]
    fn test_hausdorff_2d_symmetric() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data = FdMatrix::from_column_major(
            (0..(n * n_points))
                .map(|i| (i as f64 * 0.1).sin())
                .collect(),
            n,
            n_points,
        )
        .unwrap();
        let dist = hausdorff_self_2d(&data, &argvals_s, &argvals_t);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "2D Hausdorff should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hausdorff_2d_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(hausdorff_self_2d(&empty, &[], &[]).is_empty());
    }

    #[test]
    fn test_hausdorff_cross_1d() {
        let n1 = 3;
        let n2 = 4;
        let m = 15;
        let argvals = uniform_grid(m);
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n1,
            m,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
            n2,
            m,
        )
        .unwrap();
        let dist = hausdorff_cross_1d(&data1, &data2, &argvals);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(
                    dist[(i, j)] >= 0.0,
                    "Hausdorff cross distance should be non-negative"
                );
                assert!(
                    dist[(i, j)].is_finite(),
                    "Hausdorff cross distance should be finite"
                );
            }
        }
        let self_dist = hausdorff_self_1d(&data1, &argvals);
        let cross_self = hausdorff_cross_1d(&data1, &data1, &argvals);
        for i in 0..n1 {
            assert!(
                (cross_self[(i, i)] - self_dist[(i, i)]).abs() < 1e-10,
                "Cross-self diagonal should match self diagonal at {}",
                i
            );
        }
    }

    #[test]
    fn test_dtw_cross_1d() {
        let n1 = 3;
        let n2 = 4;
        let m = 15;
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n1,
            m,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
            n2,
            m,
        )
        .unwrap();
        let dist = dtw_cross_1d(&data1, &data2, 2.0, 5);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(
                    dist[(i, j)] >= 0.0,
                    "DTW cross distance should be non-negative"
                );
                assert!(
                    dist[(i, j)].is_finite(),
                    "DTW cross distance should be finite"
                );
            }
        }
        let data_same = FdMatrix::from_column_major(
            (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n1,
            m,
        )
        .unwrap();
        let cross_self = dtw_cross_1d(&data_same, &data_same, 2.0, 5);
        for i in 0..n1 {
            assert!(
                cross_self[(i, i)] < 1e-8,
                "DTW self-distance should be ~0, got {}",
                cross_self[(i, i)]
            );
        }
    }

    #[test]
    fn test_fourier_cross_1d() {
        let n1 = 3;
        let n2 = 4;
        let m = 32;
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n1,
            m,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
            n2,
            m,
        )
        .unwrap();
        let dist = fourier_cross_1d(&data1, &data2, 5);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(
                    dist[(i, j)] >= 0.0,
                    "Fourier cross distance should be non-negative"
                );
                assert!(
                    dist[(i, j)].is_finite(),
                    "Fourier cross distance should be finite"
                );
            }
        }
    }

    #[test]
    fn test_hshift_cross_1d() {
        let n1 = 3;
        let n2 = 4;
        let m = 20;
        let argvals = uniform_grid(m);
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
            n1,
            m,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
            n2,
            m,
        )
        .unwrap();
        let dist = hshift_cross_1d(&data1, &data2, &argvals, 3);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(
                    dist[(i, j)] >= 0.0,
                    "Hshift cross distance should be non-negative"
                );
                assert!(
                    dist[(i, j)].is_finite(),
                    "Hshift cross distance should be finite"
                );
            }
        }
    }

    #[test]
    fn test_hausdorff_self_2d_properties() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data = FdMatrix::from_column_major(
            (0..(n * n_points))
                .map(|i| (i as f64 * 0.1).sin())
                .collect(),
            n,
            n_points,
        )
        .unwrap();
        let dist = hausdorff_self_2d(&data, &argvals_s, &argvals_t);
        for i in 0..n {
            assert!(
                dist[(i, i)].abs() < 1e-10,
                "Hausdorff 2D self-distance should be zero on diagonal"
            );
        }
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Hausdorff 2D should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hausdorff_cross_2d() {
        let n1 = 2;
        let n2 = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * n_points))
                .map(|i| (i as f64 * 0.1).sin())
                .collect(),
            n1,
            n_points,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * n_points))
                .map(|i| (i as f64 * 0.2).cos())
                .collect(),
            n2,
            n_points,
        )
        .unwrap();
        let dist = hausdorff_cross_2d(&data1, &data2, &argvals_s, &argvals_t);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(
                    dist[(i, j)] >= 0.0,
                    "Hausdorff cross 2D should be non-negative"
                );
                assert!(
                    dist[(i, j)].is_finite(),
                    "Hausdorff cross 2D should be finite"
                );
            }
        }
    }

    #[test]
    fn test_lp_cross_2d() {
        let n1 = 2;
        let n2 = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data1 = FdMatrix::from_column_major(
            (0..(n1 * n_points))
                .map(|i| (i as f64 * 0.1).sin())
                .collect(),
            n1,
            n_points,
        )
        .unwrap();
        let data2 = FdMatrix::from_column_major(
            (0..(n2 * n_points))
                .map(|i| (i as f64 * 0.2).cos())
                .collect(),
            n2,
            n_points,
        )
        .unwrap();
        let dist = lp_cross_2d(&data1, &data2, &argvals_s, &argvals_t, 2.0, &[]);
        assert_eq!(dist.nrows(), n1);
        assert_eq!(dist.ncols(), n2);
        for j in 0..n2 {
            for i in 0..n1 {
                assert!(dist[(i, j)] >= 0.0, "Lp cross 2D should be non-negative");
                assert!(dist[(i, j)].is_finite(), "Lp cross 2D should be finite");
            }
        }
        let user_weights: Vec<f64> = vec![1.0; n_points];
        let dist_w = lp_cross_2d(&data1, &data2, &argvals_s, &argvals_t, 2.0, &user_weights);
        assert_eq!(dist_w.nrows(), n1);
        assert_eq!(dist_w.ncols(), n2);
    }

    #[test]
    fn test_lp_self_2d_with_user_weights() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data = FdMatrix::from_column_major(
            (0..(n * n_points)).map(|i| i as f64 * 0.1).collect(),
            n,
            n_points,
        )
        .unwrap();
        let user_weights: Vec<f64> = vec![2.0; n_points];
        let dist = lp_self_2d(&data, &argvals_s, &argvals_t, 2.0, &user_weights);
        assert_eq!(dist.nrows(), n);
        assert_eq!(dist.ncols(), n);
        for i in 0..n {
            assert!(
                dist[(i, i)].abs() < 1e-10,
                "Weighted Lp 2D self-distance should be zero on diagonal"
            );
        }
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Weighted Lp 2D should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_nan_lp_no_panic() {
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data_vec = vec![1.0; 3 * m];
        data_vec[5] = f64::NAN;
        let data = FdMatrix::from_column_major(data_vec, 3, m).unwrap();
        let w = vec![1.0; m];
        let dm = lp_self_1d(&data, &argvals, 2.0, &w);
        assert_eq!(dm.nrows(), 3);
    }

    #[test]
    fn test_n1_self_metric() {
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let data = FdMatrix::from_column_major(vec![1.0; m], 1, m).unwrap();
        let w = vec![1.0; m];
        let dm = lp_self_1d(&data, &argvals, 2.0, &w);
        assert_eq!(dm.shape(), (1, 1));
        assert!(dm[(0, 0)].abs() < 1e-12);
    }

    #[test]
    fn test_inf_hausdorff() {
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data_vec = vec![1.0; 2 * m];
        data_vec[0] = f64::INFINITY;
        let data = FdMatrix::from_column_major(data_vec, 2, m).unwrap();
        let dm = hausdorff_self_1d(&data, &argvals);
        assert_eq!(dm.nrows(), 2);
        // Should not panic
    }

    #[test]
    fn test_non_uniform_lp() {
        // Non-uniform grid
        let argvals = vec![0.0, 0.1, 0.5, 1.0];
        let m = argvals.len();
        let data_vec: Vec<f64> = vec![
            1.0, 2.0, // col 0
            1.0, 2.0, // col 1
            1.0, 2.0, // col 2
            1.0, 2.0, // col 3
        ];
        let data = FdMatrix::from_column_major(data_vec, 2, m).unwrap();
        let w = vec![1.0; m];
        let dm = lp_self_1d(&data, &argvals, 2.0, &w);
        assert_eq!(dm.shape(), (2, 2));
        // Constant offset curves: distance should be > 0
        assert!(dm[(0, 1)] > 0.0);
    }

    // ── Soft-DTW tests ──

    #[test]
    fn test_softmin3_approaches_hard_min() {
        let (a, b, c) = (1.0, 3.0, 5.0);
        // Small gamma → hard min
        let result = softmin3(a, b, c, 0.001);
        assert!(
            (result - 1.0).abs() < 0.01,
            "softmin3 with small gamma should approach hard min, got {result}"
        );
    }

    #[test]
    fn test_softmin3_large_gamma() {
        let (a, b, c) = (1.0, 3.0, 5.0);
        // Large gamma → approaches negative bias from ln(3)
        let result = softmin3(a, b, c, 1000.0);
        // With very large gamma, softmin3 ≈ min - gamma * ln(3)
        assert!(
            result.is_finite(),
            "softmin3 with large gamma should be finite"
        );
    }

    #[test]
    fn test_softmin3_stability_large_values() {
        let result = softmin3(1e300, 1e300 + 1.0, 1e300 + 2.0, 1.0);
        assert!(
            result.is_finite(),
            "softmin3 should handle large values without overflow"
        );
    }

    #[test]
    fn test_soft_dtw_identical_series() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let div = soft_dtw_divergence(&x, &x, 1.0);
        assert!(
            div.abs() < 1e-10,
            "Divergence of identical series should be ~0, got {div}"
        );
    }

    #[test]
    fn test_soft_dtw_converges_to_hard_dtw() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        let hard = dtw_distance(&x, &y, 2.0, 10);
        let soft = soft_dtw_distance(&x, &y, 0.001);
        assert!(
            (soft - hard).abs() < 0.1,
            "Soft-DTW with small gamma should approach hard DTW²: soft={soft}, hard={hard}"
        );
    }

    #[test]
    fn test_soft_dtw_self_symmetric() {
        let n = 4;
        let m = 10;
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.2).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let dist = soft_dtw_self_1d(&data, 1.0);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                    "Soft-DTW matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_soft_dtw_cross_vs_self() {
        let n = 3;
        let m = 10;
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.2).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let cross = soft_dtw_cross_1d(&data, &data, 1.0);
        let self_mat = soft_dtw_self_1d(&data, 1.0);
        // Off-diagonal entries should match (diagonal differs because self_distance_matrix
        // leaves diagonal as 0, but sdtw(x,x) > 0 for soft-DTW)
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    assert!(
                        (cross[(i, j)] - self_mat[(i, j)]).abs() < 1e-10,
                        "Cross(data,data) should match self at ({i},{j})"
                    );
                }
            }
        }
    }

    #[test]
    fn test_soft_dtw_divergence_nonneg() {
        let n = 4;
        let m = 10;
        let data = FdMatrix::from_column_major(
            (0..(n * m)).map(|i| (i as f64 * 0.3).sin()).collect(),
            n,
            m,
        )
        .unwrap();
        let div = soft_dtw_div_self_1d(&data, 1.0);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    div[(i, j)] >= -1e-10,
                    "Divergence should be non-negative, got {} at ({i},{j})",
                    div[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_soft_dtw_single_point() {
        let x = vec![3.0];
        let y = vec![5.0];
        let dist = soft_dtw_distance(&x, &y, 1.0);
        let expected = (3.0 - 5.0_f64).powi(2);
        assert!(
            (dist - expected).abs() < 1e-10,
            "Single-point sdtw should be (a-b)², got {dist}, expected {expected}"
        );
    }

    #[test]
    fn test_soft_dtw_barycenter_identical() {
        let m = 10;
        let series: Vec<f64> = (0..m).map(|i| (i as f64 * 0.3).sin()).collect();
        // Stack 5 identical copies
        let mut flat = Vec::with_capacity(5 * m);
        for _ in 0..5 {
            flat.extend_from_slice(&series);
        }
        // Column-major: for n=5, m=10
        let mut col_major = vec![0.0; 5 * m];
        for i in 0..5 {
            for j in 0..m {
                col_major[i + j * 5] = flat[i * m + j];
            }
        }
        let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
        let result = soft_dtw_barycenter(&data, 1.0, 50, 1e-6);
        for j in 0..m {
            assert!(
                (result.barycenter[j] - series[j]).abs() < 0.5,
                "Barycenter of identical series should be close to the series at j={j}"
            );
        }
    }

    #[test]
    fn test_soft_dtw_barycenter_shifted() {
        let m = 20;
        let n = 3;
        // Create shifted copies: sin(t), sin(t)+1, sin(t)+2
        let mut col_major = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                let t = j as f64 / (m - 1) as f64;
                col_major[i + j * n] = (2.0 * PI * t).sin() + i as f64;
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        let result = soft_dtw_barycenter(&data, 1.0, 100, 1e-6);
        // Barycenter should be approximately sin(t)+1 (the middle)
        let mean_val: f64 = result.barycenter.iter().sum::<f64>() / m as f64;
        assert!(
            (mean_val - 1.0).abs() < 0.5,
            "Barycenter mean should be ~1.0 (middle of shifts), got {mean_val}"
        );
    }

    #[test]
    fn test_soft_dtw_empty() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(soft_dtw_self_1d(&empty, 1.0).is_empty());
        assert!(soft_dtw_cross_1d(&empty, &empty, 1.0).is_empty());
        assert!(soft_dtw_div_self_1d(&empty, 1.0).is_empty());
    }

    #[test]
    fn test_soft_dtw_gamma_effect() {
        let x = vec![0.0, 1.0, 0.0];
        let y = vec![0.0, 0.0, 1.0];
        let d_small = soft_dtw_distance(&x, &y, 0.01);
        let d_large = soft_dtw_distance(&x, &y, 10.0);
        // Larger gamma produces more smoothing (smaller soft-dtw value due to more averaging)
        assert!(
            d_small > d_large || (d_small - d_large).abs() < 1e-5,
            "Larger gamma should generally produce smaller or equal soft-DTW: small={d_small}, large={d_large}"
        );
    }

    // ── Reference-value tests (tslearn) ─────────────────────────────────────

    #[test]
    fn test_soft_dtw_reference_tslearn_pairwise() {
        // Reference: tslearn.metrics.soft_dtw with (n_timestamps, 1) arrays
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let dxy = soft_dtw_distance(&x, &y, 1.0);
        let dxx = soft_dtw_distance(&x, &x, 1.0);
        let dyy = soft_dtw_distance(&y, &y, 1.0);

        // tslearn reference values (gamma=1.0)
        let dxy_ref = -0.277175357237551;
        let dxx_ref = -2.488408256052583;
        let dyy_ref = -2.488408256052583;

        let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-10);
        assert!(
            rel(dxy, dxy_ref) < 1e-6,
            "d(x,y): got {dxy}, expected {dxy_ref}"
        );
        assert!(
            rel(dxx, dxx_ref) < 1e-6,
            "d(x,x): got {dxx}, expected {dxx_ref}"
        );
        assert!(
            rel(dyy, dyy_ref) < 1e-6,
            "d(y,y): got {dyy}, expected {dyy_ref}"
        );

        // Divergence = d(x,y) - 0.5*(d(x,x) + d(y,y))
        let div = dxy - 0.5 * (dxx + dyy);
        let div_ref = 2.211232898815032;
        assert!(
            rel(div, div_ref) < 1e-6,
            "divergence: got {div}, expected {div_ref}"
        );
    }

    #[test]
    fn test_soft_dtw_reference_tslearn_gamma_sweep() {
        // Reference: tslearn.metrics.soft_dtw at multiple gamma values
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let cases: [(f64, f64); 3] = [
            (0.1, 1.999963681086807),
            (1.0, -0.277175357237551),
            (10.0, -46.741_092_332_890_21),
        ];

        for (gamma, expected) in cases {
            let actual = soft_dtw_distance(&x, &y, gamma);
            let denom = expected.abs().max(1e-10);
            assert!(
                (actual - expected).abs() / denom < 1e-5,
                "gamma={gamma}: got {actual}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_soft_dtw_divergence_reference_tslearn() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let div_xy = soft_dtw_divergence(&x, &y, 1.0);
        let div_ref = 2.211232898815032;
        assert!(
            (div_xy - div_ref).abs() / div_ref.abs() < 1e-6,
            "divergence(x,y): got {div_xy}, expected {div_ref}"
        );

        let div_xx = soft_dtw_divergence(&x, &x, 1.0);
        assert!(
            div_xx.abs() < 1e-6,
            "divergence(x,x) should be ~0, got {div_xx}"
        );
    }
}
