//! Distance metrics and semimetrics for functional data.
//!
//! This module provides various distance measures including:
//! - Lp distances (L1, L2, L∞)
//! - Hausdorff distance
//! - Dynamic Time Warping (DTW)
//! - Fourier-based semimetric
//! - Horizontal shift semimetric
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
fn self_distance_matrix(
    n: usize,
    compute: impl Fn(usize, usize) -> f64 + Sync,
) -> FdMatrix {
    // Pre-allocate flat buffer for upper triangle: n*(n-1)/2 entries
    let tri_len = n * (n - 1) / 2;
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| compute(i, j))
                .collect::<Vec<_>>()
        })
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
        .flat_map(|i| {
            (0..n2)
                .map(|j| compute(i, j))
                .collect::<Vec<_>>()
        })
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
fn dtw_dp_loop(
    x: &[f64],
    y: &[f64],
    w: usize,
    cost_fn: impl Fn(f64, f64) -> f64,
) -> f64 {
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
        dtw_distance(
            &rm1[i * m1..(i + 1) * m1],
            &rm2[j * m2..(j + 1) * m2],
            p,
            w,
        )
    })
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
}
