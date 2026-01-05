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
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// Compute Lp distance matrix between two sets of functional data.
///
/// # Arguments
/// * `data1` - First dataset (column-major, n1 x n_points)
/// * `data2` - Second dataset (column-major, n2 x n_points)
/// * `n1` - Number of observations in first set
/// * `n2` - Number of observations in second set
/// * `n_points` - Number of evaluation points
/// * `argvals` - Evaluation points for integration
/// * `p` - Order of the norm
/// * `weights` - Optional user weights (empty vector for none)
///
/// # Returns
/// Flattened distance matrix (column-major, n1 x n2)
pub fn lp_cross_1d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    n_points: usize,
    argvals: &[f64],
    p: f64,
    user_weights: &[f64],
) -> Vec<f64> {
    if n1 == 0 || n2 == 0 || n_points == 0 || argvals.len() != n_points {
        return Vec::new();
    }

    let base_weights = simpsons_weights(argvals);
    let weights: Vec<f64> = if user_weights.len() == n_points {
        base_weights
            .iter()
            .zip(user_weights.iter())
            .map(|(b, u)| b * u)
            .collect()
    } else {
        base_weights
    };

    let distances: Vec<f64> = iter_maybe_parallel!(0..n2)
        .flat_map(|j| {
            (0..n1)
                .map(|i| {
                    let mut integral = 0.0;
                    for k in 0..n_points {
                        let diff = (data1[i + k * n1] - data2[j + k * n2]).abs();
                        integral += diff.powf(p) * weights[k];
                    }
                    integral.powf(1.0 / p)
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    distances
}

/// Compute Lp distance matrix for self-distances (symmetric).
///
/// Returns flattened symmetric distance matrix (column-major, n x n).
pub fn lp_self_1d(
    data: &[f64],
    n: usize,
    n_points: usize,
    argvals: &[f64],
    p: f64,
    user_weights: &[f64],
) -> Vec<f64> {
    if n == 0 || n_points == 0 || argvals.len() != n_points {
        return Vec::new();
    }

    let base_weights = simpsons_weights(argvals);
    let weights: Vec<f64> = if user_weights.len() == n_points {
        base_weights
            .iter()
            .zip(user_weights.iter())
            .map(|(b, u)| b * u)
            .collect()
    } else {
        base_weights
    };

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let mut integral = 0.0;
                    for k in 0..n_points {
                        let diff = (data[i + k * n] - data[j + k * n]).abs();
                        integral += diff.powf(p) * weights[k];
                    }
                    (i, j, integral.powf(1.0 / p))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute Lp distance for 2D functional data (surfaces).
pub fn lp_cross_2d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    argvals_s: &[f64],
    argvals_t: &[f64],
    p: f64,
    user_weights: &[f64],
) -> Vec<f64> {
    let n_points = argvals_s.len() * argvals_t.len();
    if n1 == 0 || n2 == 0 || n_points == 0 {
        return Vec::new();
    }

    let base_weights = simpsons_weights_2d(argvals_s, argvals_t);
    let weights: Vec<f64> = if user_weights.len() == n_points {
        base_weights
            .iter()
            .zip(user_weights.iter())
            .map(|(bw, uw)| bw * uw)
            .collect()
    } else {
        base_weights
    };

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| {
                    let mut sum = 0.0;
                    for k in 0..n_points {
                        let diff = (data1[i + k * n1] - data2[j + k * n2]).abs();
                        sum += weights[k] * diff.powf(p);
                    }
                    sum.powf(1.0 / p)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    distances
}

/// Compute Lp self-distance matrix for 2D functional data (symmetric).
pub fn lp_self_2d(
    data: &[f64],
    n: usize,
    argvals_s: &[f64],
    argvals_t: &[f64],
    p: f64,
    user_weights: &[f64],
) -> Vec<f64> {
    let n_points = argvals_s.len() * argvals_t.len();
    if n == 0 || n_points == 0 {
        return Vec::new();
    }

    let base_weights = simpsons_weights_2d(argvals_s, argvals_t);
    let weights: Vec<f64> = if user_weights.len() == n_points {
        base_weights
            .iter()
            .zip(user_weights.iter())
            .map(|(bw, uw)| bw * uw)
            .collect()
    } else {
        base_weights
    };

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let mut sum = 0.0;
                    for k in 0..n_points {
                        let diff = (data[i + k * n] - data[j + k * n]).abs();
                        sum += weights[k] * diff.powf(p);
                    }
                    (i, j, sum.powf(1.0 / p))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute Hausdorff distance matrix for self-distances (symmetric).
///
/// The Hausdorff distance treats curves as sets of points (t, f(t)) in 2D space.
pub fn hausdorff_self_1d(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> Vec<f64> {
    if n == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    // Precompute squared time differences
    let mtt: Vec<f64> = {
        let mut result = Vec::with_capacity(m * m);
        for s in 0..m {
            for t in 0..m {
                let diff = argvals[s] - argvals[t];
                result.push(diff * diff);
            }
        }
        result
    };

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    // For each row s, find min over columns t
                    let max_row_min = (0..m)
                        .map(|s| {
                            let x_s = data[i + s * n];
                            (0..m)
                                .map(|t| {
                                    let y_t = data[j + t * n];
                                    let val_diff = x_s - y_t;
                                    (val_diff * val_diff + mtt[s * m + t]).sqrt()
                                })
                                .fold(f64::INFINITY, |a, b| a.min(b))
                        })
                        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

                    // For each column t, find min over rows s
                    let max_col_min = (0..m)
                        .map(|t| {
                            let y_t = data[j + t * n];
                            (0..m)
                                .map(|s| {
                                    let x_s = data[i + s * n];
                                    let val_diff = x_s - y_t;
                                    (val_diff * val_diff + mtt[s * m + t]).sqrt()
                                })
                                .fold(f64::INFINITY, |a, b| a.min(b))
                        })
                        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

                    let hausdorff_dist = max_row_min.max(max_col_min);
                    (i, j, hausdorff_dist)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute Hausdorff cross-distances for 1D functional data.
pub fn hausdorff_cross_1d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    m: usize,
    argvals: &[f64],
) -> Vec<f64> {
    if n1 == 0 || n2 == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    // Precompute squared time differences
    let mtt: Vec<f64> = {
        let mut result = Vec::with_capacity(m * m);
        for s in 0..m {
            for t in 0..m {
                let diff = argvals[s] - argvals[t];
                result.push(diff * diff);
            }
        }
        result
    };

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| {
                    let max_row_min = (0..m)
                        .map(|s| {
                            let x_s = data1[i + s * n1];
                            (0..m)
                                .map(|t| {
                                    let y_t = data2[j + t * n2];
                                    let val_diff = x_s - y_t;
                                    (val_diff * val_diff + mtt[s * m + t]).sqrt()
                                })
                                .fold(f64::INFINITY, |a, b| a.min(b))
                        })
                        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

                    let max_col_min = (0..m)
                        .map(|t| {
                            let y_t = data2[j + t * n2];
                            (0..m)
                                .map(|s| {
                                    let x_s = data1[i + s * n1];
                                    let val_diff = x_s - y_t;
                                    (val_diff * val_diff + mtt[s * m + t]).sqrt()
                                })
                                .fold(f64::INFINITY, |a, b| a.min(b))
                        })
                        .fold(f64::NEG_INFINITY, |a, b| a.max(b));

                    max_row_min.max(max_col_min)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    distances
}

/// Compute DTW distance between two time series.
///
/// Uses Lp distance with optional Sakoe-Chiba window constraint.
pub fn dtw_distance(x: &[f64], y: &[f64], p: f64, w: usize) -> f64 {
    let n = x.len();
    let m = y.len();

    let mut dtw = vec![vec![0.0_f64; m + 1]; n + 1];

    // Initialize boundaries
    for j in 0..=m {
        dtw[0][j] = f64::INFINITY;
    }
    for i in 0..=n {
        dtw[i][0] = f64::INFINITY;
    }
    dtw[0][0] = 0.0;

    // Initialize cells within the window band
    for i in 1..=n {
        let r_i = i + 1;
        let j_start_r = 2.max(r_i as isize - w as isize) as usize;
        let j_end_r = (m + 1).min(r_i + w);

        for j_r in j_start_r..=j_end_r {
            let j = j_r - 1;
            if j <= m {
                dtw[i][j] = 0.0;
            }
        }
    }

    // Fill the DTW matrix
    for i in 1..=n {
        let r_i = i + 1;
        let j_start_r = 2.max(r_i as isize - w as isize) as usize;
        let j_end_r = (m + 1).min(r_i + w);

        for j_r in j_start_r..=j_end_r {
            let j = j_r - 1;
            if j <= m && j >= 1 {
                let cost = (x[i - 1] - y[j - 1]).abs().powf(p);
                dtw[i][j] = cost + dtw[i - 1][j].min(dtw[i][j - 1]).min(dtw[i - 1][j - 1]);
            }
        }
    }

    dtw[n][m]
}

/// Compute DTW distance matrix for self-distances (symmetric).
pub fn dtw_self_1d(data: &[f64], n: usize, m: usize, p: f64, w: usize) -> Vec<f64> {
    if n == 0 || m == 0 {
        return Vec::new();
    }

    // Extract curves as vectors
    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let dist = dtw_distance(&curves[i], &curves[j], p, w);
                    (i, j, dist)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute DTW cross-distance matrix.
pub fn dtw_cross_1d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    m1: usize,
    m2: usize,
    p: f64,
    w: usize,
) -> Vec<f64> {
    if n1 == 0 || n2 == 0 || m1 == 0 || m2 == 0 {
        return Vec::new();
    }

    let curves1: Vec<Vec<f64>> = (0..n1)
        .map(|i| (0..m1).map(|j| data1[i + j * n1]).collect())
        .collect();
    let curves2: Vec<Vec<f64>> = (0..n2)
        .map(|i| (0..m2).map(|j| data2[i + j * n2]).collect())
        .collect();

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| dtw_distance(&curves1[i], &curves2[j], p, w))
                .collect::<Vec<_>>()
        })
        .collect();

    distances
}

/// Compute Fourier coefficients for a curve using FFT.
fn fft_coefficients(data: &[f64], nfreq: usize) -> Vec<f64> {
    let n = data.len();
    let nfreq = nfreq.min(n / 2);

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();

    fft.process(&mut buffer);

    buffer
        .iter()
        .take(nfreq + 1)
        .map(|c| c.norm() / n as f64)
        .collect()
}

/// Compute semimetric based on Fourier coefficients for self-distances.
pub fn fourier_self_1d(data: &[f64], n: usize, m: usize, nfreq: usize) -> Vec<f64> {
    if n == 0 || m == 0 {
        return Vec::new();
    }

    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

    let coeffs: Vec<Vec<f64>> = iter_maybe_parallel!(curves)
        .map(|curve| fft_coefficients(&curve, nfreq))
        .collect();

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let dist_sq: f64 = coeffs[i]
                        .iter()
                        .zip(coeffs[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    (i, j, dist_sq.sqrt())
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute semimetric based on Fourier coefficients for cross-distances.
pub fn fourier_cross_1d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    m: usize,
    nfreq: usize,
) -> Vec<f64> {
    if n1 == 0 || n2 == 0 || m == 0 {
        return Vec::new();
    }

    let curves1: Vec<Vec<f64>> = (0..n1)
        .map(|i| (0..m).map(|j| data1[i + j * n1]).collect())
        .collect();
    let curves2: Vec<Vec<f64>> = (0..n2)
        .map(|i| (0..m).map(|j| data2[i + j * n2]).collect())
        .collect();

    let coeffs1: Vec<Vec<f64>> = iter_maybe_parallel!(curves1)
        .map(|curve| fft_coefficients(&curve, nfreq))
        .collect();
    let coeffs2: Vec<Vec<f64>> = iter_maybe_parallel!(curves2)
        .map(|curve| fft_coefficients(&curve, nfreq))
        .collect();

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| {
                    let dist_sq: f64 = coeffs1[i]
                        .iter()
                        .zip(coeffs2[j].iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    dist_sq.sqrt()
                })
                .collect::<Vec<_>>()
        })
        .collect();

    distances
}

/// Compute minimum L2 distance after horizontal shift between two curves.
fn hshift_distance(x: &[f64], y: &[f64], weights: &[f64], max_shift: usize) -> f64 {
    let n = x.len();
    if n == 0 || y.len() != n || weights.len() != n {
        return f64::INFINITY;
    }

    let mut min_dist = f64::INFINITY;

    for shift in -(max_shift as i32)..=(max_shift as i32) {
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
            let dist = sum.sqrt();
            if dist < min_dist {
                min_dist = dist;
            }
        }
    }

    min_dist
}

/// Compute semimetric based on horizontal shift for self-distances.
pub fn hshift_self_1d(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    max_shift: usize,
) -> Vec<f64> {
    if n == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    let curves: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..m).map(|j| data[i + j * n]).collect())
        .collect();

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let dist = hshift_distance(&curves[i], &curves[j], &weights, max_shift);
                    (i, j, dist)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute semimetric based on horizontal shift for cross-distances.
pub fn hshift_cross_1d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    m: usize,
    argvals: &[f64],
    max_shift: usize,
) -> Vec<f64> {
    if n1 == 0 || n2 == 0 || m == 0 || argvals.len() != m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    let curves1: Vec<Vec<f64>> = (0..n1)
        .map(|i| (0..m).map(|j| data1[i + j * n1]).collect())
        .collect();
    let curves2: Vec<Vec<f64>> = (0..n2)
        .map(|i| (0..m).map(|j| data2[i + j * n2]).collect())
        .collect();

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| hshift_distance(&curves1[i], &curves2[j], &weights, max_shift))
                .collect::<Vec<_>>()
        })
        .collect();

    distances
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

/// Compute Hausdorff self-distance for 2D surfaces.
pub fn hausdorff_self_2d(data: &[f64], n: usize, argvals_s: &[f64], argvals_t: &[f64]) -> Vec<f64> {
    let m1 = argvals_s.len();
    let m2 = argvals_t.len();
    let n_points = m1 * m2;

    if n == 0 || n_points == 0 {
        return Vec::new();
    }

    let surfaces: Vec<Vec<(f64, f64, f64)>> = (0..n)
        .map(|curve| {
            let mut points = Vec::with_capacity(n_points);
            for i in 0..m1 {
                for j in 0..m2 {
                    let k = i * m2 + j;
                    points.push((argvals_s[i], argvals_t[j], data[curve + k * n]));
                }
            }
            points
        })
        .collect();

    let upper_triangle: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            ((i + 1)..n)
                .map(|j| {
                    let dist = hausdorff_3d(&surfaces[i], &surfaces[j]);
                    (i, j, dist)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut distances = vec![0.0; n * n];
    for (i, j, dist) in upper_triangle {
        distances[i + j * n] = dist;
        distances[j + i * n] = dist;
    }

    distances
}

/// Compute Hausdorff cross-distance for 2D surfaces.
pub fn hausdorff_cross_2d(
    data1: &[f64],
    data2: &[f64],
    n1: usize,
    n2: usize,
    argvals_s: &[f64],
    argvals_t: &[f64],
) -> Vec<f64> {
    let m1 = argvals_s.len();
    let m2 = argvals_t.len();
    let n_points = m1 * m2;

    if n1 == 0 || n2 == 0 || n_points == 0 {
        return Vec::new();
    }

    let surfaces1: Vec<Vec<(f64, f64, f64)>> = (0..n1)
        .map(|curve| {
            let mut points = Vec::with_capacity(n_points);
            for i in 0..m1 {
                for j in 0..m2 {
                    let k = i * m2 + j;
                    points.push((argvals_s[i], argvals_t[j], data1[curve + k * n1]));
                }
            }
            points
        })
        .collect();

    let surfaces2: Vec<Vec<(f64, f64, f64)>> = (0..n2)
        .map(|curve| {
            let mut points = Vec::with_capacity(n_points);
            for i in 0..m1 {
                for j in 0..m2 {
                    let k = i * m2 + j;
                    points.push((argvals_s[i], argvals_t[j], data2[curve + k * n2]));
                }
            }
            points
        })
        .collect();

    let distances: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| {
            (0..n2)
                .map(|j| hausdorff_3d(&surfaces1[i], &surfaces2[j]))
                .collect::<Vec<_>>()
        })
        .collect();

    distances
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    // ============== Lp distance tests ==============

    #[test]
    fn test_lp_self_distance() {
        // Two curves: [0, 1] and [1, 2]
        let data = vec![0.0, 1.0, 1.0, 2.0]; // column-major
        let argvals = vec![0.0, 1.0];
        let dist = lp_self_1d(&data, 2, 2, &argvals, 2.0, &[]);
        // L2 distance should be 1.0
        assert!((dist[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_lp_self_symmetric() {
        let n = 5;
        let m = 20;
        let argvals = uniform_grid(m);
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
            }
        }
        let dist = lp_self_1d(&data, n, m, &argvals, 2.0, &[]);

        // Check symmetry
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
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
        let data: Vec<f64> = (0..(n * m)).map(|i| i as f64 * 0.1).collect();
        let dist = lp_self_1d(&data, n, m, &argvals, 2.0, &[]);

        for i in 0..n {
            assert!(
                dist[i + i * n].abs() < 1e-10,
                "Self-distance should be zero"
            );
        }
    }

    #[test]
    fn test_lp_cross_shape() {
        let n1 = 3;
        let n2 = 4;
        let m = 20;
        let argvals = uniform_grid(m);
        let data1: Vec<f64> = (0..(n1 * m)).map(|i| i as f64 * 0.1).collect();
        let data2: Vec<f64> = (0..(n2 * m)).map(|i| i as f64 * 0.2).collect();
        let dist = lp_cross_1d(&data1, &data2, n1, n2, m, &argvals, 2.0, &[]);

        assert_eq!(dist.len(), n1 * n2);
    }

    #[test]
    fn test_lp_invalid() {
        assert!(lp_self_1d(&[], 0, 0, &[], 2.0, &[]).is_empty());
        assert!(lp_cross_1d(&[], &[], 0, 0, 0, &[], 2.0, &[]).is_empty());
    }

    // ============== DTW distance tests ==============

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
        let data: Vec<f64> = (0..(n * m)).map(|i| i as f64 * 0.1).collect();
        let dist = dtw_self_1d(&data, n, m, 2.0, 5);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
                    "DTW matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_dtw_invalid() {
        assert!(dtw_self_1d(&[], 0, 0, 2.0, 5).is_empty());
    }

    // ============== Hausdorff distance tests ==============

    #[test]
    fn test_hausdorff_self_symmetric() {
        let n = 4;
        let m = 15;
        let argvals = uniform_grid(m);
        let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect();
        let dist = hausdorff_self_1d(&data, n, m, &argvals);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
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
        let data: Vec<f64> = (0..(n * m)).map(|i| i as f64 * 0.1).collect();
        let dist = hausdorff_self_1d(&data, n, m, &argvals);

        for i in 0..n {
            assert!(
                dist[i + i * n].abs() < 1e-10,
                "Self-distance should be zero"
            );
        }
    }

    #[test]
    fn test_hausdorff_invalid() {
        assert!(hausdorff_self_1d(&[], 0, 0, &[]).is_empty());
    }

    // ============== Fourier semimetric tests ==============

    #[test]
    fn test_fourier_self_symmetric() {
        let n = 4;
        let m = 32; // Power of 2 for FFT efficiency
        let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect();
        let dist = fourier_self_1d(&data, n, m, 5);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
                    "Fourier distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_fourier_self_diagonal_zero() {
        let n = 3;
        let m = 32;
        let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64 * 0.2).cos()).collect();
        let dist = fourier_self_1d(&data, n, m, 8);

        for i in 0..n {
            assert!(
                dist[i + i * n].abs() < 1e-10,
                "Self-distance should be zero"
            );
        }
    }

    #[test]
    fn test_fourier_invalid() {
        assert!(fourier_self_1d(&[], 0, 0, 5).is_empty());
    }

    // ============== Horizontal shift semimetric tests ==============

    #[test]
    fn test_hshift_self_symmetric() {
        let n = 4;
        let m = 20;
        let argvals = uniform_grid(m);
        let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect();
        let dist = hshift_self_1d(&data, n, m, &argvals, 3);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
                    "Hshift distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hshift_invalid() {
        assert!(hshift_self_1d(&[], 0, 0, &[], 3).is_empty());
    }

    // ============== 3D Hausdorff tests ==============

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

    // ============== 2D surface tests ==============

    #[test]
    fn test_lp_2d_symmetric() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data: Vec<f64> = (0..(n * n_points)).map(|i| i as f64 * 0.1).collect();
        let dist = lp_self_2d(&data, n, &argvals_s, &argvals_t, 2.0, &[]);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
                    "2D Lp distance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_lp_2d_invalid() {
        assert!(lp_self_2d(&[], 0, &[], &[], 2.0, &[]).is_empty());
    }

    #[test]
    fn test_hausdorff_2d_symmetric() {
        let n = 3;
        let m1 = 4;
        let m2 = 5;
        let argvals_s = uniform_grid(m1);
        let argvals_t = uniform_grid(m2);
        let n_points = m1 * m2;
        let data: Vec<f64> = (0..(n * n_points))
            .map(|i| (i as f64 * 0.1).sin())
            .collect();
        let dist = hausdorff_self_2d(&data, n, &argvals_s, &argvals_t);

        for i in 0..n {
            for j in 0..n {
                assert!(
                    (dist[i + j * n] - dist[j + i * n]).abs() < 1e-10,
                    "2D Hausdorff should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_hausdorff_2d_invalid() {
        assert!(hausdorff_self_2d(&[], 0, &[], &[]).is_empty());
    }
}
