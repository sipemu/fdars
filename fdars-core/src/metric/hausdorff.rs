//! Hausdorff distance for 1D and 2D functional data.

use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, self_distance_matrix};

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
                .fold(f64::INFINITY, f64::min)
        })
        .fold(f64::NEG_INFINITY, f64::max)
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
                    let k = i + j * m1;
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
