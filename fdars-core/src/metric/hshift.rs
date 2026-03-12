//! Horizontal shift semimetric for functional data.

use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, self_distance_matrix};

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
