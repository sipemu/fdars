//! Dynamic Time Warping (DTW) distance.

use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, self_distance_matrix};

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
