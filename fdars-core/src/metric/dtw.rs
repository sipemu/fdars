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
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::dtw_self_1d;
///
/// let data = FdMatrix::from_column_major(
///     (0..40).map(|i| (i as f64 * 0.1).sin()).collect(),
///     4, 10,
/// ).unwrap();
/// let dist = dtw_self_1d(&data, 2.0, 10);
/// assert_eq!(dist.shape(), (4, 4));
/// assert!((dist[(0, 0)]).abs() < 1e-10);
/// assert!((dist[(0, 1)] - dist[(1, 0)]).abs() < 1e-10);
/// ```
pub fn dtw_self_1d(data: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n = data.nrows();
    if n == 0 || data.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    self_distance_matrix(n, |i, j| dtw_distance(&rows[i], &rows[j], p, w))
}

/// Compute DTW cross-distance matrix.
pub fn dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    if n1 == 0 || n2 == 0 || data1.ncols() == 0 || data2.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows1: Vec<Vec<f64>> = (0..n1).map(|i| data1.row(i)).collect();
    let rows2: Vec<Vec<f64>> = (0..n2).map(|i| data2.row(i)).collect();
    cross_distance_matrix(n1, n2, |i, j| dtw_distance(&rows1[i], &rows2[j], p, w))
}
