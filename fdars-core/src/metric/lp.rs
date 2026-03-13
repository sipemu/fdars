//! Lp distance metrics for 1D and 2D functional data.

use crate::helpers::{simpsons_weights, simpsons_weights_2d};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::{lp_weighted_distance, merge_weights};

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
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::lp_cross_1d;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data1 = FdMatrix::from_column_major(
///     (0..30).map(|i| (i as f64 * 0.1).sin()).collect(), 3, 10,
/// ).unwrap();
/// let data2 = FdMatrix::from_column_major(
///     (0..20).map(|i| (i as f64 * 0.2).cos()).collect(), 2, 10,
/// ).unwrap();
/// let dist = lp_cross_1d(&data1, &data2, &argvals, 2.0, &[]);
/// assert_eq!(dist.shape(), (3, 2));
/// assert!(dist[(0, 0)] >= 0.0);
/// ```
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
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::lp_self_1d;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
/// assert_eq!(dist.shape(), (5, 5));
/// // Diagonal should be zero, matrix should be symmetric
/// assert!((dist[(0, 0)]).abs() < 1e-10);
/// assert!((dist[(0, 1)] - dist[(1, 0)]).abs() < 1e-10);
/// ```
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
