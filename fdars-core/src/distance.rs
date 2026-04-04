//! Generic pairwise distance computation.
//!
//! Many FDA algorithms need symmetric n x n distance matrices computed from
//! an upper-triangle loop. This module provides a single generic builder
//! ([`pairwise_distance_matrix`]) plus concrete wrappers for the two most
//! common metrics (L2 functional distance and Euclidean multivariate
//! distance). A cross-distance helper is also included.

use crate::helpers::{l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute symmetric n x n pairwise distance matrix using a custom distance function.
///
/// The closure `dist_fn(i, j)` is called for each pair where `i < j`. The
/// result is symmetric with zeros on the diagonal.
///
/// # Examples
///
/// ```
/// use fdars_core::distance::pairwise_distance_matrix;
///
/// let mat = pairwise_distance_matrix(3, |i, j| (i as f64 - j as f64).abs());
/// assert_eq!(mat.shape(), (3, 3));
/// assert!((mat[(0, 2)] - 2.0).abs() < 1e-15);
/// assert!((mat[(2, 0)] - 2.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn pairwise_distance_matrix<F>(n: usize, dist_fn: F) -> FdMatrix
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    let pair_dists: Vec<(usize, usize, f64)> = iter_maybe_parallel!(pairs)
        .map(|(i, j)| (i, j, dist_fn(i, j)))
        .collect();

    let mut mat = FdMatrix::zeros(n, n);
    for (i, j, d) in pair_dists {
        mat[(i, j)] = d;
        mat[(j, i)] = d;
    }
    mat
}

/// Compute n x n pairwise L2 distance matrix for functional data.
///
/// Uses Simpson's-rule integration weights derived from `argvals`.
///
/// # Examples
///
/// ```
/// use fdars_core::distance::l2_distance_matrix;
/// use fdars_core::matrix::FdMatrix;
///
/// let data = FdMatrix::zeros(5, 10);
/// let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let mat = l2_distance_matrix(&data, &t);
/// assert_eq!(mat.shape(), (5, 5));
/// ```
#[must_use]
pub fn l2_distance_matrix(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let weights = simpsons_weights(argvals);
    let n = data.nrows();
    pairwise_distance_matrix(n, |i, j| l2_distance(&data.row(i), &data.row(j), &weights))
}

/// Compute n x n pairwise Euclidean distance matrix for multivariate data.
///
/// # Examples
///
/// ```
/// use fdars_core::distance::euclidean_distance_matrix;
/// use fdars_core::matrix::FdMatrix;
///
/// let mut data = FdMatrix::zeros(3, 2);
/// data[(0, 0)] = 0.0; data[(0, 1)] = 0.0;
/// data[(1, 0)] = 3.0; data[(1, 1)] = 4.0;
/// data[(2, 0)] = 0.0; data[(2, 1)] = 0.0;
/// let mat = euclidean_distance_matrix(&data);
/// assert!((mat[(0, 1)] - 5.0).abs() < 1e-12);
/// assert!((mat[(0, 2)]).abs() < 1e-12);
/// ```
#[must_use]
pub fn euclidean_distance_matrix(data: &FdMatrix) -> FdMatrix {
    let n = data.nrows();
    let p = data.ncols();
    pairwise_distance_matrix(n, |i, j| {
        let mut d2 = 0.0;
        for k in 0..p {
            let diff = data[(i, k)] - data[(j, k)];
            d2 += diff * diff;
        }
        d2.sqrt()
    })
}

/// Compute n_new x n_train cross-distance matrix using a custom distance function.
///
/// The closure `dist_fn(i, j)` receives the index `i` into the "new" set
/// and `j` into the "train" set. The result matrix has shape
/// `(n_new, n_train)`.
///
/// # Examples
///
/// ```
/// use fdars_core::distance::cross_distance_matrix;
///
/// let mat = cross_distance_matrix(3, 5, |i, j| (i + j) as f64);
/// assert_eq!(mat.shape(), (3, 5));
/// assert!((mat[(2, 4)] - 6.0).abs() < 1e-15);
/// ```
#[must_use]
pub fn cross_distance_matrix<F>(n_new: usize, n_train: usize, dist_fn: F) -> FdMatrix
where
    F: Fn(usize, usize) -> f64 + Sync,
{
    let pairs: Vec<(usize, usize)> = (0..n_new)
        .flat_map(|i| (0..n_train).map(move |j| (i, j)))
        .collect();

    let pair_dists: Vec<(usize, usize, f64)> = iter_maybe_parallel!(pairs)
        .map(|(i, j)| (i, j, dist_fn(i, j)))
        .collect();

    let mut mat = FdMatrix::zeros(n_new, n_train);
    for (i, j, d) in pair_dists {
        mat[(i, j)] = d;
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_symmetric() {
        let mat = pairwise_distance_matrix(4, |i, j| (i as f64 - j as f64).abs());
        assert_eq!(mat.shape(), (4, 4));
        for i in 0..4 {
            assert!(mat[(i, i)].abs() < 1e-15);
            for j in 0..4 {
                assert!((mat[(i, j)] - mat[(j, i)]).abs() < 1e-15);
            }
        }
        assert!((mat[(0, 3)] - 3.0).abs() < 1e-15);
    }

    #[test]
    fn l2_matrix_smoke() {
        let data = FdMatrix::zeros(5, 10);
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let mat = l2_distance_matrix(&data, &t);
        assert_eq!(mat.shape(), (5, 5));
        // All zeros -> all distances zero
        for i in 0..5 {
            for j in 0..5 {
                assert!(mat[(i, j)].abs() < 1e-15);
            }
        }
    }

    #[test]
    fn l2_matrix_nonzero() {
        // Two curves: one constant 0, one constant 1
        let mut data = FdMatrix::zeros(2, 10);
        for j in 0..10 {
            data[(1, j)] = 1.0;
        }
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let mat = l2_distance_matrix(&data, &t);
        // L2 distance between 0 and 1 on [0,1] = sqrt(1) = 1
        assert!((mat[(0, 1)] - 1.0).abs() < 0.1);
        assert!((mat[(1, 0)] - mat[(0, 1)]).abs() < 1e-15);
    }

    #[test]
    fn euclidean_matrix_smoke() {
        let mut data = FdMatrix::zeros(3, 2);
        data[(0, 0)] = 0.0;
        data[(0, 1)] = 0.0;
        data[(1, 0)] = 3.0;
        data[(1, 1)] = 4.0;
        data[(2, 0)] = 0.0;
        data[(2, 1)] = 0.0;
        let mat = euclidean_distance_matrix(&data);
        assert!((mat[(0, 1)] - 5.0).abs() < 1e-12);
        assert!((mat[(0, 2)]).abs() < 1e-12);
        assert!((mat[(1, 2)] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn cross_distance_dims() {
        let mat = cross_distance_matrix(3, 5, |i, j| (i + j) as f64);
        assert_eq!(mat.shape(), (3, 5));
        assert!((mat[(0, 0)]).abs() < 1e-15);
        assert!((mat[(2, 4)] - 6.0).abs() < 1e-15);
    }

    #[test]
    fn pairwise_n_zero() {
        let mat = pairwise_distance_matrix(0, |_i, _j| 1.0);
        assert_eq!(mat.shape(), (0, 0));
    }

    #[test]
    fn pairwise_n_one() {
        let mat = pairwise_distance_matrix(1, |_i, _j| 1.0);
        assert_eq!(mat.shape(), (1, 1));
        assert!(mat[(0, 0)].abs() < 1e-15);
    }
}
