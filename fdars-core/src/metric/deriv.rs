//! Derivative-based semimetric for functional data.
//!
//! Computes L2 distances between the k-th finite-difference derivatives
//! of functional observations, using trapezoidal integration weights.

use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, merge_weights, self_distance_matrix};

/// Compute the `nderiv`-th finite-difference derivative for each row.
///
/// After one round of differencing a curve with `m` points becomes `m-1`
/// points; after `nderiv` rounds it has `m - nderiv` points.  The
/// corresponding argument values are the midpoints of successive intervals.
fn finite_differences(data: &FdMatrix, argvals: &[f64], nderiv: usize) -> (FdMatrix, Vec<f64>) {
    let n = data.nrows();
    let m = data.ncols();

    if nderiv == 0 || m <= nderiv {
        return (data.clone(), argvals.to_vec());
    }

    // Work with row-major representation for convenience.
    let mut curves: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let mut grid = argvals.to_vec();

    for _ in 0..nderiv {
        let len = grid.len();
        let new_len = len - 1;
        let mut new_curves = Vec::with_capacity(n);
        let mut new_grid = Vec::with_capacity(new_len);

        for k in 0..new_len {
            let dt = grid[k + 1] - grid[k];
            if new_grid.is_empty() || new_grid.len() < new_len {
                new_grid.push(0.5 * (grid[k] + grid[k + 1]));
            }
            let _ = dt; // used below per-curve
        }

        for curve in &curves {
            let deriv: Vec<f64> = (0..new_len)
                .map(|k| {
                    let dt = grid[k + 1] - grid[k];
                    if dt.abs() < 1e-30 {
                        0.0
                    } else {
                        (curve[k + 1] - curve[k]) / dt
                    }
                })
                .collect();
            new_curves.push(deriv);
        }

        curves = new_curves;
        grid = new_grid;
    }

    // Pack back into column-major FdMatrix.
    let new_m = grid.len();
    let mut flat = vec![0.0; n * new_m];
    for i in 0..n {
        for j in 0..new_m {
            flat[i + j * n] = curves[i][j];
        }
    }
    let mat = FdMatrix::from_column_major(flat, n, new_m)
        .expect("dimension invariant: data.len() == n * new_m");
    (mat, grid)
}

/// Compute the derivative-based semimetric self-distance matrix.
///
/// Each curve is differentiated `nderiv` times via finite differences,
/// and the L2 distance between derivative curves is computed using
/// Simpson's integration weights (optionally merged with `user_weights`).
///
/// # Arguments
///
/// * `data` - Functional data matrix (`n x m`, column-major).
/// * `argvals` - Grid of evaluation points (length `m`).
/// * `nderiv` - Order of differentiation (0 returns plain L2 distance).
/// * `user_weights` - Optional per-point weights; pass `&[]` for none.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::deriv_self_1d;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 20,
/// ).unwrap();
/// let dist = deriv_self_1d(&data, &argvals, 1, &[]);
/// assert_eq!(dist.shape(), (5, 5));
/// assert!(dist[(0, 0)].abs() < 1e-10);
/// ```
pub fn deriv_self_1d(
    data: &FdMatrix,
    argvals: &[f64],
    nderiv: usize,
    user_weights: &[f64],
) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || argvals.len() != m {
        return FdMatrix::zeros(0, 0);
    }

    let (dmat, dgrid) = finite_differences(data, argvals, nderiv);
    let dm = dmat.ncols();
    if dm == 0 {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights(&dgrid), user_weights);

    self_distance_matrix(n, |i, j| {
        let mut sum = 0.0;
        for k in 0..dm {
            let diff = dmat[(i, k)] - dmat[(j, k)];
            sum += diff * diff * weights[k];
        }
        sum.sqrt()
    })
}

/// Compute the derivative-based semimetric cross-distance matrix.
///
/// Each curve in both datasets is differentiated `nderiv` times, then
/// pairwise L2 distances between derivative curves are computed.
///
/// # Arguments
///
/// * `data1` - First dataset (`n1 x m`, column-major).
/// * `data2` - Second dataset (`n2 x m`, column-major).
/// * `argvals` - Grid of evaluation points (length `m`).
/// * `nderiv` - Order of differentiation.
/// * `user_weights` - Optional per-point weights; pass `&[]` for none.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::deriv_cross_1d;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data1 = FdMatrix::from_column_major(
///     (0..60).map(|i| (i as f64 * 0.1).sin()).collect(), 3, 20,
/// ).unwrap();
/// let data2 = FdMatrix::from_column_major(
///     (0..40).map(|i| (i as f64 * 0.2).cos()).collect(), 2, 20,
/// ).unwrap();
/// let dist = deriv_cross_1d(&data1, &data2, &argvals, 1, &[]);
/// assert_eq!(dist.shape(), (3, 2));
/// ```
pub fn deriv_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    nderiv: usize,
    user_weights: &[f64],
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();

    if n1 == 0 || n2 == 0 || m == 0 || argvals.len() != m || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }

    let (dmat1, dgrid) = finite_differences(data1, argvals, nderiv);
    let (dmat2, _) = finite_differences(data2, argvals, nderiv);
    let dm = dmat1.ncols();
    if dm == 0 || dmat2.ncols() != dm {
        return FdMatrix::zeros(0, 0);
    }

    let weights = merge_weights(simpsons_weights(&dgrid), user_weights);

    cross_distance_matrix(n1, n2, |i, j| {
        let mut sum = 0.0;
        for k in 0..dm {
            let diff = dmat1[(i, k)] - dmat2[(j, k)];
            sum += diff * diff * weights[k];
        }
        sum.sqrt()
    })
}
