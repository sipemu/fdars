//! Basis projection and reconstruction for functional data.

use super::bspline::bspline_basis;
use super::fourier::fourier_basis;
use super::helpers::svd_pseudoinverse;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of basis projection.
pub struct BasisProjectionResult {
    /// Coefficient matrix (n_samples x n_basis)
    pub coefficients: FdMatrix,
    /// Number of basis functions used
    pub n_basis: usize,
}

/// Project functional data to basis coefficients.
///
/// # Arguments
/// * `data` - Column-major FdMatrix (n x m)
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of basis functions
/// * `basis_type` - 0 = B-spline, 1 = Fourier
pub fn fdata_to_basis_1d(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: i32,
) -> Option<BasisProjectionResult> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || argvals.len() != m || nbasis < 2 {
        return None;
    }

    let basis = if basis_type == 1 {
        fourier_basis(argvals, nbasis)
    } else {
        // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
        bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4)
    };

    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let btb_inv = svd_pseudoinverse(&btb)?;
    let proj = btb_inv * b_mat.transpose();

    let coefs: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            (0..actual_nbasis)
                .map(|k| {
                    let mut sum = 0.0;
                    for j in 0..m {
                        sum += proj[(k, j)] * curve[j];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Some(BasisProjectionResult {
        coefficients: FdMatrix::from_column_major(coefs, n, actual_nbasis)
            .expect("dimension invariant: data.len() == n * m"),
        n_basis: actual_nbasis,
    })
}

/// Reconstruct functional data from basis coefficients.
pub fn basis_to_fdata_1d(
    coefs: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: i32,
) -> FdMatrix {
    let n = coefs.nrows();
    let coefs_ncols = coefs.ncols();
    let m = argvals.len();
    if n == 0 || m == 0 || nbasis < 2 {
        return FdMatrix::zeros(0, 0);
    }

    let basis = if basis_type == 1 {
        fourier_basis(argvals, nbasis)
    } else {
        // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
        bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4)
    };

    let actual_nbasis = basis.len() / m;

    let flat: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            (0..m)
                .map(|j| {
                    let mut sum = 0.0;
                    for k in 0..actual_nbasis.min(coefs_ncols) {
                        sum += coefs[(i, k)] * basis[j + k * m];
                    }
                    sum
                })
                .collect::<Vec<_>>()
        })
        .collect();

    FdMatrix::from_column_major(flat, n, m).expect("dimension invariant: data.len() == n * m")
}
