//! Basis-coefficient semimetric for functional data.
//!
//! Each curve is projected onto a chosen basis (B-spline or Fourier) and
//! the Euclidean distance between the resulting coefficient vectors is
//! used as a semimetric.

use crate::basis::{fdata_to_basis, ProjectionBasisType};
use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, self_distance_matrix};

/// Euclidean distance between two rows of an `FdMatrix`.
#[inline]
fn row_euclidean(mat: &FdMatrix, i: usize, j: usize, ncols: usize) -> f64 {
    let mut sum = 0.0;
    for k in 0..ncols {
        let diff = mat[(i, k)] - mat[(j, k)];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Euclidean distance between row `i` of `mat1` and row `j` of `mat2`.
#[inline]
fn row_euclidean_cross(mat1: &FdMatrix, i: usize, mat2: &FdMatrix, j: usize, ncols: usize) -> f64 {
    let mut sum = 0.0;
    for k in 0..ncols {
        let diff = mat1[(i, k)] - mat2[(j, k)];
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Compute the basis-coefficient semimetric self-distance matrix.
///
/// Each curve is projected onto `nbasis` functions of the specified
/// `basis_type`, and the Euclidean distance in coefficient space is
/// returned.
///
/// # Arguments
///
/// * `data` - Functional data matrix (`n x m`, column-major).
/// * `argvals` - Evaluation grid (length `m`).
/// * `nbasis` - Number of basis functions.
/// * `basis_type` - [`ProjectionBasisType::Bspline`] or [`ProjectionBasisType::Fourier`].
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::basis::ProjectionBasisType;
/// use fdars_core::metric::basis_coef_self_1d;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..100).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 20,
/// ).unwrap();
/// let dist = basis_coef_self_1d(&data, &argvals, 7, ProjectionBasisType::Fourier);
/// assert_eq!(dist.shape(), (5, 5));
/// assert!(dist[(0, 0)].abs() < 1e-10);
/// ```
pub fn basis_coef_self_1d(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: ProjectionBasisType,
) -> FdMatrix {
    let n = data.nrows();
    if n == 0 || data.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }

    let proj = match fdata_to_basis(data, argvals, nbasis, basis_type) {
        Some(p) => p,
        None => return FdMatrix::zeros(0, 0),
    };
    let coefs = &proj.coefficients;
    let nb = proj.n_basis;

    self_distance_matrix(n, |i, j| row_euclidean(coefs, i, j, nb))
}

/// Compute the basis-coefficient semimetric cross-distance matrix.
///
/// Both datasets are independently projected onto the same basis
/// specification, and pairwise Euclidean distances in coefficient space
/// are computed.
///
/// # Arguments
///
/// * `data1` - First dataset (`n1 x m`, column-major).
/// * `data2` - Second dataset (`n2 x m`, column-major).
/// * `argvals` - Evaluation grid (length `m`).
/// * `nbasis` - Number of basis functions.
/// * `basis_type` - [`ProjectionBasisType::Bspline`] or [`ProjectionBasisType::Fourier`].
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::basis::ProjectionBasisType;
/// use fdars_core::metric::basis_coef_cross_1d;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data1 = FdMatrix::from_column_major(
///     (0..60).map(|i| (i as f64 * 0.1).sin()).collect(), 3, 20,
/// ).unwrap();
/// let data2 = FdMatrix::from_column_major(
///     (0..40).map(|i| (i as f64 * 0.2).cos()).collect(), 2, 20,
/// ).unwrap();
/// let dist = basis_coef_cross_1d(&data1, &data2, &argvals, 7, ProjectionBasisType::Fourier);
/// assert_eq!(dist.shape(), (3, 2));
/// ```
pub fn basis_coef_cross_1d(
    data1: &FdMatrix,
    data2: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: ProjectionBasisType,
) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();

    if n1 == 0 || n2 == 0 || m == 0 || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }

    let proj1 = match fdata_to_basis(data1, argvals, nbasis, basis_type) {
        Some(p) => p,
        None => return FdMatrix::zeros(0, 0),
    };
    let proj2 = match fdata_to_basis(data2, argvals, nbasis, basis_type) {
        Some(p) => p,
        None => return FdMatrix::zeros(0, 0),
    };
    let coefs1 = &proj1.coefficients;
    let coefs2 = &proj2.coefficients;
    let nb = proj1.n_basis.min(proj2.n_basis);

    cross_distance_matrix(n1, n2, |i, j| row_euclidean_cross(coefs1, i, coefs2, j, nb))
}
