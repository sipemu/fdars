//! Basis projection and reconstruction for functional data.

use super::bspline::bspline_basis;
use super::fourier::fourier_basis;
use super::helpers::svd_pseudoinverse;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Simple basis type selector for projection functions.
///
/// Used by [`fdata_to_basis_1d`] and [`basis_to_fdata_1d`] to choose between
/// B-spline and Fourier basis systems. For penalized smoothing with additional
/// parameters (order, period), see [`BasisType`](crate::smooth_basis::BasisType).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ProjectionBasisType {
    /// B-spline basis (order 4 / cubic).
    Bspline,
    /// Fourier basis.
    Fourier,
}

impl ProjectionBasisType {
    /// Convert from legacy integer encoding (0 = B-spline, 1 = Fourier).
    ///
    /// Returns `Bspline` for any value other than 1, matching the previous
    /// `if basis_type == 1 { Fourier } else { Bspline }` convention.
    #[must_use]
    pub fn from_i32(value: i32) -> Self {
        if value == 1 {
            Self::Fourier
        } else {
            Self::Bspline
        }
    }

    /// Convert to the legacy integer encoding (0 = B-spline, 1 = Fourier).
    #[must_use]
    pub fn to_i32(self) -> i32 {
        match self {
            Self::Bspline => 0,
            Self::Fourier => 1,
        }
    }
}

/// Result of basis projection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct BasisProjectionResult {
    /// Coefficient matrix (n_samples x n_basis)
    pub coefficients: FdMatrix,
    /// Number of basis functions used
    pub n_basis: usize,
}

/// Evaluate the appropriate basis on `argvals`.
fn evaluate_projection_basis(
    argvals: &[f64],
    nbasis: usize,
    basis_type: ProjectionBasisType,
) -> Vec<f64> {
    match basis_type {
        ProjectionBasisType::Fourier => fourier_basis(argvals, nbasis),
        ProjectionBasisType::Bspline => {
            // For order 4 B-splines: nbasis = nknots + order, so nknots = nbasis - 4
            bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4)
        }
    }
}

/// Project functional data to basis coefficients.
///
/// # Arguments
/// * `data` - Column-major FdMatrix (n x m)
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of basis functions
/// * `basis_type` - Basis type to use
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::basis::projection::{fdata_to_basis, ProjectionBasisType};
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     argvals.iter().map(|&t| (t * 6.0).sin()).collect(),
///     1, 20,
/// ).unwrap();
/// let result = fdata_to_basis(&data, &argvals, 7, ProjectionBasisType::Fourier).unwrap();
/// assert_eq!(result.coefficients.nrows(), 1);
/// assert_eq!(result.n_basis, 7);
/// ```
pub fn fdata_to_basis(
    data: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: ProjectionBasisType,
) -> Option<BasisProjectionResult> {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 || argvals.len() != m || nbasis < 2 {
        return None;
    }

    let basis = evaluate_projection_basis(argvals, nbasis, basis_type);

    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let btb_inv = svd_pseudoinverse(&btb)?;
    let proj = btb_inv * b_mat.transpose();

    // Per-curve coefficient rows. Collect as `Vec<Vec<f64>>` (one row per curve)
    // and scatter into the column-major FdMatrix via `[(i, k)]` indexing. The
    // previous code built a curve-major flat buffer and passed it to
    // `from_column_major`, which transposes/scrambles the result whenever
    // `n > 1` (it round-trips only for a single curve) — GH #33.
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
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

    let mut coefficients = FdMatrix::zeros(n, actual_nbasis);
    for (i, row) in rows.iter().enumerate() {
        for (k, &c) in row.iter().enumerate() {
            coefficients[(i, k)] = c;
        }
    }

    Some(BasisProjectionResult {
        coefficients,
        n_basis: actual_nbasis,
    })
}

/// Project functional data to basis coefficients (legacy i32 interface).
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
    fdata_to_basis(
        data,
        argvals,
        nbasis,
        ProjectionBasisType::from_i32(basis_type),
    )
}

/// Reconstruct functional data from basis coefficients.
///
/// # Arguments
/// * `coefs` - Coefficient matrix (n x n_basis)
/// * `argvals` - Evaluation points
/// * `nbasis` - Number of basis functions
/// * `basis_type` - Basis type to use
pub fn basis_to_fdata(
    coefs: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: ProjectionBasisType,
) -> FdMatrix {
    let n = coefs.nrows();
    let coefs_ncols = coefs.ncols();
    let m = argvals.len();
    if n == 0 || m == 0 || nbasis < 2 {
        return FdMatrix::zeros(0, 0);
    }

    let basis = evaluate_projection_basis(argvals, nbasis, basis_type);

    let actual_nbasis = basis.len() / m;

    // Per-curve reconstruction rows, scattered into the column-major FdMatrix
    // via `[(i, j)]`. As in `fdata_to_basis`, building a curve-major flat buffer
    // and passing it to `from_column_major` transposes the output for n > 1
    // (GH #33).
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
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

    let mut out = FdMatrix::zeros(n, m);
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    out
}

/// Reconstruct functional data from basis coefficients (legacy i32 interface).
pub fn basis_to_fdata_1d(
    coefs: &FdMatrix,
    argvals: &[f64],
    nbasis: usize,
    basis_type: i32,
) -> FdMatrix {
    basis_to_fdata(
        coefs,
        argvals,
        nbasis,
        ProjectionBasisType::from_i32(basis_type),
    )
}
