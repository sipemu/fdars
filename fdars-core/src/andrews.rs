//! Andrews curves transformation for multivariate and functional data.
//!
//! Andrews curves map p-dimensional observations to functional curves using
//! a Fourier-like bijection:
//!
//! f_x(t) = x₁/√2 + x₂·sin(t) + x₃·cos(t) + x₄·sin(2t) + x₅·cos(2t) + …
//!
//! for t ∈ \[−π, π\].
//!
//! This provides a dimension-reduction visualization: each multivariate
//! observation becomes a curve, and observations that are close in the
//! original space produce curves that are close pointwise.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use std::f64::consts::PI;

/// Result of Andrews curves transformation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AndrewsResult {
    /// Transformed functional data (n × n_grid).
    pub curves: FdMatrix,
    /// Grid points in \[−π, π\], length `n_grid`.
    pub argvals: Vec<f64>,
    /// Number of variables used (p).
    pub n_vars: usize,
}

/// Andrews loadings: transform principal component loadings to Andrews curves.
///
/// Each column of a rotation matrix (from FPCA) is treated as a
/// p-dimensional observation and projected into Andrews space,
/// producing one Andrews curve per component.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AndrewsLoadings {
    /// Loading curves (ncomp × n_grid).
    pub loadings: FdMatrix,
    /// Grid points in \[−π, π\], length `n_grid`.
    pub argvals: Vec<f64>,
    /// Number of variables/components.
    pub n_vars: usize,
}

/// Evaluate the k-th Andrews basis function at t.
///
/// The basis sequence is:
/// - k = 0: 1/√2
/// - k = 1: sin(t)
/// - k = 2: cos(t)
/// - k = 3: sin(2t)
/// - k = 4: cos(2t)
/// - …
#[inline]
fn andrews_basis(t: f64, k: usize) -> f64 {
    if k == 0 {
        return std::f64::consts::FRAC_1_SQRT_2;
    }
    let j = k.div_ceil(2) as f64;
    if k % 2 == 1 {
        (j * t).sin()
    } else {
        (j * t).cos()
    }
}

/// Transform multivariate observations to Andrews curves.
///
/// Each row of `data` (an n × p matrix) is a p-dimensional observation.
/// The function evaluates the Andrews representation on a uniform grid
/// of `n_grid` points in \[−π, π\].
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `n_grid` is zero.
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or zero columns.
#[must_use = "returns the Andrews curves without modifying the input"]
pub fn andrews_transform(data: &FdMatrix, n_grid: usize) -> Result<AndrewsResult, FdarError> {
    let (n, p) = data.shape();
    if n == 0 || p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-zero rows and columns".to_string(),
            actual: format!("{n} x {p}"),
        });
    }
    if n_grid == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be at least 1".to_string(),
        });
    }

    let argvals = make_grid(n_grid);
    let mut curves = FdMatrix::zeros(n, n_grid);

    // Pre-compute basis values: basis_vals[g][k] = andrews_basis(argvals[g], k)
    let basis_vals: Vec<Vec<f64>> = argvals
        .iter()
        .map(|&t| (0..p).map(|k| andrews_basis(t, k)).collect())
        .collect();

    for i in 0..n {
        for (g, bv) in basis_vals.iter().enumerate() {
            let mut val = 0.0;
            for k in 0..p {
                val += data[(i, k)] * bv[k];
            }
            curves[(i, g)] = val;
        }
    }

    Ok(AndrewsResult {
        curves,
        argvals,
        n_vars: p,
    })
}

/// Transform FPCA loadings to Andrews curves.
///
/// Given a rotation matrix of shape m × ncomp (as produced by
/// [`crate::regression::fdata_to_pc_1d`]), this function treats each of
/// the `ncomp` columns as an m-dimensional observation and computes its
/// Andrews curve representation.  The result contains `ncomp` curves,
/// each evaluated on `n_grid` points in \[−π, π\].
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `n_grid` is zero.
/// Returns [`FdarError::InvalidDimension`] if `rotation` has zero rows or zero columns.
#[must_use = "returns the Andrews loadings without modifying the input"]
pub fn andrews_loadings(rotation: &FdMatrix, n_grid: usize) -> Result<AndrewsLoadings, FdarError> {
    let (m, ncomp) = rotation.shape();
    if m == 0 || ncomp == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "rotation",
            expected: "non-zero rows and columns".to_string(),
            actual: format!("{m} x {ncomp}"),
        });
    }
    if n_grid == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_grid",
            message: "must be at least 1".to_string(),
        });
    }

    // Transpose rotation (m × ncomp) to (ncomp × m) so each component
    // becomes a row (an m-dimensional observation).
    let mut transposed_data = vec![0.0; ncomp * m];
    for j in 0..ncomp {
        let col = rotation.column(j);
        for i in 0..m {
            // transposed is ncomp × m in column-major:
            // element (row=j, col=i) => index j + i * ncomp
            transposed_data[j + i * ncomp] = col[i];
        }
    }
    let transposed = FdMatrix::from_column_major(transposed_data, ncomp, m)?;

    let result = andrews_transform(&transposed, n_grid)?;

    Ok(AndrewsLoadings {
        loadings: result.curves,
        argvals: result.argvals,
        n_vars: m,
    })
}

/// Create a uniform grid of `n` points in \[−π, π\].
fn make_grid(n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![0.0];
    }
    let step = 2.0 * PI / (n - 1) as f64;
    (0..n).map(|i| -PI + i as f64 * step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build data from row-major layout.
    fn row_major_matrix(data: &[f64], nrows: usize, ncols: usize) -> FdMatrix {
        let mut col_major = vec![0.0; nrows * ncols];
        for i in 0..nrows {
            for j in 0..ncols {
                col_major[i + j * nrows] = data[i * ncols + j];
            }
        }
        FdMatrix::from_column_major(col_major, nrows, ncols).unwrap()
    }

    #[test]
    fn andrews_basis_values() {
        let t = 1.0;
        assert!((andrews_basis(t, 0) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
        assert!((andrews_basis(t, 1) - t.sin()).abs() < 1e-12);
        assert!((andrews_basis(t, 2) - t.cos()).abs() < 1e-12);
        assert!((andrews_basis(t, 3) - (2.0 * t).sin()).abs() < 1e-12);
        assert!((andrews_basis(t, 4) - (2.0 * t).cos()).abs() < 1e-12);
        assert!((andrews_basis(t, 5) - (3.0 * t).sin()).abs() < 1e-12);
        assert!((andrews_basis(t, 6) - (3.0 * t).cos()).abs() < 1e-12);
    }

    #[test]
    fn constant_curve_from_unit_first_var() {
        // [1, 0, 0] → f(t) = 1/√2 for all t
        let data = row_major_matrix(&[1.0, 0.0, 0.0], 1, 3);
        let result = andrews_transform(&data, 50).unwrap();
        assert_eq!(result.curves.nrows(), 1);
        assert_eq!(result.curves.ncols(), 50);
        for g in 0..50 {
            assert!(
                (result.curves[(0, g)] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12,
                "grid point {g}: expected 1/sqrt(2), got {}",
                result.curves[(0, g)]
            );
        }
    }

    #[test]
    fn sin_curve_from_unit_second_var() {
        // [0, 1, 0] → f(t) = sin(t)
        let data = row_major_matrix(&[0.0, 1.0, 0.0], 1, 3);
        let result = andrews_transform(&data, 100).unwrap();
        for (g, &t) in result.argvals.iter().enumerate() {
            assert!(
                (result.curves[(0, g)] - t.sin()).abs() < 1e-12,
                "at t={t}: expected sin(t)={}, got {}",
                t.sin(),
                result.curves[(0, g)]
            );
        }
    }

    #[test]
    fn cos_curve_from_unit_third_var() {
        // [0, 0, 1] → f(t) = cos(t)
        let data = row_major_matrix(&[0.0, 0.0, 1.0], 1, 3);
        let result = andrews_transform(&data, 100).unwrap();
        for (g, &t) in result.argvals.iter().enumerate() {
            assert!(
                (result.curves[(0, g)] - t.cos()).abs() < 1e-12,
                "at t={t}: expected cos(t)={}, got {}",
                t.cos(),
                result.curves[(0, g)]
            );
        }
    }

    #[test]
    fn correct_output_dimensions() {
        let data = row_major_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let result = andrews_transform(&data, 75).unwrap();
        assert_eq!(result.curves.nrows(), 2);
        assert_eq!(result.curves.ncols(), 75);
        assert_eq!(result.argvals.len(), 75);
        assert_eq!(result.n_vars, 3);
    }

    #[test]
    fn error_on_empty_data() {
        let data = FdMatrix::zeros(0, 0);
        let err = andrews_transform(&data, 50).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn error_on_zero_grid() {
        let data = row_major_matrix(&[1.0, 2.0], 1, 2);
        let err = andrews_transform(&data, 0).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }));
    }

    #[test]
    fn error_on_zero_rows() {
        let data = FdMatrix::zeros(0, 3);
        let err = andrews_transform(&data, 50).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn error_on_zero_cols() {
        let data = FdMatrix::zeros(5, 0);
        let err = andrews_transform(&data, 50).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn andrews_loadings_correct_shape() {
        // rotation: m=10 grid points, ncomp=3 components → 3 Andrews curves
        let rotation = FdMatrix::zeros(10, 3);
        let result = andrews_loadings(&rotation, 50).unwrap();
        assert_eq!(result.loadings.nrows(), 3);
        assert_eq!(result.loadings.ncols(), 50);
        assert_eq!(result.argvals.len(), 50);
        assert_eq!(result.n_vars, 10);
    }

    #[test]
    fn andrews_loadings_identity_column() {
        // rotation: 3 × 1 matrix with column [1, 0, 0]
        // This should give the same Andrews curve as transforming [1, 0, 0]
        let rotation = row_major_matrix(&[1.0, 0.0, 0.0], 3, 1);
        let loadings = andrews_loadings(&rotation, 50).unwrap();

        let data = row_major_matrix(&[1.0, 0.0, 0.0], 1, 3);
        let direct = andrews_transform(&data, 50).unwrap();

        for g in 0..50 {
            assert!(
                (loadings.loadings[(0, g)] - direct.curves[(0, g)]).abs() < 1e-12,
                "grid point {g}: loadings {} vs direct {}",
                loadings.loadings[(0, g)],
                direct.curves[(0, g)]
            );
        }
    }

    #[test]
    fn deterministic_output() {
        let data = row_major_matrix(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let r1 = andrews_transform(&data, 50).unwrap();
        let r2 = andrews_transform(&data, 50).unwrap();
        assert_eq!(r1.curves, r2.curves);
        assert_eq!(r1.argvals, r2.argvals);
    }

    #[test]
    fn grid_endpoints() {
        let data = row_major_matrix(&[1.0], 1, 1);
        let result = andrews_transform(&data, 101).unwrap();
        assert!((result.argvals[0] - (-PI)).abs() < 1e-12);
        assert!((result.argvals[100] - PI).abs() < 1e-12);
    }

    #[test]
    fn single_grid_point() {
        let data = row_major_matrix(&[1.0, 2.0], 1, 2);
        let result = andrews_transform(&data, 1).unwrap();
        assert_eq!(result.curves.ncols(), 1);
        assert_eq!(result.argvals.len(), 1);
        // t = 0, so f(0) = 1/√2 + 2*sin(0) = 1/√2
        assert!((result.curves[(0, 0)] - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-12);
    }

    #[test]
    fn linearity() {
        // Andrews transform is linear: T(a*x + b*y) = a*T(x) + b*T(y)
        let x = row_major_matrix(&[1.0, 2.0, 3.0], 1, 3);
        let y = row_major_matrix(&[4.0, -1.0, 0.5], 1, 3);
        let combined = row_major_matrix(
            &[
                2.0 * 1.0 + 3.0 * 4.0,
                2.0 * 2.0 - 3.0,
                2.0 * 3.0 + 3.0 * 0.5,
            ],
            1,
            3,
        );

        let n_grid = 50;
        let tx = andrews_transform(&x, n_grid).unwrap();
        let ty = andrews_transform(&y, n_grid).unwrap();
        let tc = andrews_transform(&combined, n_grid).unwrap();

        for g in 0..n_grid {
            let expected = 2.0 * tx.curves[(0, g)] + 3.0 * ty.curves[(0, g)];
            assert!(
                (tc.curves[(0, g)] - expected).abs() < 1e-10,
                "linearity failed at grid point {g}: {} vs {expected}",
                tc.curves[(0, g)]
            );
        }
    }

    #[test]
    fn andrews_loadings_error_on_empty() {
        let rotation = FdMatrix::zeros(0, 0);
        let err = andrews_loadings(&rotation, 50).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn andrews_loadings_error_on_zero_grid() {
        let rotation = FdMatrix::zeros(5, 2);
        let err = andrews_loadings(&rotation, 0).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }));
    }
}
