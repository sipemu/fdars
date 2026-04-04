//! Common input validation helpers.
//!
//! These utilities centralise the dimension-checking boilerplate that recurs
//! across regression, classification, depth, and alignment entry points.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Validate functional data dimensions.
///
/// Checks that `data` has at least 1 row and 1 column, and that `argvals`
/// length matches the number of columns.
///
/// Returns `(n, m)` on success.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] when any check fails.
///
/// # Examples
///
/// ```
/// use fdars_core::validation::validate_fdata;
/// use fdars_core::matrix::FdMatrix;
///
/// let data = FdMatrix::zeros(10, 50);
/// let t: Vec<f64> = (0..50).map(|i| i as f64).collect();
/// let (n, m) = validate_fdata(&data, &t).unwrap();
/// assert_eq!((n, m), (10, 50));
/// ```
pub fn validate_fdata(data: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0 rows".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "m > 0 columns".to_string(),
            actual: format!("m = {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    Ok((n, m))
}

/// Validate that a response vector matches the data row count.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `y.len() != n`.
///
/// # Examples
///
/// ```
/// use fdars_core::validation::validate_response;
///
/// validate_response(&[1.0, 2.0, 3.0], 3).unwrap();
/// assert!(validate_response(&[1.0, 2.0], 3).is_err());
/// ```
pub fn validate_response(y: &[f64], n: usize) -> Result<(), FdarError> {
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements"),
            actual: format!("{} elements", y.len()),
        });
    }
    Ok(())
}

/// Validate class labels match data dimensions and have at least `min_classes` classes.
///
/// Labels are expected to be 0-indexed. Returns the number of distinct
/// classes (i.e. `max(y) + 1`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `y.len() != n`, or
/// [`FdarError::InvalidParameter`] if fewer than `min_classes` classes are found.
///
/// # Examples
///
/// ```
/// use fdars_core::validation::validate_labels;
///
/// let n_classes = validate_labels(&[0, 1, 0, 1], 4, 2).unwrap();
/// assert_eq!(n_classes, 2);
/// assert!(validate_labels(&[0, 0, 0], 3, 2).is_err());
/// ```
pub fn validate_labels(y: &[usize], n: usize, min_classes: usize) -> Result<usize, FdarError> {
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} elements"),
            actual: format!("{} elements", y.len()),
        });
    }
    let n_classes = y.iter().copied().max().map_or(0, |m| m + 1);
    if n_classes < min_classes {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least {min_classes} classes, got {n_classes}"),
        });
    }
    Ok(n_classes)
}

/// Validate that a distance matrix is square and optionally matches an expected size.
///
/// Returns the number of rows/columns `n`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the matrix is not square or
/// does not match the expected size.
///
/// # Examples
///
/// ```
/// use fdars_core::validation::validate_dist_mat;
/// use fdars_core::matrix::FdMatrix;
///
/// let dm = FdMatrix::zeros(5, 5);
/// assert_eq!(validate_dist_mat(&dm, None).unwrap(), 5);
/// assert_eq!(validate_dist_mat(&dm, Some(5)).unwrap(), 5);
/// assert!(validate_dist_mat(&dm, Some(4)).is_err());
/// ```
pub fn validate_dist_mat(
    dist_mat: &FdMatrix,
    expected_n: Option<usize>,
) -> Result<usize, FdarError> {
    let n = dist_mat.nrows();
    if dist_mat.ncols() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "dist_mat",
            expected: format!("{n} x {n} (square)"),
            actual: format!("{} x {}", n, dist_mat.ncols()),
        });
    }
    if let Some(exp) = expected_n {
        if n != exp {
            return Err(FdarError::InvalidDimension {
                parameter: "dist_mat",
                expected: format!("{exp} x {exp}"),
                actual: format!("{n} x {n}"),
            });
        }
    }
    Ok(n)
}

/// Validate and clamp the `ncomp` parameter.
///
/// Returns `min(ncomp, n, m)` after ensuring `ncomp >= 1`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp == 0`.
///
/// # Examples
///
/// ```
/// use fdars_core::validation::validate_ncomp;
///
/// assert_eq!(validate_ncomp(5, 10, 20).unwrap(), 5);
/// assert_eq!(validate_ncomp(100, 10, 20).unwrap(), 10);
/// assert!(validate_ncomp(0, 10, 20).is_err());
/// ```
pub fn validate_ncomp(ncomp: usize, n: usize, m: usize) -> Result<usize, FdarError> {
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }
    Ok(ncomp.min(n).min(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── validate_fdata ──────────────────────────────────────────────────

    #[test]
    fn fdata_ok() {
        let data = FdMatrix::zeros(10, 50);
        let t: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let (n, m) = validate_fdata(&data, &t).unwrap();
        assert_eq!((n, m), (10, 50));
    }

    #[test]
    fn fdata_zero_rows() {
        let data = FdMatrix::zeros(0, 5);
        let t = vec![0.0; 5];
        assert!(validate_fdata(&data, &t).is_err());
    }

    #[test]
    fn fdata_zero_cols() {
        let data = FdMatrix::zeros(5, 0);
        assert!(validate_fdata(&data, &[]).is_err());
    }

    #[test]
    fn fdata_argvals_mismatch() {
        let data = FdMatrix::zeros(5, 10);
        let t = vec![0.0; 8];
        assert!(validate_fdata(&data, &t).is_err());
    }

    // ── validate_response ───────────────────────────────────────────────

    #[test]
    fn response_ok() {
        validate_response(&[1.0, 2.0, 3.0], 3).unwrap();
    }

    #[test]
    fn response_mismatch() {
        assert!(validate_response(&[1.0, 2.0], 3).is_err());
    }

    // ── validate_labels ─────────────────────────────────────────────────

    #[test]
    fn labels_ok() {
        let nc = validate_labels(&[0, 1, 0, 1], 4, 2).unwrap();
        assert_eq!(nc, 2);
    }

    #[test]
    fn labels_too_few_classes() {
        assert!(validate_labels(&[0, 0, 0], 3, 2).is_err());
    }

    #[test]
    fn labels_length_mismatch() {
        assert!(validate_labels(&[0, 1], 3, 2).is_err());
    }

    // ── validate_dist_mat ───────────────────────────────────────────────

    #[test]
    fn dist_mat_ok() {
        let dm = FdMatrix::zeros(5, 5);
        assert_eq!(validate_dist_mat(&dm, None).unwrap(), 5);
        assert_eq!(validate_dist_mat(&dm, Some(5)).unwrap(), 5);
    }

    #[test]
    fn dist_mat_not_square() {
        let dm = FdMatrix::zeros(5, 3);
        assert!(validate_dist_mat(&dm, None).is_err());
    }

    #[test]
    fn dist_mat_wrong_size() {
        let dm = FdMatrix::zeros(5, 5);
        assert!(validate_dist_mat(&dm, Some(4)).is_err());
    }

    // ── validate_ncomp ──────────────────────────────────────────────────

    #[test]
    fn ncomp_ok() {
        assert_eq!(validate_ncomp(5, 10, 20).unwrap(), 5);
    }

    #[test]
    fn ncomp_clamped() {
        assert_eq!(validate_ncomp(100, 10, 20).unwrap(), 10);
        assert_eq!(validate_ncomp(100, 20, 10).unwrap(), 10);
    }

    #[test]
    fn ncomp_zero() {
        assert!(validate_ncomp(0, 10, 20).is_err());
    }
}
