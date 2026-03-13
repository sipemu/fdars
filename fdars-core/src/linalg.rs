//! Shared linear algebra helpers: Cholesky, forward-solve, Mahalanobis distance.

use crate::error::FdarError;

/// Cholesky factorization of a d×d row-major symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor L such that `mat = L * L^T`,
/// stored as a d×d flat row-major array.
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] if the matrix is not positive-definite
/// (non-positive diagonal encountered during factorization).
pub(crate) fn cholesky_d(mat: &[f64], d: usize) -> Result<Vec<f64>, FdarError> {
    let mut l = vec![0.0; d * d];
    for j in 0..d {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * d + k] * l[j * d + k];
        }
        let diag = mat[j * d + j] - sum;
        if diag <= 0.0 {
            return Err(FdarError::ComputationFailed {
                operation: "cholesky_d",
                detail: format!("non-positive diagonal at index {j}; matrix may not be positive-definite — check for collinear inputs or add regularization"),
            });
        }
        l[j * d + j] = diag.sqrt();
        for i in (j + 1)..d {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * d + k] * l[j * d + k];
            }
            l[i * d + j] = (mat[i * d + j] - s) / l[j * d + j];
        }
    }
    Ok(l)
}

/// Forward substitution: solve `L * x = b` where L is d×d lower-triangular (row-major).
pub(crate) fn forward_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut x = vec![0.0; d];
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * d + j] * x[j];
        }
        x[i] = (b[i] - s) / l[i * d + i];
    }
    x
}

/// Mahalanobis distance squared: `(x - mu)^T Sigma^{-1} (x - mu)` via Cholesky factor L.
pub(crate) fn mahalanobis_sq(x: &[f64], mu: &[f64], chol: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let y = forward_solve(chol, &diff, d);
    y.iter().map(|&v| v * v).sum()
}

/// Log-determinant from Cholesky factor L: `2 * sum(ln(L_ii))`.
pub(crate) fn log_det_from_cholesky(l: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += l[i * d + i].ln();
    }
    2.0 * s
}
