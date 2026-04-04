//! Shared linear algebra helpers: Cholesky, forward/backward substitution, OLS utilities,
//! Mahalanobis distance.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

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

// ---------------------------------------------------------------------------
// Unified Cholesky / OLS helpers (consolidation of duplicated code from
// scalar_on_function, function_on_scalar, function_on_scalar_2d, famm)
// ---------------------------------------------------------------------------

/// Cholesky factorization: A = LL'.
///
/// Input: flat row-major p-by-p symmetric positive-definite matrix.
/// Returns the lower-triangular factor L (p-by-p flat row-major).
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] if the matrix is singular or
/// near-singular (diagonal element <= 1e-12 during factorization).
pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut diag = a[j * p + j];
        for k in 0..j {
            diag -= l[j * p + k] * l[j * p + k];
        }
        if diag <= 1e-12 {
            return Err(FdarError::ComputationFailed {
                operation: "Cholesky factorization",
                detail: "matrix is singular or near-singular; try reducing ncomp or check for collinear FPC scores".into(),
            });
        }
        l[j * p + j] = diag.sqrt();
        for i in (j + 1)..p {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = s / l[j * p + j];
        }
    }
    Ok(l)
}

/// Solve Lz = b (forward substitution) then L'x = z (backward substitution).
///
/// L is a p-by-p lower-triangular matrix stored flat row-major.
pub(crate) fn cholesky_forward_back(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut z = b.to_vec();
    for j in 0..p {
        for k in 0..j {
            z[j] -= l[j * p + k] * z[k];
        }
        z[j] /= l[j * p + j];
    }
    for j in (0..p).rev() {
        for k in (j + 1)..p {
            z[j] -= l[k * p + j] * z[k];
        }
        z[j] /= l[j * p + j];
    }
    z
}

/// Solve Ax = b via Cholesky decomposition (A must be symmetric positive definite).
pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let l = cholesky_factor(a, p)?;
    Ok(cholesky_forward_back(&l, b, p))
}

/// Compute X'X (symmetric, p-by-p stored flat row-major).
pub(crate) fn compute_xtx(x: &FdMatrix) -> Vec<f64> {
    let (n, p) = x.shape();
    let mut xtx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, k)] * x[(i, j)];
            }
            xtx[k * p + j] = s;
            xtx[j * p + k] = s;
        }
    }
    xtx
}
