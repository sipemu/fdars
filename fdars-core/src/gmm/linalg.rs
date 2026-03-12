//! Cholesky helpers for GMM covariance operations.

/// Cholesky factorization of a d×d matrix (row-major flat). Returns lower-triangular L.
pub(super) fn cholesky_d(mat: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for j in 0..d {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * d + k] * l[j * d + k];
        }
        let diag = mat[j * d + j] - sum;
        if diag <= 0.0 {
            return None;
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
    Some(l)
}

/// Log-determinant from Cholesky factor L: 2 * sum(log(L_ii)).
pub(super) fn log_det_from_cholesky(l: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += l[i * d + i].ln();
    }
    2.0 * s
}

/// Solve L * x = b (forward substitution).
pub(super) fn forward_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
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

/// Compute (z - mu)^T Sigma^{-1} (z - mu) using Cholesky factor L.
/// Also returns log|Sigma| = log_det_from_cholesky(L).
pub(super) fn mahalanobis_sq(z: &[f64], mu: &[f64], chol: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = z.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let y = forward_solve(chol, &diff, d);
    y.iter().map(|&v| v * v).sum()
}
