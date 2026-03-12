//! LDA: Linear Discriminant Analysis internals.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    build_feature_matrix, compute_accuracy, confusion_matrix, remap_labels, ClassifResult,
};

/// Compute per-class means, counts, and priors from labeled features.
fn class_means_and_priors(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> (Vec<Vec<f64>>, Vec<usize>, Vec<f64>) {
    let n = features.nrows();
    let d = features.ncols();
    let mut counts = vec![0usize; g];
    let mut class_means = vec![vec![0.0; d]; g];
    for i in 0..n {
        let c = labels[i];
        counts[c] += 1;
        for j in 0..d {
            class_means[c][j] += features[(i, j)];
        }
    }
    for c in 0..g {
        if counts[c] > 0 {
            for j in 0..d {
                class_means[c][j] /= counts[c] as f64;
            }
        }
    }
    let priors: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();
    (class_means, counts, priors)
}

/// Compute pooled within-class covariance (symmetric, regularized).
fn pooled_within_cov(
    features: &FdMatrix,
    labels: &[usize],
    class_means: &[Vec<f64>],
    g: usize,
) -> Vec<f64> {
    let n = features.nrows();
    let d = features.ncols();
    let mut cov = vec![0.0; d * d];
    for i in 0..n {
        let c = labels[i];
        for r in 0..d {
            let dr = features[(i, r)] - class_means[c][r];
            for s in r..d {
                let val = dr * (features[(i, s)] - class_means[c][s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    let scale = (n - g).max(1) as f64;
    for v in cov.iter_mut() {
        *v /= scale;
    }
    for j in 0..d {
        cov[j * d + j] += 1e-6;
    }
    cov
}

/// Compute per-class means and pooled within-class covariance.
pub(crate) fn lda_params(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
    let (class_means, _counts, priors) = class_means_and_priors(features, labels, g);
    let cov = pooled_within_cov(features, labels, &class_means, g);
    (class_means, cov, priors)
}

/// Cholesky factorization of d×d row-major matrix.
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
                detail: format!("non-positive diagonal at index {j}"),
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

/// Forward solve L * x = b.
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

/// Mahalanobis distance squared: (x-mu)^T Sigma^{-1} (x-mu) via Cholesky.
pub(crate) fn mahalanobis_sq(x: &[f64], mu: &[f64], chol: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = x.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let y = forward_solve(chol, &diff, d);
    y.iter().map(|&v| v * v).sum()
}

/// LDA prediction: assign to class minimizing Mahalanobis distance (with prior).
pub(crate) fn lda_predict(
    features: &FdMatrix,
    class_means: &[Vec<f64>],
    cov_chol: &[f64],
    priors: &[f64],
    g: usize,
) -> Vec<usize> {
    let n = features.nrows();
    let d = features.ncols();

    (0..n)
        .map(|i| {
            let xi: Vec<f64> = (0..d).map(|j| features[(i, j)]).collect();
            let mut best_class = 0;
            let mut best_score = f64::NEG_INFINITY;
            for c in 0..g {
                let maha = mahalanobis_sq(&xi, &class_means[c], cov_chol, d);
                let score = priors[c].max(1e-15).ln() - 0.5 * maha;
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }
            best_class
        })
        .collect()
}

/// FPC + LDA classification.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Class labels (length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
/// Returns [`FdarError::ComputationFailed`] if the pooled covariance Cholesky factorization fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_lda(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<ClassifResult, FdarError> {
    let n = data.nrows();
    if n == 0 || y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/y",
            expected: "n > 0 and y.len() == n".to_string(),
            actual: format!("n={}, y.len()={}", n, y.len()),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be > 0".to_string(),
        });
    }

    let (labels, g) = remap_labels(y);
    if g < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("need at least 2 classes, got {g}"),
        });
    }

    let (features, _mean, _rotation) = build_feature_matrix(data, scalar_covariates, ncomp)?;
    let d = features.ncols();
    let (class_means, cov, priors) = lda_params(&features, &labels, g);
    let chol = cholesky_d(&cov, d)?;

    let predicted = lda_predict(&features, &class_means, &chol, &priors, g);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifResult {
        predicted,
        probabilities: None,
        accuracy,
        confusion,
        n_classes: g,
        ncomp: features.ncols().min(ncomp),
    })
}
