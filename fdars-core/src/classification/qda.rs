//! QDA: Quadratic Discriminant Analysis internals.

use crate::error::FdarError;
use crate::linalg::{cholesky_d, log_det_from_cholesky, mahalanobis_sq};
use crate::matrix::FdMatrix;

use super::{
    build_feature_matrix, class_means_and_priors, compute_accuracy, confusion_matrix, remap_labels,
    ClassifResult,
};

/// Accumulate symmetric covariance from feature rows.
fn accumulate_class_cov(
    features: &FdMatrix,
    members: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for &i in members {
        for r in 0..d {
            let dr = features[(i, r)] - mean[r];
            for s in r..d {
                let val = dr * (features[(i, s)] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Per-class covariance matrices.
fn qda_class_covariances(
    features: &FdMatrix,
    labels: &[usize],
    class_means: &[Vec<f64>],
    g: usize,
) -> Vec<Vec<f64>> {
    let n = features.nrows();
    let d = features.ncols();

    (0..g)
        .map(|c| {
            let members: Vec<usize> = (0..n).filter(|&i| labels[i] == c).collect();
            let nc = members.len();
            let divisor = (nc.saturating_sub(1)).max(1) as f64;
            let mut cov = accumulate_class_cov(features, &members, &class_means[c], d);
            for v in &mut cov {
                *v /= divisor;
            }
            for j in 0..d {
                cov[j * d + j] += 1e-6;
            }
            cov
        })
        .collect()
}

/// Compute QDA parameters: class means, Cholesky factors, log-dets, priors.
pub(crate) fn build_qda_params(
    features: &FdMatrix,
    labels: &[usize],
    g: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, Vec<f64>), FdarError> {
    let d = features.ncols();
    let (class_means, _counts, priors) = class_means_and_priors(features, labels, g);
    let class_covs = qda_class_covariances(features, labels, &class_means, g);
    let mut class_chols = Vec::with_capacity(g);
    let mut class_log_dets = Vec::with_capacity(g);
    for cov in &class_covs {
        let chol = cholesky_d(cov, d)?;
        class_log_dets.push(log_det_from_cholesky(&chol, d));
        class_chols.push(chol);
    }
    Ok((class_means, class_chols, class_log_dets, priors))
}

/// QDA prediction: per-class covariance.
pub(crate) fn qda_predict(
    features: &FdMatrix,
    class_means: &[Vec<f64>],
    class_chols: &[Vec<f64>],
    class_log_dets: &[f64],
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
                let maha = mahalanobis_sq(&xi, &class_means[c], &class_chols[c], d);
                let score = priors[c].max(1e-15).ln() - 0.5 * (class_log_dets[c] + maha);
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }
            best_class
        })
        .collect()
}

/// FPC + QDA classification.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
/// Returns [`FdarError::ComputationFailed`] if a per-class covariance Cholesky factorization fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_qda(
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

    let (class_means, class_chols, class_log_dets, priors) =
        build_qda_params(&features, &labels, g)?;

    let predicted = qda_predict(
        &features,
        &class_means,
        &class_chols,
        &class_log_dets,
        &priors,
        g,
    );
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
