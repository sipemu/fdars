//! ClassifFit: fitted classification model with explainability support.

use crate::error::FdarError;
use crate::explain_generic::{FpcPredictor, TaskType};
use crate::matrix::FdMatrix;

use super::knn::knn_predict_loo;
use super::lda::{lda_params, lda_predict};
use super::qda::{build_qda_params, qda_predict};
use super::{
    build_feature_matrix, compute_accuracy, confusion_matrix, remap_labels, ClassifCvResult,
    ClassifResult,
};
use crate::linalg::{cholesky_d, mahalanobis_sq};

use super::cv::fclassif_cv;

/// Classification method with stored parameters for prediction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClassifMethod {
    /// Linear Discriminant Analysis.
    Lda {
        class_means: Vec<Vec<f64>>,
        cov_chol: Vec<f64>,
        priors: Vec<f64>,
        n_classes: usize,
    },
    /// Quadratic Discriminant Analysis.
    Qda {
        class_means: Vec<Vec<f64>>,
        class_chols: Vec<Vec<f64>>,
        class_log_dets: Vec<f64>,
        priors: Vec<f64>,
        n_classes: usize,
    },
    /// k-Nearest Neighbors.
    Knn {
        training_scores: FdMatrix,
        training_labels: Vec<usize>,
        k: usize,
        n_classes: usize,
    },
}

/// A fitted classification model that retains FPCA components for explainability.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ClassifFit {
    /// Classification result (predicted labels, accuracy, confusion matrix).
    pub result: ClassifResult,
    /// FPCA mean function (length m).
    pub fpca_mean: Vec<f64>,
    /// FPCA rotation matrix (m × ncomp).
    pub fpca_rotation: FdMatrix,
    /// FPCA scores (n × ncomp).
    pub fpca_scores: FdMatrix,
    /// Number of FPC components used.
    pub ncomp: usize,
    /// The classification method with stored parameters.
    pub method: ClassifMethod,
}

/// FPC + LDA classification, retaining FPCA and LDA parameters for explainability.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
/// Returns [`FdarError::ComputationFailed`] if the pooled covariance Cholesky factorization fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_lda_fit(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<ClassifFit, FdarError> {
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

    // _fit variants use FPCA-only features (no scalar_covariates) so that stored
    // dimensions are consistent with FpcPredictor::project() / predict_from_scores().
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = scalar_covariates; // acknowledged but not used — see docstring
    let d = features.ncols();
    let (class_means, cov, priors) = lda_params(&features, &labels, g);
    let chol = cholesky_d(&cov, d)?;

    let predicted = lda_predict(&features, &class_means, &chol, &priors, g);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features,
        ncomp: d,
        method: ClassifMethod::Lda {
            class_means,
            cov_chol: chol,
            priors,
            n_classes: g,
        },
    })
}

/// FPC + QDA classification, retaining FPCA and QDA parameters for explainability.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
/// Returns [`FdarError::ComputationFailed`] if a per-class covariance Cholesky factorization fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_qda_fit(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<ClassifFit, FdarError> {
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

    // _fit variants use FPCA-only features — see fclassif_lda_fit comment.
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = scalar_covariates;
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
    let d = features.ncols();

    Ok(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features,
        ncomp: d,
        method: ClassifMethod::Qda {
            class_means,
            class_chols,
            class_log_dets,
            priors,
            n_classes: g,
        },
    })
}

/// FPC + k-NN classification, retaining FPCA and training data for explainability.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or `y.len() != n`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if `k_nn` is zero.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
/// Returns [`FdarError::ComputationFailed`] if the SVD decomposition in FPCA fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_knn_fit(
    data: &FdMatrix,
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    k_nn: usize,
) -> Result<ClassifFit, FdarError> {
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
    if k_nn == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k_nn",
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

    // _fit variants use FPCA-only features — see fclassif_lda_fit comment.
    let (features, mean, rotation) = build_feature_matrix(data, None, ncomp)?;
    let _ = scalar_covariates;
    let d = features.ncols();

    let predicted = knn_predict_loo(&features, &labels, g, d, k_nn);
    let accuracy = compute_accuracy(&labels, &predicted);
    let confusion = confusion_matrix(&labels, &predicted, g);

    Ok(ClassifFit {
        result: ClassifResult {
            predicted,
            probabilities: None,
            accuracy,
            confusion,
            n_classes: g,
            ncomp: d,
        },
        fpca_mean: mean.clone(),
        fpca_rotation: rotation,
        fpca_scores: features.clone(),
        ncomp: d,
        method: ClassifMethod::Knn {
            training_scores: features,
            training_labels: labels,
            k: k_nn,
            n_classes: g,
        },
    })
}

// ---------------------------------------------------------------------------
// FpcPredictor impl for ClassifFit
// ---------------------------------------------------------------------------

impl FpcPredictor for ClassifFit {
    fn fpca_mean(&self) -> &[f64] {
        &self.fpca_mean
    }

    fn fpca_rotation(&self) -> &FdMatrix {
        &self.fpca_rotation
    }

    fn ncomp(&self) -> usize {
        self.ncomp
    }

    fn training_scores(&self) -> &FdMatrix {
        &self.fpca_scores
    }

    fn task_type(&self) -> TaskType {
        match &self.method {
            ClassifMethod::Lda { n_classes, .. }
            | ClassifMethod::Qda { n_classes, .. }
            | ClassifMethod::Knn { n_classes, .. } => {
                if *n_classes == 2 {
                    TaskType::BinaryClassification
                } else {
                    TaskType::MulticlassClassification(*n_classes)
                }
            }
        }
    }

    fn predict_from_scores(&self, scores: &[f64], _scalar_covariates: Option<&[f64]>) -> f64 {
        match &self.method {
            ClassifMethod::Lda {
                class_means,
                cov_chol,
                priors,
                n_classes,
            } => {
                let g = *n_classes;
                let d = scores.len();
                if g == 2 {
                    // Return P(Y=1) via softmax of discriminant scores
                    let score0 = priors[0].max(1e-15).ln()
                        - 0.5 * mahalanobis_sq(scores, &class_means[0], cov_chol, d);
                    let score1 = priors[1].max(1e-15).ln()
                        - 0.5 * mahalanobis_sq(scores, &class_means[1], cov_chol, d);
                    let max_s = score0.max(score1);
                    let exp0 = (score0 - max_s).exp();
                    let exp1 = (score1 - max_s).exp();
                    exp1 / (exp0 + exp1)
                } else {
                    // Return predicted class as f64
                    let mut best_class = 0;
                    let mut best_score = f64::NEG_INFINITY;
                    for c in 0..g {
                        let maha = mahalanobis_sq(scores, &class_means[c], cov_chol, d);
                        let s = priors[c].max(1e-15).ln() - 0.5 * maha;
                        if s > best_score {
                            best_score = s;
                            best_class = c;
                        }
                    }
                    best_class as f64
                }
            }
            ClassifMethod::Qda {
                class_means,
                class_chols,
                class_log_dets,
                priors,
                n_classes,
            } => {
                let g = *n_classes;
                let d = scores.len();
                if g == 2 {
                    // Return P(Y=1) via softmax of discriminant scores
                    let score0 = priors[0].max(1e-15).ln()
                        - 0.5
                            * (class_log_dets[0]
                                + mahalanobis_sq(scores, &class_means[0], &class_chols[0], d));
                    let score1 = priors[1].max(1e-15).ln()
                        - 0.5
                            * (class_log_dets[1]
                                + mahalanobis_sq(scores, &class_means[1], &class_chols[1], d));
                    let max_s = score0.max(score1);
                    let exp0 = (score0 - max_s).exp();
                    let exp1 = (score1 - max_s).exp();
                    exp1 / (exp0 + exp1)
                } else {
                    let mut best_class = 0;
                    let mut best_score = f64::NEG_INFINITY;
                    for c in 0..g {
                        let maha = mahalanobis_sq(scores, &class_means[c], &class_chols[c], d);
                        let s = priors[c].max(1e-15).ln() - 0.5 * (class_log_dets[c] + maha);
                        if s > best_score {
                            best_score = s;
                            best_class = c;
                        }
                    }
                    best_class as f64
                }
            }
            ClassifMethod::Knn {
                training_scores,
                training_labels,
                k,
                n_classes,
            } => {
                let g = *n_classes;
                let n_train = training_scores.nrows();
                let d = scores.len();
                let k_nn = (*k).min(n_train);

                let mut dists: Vec<(f64, usize)> = (0..n_train)
                    .map(|j| {
                        let d_sq: f64 = (0..d)
                            .map(|c| (scores[c] - training_scores[(j, c)]).powi(2))
                            .sum();
                        (d_sq, training_labels[j])
                    })
                    .collect();
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                let mut votes = vec![0usize; g];
                for &(_, label) in dists.iter().take(k_nn) {
                    if label < g {
                        votes[label] += 1;
                    }
                }

                if g == 2 {
                    // Return proportion voting for class 1 as probability
                    votes[1] as f64 / k_nn as f64
                } else {
                    // Return majority vote class as f64
                    votes
                        .iter()
                        .enumerate()
                        .max_by_key(|&(_, &v)| v)
                        .map_or(0.0, |(c, _)| c as f64)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Class probability vectors (for conformal prediction sets)
// ---------------------------------------------------------------------------

/// Compute full class probability vectors for each observation.
///
/// Returns `n × g` probability vectors suitable for conformal classification.
/// For each observation, the probabilities sum to 1.
pub(crate) fn classif_predict_probs(fit: &ClassifFit, scores: &FdMatrix) -> Vec<Vec<f64>> {
    let n = scores.nrows();
    let d = scores.ncols();
    match &fit.method {
        ClassifMethod::Lda {
            class_means,
            cov_chol,
            priors,
            n_classes,
        } => {
            let g = *n_classes;
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let disc: Vec<f64> = (0..g)
                        .map(|c| {
                            priors[c].max(1e-15).ln()
                                - 0.5 * mahalanobis_sq(&x, &class_means[c], cov_chol, d)
                        })
                        .collect();
                    softmax(&disc)
                })
                .collect()
        }
        ClassifMethod::Qda {
            class_means,
            class_chols,
            class_log_dets,
            priors,
            n_classes,
        } => {
            let g = *n_classes;
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let disc: Vec<f64> = (0..g)
                        .map(|c| {
                            priors[c].max(1e-15).ln()
                                - 0.5
                                    * (class_log_dets[c]
                                        + mahalanobis_sq(&x, &class_means[c], &class_chols[c], d))
                        })
                        .collect();
                    softmax(&disc)
                })
                .collect()
        }
        ClassifMethod::Knn {
            training_scores,
            training_labels,
            k,
            n_classes,
        } => {
            let g = *n_classes;
            let n_train = training_scores.nrows();
            let k_nn = (*k).min(n_train);
            (0..n)
                .map(|i| {
                    let x: Vec<f64> = (0..d).map(|j| scores[(i, j)]).collect();
                    let mut dists: Vec<(f64, usize)> = (0..n_train)
                        .map(|j| {
                            let d_sq: f64 = (0..d)
                                .map(|c| (x[c] - training_scores[(j, c)]).powi(2))
                                .sum();
                            (d_sq, training_labels[j])
                        })
                        .collect();
                    dists
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    let mut votes = vec![0usize; g];
                    for &(_, label) in dists.iter().take(k_nn) {
                        if label < g {
                            votes[label] += 1;
                        }
                    }
                    votes.iter().map(|&v| v as f64 / k_nn as f64).collect()
                })
                .collect()
        }
    }
}

/// Softmax of a vector of log-scores → probabilities.
fn softmax(scores: &[f64]) -> Vec<f64> {
    let max_s = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// ─── Config-based API ───────────────────────────────────────────────────────

/// Configuration for [`fclassif_cv`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifCvConfig {
    /// Classification method name (one of "lda", "qda", "knn", "kernel", "dd").
    pub method: String,
    /// Number of FPC components.
    pub ncomp: usize,
    /// Number of cross-validation folds.
    pub nfold: usize,
    /// Random seed for fold assignment.
    pub seed: u64,
}

impl Default for ClassifCvConfig {
    fn default() -> Self {
        Self {
            method: "lda".to_string(),
            ncomp: 3,
            nfold: 5,
            seed: 42,
        }
    }
}

/// Cross-validated classification using a configuration struct.
///
/// Equivalent to [`fclassif_cv`] but bundles method parameters in [`ClassifCvConfig`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `config.nfold < 2` or `config.nfold > n`.
/// Returns [`FdarError::InvalidParameter`] if `y` contains fewer than 2 distinct classes.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fclassif_cv_with_config(
    data: &FdMatrix,
    argvals: &[f64],
    y: &[usize],
    scalar_covariates: Option<&FdMatrix>,
    config: &ClassifCvConfig,
) -> Result<ClassifCvResult, FdarError> {
    fclassif_cv(
        data,
        argvals,
        y,
        scalar_covariates,
        &config.method,
        config.ncomp,
        config.nfold,
        config.seed,
    )
}
