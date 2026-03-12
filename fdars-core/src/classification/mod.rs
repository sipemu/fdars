//! Functional classification with mixed scalar/functional predictors.
//!
//! Implements supervised classification for functional data using:
//! - [`fclassif_lda`] / [`fclassif_qda`] — FPC + LDA/QDA pipeline
//! - [`fclassif_knn`] — FPC + k-NN classifier
//! - [`fclassif_kernel`] — Nonparametric kernel classifier with mixed predictors
//! - [`fclassif_dd`] — Depth-based DD-classifier
//! - [`fclassif_cv`] — Cross-validated error rate
//!
//! ## Parameter ordering convention
//! All classifiers: `(data, y, [argvals,] [scalar_covariates,] method-params)`

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

pub mod cv;
pub mod dd;
pub mod fit;
pub mod kernel;
pub mod knn;
pub mod lda;
pub mod qda;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Classification result.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifResult {
    /// Predicted class labels (length n)
    pub predicted: Vec<usize>,
    /// Posterior/membership probabilities (n x G) — if available
    pub probabilities: Option<FdMatrix>,
    /// Training accuracy
    pub accuracy: f64,
    /// Confusion matrix (G x G): row = true, col = predicted
    pub confusion: Vec<Vec<usize>>,
    /// Number of classes
    pub n_classes: usize,
    /// Number of FPC components used
    pub ncomp: usize,
}

/// Cross-validation result.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassifCvResult {
    /// Mean error rate across folds
    pub error_rate: f64,
    /// Per-fold error rates
    pub fold_errors: Vec<f64>,
    /// Best ncomp (if tuned)
    pub best_ncomp: usize,
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/// Count distinct classes and remap labels to 0..G-1.
pub(crate) fn remap_labels(y: &[usize]) -> (Vec<usize>, usize) {
    let mut labels: Vec<usize> = y.to_vec();
    let mut unique: Vec<usize> = y.to_vec();
    unique.sort_unstable();
    unique.dedup();
    let g = unique.len();
    for label in &mut labels {
        *label = unique.iter().position(|&u| u == *label).unwrap_or(0);
    }
    (labels, g)
}

/// Build confusion matrix (G x G).
fn confusion_matrix(true_labels: &[usize], pred_labels: &[usize], g: usize) -> Vec<Vec<usize>> {
    let mut cm = vec![vec![0usize; g]; g];
    for (&t, &p) in true_labels.iter().zip(pred_labels.iter()) {
        if t < g && p < g {
            cm[t][p] += 1;
        }
    }
    cm
}

/// Compute per-class means, counts, and priors from labeled features.
pub(crate) fn class_means_and_priors(
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

/// Accuracy from labels.
fn compute_accuracy(true_labels: &[usize], pred_labels: &[usize]) -> f64 {
    let n = true_labels.len();
    if n == 0 {
        return 0.0;
    }
    let correct = true_labels
        .iter()
        .zip(pred_labels.iter())
        .filter(|(&t, &p)| t == p)
        .count();
    correct as f64 / n as f64
}

/// Extract FPC scores and append optional scalar covariates.
pub(crate) fn build_feature_matrix(
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<(FdMatrix, Vec<f64>, FdMatrix), FdarError> {
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let n = data.nrows();
    let d_pc = fpca.scores.ncols();
    let d_cov = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
    let d = d_pc + d_cov;

    let mut features = FdMatrix::zeros(n, d);
    for i in 0..n {
        for j in 0..d_pc {
            features[(i, j)] = fpca.scores[(i, j)];
        }
        if let Some(cov) = scalar_covariates {
            for j in 0..d_cov {
                features[(i, d_pc + j)] = cov[(i, j)];
            }
        }
    }

    Ok((features, fpca.mean, fpca.rotation))
}

// ---------------------------------------------------------------------------
// Re-exports — preserves the external API
// ---------------------------------------------------------------------------

pub use cv::fclassif_cv;
pub use dd::fclassif_dd;
pub(crate) use fit::classif_predict_probs;
pub use fit::{
    fclassif_cv_with_config, fclassif_knn_fit, fclassif_lda_fit, fclassif_qda_fit, ClassifCvConfig,
    ClassifFit, ClassifMethod,
};
pub use kernel::fclassif_kernel;
pub use knn::fclassif_knn;
pub use lda::fclassif_lda;
pub use qda::fclassif_qda;
