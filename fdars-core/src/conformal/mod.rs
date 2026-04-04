//! Conformal prediction intervals and prediction sets.
//!
//! Provides distribution-free uncertainty quantification for all regression and
//! classification frameworks in fdars:
//!
//! **Regression** (prediction intervals):
//! - [`conformal_fregre_lm`] — Linear functional regression
//! - [`conformal_fregre_np`] — Nonparametric kernel regression
//! - [`conformal_elastic_regression`] — Elastic alignment regression
//! - [`conformal_elastic_pcr`] — Elastic principal component regression
//! - [`conformal_generic_regression`] — Any [`FpcPredictor`](crate::explain_generic::FpcPredictor) model
//! - [`cv_conformal_regression`] — Cross-conformal (CV+) with closure
//! - [`jackknife_plus_regression`] — Jackknife+ with closure
//!
//! **Classification** (prediction sets):
//! - [`conformal_classif`] — LDA / QDA / kNN classifiers
//! - [`conformal_logistic`] — Functional logistic regression
//! - [`conformal_elastic_logistic`] — Elastic logistic regression
//! - [`conformal_generic_classification`] — Any [`FpcPredictor`](crate::explain_generic::FpcPredictor) model
//! - [`cv_conformal_classification`] — Cross-conformal (CV+) with closure

use crate::error::FdarError;
use crate::matrix::FdMatrix;

pub mod classification;
pub mod cv;
pub mod elastic;
pub mod generic;
pub mod regression;

#[cfg(test)]
mod tests;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Split-conformal method variant.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ConformalMethod {
    /// Random split into proper-training and calibration.
    Split,
    /// K-fold cross-conformal (CV+).
    CrossConformal { n_folds: usize },
    /// Leave-one-out jackknife+.
    JackknifePlus,
}

/// Non-conformity score type for classification.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum ClassificationScore {
    /// Least Ambiguous set-valued Classifier: `s = 1 - P(true class)`.
    Lac,
    /// Adaptive Prediction Sets: cumulative sorted probabilities.
    Aps,
}

/// Conformal prediction intervals for regression.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConformalRegressionResult {
    /// Point predictions on test data.
    pub predictions: Vec<f64>,
    /// Lower bounds of prediction intervals.
    pub lower: Vec<f64>,
    /// Upper bounds of prediction intervals.
    pub upper: Vec<f64>,
    /// Quantile of calibration residuals.
    pub residual_quantile: f64,
    /// Empirical coverage on calibration set.
    pub coverage: f64,
    /// Absolute residuals on calibration set.
    pub calibration_scores: Vec<f64>,
    /// Method used.
    pub method: ConformalMethod,
}

/// Conformal prediction sets for classification.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct ConformalClassificationResult {
    /// Argmax predictions for each test observation.
    pub predicted_classes: Vec<usize>,
    /// Set of plausible classes per test observation.
    pub prediction_sets: Vec<Vec<usize>>,
    /// Size of each prediction set.
    pub set_sizes: Vec<usize>,
    /// Mean prediction set size.
    pub average_set_size: f64,
    /// Empirical coverage on calibration set.
    pub coverage: f64,
    /// Non-conformity scores on calibration set.
    pub calibration_scores: Vec<f64>,
    /// Quantile of calibration scores.
    pub score_quantile: f64,
    /// Method used.
    pub method: ConformalMethod,
    /// Score type used.
    pub score_type: ClassificationScore,
}

/// Configuration for split-conformal prediction.
///
/// Collects the common tuning parameters shared by all conformal prediction
/// functions, with sensible defaults obtained via [`ConformalConfig::default()`].
///
/// # Example
/// ```no_run
/// use fdars_core::conformal::ConformalConfig;
///
/// let mut config = ConformalConfig::default();
/// config.alpha = 0.05; // 95% coverage
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ConformalConfig {
    /// Fraction of data reserved for calibration (default: 0.25).
    pub cal_fraction: f64,
    /// Miscoverage level, e.g. 0.1 for 90% intervals (default: 0.1).
    pub alpha: f64,
    /// Random seed for the calibration/training split (default: 42).
    pub seed: u64,
}

impl Default for ConformalConfig {
    fn default() -> Self {
        Self {
            cal_fraction: 0.25,
            alpha: 0.1,
            seed: 42,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Core helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Split indices into proper-training and calibration sets.
pub(super) fn conformal_split(n: usize, cal_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    use rand::prelude::*;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut all_idx: Vec<usize> = (0..n).collect();
    all_idx.shuffle(&mut rng);
    let n_cal = ((n as f64 * cal_fraction).round() as usize)
        .max(2)
        .min(n - 2);
    let n_proper = n - n_cal;
    let proper_idx = all_idx[..n_proper].to_vec();
    let cal_idx = all_idx[n_proper..].to_vec();
    (proper_idx, cal_idx)
}

/// Compute conformal quantile: the k-th smallest score where k = ceil((n+1)*(1-alpha)).
///
/// Uses exact order statistic (no interpolation) to preserve the finite-sample
/// coverage guarantee. Returns `f64::INFINITY` when k > n (conservative: infinite
/// interval gives 100% coverage).
pub(super) fn conformal_quantile(scores: &mut [f64], alpha: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    crate::helpers::sort_nan_safe(scores);
    let k = ((n + 1) as f64 * (1.0 - alpha)).ceil() as usize;
    if k > n {
        return f64::INFINITY;
    }
    scores[k.saturating_sub(1)]
}

/// Empirical coverage: fraction of scores <= quantile.
pub(super) fn empirical_coverage(scores: &[f64], quantile: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    scores.iter().filter(|&&s| s <= quantile).count() as f64 / n as f64
}

// Re-export canonical quantile from helpers (removes dead code duplicate).
#[allow(unused_imports)]
pub(super) use crate::helpers::quantile_sorted;

/// Build regression result from calibration residuals and test predictions.
pub(super) fn build_regression_result(
    mut cal_residuals: Vec<f64>,
    test_predictions: Vec<f64>,
    alpha: f64,
    method: ConformalMethod,
) -> ConformalRegressionResult {
    let residual_quantile = conformal_quantile(&mut cal_residuals, alpha);
    let coverage = empirical_coverage(&cal_residuals, residual_quantile);
    let lower = test_predictions
        .iter()
        .map(|&p| p - residual_quantile)
        .collect();
    let upper = test_predictions
        .iter()
        .map(|&p| p + residual_quantile)
        .collect();
    ConformalRegressionResult {
        predictions: test_predictions,
        lower,
        upper,
        residual_quantile,
        coverage,
        calibration_scores: cal_residuals,
        method,
    }
}

/// Compute LAC non-conformity score: 1 - P(true class).
pub(super) fn lac_score(probs: &[f64], true_class: usize) -> f64 {
    if true_class < probs.len() {
        1.0 - probs[true_class]
    } else {
        1.0
    }
}

/// Compute APS non-conformity score: cumulative probability until true class is included.
pub(super) fn aps_score(probs: &[f64], true_class: usize) -> f64 {
    let g = probs.len();
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0;
    for &c in &order {
        cum += probs[c];
        if c == true_class {
            return cum;
        }
    }
    1.0
}

/// Build LAC prediction set: include class k if 1 - P(k) <= quantile.
pub(super) fn lac_prediction_set(probs: &[f64], quantile: f64) -> Vec<usize> {
    (0..probs.len())
        .filter(|&k| 1.0 - probs[k] <= quantile)
        .collect()
}

/// Build APS prediction set: include classes in descending probability order until cumulative >= quantile.
///
/// The APS non-conformity score is the cumulative probability until the true class
/// is included. A class k is in the prediction set if its APS score <= the calibration
/// quantile, which means we include classes until cumulative probability reaches the quantile.
pub(super) fn aps_prediction_set(probs: &[f64], quantile: f64) -> Vec<usize> {
    let g = probs.len();
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0;
    let mut set = Vec::new();
    for &c in &order {
        set.push(c);
        cum += probs[c];
        if cum >= quantile {
            break;
        }
    }
    if set.is_empty() && g > 0 {
        set.push(order[0]);
    }
    set
}

/// Build classification result from calibration scores and test probabilities.
pub(super) fn build_classification_result(
    mut cal_scores: Vec<f64>,
    test_probs: &[Vec<f64>],
    test_pred_classes: Vec<usize>,
    alpha: f64,
    method: ConformalMethod,
    score_type: ClassificationScore,
) -> ConformalClassificationResult {
    let score_quantile = conformal_quantile(&mut cal_scores, alpha);
    let coverage = empirical_coverage(&cal_scores, score_quantile);

    let prediction_sets: Vec<Vec<usize>> = test_probs
        .iter()
        .map(|probs| match score_type {
            ClassificationScore::Lac => lac_prediction_set(probs, score_quantile),
            ClassificationScore::Aps => aps_prediction_set(probs, score_quantile),
        })
        .collect();

    let set_sizes: Vec<usize> = prediction_sets.iter().map(std::vec::Vec::len).collect();
    let average_set_size = if set_sizes.is_empty() {
        0.0
    } else {
        set_sizes.iter().sum::<usize>() as f64 / set_sizes.len() as f64
    };

    ConformalClassificationResult {
        predicted_classes: test_pred_classes,
        prediction_sets,
        set_sizes,
        average_set_size,
        coverage,
        calibration_scores: cal_scores,
        score_quantile,
        method,
        score_type,
    }
}

/// Compute non-conformity scores for classification calibration.
pub(super) fn compute_cal_scores(
    probs: &[Vec<f64>],
    true_classes: &[usize],
    score_type: ClassificationScore,
) -> Vec<f64> {
    probs
        .iter()
        .zip(true_classes.iter())
        .map(|(p, &y)| match score_type {
            ClassificationScore::Lac => lac_score(p, y),
            ClassificationScore::Aps => aps_score(p, y),
        })
        .collect()
}

/// Vertically stack two matrices with the same number of columns.
pub(super) fn vstack(a: &FdMatrix, b: &FdMatrix) -> FdMatrix {
    let m = a.ncols();
    debug_assert_eq!(m, b.ncols());
    let na = a.nrows();
    let nb = b.nrows();
    let mut out = FdMatrix::zeros(na + nb, m);
    for j in 0..m {
        for i in 0..na {
            out[(i, j)] = a[(i, j)];
        }
        for i in 0..nb {
            out[(na + i, j)] = b[(i, j)];
        }
    }
    out
}

/// Vertically stack two optional matrices.
pub(super) fn vstack_opt(a: Option<&FdMatrix>, b: Option<&FdMatrix>) -> Option<FdMatrix> {
    match (a, b) {
        (Some(a), Some(b)) => Some(vstack(a, b)),
        _ => None,
    }
}

/// Subset a usize vector by indices.
pub(super) fn subset_vec_usize(v: &[usize], indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| v[i]).collect()
}

/// Subset an i8 vector by indices.
pub(super) fn subset_vec_i8(v: &[i8], indices: &[usize]) -> Vec<i8> {
    indices.iter().map(|&i| v[i]).collect()
}

/// Argmax of a probability vector.
pub(super) fn argmax(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i)
}

/// Validate common inputs for split conformal.
pub(super) fn validate_split_inputs(
    n: usize,
    n_test: usize,
    cal_fraction: f64,
    alpha: f64,
) -> Result<(), FdarError> {
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 4 observations".to_string(),
            actual: format!("{n}"),
        });
    }
    if n_test == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: "at least 1 observation".to_string(),
            actual: "0".to_string(),
        });
    }
    if cal_fraction <= 0.0 || cal_fraction >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "cal_fraction",
            message: format!("must be in (0, 1), got {cal_fraction}"),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Re-exports — preserves the external API
// ═══════════════════════════════════════════════════════════════════════════

pub use classification::{conformal_classif, conformal_elastic_logistic, conformal_logistic};
pub use cv::{cv_conformal_classification, cv_conformal_regression, jackknife_plus_regression};
pub use elastic::{
    conformal_elastic_pcr, conformal_elastic_pcr_with_config, conformal_elastic_regression,
    conformal_elastic_regression_with_config,
};
pub use generic::{conformal_generic_classification, conformal_generic_regression};
pub use regression::{conformal_fregre_lm, conformal_fregre_np};
