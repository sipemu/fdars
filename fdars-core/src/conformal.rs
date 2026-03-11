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
//! - [`conformal_generic_regression`] — Any [`FpcPredictor`] model
//! - [`cv_conformal_regression`] — Cross-conformal (CV+) with closure
//! - [`jackknife_plus_regression`] — Jackknife+ with closure
//!
//! **Classification** (prediction sets):
//! - [`conformal_classif`] — LDA / QDA / kNN classifiers
//! - [`conformal_logistic`] — Functional logistic regression
//! - [`conformal_elastic_logistic`] — Elastic logistic regression
//! - [`conformal_generic_classification`] — Any [`FpcPredictor`] model
//! - [`cv_conformal_classification`] — Cross-conformal (CV+) with closure

use crate::classification::{
    classif_predict_probs, fclassif_knn_fit, fclassif_lda_fit, fclassif_qda_fit, ClassifFit,
};
use crate::cv::{create_folds, fold_indices, subset_rows, subset_vec};
use crate::elastic_regression::{
    elastic_logistic, elastic_pcr, elastic_regression, ElasticPcrResult, ElasticRegressionResult,
    PcaMethod,
};
use crate::explain::{project_scores, subsample_rows};
use crate::explain_generic::{FpcPredictor, TaskType};
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{
    fregre_lm, fregre_np_mixed, functional_logistic, predict_fregre_lm, predict_fregre_np,
};
use rand::prelude::*;

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Split-conformal method variant.
#[derive(Debug, Clone, Copy)]
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
pub enum ClassificationScore {
    /// Least Ambiguous set-valued Classifier: `s = 1 - P(true class)`.
    Lac,
    /// Adaptive Prediction Sets: cumulative sorted probabilities.
    Aps,
}

/// Conformal prediction intervals for regression.
#[derive(Debug, Clone)]
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

// ═══════════════════════════════════════════════════════════════════════════
// Core helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Split indices into proper-training and calibration sets.
fn conformal_split(n: usize, cal_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
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

/// Compute conformal quantile: ceil((n_cal + 1) * (1 - alpha)) / n_cal, clamped to [0, 1].
fn conformal_quantile(scores: &mut [f64], alpha: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_level = (((n + 1) as f64 * (1.0 - alpha)).ceil() / n as f64).min(1.0);
    quantile_sorted(scores, q_level)
}

/// Empirical coverage: fraction of scores ≤ quantile.
fn empirical_coverage(scores: &[f64], quantile: f64) -> f64 {
    let n = scores.len();
    if n == 0 {
        return 0.0;
    }
    scores.iter().filter(|&&s| s <= quantile).count() as f64 / n as f64
}

/// Quantile of a pre-sorted slice using linear interpolation.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted[0];
    }
    let idx = q * (n - 1) as f64;
    let lo = (idx.floor() as usize).min(n - 1);
    let hi = (idx.ceil() as usize).min(n - 1);
    if lo == hi {
        sorted[lo]
    } else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Build regression result from calibration residuals and test predictions.
fn build_regression_result(
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
fn lac_score(probs: &[f64], true_class: usize) -> f64 {
    if true_class < probs.len() {
        1.0 - probs[true_class]
    } else {
        1.0
    }
}

/// Compute APS non-conformity score: cumulative probability until true class is included.
fn aps_score(probs: &[f64], true_class: usize) -> f64 {
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

/// Build LAC prediction set: include class k if 1 - P(k) ≤ quantile.
fn lac_prediction_set(probs: &[f64], quantile: f64) -> Vec<usize> {
    (0..probs.len())
        .filter(|&k| 1.0 - probs[k] <= quantile)
        .collect()
}

/// Build APS prediction set: include classes in descending probability order until cumulative ≥ 1 - quantile.
fn aps_prediction_set(probs: &[f64], quantile: f64) -> Vec<usize> {
    let g = probs.len();
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let threshold = 1.0 - quantile;
    let mut cum = 0.0;
    let mut set = Vec::new();
    for &c in &order {
        set.push(c);
        cum += probs[c];
        if cum >= threshold {
            break;
        }
    }
    if set.is_empty() && g > 0 {
        set.push(order[0]);
    }
    set
}

/// Build classification result from calibration scores and test probabilities.
fn build_classification_result(
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

    let set_sizes: Vec<usize> = prediction_sets.iter().map(|s| s.len()).collect();
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
fn compute_cal_scores(
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

/// Subset a usize vector by indices.
fn subset_vec_usize(v: &[usize], indices: &[usize]) -> Vec<usize> {
    indices.iter().map(|&i| v[i]).collect()
}

/// Subset an i8 vector by indices.
fn subset_vec_i8(v: &[i8], indices: &[usize]) -> Vec<i8> {
    indices.iter().map(|&i| v[i]).collect()
}

/// Argmax of a probability vector.
fn argmax(probs: &[f64]) -> usize {
    probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Validate common inputs for split conformal.
fn validate_split_inputs(n: usize, n_test: usize, cal_fraction: f64, alpha: f64) -> bool {
    n >= 4 && n_test > 0 && cal_fraction > 0.0 && cal_fraction < 1.0 && alpha > 0.0 && alpha < 1.0
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. Split Conformal Regression (with refit)
// ═══════════════════════════════════════════════════════════════════════════

/// Split-conformal prediction intervals for functional linear regression.
///
/// Splits data, refits [`fregre_lm`] on the proper-training subset,
/// computes absolute residuals on the calibration set, then applies
/// the conformal quantile to construct prediction intervals.
///
/// # Arguments
/// * `data` — Training functional data (n × m)
/// * `y` — Training response (length n)
/// * `test_data` — Test functional data (n_test × m)
/// * `scalar_train` / `scalar_test` — Optional scalar covariates
/// * `ncomp` — Number of FPC components
/// * `cal_fraction` — Fraction for calibration (0, 1)
/// * `alpha` — Miscoverage level (e.g. 0.1 for 90 % intervals)
/// * `seed` — Random seed
pub fn conformal_fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    ncomp: usize,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);
    if proper_idx.len() < ncomp + 2 {
        return None;
    }

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec(y, &proper_idx);
    let proper_sc = scalar_train.map(|sc| subsample_rows(sc, &proper_idx));

    let refit = fregre_lm(&proper_data, &proper_y, proper_sc.as_ref(), ncomp)?;

    // Calibration residuals
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_sc = scalar_train.map(|sc| subsample_rows(sc, &cal_idx));
    let cal_preds = predict_fregre_lm(&refit, &cal_data, cal_sc.as_ref());
    let cal_residuals: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (y[orig] - cal_preds[i]).abs())
        .collect();

    // Test predictions
    let test_preds = predict_fregre_lm(&refit, test_data, scalar_test);

    Some(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for nonparametric kernel regression.
///
/// Refits [`fregre_np_mixed`] on the proper-training subset.
pub fn conformal_fregre_np(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    argvals: &[f64],
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    h_func: f64,
    h_scalar: f64,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec(y, &proper_idx);
    let proper_sc = scalar_train.map(|sc| subsample_rows(sc, &proper_idx));

    // Validate that fregre_np_mixed can fit
    let _fit = fregre_np_mixed(
        &proper_data,
        &proper_y,
        argvals,
        proper_sc.as_ref(),
        h_func,
        h_scalar,
    )?;

    // Calibration predictions
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_sc = scalar_train.map(|sc| subsample_rows(sc, &cal_idx));
    let cal_preds = predict_fregre_np(
        &proper_data,
        &proper_y,
        proper_sc.as_ref(),
        &cal_data,
        cal_sc.as_ref(),
        argvals,
        h_func,
        h_scalar,
    );
    let cal_residuals: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (y[orig] - cal_preds[i]).abs())
        .collect();

    // Test predictions
    let test_preds = predict_fregre_np(
        &proper_data,
        &proper_y,
        proper_sc.as_ref(),
        test_data,
        scalar_test,
        argvals,
        h_func,
        h_scalar,
    );

    Some(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for elastic regression.
///
/// Refits [`elastic_regression`] on the proper-training subset and predicts
/// on calibration / test data using the estimated β(t) and warping.
pub fn conformal_elastic_regression(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    argvals: &[f64],
    ncomp_beta: usize,
    lambda: f64,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec(y, &proper_idx);

    let refit = elastic_regression(
        &proper_data,
        &proper_y,
        argvals,
        ncomp_beta,
        lambda,
        20,
        1e-4,
    )?;

    // Calibration predictions
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_preds = predict_elastic_reg(&refit, &cal_data, argvals);
    let cal_residuals: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (y[orig] - cal_preds[i]).abs())
        .collect();

    // Test predictions
    let test_preds = predict_elastic_reg(&refit, test_data, argvals);

    Some(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for elastic PCR.
///
/// Refits [`elastic_pcr`] on the proper-training subset.
pub fn conformal_elastic_pcr(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    pca_method: PcaMethod,
    lambda: f64,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec(y, &proper_idx);

    let refit = elastic_pcr(
        &proper_data,
        &proper_y,
        argvals,
        ncomp,
        pca_method,
        lambda,
        20,
        1e-4,
    )?;

    // Calibration predictions
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_preds = predict_elastic_pcr_fn(&refit, &cal_data, argvals)?;
    let cal_residuals: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (y[orig] - cal_preds[i]).abs())
        .collect();

    // Test predictions
    let test_preds = predict_elastic_pcr_fn(&refit, test_data, argvals)?;

    Some(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// Elastic prediction helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Predict from elastic regression result on new data.
///
/// Aligns new curves to the estimated β(t) and computes inner products.
fn predict_elastic_reg(
    result: &ElasticRegressionResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let weights = crate::helpers::simpsons_weights(argvals);
    let q_new = crate::alignment::srsf_transform(new_data, argvals);

    let mut preds = vec![0.0; n_new];
    for i in 0..n_new {
        let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
        // Align to β via DP
        let gam = crate::alignment::dp_alignment_core(&result.beta, &qi, argvals, 0.0);
        let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
        let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
        let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

        preds[i] = result.alpha;
        for j in 0..m {
            let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
            preds[i] += q_aligned * result.beta[j] * weights[j];
        }
    }
    preds
}

/// Predict from elastic PCR result on new data.
///
/// Aligns new curves to the Karcher mean, projects onto stored FPCA components,
/// and applies the linear model.
fn predict_elastic_pcr_fn(
    result: &ElasticPcrResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Option<Vec<f64>> {
    let (n_new, m) = new_data.shape();
    let km = &result.karcher;

    // Use the stored mean SRSF from the Karcher mean result
    let mean_srsf = &km.mean_srsf;
    let q_new = crate::alignment::srsf_transform(new_data, argvals);

    // Get PC scores for new curves
    let scores = match result.pca_method {
        PcaMethod::Vertical => {
            let fpca = result.vert_fpca.as_ref()?;
            let ncomp = fpca.scores.ncols();
            // eigenfunctions_q is (ncomp × (m+1)), use first m columns
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
                let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
                let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
                let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

                // Project aligned SRSF onto eigenfunctions
                for k in 0..ncomp {
                    let mut s = 0.0;
                    for j in 0..m {
                        let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
                        let centered = q_aligned - mean_srsf[j.min(mean_srsf.len() - 1)];
                        // eigenfunctions_q is (ncomp × (m+1)): row k, column j
                        s += centered * fpca.eigenfunctions_q[(k, j)];
                    }
                    sc[(i, k)] = s;
                }
            }
            sc
        }
        PcaMethod::Horizontal => {
            let fpca = result.horiz_fpca.as_ref()?;
            let ncomp = fpca.scores.ncols().min(result.coefficients.len());
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
                // Project warping function onto horizontal eigenfunctions
                let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
                let psi = crate::warping::gam_to_psi(&gam, h);
                for k in 0..ncomp {
                    let mut s = 0.0;
                    for j in 0..m {
                        let centered = psi[j] - fpca.mean_psi[j];
                        s += centered * fpca.eigenfunctions_psi[(k, j)];
                    }
                    sc[(i, k)] = s;
                }
            }
            sc
        }
        PcaMethod::Joint => {
            let fpca = result.joint_fpca.as_ref()?;
            let ncomp = fpca.scores.ncols().min(result.coefficients.len());
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
                let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
                let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
                let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

                // Joint scores via vertical component (ncomp × (m+1))
                for k in 0..ncomp {
                    let mut s = 0.0;
                    for j in 0..m {
                        let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
                        let centered = q_aligned - mean_srsf[j.min(mean_srsf.len() - 1)];
                        s += centered
                            * fpca.vert_component[(k, j.min(fpca.vert_component.ncols() - 1))];
                    }
                    sc[(i, k)] = s;
                }
            }
            sc
        }
    };

    // Apply linear model: y = alpha + sum(coef_k * score_k)
    let ncomp = result.coefficients.len();
    let mut preds = vec![0.0; n_new];
    for i in 0..n_new {
        preds[i] = result.alpha;
        for k in 0..ncomp.min(scores.ncols()) {
            preds[i] += result.coefficients[k] * scores[(i, k)];
        }
    }
    Some(preds)
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Split Conformal Classification (with refit)
// ═══════════════════════════════════════════════════════════════════════════

/// Split-conformal prediction sets for functional classifiers (LDA / QDA / kNN).
///
/// Splits data, refits the specified classifier on the proper-training subset,
/// computes non-conformity scores on calibration, then builds prediction sets
/// for test data.
///
/// # Arguments
/// * `classifier` — One of `"lda"`, `"qda"`, or `"knn"`
/// * `k_nn` — Number of neighbors (only used if `classifier == "knn"`)
/// * `score_type` — [`ClassificationScore::Lac`] or [`ClassificationScore::Aps`]
pub fn conformal_classif(
    data: &FdMatrix,
    y: &[usize],
    test_data: &FdMatrix,
    covariates_train: Option<&FdMatrix>,
    _covariates_test: Option<&FdMatrix>,
    ncomp: usize,
    classifier: &str,
    k_nn: usize,
    score_type: ClassificationScore,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalClassificationResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec_usize(y, &proper_idx);
    let proper_cov = covariates_train.map(|c| subsample_rows(c, &proper_idx));

    // Fit classifier on proper-training
    let fit: ClassifFit = match classifier {
        "lda" => fclassif_lda_fit(&proper_data, &proper_y, proper_cov.as_ref(), ncomp)?,
        "qda" => fclassif_qda_fit(&proper_data, &proper_y, proper_cov.as_ref(), ncomp)?,
        "knn" => fclassif_knn_fit(&proper_data, &proper_y, proper_cov.as_ref(), ncomp, k_nn)?,
        _ => return None,
    };

    // Get calibration probabilities
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_scores_mat = fit.project(&cal_data);
    let cal_probs = classif_predict_probs(&fit, &cal_scores_mat);
    let cal_true = subset_vec_usize(y, &cal_idx);
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Get test probabilities
    let test_scores_mat = fit.project(test_data);
    let test_probs = classif_predict_probs(&fit, &test_scores_mat);
    let test_pred_classes: Vec<usize> = test_probs.iter().map(|p| argmax(p)).collect();

    Some(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}

/// Split-conformal prediction sets for functional logistic regression.
///
/// Refits [`functional_logistic`] on the proper-training subset.
/// Binary classification → prediction sets of size 1 or 2.
pub fn conformal_logistic(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    ncomp: usize,
    max_iter: usize,
    tol: f64,
    score_type: ClassificationScore,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalClassificationResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);
    if proper_idx.len() < ncomp + 2 {
        return None;
    }

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec(y, &proper_idx);
    let proper_sc = scalar_train.map(|sc| subsample_rows(sc, &proper_idx));

    let refit = functional_logistic(
        &proper_data,
        &proper_y,
        proper_sc.as_ref(),
        ncomp,
        max_iter,
        tol,
    )?;

    // Calibration: get probabilities
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_sc = scalar_train.map(|sc| subsample_rows(sc, &cal_idx));
    let cal_scores_mat = project_scores(
        &cal_data,
        &refit.fpca.mean,
        &refit.fpca.rotation,
        refit.ncomp,
    );
    let cal_probs = logistic_probs_from_scores(&refit, &cal_scores_mat, cal_sc.as_ref());
    let cal_true: Vec<usize> = cal_idx.iter().map(|&i| y[i] as usize).collect();
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Test: get probabilities
    let test_scores_mat = project_scores(
        test_data,
        &refit.fpca.mean,
        &refit.fpca.rotation,
        refit.ncomp,
    );
    let test_probs = logistic_probs_from_scores(&refit, &test_scores_mat, scalar_test);
    let test_pred_classes: Vec<usize> = test_probs.iter().map(|p| argmax(p)).collect();

    Some(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}

/// Split-conformal prediction sets for elastic logistic regression.
///
/// Refits [`elastic_logistic`] on the proper-training subset.
pub fn conformal_elastic_logistic(
    data: &FdMatrix,
    y: &[i8],
    test_data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    score_type: ClassificationScore,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalClassificationResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)
        || y.len() != n
        || data.ncols() != test_data.ncols()
    {
        return None;
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec_i8(y, &proper_idx);

    let refit = elastic_logistic(&proper_data, &proper_y, argvals, 20, lambda, 50, 1e-4)?;

    // Calibration probabilities
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_probs = predict_elastic_logistic_probs(&refit, &cal_data, argvals);
    let cal_true: Vec<usize> = cal_idx
        .iter()
        .map(|&i| if y[i] == 1 { 1 } else { 0 })
        .collect();
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Test probabilities
    let test_probs = predict_elastic_logistic_probs(&refit, test_data, argvals);
    let test_pred_classes: Vec<usize> = test_probs.iter().map(|p| argmax(p)).collect();

    Some(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}

/// Helper: get binary class probabilities from functional logistic result.
fn logistic_probs_from_scores(
    fit: &crate::scalar_on_function::FunctionalLogisticResult,
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Vec<Vec<f64>> {
    let n = scores.nrows();
    let ncomp = fit.ncomp;
    (0..n)
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| scores[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> =
                scalar_covariates.map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            let mut eta = fit.coefficients[0];
            for k in 0..ncomp {
                eta += fit.coefficients[1 + k] * s[k];
            }
            if let Some(ref sc) = sc_row {
                for (j, &v) in sc.iter().enumerate() {
                    if j < fit.gamma.len() {
                        eta += fit.gamma[j] * v;
                    }
                }
            }
            let p1 = crate::scalar_on_function::sigmoid(eta);
            vec![1.0 - p1, p1]
        })
        .collect()
}

/// Helper: predict binary probabilities from elastic logistic result.
fn predict_elastic_logistic_probs(
    result: &crate::elastic_regression::ElasticLogisticResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<Vec<f64>> {
    let (n_new, m) = new_data.shape();
    let weights = crate::helpers::simpsons_weights(argvals);
    let q_new = crate::alignment::srsf_transform(new_data, argvals);

    (0..n_new)
        .map(|i| {
            let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
            let gam = crate::alignment::dp_alignment_core(&result.beta, &qi, argvals, 0.0);
            let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
            let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
            let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

            let mut eta = result.alpha;
            for j in 0..m {
                let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
                eta += q_aligned * result.beta[j] * weights[j];
            }
            let p1 = 1.0 / (1.0 + (-eta).exp());
            vec![1.0 - p1, p1]
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Generic Conformal via FpcPredictor
// ═══════════════════════════════════════════════════════════════════════════

/// Generic split-conformal prediction intervals for any [`FpcPredictor`] model.
///
/// Does **not** refit — uses the full model's predictions and calibrates on a
/// held-out portion of the training data. Coverage is approximately correct
/// (slightly optimistic vs. split conformal with refit).
pub fn conformal_generic_regression(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha) || y.len() != n {
        return None;
    }

    let (_proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    // Predict on calibration using full model
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_sc = scalar_train.map(|sc| subsample_rows(sc, &cal_idx));
    let cal_scores_mat = model.project(&cal_data);
    let ncomp = model.ncomp();

    let cal_preds: Vec<f64> = (0..cal_idx.len())
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| cal_scores_mat[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> = cal_sc
                .as_ref()
                .map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            model.predict_from_scores(&s, sc_row.as_deref())
        })
        .collect();

    let cal_residuals: Vec<f64> = cal_idx
        .iter()
        .enumerate()
        .map(|(i, &orig)| (y[orig] - cal_preds[i]).abs())
        .collect();

    // Predict on test
    let test_scores_mat = model.project(test_data);
    let test_preds: Vec<f64> = (0..test_data.nrows())
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| test_scores_mat[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> =
                scalar_test.map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            model.predict_from_scores(&s, sc_row.as_deref())
        })
        .collect();

    Some(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Generic split-conformal prediction sets for any [`FpcPredictor`] classification model.
///
/// Does **not** refit — uses the full model's predictions. For binary classification,
/// uses `predict_from_scores` which returns P(Y=1); for multiclass, returns class
/// label as f64 (so prediction sets may be less informative).
pub fn conformal_generic_classification(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[usize],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    score_type: ClassificationScore,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Option<ConformalClassificationResult> {
    let n = data.nrows();
    if !validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha) || y.len() != n {
        return None;
    }

    let n_classes = match model.task_type() {
        TaskType::BinaryClassification => 2,
        TaskType::MulticlassClassification(g) => g,
        TaskType::Regression => return None,
    };

    let (_proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);
    let ncomp = model.ncomp();

    // Calibration probabilities
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_sc = scalar_train.map(|sc| subsample_rows(sc, &cal_idx));
    let cal_scores_mat = model.project(&cal_data);
    let cal_probs: Vec<Vec<f64>> = (0..cal_idx.len())
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| cal_scores_mat[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> = cal_sc
                .as_ref()
                .map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            let pred = model.predict_from_scores(&s, sc_row.as_deref());
            if n_classes == 2 {
                vec![1.0 - pred, pred]
            } else {
                // For multiclass FpcPredictor, pred is the class label.
                // Build a one-hot-like probability (hard assignment).
                let c = pred.round() as usize;
                let mut probs = vec![0.0; n_classes];
                if c < n_classes {
                    probs[c] = 1.0;
                }
                probs
            }
        })
        .collect();

    let cal_true = subset_vec_usize(y, &cal_idx);
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Test probabilities
    let test_scores_mat = model.project(test_data);
    let test_probs: Vec<Vec<f64>> = (0..test_data.nrows())
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| test_scores_mat[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> =
                scalar_test.map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            let pred = model.predict_from_scores(&s, sc_row.as_deref());
            if n_classes == 2 {
                vec![1.0 - pred, pred]
            } else {
                let c = pred.round() as usize;
                let mut probs = vec![0.0; n_classes];
                if c < n_classes {
                    probs[c] = 1.0;
                }
                probs
            }
        })
        .collect();

    let test_pred_classes: Vec<usize> = test_probs.iter().map(|p| argmax(p)).collect();

    Some(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Cross-Conformal (CV+)
// ═══════════════════════════════════════════════════════════════════════════

/// Cross-conformal (CV+) prediction intervals for regression.
///
/// Uses K-fold CV: each fold produces out-of-fold predictions that serve
/// as calibration residuals, so no data is "wasted" on calibration.
///
/// The `fit_predict` closure takes `(train_data, train_y, train_sc, predict_data, predict_sc)`
/// and returns `Some((cal_preds, test_preds))` — predictions on held-out fold
/// and on test data.
pub fn cv_conformal_regression(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    fit_predict: impl Fn(
        &FdMatrix,
        &[f64],
        Option<&FdMatrix>,
        &FdMatrix,
        Option<&FdMatrix>,
    ) -> Option<(Vec<f64>, Vec<f64>)>,
    n_folds: usize,
    alpha: f64,
    seed: u64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    let n_test = test_data.nrows();
    if n < 4 || n_test == 0 || y.len() != n || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }
    let n_folds = n_folds.max(2).min(n);

    let folds = create_folds(n, n_folds, seed);
    let mut all_cal_residuals = vec![0.0; n];
    let mut test_preds_sum = vec![0.0; n_test];
    let mut n_models = 0usize;

    for fold in 0..n_folds {
        let (train_idx, test_idx) = fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let train_data = subset_rows(data, &train_idx);
        let train_y = subset_vec(y, &train_idx);
        let train_sc = scalar_train.map(|sc| subset_rows(sc, &train_idx));
        let cal_data = subset_rows(data, &test_idx);
        let cal_sc = scalar_train.map(|sc| subset_rows(sc, &test_idx));

        // Get calibration predictions (on held-out fold)
        let (cal_preds, _) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            &cal_data,
            cal_sc.as_ref(),
        )?;

        // Store calibration residuals at their original positions
        for (local_i, &orig_i) in test_idx.iter().enumerate() {
            if local_i < cal_preds.len() {
                all_cal_residuals[orig_i] = (y[orig_i] - cal_preds[local_i]).abs();
            }
        }

        // Get test predictions from this fold's model
        let (_, test_preds_fold) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            test_data,
            scalar_test,
        )?;

        for j in 0..n_test {
            if j < test_preds_fold.len() {
                test_preds_sum[j] += test_preds_fold[j];
            }
        }
        n_models += 1;
    }

    if n_models == 0 {
        return None;
    }

    // Average test predictions across folds
    let test_predictions: Vec<f64> = test_preds_sum
        .iter()
        .map(|&s| s / n_models as f64)
        .collect();

    Some(build_regression_result(
        all_cal_residuals,
        test_predictions,
        alpha,
        ConformalMethod::CrossConformal { n_folds },
    ))
}

/// Cross-conformal (CV+) prediction sets for classification.
///
/// The `fit_predict_probs` closure returns `(cal_probs, test_probs)` — vectors
/// of class probability vectors.
pub fn cv_conformal_classification(
    data: &FdMatrix,
    y: &[usize],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    fit_predict_probs: impl Fn(
        &FdMatrix,
        &[usize],
        Option<&FdMatrix>,
        &FdMatrix,
        Option<&FdMatrix>,
    ) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>,
    n_folds: usize,
    score_type: ClassificationScore,
    alpha: f64,
    seed: u64,
) -> Option<ConformalClassificationResult> {
    let n = data.nrows();
    let n_test = test_data.nrows();
    if n < 4 || n_test == 0 || y.len() != n || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }
    let n_classes = *y.iter().max()? + 1;
    let n_folds = n_folds.max(2).min(n);

    let folds = create_folds(n, n_folds, seed);
    let mut all_cal_scores = vec![0.0; n];
    let mut test_probs_sum: Vec<Vec<f64>> = vec![vec![0.0; n_classes]; n_test];
    let mut n_models = 0usize;

    for fold in 0..n_folds {
        let (train_idx, test_idx) = fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let train_data = subset_rows(data, &train_idx);
        let train_y = subset_vec_usize(y, &train_idx);
        let train_sc = scalar_train.map(|sc| subset_rows(sc, &train_idx));
        let cal_data = subset_rows(data, &test_idx);
        let cal_sc = scalar_train.map(|sc| subset_rows(sc, &test_idx));

        let (cal_probs, _) = fit_predict_probs(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            &cal_data,
            cal_sc.as_ref(),
        )?;

        // Calibration scores
        let cal_true = subset_vec_usize(y, &test_idx);
        let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);
        for (local_i, &orig_i) in test_idx.iter().enumerate() {
            if local_i < cal_scores.len() {
                all_cal_scores[orig_i] = cal_scores[local_i];
            }
        }

        // Test probabilities from this fold
        let (_, test_probs) = fit_predict_probs(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            test_data,
            scalar_test,
        )?;

        for j in 0..n_test.min(test_probs.len()) {
            for c in 0..n_classes.min(test_probs[j].len()) {
                test_probs_sum[j][c] += test_probs[j][c];
            }
        }
        n_models += 1;
    }

    if n_models == 0 {
        return None;
    }

    // Average test probabilities
    let test_probs_avg: Vec<Vec<f64>> = test_probs_sum
        .iter()
        .map(|probs| probs.iter().map(|&p| p / n_models as f64).collect())
        .collect();
    let test_pred_classes: Vec<usize> = test_probs_avg.iter().map(|p| argmax(p)).collect();

    Some(build_classification_result(
        all_cal_scores,
        &test_probs_avg,
        test_pred_classes,
        alpha,
        ConformalMethod::CrossConformal { n_folds },
        score_type,
    ))
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Jackknife+
// ═══════════════════════════════════════════════════════════════════════════

/// Jackknife+ prediction intervals for regression.
///
/// LOO-based conformal: for each i = 1..n, fits model on all data except i,
/// computes LOO residual and test predictions. Constructs intervals from the
/// distribution of signed residuals.
///
/// Requires n refits, so this is the most sample-efficient but most expensive method.
///
/// The `fit_predict` closure takes `(train_data, train_y, train_sc, predict_data, predict_sc)`
/// and returns `Some((loo_pred, test_preds))`.
pub fn jackknife_plus_regression(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    fit_predict: impl Fn(
        &FdMatrix,
        &[f64],
        Option<&FdMatrix>,
        &FdMatrix,
        Option<&FdMatrix>,
    ) -> Option<(Vec<f64>, Vec<f64>)>,
    alpha: f64,
) -> Option<ConformalRegressionResult> {
    let n = data.nrows();
    let n_test = test_data.nrows();
    if n < 4 || n_test == 0 || y.len() != n || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }

    let mut loo_residuals = vec![0.0; n];
    // For each test point, store predictions from all n LOO models
    let mut test_preds_all = vec![vec![0.0; n]; n_test];

    for i in 0..n {
        let train_idx: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        let loo_idx = vec![i];

        let train_data = subset_rows(data, &train_idx);
        let train_y = subset_vec(y, &train_idx);
        let train_sc = scalar_train.map(|sc| subset_rows(sc, &train_idx));
        let loo_data = subset_rows(data, &loo_idx);
        let loo_sc = scalar_train.map(|sc| subset_rows(sc, &loo_idx));

        // Predict on LOO observation
        let (loo_pred, _) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            &loo_data,
            loo_sc.as_ref(),
        )?;

        loo_residuals[i] = (y[i] - loo_pred[0]).abs();

        // Predict on test data
        let (_, test_preds) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            test_data,
            scalar_test,
        )?;

        for j in 0..n_test.min(test_preds.len()) {
            test_preds_all[j][i] = test_preds[j];
        }
    }

    // For each test point: construct interval from the distribution of
    // ŷ_{-i}(x_test) ± R_i across all i
    let q_lo = alpha / 2.0;
    let q_hi = 1.0 - alpha / 2.0;

    let mut predictions = vec![0.0; n_test];
    let mut lower = vec![0.0; n_test];
    let mut upper = vec![0.0; n_test];

    for j in 0..n_test {
        // Mean prediction
        predictions[j] = test_preds_all[j].iter().sum::<f64>() / n as f64;

        // Lower bounds: ŷ_{-i}(x_test) - R_i
        let mut lower_vals: Vec<f64> = (0..n)
            .map(|i| test_preds_all[j][i] - loo_residuals[i])
            .collect();
        lower_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lower[j] = quantile_sorted(&lower_vals, q_lo);

        // Upper bounds: ŷ_{-i}(x_test) + R_i
        let mut upper_vals: Vec<f64> = (0..n)
            .map(|i| test_preds_all[j][i] + loo_residuals[i])
            .collect();
        upper_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        upper[j] = quantile_sorted(&upper_vals, q_hi);
    }

    // Coverage on LOO (using mean prediction)
    let residual_quantile = {
        let mut r = loo_residuals.clone();
        conformal_quantile(&mut r, alpha)
    };
    let coverage = empirical_coverage(&loo_residuals, residual_quantile);

    Some(ConformalRegressionResult {
        predictions,
        lower,
        upper,
        residual_quantile,
        coverage,
        calibration_scores: loo_residuals,
        method: ConformalMethod::JackknifePlus,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn make_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>, FdMatrix) {
        let mut rng = StdRng::seed_from_u64(seed);
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let a = rng.gen::<f64>() * 2.0 - 1.0;
            let b = rng.gen::<f64>() * 2.0 - 1.0;
            for j in 0..m {
                data[(i, j)] = a * (2.0 * PI * argvals[j]).sin()
                    + b * (4.0 * PI * argvals[j]).cos()
                    + 0.1 * rng.gen::<f64>();
            }
            y[i] = 2.0 * a + 3.0 * b + 0.5 * rng.gen::<f64>();
        }
        let n_test = 5;
        let mut test_data = FdMatrix::zeros(n_test, m);
        for i in 0..n_test {
            let a = rng.gen::<f64>() * 2.0 - 1.0;
            let b = rng.gen::<f64>() * 2.0 - 1.0;
            for j in 0..m {
                test_data[(i, j)] = a * (2.0 * PI * argvals[j]).sin()
                    + b * (4.0 * PI * argvals[j]).cos()
                    + 0.1 * rng.gen::<f64>();
            }
        }
        (data, y, test_data)
    }

    fn make_classif_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<usize>, FdMatrix) {
        let mut rng = StdRng::seed_from_u64(seed);
        let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0usize; n];
        for i in 0..n {
            let class = if i < n / 2 { 0 } else { 1 };
            y[i] = class;
            let offset = if class == 0 { -1.0 } else { 1.0 };
            for j in 0..m {
                data[(i, j)] = offset * (2.0 * PI * argvals[j]).sin() + 0.3 * rng.gen::<f64>();
            }
        }
        let n_test = 4;
        let mut test_data = FdMatrix::zeros(n_test, m);
        for i in 0..n_test {
            let offset = if i < 2 { -1.0 } else { 1.0 };
            for j in 0..m {
                test_data[(i, j)] = offset * (2.0 * PI * argvals[j]).sin() + 0.3 * rng.gen::<f64>();
            }
        }
        (data, y, test_data)
    }

    // ── Core helper tests ────────────────────────────────────────────────

    #[test]
    fn test_conformal_split_sizes() {
        let (proper, cal) = conformal_split(100, 0.2, 42);
        assert_eq!(proper.len() + cal.len(), 100);
        assert!((cal.len() as f64 - 20.0).abs() <= 2.0);
    }

    #[test]
    fn test_conformal_quantile_monotonic() {
        let mut scores: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let q1 = conformal_quantile(&mut scores.clone(), 0.1);
        let q2 = conformal_quantile(&mut scores, 0.2);
        assert!(
            q1 >= q2,
            "Lower alpha should give wider intervals (higher quantile)"
        );
    }

    #[test]
    fn test_lac_and_aps_scores() {
        let probs = vec![0.7, 0.2, 0.1];
        assert!((lac_score(&probs, 0) - 0.3).abs() < 1e-10);
        assert!((lac_score(&probs, 1) - 0.8).abs() < 1e-10);

        // APS: for true class 0, sorted order is [0, 1, 2], cumulative at class 0 = 0.7
        let aps0 = aps_score(&probs, 0);
        assert!((aps0 - 0.7).abs() < 1e-10);

        // APS: for true class 2, sorted order is [0, 1, 2], cumulative at class 2 = 1.0
        let aps2 = aps_score(&probs, 2);
        assert!((aps2 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_prediction_sets_lac() {
        let probs = vec![0.7, 0.2, 0.1];
        // quantile = 0.5: include class k if 1 - P(k) ≤ 0.5 → P(k) ≥ 0.5 → only class 0
        let set = lac_prediction_set(&probs, 0.5);
        assert_eq!(set, vec![0]);

        // quantile = 0.9: include class k if 1 - P(k) ≤ 0.9 → P(k) ≥ 0.1 → all classes
        let set = lac_prediction_set(&probs, 0.9);
        assert_eq!(set, vec![0, 1, 2]);
    }

    #[test]
    fn test_prediction_sets_aps() {
        let probs = vec![0.7, 0.2, 0.1];
        // quantile = 0.5 → threshold = 0.5: include classes until cumulative ≥ 0.5
        // Sorted: [0(0.7), 1(0.2), 2(0.1)]. Cumulative: 0.7 ≥ 0.5 → {0}
        let set = aps_prediction_set(&probs, 0.5);
        assert_eq!(set, vec![0]);

        // quantile = 0.15 → threshold = 0.85: need cumulative ≥ 0.85 → {0, 1}
        let set = aps_prediction_set(&probs, 0.15);
        assert_eq!(set, vec![0, 1]);
    }

    // ── Regression integration tests ─────────────────────────────────────

    #[test]
    fn test_conformal_fregre_lm_basic() {
        let (data, y, test_data) = make_test_data(40, 20, 42);
        let result = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.1, 42);
        let r = result.unwrap();
        assert_eq!(r.predictions.len(), 5);
        assert_eq!(r.lower.len(), 5);
        assert_eq!(r.upper.len(), 5);
        // Intervals should have positive width
        for i in 0..5 {
            assert!(r.upper[i] > r.lower[i]);
        }
        // Coverage on calibration set should be reasonable
        assert!(r.coverage >= 0.5);
    }

    #[test]
    fn test_conformal_fregre_np_basic() {
        let (data, y, test_data) = make_test_data(30, 15, 123);
        let argvals: Vec<f64> = (0..15).map(|j| j as f64 / 14.0).collect();
        let result = conformal_fregre_np(
            &data, &y, &test_data, &argvals, None, None, 1.0, 1.0, 0.3, 0.1, 123,
        );
        let r = result.unwrap();
        assert_eq!(r.predictions.len(), 5);
        for i in 0..5 {
            assert!(r.upper[i] > r.lower[i]);
        }
    }

    // ── Classification integration tests ─────────────────────────────────

    #[test]
    fn test_conformal_classif_lda() {
        let (data, y, test_data) = make_classif_data(40, 20, 42);
        let result = conformal_classif(
            &data,
            &y,
            &test_data,
            None,
            None,
            3,
            "lda",
            5,
            ClassificationScore::Lac,
            0.3,
            0.1,
            42,
        );
        let r = result.unwrap();
        assert_eq!(r.prediction_sets.len(), 4);
        // All prediction sets should be non-empty
        for set in &r.prediction_sets {
            assert!(!set.is_empty());
        }
        assert!(r.average_set_size >= 1.0);
    }

    #[test]
    fn test_conformal_classif_aps() {
        let (data, y, test_data) = make_classif_data(40, 20, 42);
        let result = conformal_classif(
            &data,
            &y,
            &test_data,
            None,
            None,
            3,
            "lda",
            5,
            ClassificationScore::Aps,
            0.3,
            0.1,
            42,
        );
        let r = result.unwrap();
        assert_eq!(r.prediction_sets.len(), 4);
        for set in &r.prediction_sets {
            assert!(!set.is_empty());
        }
    }

    #[test]
    fn test_conformal_logistic_basic() {
        let (data, y_usize, test_data) = make_classif_data(40, 20, 42);
        let y: Vec<f64> = y_usize.iter().map(|&c| c as f64).collect();
        let result = conformal_logistic(
            &data,
            &y,
            &test_data,
            None,
            None,
            3,
            100,
            1e-4,
            ClassificationScore::Lac,
            0.3,
            0.1,
            42,
        );
        let r = result.unwrap();
        assert_eq!(r.prediction_sets.len(), 4);
        for set in &r.prediction_sets {
            assert!(!set.is_empty());
            // Binary: set size should be 1 or 2
            assert!(set.len() <= 2);
        }
    }

    // ── Generic conformal tests ──────────────────────────────────────────

    #[test]
    fn test_conformal_generic_regression() {
        let (data, y, test_data) = make_test_data(40, 20, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let result =
            conformal_generic_regression(&fit, &data, &y, &test_data, None, None, 0.3, 0.1, 42);
        let r = result.unwrap();
        assert_eq!(r.predictions.len(), 5);
        for i in 0..5 {
            assert!(r.upper[i] > r.lower[i]);
        }
    }

    #[test]
    fn test_conformal_generic_classification() {
        let (data, y, test_data) = make_classif_data(40, 20, 42);
        let fit = fclassif_lda_fit(&data, &y, None, 3).unwrap();
        let result = conformal_generic_classification(
            &fit,
            &data,
            &y,
            &test_data,
            None,
            None,
            ClassificationScore::Lac,
            0.3,
            0.1,
            42,
        );
        let r = result.unwrap();
        assert_eq!(r.prediction_sets.len(), 4);
        for set in &r.prediction_sets {
            assert!(!set.is_empty());
        }
    }

    // ── CV+ conformal tests ──────────────────────────────────────────────

    #[test]
    fn test_cv_conformal_regression() {
        let (data, y, test_data) = make_test_data(40, 20, 42);
        let result = cv_conformal_regression(
            &data,
            &y,
            &test_data,
            None,
            None,
            |train_d, train_y, _train_sc, pred_d, _pred_sc| {
                let fit = fregre_lm(train_d, train_y, None, 3)?;
                let cal = predict_fregre_lm(&fit, pred_d, None);
                let test = predict_fregre_lm(&fit, pred_d, None);
                Some((cal, test))
            },
            5,
            0.1,
            42,
        );
        let r = result.unwrap();
        assert_eq!(r.predictions.len(), test_data.nrows());
        for i in 0..r.predictions.len() {
            assert!(r.upper[i] > r.lower[i]);
        }
    }

    // ── Validation tests ─────────────────────────────────────────────────

    #[test]
    fn test_invalid_inputs() {
        let data = FdMatrix::zeros(2, 5);
        let y = vec![1.0, 2.0];
        let test = FdMatrix::zeros(1, 5);
        // Too few observations
        assert!(conformal_fregre_lm(&data, &y, &test, None, None, 1, 0.3, 0.1, 42).is_none());

        // Invalid alpha
        let (data, y, test) = make_test_data(20, 10, 42);
        assert!(conformal_fregre_lm(&data, &y, &test, None, None, 2, 0.3, 0.0, 42).is_none());
        assert!(conformal_fregre_lm(&data, &y, &test, None, None, 2, 0.3, 1.0, 42).is_none());
    }

    #[test]
    fn test_alpha_affects_interval_width() {
        let (data, y, test_data) = make_test_data(40, 20, 42);
        let r1 = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.1, 42).unwrap();
        let r2 = conformal_fregre_lm(&data, &y, &test_data, None, None, 3, 0.3, 0.3, 42).unwrap();
        // Wider alpha → narrower intervals (lower quantile)
        assert!(r1.residual_quantile >= r2.residual_quantile);
    }
}
