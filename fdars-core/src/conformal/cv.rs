//! Cross-conformal (CV+) and jackknife+ prediction methods.

use crate::cv::{create_folds, fold_indices, subset_rows, subset_vec};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    argmax, build_classification_result, build_regression_result, compute_cal_scores,
    conformal_quantile, empirical_coverage, subset_vec_usize, vstack, vstack_opt,
    ClassificationScore, ConformalClassificationResult, ConformalMethod, ConformalRegressionResult,
};

/// Cross-conformal (CV+) prediction intervals for regression.
///
/// Uses K-fold CV: each fold produces out-of-fold predictions that serve
/// as calibration residuals, so no data is "wasted" on calibration.
///
/// The `fit_predict` closure takes `(train_data, train_y, train_sc, predict_data, predict_sc)`,
/// fits a model on `train_data`/`train_y`, and returns `Some((preds, _))` — predictions on
/// `predict_data`. Only the first element of the tuple is used; the second is ignored.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, or `y` length differs from the number of rows in `data`.
/// Returns [`FdarError::InvalidParameter`] if `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the `fit_predict` closure returns `None`
/// on any fold, or no valid folds are produced.
#[must_use = "expensive computation whose result should not be discarded"]
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
) -> Result<ConformalRegressionResult, FdarError> {
    let n = data.nrows();
    let n_test = test_data.nrows();
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
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
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

        // Single call: predict on combined [cal_data; test_data] to avoid double model fit
        let n_cal_fold = cal_data.nrows();
        let combined = vstack(&cal_data, test_data);
        let combined_sc = vstack_opt(cal_sc.as_ref(), scalar_test);
        let (all_preds, _) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            &combined,
            combined_sc.as_ref(),
        )
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "cv_conformal_regression",
            detail: format!("fit_predict closure returned None on fold {fold}"),
        })?;

        // Split predictions: first n_cal_fold are calibration, rest are test
        let cal_preds = &all_preds[..n_cal_fold];
        let test_preds_fold = &all_preds[n_cal_fold..];

        // Store calibration residuals at their original positions
        for (local_i, &orig_i) in test_idx.iter().enumerate() {
            if local_i < cal_preds.len() {
                all_cal_residuals[orig_i] = (y[orig_i] - cal_preds[local_i]).abs();
            }
        }

        for j in 0..n_test {
            if j < test_preds_fold.len() {
                test_preds_sum[j] += test_preds_fold[j];
            }
        }
        n_models += 1;
    }

    if n_models == 0 {
        return Err(FdarError::ComputationFailed {
            operation: "cv_conformal_regression",
            detail: "no valid folds produced".to_string(),
        });
    }

    // Average test predictions across folds
    let test_predictions: Vec<f64> = test_preds_sum
        .iter()
        .map(|&s| s / n_models as f64)
        .collect();

    Ok(build_regression_result(
        all_cal_residuals,
        test_predictions,
        alpha,
        ConformalMethod::CrossConformal { n_folds },
    ))
}

/// Run one fold of cross-conformal classification.
///
/// Fits the model on train indices, predicts on calibration + test combined,
/// and returns `(cal_scores_with_indices, test_probs)`.
fn conformal_classif_fold(
    data: &FdMatrix,
    y: &[usize],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    train_idx: &[usize],
    test_idx: &[usize],
    fold: usize,
    score_type: ClassificationScore,
    fit_predict_probs: &dyn Fn(
        &FdMatrix,
        &[usize],
        Option<&FdMatrix>,
        &FdMatrix,
        Option<&FdMatrix>,
    ) -> Option<(Vec<Vec<f64>>, Vec<Vec<f64>>)>,
) -> Result<(Vec<(usize, f64)>, Vec<Vec<f64>>), FdarError> {
    let train_data = subset_rows(data, train_idx);
    let train_y = subset_vec_usize(y, train_idx);
    let train_sc = scalar_train.map(|sc| subset_rows(sc, train_idx));
    let cal_data = subset_rows(data, test_idx);
    let cal_sc = scalar_train.map(|sc| subset_rows(sc, test_idx));

    // Single call: predict on combined [cal_data; test_data] to avoid double model fit
    let n_cal_fold = cal_data.nrows();
    let combined = vstack(&cal_data, test_data);
    let combined_sc = vstack_opt(cal_sc.as_ref(), scalar_test);
    let (all_probs, _) = fit_predict_probs(
        &train_data,
        &train_y,
        train_sc.as_ref(),
        &combined,
        combined_sc.as_ref(),
    )
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "cv_conformal_classification",
        detail: format!("fit_predict_probs closure returned None on fold {fold}"),
    })?;

    // Split predictions: first n_cal_fold are calibration, rest are test
    let cal_probs: Vec<Vec<f64>> = all_probs[..n_cal_fold].to_vec();
    let test_probs: Vec<Vec<f64>> = all_probs[n_cal_fold..].to_vec();

    // Calibration scores paired with original indices
    let cal_true = subset_vec_usize(y, test_idx);
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);
    let indexed_scores: Vec<(usize, f64)> = test_idx
        .iter()
        .enumerate()
        .filter(|&(local_i, _)| local_i < cal_scores.len())
        .map(|(local_i, &orig_i)| (orig_i, cal_scores[local_i]))
        .collect();

    Ok((indexed_scores, test_probs))
}

/// Aggregate per-fold conformal classification results into a final prediction.
///
/// Averages test probabilities across folds, determines predicted classes, and
/// builds the final [`ConformalClassificationResult`].
fn aggregate_conformal_results(
    all_cal_scores: Vec<f64>,
    test_probs_sum: &[Vec<f64>],
    n_models: usize,
    n_folds: usize,
    alpha: f64,
    score_type: ClassificationScore,
) -> Result<ConformalClassificationResult, FdarError> {
    if n_models == 0 {
        return Err(FdarError::ComputationFailed {
            operation: "cv_conformal_classification",
            detail: "no valid folds produced".to_string(),
        });
    }

    let test_probs_avg: Vec<Vec<f64>> = test_probs_sum
        .iter()
        .map(|probs| probs.iter().map(|&p| p / n_models as f64).collect())
        .collect();
    let test_pred_classes: Vec<usize> = test_probs_avg.iter().map(|p| argmax(p)).collect();

    Ok(build_classification_result(
        all_cal_scores,
        &test_probs_avg,
        test_pred_classes,
        alpha,
        ConformalMethod::CrossConformal { n_folds },
        score_type,
    ))
}

/// Cross-conformal (CV+) prediction sets for classification.
///
/// The `fit_predict_probs` closure takes `(train_data, train_y, train_sc, predict_data, predict_sc)`,
/// fits on `train_data`/`train_y`, and returns `Some((probs, _))` — probability vectors on
/// `predict_data`. Only the first element of the tuple is used.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `y` is empty (no classes can be determined).
/// Returns [`FdarError::InvalidParameter`] if `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the `fit_predict_probs` closure returns
/// `None` on any fold, or no valid folds are produced.
#[must_use = "expensive computation whose result should not be discarded"]
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
) -> Result<ConformalClassificationResult, FdarError> {
    let n = data.nrows();
    let n_test = test_data.nrows();
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
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
    }
    let n_classes = y
        .iter()
        .copied()
        .max()
        .ok_or_else(|| FdarError::InvalidDimension {
            parameter: "y",
            expected: "non-empty label vector".to_string(),
            actual: "empty".to_string(),
        })?
        + 1;
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

        let (indexed_scores, test_probs) = conformal_classif_fold(
            data,
            y,
            test_data,
            scalar_train,
            scalar_test,
            &train_idx,
            &test_idx,
            fold,
            score_type,
            &fit_predict_probs,
        )?;

        for (orig_i, score) in indexed_scores {
            all_cal_scores[orig_i] = score;
        }
        for j in 0..n_test.min(test_probs.len()) {
            for c in 0..n_classes.min(test_probs[j].len()) {
                test_probs_sum[j][c] += test_probs[j][c];
            }
        }
        n_models += 1;
    }

    aggregate_conformal_results(
        all_cal_scores,
        &test_probs_sum,
        n_models,
        n_folds,
        alpha,
        score_type,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Jackknife+
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
/// and returns `Some((predictions, _))` — predictions on `predict_data`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, or `y` length differs from the number of rows in `data`.
/// Returns [`FdarError::InvalidParameter`] if `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the `fit_predict` closure returns `None`
/// on any leave-one-out iteration.
#[must_use = "expensive computation whose result should not be discarded"]
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
) -> Result<ConformalRegressionResult, FdarError> {
    let n = data.nrows();
    let n_test = test_data.nrows();
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
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {alpha}"),
        });
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
        )
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "jackknife_plus_regression",
            detail: format!("fit_predict closure returned None on LOO iteration {i}"),
        })?;

        loo_residuals[i] = (y[i] - loo_pred[0]).abs();

        // Predict on test data
        let (test_preds, _) = fit_predict(
            &train_data,
            &train_y,
            train_sc.as_ref(),
            test_data,
            scalar_test,
        )
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "jackknife_plus_regression",
            detail: format!(
                "fit_predict closure returned None on test prediction for LOO iteration {i}"
            ),
        })?;

        for j in 0..n_test.min(test_preds.len()) {
            test_preds_all[j][i] = test_preds[j];
        }
    }

    // For each test point: construct interval from the distribution of
    // y_{-i}(x_test) +/- R_i across all i
    let q_lo = alpha / 2.0;
    let q_hi = 1.0 - alpha / 2.0;

    let mut predictions = vec![0.0; n_test];
    let mut lower = vec![0.0; n_test];
    let mut upper = vec![0.0; n_test];

    for j in 0..n_test {
        // Mean prediction
        predictions[j] = test_preds_all[j].iter().sum::<f64>() / n as f64;

        // Lower bounds: y_{-i}(x_test) - R_i
        let mut lower_vals: Vec<f64> = (0..n)
            .map(|i| test_preds_all[j][i] - loo_residuals[i])
            .collect();
        crate::helpers::sort_nan_safe(&mut lower_vals);
        // Lower bound: floor((n+1)*q_lo) as rank (Barber et al. 2021, Corollary 2)
        let lo_k = ((n + 1) as f64 * q_lo).floor() as usize;
        if lo_k == 0 {
            lower[j] = f64::NEG_INFINITY;
        } else {
            lower[j] = lower_vals[(lo_k - 1).min(n.saturating_sub(1))];
        }

        // Upper bounds: y_{-i}(x_test) + R_i
        let mut upper_vals: Vec<f64> = (0..n)
            .map(|i| test_preds_all[j][i] + loo_residuals[i])
            .collect();
        crate::helpers::sort_nan_safe(&mut upper_vals);
        let hi_k = ((n + 1) as f64 * q_hi).ceil() as usize;
        let hi_idx = if hi_k > n {
            n.saturating_sub(1)
        } else {
            (hi_k - 1).min(n.saturating_sub(1))
        };
        upper[j] = upper_vals[hi_idx];
    }

    // Coverage on LOO (using mean prediction)
    let residual_quantile = {
        let mut r = loo_residuals.clone();
        conformal_quantile(&mut r, alpha)
    };
    let coverage = empirical_coverage(&loo_residuals, residual_quantile);

    Ok(ConformalRegressionResult {
        predictions,
        lower,
        upper,
        residual_quantile,
        coverage,
        calibration_scores: loo_residuals,
        method: ConformalMethod::JackknifePlus,
    })
}
