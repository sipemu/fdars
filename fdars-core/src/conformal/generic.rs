//! Generic split-conformal prediction via [`FpcPredictor`] trait.

use crate::error::FdarError;
use crate::explain::subsample_rows;
use crate::explain_generic::{FpcPredictor, TaskType};
use crate::matrix::FdMatrix;

use super::{
    argmax, build_classification_result, build_regression_result, compute_cal_scores,
    conformal_split, subset_vec_usize, validate_split_inputs, ClassificationScore,
    ConformalClassificationResult, ConformalMethod, ConformalRegressionResult,
};

/// Generic split-conformal prediction intervals for any [`FpcPredictor`] model.
///
/// Does **not** refit — uses the full model's predictions and calibrates on a
/// held-out portion of the training data.
///
/// # Calibration indices
///
/// When `calibration_indices` is `Some(indices)`, only those rows of `data`/`y`
/// are used for calibration. The caller is responsible for ensuring the model was
/// **not** trained on these rows (e.g., they are a held-out validation set). This
/// avoids the data-leakage problem and preserves the finite-sample coverage
/// guarantee.
///
/// When `calibration_indices` is `None`, a random split is performed using
/// `cal_fraction` and `seed`.
///
/// **Warning (data leakage)**: When `calibration_indices` is `None`, the model
/// was typically trained on all data including the calibration set, so calibration
/// residuals are in-sample and systematically too small. This breaks the
/// distribution-free coverage guarantee and produces intervals that are too
/// narrow (optimistic). For valid coverage, either supply held-out
/// `calibration_indices` or use the refit-based / CV+ variants instead.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or any index in `calibration_indices` is out of bounds.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// or if `calibration_indices` contains fewer than 2 elements.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn conformal_generic_regression(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    calibration_indices: Option<&[usize]>,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Result<ConformalRegressionResult, FdarError> {
    let n = data.nrows();
    validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)?;
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }

    let cal_idx = resolve_calibration_indices(calibration_indices, n, cal_fraction, seed)?;

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

    Ok(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Generic split-conformal prediction sets for any [`FpcPredictor`] classification model.
///
/// Works with **binary classification** models only. For binary models,
/// `predict_from_scores` returns P(Y=1), which is converted to proper
/// probabilities `[1-p, p]` for conformal scoring.
///
/// # Calibration indices
///
/// When `calibration_indices` is `Some(indices)`, only those rows of `data`/`y`
/// are used for calibration. The caller is responsible for ensuring the model was
/// **not** trained on these rows (e.g., they are a held-out validation set). This
/// avoids the data-leakage problem and preserves the finite-sample coverage
/// guarantee.
///
/// When `calibration_indices` is `None`, a random split is performed using
/// `cal_fraction` and `seed`.
///
/// **Warning (data leakage)**: When `calibration_indices` is `None`, the model
/// was typically trained on all data including the calibration set, so calibration
/// scores are in-sample and systematically too small. This breaks the
/// distribution-free coverage guarantee and produces prediction sets that are
/// too small (optimistic). For valid coverage, either supply held-out
/// `calibration_indices` or use the refit-based / CV+ variants instead.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or any index in `calibration_indices` is out of bounds.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// the model's task type is `Regression`, the model's task type is
/// `MulticlassClassification` (not supported — `predict_from_scores` returns a
/// class label, not probabilities, producing degenerate one-hot conformal sets),
/// or `calibration_indices` contains fewer than 2 elements.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn conformal_generic_classification(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    y: &[usize],
    test_data: &FdMatrix,
    scalar_train: Option<&FdMatrix>,
    scalar_test: Option<&FdMatrix>,
    score_type: ClassificationScore,
    calibration_indices: Option<&[usize]>,
    cal_fraction: f64,
    alpha: f64,
    seed: u64,
) -> Result<ConformalClassificationResult, FdarError> {
    let n = data.nrows();
    validate_split_inputs(n, test_data.nrows(), cal_fraction, alpha)?;
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }

    match model.task_type() {
        TaskType::BinaryClassification => {}
        TaskType::MulticlassClassification(g) => {
            return Err(FdarError::InvalidParameter {
                parameter: "model",
                message: format!(
                    "conformal_generic_classification does not support multiclass models \
                     ({g} classes): FpcPredictor::predict_from_scores returns a class label, \
                     not probabilities, which produces degenerate one-hot conformal sets. \
                     Use cv_conformal_classification with a closure that returns proper \
                     probabilities instead."
                ),
            })
        }
        TaskType::Regression => {
            return Err(FdarError::InvalidParameter {
                parameter: "model",
                message: "expected a classification model, got regression".to_string(),
            })
        }
    };

    let cal_idx = resolve_calibration_indices(calibration_indices, n, cal_fraction, seed)?;
    let ncomp = model.ncomp();

    // Calibration probabilities (binary: [1-p, p])
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
            vec![1.0 - pred, pred]
        })
        .collect();

    let cal_true = subset_vec_usize(y, &cal_idx);
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Test probabilities (binary: [1-p, p])
    let test_scores_mat = model.project(test_data);
    let test_probs: Vec<Vec<f64>> = (0..test_data.nrows())
        .map(|i| {
            let s: Vec<f64> = (0..ncomp).map(|k| test_scores_mat[(i, k)]).collect();
            let sc_row: Option<Vec<f64>> =
                scalar_test.map(|sc| (0..sc.ncols()).map(|j| sc[(i, j)]).collect());
            let pred = model.predict_from_scores(&s, sc_row.as_deref());
            vec![1.0 - pred, pred]
        })
        .collect();

    let test_pred_classes: Vec<usize> = test_probs.iter().map(|p| argmax(p)).collect();

    Ok(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// Resolve calibration indices from explicit indices or a random split.
///
/// Returns the calibration index vector. Validates bounds and minimum size.
fn resolve_calibration_indices(
    calibration_indices: Option<&[usize]>,
    n: usize,
    cal_fraction: f64,
    seed: u64,
) -> Result<Vec<usize>, FdarError> {
    match calibration_indices {
        Some(indices) => {
            if indices.len() < 2 {
                return Err(FdarError::InvalidParameter {
                    parameter: "calibration_indices",
                    message: format!(
                        "need at least 2 calibration observations, got {}",
                        indices.len()
                    ),
                });
            }
            for (pos, &idx) in indices.iter().enumerate() {
                if idx >= n {
                    return Err(FdarError::InvalidDimension {
                        parameter: "calibration_indices",
                        expected: format!("indices in 0..{n}"),
                        actual: format!("index {idx} at position {pos}"),
                    });
                }
            }
            Ok(indices.to_vec())
        }
        None => {
            let (_proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);
            Ok(cal_idx)
        }
    }
}
