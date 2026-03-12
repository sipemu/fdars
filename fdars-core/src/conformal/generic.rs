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
/// **Warning**: The model was trained on all data including the calibration set,
/// so calibration residuals are in-sample and systematically too small. This
/// breaks the distribution-free coverage guarantee and produces intervals that
/// are too narrow (optimistic). For valid coverage, use the refit-based or CV+
/// variants instead. This function is provided as a fast heuristic only.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, or `y` length differs from the number of rows in `data`.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1).
#[must_use = "expensive computation whose result should not be discarded"]
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

    Ok(build_regression_result(
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
///
/// **Warning**: Same data leakage caveat as [`conformal_generic_regression`] —
/// the model was trained on all data including the calibration set. Coverage
/// guarantee is broken. Use refit-based variants for valid coverage.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, or `y` length differs from the number of rows in `data`.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// or the model's task type is `Regression` instead of a classification type.
#[must_use = "expensive computation whose result should not be discarded"]
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

    let n_classes = match model.task_type() {
        TaskType::BinaryClassification => 2,
        TaskType::MulticlassClassification(g) => g,
        TaskType::Regression => {
            return Err(FdarError::InvalidParameter {
                parameter: "model",
                message: "expected a classification model, got regression".to_string(),
            })
        }
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

    Ok(build_classification_result(
        cal_scores,
        &test_probs,
        test_pred_classes,
        alpha,
        ConformalMethod::Split,
        score_type,
    ))
}
