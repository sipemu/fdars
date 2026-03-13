//! Split-conformal prediction sets for classification models.

use crate::classification::{
    classif_predict_probs, fclassif_knn_fit, fclassif_lda_fit, fclassif_qda_fit, ClassifFit,
};
use crate::cv::subset_vec;
use crate::elastic_regression::elastic_logistic;
use crate::error::FdarError;
use crate::explain::{project_scores, subsample_rows};
use crate::explain_generic::FpcPredictor;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::functional_logistic;

use super::{
    argmax, build_classification_result, compute_cal_scores, conformal_split, subset_vec_i8,
    subset_vec_usize, validate_split_inputs, ClassificationScore, ConformalClassificationResult,
    ConformalMethod,
};

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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// or `classifier` is not one of `"lda"`, `"qda"`, or `"knn"`.
/// Returns [`FdarError::ComputationFailed`] if the classifier fitting fails.
#[must_use = "expensive computation whose result should not be discarded"]
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
    if data.ncols() != test_data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: format!("{} columns", data.ncols()),
            actual: format!("{} columns", test_data.ncols()),
        });
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
        _ => {
            return Err(FdarError::InvalidParameter {
                parameter: "classifier",
                message: format!(
                    "unknown classifier '{classifier}', expected 'lda', 'qda', or 'knn'"
                ),
            })
        }
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

    Ok(build_classification_result(
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
/// Binary classification -> prediction sets of size 1 or 2.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// or the proper training set is too small for the requested `ncomp`.
/// Returns [`FdarError::ComputationFailed`] if the `functional_logistic` fitting fails.
#[must_use = "expensive computation whose result should not be discarded"]
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
    if data.ncols() != test_data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: format!("{} columns", data.ncols()),
            actual: format!("{} columns", test_data.ncols()),
        });
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);
    if proper_idx.len() < ncomp + 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!(
                "proper training set size {} too small for ncomp={}",
                proper_idx.len(),
                ncomp
            ),
        });
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

    Ok(build_classification_result(
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the `elastic_logistic` fitting fails.
#[must_use = "expensive computation whose result should not be discarded"]
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
    if data.ncols() != test_data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: format!("{} columns", data.ncols()),
            actual: format!("{} columns", test_data.ncols()),
        });
    }

    let (proper_idx, cal_idx) = conformal_split(n, cal_fraction, seed);

    let proper_data = subsample_rows(data, &proper_idx);
    let proper_y = subset_vec_i8(y, &proper_idx);

    let refit = elastic_logistic(&proper_data, &proper_y, argvals, 20, lambda, 50, 1e-4)?;

    // Calibration probabilities
    let cal_data = subsample_rows(data, &cal_idx);
    let cal_probs = predict_elastic_logistic_probs(&refit, &cal_data, argvals);
    let cal_true: Vec<usize> = cal_idx.iter().map(|&i| usize::from(y[i] == 1)).collect();
    let cal_scores = compute_cal_scores(&cal_probs, &cal_true, score_type);

    // Test probabilities
    let test_probs = predict_elastic_logistic_probs(&refit, test_data, argvals);
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

// ═══════════════════════════════════════════════════════════════════════════
// Classification prediction helpers
// ═══════════════════════════════════════════════════════════════════════════

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
