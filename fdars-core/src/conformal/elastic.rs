//! Split-conformal prediction intervals for elastic regression models and
//! elastic prediction helpers.

use crate::cv::subset_vec;
use crate::elastic_regression::{
    elastic_pcr, elastic_regression, ElasticPcrResult, ElasticRegressionResult, PcaMethod,
};
use crate::error::FdarError;
use crate::explain::subsample_rows;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;

use super::{
    build_regression_result, conformal_split, validate_split_inputs, ConformalConfig,
    ConformalMethod, ConformalRegressionResult,
};

/// Split-conformal prediction intervals for elastic regression.
///
/// Refits [`elastic_regression`] on the proper-training subset and predicts
/// on calibration / test data using the estimated beta(t) and warping.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the internal `elastic_regression` fitting fails.
#[must_use = "expensive computation whose result should not be discarded"]
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
    if data.ncols() != test_data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: format!("{} columns", data.ncols()),
            actual: format!("{} columns", test_data.ncols()),
        });
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

    Ok(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for elastic PCR.
///
/// Refits [`elastic_pcr`] on the proper-training subset.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if `elastic_pcr` fitting or prediction fails.
#[must_use = "expensive computation whose result should not be discarded"]
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
    if data.ncols() != test_data.ncols() {
        return Err(FdarError::InvalidDimension {
            parameter: "test_data",
            expected: format!("{} columns", data.ncols()),
            actual: format!("{} columns", test_data.ncols()),
        });
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

    Ok(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for elastic regression using a [`ConformalConfig`].
///
/// This is the config-based alternative to [`conformal_elastic_regression`].
///
/// # Errors
///
/// Same error conditions as [`conformal_elastic_regression`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn conformal_elastic_regression_with_config(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    argvals: &[f64],
    ncomp_beta: usize,
    lambda: f64,
    config: &ConformalConfig,
) -> Result<ConformalRegressionResult, FdarError> {
    conformal_elastic_regression(
        data,
        y,
        test_data,
        argvals,
        ncomp_beta,
        lambda,
        config.cal_fraction,
        config.alpha,
        config.seed,
    )
}

/// Split-conformal prediction intervals for elastic PCR using a [`ConformalConfig`].
///
/// This is the config-based alternative to [`conformal_elastic_pcr`].
///
/// # Errors
///
/// Same error conditions as [`conformal_elastic_pcr`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn conformal_elastic_pcr_with_config(
    data: &FdMatrix,
    y: &[f64],
    test_data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    pca_method: PcaMethod,
    lambda: f64,
    config: &ConformalConfig,
) -> Result<ConformalRegressionResult, FdarError> {
    conformal_elastic_pcr(
        data,
        y,
        test_data,
        argvals,
        ncomp,
        pca_method,
        lambda,
        config.cal_fraction,
        config.alpha,
        config.seed,
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Elastic prediction helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Predict from elastic regression result on new data.
///
/// Aligns new curves to the estimated beta(t) and computes inner products.
fn predict_elastic_reg(
    result: &ElasticRegressionResult,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let (n_new, m) = new_data.shape();
    let weights = crate::helpers::simpsons_weights(argvals);
    let q_new = crate::alignment::srsf_transform(new_data, argvals);
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;

    iter_maybe_parallel!(0..n_new)
        .map(|i| {
            let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
            let gam = crate::alignment::dp_alignment_core(&result.beta, &qi, argvals, 0.0);
            let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
            let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

            let mut pred = result.alpha;
            for j in 0..m {
                let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
                pred += q_aligned * result.beta[j] * weights[j];
            }
            pred
        })
        .collect()
}

/// Project new curves onto elastic PCA eigenfunctions to obtain PC scores.
///
/// Aligns each new SRSF to the Karcher mean, then projects the centered/aligned
/// representation onto the stored eigenfunctions for the chosen PCA method.
fn rebuild_elastic_pcr_model(
    result: &ElasticPcrResult,
    q_new: &FdMatrix,
    argvals: &[f64],
) -> Result<FdMatrix, FdarError> {
    let (n_new, m) = q_new.shape();
    let mean_srsf = &result.karcher.mean_srsf;

    match result.pca_method {
        PcaMethod::Vertical => {
            let fpca = result
                .vert_fpca
                .as_ref()
                .ok_or_else(|| FdarError::ComputationFailed {
                    operation: "predict_elastic_pcr",
                    detail: "vertical FPCA result missing".to_string(),
                })?;
            let ncomp = fpca.scores.ncols();
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
                let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
                let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
                let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

                for k in 0..ncomp {
                    let mut s = 0.0;
                    for j in 0..m {
                        let q_aligned = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
                        let centered = q_aligned - mean_srsf[j.min(mean_srsf.len() - 1)];
                        s += centered * fpca.eigenfunctions_q[(k, j)];
                    }
                    sc[(i, k)] = s;
                }
            }
            Ok(sc)
        }
        PcaMethod::Horizontal => {
            let fpca = result
                .horiz_fpca
                .as_ref()
                .ok_or_else(|| FdarError::ComputationFailed {
                    operation: "predict_elastic_pcr",
                    detail: "horizontal FPCA result missing".to_string(),
                })?;
            let ncomp = fpca.scores.ncols().min(result.coefficients.len());
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
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
            Ok(sc)
        }
        PcaMethod::Joint => {
            let fpca = result
                .joint_fpca
                .as_ref()
                .ok_or_else(|| FdarError::ComputationFailed {
                    operation: "predict_elastic_pcr",
                    detail: "joint FPCA result missing".to_string(),
                })?;
            let ncomp = fpca.scores.ncols().min(result.coefficients.len());
            let mut sc = FdMatrix::zeros(n_new, ncomp);
            for i in 0..n_new {
                let qi: Vec<f64> = (0..m).map(|j| q_new[(i, j)]).collect();
                let gam = crate::alignment::dp_alignment_core(mean_srsf, &qi, argvals, 0.0);
                let q_warped = crate::alignment::reparameterize_curve(&qi, argvals, &gam);
                let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
                let gam_deriv = crate::helpers::gradient_uniform(&gam, h);

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
            Ok(sc)
        }
    }
}

/// Apply the elastic PCR linear model: y_i = alpha + sum_k(coef_k * score_ik).
fn predict_from_elastic_pca(alpha: f64, coefficients: &[f64], scores: &FdMatrix) -> Vec<f64> {
    let n = scores.nrows();
    let ncomp = coefficients.len().min(scores.ncols());
    let mut preds = vec![0.0; n];
    for i in 0..n {
        preds[i] = alpha;
        for k in 0..ncomp {
            preds[i] += coefficients[k] * scores[(i, k)];
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
) -> Result<Vec<f64>, FdarError> {
    let q_new = crate::alignment::srsf_transform(new_data, argvals);
    let scores = rebuild_elastic_pcr_model(result, &q_new, argvals)?;
    Ok(predict_from_elastic_pca(
        result.alpha,
        &result.coefficients,
        &scores,
    ))
}
