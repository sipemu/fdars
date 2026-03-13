//! Split-conformal prediction intervals for standard regression models.

use crate::cv::subset_vec;
use crate::error::FdarError;
use crate::explain::subsample_rows;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{fregre_lm, fregre_np_mixed, predict_fregre_lm, predict_fregre_np};

use super::{
    build_regression_result, conformal_split, validate_split_inputs, ConformalMethod,
    ConformalRegressionResult,
};

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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1),
/// or the proper training set is too small for the requested `ncomp`.
/// Returns [`FdarError::ComputationFailed`] if the internal `fregre_lm` model fitting fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::conformal::conformal_fregre_lm;
///
/// let data = FdMatrix::from_column_major(
///     (0..400).map(|i| (i as f64 * 0.1).sin()).collect(),
///     20, 20,
/// ).unwrap();
/// let y: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let test = FdMatrix::from_column_major(
///     (0..60).map(|i| (i as f64 * 0.15).sin()).collect(),
///     3, 20,
/// ).unwrap();
/// let result = conformal_fregre_lm(&data, &y, &test, None, None, 2, 0.5, 0.1, 42).unwrap();
/// assert_eq!(result.lower.len(), 3);
/// assert_eq!(result.upper.len(), 3);
/// assert_eq!(result.predictions.len(), 3);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
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

    Ok(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}

/// Split-conformal prediction intervals for nonparametric kernel regression.
///
/// Refits [`fregre_np_mixed`] on the proper-training subset.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 4 observations,
/// `test_data` is empty, `y` length differs from the number of rows in `data`,
/// or `data` and `test_data` have different numbers of columns.
/// Returns [`FdarError::InvalidParameter`] if `cal_fraction` or `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the internal `fregre_np_mixed` fitting fails.
#[must_use = "expensive computation whose result should not be discarded"]
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

    Ok(build_regression_result(
        cal_residuals,
        test_preds,
        alpha,
        ConformalMethod::Split,
    ))
}
