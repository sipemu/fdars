//! Logistic model, prediction, conformal, and calibration helpers.

use crate::matrix::FdMatrix;
use crate::scalar_on_function::sigmoid;

use super::stability::quantile_sorted;

/// Compute base logistic eta for one observation, excluding a given component.
pub(crate) fn logistic_eta_base(
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scores: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
    i: usize,
    ncomp: usize,
    exclude_component: usize,
) -> f64 {
    let mut eta = fit_intercept;
    for k in 0..ncomp {
        if k != exclude_component {
            eta += coefficients[1 + k] * scores[(i, k)];
        }
    }
    if let Some(sc) = scalar_covariates {
        for j in 0..gamma.len() {
            eta += gamma[j] * sc[(i, j)];
        }
    }
    eta
}

/// Compute logistic accuracy from a score matrix.
pub(crate) fn logistic_accuracy_from_scores(
    score_mat: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    y: &[f64],
    n: usize,
    ncomp: usize,
) -> f64 {
    let correct: usize = (0..n)
        .filter(|&i| {
            let mut eta = fit_intercept;
            for c in 0..ncomp {
                eta += coefficients[1 + c] * score_mat[(i, c)];
            }
            let pred = if sigmoid(eta) >= 0.5 { 1.0 } else { 0.0 };
            (pred - y[i]).abs() < 1e-10
        })
        .count();
    correct as f64 / n as f64
}

/// Compute mean logistic prediction with optional component replacements.
pub(crate) fn logistic_pdp_mean(
    scores: &FdMatrix,
    fit_intercept: f64,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    ncomp: usize,
    replacements: &[(usize, f64)],
) -> f64 {
    let p_scalar = gamma.len();
    let mut sum = 0.0;
    for i in 0..n {
        let mut eta = fit_intercept;
        for c in 0..ncomp {
            let s = replacements
                .iter()
                .find(|&&(comp, _)| comp == c)
                .map_or(scores[(i, c)], |&(_, val)| val);
            eta += coefficients[1 + c] * s;
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                eta += gamma[j] * sc[(i, j)];
            }
        }
        sum += sigmoid(eta);
    }
    sum / n as f64
}

/// Predict from FPC scores + scalar covariates using linear model coefficients.
pub(crate) fn predict_from_scores(
    scores: &FdMatrix,
    coefficients: &[f64],
    gamma: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Vec<f64> {
    let n = scores.nrows();
    let mut preds = vec![0.0; n];
    for i in 0..n {
        let mut yhat = coefficients[0];
        for k in 0..ncomp {
            yhat += coefficients[1 + k] * scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..gamma.len() {
                yhat += gamma[j] * sc[(i, j)];
            }
        }
        preds[i] = yhat;
    }
    preds
}

/// Validate inputs for conformal prediction. Returns (n_cal, n_proper) on success.
pub(crate) fn validate_conformal_inputs(
    n: usize,
    m: usize,
    n_test: usize,
    m_test: usize,
    train_y_len: usize,
    ncomp: usize,
    cal_fraction: f64,
    alpha: f64,
) -> Option<(usize, usize)> {
    let shapes_ok = n >= 4 && n == train_y_len && m > 0 && n_test > 0 && m_test == m;
    let params_ok = cal_fraction > 0.0 && cal_fraction < 1.0 && alpha > 0.0 && alpha < 1.0;
    if !(shapes_ok && params_ok) {
        return None;
    }
    let n_cal = crate::utility::f64_to_usize_clamped((n as f64 * cal_fraction).round()).max(2);
    let n_proper = n - n_cal;
    (n_proper >= ncomp + 2).then_some((n_cal, n_proper))
}

/// Compute conformal calibration quantile and coverage from absolute residuals.
pub(crate) fn conformal_quantile_and_coverage(
    calibration_scores: &[f64],
    cal_n: usize,
    alpha: f64,
) -> (f64, f64) {
    let q_level = (((cal_n + 1) as f64 * (1.0 - alpha)).ceil() / cal_n as f64).min(1.0);
    let mut sorted_scores = calibration_scores.to_vec();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let residual_quantile = quantile_sorted(&sorted_scores, q_level);

    let coverage = calibration_scores
        .iter()
        .filter(|&&s| s <= residual_quantile)
        .count() as f64
        / cal_n as f64;

    (residual_quantile, coverage)
}

/// Weighted calibration gap for a group of sorted indices.
pub(crate) fn calibration_gap_weighted(
    indices: &[usize],
    y: &[f64],
    probabilities: &[f64],
    total_n: usize,
) -> f64 {
    let cnt = indices.len();
    if cnt == 0 {
        return 0.0;
    }
    let sum_y: f64 = indices.iter().map(|&i| y[i]).sum();
    let sum_p: f64 = indices.iter().map(|&i| probabilities[i]).sum();
    let gap = (sum_y / cnt as f64 - sum_p / cnt as f64).abs();
    cnt as f64 / total_n as f64 * gap
}
