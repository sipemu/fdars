//! VIF, influence diagnostics, DFBETAS/DFFITS, prediction intervals, and LOO-CV.

use super::helpers::*;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::scalar_on_function::{
    build_design_matrix, cholesky_factor, cholesky_forward_back, compute_hat_diagonal, compute_xtx,
    FregreLmResult, FunctionalLogisticResult,
};

// ===========================================================================
// VIF (Variance Inflation Factors)
// ===========================================================================

/// Result of VIF analysis for FPC-based regression.
#[derive(Debug, Clone)]
pub struct VifResult {
    /// VIF values (length ncomp + p_scalar, excludes intercept).
    pub vif: Vec<f64>,
    /// Labels for each predictor.
    pub labels: Vec<String>,
    /// Mean VIF.
    pub mean_vif: f64,
    /// Number of predictors with VIF > 5.
    pub n_moderate: usize,
    /// Number of predictors with VIF > 10.
    pub n_severe: usize,
}

/// Variance inflation factors for FPC scores (and optional scalar covariates).
///
/// For orthogonal FPC scores without scalar covariates, VIF should be approximately 1.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_vif`] for error conditions.
pub fn fpc_vif(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<VifResult, FdarError> {
    crate::explain_generic::generic_vif(fit, data, scalar_covariates)
}

/// VIF for a functional logistic regression model.
///
/// # Errors
///
/// See [`crate::explain_generic::generic_vif`] for error conditions.
pub fn fpc_vif_logistic(
    fit: &FunctionalLogisticResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<VifResult, FdarError> {
    crate::explain_generic::generic_vif(fit, data, scalar_covariates)
}

pub(crate) fn compute_vif_from_scores(
    scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> Option<VifResult> {
    let p_scalar = scalar_covariates.map_or(0, super::super::matrix::FdMatrix::ncols);
    let p = ncomp + p_scalar;
    if p == 0 || n <= p {
        return None;
    }

    let x_noi = build_no_intercept_matrix(scores, ncomp, scalar_covariates, n);
    let xtx = compute_xtx(&x_noi);
    let l = cholesky_factor(&xtx, p)?;

    let mut vif = vec![0.0; p];
    for k in 0..p {
        let mut ek = vec![0.0; p];
        ek[k] = 1.0;
        let v = cholesky_forward_back(&l, &ek, p);
        vif[k] = v[k] * xtx[k * p + k];
    }

    let mut labels = Vec::with_capacity(p);
    for k in 0..ncomp {
        labels.push(format!("FPC_{k}"));
    }
    for j in 0..p_scalar {
        labels.push(format!("scalar_{j}"));
    }

    let mean_vif = vif.iter().sum::<f64>() / p as f64;
    let n_moderate = vif.iter().filter(|&&v| v > 5.0).count();
    let n_severe = vif.iter().filter(|&&v| v > 10.0).count();

    Some(VifResult {
        vif,
        labels,
        mean_vif,
        n_moderate,
        n_severe,
    })
}

/// Build design matrix without intercept: scores + optional scalars.
fn build_no_intercept_matrix(
    scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> FdMatrix {
    let p_scalar = scalar_covariates.map_or(0, super::super::matrix::FdMatrix::ncols);
    let p = ncomp + p_scalar;
    let mut x = FdMatrix::zeros(n, p);
    for i in 0..n {
        for k in 0..ncomp {
            x[(i, k)] = scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                x[(i, ncomp + j)] = sc[(i, j)];
            }
        }
    }
    x
}

// ===========================================================================
// Influence Diagnostics (Cook's D + Leverage)
// ===========================================================================

/// Cook's distance and leverage diagnostics for the FPC regression.
#[derive(Debug, Clone)]
pub struct InfluenceDiagnostics {
    /// Hat matrix diagonal h_ii (length n).
    pub leverage: Vec<f64>,
    /// Cook's distance D_i (length n).
    pub cooks_distance: Vec<f64>,
    /// Number of model parameters (intercept + ncomp + p_scalar).
    pub p: usize,
    /// Residual mean squared error.
    pub mse: f64,
}

/// Compute leverage and Cook's distance for a linear functional regression.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its column
/// count does not match `fit.fpca.mean`, or the number of rows is not greater
/// than the number of model parameters.
/// Returns [`FdarError::ComputationFailed`] if Cholesky factorization fails.
pub fn influence_diagnostics(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<InfluenceDiagnostics, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();

    if n <= p {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!(">{p} rows (more than parameters)"),
            actual: format!("{n}"),
        });
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p).ok_or_else(|| FdarError::ComputationFailed {
        operation: "influence_diagnostics",
        detail: "Cholesky factorization failed".into(),
    })?;
    let leverage = compute_hat_diagonal(&design, &l);

    let ss_res: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let mse = ss_res / (n - p) as f64;

    let mut cooks_distance = vec![0.0; n];
    for i in 0..n {
        let e = fit.residuals[i];
        let h = leverage[i];
        let denom = p as f64 * mse * (1.0 - h).powi(2);
        cooks_distance[i] = if denom > 0.0 { e * e * h / denom } else { 0.0 };
    }

    Ok(InfluenceDiagnostics {
        leverage,
        cooks_distance,
        p,
        mse,
    })
}

// ===========================================================================
// DFBETAS / DFFITS
// ===========================================================================

/// Result of DFBETAS/DFFITS influence diagnostics.
#[derive(Debug, Clone)]
pub struct DfbetasDffitsResult {
    /// DFBETAS values (n x p).
    pub dfbetas: FdMatrix,
    /// DFFITS values (length n).
    pub dffits: Vec<f64>,
    /// Studentized residuals (length n).
    pub studentized_residuals: Vec<f64>,
    /// Number of parameters p (including intercept).
    pub p: usize,
    /// DFBETAS cutoff: 2/sqrt(n).
    pub dfbetas_cutoff: f64,
    /// DFFITS cutoff: 2*sqrt(p/n).
    pub dffits_cutoff: f64,
}

/// DFBETAS and DFFITS for a linear functional regression model.
///
/// DFBETAS measures how much each coefficient changes when observation i is deleted.
/// DFFITS measures how much the fitted value changes when observation i is deleted.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its column
/// count does not match `fit.fpca.mean`, or the number of rows is not greater
/// than the number of model parameters.
/// Returns [`FdarError::ComputationFailed`] if Cholesky factorization fails or
/// the residual standard error is near zero.
pub fn dfbetas_dffits(
    fit: &FregreLmResult,
    data: &FdMatrix,
    scalar_covariates: Option<&FdMatrix>,
) -> Result<DfbetasDffitsResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();

    if n <= p {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!(">{p} rows (more than parameters)"),
            actual: format!("{n}"),
        });
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p).ok_or_else(|| FdarError::ComputationFailed {
        operation: "dfbetas_dffits",
        detail: "Cholesky factorization failed".into(),
    })?;
    let hat_diag = compute_hat_diagonal(&design, &l);

    let ss_res: f64 = fit.residuals.iter().map(|r| r * r).sum();
    let mse = ss_res / (n - p) as f64;
    let s = mse.sqrt();

    if s < 1e-15 {
        return Err(FdarError::ComputationFailed {
            operation: "dfbetas_dffits",
            detail: "residual standard error is near zero".into(),
        });
    }

    let se = compute_coefficient_se(&l, mse, p);

    let mut studentized_residuals = vec![0.0; n];
    let mut dffits = vec![0.0; n];
    let mut dfbetas = FdMatrix::zeros(n, p);

    for i in 0..n {
        let (t_i, dffits_i, dfb) =
            compute_obs_influence(&design, &l, fit.residuals[i], hat_diag[i], s, &se, p, i);
        studentized_residuals[i] = t_i;
        dffits[i] = dffits_i;
        for j in 0..p {
            dfbetas[(i, j)] = dfb[j];
        }
    }

    let dfbetas_cutoff = 2.0 / (n as f64).sqrt();
    let dffits_cutoff = 2.0 * (p as f64 / n as f64).sqrt();

    Ok(DfbetasDffitsResult {
        dfbetas,
        dffits,
        studentized_residuals,
        p,
        dfbetas_cutoff,
        dffits_cutoff,
    })
}

/// Compute coefficient standard errors from Cholesky factor and MSE.
fn compute_coefficient_se(l: &[f64], mse: f64, p: usize) -> Vec<f64> {
    let mut se = vec![0.0; p];
    for j in 0..p {
        let mut ej = vec![0.0; p];
        ej[j] = 1.0;
        let v = cholesky_forward_back(l, &ej, p);
        se[j] = (mse * v[j].max(0.0)).sqrt();
    }
    se
}

/// Compute DFBETAS row, DFFITS, and studentized residual for a single observation.
fn compute_obs_influence(
    design: &FdMatrix,
    l: &[f64],
    residual: f64,
    h_ii: f64,
    s: f64,
    se: &[f64],
    p: usize,
    i: usize,
) -> (f64, f64, Vec<f64>) {
    let one_minus_h = (1.0 - h_ii).max(1e-15);
    let t_i = residual / (s * one_minus_h.sqrt());
    let dffits_i = t_i * (h_ii / one_minus_h).sqrt();

    let mut xi = vec![0.0; p];
    for j in 0..p {
        xi[j] = design[(i, j)];
    }
    let inv_xtx_xi = cholesky_forward_back(l, &xi, p);
    let mut dfb = vec![0.0; p];
    for j in 0..p {
        if se[j] > 1e-15 {
            dfb[j] = inv_xtx_xi[j] * residual / (one_minus_h * se[j]);
        }
    }

    (t_i, dffits_i, dfb)
}

// ===========================================================================
// Prediction Intervals
// ===========================================================================

/// Result of prediction interval computation.
#[derive(Debug, Clone)]
pub struct PredictionIntervalResult {
    /// Point predictions y_hat_new (length n_new).
    pub predictions: Vec<f64>,
    /// Lower bounds (length n_new).
    pub lower: Vec<f64>,
    /// Upper bounds (length n_new).
    pub upper: Vec<f64>,
    /// Prediction standard errors: s * sqrt(1 + h_new) (length n_new).
    pub prediction_se: Vec<f64>,
    /// Confidence level used.
    pub confidence_level: f64,
    /// Critical value used.
    pub t_critical: f64,
    /// Residual standard error from the training model.
    pub residual_se: f64,
}

/// Prediction intervals for new observations from a linear functional regression model.
///
/// Computes prediction intervals accounting for both estimation uncertainty
/// (through the hat matrix) and residual variance.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `confidence_level` is not in (0, 1).
/// Returns [`FdarError::InvalidDimension`] if `train_data` or `new_data` has zero
/// rows, column counts do not match `fit.fpca.mean` or each other, or the number
/// of training rows is not greater than the number of model parameters.
/// Returns [`FdarError::ComputationFailed`] if Cholesky factorization fails.
pub fn prediction_intervals(
    fit: &FregreLmResult,
    train_data: &FdMatrix,
    train_scalar: Option<&FdMatrix>,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    confidence_level: f64,
) -> Result<PredictionIntervalResult, FdarError> {
    let (n_train, m) = train_data.shape();
    let (n_new, m_new) = new_data.shape();
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "confidence_level",
            message: "must be in (0, 1)".into(),
        });
    }
    if n_train == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "train_data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "train_data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    if n_new == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if m_new != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns (matching train)"),
            actual: format!("{m_new}"),
        });
    }
    let ncomp = fit.ncomp;

    let train_scores = project_scores(train_data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let train_design = build_design_matrix(&train_scores, ncomp, train_scalar, n_train);
    let p = train_design.ncols();
    if n_train <= p {
        return Err(FdarError::InvalidDimension {
            parameter: "train_data",
            expected: format!(">{p} rows (more than parameters)"),
            actual: format!("{n_train}"),
        });
    }

    let xtx = compute_xtx(&train_design);
    let l = cholesky_factor(&xtx, p).ok_or_else(|| FdarError::ComputationFailed {
        operation: "prediction_intervals",
        detail: "Cholesky factorization failed".into(),
    })?;

    let residual_se = fit.residual_se;
    let df = n_train - p;
    let t_crit = t_critical_value(confidence_level, df);

    // Project new data
    let new_scores = project_scores(new_data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);

    let mut predictions = vec![0.0; n_new];
    let mut lower = vec![0.0; n_new];
    let mut upper = vec![0.0; n_new];
    let mut prediction_se = vec![0.0; n_new];

    let p_scalar = fit.gamma.len();

    for i in 0..n_new {
        let x_new = build_design_vector(&new_scores, new_scalar, i, ncomp, p_scalar, p);
        let (yhat, lo, up, pse) =
            compute_prediction_interval_obs(&l, &fit.coefficients, &x_new, p, residual_se, t_crit);
        predictions[i] = yhat;
        lower[i] = lo;
        upper[i] = up;
        prediction_se[i] = pse;
    }

    Ok(PredictionIntervalResult {
        predictions,
        lower,
        upper,
        prediction_se,
        confidence_level,
        t_critical: t_crit,
        residual_se,
    })
}

/// Normal quantile approximation (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    let val = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);
    if p < 0.5 {
        -val
    } else {
        val
    }
}

/// t-distribution critical value with Cornish-Fisher correction for small df.
fn t_critical_value(conf: f64, df: usize) -> f64 {
    let alpha = 1.0 - conf;
    let z = normal_quantile(1.0 - alpha / 2.0);
    if df == 0 {
        return z;
    }
    let df_f = df as f64;
    let g1 = (z.powi(3) + z) / (4.0 * df_f);
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / (96.0 * df_f * df_f);
    let g3 = (3.0 * z.powi(7) + 19.0 * z.powi(5) + 17.0 * z.powi(3) - 15.0 * z)
        / (384.0 * df_f * df_f * df_f);
    z + g1 + g2 + g3
}

/// Build a design vector [1, scores, scalars] for one observation.
fn build_design_vector(
    new_scores: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    i: usize,
    ncomp: usize,
    p_scalar: usize,
    p: usize,
) -> Vec<f64> {
    let mut x = vec![0.0; p];
    x[0] = 1.0;
    for k in 0..ncomp {
        x[1 + k] = new_scores[(i, k)];
    }
    if let Some(ns) = new_scalar {
        for j in 0..p_scalar {
            x[1 + ncomp + j] = ns[(i, j)];
        }
    }
    x
}

/// Compute prediction interval for a single observation.
fn compute_prediction_interval_obs(
    l: &[f64],
    coefficients: &[f64],
    x_new: &[f64],
    p: usize,
    residual_se: f64,
    t_crit: f64,
) -> (f64, f64, f64, f64) {
    let yhat: f64 = x_new.iter().zip(coefficients).map(|(a, b)| a * b).sum();
    let v = cholesky_forward_back(l, x_new, p);
    let h_new: f64 = x_new.iter().zip(&v).map(|(a, b)| a * b).sum();
    let pred_se = residual_se * (1.0 + h_new).sqrt();
    (
        yhat,
        yhat - t_crit * pred_se,
        yhat + t_crit * pred_se,
        pred_se,
    )
}

// ===========================================================================
// LOO-CV / PRESS
// ===========================================================================

/// Result of leave-one-out cross-validation diagnostics.
#[derive(Debug, Clone)]
pub struct LooCvResult {
    /// LOO residuals: e_i / (1 - h_ii), length n.
    pub loo_residuals: Vec<f64>,
    /// PRESS statistic: sum loo_residuals^2.
    pub press: f64,
    /// LOO R^2: 1 - PRESS / TSS.
    pub loo_r_squared: f64,
    /// Hat diagonal h_ii, length n.
    pub leverage: Vec<f64>,
    /// Total sum of squares: sum (y_i - y_bar)^2.
    pub tss: f64,
}

/// LOO-CV / PRESS diagnostics for a linear functional regression model.
///
/// Uses the hat-matrix shortcut: LOO residual = e_i / (1 - h_ii).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, its column
/// count does not match `fit.fpca.mean`, `y.len()` does not match the row count,
/// or the number of rows is not greater than the number of model parameters.
/// Returns [`FdarError::ComputationFailed`] if Cholesky factorization fails or
/// the total sum of squares is zero.
pub fn loo_cv_press(
    fit: &FregreLmResult,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Result<LooCvResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: ">0 rows".into(),
            actual: "0".into(),
        });
    }
    if n != y.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n} (matching data rows)"),
            actual: format!("{}", y.len()),
        });
    }
    if m != fit.fpca.mean.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("{} columns", fit.fpca.mean.len()),
            actual: format!("{m}"),
        });
    }
    let ncomp = fit.ncomp;
    let scores = project_scores(data, &fit.fpca.mean, &fit.fpca.rotation, ncomp);
    let design = build_design_matrix(&scores, ncomp, scalar_covariates, n);
    let p = design.ncols();
    if n <= p {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!(">{p} rows (more than parameters)"),
            actual: format!("{n}"),
        });
    }

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p).ok_or_else(|| FdarError::ComputationFailed {
        operation: "loo_cv_press",
        detail: "Cholesky factorization failed".into(),
    })?;
    let leverage = compute_hat_diagonal(&design, &l);

    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    if tss == 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "loo_cv_press",
            detail: "total sum of squares is zero".into(),
        });
    }

    let mut loo_residuals = vec![0.0; n];
    let mut press = 0.0;
    for i in 0..n {
        let denom = (1.0 - leverage[i]).max(1e-15);
        loo_residuals[i] = fit.residuals[i] / denom;
        press += loo_residuals[i] * loo_residuals[i];
    }

    let loo_r_squared = 1.0 - press / tss;

    Ok(LooCvResult {
        loo_residuals,
        press,
        loo_r_squared,
        leverage,
        tss,
    })
}
