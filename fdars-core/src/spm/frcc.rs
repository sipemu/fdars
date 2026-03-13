//! Functional Regression Control Chart (FRCC).
//!
//! Monitors a functional response after adjusting for known scalar covariates
//! using function-on-scalar regression (FOSR). The residuals are then monitored
//! via FPCA-based T-squared and SPE statistics.
//!
//! This is useful when the process output (functional) depends on known inputs
//! (scalar predictors), and we want to detect deviations beyond what the inputs
//! explain.

use crate::error::FdarError;
use crate::function_on_scalar::{fosr, predict_fosr, FosrResult};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

use super::control::{spe_control_limit, t2_control_limit, ControlLimit};
use super::stats::{hotelling_t2, spe_univariate};

/// Configuration for FRCC chart construction.
#[derive(Debug, Clone, PartialEq)]
pub struct FrccConfig {
    /// Number of principal components for residual FPCA (default 5).
    pub ncomp: usize,
    /// FOSR smoothing parameter; use small positive value (default 1e-4).
    pub fosr_lambda: f64,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Fraction of data for tuning (default 0.5).
    pub tuning_fraction: f64,
    /// Random seed (default 42).
    pub seed: u64,
}

impl Default for FrccConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            fosr_lambda: 1e-4,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
        }
    }
}

/// Phase I FRCC chart.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FrccChart {
    /// Fitted FOSR model.
    pub fosr: FosrResult,
    /// FPCA on calibration residuals.
    pub residual_fpca: FpcaResult,
    /// Eigenvalues from residual FPCA.
    pub eigenvalues: Vec<f64>,
    /// T-squared control limit.
    pub t2_limit: ControlLimit,
    /// SPE control limit.
    pub spe_limit: ControlLimit,
    /// Configuration used.
    pub config: FrccConfig,
}

/// Result of FRCC monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FrccMonitorResult {
    /// T-squared values.
    pub t2: Vec<f64>,
    /// SPE values.
    pub spe: Vec<f64>,
    /// T-squared alarm flags.
    pub t2_alarm: Vec<bool>,
    /// SPE alarm flags.
    pub spe_alarm: Vec<bool>,
    /// Residual scores.
    pub residual_scores: FdMatrix,
}

/// Split indices into tuning and calibration sets (same logic as phase.rs).
fn split_indices(n: usize, tuning_fraction: f64, seed: u64) -> (Vec<usize>, Vec<usize>) {
    let n_tune = ((n as f64 * tuning_fraction).round() as usize)
        .max(2)
        .min(n - 1);

    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng_state: u64 = seed;
    for i in (1..n).rev() {
        rng_state = rng_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = (rng_state >> 33) as usize % (i + 1);
        indices.swap(i, j);
    }

    let tune_indices: Vec<usize> = indices[..n_tune].to_vec();
    let cal_indices: Vec<usize> = indices[n_tune..].to_vec();
    (tune_indices, cal_indices)
}

/// Compute residuals: observed - predicted.
fn compute_residuals(observed: &FdMatrix, predicted: &FdMatrix) -> FdMatrix {
    let (n, m) = observed.shape();
    let mut residuals = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            residuals[(i, j)] = observed[(i, j)] - predicted[(i, j)];
        }
    }
    residuals
}

/// Compute centered reconstruction (without adding back mean) for SPE.
fn centered_reconstruct(fpca: &FpcaResult, scores: &FdMatrix, ncomp: usize) -> FdMatrix {
    let n = scores.nrows();
    let m = fpca.mean.len();
    let ncomp = ncomp.min(fpca.rotation.ncols()).min(scores.ncols());

    let mut recon = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let mut val = 0.0;
            for k in 0..ncomp {
                val += scores[(i, k)] * fpca.rotation[(j, k)];
            }
            recon[(i, j)] = val;
        }
    }
    recon
}

/// Center data by subtracting mean.
fn center_data(data: &FdMatrix, mean: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    let mut centered = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            centered[(i, j)] = data[(i, j)] - mean[j];
        }
    }
    centered
}

/// Build a Functional Regression Control Chart from Phase I data.
///
/// 1. Splits data into tuning and calibration sets
/// 2. Fits FOSR on tuning set
/// 3. Computes residuals on calibration set
/// 4. Runs FPCA on calibration residuals
/// 5. Computes T-squared and SPE control limits
///
/// # Arguments
/// * `y_curves` - Functional response (n x m)
/// * `predictors` - Scalar predictors (n x p)
/// * `argvals` - Grid points (length m)
/// * `config` - FRCC configuration
///
/// # Errors
///
/// Returns errors from FOSR fitting, FPCA, or control limit estimation.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn frcc_phase1(
    y_curves: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &FrccConfig,
) -> Result<FrccChart, FdarError> {
    let (n, m) = y_curves.shape();
    let p = predictors.ncols();

    if n < 6 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_curves",
            expected: "at least 6 observations".to_string(),
            actual: format!("{n} observations"),
        });
    }
    if predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: format!("{n} rows"),
            actual: format!("{} rows", predictors.nrows()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    // Split
    let (tune_idx, cal_idx) = split_indices(n, config.tuning_fraction, config.seed);

    let tune_y = crate::cv::subset_rows(y_curves, &tune_idx);
    let tune_x = crate::cv::subset_rows(predictors, &tune_idx);
    let cal_y = crate::cv::subset_rows(y_curves, &cal_idx);
    let cal_x = crate::cv::subset_rows(predictors, &cal_idx);

    let n_tune = tune_y.nrows();
    let n_cal = cal_y.nrows();

    // Ensure enough observations for FOSR (needs n >= p + 2)
    if n_tune < p + 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_curves",
            expected: format!("tuning set with at least {} observations", p + 2),
            actual: format!("{n_tune} observations in tuning set"),
        });
    }

    // FOSR on tuning set
    let fosr_lambda = if config.fosr_lambda < 0.0 {
        1e-4
    } else {
        config.fosr_lambda
    };
    let fosr_result = fosr(&tune_y, &tune_x, fosr_lambda)?;

    // Predict on calibration set and compute residuals
    let cal_predicted = predict_fosr(&fosr_result, &cal_x);
    let cal_residuals = compute_residuals(&cal_y, &cal_predicted);

    // FPCA on calibration residuals
    let ncomp = config.ncomp.min(n_cal - 1).min(m);
    let residual_fpca = fdata_to_pc_1d(&cal_residuals, ncomp)?;
    let actual_ncomp = residual_fpca.scores.ncols();

    // Eigenvalues
    let eigenvalues: Vec<f64> = residual_fpca
        .singular_values
        .iter()
        .take(actual_ncomp)
        .map(|&sv| sv * sv / (n_cal as f64 - 1.0))
        .collect();

    // T-squared on calibration residual scores
    let _t2_cal = hotelling_t2(&residual_fpca.scores, &eigenvalues)?;

    // SPE on calibration residuals
    let cal_resid_centered = center_data(&cal_residuals, &residual_fpca.mean);
    let cal_resid_recon = centered_reconstruct(&residual_fpca, &residual_fpca.scores, actual_ncomp);
    let spe_cal = spe_univariate(&cal_resid_centered, &cal_resid_recon, argvals)?;

    // Control limits
    let t2_limit = t2_control_limit(actual_ncomp, config.alpha)?;
    let spe_limit = spe_control_limit(&spe_cal, config.alpha)?;

    Ok(FrccChart {
        fosr: fosr_result,
        residual_fpca,
        eigenvalues,
        t2_limit,
        spe_limit,
        config: config.clone(),
    })
}

/// Monitor new data against a Functional Regression Control Chart.
///
/// 1. Predicts functional response from FOSR model
/// 2. Computes residuals
/// 3. Projects residuals through FPCA
/// 4. Computes T-squared and SPE
///
/// # Arguments
/// * `chart` - Phase I FRCC chart
/// * `new_y` - New functional response (n_new x m)
/// * `new_predictors` - New scalar predictors (n_new x p)
/// * `argvals` - Grid points (length m)
///
/// # Errors
///
/// Returns errors from FOSR prediction, FPCA projection, or statistic computation.
#[must_use = "monitoring result should not be discarded"]
pub fn frcc_monitor(
    chart: &FrccChart,
    new_y: &FdMatrix,
    new_predictors: &FdMatrix,
    argvals: &[f64],
) -> Result<FrccMonitorResult, FdarError> {
    let m = chart.residual_fpca.mean.len();
    if new_y.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_y",
            expected: format!("{m} columns"),
            actual: format!("{} columns", new_y.ncols()),
        });
    }
    if new_y.nrows() != new_predictors.nrows() {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{} rows", new_y.nrows()),
            actual: format!("{} rows", new_predictors.nrows()),
        });
    }

    let ncomp = chart.eigenvalues.len();

    // Predict and compute residuals
    let predicted = predict_fosr(&chart.fosr, new_predictors);
    let residuals = compute_residuals(new_y, &predicted);

    // Project residuals through FPCA
    let residual_scores = chart.residual_fpca.project(&residuals)?;

    // T-squared
    let t2 = hotelling_t2(&residual_scores, &chart.eigenvalues)?;

    // SPE
    let resid_centered = center_data(&residuals, &chart.residual_fpca.mean);
    let resid_recon = centered_reconstruct(&chart.residual_fpca, &residual_scores, ncomp);
    let spe = spe_univariate(&resid_centered, &resid_recon, argvals)?;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > chart.t2_limit.ucl).collect();
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > chart.spe_limit.ucl).collect();

    Ok(FrccMonitorResult {
        t2,
        spe,
        t2_alarm,
        spe_alarm,
        residual_scores,
    })
}
