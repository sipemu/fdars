//! Functional Regression Control Chart (FRCC).
//!
//! Monitors a functional response after adjusting for known scalar covariates
//! using function-on-scalar regression (FOSR). The residuals are then monitored
//! via FPCA-based T-squared and SPE statistics.
//!
//! This is useful when the process output (functional) depends on known inputs
//! (scalar predictors), and we want to detect deviations beyond what the inputs
//! explain.
//!
//! # Model assessment
//!
//! R-squared = 1 - SSR/SST measures the proportion of response variance explained
//! by the FRCC model (computed pointwise across grid points and summed). For
//! monitoring, R-squared > 0.7 indicates excellent model fit; R-squared 0.5--0.7
//! is adequate; R-squared 0.3--0.5 is marginal (covariate adjustment helps but
//! residual variation is large); R-squared < 0.3 suggests the functional
//! predictors have weak predictive power and monitoring may be ineffective
//! compared to standard `spm_phase1`.
//!
//! After building the FRCC chart, verify `fosr_r_squared` to assess model
//! quality. R-squared values: > 0.5 (strong adjustment), 0.3--0.5 (moderate),
//! 0.1--0.3 (weak), < 0.1 (rejected by default threshold).
//!
//! # Residual assumptions
//!
//! The monitoring assumes regression residuals are independent across observations.
//! Autocorrelated residuals (e.g., from time-series data or batch-to-batch effects)
//! inflate SPE alarm rates because the empirical SPE distribution underestimates the
//! true variability. Check residual autocorrelation using the lag-1 sample
//! autocorrelation of the SPE sequence and consider pre-whitening (e.g., fitting an
//! AR(1) model to the residual scores) if significant.
//!
//! The SPE control limit assumes residuals are approximately independent across
//! observations. If the residuals exhibit temporal autocorrelation, consider using
//! bootstrap control limits via `spe_limit_robust()`.
//!
//! # References
//!
//! - Capezza, C., Lepore, A., Menafoglio, A., Palumbo, B. & Vantini, S.
//!   (2020). Control charts for monitoring ship operating conditions and
//!   CO2 emissions based on scalar-on-function regression. *Applied
//!   Stochastic Models in Business and Industry*, 36(3), 477--500,
//!   section 3.1 (FRCC construction), section 4 (monitoring procedure).

use crate::error::FdarError;
use crate::function_on_scalar::{fosr, predict_fosr, FosrResult};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

use super::control::{spe_control_limit, t2_control_limit, ControlLimit};
use super::phase::{center_data, centered_reconstruct, split_indices};
use super::stats::{hotelling_t2, spe_univariate};

/// Configuration for FRCC chart construction.
#[derive(Debug, Clone, PartialEq)]
pub struct FrccConfig {
    /// Number of principal components for residual FPCA (default 5).
    pub ncomp: usize,
    /// FOSR smoothing parameter; controls roughness penalty on β(t) (default 1e-4).
    /// Larger values produce smoother coefficient functions. Typical range: [1e-6, 1e-2].
    /// Use cross-validation if unsure.
    pub fosr_lambda: f64,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Fraction of data for tuning (default 0.5).
    ///
    /// Must be in (0, 1). The tuning set is used for FOSR fitting and R-squared
    /// assessment; the calibration set (1 - tuning_fraction) is used for FPCA
    /// and control limit estimation. Larger tuning fractions give more stable
    /// FOSR estimates but fewer calibration observations for control limits.
    /// For n < 50, consider tuning_fraction = 0.4 to ensure adequate calibration.
    /// For n > 200, tuning_fraction = 0.6 may improve FOSR stability.
    pub tuning_fraction: f64,
    /// Random seed (default 42).
    pub seed: u64,
    /// Minimum FOSR R² required to proceed (default 0.1).
    /// If the FOSR model explains less than this fraction of variance,
    /// frcc_phase1 returns an error suggesting standard SPM instead.
    ///
    /// Default 0.1 is a lenient threshold that catches only clearly useless
    /// models. For production use, consider 0.2--0.3. An R² of 0.3 means
    /// predictors explain 30% of functional variance --- enough for
    /// meaningful covariate adjustment.
    pub min_r_squared: f64,
}

impl Default for FrccConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            fosr_lambda: 1e-4,
            alpha: 0.05,
            tuning_fraction: 0.5,
            seed: 42,
            min_r_squared: 0.1,
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
    /// Coefficient of determination (R²) for the FOSR model on the tuning set.
    /// Values near 0 suggest the predictors explain little variance, and
    /// the FRCC may not add value over a standard SPM chart.
    pub fosr_r_squared: f64,
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

/// Build a Functional Regression Control Chart from Phase I data.
///
/// 1. Splits data into tuning and calibration sets
/// 2. Fits FOSR on tuning set
/// 3. Computes residuals on calibration set
/// 4. Runs FPCA on calibration residuals
/// 5. Computes T-squared and SPE control limits
///
/// The R² is computed on the tuning set, not the calibration set, to avoid
/// optimistic bias. The tuning set is used for both FOSR fitting and R²
/// assessment.
///
/// The SPE control limit assumes approximately independent residuals. For
/// processes with temporal structure in the residuals, use `spe_limit_robust()`
/// with bootstrap method.
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
///
/// # Example
/// ```no_run
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::frcc::{frcc_phase1, FrccConfig};
/// let n = 60;
/// let m = 10;
/// let y = FdMatrix::from_column_major(
///     (0..n*m).map(|i| (i as f64 * 0.1).sin()).collect(), n, m
/// ).unwrap();
/// let pred = FdMatrix::from_column_major(
///     (0..n).map(|i| i as f64 / n as f64).collect(), n, 1
/// ).unwrap();
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();
/// let config = FrccConfig { min_r_squared: 0.0, ..FrccConfig::default() };
/// let chart = frcc_phase1(&y, &pred, &argvals, &config).unwrap();
/// assert!(chart.fosr_r_squared >= 0.0);
/// ```
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
    if config.tuning_fraction <= 0.0 || config.tuning_fraction >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "tuning_fraction",
            message: format!(
                "tuning_fraction must be in (0, 1), got {}",
                config.tuning_fraction
            ),
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

    if n_cal < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_curves",
            expected: "calibration set with at least 2 observations".to_string(),
            actual: format!("{n_cal} observations in calibration set"),
        });
    }

    // Ensure enough observations for FOSR (needs n >= p + 2)
    if n_tune < p + 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_curves",
            expected: format!("tuning set with at least {} observations", p + 2),
            actual: format!("{n_tune} observations in tuning set"),
        });
    }

    // FOSR on tuning set
    if config.fosr_lambda < 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "fosr_lambda",
            message: format!(
                "fosr_lambda must be non-negative, got {}",
                config.fosr_lambda
            ),
        });
    }
    let fosr_lambda = config.fosr_lambda;
    let fosr_result = fosr(&tune_y, &tune_x, fosr_lambda)?;

    // Compute R² on tuning set.
    // R² is computed pointwise across all grid points, implicitly assuming a
    // uniform grid. For non-uniform grids, this gives equal weight to each
    // discrete point rather than integrating over the domain. For grids with
    // > 3x variation in spacing, consider computing a weighted R² using
    // Simpson's weights on argvals. The pointwise R² here is adequate for
    // uniform or near-uniform grids.
    //
    // The pointwise R² treats each grid point equally, which is appropriate
    // for uniform or near-uniform grids. For strongly non-uniform grids, the
    // functional R² (integrating with quadrature weights) would be more
    // appropriate but is not currently implemented. In practice, the
    // difference is small when the grid has < 3x variation in spacing.
    let tune_predicted = predict_fosr(&fosr_result, &tune_x);
    let fosr_r_squared = {
        let (n_t, m_t) = tune_y.shape();
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        // Per-point mean for total SS
        for j in 0..m_t {
            let col_mean: f64 = (0..n_t).map(|i| tune_y[(i, j)]).sum::<f64>() / n_t as f64;
            for i in 0..n_t {
                ss_res += (tune_y[(i, j)] - tune_predicted[(i, j)]).powi(2);
                ss_tot += (tune_y[(i, j)] - col_mean).powi(2);
            }
        }
        if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        }
    };

    if fosr_r_squared < config.min_r_squared {
        return Err(FdarError::ComputationFailed {
            operation: "frcc_phase1",
            detail: format!(
                "FOSR R² = {fosr_r_squared:.4}; below threshold {:.4}. \
                 Consider: (a) adding more predictors, (b) increasing the training \
                 set size, (c) reducing fosr_lambda for a less smooth fit, or \
                 (d) using standard `spm_phase1` instead. Low R² means the \
                 predictors explain little variance, so covariate adjustment \
                 provides minimal benefit and may introduce estimation noise.",
                config.min_r_squared
            ),
        });
    }

    // Predict on calibration set and compute residuals
    let cal_predicted = predict_fosr(&fosr_result, &cal_x);
    let cal_residuals = compute_residuals(&cal_y, &cal_predicted);

    // FPCA on calibration residuals
    let ncomp = config.ncomp.min(n_cal - 1).min(m);
    let residual_fpca = fdata_to_pc_1d(&cal_residuals, ncomp, argvals)?;
    let actual_ncomp = residual_fpca.scores.ncols();

    // Eigenvalues
    let eigenvalues: Vec<f64> = residual_fpca
        .singular_values
        .iter()
        .take(actual_ncomp)
        .map(|&sv| sv * sv / (n_cal as f64 - 1.0))
        .collect();

    // T² on calibration scores: computed to verify FPCA but not used for
    // control limits (which use chi² quantiles instead of empirical limits).
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
        fosr_r_squared,
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
    // Verify predictor count matches the FOSR model (beta has p rows, one per predictor).
    let expected_p = chart.fosr.beta.nrows();
    if new_predictors.ncols() != expected_p {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{expected_p} columns (predictors)"),
            actual: format!("{} columns", new_predictors.ncols()),
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
