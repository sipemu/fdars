//! Profile monitoring for functional data.
//!
//! Monitors the relationship between scalar predictors and functional
//! responses over time using Function-on-Scalar Regression (FOSR).
//! Detects changes in the coefficient functions β(t) via FPCA and T².

use crate::error::FdarError;
use crate::function_on_scalar::{fosr, FosrResult};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use crate::spm::control::{t2_control_limit, ControlLimit};
use crate::spm::stats::hotelling_t2;

/// Configuration for profile monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct ProfileMonitorConfig {
    /// FOSR smoothing parameter (default 1e-4).
    pub fosr_lambda: f64,
    /// Number of principal components for beta FPCA (default 3).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Window size for rolling FOSR (default 20).
    pub window_size: usize,
    /// Step size between windows (default 1).
    pub step_size: usize,
}

impl Default for ProfileMonitorConfig {
    fn default() -> Self {
        Self {
            fosr_lambda: 1e-4,
            ncomp: 3,
            alpha: 0.05,
            window_size: 20,
            step_size: 1,
        }
    }
}

/// Phase I profile monitoring chart.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProfileChart {
    /// FOSR result from the full reference data.
    pub reference_fosr: FosrResult,
    /// FPCA of the rolling beta coefficient functions.
    pub beta_fpca: FpcaResult,
    /// Eigenvalues: sv² / (n_windows - 1).
    pub eigenvalues: Vec<f64>,
    /// T-squared control limit for beta monitoring.
    pub t2_limit: ControlLimit,
    /// Configuration used.
    pub config: ProfileMonitorConfig,
}

/// Result of Phase II profile monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ProfileMonitorResult {
    /// Per-window beta coefficient matrices (vectorized).
    pub betas: FdMatrix,
    /// T-squared values for each window.
    pub t2: Vec<f64>,
    /// T-squared alarm flags.
    pub t2_alarm: Vec<bool>,
    /// FPC scores for the beta functions.
    pub beta_scores: FdMatrix,
}

/// Build a Phase I profile monitoring chart.
///
/// 1. Fits FOSR on the full training data to get reference β(t)
/// 2. Rolls windows over the data, fitting FOSR per window
/// 3. Vectorizes the β(t) from each window
/// 4. Runs FPCA on the vectorized betas
/// 5. Computes T-squared control limits
///
/// # Arguments
/// * `y_curves` - Response functional data (n × m)
/// * `predictors` - Scalar predictors (n × p)
/// * `argvals` - Grid points (length m)
/// * `config` - Profile monitoring configuration
///
/// # Errors
///
/// Returns errors from FOSR or FPCA computation.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn profile_phase1(
    y_curves: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &ProfileMonitorConfig,
) -> Result<ProfileChart, FdarError> {
    let (n, m) = y_curves.shape();
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
    if config.window_size < 3 {
        return Err(FdarError::InvalidParameter {
            parameter: "window_size",
            message: format!("window_size must be >= 3, got {}", config.window_size),
        });
    }
    if config.window_size > n {
        return Err(FdarError::InvalidParameter {
            parameter: "window_size",
            message: format!(
                "window_size ({}) exceeds data size ({n})",
                config.window_size
            ),
        });
    }

    // Fit reference FOSR on full data
    let reference_fosr = fosr(y_curves, predictors, config.fosr_lambda)?;

    // Rolling windows: extract per-window FOSR betas
    let beta_vecs = rolling_betas(y_curves, predictors, config)?;
    let n_windows = beta_vecs.nrows();

    if n_windows < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "enough data for at least 4 windows".to_string(),
            actual: format!("{n_windows} windows"),
        });
    }

    // FPCA on vectorized betas
    let ncomp = config.ncomp.min(n_windows - 1).min(beta_vecs.ncols());
    let beta_fpca = fdata_to_pc_1d(&beta_vecs, ncomp)?;
    let actual_ncomp = beta_fpca.scores.ncols();

    // Eigenvalues
    let eigenvalues: Vec<f64> = beta_fpca
        .singular_values
        .iter()
        .take(actual_ncomp)
        .map(|&sv| sv * sv / (n_windows as f64 - 1.0))
        .collect();

    // Control limit
    let t2_limit = t2_control_limit(actual_ncomp, config.alpha)?;

    Ok(ProfileChart {
        reference_fosr,
        beta_fpca,
        eigenvalues,
        t2_limit,
        config: config.clone(),
    })
}

/// Monitor new data against a Phase I profile chart.
///
/// 1. Fits FOSR per rolling window on new data
/// 2. Vectorizes β(t) and projects onto Phase I beta-FPCA
/// 3. Computes T-squared
///
/// # Arguments
/// * `chart` - Phase I profile chart
/// * `new_y` - New response functional data (n × m)
/// * `new_predictors` - New scalar predictors (n × p)
/// * `argvals` - Grid points (length m)
/// * `config` - Profile monitoring configuration
///
/// # Errors
///
/// Returns errors from FOSR or projection.
#[must_use = "monitoring result should not be discarded"]
pub fn profile_monitor(
    chart: &ProfileChart,
    new_y: &FdMatrix,
    new_predictors: &FdMatrix,
    _argvals: &[f64],
    config: &ProfileMonitorConfig,
) -> Result<ProfileMonitorResult, FdarError> {
    let n = new_y.nrows();
    if new_predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "new_predictors",
            expected: format!("{n} rows"),
            actual: format!("{} rows", new_predictors.nrows()),
        });
    }

    // Rolling windows on new data
    let beta_vecs = rolling_betas(new_y, new_predictors, config)?;

    // Project onto Phase I FPCA
    let beta_scores = chart.beta_fpca.project(&beta_vecs)?;

    // T-squared
    let t2 = hotelling_t2(&beta_scores, &chart.eigenvalues)?;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > chart.t2_limit.ucl).collect();

    Ok(ProfileMonitorResult {
        betas: beta_vecs,
        t2,
        t2_alarm,
        beta_scores,
    })
}

/// Extract vectorized FOSR betas from rolling windows.
fn rolling_betas(
    y_curves: &FdMatrix,
    predictors: &FdMatrix,
    config: &ProfileMonitorConfig,
) -> Result<FdMatrix, FdarError> {
    let n = y_curves.nrows();
    let m = y_curves.ncols();
    let p = predictors.ncols();

    let mut windows = Vec::new();
    let mut start = 0;
    while start + config.window_size <= n {
        windows.push(start);
        start += config.step_size;
    }

    if windows.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!(
                "at least {} observations for one window",
                config.window_size
            ),
            actual: format!("{n} observations"),
        });
    }

    // Each beta is p × m, vectorized to length p*m
    let beta_len = p * m;
    let n_windows = windows.len();
    let mut beta_mat = FdMatrix::zeros(n_windows, beta_len);

    for (w_idx, &win_start) in windows.iter().enumerate() {
        // Extract window data
        let mut y_window = FdMatrix::zeros(config.window_size, m);
        let mut pred_window = FdMatrix::zeros(config.window_size, p);
        for i in 0..config.window_size {
            for j in 0..m {
                y_window[(i, j)] = y_curves[(win_start + i, j)];
            }
            for j in 0..p {
                pred_window[(i, j)] = predictors[(win_start + i, j)];
            }
        }

        // Fit FOSR
        let fosr_result = fosr(&y_window, &pred_window, config.fosr_lambda)?;

        // Vectorize beta (p × m) into row of beta_mat
        // fosr_result.beta is p × m where row j = βⱼ(t)
        for j in 0..p {
            for t in 0..m {
                beta_mat[(w_idx, j * m + t)] = fosr_result.beta[(j, t)];
            }
        }
    }

    Ok(beta_mat)
}
