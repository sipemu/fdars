//! Adaptive EWMA (AMFEWMA) monitoring for functional data.
//!
//! Extends standard EWMA by dynamically adjusting the smoothing parameter
//! λ_t based on the magnitude of observed deviations. This provides
//! robustness across unknown shift sizes: small λ for persistent small shifts,
//! large λ for sudden large shifts.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::chi_squared::chi2_quantile;
use super::phase::SpmChart;

/// Configuration for adaptive EWMA (AMFEWMA) monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct AmewmaConfig {
    /// Minimum EWMA smoothing parameter (default 0.05).
    pub lambda_min: f64,
    /// Maximum EWMA smoothing parameter (default 0.95).
    pub lambda_max: f64,
    /// Initial smoothing parameter (default 0.2).
    pub lambda_init: f64,
    /// Smoothing parameter for the adaptive weight estimator (default 0.1).
    /// Controls how quickly the adaptive weight reacts to changes.
    pub eta: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
}

impl Default for AmewmaConfig {
    fn default() -> Self {
        Self {
            lambda_min: 0.05,
            lambda_max: 0.95,
            lambda_init: 0.2,
            eta: 0.1,
            ncomp: 5,
            alpha: 0.05,
        }
    }
}

/// Result of adaptive EWMA monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AmewmaMonitorResult {
    /// Adaptively-smoothed score matrix (n × ncomp).
    pub smoothed_scores: FdMatrix,
    /// Adaptive EWMA T² statistic for each observation.
    pub t2_statistic: Vec<f64>,
    /// Adaptive lambda values used at each time step.
    pub lambda_t: Vec<f64>,
    /// Upper control limit.
    pub ucl: f64,
    /// Alarm flags.
    pub alarm: Vec<bool>,
}

/// Run adaptive EWMA (AMFEWMA) monitoring on sequential functional data.
///
/// 1. Projects each observation through the chart's FPCA
/// 2. Standardizes scores by dividing by sqrt(eigenvalue)
/// 3. Adaptively adjusts the EWMA smoothing parameter λ_t based on
///    the magnitude of the prediction error signal
/// 4. Computes a T²-type statistic using the adaptive covariance
/// 5. Sets the UCL from the chi-squared distribution
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `_argvals` - Grid points (length m), reserved for future use
/// * `config` - Adaptive EWMA configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `sequential_data` column count
/// does not match the chart, or [`FdarError::InvalidParameter`] if any
/// configuration parameter is out of range.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_amewma_monitor(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    _argvals: &[f64],
    config: &AmewmaConfig,
) -> Result<AmewmaMonitorResult, FdarError> {
    // --- Validation ---
    let m = chart.fpca.mean.len();
    if sequential_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "sequential_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", sequential_data.ncols()),
        });
    }
    if config.lambda_min <= 0.0 || config.lambda_min > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda_min",
            message: format!("lambda_min must be in (0, 1], got {}", config.lambda_min),
        });
    }
    if config.lambda_max <= 0.0 || config.lambda_max > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda_max",
            message: format!("lambda_max must be in (0, 1], got {}", config.lambda_max),
        });
    }
    if config.lambda_min > config.lambda_max {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda_min",
            message: format!(
                "lambda_min ({}) must be <= lambda_max ({})",
                config.lambda_min, config.lambda_max
            ),
        });
    }
    if config.lambda_init < config.lambda_min || config.lambda_init > config.lambda_max {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda_init",
            message: format!(
                "lambda_init ({}) must be in [lambda_min, lambda_max] = [{}, {}]",
                config.lambda_init, config.lambda_min, config.lambda_max
            ),
        });
    }
    if config.eta <= 0.0 || config.eta > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "eta",
            message: format!("eta must be in (0, 1], got {}", config.eta),
        });
    }

    let ncomp = chart.eigenvalues.len().min(config.ncomp);
    let n = sequential_data.nrows();

    // --- Project all observations ---
    let all_scores = chart.fpca.project(sequential_data)?;

    // Truncate to ncomp columns if needed
    let scores = if all_scores.ncols() > ncomp {
        let mut trunc = FdMatrix::zeros(n, ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                trunc[(i, k)] = all_scores[(i, k)];
            }
        }
        trunc
    } else {
        all_scores
    };
    let actual_ncomp = scores.ncols();

    // --- Standardize: z[i,l] = score[i,l] / sqrt(eigenvalue[l]) ---
    let mut z = FdMatrix::zeros(n, actual_ncomp);
    for i in 0..n {
        for l in 0..actual_ncomp {
            let ev = chart.eigenvalues[l];
            z[(i, l)] = if ev > 0.0 {
                scores[(i, l)] / ev.sqrt()
            } else {
                0.0
            };
        }
    }

    // --- Adaptive EWMA loop ---
    let mut smoothed = FdMatrix::zeros(n, actual_ncomp);
    let mut t2_statistic = Vec::with_capacity(n);
    let mut lambda_vals = Vec::with_capacity(n);

    // Q_0 initialized from lambda_init to set the initial adaptive state
    let mut q_prev = config.lambda_init;

    for i in 0..n {
        // Compute prediction error: e_t = z_i (standardized scores)
        // Average squared prediction error: e_t^T e_t / ncomp
        let mut e_sq_sum = 0.0;
        for l in 0..actual_ncomp {
            e_sq_sum += z[(i, l)] * z[(i, l)];
        }
        let e_sq_avg = e_sq_sum / actual_ncomp as f64;

        // Update adaptive weight estimator
        let q_t = config.eta * e_sq_avg + (1.0 - config.eta) * q_prev;
        let lambda_cur = config.lambda_min.max(config.lambda_max.min(q_t));
        q_prev = q_t;

        // EWMA with adaptive lambda: S_t = lambda_t * z_i + (1 - lambda_t) * S_{t-1}
        for l in 0..actual_ncomp {
            let s_prev = if i > 0 { smoothed[(i - 1, l)] } else { 0.0 };
            smoothed[(i, l)] = lambda_cur * z[(i, l)] + (1.0 - lambda_cur) * s_prev;
        }

        // T² statistic using approximate asymptotic covariance:
        // Σ_approx = (lambda_t / (2 - lambda_t)) * I
        // T²_t = S_t^T * Σ_approx^{-1} * S_t = sum(S_t[l]^2 * (2 - lambda_t) / lambda_t)
        let scale = (2.0 - lambda_cur) / lambda_cur;
        let mut t2 = 0.0;
        for l in 0..actual_ncomp {
            t2 += smoothed[(i, l)] * smoothed[(i, l)] * scale;
        }
        t2_statistic.push(t2);
        lambda_vals.push(lambda_cur);
    }

    // --- UCL from chi-squared ---
    let ucl = chi2_quantile(1.0 - config.alpha, actual_ncomp);

    // --- Alarms ---
    let alarm: Vec<bool> = t2_statistic.iter().map(|&v| v > ucl).collect();

    Ok(AmewmaMonitorResult {
        smoothed_scores: smoothed,
        t2_statistic,
        lambda_t: lambda_vals,
        ucl,
        alarm,
    })
}
