//! Adaptive EWMA (AMFEWMA) monitoring for functional data.
//!
//! Extends standard EWMA by dynamically adjusting the smoothing parameter
//! λ_t based on the magnitude of observed deviations (Sparks, 2000). This
//! provides robustness across unknown shift sizes: small λ for persistent
//! small shifts, large λ for sudden large shifts.
//!
//! Key implementation details:
//! - Causal adaptation: λ_t is determined from data up to time t-1, applied
//!   at time t (avoids look-ahead bias).
//! - Time-dependent covariance: the T² statistic uses the exact cumulative
//!   variance of the EWMA with time-varying weights, rather than the
//!   asymptotic approximation.
//!
//! # References
//!
//! - Sparks, R.S. (2000). CUSUM charts for signalling varying location
//!   shifts, Eqs. 3--5, pp. 162--164. *Journal of Quality Technology*,
//!   32(2), 157--171.
//! - Capizzi, G. & Masarotto, G. (2003). An adaptive exponentially weighted
//!   moving average control chart, §2, pp. 200--202. *Technometrics*,
//!   45(3), 199--207.
//!
//! # Parameter guidance
//!
//! - `eta`: Controls adaptation speed. Small η (0.05-0.1) for gradual adaptation,
//!   large η (0.3-0.5) for fast reaction. Default 0.1 is conservative.
//! - `error_clip`: Recommended when data may contain outliers. A value of 9.0
//!   corresponds to 3σ clipping for standardized scores.
//! - `lambda_min`/`lambda_max`: Typical range [0.05, 0.95]. Narrow ranges reduce
//!   adaptivity but improve stability.

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
    ///
    /// Controls adaptation speed. Small eta (0.05--0.1) provides gradual
    /// adaptation suitable for slowly drifting processes; large eta (0.3--0.5)
    /// reacts quickly to abrupt shifts but may over-adapt to noise. Default
    /// 0.1 balances responsiveness and stability.
    pub eta: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Optional upper bound for the squared error magnitude (default None).
    /// When set, clips e²_avg to this value to reduce sensitivity to outlier
    /// observations. A value of 9.0 corresponds to 3σ clipping for unit-variance
    /// standardized scores.
    pub error_clip: Option<f64>,
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
            error_clip: None,
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
/// * `argvals` - Grid points (length m)
/// * `config` - Adaptive EWMA configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `sequential_data` column count
/// does not match the chart, or [`FdarError::InvalidParameter`] if any
/// configuration parameter is out of range.
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, SpmConfig};
/// use fdars_core::spm::amewma::{spm_amewma_monitor, AmewmaConfig};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let chart = spm_phase1(&data, &argvals, &SpmConfig { ncomp: 2, ..SpmConfig::default() }).unwrap();
/// let new_data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10
/// ).unwrap();
/// let result = spm_amewma_monitor(&chart, &new_data, &argvals, &AmewmaConfig::default()).unwrap();
/// assert_eq!(result.lambda_t.len(), 5);
/// ```
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
    if let Some(clip) = config.error_clip {
        if clip <= 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "error_clip",
                message: format!("error_clip must be positive, got {clip}"),
            });
        }
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
    // Sparks (2000) variant: adaptive lambda with time-dependent covariance.
    //
    // The adaptive weight estimator Q_t = eta * e^2_avg + (1 - eta) * Q_{t-1}
    // tracks the second moment of the standardized prediction error.
    // lambda_t = clamp(Q_t, lambda_min, lambda_max) maps this to a smoothing
    // parameter: large Q -> large lambda -> fast response to large shifts.
    let mut smoothed = FdMatrix::zeros(n, actual_ncomp);
    let mut t2_statistic = Vec::with_capacity(n);
    let mut lambda_vals = Vec::with_capacity(n);

    // Q_0 initialized from lambda_init to set the initial adaptive state
    let mut q_prev = config.lambda_init;
    let mut lambda_prev = config.lambda_init;
    // Cumulative variance factor for time-dependent covariance:
    // Var(S_t) = sum_{j=1}^{t} lambda_j^2 * prod_{k=j+1}^{t} (1-lambda_k)^2
    // This generalizes the constant-lambda EWMA variance
    // lambda/(2-lambda)[1-(1-lambda)^{2t}] to the time-varying case.
    let mut sum_lambda_sq_prod = 0.0_f64;

    for i in 0..n {
        // Step 1: Apply EWMA with PREVIOUS lambda (proper causal ordering)
        let lambda_cur = lambda_prev;

        // EWMA with current lambda: S_t = lambda_t * z_i + (1 - lambda_t) * S_{t-1}
        for l in 0..actual_ncomp {
            let s_prev = if i > 0 { smoothed[(i - 1, l)] } else { 0.0 };
            smoothed[(i, l)] = lambda_cur * z[(i, l)] + (1.0 - lambda_cur) * s_prev;
        }

        // Time-dependent covariance correction:
        // Var(S_t) = sum_{j=1}^{t} lambda_j^2 * prod_{k=j+1}^{t} (1-lambda_k)^2
        // For diagonal covariance with standardized scores, this simplifies to
        // a scalar factor times identity.
        sum_lambda_sq_prod =
            lambda_cur * lambda_cur + (1.0 - lambda_cur).powi(2) * sum_lambda_sq_prod;
        // Use the time-dependent variance factor for the T² statistic
        let var_factor = sum_lambda_sq_prod.max(1e-15);

        // T² statistic = S_t^T * (var_factor * I)^{-1} * S_t = sum(S_t[l]^2) / var_factor
        let mut t2 = 0.0;
        for l in 0..actual_ncomp {
            t2 += smoothed[(i, l)] * smoothed[(i, l)] / var_factor;
        }
        t2_statistic.push(t2);
        lambda_vals.push(lambda_cur);

        // Step 2: NOW update adaptive weight for the NEXT time step
        // Compute prediction error: e_t = z_i (standardized scores)
        let mut e_sq_sum = 0.0;
        for l in 0..actual_ncomp {
            e_sq_sum += z[(i, l)] * z[(i, l)];
        }
        let e_sq_avg = e_sq_sum / actual_ncomp as f64;
        let e_sq_avg = if let Some(clip) = config.error_clip {
            e_sq_avg.min(clip)
        } else {
            e_sq_avg
        };

        // Update adaptive weight estimator
        let q_t = config.eta * e_sq_avg + (1.0 - config.eta) * q_prev;
        lambda_prev = config.lambda_min.max(config.lambda_max.min(q_t));
        // Clamp Q_t to [0, 2 * lambda_max] to prevent unbounded growth from
        // outlier sequences. The factor of 2 allows Q to exceed lambda_max
        // temporarily (since the clamping to [lambda_min, lambda_max] happens
        // at the lambda mapping step), providing a form of memory for recent
        // large deviations.
        q_prev = q_t.clamp(0.0, config.lambda_max * 2.0);
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
