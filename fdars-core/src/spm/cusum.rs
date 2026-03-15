//! CUSUM (Cumulative Sum) monitoring for functional data.
//!
//! Provides both multivariate (Crosier's MCUSUM) and per-component
//! univariate CUSUM charts operating on FPCA scores from an SPM chart.
//! CUSUM charts are designed to detect small sustained shifts quickly,
//! complementing T² (large shifts) and EWMA (small persistent shifts).

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::phase::SpmChart;

/// Configuration for CUSUM monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct CusumConfig {
    /// CUSUM reference value (allowance parameter, default 0.5).
    /// Controls the size of shifts the chart is designed to detect.
    pub k: f64,
    /// CUSUM decision interval (threshold, default 5.0).
    pub h: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level for fallback chi-squared limit (default 0.05).
    pub alpha: f64,
    /// Whether to use multivariate CUSUM (Crosier's MCUSUM) or
    /// per-component univariate CUSUM (default: true = multivariate).
    pub multivariate: bool,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            k: 0.5,
            h: 5.0,
            ncomp: 5,
            alpha: 0.05,
            multivariate: true,
        }
    }
}

/// Result of CUSUM monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CusumMonitorResult {
    /// CUSUM statistics for each observation.
    /// For multivariate: single MCUSUM statistic per obs.
    /// For univariate: max of per-component CUSUMs per obs.
    pub cusum_statistic: Vec<f64>,
    /// Upper threshold (h or chi-squared UCL).
    pub ucl: f64,
    /// Alarm flags.
    pub alarm: Vec<bool>,
    /// Score matrix from FPCA projection (n × ncomp).
    pub scores: FdMatrix,
    /// Per-component CUSUM+ values (n × ncomp), only for univariate mode.
    pub cusum_plus: Option<FdMatrix>,
    /// Per-component CUSUM- values (n × ncomp), only for univariate mode.
    pub cusum_minus: Option<FdMatrix>,
}

/// Validate CUSUM configuration parameters.
fn validate_config(config: &CusumConfig) -> Result<(), FdarError> {
    if config.k <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("k must be positive, got {}", config.k),
        });
    }
    if config.h <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "h",
            message: format!("h must be positive, got {}", config.h),
        });
    }
    Ok(())
}

/// Validate dimensions of sequential data against the chart.
fn validate_dimensions(chart: &SpmChart, sequential_data: &FdMatrix) -> Result<(), FdarError> {
    let m = chart.fpca.mean.len();
    if sequential_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "sequential_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", sequential_data.ncols()),
        });
    }
    Ok(())
}

/// Project data and standardize scores by eigenvalue square roots.
///
/// Returns the truncated, standardized score matrix (n × ncomp) and the
/// actual number of components used.
fn project_and_standardize(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    ncomp_requested: usize,
) -> Result<(FdMatrix, usize), FdarError> {
    let all_scores = chart.fpca.project(sequential_data)?;
    let ncomp = chart.eigenvalues.len().min(ncomp_requested);
    let n = sequential_data.nrows();

    let mut z = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for l in 0..ncomp {
            z[(i, l)] = all_scores[(i, l)] / chart.eigenvalues[l].sqrt();
        }
    }
    Ok((z, ncomp))
}

/// Compute multivariate CUSUM (Crosier's MCUSUM) statistics.
///
/// Returns `(cusum_statistic, alarm)`. When `restart` is true, the
/// cumulative sum vector is reset to zero after each alarm.
fn mcusum_core(z: &FdMatrix, ncomp: usize, k: f64, h: f64, restart: bool) -> (Vec<f64>, Vec<bool>) {
    let n = z.nrows();
    let mut s = vec![0.0; ncomp];
    let mut cusum_statistic = Vec::with_capacity(n);
    let mut alarm = Vec::with_capacity(n);

    for i in 0..n {
        // S_new = S + z_i
        for l in 0..ncomp {
            s[l] += z[(i, l)];
        }

        // C = ||S_new||
        let c: f64 = s.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if c > k {
            // Shrink toward origin by k
            let scale = (c - k) / c;
            for l in 0..ncomp {
                s[l] *= scale;
            }
        } else {
            // Reset to zero
            s.fill(0.0);
        }

        let stat = s.iter().map(|&v| v * v).sum::<f64>().sqrt();
        let is_alarm = stat > h;
        cusum_statistic.push(stat);
        alarm.push(is_alarm);

        if restart && is_alarm {
            s.fill(0.0);
        }
    }

    (cusum_statistic, alarm)
}

/// Compute per-component univariate CUSUM statistics.
///
/// Returns `(cusum_statistic, alarm, cusum_plus_mat, cusum_minus_mat)`.
/// When `restart` is true, the per-component accumulators are reset to zero
/// after each alarm.
fn univariate_cusum_core(
    z: &FdMatrix,
    ncomp: usize,
    k: f64,
    h: f64,
    restart: bool,
) -> (Vec<f64>, Vec<bool>, FdMatrix, FdMatrix) {
    let n = z.nrows();
    let mut cp = vec![0.0; ncomp];
    let mut cm = vec![0.0; ncomp];
    let mut cusum_plus_mat = FdMatrix::zeros(n, ncomp);
    let mut cusum_minus_mat = FdMatrix::zeros(n, ncomp);
    let mut cusum_statistic = Vec::with_capacity(n);
    let mut alarm = Vec::with_capacity(n);

    for i in 0..n {
        for l in 0..ncomp {
            cp[l] = (cp[l] + z[(i, l)] - k).max(0.0);
            cm[l] = (cm[l] - z[(i, l)] - k).max(0.0);
            cusum_plus_mat[(i, l)] = cp[l];
            cusum_minus_mat[(i, l)] = cm[l];
        }

        let mut max_val = 0.0_f64;
        for l in 0..ncomp {
            max_val = max_val.max(cp[l]).max(cm[l]);
        }

        let is_alarm = max_val > h;
        cusum_statistic.push(max_val);
        alarm.push(is_alarm);

        if restart && is_alarm {
            for l in 0..ncomp {
                cp[l] = 0.0;
                cm[l] = 0.0;
            }
        }
    }

    (cusum_statistic, alarm, cusum_plus_mat, cusum_minus_mat)
}

/// Run CUSUM monitoring on sequential functional data.
///
/// 1. Projects each observation through the chart's FPCA
/// 2. Standardizes scores by eigenvalue square roots (unit variance)
/// 3. Computes either multivariate (Crosier's MCUSUM) or per-component
///    univariate CUSUM statistics
/// 4. Flags alarms where the statistic exceeds the threshold `h`
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `_argvals` - Grid points (length m), reserved for future use
/// * `config` - CUSUM configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if data columns do not match the chart.
/// Returns [`FdarError::InvalidParameter`] if `k` or `h` are non-positive.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_cusum_monitor(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    _argvals: &[f64],
    config: &CusumConfig,
) -> Result<CusumMonitorResult, FdarError> {
    validate_config(config)?;
    validate_dimensions(chart, sequential_data)?;

    let (z, ncomp) = project_and_standardize(chart, sequential_data, config.ncomp)?;

    if config.multivariate {
        let (cusum_statistic, alarm) = mcusum_core(&z, ncomp, config.k, config.h, false);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: None,
            cusum_minus: None,
        })
    } else {
        let (cusum_statistic, alarm, cusum_plus_mat, cusum_minus_mat) =
            univariate_cusum_core(&z, ncomp, config.k, config.h, false);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: Some(cusum_plus_mat),
            cusum_minus: Some(cusum_minus_mat),
        })
    }
}

/// Run CUSUM monitoring with automatic restart after each alarm.
///
/// Identical to [`spm_cusum_monitor`] but resets the cumulative sum
/// accumulators to zero after each alarm. This prevents a single large
/// shift from producing a long streak of consecutive alarms.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `_argvals` - Grid points (length m), reserved for future use
/// * `config` - CUSUM configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if data columns do not match the chart.
/// Returns [`FdarError::InvalidParameter`] if `k` or `h` are non-positive.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_cusum_monitor_with_restart(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    _argvals: &[f64],
    config: &CusumConfig,
) -> Result<CusumMonitorResult, FdarError> {
    validate_config(config)?;
    validate_dimensions(chart, sequential_data)?;

    let (z, ncomp) = project_and_standardize(chart, sequential_data, config.ncomp)?;

    if config.multivariate {
        let (cusum_statistic, alarm) = mcusum_core(&z, ncomp, config.k, config.h, true);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: None,
            cusum_minus: None,
        })
    } else {
        let (cusum_statistic, alarm, cusum_plus_mat, cusum_minus_mat) =
            univariate_cusum_core(&z, ncomp, config.k, config.h, true);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: Some(cusum_plus_mat),
            cusum_minus: Some(cusum_minus_mat),
        })
    }
}
