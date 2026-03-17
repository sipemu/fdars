//! CUSUM (Cumulative Sum) monitoring for functional data.
//!
//! Provides both multivariate (Crosier's MCUSUM) and per-component
//! univariate CUSUM charts operating on FPCA scores from an SPM chart.
//! CUSUM charts are designed to detect small sustained shifts quickly,
//! complementing T² (large shifts) and EWMA (small persistent shifts).
//!
//! CUSUM is optimal for detecting sustained shifts of known magnitude
//! delta (set k ~ delta/2). For unknown shift sizes, consider adaptive
//! EWMA ([`super::amewma::spm_amewma_monitor`]) which automatically
//! adjusts sensitivity.
//!
//! # References
//!
//! - Page, E.S. (1954). Continuous inspection schemes. *Biometrika*,
//!   41(1-2), 100-115.
//! - Crosier, R.B. (1988). Multivariate generalizations of cumulative sum
//!   quality-control schemes. *Technometrics*, 30(3), 291-303.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::phase::{center_data, centered_reconstruct, SpmChart};
use super::stats::spe_univariate;

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
    /// Whether to restart the CUSUM accumulator after each alarm (default false).
    /// When true, the accumulator resets to zero after crossing the threshold,
    /// which can improve sensitivity for detecting subsequent shifts.
    pub restart: bool,
}

impl Default for CusumConfig {
    fn default() -> Self {
        Self {
            k: 0.5,
            h: 5.0,
            ncomp: 5,
            alpha: 0.05,
            multivariate: true,
            restart: false,
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
    /// Score matrix from FPCA projection (n × ncomp, standardized).
    pub scores: FdMatrix,
    /// Per-component CUSUM+ values (n × ncomp), only for univariate mode.
    pub cusum_plus: Option<FdMatrix>,
    /// Per-component CUSUM- values (n × ncomp), only for univariate mode.
    pub cusum_minus: Option<FdMatrix>,
    /// SPE values (reconstruction error).
    pub spe: Vec<f64>,
    /// SPE control limit.
    pub spe_limit: f64,
    /// SPE alarm flags.
    pub spe_alarm: Vec<bool>,
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
/// # Parameter guidance
///
/// Common (k, h) pairs for Hotelling T² monitoring and their approximate
/// ARL₀ (Crosier, 1988):
/// - k=0.5, h=5.0: ARL₀ ≈ 370 (standard choice for detecting 1σ shifts)
/// - k=0.25, h=8.0: ARL₀ ≈ 370 (better for smaller shifts)
/// - k=1.0, h=4.0: ARL₀ ≈ 370 (faster response to larger shifts)
///
/// Edge cases: k = 0 makes the CUSUM equivalent to a cumulative sum (high
/// sensitivity but high false alarm rate). Very large h reduces false
/// alarms but delays detection. Typical starting point: k = 0.5 (detects
/// 1-sigma shifts), h = 4-5.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `argvals` - Grid points (length m), reserved for future use
/// * `config` - CUSUM configuration
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, SpmConfig};
/// use fdars_core::spm::cusum::{spm_cusum_monitor, CusumConfig};
/// // Build tiny Phase I chart
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let chart = spm_phase1(&data, &argvals, &SpmConfig { ncomp: 2, ..SpmConfig::default() }).unwrap();
/// let new_data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10
/// ).unwrap();
/// let result = spm_cusum_monitor(&chart, &new_data, &argvals, &CusumConfig::default()).unwrap();
/// assert_eq!(result.cusum_statistic.len(), 5);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if data columns do not match the chart.
/// Returns [`FdarError::InvalidParameter`] if `k` or `h` are non-positive.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_cusum_monitor(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    argvals: &[f64],
    config: &CusumConfig,
) -> Result<CusumMonitorResult, FdarError> {
    validate_config(config)?;
    validate_dimensions(chart, sequential_data)?;

    let (z, ncomp) = project_and_standardize(chart, sequential_data, config.ncomp)?;

    // SPE from reconstruction error (un-standardize scores for reconstruction)
    let n = sequential_data.nrows();
    let mut raw_scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for l in 0..ncomp {
            raw_scores[(i, l)] = z[(i, l)] * chart.eigenvalues[l].sqrt();
        }
    }
    let centered = center_data(sequential_data, &chart.fpca.mean);
    let recon_centered = centered_reconstruct(&chart.fpca, &raw_scores, ncomp);
    let spe = spe_univariate(&centered, &recon_centered, argvals)?;
    let spe_limit = chart.spe_limit.ucl;
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > spe_limit).collect();

    if config.multivariate {
        let (cusum_statistic, alarm) = mcusum_core(&z, ncomp, config.k, config.h, config.restart);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: None,
            cusum_minus: None,
            spe,
            spe_limit,
            spe_alarm,
        })
    } else {
        let (cusum_statistic, alarm, cusum_plus_mat, cusum_minus_mat) =
            univariate_cusum_core(&z, ncomp, config.k, config.h, config.restart);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: Some(cusum_plus_mat),
            cusum_minus: Some(cusum_minus_mat),
            spe,
            spe_limit,
            spe_alarm,
        })
    }
}

/// Run CUSUM monitoring with automatic restart after each alarm.
///
/// Identical to [`spm_cusum_monitor`] but resets the cumulative sum
/// accumulators to zero after each alarm. This prevents a single large
/// shift from producing a long streak of consecutive alarms.
///
/// After an alarm, the CUSUM accumulator resets to zero, making the chart
/// sensitive to subsequent shifts. Without restart, the accumulator remains
/// elevated after the first alarm, potentially masking new events. Use
/// restart when monitoring for intermittent faults.
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `argvals` - Grid points (length m), reserved for future use
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
    argvals: &[f64],
    config: &CusumConfig,
) -> Result<CusumMonitorResult, FdarError> {
    validate_config(config)?;
    validate_dimensions(chart, sequential_data)?;

    let (z, ncomp) = project_and_standardize(chart, sequential_data, config.ncomp)?;

    // SPE from reconstruction error
    let n = sequential_data.nrows();
    let mut raw_scores = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for l in 0..ncomp {
            raw_scores[(i, l)] = z[(i, l)] * chart.eigenvalues[l].sqrt();
        }
    }
    let centered = center_data(sequential_data, &chart.fpca.mean);
    let recon_centered = centered_reconstruct(&chart.fpca, &raw_scores, ncomp);
    let spe = spe_univariate(&centered, &recon_centered, argvals)?;
    let spe_limit = chart.spe_limit.ucl;
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > spe_limit).collect();

    if config.multivariate {
        let (cusum_statistic, alarm) = mcusum_core(&z, ncomp, config.k, config.h, true);
        Ok(CusumMonitorResult {
            cusum_statistic,
            ucl: config.h,
            alarm,
            scores: z,
            cusum_plus: None,
            cusum_minus: None,
            spe,
            spe_limit,
            spe_alarm,
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
            spe,
            spe_limit,
            spe_alarm,
        })
    }
}
