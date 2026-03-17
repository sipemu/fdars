//! Iterative Phase I chart construction for SPM.
//!
//! Repeatedly builds SPM charts and removes out-of-control observations
//! until convergence, producing a cleaner in-control reference dataset.
//! This addresses the common problem of Phase I data contamination
//! where outliers distort the FPCA and control limits.
//!
//! # Convergence properties
//!
//! The iterative Phase I procedure converges when no new outliers are removed
//! between iterations. Convergence is guaranteed in at most n iterations (each
//! iteration removes at least one outlier or terminates). In practice, 3--5
//! iterations suffice for typical contamination levels (5--15% outliers).
//! Non-convergence (oscillation) can occur when the contamination fraction
//! is near the breakdown point of the underlying T-squared / SPE statistics.
//!
//! # Breakdown point
//!
//! The procedure's breakdown point depends on the initial T-squared threshold.
//! With alpha = 0.05 and chi-squared limits, the expected breakdown is roughly
//! 50% for the T-squared statistic (Rousseeuw & Leroy, 1987, section 1.3,
//! pp. 10--12). For contamination above the breakdown point, consider robust
//! initialization via projection pursuit or minimum covariance determinant
//! (MCD) before applying the iterative procedure.
//!
//! # References
//!
//! - Sullivan, J.H. & Woodall, W.H. (1996). A comparison of multivariate
//!   control charts for individual observations. *Journal of Quality
//!   Technology*, 28(4), 398--408, section 3 (iterative Phase I procedure).
//! - Chenouri, S., Steiner, S.H. & Variyath, A.M. (2009). A multivariate
//!   robust control chart for individual observations. *Journal of Quality
//!   Technology*, 41(3), 259--271, section 2 (robust alternatives).
//! - Rousseeuw, P.J. & Leroy, A.M. (1987). *Robust Regression and Outlier
//!   Detection*. Wiley, section 1.3, pp. 10--12 (breakdown point),
//!   section 4.1, pp. 116--119 (iterative reweighting).

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::phase::{spm_monitor, spm_phase1, SpmChart, SpmConfig};

/// Configuration for iterative Phase I chart construction.
///
/// The iterative approach assumes outliers are a minority of the data.
/// When more than `max_removal_fraction` of the original data would be
/// removed, the procedure stops early, preserving the remaining data
/// for analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct IterativePhase1Config {
    /// Base SPM configuration.
    pub spm: SpmConfig,
    /// Maximum number of iterations (default 10).
    pub max_iterations: usize,
    /// Remove observations exceeding the T-squared limit (default true).
    pub remove_t2_outliers: bool,
    /// Remove observations exceeding the SPE limit (default true).
    pub remove_spe_outliers: bool,
    /// Maximum cumulative fraction of original data that can be removed (default 0.3).
    /// Iteration stops if the next removal batch would push the total removed
    /// count above this fraction of the original dataset size.
    ///
    /// This acts as a safeguard against breakdown: if more than 30% of the data
    /// is flagged, the in-control model is likely misspecified rather than there
    /// being isolated outliers (Rousseeuw & Leroy, 1987, section 4.1, pp. 116--119).
    ///
    /// If removal rates don't decrease across iterations (e.g., oscillating
    /// around 0.3--0.5), the process likely has sustained non-stationarity
    /// rather than isolated outliers. Consider increasing `alpha` or
    /// investigating the data for structural changes.
    pub max_removal_fraction: f64,
}

impl Default for IterativePhase1Config {
    fn default() -> Self {
        Self {
            spm: SpmConfig::default(),
            max_iterations: 10,
            remove_t2_outliers: true,
            remove_spe_outliers: true,
            max_removal_fraction: 0.3,
        }
    }
}

/// Result of iterative Phase I chart construction.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct IterativePhase1Result {
    /// Final SPM chart after outlier removal.
    pub chart: SpmChart,
    /// Number of iterations performed.
    pub n_iterations: usize,
    /// Indices of removed observations (relative to original data).
    pub removed_indices: Vec<usize>,
    /// Number of observations remaining.
    pub n_remaining: usize,
    /// History of observations removed per iteration.
    pub removal_history: Vec<Vec<usize>>,
    /// Fraction of observations removed per iteration (convergence diagnostic).
    /// A decreasing sequence indicates convergence. Rates > 0.5 at any
    /// iteration suggest the control limits may be too tight or the process
    /// is genuinely unstable.
    pub removal_rates: Vec<f64>,
}

/// Iteratively build a Phase I SPM chart by removing out-of-control observations.
///
/// Standard Phase I (`spm_phase1`) builds the chart once. However, if the training
/// data contains outliers, the chart may be contaminated. This function repeatedly:
///
/// 1. Builds a chart from the current clean data
/// 2. Monitors all current data against the chart
/// 3. Removes observations flagged as out-of-control
/// 4. Repeats until no more observations are removed or the maximum number of
///    iterations is reached
///
/// # Arguments
/// * `data` - In-control functional data (n x m)
/// * `argvals` - Grid points (length m)
/// * `config` - Iterative Phase I configuration
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::iterative::{spm_phase1_iterative, IterativePhase1Config};
/// use fdars_core::spm::phase::SpmConfig;
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = IterativePhase1Config {
///     spm: SpmConfig { ncomp: 2, ..SpmConfig::default() },
///     ..IterativePhase1Config::default()
/// };
/// let result = spm_phase1_iterative(&data, &argvals, &config).unwrap();
/// assert!(result.n_iterations <= config.max_iterations);
/// ```
///
/// # Errors
///
/// Returns `FdarError::InvalidParameter` if `max_iterations < 1` or
/// `max_removal_fraction` is not in (0, 1]. Dimension errors are propagated
/// from `spm_phase1`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn spm_phase1_iterative(
    data: &FdMatrix,
    argvals: &[f64],
    config: &IterativePhase1Config,
) -> Result<IterativePhase1Result, FdarError> {
    // Validate iterative-specific parameters.
    // alpha must be in (0, 1). Smaller alpha values (e.g., 0.01) produce wider
    // control limits and remove fewer observations per iteration, yielding a more
    // conservative procedure. Larger alpha (e.g., 0.10) is more aggressive and
    // converges faster but risks removing in-control observations (masking).
    // The default alpha = 0.05 balances sensitivity and specificity for typical
    // contamination levels (5--15%).
    if config.spm.alpha <= 0.0 || config.spm.alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in (0, 1), got {}", config.spm.alpha),
        });
    }
    if config.max_iterations < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iterations",
            message: format!(
                "max_iterations must be at least 1, got {}",
                config.max_iterations
            ),
        });
    }
    if config.max_removal_fraction <= 0.0 || config.max_removal_fraction > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_removal_fraction",
            message: format!(
                "max_removal_fraction must be in (0, 1], got {}",
                config.max_removal_fraction
            ),
        });
    }

    let n_original = data.nrows();
    let mut remaining_indices: Vec<usize> = (0..n_original).collect();
    let mut all_removed: Vec<usize> = vec![];
    let mut removal_history: Vec<Vec<usize>> = vec![];
    let mut removal_rates: Vec<f64> = vec![];

    let mut chart = None;

    for _ in 0..config.max_iterations {
        // Build chart from current data
        let current_data = crate::cv::subset_rows(data, &remaining_indices);
        let current_chart = spm_phase1(&current_data, argvals, &config.spm)?;

        // Monitor the same data against the chart
        let monitor = spm_monitor(&current_chart, &current_data, argvals)?;

        // Identify out-of-control observations
        let n_current = remaining_indices.len();
        let mut flagged_local: Vec<usize> = Vec::new();
        for i in 0..n_current {
            let is_flagged = (config.remove_t2_outliers && monitor.t2_alarm[i])
                || (config.remove_spe_outliers && monitor.spe_alarm[i]);
            if is_flagged {
                flagged_local.push(i);
            }
        }

        // Converged: no observations flagged
        if flagged_local.is_empty() {
            chart = Some(current_chart);
            break;
        }

        // Check cumulative removal: total removed so far (including this batch)
        // against the maximum allowed fraction of the ORIGINAL dataset.
        // The 0.5 removal rate threshold is a practical heuristic: if more
        // than half the remaining data is flagged in one iteration, the
        // in-control model is likely misspecified rather than there being
        // individual outliers. This aligns with the breakdown point of
        // classical outlier detection methods (Rousseeuw & Leroy, 1987).
        let total_removed = all_removed.len() + flagged_local.len();
        if total_removed as f64 / n_original as f64 > config.max_removal_fraction {
            chart = Some(current_chart);
            break;
        }

        // Check if remaining after removal would be too few
        let n_after = n_current - flagged_local.len();
        if n_after < 4 {
            chart = Some(current_chart);
            break;
        }

        // Map flagged local indices back to original indices
        let flagged_original: Vec<usize> = flagged_local
            .iter()
            .map(|&i| remaining_indices[i])
            .collect();

        // Update remaining_indices by removing flagged ones
        let flagged_set: std::collections::HashSet<usize> = flagged_local.iter().copied().collect();
        remaining_indices = remaining_indices
            .iter()
            .enumerate()
            .filter(|(local_i, _)| !flagged_set.contains(local_i))
            .map(|(_, &orig_i)| orig_i)
            .collect();

        let removal_rate = flagged_original.len() as f64 / n_current as f64;
        removal_rates.push(removal_rate);
        all_removed.extend_from_slice(&flagged_original);
        removal_history.push(flagged_original);
    }

    // Build the final chart if we exhausted iterations without converging
    let final_chart = match chart {
        Some(c) => c,
        None => {
            let final_data = crate::cv::subset_rows(data, &remaining_indices);
            spm_phase1(&final_data, argvals, &config.spm)?
        }
    };

    Ok(IterativePhase1Result {
        chart: final_chart,
        n_iterations: removal_history.len(),
        removed_indices: all_removed,
        n_remaining: remaining_indices.len(),
        removal_history,
        removal_rates,
    })
}
