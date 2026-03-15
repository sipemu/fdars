//! Iterative Phase I chart construction for SPM.
//!
//! Repeatedly builds SPM charts and removes out-of-control observations
//! until convergence, producing a cleaner in-control reference dataset.
//! This addresses the common problem of Phase I data contamination
//! where outliers distort the FPCA and control limits.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::phase::{spm_monitor, spm_phase1, SpmChart, SpmConfig};

/// Configuration for iterative Phase I chart construction.
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
    /// Maximum fraction of data that can be removed (default 0.3).
    /// Stops iteration if more data would be removed.
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
    // Validate iterative-specific parameters
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

        // Check if removing these would exceed max_removal_fraction
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
    })
}
