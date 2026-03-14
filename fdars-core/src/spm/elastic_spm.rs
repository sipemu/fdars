//! Phase-aware (elastic) SPM monitoring.
//!
//! Separates amplitude and phase variation using elastic alignment,
//! then monitors each component independently. This prevents phase
//! variation from masking amplitude shifts and vice versa.

use crate::alignment::{align_to_target, karcher_mean};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::phase::{spm_monitor, spm_phase1, SpmChart, SpmConfig, SpmMonitorResult};

/// Configuration for elastic SPM.
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticSpmConfig {
    /// Base SPM configuration.
    pub spm: SpmConfig,
    /// Alignment regularization parameter (default 0.0).
    pub align_lambda: f64,
    /// Whether to also monitor phase variation (default true).
    pub monitor_phase: bool,
    /// Number of PCs for warping function FPCA (default 3).
    pub warp_ncomp: usize,
}

impl Default for ElasticSpmConfig {
    fn default() -> Self {
        Self {
            spm: SpmConfig::default(),
            align_lambda: 0.0,
            monitor_phase: true,
            warp_ncomp: 3,
        }
    }
}

/// Phase I elastic SPM chart.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticSpmChart {
    /// Karcher mean of the training data.
    pub karcher_mean: Vec<f64>,
    /// SPM chart for amplitude (aligned data).
    pub amplitude_chart: SpmChart,
    /// SPM chart for phase (warping functions), if monitoring phase.
    pub phase_chart: Option<SpmChart>,
    /// Configuration used.
    pub config: ElasticSpmConfig,
}

/// Result of Phase II elastic SPM monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticSpmMonitorResult {
    /// Amplitude monitoring result.
    pub amplitude: SpmMonitorResult,
    /// Phase monitoring result (if monitoring phase).
    pub phase: Option<SpmMonitorResult>,
    /// Aligned new data.
    pub aligned_data: FdMatrix,
    /// Warping functions for new data.
    pub warping_functions: FdMatrix,
}

/// Build a Phase I elastic SPM chart.
///
/// 1. Computes the Karcher mean of the training data
/// 2. Aligns all observations to the Karcher mean
/// 3. Builds an SPM chart on the aligned data (amplitude)
/// 4. Optionally builds an SPM chart on the warping functions (phase)
///
/// # Arguments
/// * `data` - In-control functional data (n x m)
/// * `argvals` - Grid points (length m)
/// * `config` - Elastic SPM configuration
///
/// # Errors
///
/// Returns errors from alignment or SPM chart construction.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_spm_phase1(
    data: &FdMatrix,
    argvals: &[f64],
    config: &ElasticSpmConfig,
) -> Result<ElasticSpmChart, FdarError> {
    let (n, m) = data.shape();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 4 observations".to_string(),
            actual: format!("{n} observations"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    // Step 1: Karcher mean
    let km_result = karcher_mean(data, argvals, 20, 1e-4, config.align_lambda);

    // Step 2: Align all to Karcher mean
    let alignment = align_to_target(data, &km_result.mean, argvals, config.align_lambda);

    // Step 3: SPM on aligned data (amplitude)
    let amplitude_chart = spm_phase1(&alignment.aligned_data, argvals, &config.spm)?;

    // Step 4: Optionally SPM on warping functions (phase)
    let phase_chart = if config.monitor_phase {
        let phase_config = SpmConfig {
            ncomp: config.warp_ncomp,
            alpha: config.spm.alpha,
            tuning_fraction: config.spm.tuning_fraction,
            seed: config.spm.seed.wrapping_add(1),
        };
        Some(spm_phase1(&alignment.gammas, argvals, &phase_config)?)
    } else {
        None
    };

    Ok(ElasticSpmChart {
        karcher_mean: km_result.mean,
        amplitude_chart,
        phase_chart,
        config: config.clone(),
    })
}

/// Monitor new data against a Phase I elastic SPM chart.
///
/// 1. Aligns new observations to the stored Karcher mean
/// 2. Monitors aligned data through the amplitude chart
/// 3. Optionally monitors warping functions through the phase chart
///
/// # Arguments
/// * `chart` - Phase I elastic SPM chart
/// * `new_data` - New functional observations (n_new x m)
/// * `argvals` - Grid points (length m)
///
/// # Errors
///
/// Returns errors from alignment or monitoring.
#[must_use = "monitoring result should not be discarded"]
pub fn elastic_spm_monitor(
    chart: &ElasticSpmChart,
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<ElasticSpmMonitorResult, FdarError> {
    let m = chart.karcher_mean.len();
    if new_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", new_data.ncols()),
        });
    }

    // Align new data to Karcher mean
    let alignment = align_to_target(
        new_data,
        &chart.karcher_mean,
        argvals,
        chart.config.align_lambda,
    );

    // Monitor amplitude
    let amplitude = spm_monitor(&chart.amplitude_chart, &alignment.aligned_data, argvals)?;

    // Optionally monitor phase
    let phase = if let Some(ref phase_chart) = chart.phase_chart {
        Some(spm_monitor(phase_chart, &alignment.gammas, argvals)?)
    } else {
        None
    };

    Ok(ElasticSpmMonitorResult {
        amplitude,
        phase,
        aligned_data: alignment.aligned_data,
        warping_functions: alignment.gammas,
    })
}
