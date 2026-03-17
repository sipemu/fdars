//! Phase-aware (elastic) SPM monitoring.
//!
//! Separates amplitude and phase variation using elastic alignment,
//! then monitors each component independently. This prevents phase
//! variation from masking amplitude shifts and vice versa.
//!
//! The amplitude and phase decomposition assumes the SRSF (Square Root
//! Slope Function) representation. This requires the input functions to
//! be absolutely continuous. Highly discontinuous or noisy data should be
//! smoothed before elastic alignment.
//!
//! Elastic SPM is most beneficial when significant phase variation exists
//! (e.g., time-warped peaks, shifted features). If phase variation is
//! negligible, standard `spm_phase1` on unaligned data is simpler and
//! equally effective.
//!
//! To assess whether elastic SPM is warranted, compare ARL0 or alarm rates
//! between `elastic_spm_phase1` and standard `spm_phase1` on the same data.
//! If phase variation is negligible, both will perform similarly.
//!
//! # References
//!
//! - Srivastava, A., Wu, W., Kurtek, S., Klassen, E. & Marron, J.S. (2011).
//!   Registration of functional data using Fisher-Rao metric. arXiv:1103.3817.
//! - Tucker, J.D., Wu, W. & Srivastava, A. (2013). Generative models for
//!   functional data using phase and amplitude separation. *Computational
//!   Statistics & Data Analysis*, 61, 50-66.
//!
//! # Design notes
//!
//! - Amplitude and phase charts are monitored independently. If amplitude and
//!   phase variation are correlated in your process, consider using a joint
//!   monitoring approach or adjusting significance levels for multiple testing.
//! - `align_lambda`: Controls alignment regularization. 0.0 (default) gives pure
//!   elastic alignment; increase toward 1.0 to penalize warping and preserve
//!   more amplitude structure.

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
    ///
    /// Warping functions are typically lower-dimensional than amplitude,
    /// so fewer components suffice. Default 3 captures the dominant modes
    /// of phase variation.
    pub warp_ncomp: usize,
    /// Maximum iterations for Karcher mean computation (default 20).
    ///
    /// Convergence is checked at `karcher_tolerance` (relative SRSF change).
    /// Increase for highly variable data or if alignment residuals are large.
    pub max_karcher_iterations: usize,
    /// Convergence tolerance for Karcher mean (relative SRSF change).
    /// Default 1e-4. Decrease for higher precision at the cost of more
    /// iterations.
    pub karcher_tolerance: f64,
}

impl Default for ElasticSpmConfig {
    fn default() -> Self {
        Self {
            spm: SpmConfig::default(),
            align_lambda: 0.0,
            monitor_phase: true,
            warp_ncomp: 3,
            max_karcher_iterations: 20,
            karcher_tolerance: 1e-4,
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
    /// Mean integrated squared alignment residual (lower = better alignment).
    /// Computed as the average of ||f_i composed with gamma_i - mu||^2 across training
    /// observations. Values near 0 indicate excellent alignment (all curves collapse
    /// onto the mean). Compare against the mean squared distance before alignment
    /// (i.e., ||f_i - mu_arithmetic||^2) to quantify alignment benefit. A reduction
    /// of > 50% indicates substantial phase variation was present. The residual
    /// itself has no universal threshold; interpret relative to the unaligned
    /// baseline.
    pub mean_alignment_residual: f64,
}

/// Result of Phase II elastic SPM monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticSpmMonitorResult {
    /// Amplitude monitoring result.
    pub amplitude: SpmMonitorResult,
    /// Phase monitoring result (if monitoring phase).
    pub phase: Option<SpmMonitorResult>,
    /// The new observations after alignment to the Phase I Karcher mean.
    /// Useful for visual inspection and downstream analysis.
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
/// Returns errors from alignment or SPM chart construction. Alignment failures
/// (e.g., non-convergence of Karcher mean) propagate as errors. If alignment
/// fails, common remedies: (a) increase `max_karcher_iterations`, (b) pre-smooth
/// the data, (c) increase `align_lambda` toward 1.0 to regularize warping, or
/// (d) fall back to standard `spm_phase1` which monitors without alignment.
///
/// Returns `InvalidParameter` if `align_lambda` is outside [0, 1].
///
/// # Example
/// ```no_run
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::elastic_spm::{elastic_spm_phase1, ElasticSpmConfig};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let config = ElasticSpmConfig { monitor_phase: false, ..ElasticSpmConfig::default() };
/// let chart = elastic_spm_phase1(&data, &argvals, &config).unwrap();
/// assert!(chart.mean_alignment_residual >= 0.0);
/// ```
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
    if config.monitor_phase && config.warp_ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "warp_ncomp",
            message: "warp_ncomp must be >= 1 when monitor_phase is true".to_string(),
        });
    }
    if config.align_lambda < 0.0 || config.align_lambda > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "align_lambda",
            message: format!(
                "align_lambda must be in [0, 1], got {}",
                config.align_lambda
            ),
        });
    }

    // Step 1: Karcher mean
    let km_result = karcher_mean(
        data,
        argvals,
        config.max_karcher_iterations,
        config.karcher_tolerance,
        config.align_lambda,
    );

    // Step 2: Align all to Karcher mean
    let alignment = align_to_target(data, &km_result.mean, argvals, config.align_lambda);

    // Compute alignment quality: mean integrated squared residual
    let (n_aligned, m_aligned) = alignment.aligned_data.shape();
    let mean_alignment_residual = if m_aligned > 0 && n_aligned > 0 {
        let mut total_residual = 0.0;
        for i in 0..n_aligned {
            let mut sq_diff = 0.0;
            for j in 0..m_aligned {
                let d = alignment.aligned_data[(i, j)] - km_result.mean[j];
                sq_diff += d * d;
            }
            total_residual += sq_diff / m_aligned as f64;
        }
        total_residual / n_aligned as f64
    } else {
        0.0
    };

    // Step 3: SPM on aligned data (amplitude)
    let amplitude_chart = spm_phase1(&alignment.aligned_data, argvals, &config.spm)?;

    // Step 4: Optionally SPM on warping functions (phase)
    let phase_chart = if config.monitor_phase {
        let phase_config = SpmConfig {
            ncomp: config.warp_ncomp,
            alpha: config.spm.alpha,
            tuning_fraction: config.spm.tuning_fraction,
            // Use a distinct seed for the phase chart's tuning/calibration split
            // to ensure independent partitioning from the amplitude chart.
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
        mean_alignment_residual,
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
