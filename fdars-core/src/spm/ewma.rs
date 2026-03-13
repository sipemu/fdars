//! EWMA (Exponentially Weighted Moving Average) smoothing for SPM.
//!
//! Applies EWMA smoothing to FPC scores before computing monitoring statistics.
//! This increases sensitivity to small persistent shifts in the process.
//! The T-squared statistic uses adjusted eigenvalues that account for the
//! variance reduction from smoothing.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::chi_squared::chi2_quantile;
use super::phase::SpmChart;
use super::stats::{hotelling_t2, spe_univariate};

/// Configuration for EWMA monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct EwmaConfig {
    /// EWMA smoothing parameter in (0, 1] (default 0.2).
    /// lambda = 1 gives raw scores (no smoothing).
    pub lambda: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.2,
            ncomp: 5,
            alpha: 0.05,
        }
    }
}

/// Result of EWMA-based monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct EwmaMonitorResult {
    /// Smoothed score matrix (n x ncomp).
    pub smoothed_scores: FdMatrix,
    /// T-squared values computed on smoothed scores.
    pub t2: Vec<f64>,
    /// SPE values (from reconstruction error, not smoothed).
    pub spe: Vec<f64>,
    /// T-squared control limit for EWMA.
    pub t2_limit: f64,
    /// SPE control limit.
    pub spe_limit: f64,
    /// T-squared alarm flags.
    pub t2_alarm: Vec<bool>,
    /// SPE alarm flags.
    pub spe_alarm: Vec<bool>,
}

/// Apply EWMA smoothing to a sequence of score vectors.
///
/// Z_t = lambda * xi_t + (1 - lambda) * Z_{t-1}, with Z_0 = 0.
///
/// # Arguments
/// * `scores` - Score matrix (n x ncomp), rows are sequential observations
/// * `lambda` - Smoothing parameter in (0, 1]
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `lambda` is not in (0, 1].
pub fn ewma_scores(scores: &FdMatrix, lambda: f64) -> Result<FdMatrix, FdarError> {
    if lambda <= 0.0 || lambda > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("lambda must be in (0, 1], got {lambda}"),
        });
    }

    let (n, ncomp) = scores.shape();
    let mut smoothed = FdMatrix::zeros(n, ncomp);

    // Z_0 = lambda * xi_0 (since Z_{-1} = 0)
    if n > 0 {
        for k in 0..ncomp {
            smoothed[(0, k)] = lambda * scores[(0, k)];
        }
    }

    for i in 1..n {
        for k in 0..ncomp {
            smoothed[(i, k)] = lambda * scores[(i, k)] + (1.0 - lambda) * smoothed[(i - 1, k)];
        }
    }

    Ok(smoothed)
}

/// Run EWMA-based monitoring on sequential functional data.
///
/// 1. Projects each observation through the chart's FPCA
/// 2. Applies EWMA smoothing to the scores
/// 3. Computes T-squared on smoothed scores with adjusted eigenvalues
/// 4. Computes SPE from reconstruction error (unsmoothed)
/// 5. Sets limits from chi-squared distribution
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n x m), rows in time order
/// * `argvals` - Grid points (length m)
/// * `config` - EWMA configuration
///
/// # Errors
///
/// Returns errors from projection, EWMA smoothing, or statistic computation.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_ewma_monitor(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    argvals: &[f64],
    config: &EwmaConfig,
) -> Result<EwmaMonitorResult, FdarError> {
    let m = chart.fpca.mean.len();
    if sequential_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "sequential_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", sequential_data.ncols()),
        });
    }

    let ncomp = chart.eigenvalues.len().min(config.ncomp);

    // Project all observations
    let scores = chart.fpca.project(sequential_data)?;

    // Truncate to ncomp columns if needed
    let scores = if scores.ncols() > ncomp {
        let n = scores.nrows();
        let mut trunc = FdMatrix::zeros(n, ncomp);
        for i in 0..n {
            for k in 0..ncomp {
                trunc[(i, k)] = scores[(i, k)];
            }
        }
        trunc
    } else {
        scores
    };
    let actual_ncomp = scores.ncols();

    // EWMA smooth
    let smoothed = ewma_scores(&scores, config.lambda)?;

    // Adjusted eigenvalues for EWMA: lambda_adj = eigenvalue * lambda / (2 - lambda)
    let adj_eigenvalues: Vec<f64> = chart
        .eigenvalues
        .iter()
        .take(actual_ncomp)
        .map(|&ev| ev * config.lambda / (2.0 - config.lambda))
        .collect();

    // T-squared on smoothed scores
    let t2 = hotelling_t2(&smoothed, &adj_eigenvalues)?;

    // SPE from reconstruction error (uses raw scores, not smoothed)
    let centered = {
        let n = sequential_data.nrows();
        let mut c = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                c[(i, j)] = sequential_data[(i, j)] - chart.fpca.mean[j];
            }
        }
        c
    };

    let recon_centered = {
        let n = scores.nrows();
        let mut r = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                let mut val = 0.0;
                for k in 0..actual_ncomp {
                    val += scores[(i, k)] * chart.fpca.rotation[(j, k)];
                }
                r[(i, j)] = val;
            }
        }
        r
    };

    let spe = spe_univariate(&centered, &recon_centered, argvals)?;

    // Control limits
    let t2_limit = chi2_quantile(1.0 - config.alpha, actual_ncomp);
    let spe_limit = chart.spe_limit.ucl;

    // Alarms
    let t2_alarm: Vec<bool> = t2.iter().map(|&v| v > t2_limit).collect();
    let spe_alarm: Vec<bool> = spe.iter().map(|&v| v > spe_limit).collect();

    Ok(EwmaMonitorResult {
        smoothed_scores: smoothed,
        t2,
        spe,
        t2_limit,
        spe_limit,
        t2_alarm,
        spe_alarm,
    })
}
