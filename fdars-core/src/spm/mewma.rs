//! Multivariate EWMA (MEWMA) monitoring for functional data.
//!
//! Extends the standard EWMA by computing a multivariate T²-type statistic
//! on the EWMA-smoothed score vector, using either the asymptotic or
//! exact time-dependent covariance matrix.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::chi_squared::chi2_quantile;
use super::ewma::ewma_scores;
use super::phase::SpmChart;
use super::stats::hotelling_t2;

/// Configuration for MEWMA monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct MewmaConfig {
    /// EWMA smoothing parameter in (0, 1] (default 0.2).
    pub lambda: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Use asymptotic covariance (true) or exact time-dependent (false).
    /// Default: true.
    pub asymptotic: bool,
}

impl Default for MewmaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.2,
            ncomp: 5,
            alpha: 0.05,
            asymptotic: true,
        }
    }
}

/// Result of MEWMA monitoring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MewmaMonitorResult {
    /// EWMA-smoothed score matrix (n × ncomp).
    pub smoothed_scores: FdMatrix,
    /// MEWMA T² statistic for each observation.
    pub mewma_statistic: Vec<f64>,
    /// Upper control limit.
    pub ucl: f64,
    /// Alarm flags.
    pub alarm: Vec<bool>,
}

/// Run MEWMA monitoring on sequential functional data.
///
/// 1. Projects data through the chart's FPCA
/// 2. Applies EWMA smoothing: Z_t = λξ_t + (1-λ)Z_{t-1}
/// 3. Computes covariance:
///    - Asymptotic: Σ_Z = (λ/(2-λ)) · diag(eigenvalues)
///    - Exact: Σ_Z(t) = (λ/(2-λ))(1-(1-λ)^{2t}) · diag(eigenvalues)
/// 4. MEWMA statistic: T_t = Z_t^T Σ_Z^{-1} Z_t
/// 5. UCL = chi2_quantile(1-α, ncomp)
///
/// # Arguments
/// * `chart` - Phase I SPM chart
/// * `sequential_data` - Sequential functional data (n × m), rows in time order
/// * `argvals` - Grid points (length m)
/// * `config` - MEWMA configuration
///
/// # Errors
///
/// Returns errors from projection, EWMA smoothing, or statistic computation.
#[must_use = "monitoring result should not be discarded"]
pub fn spm_mewma_monitor(
    chart: &SpmChart,
    sequential_data: &FdMatrix,
    _argvals: &[f64],
    config: &MewmaConfig,
) -> Result<MewmaMonitorResult, FdarError> {
    let m = chart.fpca.mean.len();
    if sequential_data.ncols() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "sequential_data",
            expected: format!("{m} columns"),
            actual: format!("{} columns", sequential_data.ncols()),
        });
    }
    if config.lambda <= 0.0 || config.lambda > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("lambda must be in (0, 1], got {}", config.lambda),
        });
    }

    let ncomp = chart.eigenvalues.len().min(config.ncomp);
    let n = sequential_data.nrows();

    // Project all observations
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

    // EWMA smooth
    let smoothed = ewma_scores(&scores, config.lambda)?;

    // Compute MEWMA statistic
    let mewma_statistic = if config.asymptotic {
        // Asymptotic: Σ_Z^{-1} diag entry = (2-λ)/(λ * eigenvalue)
        let adj_eigenvalues: Vec<f64> = chart
            .eigenvalues
            .iter()
            .take(actual_ncomp)
            .map(|&ev| ev * config.lambda / (2.0 - config.lambda))
            .collect();
        hotelling_t2(&smoothed, &adj_eigenvalues)?
    } else {
        // Exact time-dependent covariance
        let lambda = config.lambda;
        let one_minus_lambda = 1.0 - lambda;
        let lambda_factor = lambda / (2.0 - lambda);

        let mut stats = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i + 1) as f64;
            let time_factor = 1.0 - one_minus_lambda.powf(2.0 * t);
            let mut t2 = 0.0;
            for l in 0..actual_ncomp {
                let sigma_z_inv = 1.0 / (lambda_factor * time_factor * chart.eigenvalues[l]);
                let z = smoothed[(i, l)];
                t2 += z * z * sigma_z_inv;
            }
            stats.push(t2);
        }
        stats
    };

    // UCL from chi-squared
    let ucl = chi2_quantile(1.0 - config.alpha, actual_ncomp);

    // Alarms
    let alarm: Vec<bool> = mewma_statistic.iter().map(|&v| v > ucl).collect();

    Ok(MewmaMonitorResult {
        smoothed_scores: smoothed,
        mewma_statistic,
        ucl,
        alarm,
    })
}
