//! EWMA (Exponentially Weighted Moving Average) smoothing for SPM.
//!
//! Applies EWMA smoothing to FPC scores before computing monitoring statistics.
//! This increases sensitivity to small persistent shifts in the process.
//! The T-squared statistic uses adjusted eigenvalues that account for the
//! variance reduction from smoothing.
//!
//! # Comparison with other charts
//!
//! For detecting small persistent shifts (< 1 sigma), EWMA with lambda around 0.1
//! outperforms Shewhart-type T-squared charts. For large shifts (> 2 sigma),
//! standard T-squared is preferable as EWMA introduces detection lag. See also
//! [`spm_mewma_monitor`](super::mewma::spm_mewma_monitor) for multivariate EWMA
//! and [`spm_amewma_monitor`](super::amewma::spm_amewma_monitor) for adaptive
//! smoothing.
//!
//! # Choosing lambda
//!
//! The smoothing parameter `lambda` controls the trade-off between sensitivity
//! and robustness. Small lambda (0.05--0.1) detects persistent small shifts;
//! large lambda (0.3--0.5) detects sudden large shifts. `lambda = 0.2` is a
//! common default balancing both. For ARL_0 ~ 370, typical (lambda, L) pairs:
//! (0.05, 2.615), (0.10, 2.814), (0.20, 2.962), (0.50, 3.071). See Lucas &
//! Saccucci (1990), Table 3.
//!
//! # References
//!
//! - Roberts, S.W. (1959). Control chart tests based on geometric moving
//!   averages. *Technometrics*, 1(3), 239-250.
//! - Lucas, J.M. & Saccucci, M.S. (1990). Exponentially weighted moving
//!   average control schemes: properties and enhancements. *Technometrics*,
//!   32(1), 1-12.
//! - Lowry, C.A., Woodall, W.H., Champ, C.W. & Rigdon, S.E. (1992). A
//!   multivariate exponentially weighted moving average control chart.
//!   *Technometrics*, 34(1), 46-53.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::chi_squared::chi2_quantile;
use super::phase::SpmChart;
use super::stats::{hotelling_t2, spe_univariate};

/// Configuration for EWMA monitoring.
#[derive(Debug, Clone, PartialEq)]
pub struct EwmaConfig {
    /// EWMA smoothing parameter in (0, 1] (default 0.2).
    /// `lambda = 1` gives raw scores (no smoothing).
    ///
    /// `lambda` controls the trade-off between sensitivity and robustness.
    /// Small lambda (0.05--0.1) detects persistent small shifts; large lambda
    /// (0.3--0.5) detects sudden large shifts. `lambda = 0.2` is a common
    /// default balancing both. For ARL_0 ~ 370, typical (lambda, L) pairs:
    /// (0.05, 2.615), (0.10, 2.814), (0.20, 2.962), (0.50, 3.071). See Lucas
    /// & Saccucci (1990), Table 3.
    pub lambda: f64,
    /// Number of principal components (default 5).
    pub ncomp: usize,
    /// Significance level (default 0.05).
    pub alpha: f64,
    /// Use exact time-dependent covariance correction (default false).
    /// When true, the T² statistic accounts for the EWMA startup transient
    /// where variance grows from 0 to the asymptotic value (Lowry et al., 1992).
    /// This prevents inflated T² values at early time steps.
    ///
    /// Recommended `true` for sequential monitoring with n < 50 observations.
    /// For large Phase II batches, the asymptotic approximation (`false`) is
    /// adequate and slightly faster.
    pub exact_covariance: bool,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            lambda: 0.2,
            ncomp: 5,
            alpha: 0.05,
            exact_covariance: false,
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
    ///
    /// Computed from raw (unsmoothed) scores to preserve instantaneous fault
    /// detection. See module doc for rationale.
    pub spe: Vec<f64>,
    /// T-squared control limit for EWMA.
    ///
    /// When using asymptotic covariance (`exact_covariance: false`), this UCL
    /// may be slightly liberal during the first ~20 observations because the
    /// chi-squared limit ignores the EWMA startup transient. Set
    /// `exact_covariance: true` for precise startup behavior.
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
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::ewma::ewma_scores;
/// let scores = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 0.5, 1.0, 1.5], 3, 2).unwrap();
/// let smoothed = ewma_scores(&scores, 0.2).unwrap();
/// assert_eq!(smoothed.shape(), (3, 2));
/// // First row: lambda * score (since Z_0 = lambda * x_0)
/// assert!((smoothed[(0, 0)] - 0.2).abs() < 1e-10);
/// ```
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
/// When using asymptotic covariance (`exact_covariance: false`), the UCL may
/// be slightly liberal during the first ~20 observations. Set
/// `exact_covariance: true` for precise startup behavior.
///
/// # Notes
///
/// SPE is computed from unsmoothed (raw) scores rather than the EWMA-smoothed
/// scores. EWMA smoothing would spread reconstruction error across time,
/// reducing the diagnostic specificity of SPE alarms. The smoothed T-squared
/// catches persistent mean shifts while raw SPE catches instantaneous model
/// violations.
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
///
/// # Example
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::phase::{spm_phase1, SpmConfig};
/// use fdars_core::spm::ewma::{spm_ewma_monitor, EwmaConfig};
/// let data = FdMatrix::from_column_major(
///     (0..200).map(|i| (i as f64 * 0.1).sin()).collect(), 20, 10
/// ).unwrap();
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let chart = spm_phase1(&data, &argvals, &SpmConfig { ncomp: 2, ..SpmConfig::default() }).unwrap();
/// let new_data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(), 5, 10
/// ).unwrap();
/// let result = spm_ewma_monitor(&chart, &new_data, &argvals, &EwmaConfig::default()).unwrap();
/// assert_eq!(result.t2.len(), 5);
/// ```
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
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be at least 1".to_string(),
        });
    }

    // Project all observations
    let scores = chart.fpca.project(sequential_data)?;

    // Truncate to ncomp columns if needed.
    // NOTE: This pattern is intentionally duplicated across ewma, amewma, and
    // mewma modules for clarity—each monitor owns its projection pipeline and
    // the inline truncation keeps the data flow explicit.
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
    let n = smoothed.nrows();
    let t2 = if config.exact_covariance {
        // Exact variance formula: Var(Z_t) = λ/(2-λ) · [1 - (1-λ)^{2(t+1)}] · eigenvalue.
        // See Lowry et al. (1992), Equation (4).
        let lambda = config.lambda;
        let one_minus_lambda = 1.0 - lambda;
        let lambda_factor = lambda / (2.0 - lambda);
        let mut stats = Vec::with_capacity(n);
        for i in 0..n {
            let t = (i + 1) as f64;
            let time_factor = 1.0 - one_minus_lambda.powf(2.0 * t);
            let mut t2_val = 0.0;
            for l in 0..actual_ncomp {
                let adj_ev = lambda_factor * time_factor * chart.eigenvalues[l];
                let z = smoothed[(i, l)];
                t2_val += z * z / adj_ev;
            }
            stats.push(t2_val);
        }
        stats
    } else {
        hotelling_t2(&smoothed, &adj_eigenvalues)?
    };

    // SPE from reconstruction error (uses raw scores, not smoothed).
    // SPE is computed from unsmoothed scores because EWMA smoothing would
    // spread reconstruction error across time, reducing the diagnostic
    // specificity of SPE alarms. Smoothed T-squared catches persistent mean
    // shifts; raw SPE catches instantaneous model violations.
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
