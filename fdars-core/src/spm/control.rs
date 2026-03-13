//! Control limits for SPM monitoring statistics.
//!
//! Provides upper control limits (UCL) for T-squared and SPE statistics:
//! - T-squared: chi-squared distribution quantile
//! - SPE: moment-matched chi-squared approximation (Box, 1954)

use super::chi_squared::chi2_quantile;
use crate::error::FdarError;

/// A control limit for a monitoring statistic.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ControlLimit {
    /// Upper control limit value.
    pub ucl: f64,
    /// Significance level used.
    pub alpha: f64,
    /// Human-readable description of how the limit was computed.
    pub description: String,
}

/// Compute the T-squared control limit based on the chi-squared distribution.
///
/// UCL = chi2_quantile(1 - alpha, ncomp)
///
/// # Arguments
/// * `ncomp` - Number of principal components (degrees of freedom)
/// * `alpha` - Significance level (e.g. 0.05)
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is 0 or `alpha` is not in (0, 1).
pub fn t2_control_limit(ncomp: usize, alpha: f64) -> Result<ControlLimit, FdarError> {
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be at least 1".to_string(),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in (0, 1), got {alpha}"),
        });
    }

    let ucl = chi2_quantile(1.0 - alpha, ncomp);

    Ok(ControlLimit {
        ucl,
        alpha,
        description: format!("T2 ~ chi2({ncomp}), alpha={alpha}"),
    })
}

/// Compute the SPE control limit using moment-matched chi-squared approximation.
///
/// Estimates parameters a and b such that SPE ~ a * chi2(b):
///   a = var(spe) / (2 * mean(spe))
///   b = 2 * mean(spe)^2 / var(spe)
///   UCL = a * chi2_quantile(1 - alpha, round(b))
///
/// # Arguments
/// * `spe_values` - In-control SPE values from calibration data
/// * `alpha` - Significance level (e.g. 0.05)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `spe_values` has fewer than 2 elements.
/// Returns [`FdarError::InvalidParameter`] if `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the estimated variance is zero or negative.
pub fn spe_control_limit(spe_values: &[f64], alpha: f64) -> Result<ControlLimit, FdarError> {
    let n = spe_values.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "spe_values",
            expected: "at least 2 values".to_string(),
            actual: format!("{n} values"),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in (0, 1), got {alpha}"),
        });
    }

    let mean: f64 = spe_values.iter().sum::<f64>() / n as f64;
    let var: f64 = spe_values.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);

    if mean <= 0.0 || var <= 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "spe_control_limit",
            detail: format!(
                "SPE mean={mean:.6}, var={var:.6}; cannot estimate chi-squared parameters from non-positive moments"
            ),
        });
    }

    let a = var / (2.0 * mean);
    let b = 2.0 * mean * mean / var;
    let b_int = b.round().max(1.0) as usize;

    let ucl = a * chi2_quantile(1.0 - alpha, b_int);

    Ok(ControlLimit {
        ucl,
        alpha,
        description: format!("SPE ~ {a:.4} * chi2({b_int}), alpha={alpha}"),
    })
}
