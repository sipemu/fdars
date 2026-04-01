//! Control limits for SPM monitoring statistics.
//!
//! Provides upper control limits (UCL) for T-squared and SPE statistics:
//! - **T-squared**: chi-squared distribution quantile. For finite calibration
//!   samples of size *n*, T² follows `(n·ncomp/(n−ncomp))·F(ncomp, n−ncomp)`
//!   rather than `χ²(ncomp)`. The chi-squared limit is the large-sample
//!   (*n* → ∞) limit.
//! - **SPE**: moment-matched chi-squared approximation (Box, 1954, Theorem 1,
//!   pp. 292–295). The derivation matches the first two moments of the SPE
//!   distribution to a scaled chi-squared: `E[a·χ²(b)] = a·b = mean`,
//!   `Var[a·χ²(b)] = 2a²·b = var`, giving `a = var/(2·mean)`,
//!   `b = 2·mean²/var`.
//!
//! # Accuracy
//!
//! The moment-matching approximation is exact when SPE follows a scaled
//! chi-squared distribution (holds under Gaussian scores). For non-Gaussian
//! data, the approximation error is O(κ₄) where κ₄ is the excess kurtosis
//! of the SPE distribution. Use [`spe_moment_match_diagnostic`] to assess
//! adequacy.
//!
//! # References
//!
//! - Box, G.E.P. (1954). Some theorems on quadratic forms applied in the
//!   study of analysis of variance problems, I. *Annals of Mathematical
//!   Statistics*, 25(2), 290–302. Theorem 1, pp. 292–295.
//! - Woodall, W.H. & Ncube, M.M. (1985). Multivariate CUSUM quality-control
//!   procedures. *Technometrics*, 27(3), 285–292. §2, pp. 286–288.

use super::chi_squared::chi2_quantile;
use crate::error::FdarError;

/// A control limit for a monitoring statistic.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
/// Based on the chi-squared approximation to the Hotelling T² distribution.
/// For small calibration samples (n < 30), this may be anti-conservative;
/// consider `t2_limit_robust()` with bootstrap method.
///
/// # Finite-sample correction
///
/// For finite calibration samples of size *n*, T² follows
/// `(n·ncomp/(n−ncomp))·F(ncomp, n−ncomp)` rather than `χ²(ncomp)`. The
/// chi-squared limit used here is the large-sample (*n* → ∞) limit. For
/// *n* < 30, the F-based limit should be preferred (Woodall & Ncube, 1985,
/// §2, pp. 286–288).
///
/// # Arguments
/// * `ncomp` - Number of principal components (degrees of freedom)
/// * `alpha` - Significance level (e.g. 0.05)
///
/// # Example
///
/// ```
/// use fdars_core::spm::control::t2_control_limit;
/// let limit = t2_control_limit(3, 0.05).unwrap();
/// assert!(limit.ucl > 0.0);
/// assert!((limit.ucl - 7.815).abs() < 0.01); // chi2(0.95, 3)
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is 0 or `alpha` is not in (0, 1).
#[must_use = "control limit should not be discarded"]
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
/// Uses the Box (1954, Theorem 1, pp. 292–295) moment-matching approximation:
/// SPE ~ a * chi²(b) where `a = var/(2·mean)`, `b = ceil(2·mean²/var)`. This
/// is accurate when SPE values are approximately chi-squared distributed,
/// which holds when the number of retained components is moderate (5–20)
/// and sample size is adequate (n > 30).
///
/// Estimates parameters a and b such that SPE ~ a * chi2(b):
///   a = var(spe) / (2 * mean(spe))
///   b = 2 * mean(spe)^2 / var(spe)
///   UCL = a * chi2_quantile(1 - alpha, ceil(b))
///
/// # Accuracy
///
/// The moment-matching approximation is exact when SPE follows a scaled
/// chi-squared distribution (holds under Gaussian scores). For non-Gaussian
/// data, the approximation error is O(κ₄) where κ₄ is the excess kurtosis
/// of the SPE distribution. Use [`spe_moment_match_diagnostic`] to check.
///
/// # Rounding choice
///
/// Using `ceil()` rather than `round()` for the degrees-of-freedom parameter
/// *b* gives a conservative (wider) control limit, ensuring the nominal
/// false-alarm rate is not exceeded.
///
/// # Arguments
/// * `spe_values` - In-control SPE values from calibration data
/// * `alpha` - Significance level (e.g. 0.05)
///
/// # Example
///
/// ```
/// use fdars_core::spm::control::spe_control_limit;
/// let spe_values = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.8, 2.2, 1.2, 2.8, 1.6];
/// let limit = spe_control_limit(&spe_values, 0.05).unwrap();
/// assert!(limit.ucl > 0.0);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `spe_values` has fewer than 2 elements.
/// Returns [`FdarError::InvalidParameter`] if `alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if the estimated variance is zero or negative.
#[must_use = "control limit should not be discarded"]
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
    // Using ceil() rather than round() gives a conservative (wider) control
    // limit, ensuring the nominal false-alarm rate is not exceeded.
    let b_int = b.ceil().max(1.0) as usize;

    let ucl = a * chi2_quantile(1.0 - alpha, b_int);

    Ok(ControlLimit {
        ucl,
        alpha,
        description: format!("SPE ~ {a:.4} * chi2({b_int}), alpha={alpha}"),
    })
}

/// Diagnostic for the SPE moment-match chi-squared approximation.
///
/// Computes the excess kurtosis of the SPE values and compares it to the
/// theoretical kurtosis of the fitted chi-squared distribution (12/b).
/// A large discrepancy suggests the chi-squared approximation may be poor.
///
/// Returns `(excess_kurtosis, theoretical_kurtosis, is_adequate)` where
/// `is_adequate` is true when the absolute difference is within 50% of
/// the theoretical value. Interpretation: `is_adequate == true` (ratio
/// within 50%) indicates the chi-squared model fits well. When inadequate,
/// the SPE distribution may be far from chi-squared (possibly multi-modal
/// or heavy-tailed); in that case, use `spe_limit_robust()` with bootstrap
/// or KDE method instead.
///
/// # Arguments
/// * `spe_values` - In-control SPE values
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if fewer than 4 values are provided.
#[must_use = "diagnostic result should not be discarded"]
pub fn spe_moment_match_diagnostic(spe_values: &[f64]) -> Result<(f64, f64, bool), FdarError> {
    let n = spe_values.len();
    if n < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "spe_values",
            expected: "at least 4 values".to_string(),
            actual: format!("{n} values"),
        });
    }

    let mean: f64 = spe_values.iter().sum::<f64>() / n as f64;
    let var: f64 = spe_values.iter().map(|&s| (s - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);

    if var <= 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "spe_moment_match_diagnostic",
            detail: "variance is zero".to_string(),
        });
    }

    // Excess kurtosis: E[(X-mu)^4]/sigma^4 - 3
    let m4: f64 = spe_values.iter().map(|&s| (s - mean).powi(4)).sum::<f64>() / n as f64;
    let excess_kurtosis = m4 / (var * var) - 3.0;

    // Theoretical: for chi2(b) scaled by a, kurtosis = 12/b
    // b = 2 * mean^2 / var
    let b = 2.0 * mean * mean / var;
    let theoretical_kurtosis = if b > 0.0 { 12.0 / b } else { f64::INFINITY };

    // Adequate if |observed - theoretical| <= 0.5 * theoretical
    let is_adequate = if theoretical_kurtosis.is_finite() {
        (excess_kurtosis - theoretical_kurtosis).abs() <= 0.5 * theoretical_kurtosis
    } else {
        false
    };

    Ok((excess_kurtosis, theoretical_kurtosis, is_adequate))
}
