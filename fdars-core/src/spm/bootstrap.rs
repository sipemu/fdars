//! Bootstrap and robust control limit estimation.
//!
//! Provides alternative methods for computing T-squared and SPE control
//! limits when the parametric chi-squared assumption may not hold.

use crate::error::FdarError;
use crate::helpers::sort_nan_safe;
use crate::iter_maybe_parallel;

use super::chi_squared::regularized_gamma_p;
use super::control::{spe_control_limit, t2_control_limit, ControlLimit};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Method for computing control limits.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ControlLimitMethod {
    /// Standard parametric (chi-squared) limits.
    Parametric,
    /// Empirical quantile from the observed values.
    Empirical,
    /// Bootstrap resampling to estimate the quantile.
    Bootstrap {
        /// Number of bootstrap resamples.
        n_bootstrap: usize,
        /// Random seed for reproducibility.
        seed: u64,
    },
    /// Kernel density estimation with Gaussian kernel.
    KernelDensity {
        /// Bandwidth (None for Silverman's rule of thumb).
        bandwidth: Option<f64>,
    },
}

/// Compute a robust T-squared control limit.
///
/// # Arguments
/// * `t2_values` - Observed T-squared values from calibration data
/// * `ncomp` - Number of principal components
/// * `alpha` - Significance level
/// * `method` - Control limit method
///
/// # Errors
///
/// Returns errors from the underlying limit computation method.
#[must_use = "control limit should not be discarded"]
pub fn t2_limit_robust(
    t2_values: &[f64],
    ncomp: usize,
    alpha: f64,
    method: &ControlLimitMethod,
) -> Result<ControlLimit, FdarError> {
    validate_inputs(t2_values, alpha)?;

    match method {
        ControlLimitMethod::Parametric => t2_control_limit(ncomp, alpha),
        ControlLimitMethod::Empirical => {
            let ucl = empirical_quantile(t2_values, 1.0 - alpha);
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("T2 empirical quantile, alpha={alpha}"),
            })
        }
        ControlLimitMethod::Bootstrap { n_bootstrap, seed } => {
            let ucl = bootstrap_quantile(t2_values, 1.0 - alpha, *n_bootstrap, *seed);
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("T2 bootstrap ({n_bootstrap} resamples), alpha={alpha}"),
            })
        }
        ControlLimitMethod::KernelDensity { bandwidth } => {
            let ucl = kde_quantile(t2_values, 1.0 - alpha, *bandwidth)?;
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("T2 KDE, alpha={alpha}"),
            })
        }
    }
}

/// Compute a robust SPE control limit.
///
/// # Arguments
/// * `spe_values` - Observed SPE values from calibration data
/// * `alpha` - Significance level
/// * `method` - Control limit method
///
/// # Errors
///
/// Returns errors from the underlying limit computation method.
#[must_use = "control limit should not be discarded"]
pub fn spe_limit_robust(
    spe_values: &[f64],
    alpha: f64,
    method: &ControlLimitMethod,
) -> Result<ControlLimit, FdarError> {
    validate_inputs(spe_values, alpha)?;

    match method {
        ControlLimitMethod::Parametric => spe_control_limit(spe_values, alpha),
        ControlLimitMethod::Empirical => {
            let ucl = empirical_quantile(spe_values, 1.0 - alpha);
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("SPE empirical quantile, alpha={alpha}"),
            })
        }
        ControlLimitMethod::Bootstrap { n_bootstrap, seed } => {
            let ucl = bootstrap_quantile(spe_values, 1.0 - alpha, *n_bootstrap, *seed);
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("SPE bootstrap ({n_bootstrap} resamples), alpha={alpha}"),
            })
        }
        ControlLimitMethod::KernelDensity { bandwidth } => {
            let ucl = kde_quantile(spe_values, 1.0 - alpha, *bandwidth)?;
            Ok(ControlLimit {
                ucl,
                alpha,
                description: format!("SPE KDE, alpha={alpha}"),
            })
        }
    }
}

fn validate_inputs(values: &[f64], alpha: f64) -> Result<(), FdarError> {
    if values.len() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "values",
            expected: "at least 2 values".to_string(),
            actual: format!("{} values", values.len()),
        });
    }
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in (0, 1), got {alpha}"),
        });
    }
    Ok(())
}

/// Empirical quantile: sort, return values[ceil(p*n) - 1].
fn empirical_quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sort_nan_safe(&mut sorted);
    let n = sorted.len();
    let idx = ((p * n as f64).ceil() as usize)
        .saturating_sub(1)
        .min(n - 1);
    sorted[idx]
}

/// Bootstrap quantile: resample `n_bootstrap` times, compute p-quantile each,
/// return mean of bootstrap quantiles.
fn bootstrap_quantile(values: &[f64], p: f64, n_bootstrap: usize, seed: u64) -> f64 {
    let n = values.len();

    let quantiles: Vec<f64> = iter_maybe_parallel!((0..n_bootstrap).collect::<Vec<_>>())
        .map(|rep| {
            let mut rng = StdRng::seed_from_u64(seed + rep as u64);
            let mut sample: Vec<f64> = (0..n).map(|_| values[rng.gen_range(0..n)]).collect();
            sort_nan_safe(&mut sample);
            let idx = ((p * n as f64).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1);
            sample[idx]
        })
        .collect();

    quantiles.iter().sum::<f64>() / quantiles.len() as f64
}

/// KDE quantile: Gaussian kernel density, Silverman bandwidth, bisection.
fn kde_quantile(values: &[f64], p: f64, bandwidth: Option<f64>) -> Result<f64, FdarError> {
    let n = values.len() as f64;
    let mean: f64 = values.iter().sum::<f64>() / n;
    let var: f64 = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();

    if std <= 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "kde_quantile",
            detail: "standard deviation is zero; cannot estimate KDE".to_string(),
        });
    }

    // Silverman's rule of thumb
    let h = bandwidth.unwrap_or_else(|| {
        let iqr = {
            let mut sorted = values.to_vec();
            sort_nan_safe(&mut sorted);
            let q75_idx = ((0.75 * n).ceil() as usize)
                .saturating_sub(1)
                .min(sorted.len() - 1);
            let q25_idx = ((0.25 * n).ceil() as usize)
                .saturating_sub(1)
                .min(sorted.len() - 1);
            sorted[q75_idx] - sorted[q25_idx]
        };
        let s = std.min(iqr / 1.34);
        let s = if s > 0.0 { s } else { std };
        0.9 * s * n.powf(-0.2)
    });

    if h <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("bandwidth must be positive, got {h}"),
        });
    }

    // KDE CDF: F_hat(x) = (1/n) * sum_i Phi((x - x_i) / h)
    // where Phi is the standard normal CDF
    let kde_cdf = |x: f64| -> f64 {
        let sum: f64 = values.iter().map(|&xi| normal_cdf((x - xi) / h)).sum();
        sum / n
    };

    // Bisection to find x such that kde_cdf(x) = p
    let mut sorted = values.to_vec();
    sort_nan_safe(&mut sorted);
    let mut lo = sorted[0] - 5.0 * h;
    let mut hi = sorted[sorted.len() - 1] + 5.0 * h;

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if kde_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if (hi - lo) < 1e-10 * (1.0 + hi.abs()) {
            break;
        }
    }

    Ok((lo + hi) / 2.0)
}

/// Standard normal CDF using the regularized gamma function.
///
/// Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
///
/// The error function is computed via `regularized_gamma_p(0.5, x^2)`,
/// exploiting the identity erf(x) = P(0.5, x^2) for x >= 0.
fn normal_cdf(x: f64) -> f64 {
    if x >= 0.0 {
        0.5 * (1.0 + erf_via_gamma(x))
    } else {
        0.5 * (1.0 - erf_via_gamma(-x))
    }
}

/// erf(x) for x >= 0 via the regularized lower incomplete gamma function.
///
/// Uses the identity: erf(x) = P(0.5, x^2), which follows from the
/// substitution u = sqrt(t) in the integral representation of P(0.5, x^2).
fn erf_via_gamma(x: f64) -> f64 {
    regularized_gamma_p(0.5, x * x)
}
