use super::helpers::{build_band, percentile_sorted};
use super::{MultiplierDistribution, ToleranceBand};
use crate::error::FdarError;
use crate::fdata::mean_1d;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::smoothing::local_polynomial;
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── SCB for the Mean (Degras) ──────────────────────────────────────────────

/// Compute pointwise residual standard deviation (using n-1 divisor to match R's sd()).
pub(super) fn residual_sigma(data: &FdMatrix, center: &[f64], n: usize, m: usize) -> Vec<f64> {
    let nf_minus1 = (n as f64 - 1.0).max(1.0);
    (0..m)
        .map(|j| {
            let var: f64 = (0..n).map(|i| (data[(i, j)] - center[j]).powi(2)).sum();
            (var / nf_minus1).sqrt().max(1e-15)
        })
        .collect()
}

/// Generate n multiplier weights for Degras bootstrap.
pub(super) fn generate_multiplier_weights(
    rng: &mut StdRng,
    n: usize,
    multiplier: MultiplierDistribution,
) -> Vec<f64> {
    (0..n)
        .map(|_| match multiplier {
            MultiplierDistribution::Gaussian => rng.sample::<f64, _>(StandardNormal),
            MultiplierDistribution::Rademacher => {
                if rng.gen_bool(0.5) {
                    1.0
                } else {
                    -1.0
                }
            }
        })
        .collect()
}

/// Compute a simultaneous confidence band for the mean function (Degras method).
///
/// Uses a multiplier bootstrap to estimate the critical value for a
/// simultaneous confidence band around the mean.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `argvals` — Evaluation points (length m)
/// * `bandwidth` — Kernel bandwidth for local polynomial smoothing
/// * `nb` — Number of bootstrap replicates
/// * `confidence` — Confidence level (e.g., 0.95)
/// * `multiplier` — [`MultiplierDistribution::Gaussian`] or [`MultiplierDistribution::Rademacher`]
///
/// # Returns
/// `Ok(ToleranceBand)` on success, or `Err(FdarError)` if inputs are invalid.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has fewer than 3 rows,
/// zero columns, or if `argvals` length does not match the number of columns.
/// Returns [`FdarError::InvalidParameter`] if `bandwidth` is not positive,
/// `nb` is zero, or `confidence` is not in the open interval (0, 1).
///
/// # Examples
///
/// ```
/// use fdars_core::simulation::{sim_fundata, EFunType, EValType};
/// use fdars_core::tolerance::{scb_mean_degras, MultiplierDistribution};
///
/// let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
/// let data = sim_fundata(50, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));
///
/// let band = scb_mean_degras(&data, &t, 0.15, 200, 0.95, MultiplierDistribution::Gaussian).unwrap();
/// assert_eq!(band.center.len(), 50);
/// assert!(band.lower.iter().zip(band.upper.iter()).all(|(l, u)| l < u));
/// ```
pub fn scb_mean_degras(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    nb: usize,
    confidence: f64,
    multiplier: MultiplierDistribution,
) -> Result<ToleranceBand, FdarError> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows and 1 column".to_string(),
            actual: format!("{n} x {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (matching data columns)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("must be positive, got {bandwidth}"),
        });
    }
    if nb == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "nb",
            message: "must be >= 1".to_string(),
        });
    }
    if confidence <= 0.0 || confidence >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "confidence",
            message: format!("must be in (0, 1), got {confidence}"),
        });
    }

    let raw_mean = mean_1d(data);
    let center = local_polynomial(argvals, &raw_mean, argvals, bandwidth, 1, "epanechnikov")?;
    let sigma_hat = residual_sigma(data, &center, n, m);

    let sqrt_n = (n as f64).sqrt();
    let mut sup_stats: Vec<f64> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(42 + b as u64);
            let weights = generate_multiplier_weights(&mut rng, n, multiplier);
            (0..m)
                .map(|j| {
                    let z: f64 = (0..n)
                        .map(|i| weights[i] * (data[(i, j)] - center[j]))
                        .sum::<f64>()
                        / (sqrt_n * sigma_hat[j]);
                    z.abs()
                })
                .fold(0.0_f64, f64::max)
        })
        .collect();

    let c = percentile_sorted(&mut sup_stats, confidence);
    let half_width: Vec<f64> = sigma_hat.iter().map(|&s| c * s / sqrt_n).collect();

    Ok(build_band(center, half_width))
}
