//! Warping function statistics: mean, variance, confidence bands.
//!
//! After elastic alignment, the warping functions contain information about
//! phase variation. This module provides summary statistics and uncertainty
//! quantification for sets of warping functions.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::warping::{
    exp_map_sphere, gam_to_psi, inv_exp_map_sphere, l2_norm_l2, phase_distance, psi_to_gam,
};

/// Statistics computed on a set of warping functions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct WarpStatistics {
    /// Pointwise mean warp (length m).
    pub mean: Vec<f64>,
    /// Pointwise variance (length m).
    pub variance: Vec<f64>,
    /// Pointwise standard deviation (length m).
    pub std_dev: Vec<f64>,
    /// Lower confidence band (length m).
    pub lower_band: Vec<f64>,
    /// Upper confidence band (length m).
    pub upper_band: Vec<f64>,
    /// Karcher mean warp on the Hilbert sphere (length m).
    pub karcher_mean_warp: Vec<f64>,
    /// Per-warp geodesic distances from Karcher mean (length n).
    pub geodesic_distances: Vec<f64>,
}

/// Inverse normal CDF (probit) via rational approximation (Abramowitz & Stegun 26.2.23).
fn normal_quantile(p: f64) -> f64 {
    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    let (sign, q) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let t = (-2.0 * (1.0 - q).ln()).sqrt();
    let numerator = C0 + C1 * t + C2 * t * t;
    let denominator = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;
    sign * (t - numerator / denominator)
}

/// Compute summary statistics for a set of warping functions.
///
/// Given an n x m matrix of warping functions (one per row) and the common
/// evaluation grid, computes pointwise statistics, confidence bands, the
/// Karcher mean warp on the Hilbert sphere, and per-warp geodesic distances.
///
/// # Arguments
/// * `gammas` — Warping functions matrix (n x m), one warp per row.
/// * `argvals` — Common evaluation points (length m).
/// * `confidence_level` — Confidence level for the bands (e.g. 0.95).
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if n < 2 or m does not match.
/// Returns `FdarError::InvalidParameter` if confidence_level is not in (0, 1).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn warp_statistics(
    gammas: &FdMatrix,
    argvals: &[f64],
    confidence_level: f64,
) -> Result<WarpStatistics, FdarError> {
    let (n, m) = gammas.shape();

    // Validate dimensions
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "gammas",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "gammas",
            expected: "at least 2 columns".to_string(),
            actual: format!("{m} columns"),
        });
    }

    // Validate confidence level
    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "confidence_level",
            message: format!("must be in (0, 1), got {confidence_level}"),
        });
    }

    let nf = n as f64;

    // ── Step 1: Pointwise mean, variance, std_dev ──

    let mut mean = vec![0.0; m];
    let mut variance = vec![0.0; m];

    for j in 0..m {
        let col = gammas.column(j);
        let mu = col.iter().sum::<f64>() / nf;
        mean[j] = mu;
        let var = col.iter().map(|&v| (v - mu) * (v - mu)).sum::<f64>() / (nf - 1.0);
        variance[j] = var;
    }

    let std_dev: Vec<f64> = variance.iter().map(|&v| v.sqrt()).collect();

    // ── Step 2: Confidence bands ──

    let alpha = 1.0 - confidence_level;
    let z = normal_quantile(1.0 - alpha / 2.0);
    let sqrt_n = nf.sqrt();

    let lower_band: Vec<f64> = mean
        .iter()
        .zip(std_dev.iter())
        .map(|(&mu, &sd)| mu - z * sd / sqrt_n)
        .collect();
    let upper_band: Vec<f64> = mean
        .iter()
        .zip(std_dev.iter())
        .map(|(&mu, &sd)| mu + z * sd / sqrt_n)
        .collect();

    // ── Step 3: Karcher mean warp on the Hilbert sphere ──

    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;

    // Uniform time grid on [0,1]
    let time_01: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let h = 1.0 / (m - 1) as f64;

    // Convert all warps to psi representation
    let mut psis: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let row = gammas.row(i);
        let gam_01: Vec<f64> = row.iter().map(|&g| (g - t0) / domain).collect();
        psis.push(gam_to_psi(&gam_01, h));
    }

    // Iterative Karcher mean on the sphere
    let mut psi_mean = psis[0].clone();
    let max_iter = 20;
    let step_size = 0.5;

    for _ in 0..max_iter {
        // Compute mean tangent vector
        let mut mean_tangent = vec![0.0; m];
        for psi_i in &psis {
            let v = inv_exp_map_sphere(&psi_mean, psi_i, &time_01);
            for (mt, vi) in mean_tangent.iter_mut().zip(v.iter()) {
                *mt += vi / nf;
            }
        }

        // Check convergence
        let tangent_norm = l2_norm_l2(&mean_tangent, &time_01);
        if tangent_norm < 1e-10 {
            break;
        }

        // Take a step along the mean tangent direction
        let step_tangent: Vec<f64> = mean_tangent.iter().map(|&v| v * step_size).collect();
        psi_mean = exp_map_sphere(&psi_mean, &step_tangent, &time_01);

        // Re-normalize to unit sphere
        let norm = l2_norm_l2(&psi_mean, &time_01);
        if norm > 1e-10 {
            for v in &mut psi_mean {
                *v /= norm;
            }
        }
    }

    // Convert Karcher mean psi back to warping function
    let karcher_gam_01 = psi_to_gam(&psi_mean, &time_01);
    let mut karcher_mean_warp: Vec<f64> = karcher_gam_01.iter().map(|&g| t0 + g * domain).collect();
    crate::warping::normalize_warp(&mut karcher_mean_warp, argvals);

    // ── Step 4: Geodesic distances ──

    let geodesic_distances: Vec<f64> = (0..n)
        .map(|i| {
            let row = gammas.row(i);
            phase_distance(&row, argvals)
        })
        .collect();

    Ok(WarpStatistics {
        mean,
        variance,
        std_dev,
        lower_band,
        upper_band,
        karcher_mean_warp,
        geodesic_distances,
    })
}
