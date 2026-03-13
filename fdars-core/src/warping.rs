//! Warping function utilities and Hilbert sphere geometry.
//!
//! This module provides operations on warping (reparameterization) functions,
//! including their Hilbert sphere representation via `ψ(t) = √γ'(t)`.
//!
//! Key capabilities:
//! - [`gam_to_psi`] / [`psi_to_gam`] — Convert between warping functions and sphere
//! - [`exp_map_sphere`] / [`inv_exp_map_sphere`] — Riemannian exponential / log maps
//! - [`normalize_warp`] / [`invert_gamma`] — Warp normalization and inversion
//! - [`phase_distance`] — Geodesic distance from a warp to the identity

use crate::helpers::{cumulative_trapz, gradient_uniform, linear_interp, trapz};
use crate::smoothing::nadaraya_watson;

/// Ensure γ is a valid warping: monotone non-decreasing, with correct boundary values.
pub fn normalize_warp(gamma: &mut [f64], argvals: &[f64]) {
    let n = gamma.len();
    if n == 0 {
        return;
    }

    // Fix boundaries
    gamma[0] = argvals[0];
    gamma[n - 1] = argvals[n - 1];

    // Enforce monotonicity
    for i in 1..n {
        if gamma[i] < gamma[i - 1] {
            gamma[i] = gamma[i - 1];
        }
    }
}

/// Convert warping function to Hilbert sphere representation: ψ = √γ'.
pub fn gam_to_psi(gam: &[f64], h: f64) -> Vec<f64> {
    gradient_uniform(gam, h)
        .iter()
        .map(|&g| g.max(0.0).sqrt())
        .collect()
}

/// Convert warping function to smoothed Hilbert sphere representation.
///
/// Like [`gam_to_psi`], but smooths γ before differentiating to remove
/// DP grid kinks. Matches Python fdasrsf's `SqrtMean(smooth=True)` which
/// uses spline smoothing (s=1e-4) before computing ψ = √γ'.
///
/// We use Nadaraya-Watson kernel smoothing on γ with bandwidth proportional
/// to the grid spacing, then differentiate the smoothed result. This prevents
/// derivative spikes from propagating into TSRVF tangent vectors and FPCA.
pub fn gam_to_psi_smooth(gam: &[f64], h: f64) -> Vec<f64> {
    let m = gam.len();
    if m < 3 {
        return gam_to_psi(gam, h);
    }

    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Smooth gamma with Nadaraya-Watson (bandwidth = 2 grid spacings).
    // This removes DP kinks while preserving the overall warp shape.
    let bandwidth = 2.0 * h;
    let gam_smooth = nadaraya_watson(&time, gam, &time, bandwidth, "gaussian");

    gradient_uniform(&gam_smooth, h)
        .iter()
        .map(|&g| g.max(0.0).sqrt())
        .collect()
}

/// Convert ψ back to warping function: γ = cumtrapz(ψ²), normalized to \[0,1\].
pub fn psi_to_gam(psi: &[f64], time: &[f64]) -> Vec<f64> {
    let psi_sq: Vec<f64> = psi.iter().map(|&p| p * p).collect();
    let gam = cumulative_trapz(&psi_sq, time);
    let min_val = gam.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = gam.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_val - min_val).max(1e-10);
    gam.iter().map(|&v| (v - min_val) / range).collect()
}

/// L2 inner product: ∫ψ₁·ψ₂ dt via trapezoidal rule.
pub fn inner_product_l2(psi1: &[f64], psi2: &[f64], time: &[f64]) -> f64 {
    let prod: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
    trapz(&prod, time)
}

/// L2 norm: √(∫ψ² dt).
pub fn l2_norm_l2(psi: &[f64], time: &[f64]) -> f64 {
    inner_product_l2(psi, psi, time).max(0.0).sqrt()
}

/// Inverse exponential (log) map on the Hilbert sphere.
/// Returns tangent vector at `mu` pointing toward `psi`.
pub fn inv_exp_map_sphere(mu: &[f64], psi: &[f64], time: &[f64]) -> Vec<f64> {
    let ip = inner_product_l2(mu, psi, time).clamp(-1.0, 1.0);
    let theta = ip.acos();
    if theta < 1e-10 {
        vec![0.0; mu.len()]
    } else {
        let coeff = theta / theta.sin();
        let cos_theta = theta.cos();
        mu.iter()
            .zip(psi.iter())
            .map(|(&m, &p)| coeff * (p - cos_theta * m))
            .collect()
    }
}

/// Exponential map on the Hilbert sphere.
/// Moves from `psi` along tangent vector `v`.
pub fn exp_map_sphere(psi: &[f64], v: &[f64], time: &[f64]) -> Vec<f64> {
    let v_norm = l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        psi.to_vec()
    } else {
        let cos_n = v_norm.cos();
        let sin_n = v_norm.sin();
        psi.iter()
            .zip(v.iter())
            .map(|(&p, &vi)| cos_n * p + sin_n * vi / v_norm)
            .collect()
    }
}

/// Invert a warping function: find γ⁻¹ such that γ⁻¹(γ(t)) = t.
/// `gam` and `time` are both on \[0,1\].
pub fn invert_gamma(gam: &[f64], time: &[f64]) -> Vec<f64> {
    let n = time.len();
    let mut gam_inv: Vec<f64> = time.iter().map(|&t| linear_interp(gam, time, t)).collect();
    gam_inv[0] = time[0];
    gam_inv[n - 1] = time[n - 1];
    gam_inv
}

/// Geodesic distance from a warping function to the identity on the Hilbert sphere.
///
/// Computes `arccos(⟨ψ/‖ψ‖, 1/‖1‖⟩_L2)` where `ψ = √γ'`.
///
/// # Arguments
/// * `gamma` — Warping function values (length m)
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// Geodesic distance (≥ 0). Returns 0 for the identity warp.
pub fn phase_distance(gamma: &[f64], argvals: &[f64]) -> f64 {
    let m = gamma.len();
    if m < 2 {
        return 0.0;
    }

    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;

    // Work on [0,1] internally
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    // Convert gamma to [0,1] and compute psi
    let gam_01: Vec<f64> = (0..m).map(|j| (gamma[j] - t0) / domain).collect();
    let psi = gam_to_psi(&gam_01, binsize);

    // Normalize psi to unit sphere
    let psi_norm = l2_norm_l2(&psi, &time);
    if psi_norm < 1e-10 {
        return 0.0;
    }
    let psi_unit: Vec<f64> = psi.iter().map(|&p| p / psi_norm).collect();

    // Identity warp psi = constant 1, normalized
    let id_raw = vec![1.0; m];
    let id_norm = l2_norm_l2(&id_raw, &time);
    let id_unit: Vec<f64> = id_raw.iter().map(|&v| v / id_norm).collect();

    // Geodesic distance = arccos(inner product)
    let ip = inner_product_l2(&psi_unit, &id_unit, &time).clamp(-1.0, 1.0);
    ip.acos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn test_gam_psi_round_trip() {
        let m = 101;
        let time = uniform_grid(m);
        let h = 1.0 / (m - 1) as f64;

        // Start with identity warp
        let gam = time.clone();
        let psi = gam_to_psi(&gam, h);
        let gam_recovered = psi_to_gam(&psi, &time);

        for j in 0..m {
            assert!(
                (gam_recovered[j] - time[j]).abs() < 0.02,
                "Round trip failed at j={j}: got {}, expected {}",
                gam_recovered[j],
                time[j]
            );
        }
    }

    #[test]
    fn test_normalize_warp_properties() {
        let t = uniform_grid(20);
        let mut gamma = vec![0.1; 20];
        normalize_warp(&mut gamma, &t);

        assert_eq!(gamma[0], t[0]);
        assert_eq!(gamma[19], t[19]);
        for i in 1..20 {
            assert!(gamma[i] >= gamma[i - 1]);
        }
    }

    #[test]
    fn test_invert_gamma_identity() {
        let m = 50;
        let time = uniform_grid(m);
        let inv = invert_gamma(&time, &time);
        for j in 0..m {
            assert!(
                (inv[j] - time[j]).abs() < 1e-12,
                "Inverting identity should give identity at j={j}"
            );
        }
    }

    #[test]
    fn test_sphere_round_trip() {
        let m = 21;
        let time = uniform_grid(m);

        // Construct two unit vectors on the sphere
        let raw1 = vec![1.0; m];
        let norm1 = l2_norm_l2(&raw1, &time);
        let psi1: Vec<f64> = raw1.iter().map(|&v| v / norm1).collect();

        let raw2: Vec<f64> = time
            .iter()
            .map(|&t| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin())
            .collect();
        let norm2 = l2_norm_l2(&raw2, &time);
        let psi2: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

        let v = inv_exp_map_sphere(&psi1, &psi2, &time);
        let recovered = exp_map_sphere(&psi1, &v, &time);

        let diff: Vec<f64> = psi2
            .iter()
            .zip(recovered.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .collect();
        let l2_err = trapz(&diff, &time).max(0.0).sqrt();
        assert!(
            l2_err < 1e-12,
            "Sphere round-trip error = {l2_err:.2e}, expected < 1e-12"
        );
    }

    #[test]
    fn test_phase_distance_identity_zero() {
        let m = 101;
        let t = uniform_grid(m);
        let d = phase_distance(&t, &t);
        assert!(
            d < 1e-6,
            "Phase distance of identity warp should be ~0, got {d}"
        );
    }

    #[test]
    fn test_phase_distance_nonidentity_positive() {
        let m = 101;
        let t = uniform_grid(m);
        let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect(); // quadratic warp
        let d = phase_distance(&gamma, &t);
        assert!(
            d > 0.01,
            "Phase distance of non-identity warp should be > 0, got {d}"
        );
    }
}
