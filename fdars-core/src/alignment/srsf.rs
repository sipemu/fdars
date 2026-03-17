//! SRSF (Square-Root Slope Function) transforms and warping utilities.

use crate::error::FdarError;
use crate::fdata::deriv_1d;
use crate::helpers::{cumulative_trapz, linear_interp};
use crate::matrix::FdMatrix;

// ─── SRSF Transform and Inverse ─────────────────────────────────────────────

/// Compute the Square-Root Slope Function (SRSF) transform.
///
/// For each curve f, the SRSF is: `q(t) = sign(f'(t)) * sqrt(|f'(t)|)`
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// FdMatrix of SRSFs with the same shape as input.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::alignment::srsf_transform;
///
/// let argvals: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let data = FdMatrix::from_column_major(
///     argvals.iter().map(|&t| (t * 6.0).sin()).collect(),
///     1, 20,
/// ).unwrap();
/// let srsf = srsf_transform(&data, &argvals);
/// assert_eq!(srsf.shape(), (1, 20));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn srsf_transform(data: &FdMatrix, argvals: &[f64]) -> FdMatrix {
    let (n, m) = data.shape();
    if n == 0 || m == 0 || argvals.len() != m {
        return FdMatrix::zeros(n, m);
    }

    let deriv = deriv_1d(data, argvals, 1);

    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            let d = deriv[(i, j)];
            result[(i, j)] = d.signum() * d.abs().sqrt();
        }
    }
    result
}

/// Reconstruct a curve from its SRSF representation.
///
/// Given SRSF q and initial value f0, reconstructs: `f(t) = f0 + ∫₀ᵗ q(s)|q(s)| ds`
///
/// # Arguments
/// * `q` — SRSF values (length m)
/// * `argvals` — Evaluation points (length m)
/// * `f0` — Initial value f(argvals\[0\])
///
/// # Returns
/// Reconstructed curve values.
pub fn srsf_inverse(q: &[f64], argvals: &[f64], f0: f64) -> Vec<f64> {
    let m = q.len();
    if m == 0 {
        return Vec::new();
    }

    // Integrand: q(s) * |q(s)|
    let integrand: Vec<f64> = q.iter().map(|&qi| qi * qi.abs()).collect();
    let integral = cumulative_trapz(&integrand, argvals);

    integral.iter().map(|&v| f0 + v).collect()
}

// ─── Reparameterization ─────────────────────────────────────────────────────

/// Reparameterize a curve by a warping function.
///
/// Computes `f(γ(t))` via linear interpolation.
///
/// # Arguments
/// * `f` — Curve values (length m)
/// * `argvals` — Evaluation points (length m)
/// * `gamma` — Warping function values (length m)
pub fn reparameterize_curve(f: &[f64], argvals: &[f64], gamma: &[f64]) -> Vec<f64> {
    gamma
        .iter()
        .map(|&g| linear_interp(argvals, f, g))
        .collect()
}

/// Compose two warping functions: `(γ₁ ∘ γ₂)(t) = γ₁(γ₂(t))`.
///
/// # Arguments
/// * `gamma1` — Outer warping function (length m)
/// * `gamma2` — Inner warping function (length m)
/// * `argvals` — Evaluation points (length m)
pub fn compose_warps(gamma1: &[f64], gamma2: &[f64], argvals: &[f64]) -> Vec<f64> {
    gamma2
        .iter()
        .map(|&g| linear_interp(argvals, gamma1, g))
        .collect()
}

/// Compute a single SRSF from a slice (single-row convenience).
pub(crate) fn srsf_single(f: &[f64], argvals: &[f64]) -> Vec<f64> {
    let m = f.len();
    let mat = FdMatrix::from_slice(f, 1, m).expect("dimension invariant: data.len() == n * m");
    let q_mat = srsf_transform(&mat, argvals);
    q_mat.row(0)
}

/// Compute the inverse of a warping function.
///
/// Given γ: \[a,b\] → \[a,b\], computes γ⁻¹ such that γ⁻¹(γ(t)) ≈ t.
/// The inverse is computed by mapping to \[0,1\], calling the sphere-based
/// inverse from the warping module, and mapping back to the original domain.
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if lengths do not match or m < 2.
pub fn invert_warp(gamma: &[f64], argvals: &[f64]) -> Result<Vec<f64>, FdarError> {
    let m = gamma.len();
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "gamma",
            expected: format!("length {}", argvals.len()),
            actual: format!("length {m}"),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "gamma",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    if domain <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: format!("domain must be positive, got {domain}"),
        });
    }
    // Normalize to [0,1]
    let gam_01: Vec<f64> = gamma.iter().map(|&g| (g - t0) / domain).collect();
    let time_01: Vec<f64> = argvals.iter().map(|&t| (t - t0) / domain).collect();
    // Invert using warping module
    let inv_01 = crate::warping::invert_gamma(&gam_01, &time_01);
    // Map back to original domain
    let mut result: Vec<f64> = inv_01.iter().map(|&g| t0 + g * domain).collect();
    crate::warping::normalize_warp(&mut result, argvals);
    Ok(result)
}

/// Verify roundtrip accuracy: max |γ(γ⁻¹(t)) - t| over the domain.
///
/// Returns the maximum absolute deviation. Values near 0 indicate
/// a high-quality inverse. Typical values for smooth warps: < 1e-10.
pub fn warp_inverse_error(gamma: &[f64], gamma_inv: &[f64], argvals: &[f64]) -> f64 {
    // compose gamma with gamma_inv, compare to identity
    let roundtrip = compose_warps(gamma, gamma_inv, argvals);
    roundtrip
        .iter()
        .zip(argvals.iter())
        .map(|(&r, &t)| (r - t).abs())
        .fold(0.0_f64, f64::max)
}
