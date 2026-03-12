//! SRSF (Square-Root Slope Function) transforms and warping utilities.

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
    let mat = FdMatrix::from_slice(f, 1, m).unwrap();
    let q_mat = srsf_transform(&mat, argvals);
    q_mat.row(0)
}
