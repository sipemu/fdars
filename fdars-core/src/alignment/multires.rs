//! Multi-resolution elastic alignment: coarse DP + fine gradient refinement.
//!
//! Standard DP alignment has O(m²) complexity. Multi-resolution alignment
//! runs DP on a coarsened grid first, then refines the warp using gradient
//! descent on the original resolution, giving faster alignment for long curves.

use super::pairwise::elastic_align_pair;
use super::srsf::{reparameterize_curve, srsf_single};
use super::{dp_alignment_core, AlignmentResult};
use crate::error::FdarError;
use crate::helpers::{l2_distance, linear_interp, simpsons_weights};
use crate::warping::normalize_warp;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for multi-resolution alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiresConfig {
    /// Coarsening factor: the coarse grid has `m / coarsen_factor` points.
    /// Must be >= 2. Default 4.
    pub coarsen_factor: usize,
    /// Number of gradient refinement steps on the fine grid.
    /// Default 10.
    pub n_refine_steps: usize,
    /// Gradient descent step size for refinement.
    /// Default 0.01.
    pub step_size: f64,
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
}

impl Default for MultiresConfig {
    fn default() -> Self {
        Self {
            coarsen_factor: 4,
            n_refine_steps: 10,
            step_size: 0.01,
            lambda: 0.0,
        }
    }
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Align curve `f2` to `f1` using multi-resolution elastic alignment.
///
/// 1. **Coarse stage**: Subsample both SRSFs to a coarser grid, run DP,
///    interpolate the resulting warp back to full resolution.
/// 2. **Fine stage**: Starting from the coarse warp, run gradient descent
///    steps to locally refine the warp on the full-resolution grid.
///
/// For short curves (m < 2 * coarsen_factor), falls back to standard DP.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `config` — Multi-resolution configuration
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if lengths do not match or m < 2.
/// Returns [`FdarError::InvalidParameter`] if `coarsen_factor < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_align_pair_multires(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    config: &MultiresConfig,
) -> Result<AlignmentResult, FdarError> {
    let m = f1.len();

    if m != f2.len() || m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "f1/f2/argvals",
            expected: format!("equal lengths, f1 has {m}"),
            actual: format!("f2 has {}, argvals has {}", f2.len(), argvals.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }
    if config.coarsen_factor < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "coarsen_factor",
            message: format!("must be >= 2, got {}", config.coarsen_factor),
        });
    }

    // For short curves, fall back to standard alignment
    if m < 2 * config.coarsen_factor {
        let result = elastic_align_pair(f1, f2, argvals, config.lambda);
        return Ok(result);
    }

    let q1 = srsf_single(f1, argvals);
    let q2 = srsf_single(f2, argvals);

    // ── Stage 1: Coarse DP ──
    let m_coarse = (m / config.coarsen_factor).max(4);
    let coarse_argvals = subsample_grid(argvals, m_coarse);
    let coarse_q1 = subsample_values(&q1, argvals, &coarse_argvals);
    let coarse_q2 = subsample_values(&q2, argvals, &coarse_argvals);

    let coarse_gamma = dp_alignment_core(&coarse_q1, &coarse_q2, &coarse_argvals, config.lambda);

    // Interpolate coarse warp to fine grid
    let mut gamma: Vec<f64> = argvals
        .iter()
        .map(|&t| linear_interp(&coarse_argvals, &coarse_gamma, t))
        .collect();
    normalize_warp(&mut gamma, argvals);

    // ── Stage 2: Gradient refinement ──
    for _ in 0..config.n_refine_steps {
        // Compute current cost and gradient
        let f2_warped = reparameterize_curve(f2, argvals, &gamma);
        let q2_warped = srsf_single(&f2_warped, argvals);

        // Approximate gradient: dJ/dγ_j ≈ -2(q1_j - q2_warped_j) * dq2/dγ_j
        // We use a finite-difference approximation for simplicity
        let h = 1.0 / (m as f64 * 10.0);
        let weights = simpsons_weights(argvals);
        let _current_dist = l2_distance(&q1, &q2_warped, &weights);

        let mut improved = false;
        for j in 1..m - 1 {
            // Perturb gamma[j] and measure cost change
            let orig = gamma[j];

            gamma[j] = orig + h;
            // Ensure monotonicity
            if gamma[j] <= gamma[j - 1] || gamma[j] >= gamma[j + 1] {
                gamma[j] = orig;
                continue;
            }

            let f2_pert = reparameterize_curve(f2, argvals, &gamma);
            let q2_pert = srsf_single(&f2_pert, argvals);
            let dist_plus = l2_distance(&q1, &q2_pert, &weights);

            gamma[j] = orig - h;
            if gamma[j] <= gamma[j - 1] || gamma[j] >= gamma[j + 1] {
                gamma[j] = orig;
                continue;
            }

            let f2_pert2 = reparameterize_curve(f2, argvals, &gamma);
            let q2_pert2 = srsf_single(&f2_pert2, argvals);
            let dist_minus = l2_distance(&q1, &q2_pert2, &weights);

            // Central difference gradient
            let grad = (dist_plus - dist_minus) / (2.0 * h);

            // Gradient step
            let new_val = orig - config.step_size * grad;
            // Clamp to maintain monotonicity
            let lo = gamma[j - 1] + 1e-12;
            let hi = gamma[j + 1] - 1e-12;
            gamma[j] = new_val.clamp(lo, hi);

            if (gamma[j] - orig).abs() > 1e-15 {
                improved = true;
            }
        }

        if !improved {
            break;
        }

        normalize_warp(&mut gamma, argvals);
    }

    // ── Final alignment ──
    let f_aligned = reparameterize_curve(f2, argvals, &gamma);
    let q_aligned = srsf_single(&f_aligned, argvals);
    let weights = simpsons_weights(argvals);
    let distance = l2_distance(&q1, &q_aligned, &weights);

    Ok(AlignmentResult {
        gamma,
        f_aligned,
        distance,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Create a uniform subsample of a grid.
fn subsample_grid(argvals: &[f64], m_coarse: usize) -> Vec<f64> {
    let m = argvals.len();
    if m_coarse >= m {
        return argvals.to_vec();
    }
    (0..m_coarse)
        .map(|i| {
            let idx_f = i as f64 * (m - 1) as f64 / (m_coarse - 1) as f64;
            let lo = idx_f.floor() as usize;
            let hi = idx_f.ceil().min((m - 1) as f64) as usize;
            let frac = idx_f - lo as f64;
            argvals[lo] * (1.0 - frac) + argvals[hi] * frac
        })
        .collect()
}

/// Interpolate values from the fine grid to a coarser grid.
fn subsample_values(values: &[f64], fine_grid: &[f64], coarse_grid: &[f64]) -> Vec<f64> {
    coarse_grid
        .iter()
        .map(|&t| linear_interp(fine_grid, values, t))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn multires_identity() {
        let m = 50;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();

        let config = MultiresConfig::default();
        let result = elastic_align_pair_multires(&f, &f, &t, &config).unwrap();

        assert!(
            result.distance < 0.5,
            "identical curves should have near-zero distance, got {}",
            result.distance
        );
    }

    #[test]
    fn multires_phase_shifted() {
        let m = 60;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();
        let f2: Vec<f64> = t.iter().map(|&x| ((x + 0.1) * 6.0).sin()).collect();

        let config = MultiresConfig::default();
        let result = elastic_align_pair_multires(&f1, &f2, &t, &config).unwrap();

        // Should produce a reasonable alignment
        let standard = elastic_align_pair(&f1, &f2, &t, 0.0);
        // Multi-res may be slightly worse but should not be dramatically worse
        assert!(
            result.distance < standard.distance * 2.0 + 0.5,
            "multi-res distance ({}) should be comparable to standard ({})",
            result.distance,
            standard.distance,
        );
    }

    #[test]
    fn multires_falls_back_short_curves() {
        let m = 6;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&x| x * x).collect();
        let f2: Vec<f64> = t.iter().map(|&x| x * x + 0.1).collect();

        let config = MultiresConfig {
            coarsen_factor: 4,
            ..Default::default()
        };
        let result = elastic_align_pair_multires(&f1, &f2, &t, &config).unwrap();
        assert_eq!(result.gamma.len(), m);
        assert_eq!(result.f_aligned.len(), m);
    }

    #[test]
    fn multires_rejects_bad_coarsen_factor() {
        let t = uniform_grid(20);
        let f: Vec<f64> = t.to_vec();
        let config = MultiresConfig {
            coarsen_factor: 1,
            ..Default::default()
        };
        assert!(elastic_align_pair_multires(&f, &f, &t, &config).is_err());
    }

    #[test]
    fn multires_config_default() {
        let config = MultiresConfig::default();
        assert_eq!(config.coarsen_factor, 4);
        assert_eq!(config.n_refine_steps, 10);
        assert!((config.step_size - 0.01).abs() < f64::EPSILON);
        assert!((config.lambda - 0.0).abs() < f64::EPSILON);
    }
}
