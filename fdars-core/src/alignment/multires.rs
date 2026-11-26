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
///
/// Construct via `MultiresConfig::default()`, then assign the fields you need (e.g. `let mut c = MultiresConfig::default(); c.field = …;`). This struct is `#[non_exhaustive]`, so external crates cannot build it with a struct literal — not even functional-update `..Default::default()` form.
#[non_exhaustive]
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
    refine_gamma(f2, argvals, &q1, config, &mut gamma);

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

/// Penalised elastic objective minimised by the refinement stage:
///
/// `E(γ) = ‖q₁ − SRSF(f₂∘γ)‖²_{L²} + λ · Σⱼ (Δγⱼ/Δtⱼ − 1)² Δtⱼ`
///
/// i.e. the squared elastic distance reported in [`AlignmentResult::distance`]
/// plus the first-order (`l2gam`) roughness penalty used by the DP
/// ([`super::dp_lambda_penalty`]) evaluated on the piecewise-linear warp.
fn refine_objective(
    f2: &[f64],
    argvals: &[f64],
    q1: &[f64],
    weights: &[f64],
    lambda: f64,
    gamma: &[f64],
) -> f64 {
    let f2_warp = reparameterize_curve(f2, argvals, gamma);
    let q2_warp = srsf_single(&f2_warp, argvals);
    let d = l2_distance(q1, &q2_warp, weights);
    let mut obj = d * d;
    if lambda > 0.0 {
        let pen: f64 = argvals
            .windows(2)
            .zip(gamma.windows(2))
            .map(|(t, g)| {
                let dt = t[1] - t[0];
                let slope = (g[1] - g[0]) / dt;
                (slope - 1.0).powi(2) * dt
            })
            .sum();
        obj += lambda * pen;
    }
    obj
}

/// Shared, immutable inputs of the refinement objective.
struct RefineCtx<'a> {
    f2: &'a [f64],
    argvals: &'a [f64],
    q1: &'a [f64],
    weights: Vec<f64>,
    lambda: f64,
    h: f64,
    step_size: f64,
}

impl RefineCtx<'_> {
    fn objective(&self, gamma: &[f64]) -> f64 {
        refine_objective(
            self.f2,
            self.argvals,
            self.q1,
            &self.weights,
            self.lambda,
            gamma,
        )
    }
}

/// Maximum number of step halvings in the per-coordinate backtracking search.
const MAX_BACKTRACK: usize = 10;

/// Run gradient refinement on `gamma` for up to `config.n_refine_steps` sweeps.
///
/// Each sweep is a Gauss–Seidel pass of finite-difference gradient steps on the
/// interior grid values of `gamma`, with a backtracking descent check: a
/// coordinate update is accepted only if it strictly decreases the penalised
/// objective [`refine_objective`] (which includes `config.lambda`). The
/// objective is therefore non-increasing, so refinement can never make the
/// coarse-DP warp worse (GH #81). Monotonicity of `gamma` is preserved at every
/// step; stops early when a sweep accepts no update.
fn refine_gamma(
    f2: &[f64],
    argvals: &[f64],
    q1: &[f64],
    config: &MultiresConfig,
    gamma: &mut [f64],
) {
    let m = argvals.len();
    let ctx = RefineCtx {
        f2,
        argvals,
        q1,
        weights: simpsons_weights(argvals),
        lambda: config.lambda,
        h: 1.0 / (m as f64 * 10.0),
        step_size: config.step_size,
    };
    let mut obj = ctx.objective(gamma);

    for _ in 0..config.n_refine_steps {
        let mut improved = false;
        for j in 1..m - 1 {
            if let Some(new_obj) = refine_gamma_step(&ctx, gamma, j, obj) {
                obj = new_obj;
                improved = true;
            }
        }
        if !improved {
            break;
        }
        let before: Vec<f64> = gamma.to_vec();
        normalize_warp(gamma, argvals);
        let normalized_obj = ctx.objective(gamma);
        if normalized_obj > obj {
            gamma.copy_from_slice(&before);
        } else {
            obj = normalized_obj;
        }
    }
}

/// Objective with `gamma[j]` temporarily set to `val` (`None` if that would
/// break strict monotonicity). `gamma` is restored before returning.
fn objective_at(ctx: &RefineCtx<'_>, gamma: &mut [f64], j: usize, val: f64) -> Option<f64> {
    if val <= gamma[j - 1] || val >= gamma[j + 1] {
        return None;
    }
    let orig = gamma[j];
    gamma[j] = val;
    let obj = ctx.objective(gamma);
    gamma[j] = orig;
    Some(obj)
}

/// Apply one descent-checked finite-difference gradient step to `gamma[j]`.
///
/// Returns the new objective if an update strictly decreasing `current_obj`
/// was accepted (`gamma[j]` then holds the new value); otherwise leaves
/// `gamma[j]` unchanged and returns `None`. The step `step_size · ∂E/∂γⱼ` is
/// clamped to keep `gamma` strictly monotone and halved up to
/// [`MAX_BACKTRACK`] times until the objective decreases.
fn refine_gamma_step(
    ctx: &RefineCtx<'_>,
    gamma: &mut [f64],
    j: usize,
    current_obj: f64,
) -> Option<f64> {
    let orig = gamma[j];
    let obj_plus = objective_at(ctx, gamma, j, orig + ctx.h)?;
    let obj_minus = objective_at(ctx, gamma, j, orig - ctx.h)?;
    let grad = (obj_plus - obj_minus) / (2.0 * ctx.h);
    if !grad.is_finite() || grad == 0.0 {
        return None;
    }
    let lo = gamma[j - 1] + 1e-12;
    let hi = gamma[j + 1] - 1e-12;
    let mut step = ctx.step_size;
    for _ in 0..=MAX_BACKTRACK {
        let cand = (orig - step * grad).clamp(lo, hi);
        if (cand - orig).abs() <= 1e-15 {
            return None;
        }
        if let Some(obj) = objective_at(ctx, gamma, j, cand) {
            if obj < current_obj {
                gamma[j] = cand;
                return Some(obj);
            }
        }
        step *= 0.5;
    }
    None
}

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
    use std::f64::consts::PI;

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

    /// GH #81 repro (m = 400): the issue's warped two-frequency sinusoid.
    fn gh81_pair(m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let t = uniform_grid(m);
        let base: Vec<f64> = t
            .iter()
            .map(|&x| (4.0 * PI * x).sin() + 0.5 * (10.0 * PI * x).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&x| linear_interp(&t, &base, (x + 0.09 * (PI * x).sin()).clamp(0.0, 1.0)))
            .collect();
        (t, base, f2)
    }

    fn cfg(n_refine_steps: usize, step_size: f64, lambda: f64) -> MultiresConfig {
        MultiresConfig {
            n_refine_steps,
            step_size,
            lambda,
            ..MultiresConfig::default()
        }
    }

    /// Refinement must never increase the elastic distance of the coarse-DP
    /// warp it starts from, for any step count / step size (GH #81: it used to
    /// raise 0.354 -> 0.639 with defaults and to ~2.7 with 40 steps of 0.1).
    #[test]
    fn multires_refinement_never_increases_distance() {
        for m in [101, 400] {
            let (t, base, f2) = gh81_pair(m);
            let coarse = elastic_align_pair_multires(&base, &f2, &t, &cfg(0, 0.01, 0.0)).unwrap();
            for (steps, step) in [(10, 0.01), (40, 0.1), (5, 1.0)] {
                let r =
                    elastic_align_pair_multires(&base, &f2, &t, &cfg(steps, step, 0.0)).unwrap();
                assert!(
                    r.distance <= coarse.distance,
                    "m={m} steps={steps} step={step}: refined distance {} > coarse {}",
                    r.distance,
                    coarse.distance
                );
                for j in 1..m {
                    assert!(r.gamma[j] >= r.gamma[j - 1], "warp not monotone at {j}");
                }
            }
        }
    }

    /// With `lambda > 0` the refinement minimises the *penalised* objective:
    /// it must not increase it relative to the coarse warp, and a large
    /// lambda must pull the refined warp towards the identity (GH #81).
    #[test]
    fn multires_refinement_respects_lambda() {
        let m = 101;
        let (t, base, f2) = gh81_pair(m);
        let q1 = srsf_single(&base, &t);
        let w = simpsons_weights(&t);
        for lambda in [0.1, 10.0] {
            let coarse =
                elastic_align_pair_multires(&base, &f2, &t, &cfg(0, 0.01, lambda)).unwrap();
            let r = elastic_align_pair_multires(&base, &f2, &t, &cfg(20, 0.1, lambda)).unwrap();
            let e0 = refine_objective(&f2, &t, &q1, &w, lambda, &coarse.gamma);
            let e1 = refine_objective(&f2, &t, &q1, &w, lambda, &r.gamma);
            assert!(
                e1 <= e0,
                "lambda={lambda}: penalised objective rose {e0} -> {e1}"
            );
        }
        // Huge lambda: refining from the coarse warp must not move it away from
        // the identity (the penalty dominates the data term).
        let lambda = 1e6;
        let coarse = elastic_align_pair_multires(&base, &f2, &t, &cfg(0, 0.01, lambda)).unwrap();
        let r = elastic_align_pair_multires(&base, &f2, &t, &cfg(20, 0.1, lambda)).unwrap();
        let dev = |g: &[f64]| {
            g.iter()
                .zip(&t)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max)
        };
        assert!(dev(&r.gamma) <= dev(&coarse.gamma) + 1e-12);
    }
}
