//! Elastic partial matching: find the best-aligned subcurve of a longer curve.
//!
//! Standard elastic alignment requires both curves to span the full domain.
//! Partial matching relaxes this: given a template curve and a longer curve,
//! it finds the contiguous subdomain of the longer curve that best matches
//! the template in the elastic metric.

use super::srsf::srsf_single;
use super::{dp_edge_weight, dp_lambda_penalty, dp_path_to_gamma};
use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights};

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for elastic partial matching.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialMatchConfig {
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
    /// Minimum fraction of the target curve that the match must span.
    /// Must be in (0, 1]. Default 0.5.
    pub min_span: f64,
}

impl Default for PartialMatchConfig {
    fn default() -> Self {
        Self {
            lambda: 0.0,
            min_span: 0.5,
        }
    }
}

/// Result of elastic partial matching.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PartialMatchResult {
    /// Start index in the target curve where the best match begins.
    pub start_index: usize,
    /// End index (inclusive) in the target curve where the best match ends.
    pub end_index: usize,
    /// Warping function mapping template domain to the matched subdomain.
    pub gamma: Vec<f64>,
    /// Elastic distance of the best partial match.
    pub distance: f64,
    /// Fraction of the target domain spanned by the match.
    pub domain_fraction: f64,
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Find the best elastic partial match of `template` within `target`.
///
/// Slides a variable-length window over the target curve and performs
/// elastic alignment of the template to each window, returning the
/// window position and warp with minimum elastic distance.
///
/// # Arguments
/// * `template` — Short template curve (length m_t)
/// * `target` — Longer target curve (length m_f)
/// * `argvals_template` — Evaluation points for the template (length m_t)
/// * `argvals_target` — Evaluation points for the target (length m_f)
/// * `config` — Partial matching configuration
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if lengths are inconsistent.
/// Returns [`FdarError::InvalidParameter`] if `min_span` is not in (0, 1].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_partial_match(
    template: &[f64],
    target: &[f64],
    argvals_template: &[f64],
    argvals_target: &[f64],
    config: &PartialMatchConfig,
) -> Result<PartialMatchResult, FdarError> {
    let m_t = template.len();
    let m_f = target.len();

    if m_t != argvals_template.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_template",
            expected: format!("{m_t}"),
            actual: format!("{}", argvals_template.len()),
        });
    }
    if m_f != argvals_target.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals_target",
            expected: format!("{m_f}"),
            actual: format!("{}", argvals_target.len()),
        });
    }
    if m_t < 2 || m_f < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "template/target",
            expected: "length >= 2".to_string(),
            actual: format!("template={m_t}, target={m_f}"),
        });
    }
    if config.min_span <= 0.0 || config.min_span > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "min_span",
            message: format!("must be in (0, 1], got {}", config.min_span),
        });
    }

    let q_template = srsf_single(template, argvals_template);

    // Minimum window size (in grid points) based on min_span
    let min_window = ((m_f as f64 * config.min_span).ceil() as usize).max(2);

    let mut best_start = 0;
    let mut best_end = m_f - 1;
    let mut best_dist = f64::INFINITY;
    let mut best_gamma = argvals_template.to_vec();

    // Iterate over window sizes from min_window to m_f
    // Use a coarse grid of window sizes for efficiency
    let n_sizes = 5.min(m_f - min_window + 1);
    let sizes: Vec<usize> = if n_sizes <= 1 {
        vec![m_f]
    } else {
        (0..n_sizes)
            .map(|i| min_window + i * (m_f - min_window) / (n_sizes - 1))
            .collect()
    };

    for &win_size in &sizes {
        let step = (win_size / 10).max(1);
        let mut start = 0;
        while start + win_size <= m_f {
            let end = start + win_size - 1;

            // Extract sub-argvals and sub-curve
            let sub_argvals: Vec<f64> = (0..m_t)
                .map(|i| {
                    argvals_target[start]
                        + (argvals_target[end] - argvals_target[start]) * i as f64
                            / (m_t - 1) as f64
                })
                .collect();

            // Interpolate target onto sub_argvals
            let sub_target: Vec<f64> = sub_argvals
                .iter()
                .map(|&t| interp_target(target, argvals_target, t))
                .collect();

            let q_sub = srsf_single(&sub_target, argvals_template);

            // DP alignment on the shared template grid
            let gamma = dp_align_partial(&q_template, &q_sub, argvals_template, config.lambda);

            // Compute distance
            let sub_aligned: Vec<f64> = argvals_template
                .iter()
                .map(|&t| {
                    interp_target(
                        &sub_target,
                        argvals_template,
                        interp_target(&gamma, argvals_template, t),
                    )
                })
                .collect();
            let q_aligned = srsf_single(&sub_aligned, argvals_template);
            let weights = simpsons_weights(argvals_template);
            let dist = l2_distance(&q_template, &q_aligned, &weights);

            if dist < best_dist {
                best_dist = dist;
                best_start = start;
                best_end = end;
                best_gamma = gamma;
            }

            start += step;
        }
    }

    let total_domain = argvals_target[m_f - 1] - argvals_target[0];
    let match_domain = argvals_target[best_end] - argvals_target[best_start];
    let domain_fraction = if total_domain > 0.0 {
        match_domain / total_domain
    } else {
        1.0
    };

    Ok(PartialMatchResult {
        start_index: best_start,
        end_index: best_end,
        gamma: best_gamma,
        distance: best_dist,
        domain_fraction,
    })
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Linear interpolation of a curve at point `t`.
fn interp_target(values: &[f64], grid: &[f64], t: f64) -> f64 {
    let n = grid.len();
    if n == 0 {
        return 0.0;
    }
    if t <= grid[0] {
        return values[0];
    }
    if t >= grid[n - 1] {
        return values[n - 1];
    }
    // Binary search for the interval
    let mut lo = 0;
    let mut hi = n - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if grid[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let frac = (t - grid[lo]) / (grid[hi] - grid[lo]);
    values[lo] * (1.0 - frac) + values[hi] * frac
}

/// DP alignment for partial matching (same grid for both SRSFs).
fn dp_align_partial(q1: &[f64], q2: &[f64], argvals: &[f64], lambda: f64) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return argvals.to_vec();
    }

    let norm1 = q1.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let norm2 = q2.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    let q1n: Vec<f64> = q1.iter().map(|&v| v / norm1).collect();
    let q2n: Vec<f64> = q2.iter().map(|&v| v / norm2).collect();

    let path = super::dp_grid_solve(m, m, |sr, sc, tr, tc| {
        dp_edge_weight(&q1n, &q2n, argvals, sc, tc, sr, tr)
            + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
    });

    dp_path_to_gamma(&path, argvals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn partial_match_identity() {
        let m = 30;
        let t = uniform_grid(m);
        let f: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();

        let config = PartialMatchConfig {
            min_span: 0.5,
            ..Default::default()
        };
        let result = elastic_partial_match(&f, &f, &t, &t, &config).unwrap();

        assert!(
            result.distance < 0.5,
            "matching a curve to itself should give small distance, got {}",
            result.distance
        );
    }

    #[test]
    fn partial_match_subcurve() {
        let m = 40;
        let t = uniform_grid(m);
        let target: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();

        // Template is roughly the middle portion
        let m_t = 20;
        let t_template = uniform_grid(m_t);
        let template: Vec<f64> = t_template
            .iter()
            .map(|&x| ((x * 0.5 + 0.25) * 6.0).sin())
            .collect();

        let config = PartialMatchConfig {
            min_span: 0.3,
            ..Default::default()
        };
        let result = elastic_partial_match(&template, &target, &t_template, &t, &config).unwrap();

        assert!(result.start_index < result.end_index);
        assert!(result.domain_fraction >= 0.3);
        assert!(result.gamma.len() == m_t);
    }

    #[test]
    fn partial_match_rejects_bad_min_span() {
        let t = uniform_grid(10);
        let f: Vec<f64> = t.iter().map(|&x| x * x).collect();
        let config = PartialMatchConfig {
            min_span: 0.0,
            ..Default::default()
        };
        assert!(elastic_partial_match(&f, &f, &t, &t, &config).is_err());
    }

    #[test]
    fn partial_match_config_default() {
        let config = PartialMatchConfig::default();
        assert!((config.lambda - 0.0).abs() < f64::EPSILON);
        assert!((config.min_span - 0.5).abs() < f64::EPSILON);
    }
}
