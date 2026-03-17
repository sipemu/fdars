//! Bayesian pairwise alignment via pCN MCMC on the Hilbert sphere.

use super::dp_alignment_core;
use super::srsf::{reparameterize_curve, srsf_single};
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::warping::{
    exp_map_sphere, gam_to_psi, inner_product_l2, inv_exp_map_sphere, l2_norm_l2, normalize_warp,
    psi_to_gam,
};

use rand::prelude::*;
use rand_distr::StandardNormal;

// ─── Config / Result ─────────────────────────────────────────────────────────

/// Configuration for Bayesian pairwise alignment.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianAlignConfig {
    /// Number of posterior samples to retain (after burn-in).
    pub n_samples: usize,
    /// Number of burn-in iterations to discard.
    pub burn_in: usize,
    /// pCN step size beta in (0, 1).
    pub step_size: f64,
    /// Variance scaling for random tangent-vector proposals.
    pub proposal_variance: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for BayesianAlignConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            burn_in: 200,
            step_size: 0.1,
            proposal_variance: 1.0,
            seed: 42,
        }
    }
}

/// Result of Bayesian pairwise alignment.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BayesianAlignmentResult {
    /// Posterior warping function samples (n_samples x m), after burn-in.
    pub posterior_gammas: FdMatrix,
    /// Pointwise posterior mean warping function (length m).
    pub posterior_mean_gamma: Vec<f64>,
    /// Pointwise 2.5% credible band (length m).
    pub credible_lower: Vec<f64>,
    /// Pointwise 97.5% credible band (length m).
    pub credible_upper: Vec<f64>,
    /// MCMC acceptance rate.
    pub acceptance_rate: f64,
    /// f2 aligned to f1 using the posterior mean warping function.
    pub f_aligned_mean: Vec<f64>,
}

// ─── Bayesian Alignment ─────────────────────────────────────────────────────

/// Compute the SRSF-based log-likelihood for a warping function.
///
/// `log_lik = -0.5 * sum_j(w[j] * (q1[j] - q2_gamma[j])^2)`
/// where q2_gamma is the SRSF of f2 composed with gamma (with sqrt(gamma') factor).
fn log_likelihood(q1: &[f64], q2: &[f64], argvals: &[f64], gamma: &[f64], weights: &[f64]) -> f64 {
    let m = q1.len();
    let q2_warped = reparameterize_curve(q2, argvals, gamma);

    // Compute gamma' via finite differences
    let mut gamma_dot = vec![0.0; m];
    gamma_dot[0] = (gamma[1] - gamma[0]) / (argvals[1] - argvals[0]);
    for j in 1..(m - 1) {
        gamma_dot[j] = (gamma[j + 1] - gamma[j - 1]) / (argvals[j + 1] - argvals[j - 1]);
    }
    gamma_dot[m - 1] = (gamma[m - 1] - gamma[m - 2]) / (argvals[m - 1] - argvals[m - 2]);

    let mut ll = 0.0;
    for j in 0..m {
        let q2g = q2_warped[j] * gamma_dot[j].max(0.0).sqrt();
        let diff = q1[j] - q2g;
        ll -= 0.5 * weights[j] * diff * diff;
    }
    ll
}

/// Project a vector onto the tangent plane at a point on the sphere.
///
/// Removes the component along `psi_base`: `v - <v, psi_base> * psi_base`
fn project_to_tangent(v: &[f64], psi_base: &[f64], time: &[f64]) -> Vec<f64> {
    let ip = inner_product_l2(v, psi_base, time);
    v.iter()
        .zip(psi_base.iter())
        .map(|(&vi, &pi)| vi - ip * pi)
        .collect()
}

/// Perform Bayesian pairwise alignment of f2 to f1 via pCN MCMC on the
/// Hilbert sphere.
///
/// Uses a preconditioned Crank-Nicolson (pCN) proposal in the tangent space
/// of the identity warping function on the Hilbert sphere. The DP-optimal
/// alignment serves as initialization.
///
/// # Arguments
/// * `f1` — Target curve (length m)
/// * `f2` — Curve to align (length m)
/// * `argvals` — Evaluation points (length m)
/// * `config` — MCMC configuration
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if lengths don't match or m < 2.
/// Returns `FdarError::InvalidParameter` if config values are out of range.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bayesian_align_pair(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    config: &BayesianAlignConfig,
) -> Result<BayesianAlignmentResult, FdarError> {
    let m = f1.len();

    // ── Validation ──────────────────────────────────────────────────────
    if m != f2.len() || m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "f1/f2/argvals",
            expected: format!("all length {m}"),
            actual: format!("f1={}, f2={}, argvals={}", m, f2.len(), argvals.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }
    if config.n_samples == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be > 0".to_string(),
        });
    }
    if config.step_size <= 0.0 || config.step_size >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "step_size",
            message: format!("step_size must be in (0, 1), got {}", config.step_size),
        });
    }

    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    // Compute SRSFs
    let q1 = srsf_single(f1, argvals);
    let q2 = srsf_single(f2, argvals);

    // Simpson's weights for log-likelihood
    let weights = simpsons_weights(argvals);

    // Identity warp psi on sphere: constant 1, normalized
    let psi_id: Vec<f64> = {
        let raw = vec![1.0; m];
        let norm = l2_norm_l2(&raw, &time);
        raw.iter().map(|&v| v / norm).collect()
    };

    // DP initialization
    let gamma_dp = dp_alignment_core(&q1, &q2, argvals, 0.0);
    let gam_01: Vec<f64> = gamma_dp.iter().map(|&g| (g - t0) / domain).collect();
    let mut psi_curr = gam_to_psi(&gam_01, binsize);
    let psi_norm = l2_norm_l2(&psi_curr, &time);
    if psi_norm > 1e-10 {
        for v in &mut psi_curr {
            *v /= psi_norm;
        }
    }

    // Current tangent vector and log-likelihood
    let mut v_curr = inv_exp_map_sphere(&psi_id, &psi_curr, &time);
    let mut ll_curr = log_likelihood(&q1, &q2, argvals, &gamma_dp, &weights);

    let beta = config.step_size;
    let sqrt_1_beta2 = (1.0 - beta * beta).sqrt();
    let total_iter = config.n_samples + config.burn_in;

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut stored_gammas: Vec<Vec<f64>> = Vec::with_capacity(config.n_samples);
    let mut n_accepted = 0usize;

    for iter in 0..total_iter {
        // Generate random tangent vector at identity
        let xi_raw: Vec<f64> = (0..m)
            .map(|_| rng.sample::<f64, _>(StandardNormal))
            .collect();
        let xi_tangent = project_to_tangent(&xi_raw, &psi_id, &time);
        let xi_scaled: Vec<f64> = xi_tangent
            .iter()
            .map(|&v| v * config.proposal_variance.sqrt())
            .collect();

        // pCN proposal: v_prop = sqrt(1 - beta^2) * v_curr + beta * xi
        let v_prop: Vec<f64> = v_curr
            .iter()
            .zip(xi_scaled.iter())
            .map(|(&vc, &xi)| sqrt_1_beta2 * vc + beta * xi)
            .collect();

        // Map to sphere
        let psi_prop = exp_map_sphere(&psi_id, &v_prop, &time);

        // Convert to gamma
        let gam_prop_01 = psi_to_gam(&psi_prop, &time);
        let mut gamma_prop: Vec<f64> = gam_prop_01.iter().map(|&g| t0 + g * domain).collect();
        normalize_warp(&mut gamma_prop, argvals);

        // Log-likelihood of proposal
        let ll_prop = log_likelihood(&q1, &q2, argvals, &gamma_prop, &weights);

        // Accept/reject
        let log_alpha = ll_prop - ll_curr;
        let u: f64 = rng.gen();
        if u.ln() < log_alpha {
            psi_curr = psi_prop;
            v_curr = v_prop;
            ll_curr = ll_prop;
            n_accepted += 1;

            if iter >= config.burn_in {
                stored_gammas.push(gamma_prop);
            }
        } else if iter >= config.burn_in {
            // Store current (rejected proposal keeps previous)
            let gam_curr_01 = psi_to_gam(&psi_curr, &time);
            let mut gamma_curr: Vec<f64> = gam_curr_01.iter().map(|&g| t0 + g * domain).collect();
            normalize_warp(&mut gamma_curr, argvals);
            stored_gammas.push(gamma_curr);
        }
    }

    let n_stored = stored_gammas.len();
    let acceptance_rate = n_accepted as f64 / total_iter as f64;

    // Build posterior gamma matrix
    let mut posterior_gammas = FdMatrix::zeros(n_stored, m);
    for (i, gam) in stored_gammas.iter().enumerate() {
        for j in 0..m {
            posterior_gammas[(i, j)] = gam[j];
        }
    }

    // Pointwise posterior mean
    let mut posterior_mean_gamma = vec![0.0; m];
    for j in 0..m {
        for i in 0..n_stored {
            posterior_mean_gamma[j] += posterior_gammas[(i, j)];
        }
        posterior_mean_gamma[j] /= n_stored as f64;
    }
    normalize_warp(&mut posterior_mean_gamma, argvals);

    // Pointwise credible bands (2.5% and 97.5% quantiles)
    let mut credible_lower = vec![0.0; m];
    let mut credible_upper = vec![0.0; m];
    for j in 0..m {
        let mut col: Vec<f64> = (0..n_stored).map(|i| posterior_gammas[(i, j)]).collect();
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx_lo = ((0.025 * n_stored as f64).floor() as usize).min(n_stored.saturating_sub(1));
        let idx_hi = ((0.975 * n_stored as f64).ceil() as usize).min(n_stored.saturating_sub(1));
        credible_lower[j] = col[idx_lo];
        credible_upper[j] = col[idx_hi];
    }

    // Align f2 using posterior mean gamma
    let f_aligned_mean = reparameterize_curve(f2, argvals, &posterior_mean_gamma);

    Ok(BayesianAlignmentResult {
        posterior_gammas,
        posterior_mean_gamma,
        credible_lower,
        credible_upper,
        acceptance_rate,
        f_aligned_mean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn bayesian_align_identical_curves() {
        let m = 51;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let f2 = f1.clone();

        let config = BayesianAlignConfig {
            n_samples: 200,
            burn_in: 50,
            step_size: 0.1,
            proposal_variance: 0.5,
            seed: 42,
        };
        let result = bayesian_align_pair(&f1, &f2, &t, &config).unwrap();

        // Posterior mean gamma should be close to identity
        for j in 0..m {
            assert!(
                (result.posterior_mean_gamma[j] - t[j]).abs() < 0.15,
                "posterior mean gamma at j={j} deviates too much from identity: {} vs {}",
                result.posterior_mean_gamma[j],
                t[j]
            );
        }

        // Acceptance rate should be reasonable
        assert!(
            result.acceptance_rate > 0.05,
            "acceptance rate too low: {}",
            result.acceptance_rate
        );
    }

    #[test]
    fn bayesian_align_credible_bands_order() {
        let m = 51;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let f2: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * (ti + 0.05)).sin()).collect();

        let config = BayesianAlignConfig {
            n_samples: 200,
            burn_in: 50,
            step_size: 0.15,
            proposal_variance: 0.5,
            seed: 7,
        };
        let result = bayesian_align_pair(&f1, &f2, &t, &config).unwrap();

        for j in 0..m {
            assert!(
                result.credible_lower[j] <= result.posterior_mean_gamma[j] + 1e-10,
                "lower > mean at j={j}: {} > {}",
                result.credible_lower[j],
                result.posterior_mean_gamma[j]
            );
            assert!(
                result.posterior_mean_gamma[j] <= result.credible_upper[j] + 1e-10,
                "mean > upper at j={j}: {} > {}",
                result.posterior_mean_gamma[j],
                result.credible_upper[j]
            );
        }
    }

    #[test]
    fn bayesian_align_shifted_sine() {
        let m = 51;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let shift = 0.1;
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * (ti + shift)).sin())
            .collect();

        let config = BayesianAlignConfig {
            n_samples: 300,
            burn_in: 100,
            step_size: 0.15,
            proposal_variance: 1.0,
            seed: 99,
        };
        let result = bayesian_align_pair(&f1, &f2, &t, &config).unwrap();

        // The aligned curve should be closer to f1 than the original f2
        let error_original: f64 = f1
            .iter()
            .zip(f2.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>();
        let error_aligned: f64 = f1
            .iter()
            .zip(result.f_aligned_mean.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>();

        assert!(
            error_aligned < error_original + 1e-6,
            "aligned error ({error_aligned:.4}) should be <= original ({error_original:.4})"
        );
    }

    #[test]
    fn bayesian_align_rejects_bad_config() {
        let m = 21;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
        let f2 = f1.clone();

        // n_samples = 0
        let config = BayesianAlignConfig {
            n_samples: 0,
            ..BayesianAlignConfig::default()
        };
        assert!(
            bayesian_align_pair(&f1, &f2, &t, &config).is_err(),
            "should reject n_samples=0"
        );

        // step_size = 0
        let config = BayesianAlignConfig {
            step_size: 0.0,
            ..BayesianAlignConfig::default()
        };
        assert!(
            bayesian_align_pair(&f1, &f2, &t, &config).is_err(),
            "should reject step_size=0"
        );

        // step_size = 1
        let config = BayesianAlignConfig {
            step_size: 1.0,
            ..BayesianAlignConfig::default()
        };
        assert!(
            bayesian_align_pair(&f1, &f2, &t, &config).is_err(),
            "should reject step_size=1"
        );
    }
}
