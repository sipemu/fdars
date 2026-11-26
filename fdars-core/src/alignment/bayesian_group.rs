//! Bayesian group (multiple) alignment via Z-mixture pCN MCMC.
//!
//! Jointly samples the warping function of every curve, the template SRSF and
//! the noise variance of the SRSF model
//!
//! ```text
//! q*(t_j) = q_i(γ_i(t_j)) · √γ_i'(t_j) + ε_ij ,   ε_ij ~ N(0, σ²)  iid
//! ```
//!
//! following fdasrsf 2.7.5 `fdawarp.time_warping_bayes` (a port of fdasrvf_MATLAB
//! and fdasrvf 2.6.0 `time_warping_bayes`, Lu, Herbei & Kurtek):
//!
//! * warps live on the Hilbert sphere: `ψ_i = exp_1(v_i)`, `γ_i = ∫ψ_i²`, with the
//!   shooting vector `v_i = Σ_k c_ik φ_k` in a Fourier basis of the tangent space
//!   at the identity and prior `c_ik ~ N(0, (σ_g / j)²)` (`j` = harmonic index);
//! * the template is a quadratic B-spline `q* = Σ_k a_k B_k` with prior
//!   `a_k ~ N(0, (σ_q / ⌈k/2⌉)²)`, `σ_q = 5·(P95 − P5)` of the observed SRSFs;
//! * `σ² ~ InvGamma(α₀, β₀)` is drawn exactly by Gibbs;
//! * warp coefficients (one Metropolis step per curve per iteration) and template
//!   coefficients are moved with a Z-mixture preconditioned Crank–Nicolson proposal
//!   `c' = √(1−β²)·c + β·ξ`, `ξ ~ prior`, `β` drawn from a finite mixture — small
//!   `β` explores locally, large `β` jumps between posterior modes;
//! * every `center_every` iterations the warps are re-centred (Karcher mean of the
//!   warps set to the identity, template warped accordingly), as in the reference.
//!
//! Several chains can be run (chain `k` is seeded with `seed + k`, so results do
//! not depend on thread scheduling); the result pools them and also reports
//! per-chain modes, split-R̂ diagnostics and acceptance rates.
//!
//! # Deviations from fdasrsf
//!
//! * The warp basis is the genuine tangent space at `ψ = 1` (`√2 sin(2πjt)`,
//!   `√2 cos(2πjt)`, `j = 1..M`). fdasrsf's `basis_fourier` puts a constant `√2`
//!   first, which is not tangent to the sphere, so its sampled "warps" need not
//!   end at 1 and the SRSF is extrapolated past the domain.
//! * Proposals with `ψ < 0` somewhere are rejected (exact Metropolis–Hastings for
//!   the prior truncated to valid warps); fdasrsf re-draws until valid.
//! * γ is normalised to end exactly at 1.
//! * By default ([`BayesianGroupConfig::init_dp`]) chains start from the DP alignment
//!   of each curve to the initial template instead of the identity; this changes
//!   only the starting point and avoids chains stuck in poor modes.
//! * By default ([`BayesianGroupConfig::center_draws`]) every retained draw is
//!   re-centred exactly before summarising, so the reported posterior is that of the
//!   identifiable (centred) warps and template; fdasrsf summarises raw states.
//!
//! Random-walk pCN mixes slowly when the posterior is concentrated: check
//! [`BayesianGroupResult::rhat_warps`], [`BayesianGroupResult::rhat_sigma2`],
//! [`BayesianGroupResult::ess_warps`] and [`BayesianGroupResult::mode_spread`] and lengthen
//! `n_iter` if they are not ≈ 1 / large / small.
//!
//! # infHMC sampler
//!
//! [`BayesianSampler::InfHmc`] replaces the warp and template moves (same target posterior)
//! by those of fdasrsf 2.7.5 `time_warping_bayes_infHMC` (fdasrvf 2.6.0
//! `time_warping_bayes_infHMC`): each curve's warp coefficients are moved by
//! infinite-dimensional (prior-preconditioned) Hamiltonian Monte Carlo (Beskos et al. 2011)
//! with the exact gradient of the SRSF likelihood — the Gaussian prior part of the dynamics
//! is integrated exactly by a rotation, the likelihood enters through half kicks
//! `v ← v − ε/2 · C∇Φ` — with `1..=L` leapfrog steps and per-curve step sizes adapted by
//! dual averaging during burn-in; the template is drawn exactly from its Gaussian full
//! conditional. It typically gives a much higher effective sample size per iteration than
//! pCN. Differences from fdasrsf's infHMC: the warp prior, template basis (quadratic
//! B-splines with the pCN prior) and centring schedule are those of this module, so both
//! samplers target the same posterior — given [`BayesianGroupConfig::injective_warps`],
//! which infHMC always applies (as fdasrsf's infHMC, `‖v‖ < π`): `exp_1` is 2π-periodic in
//! `‖v‖`, so without it every warp has further coefficient preimages near `‖v‖ ≈ 2πk`
//! that pCN's large steps visit but local HMC moves cannot reach.
//!
//! infHMC moves locally: a chain that settles in a poor, well-separated mode (e.g. a bump
//! aligned to the wrong peak) stays there. Run several chains and check
//! [`BayesianGroupResult::rhat_sigma2`], [`BayesianGroupResult::mode_spread`] and
//! [`BayesianGroupResult::multimodal_curves`] (Sarle's bimodality coefficient of the pooled
//! per-curve warp draws), which flag such chains.

use super::dp_alignment_core;
use super::karcher::sqrt_mean_inverse;
use super::srsf::{reparameterize_curve, srsf_single};
use crate::error::FdarError;
use crate::helpers::{gradient_uniform, linear_interp, trapz};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::warping::{inv_exp_map_sphere, l2_norm_l2};

use rand::prelude::*;
use rand_distr::{Gamma, StandardNormal};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Config / Result ─────────────────────────────────────────────────────────

/// MCMC kernel used by [`bayesian_align_group`] for the warps and the template.
///
/// # Example
/// ```
/// use fdars_core::alignment::{bayesian_align_group, BayesianGroupConfig, BayesianSampler};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 31;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let mut data = FdMatrix::zeros(3, m);
/// for (i, s) in [-0.04, 0.0, 0.04].iter().enumerate() {
///     for j in 0..m {
///         data[(i, j)] = (-(t[j] + s - 0.5).powi(2) / 0.02).exp();
///     }
/// }
/// let mut cfg = BayesianGroupConfig::default();
/// cfg.sampler = BayesianSampler::InfHmc;
/// cfg.n_iter = 300;
/// cfg.burn_in = 150;
/// cfg.n_warp_harmonics = 3;
/// cfg.n_template_basis = 12;
/// cfg.n_chains = 2;
/// let res = bayesian_align_group(&data, &t, &cfg).unwrap();
/// // Step sizes were adapted per curve; acceptance is near the 0.65 target.
/// assert_eq!(res.chains[0].warp_step_sizes.len(), 3);
/// assert!(res.warp_acceptance.iter().all(|&a| a > 0.3 && a < 0.95));
/// assert_eq!(res.ess_warps.len(), 3);
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BayesianSampler {
    /// Z-mixture preconditioned Crank–Nicolson Metropolis moves for warps and template
    /// (fdasrsf `time_warping_bayes`). The default.
    #[default]
    Pcn,
    /// Infinite-dimensional (prior-preconditioned) HMC for each curve's warp with the
    /// exact likelihood gradient, conjugate Gibbs draw of the template
    /// (fdasrsf `time_warping_bayes_infHMC`).
    InfHmc,
}

/// Configuration for [`bayesian_align_group`].
///
/// Defaults follow fdasrsf 2.7.5 `time_warping_bayes` (`iter = 2000`,
/// `burnin = iter / 2`, `alpha0 = beta0 = 0.1`, Z-mixture
/// `betas = [0.5, 0.5, 0.005, 0.0001]`, `probs = [0.1, 0.1, 0.7, 0.1]`,
/// `propvar = 4`, 10 warp harmonics, 30 template basis functions,
/// centring every 100 iterations), except that the sampling grid defaults to the
/// input grid (`n_points = 0`) instead of 200 points, chains start from the DP
/// alignment (`init_dp = true`) and summaries use re-centred draws
/// (`center_draws = true`); set both to `false` for fdasrsf's behaviour.
///
/// Construct via `BayesianGroupConfig::default()`, then assign the fields you need
/// (e.g. `let mut c = BayesianGroupConfig::default(); c.n_chains = 4;`). This struct
/// is `#[non_exhaustive]`, so external crates cannot build it with a struct literal —
/// not even functional-update `..Default::default()` form.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianGroupConfig {
    /// Total MCMC iterations per chain, including burn-in.
    pub n_iter: usize,
    /// Iterations discarded at the start of each chain (`1 ≤ burn_in < n_iter`).
    pub burn_in: usize,
    /// Keep every `thin`-th post-burn-in draw (`≥ 1`).
    pub thin: usize,
    /// Number of independent chains (`≥ 1`); chain `k` uses seed `seed + k`.
    pub n_chains: usize,
    /// Base RNG seed.
    pub seed: u64,
    /// Inverse-gamma prior shape for σ² (`> 0`).
    pub alpha0: f64,
    /// Inverse-gamma prior scale for σ² (`> 0`).
    pub beta0: f64,
    /// Z-mixture pCN step sizes, each in `(0, 1]`.
    pub pcn_betas: Vec<f64>,
    /// Mixture probabilities of the step sizes (non-negative, positive sum; normalised).
    pub pcn_probs: Vec<f64>,
    /// Prior standard deviation σ_g of the first warp harmonic (fdasrsf `propvar`).
    pub warp_prior_sd: f64,
    /// Number M of Fourier harmonics of the warp shooting vector (2M coefficients).
    pub n_warp_harmonics: usize,
    /// Number of quadratic B-spline basis functions of the template SRSF (`≥ 3`).
    pub n_template_basis: usize,
    /// Size T of the uniform sampling grid on \[0, 1\]; `0` uses the input grid size.
    pub n_points: usize,
    /// Re-centre warps / template every this many iterations (`0` disables).
    pub center_every: usize,
    /// Level of the pointwise credible bands, in `(0, 1)`.
    pub credible_level: f64,
    /// Summarise re-centred draws: each retained draw's warps are composed with the
    /// inverse of their Karcher mean (and its template warped to match) before the
    /// posterior summaries are formed. This removes the joint template/warp
    /// non-identifiability from the reported posterior (exact, no re-projection).
    /// `false` summarises the raw chain states, as fdasrsf does.
    pub center_draws: bool,
    /// Start every chain from the DP (dynamic-programming) alignment of each curve
    /// to the initial template, projected onto the warp basis and centred. Only the
    /// starting point changes (the target posterior is the same); it avoids long
    /// burn-in and chains stuck in poor modes. `false` starts at the identity warps,
    /// as fdasrsf does (`initcoef = 0`).
    pub init_dp: bool,
    /// MCMC kernel (default [`BayesianSampler::Pcn`]). The `hmc_*` fields apply only to
    /// [`BayesianSampler::InfHmc`]; `pcn_*` only to [`BayesianSampler::Pcn`].
    pub sampler: BayesianSampler,
    /// Initial infHMC step size ε (rotation angle), in `(0, π/2]`; fdasrsf `√h = 0.1`.
    pub hmc_step_size: f64,
    /// Maximum number L of leapfrog steps; each transition uses a uniform draw from `1..=L`.
    pub hmc_max_leapfrog: usize,
    /// Adapt each curve's step size by dual averaging during burn-in (then frozen).
    pub hmc_adapt: bool,
    /// Target acceptance probability of the step-size adaptation, in `(0, 1)`.
    pub hmc_target_accept: f64,
    /// Truncate the warp prior to the injectivity radius of the shooting representation,
    /// `‖v‖ < π` (fdasrsf `time_warping_bayes_infHMC`). Without it (fdasrsf
    /// `time_warping_bayes`) a warp has several coefficient preimages — `exp_1(v)` is
    /// 2π-periodic in `‖v‖` — so the coefficient posterior has shells at `‖v‖ ≈ 2πk`
    /// that pCN's large steps visit and that change the induced prior on γ. Always on
    /// for [`BayesianSampler::InfHmc`], whose local moves cannot cross between shells;
    /// default `false` for pCN (unchanged behaviour).
    pub injective_warps: bool,
}

impl Default for BayesianGroupConfig {
    fn default() -> Self {
        Self {
            n_iter: 2000,
            burn_in: 1000,
            thin: 1,
            n_chains: 1,
            seed: 42,
            alpha0: 0.1,
            beta0: 0.1,
            pcn_betas: vec![0.5, 0.5, 0.005, 0.0001],
            pcn_probs: vec![0.1, 0.1, 0.7, 0.1],
            warp_prior_sd: 4.0,
            n_warp_harmonics: 10,
            n_template_basis: 30,
            n_points: 0,
            center_every: 100,
            credible_level: 0.95,
            center_draws: true,
            init_dp: true,
            sampler: BayesianSampler::Pcn,
            hmc_step_size: 0.1,
            hmc_max_leapfrog: 10,
            hmc_adapt: true,
            hmc_target_accept: 0.65,
            injective_warps: false,
        }
    }
}

/// Per-chain output of [`bayesian_align_group`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BayesianGroupChain {
    /// Seed this chain was run with (`config.seed + k`).
    pub seed: u64,
    /// Chain posterior-mean warps (ψ-mean on the sphere), n × m, on the `argvals` scale.
    pub posterior_mean_gammas: FdMatrix,
    /// Warps of the retained draw with the highest log-likelihood (the chain's mode), n × m.
    pub mode_gammas: FdMatrix,
    /// Log-likelihood of that draw.
    pub mode_log_likelihood: f64,
    /// Template SRSF of that draw (length m, `argvals` scale).
    pub mode_template_srsf: Vec<f64>,
    /// Chain posterior mean of the template SRSF (length m, `argvals` scale).
    pub template_srsf_mean: Vec<f64>,
    /// Per-curve warp acceptance rate over all iterations after the first.
    pub warp_acceptance: Vec<f64>,
    /// Template acceptance rate over all iterations after the first.
    pub template_acceptance: f64,
    /// σ² of every retained draw.
    pub sigma2_trace: Vec<f64>,
    /// Log-likelihood of every retained draw.
    pub log_likelihood_trace: Vec<f64>,
    /// Final (adapted) infHMC step size of each curve; empty for the pCN sampler.
    pub warp_step_sizes: Vec<f64>,
}

/// Result of [`bayesian_align_group`]: posterior summaries pooled over all chains.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BayesianGroupResult {
    /// Posterior-mean warps (normalised ψ-mean, mapped back to γ), n × m, `argvals` scale.
    pub gammas: FdMatrix,
    /// Pointwise posterior-median warps, n × m.
    pub gammas_median: FdMatrix,
    /// Lower pointwise credible band of the warps, n × m.
    pub gammas_lower: FdMatrix,
    /// Upper pointwise credible band of the warps, n × m.
    pub gammas_upper: FdMatrix,
    /// Curves aligned with the posterior-mean warps, `f_i ∘ γ_i`, n × m.
    pub aligned_data: FdMatrix,
    /// Posterior mean of the template SRSF (length m, `argvals` scale).
    pub template_srsf: Vec<f64>,
    /// Lower pointwise credible band of the template SRSF.
    pub template_srsf_lower: Vec<f64>,
    /// Upper pointwise credible band of the template SRSF.
    pub template_srsf_upper: Vec<f64>,
    /// Template function reconstructed from the posterior-mean SRSF,
    /// `mean_i f_i(t_0) + ∫ q|q|` (as fdasrsf `fmean`).
    pub template: Vec<f64>,
    /// Lower pointwise credible band of the template function.
    pub template_lower: Vec<f64>,
    /// Upper pointwise credible band of the template function.
    pub template_upper: Vec<f64>,
    /// Posterior mean of the SRSF noise variance σ² (unit-domain SRSF scale).
    pub sigma2_mean: f64,
    /// Lower credible bound of σ².
    pub sigma2_lower: f64,
    /// Upper credible bound of σ².
    pub sigma2_upper: f64,
    /// Per-curve warp acceptance rate averaged over chains.
    pub warp_acceptance: Vec<f64>,
    /// Template acceptance rate averaged over chains.
    pub template_acceptance: f64,
    /// Split-R̂ of σ² across chains (single chain: its two halves).
    pub rhat_sigma2: f64,
    /// Split-R̂ of the log-likelihood.
    pub rhat_log_likelihood: f64,
    /// Per-curve split-R̂ of the warp summary `∫ (γ_i(t) − t) dt` (unit domain).
    pub rhat_warps: Vec<f64>,
    /// Per-curve between-chain spread of the chain posterior-mean warps:
    /// max over chain pairs of `‖γ̄_a − γ̄_b‖_{L²[0,1]}` (0 for one chain).
    /// Large values flag a multimodal posterior.
    pub mode_spread: Vec<f64>,
    /// Pooled warp draws, one matrix per curve (n_draws × m, `argvals` scale).
    pub gamma_draws: Vec<FdMatrix>,
    /// Pooled template-SRSF draws (n_draws × m, `argvals` scale).
    pub template_srsf_draws: FdMatrix,
    /// Pooled σ² draws (length n_draws).
    pub sigma2_draws: Vec<f64>,
    /// Per-chain summaries.
    pub chains: Vec<BayesianGroupChain>,
    /// Effective sample size of σ² (Geyer initial monotone sequence, summed over chains).
    pub ess_sigma2: f64,
    /// Effective sample size of the log-likelihood.
    pub ess_log_likelihood: f64,
    /// Per-curve effective sample size of the warp summary `∫ (γ_i(t) − t) dt`.
    pub ess_warps: Vec<f64>,
    /// Per-curve Sarle bimodality coefficient `(g² + 1) / (κ + 3(n−1)²/((n−2)(n−3)))` of the
    /// pooled draws of `∫ (γ_i(t) − t) dt` (g, κ: bias-corrected skewness and excess
    /// kurtosis). About 1/3 for a Gaussian posterior; values above 5/9 suggest a bimodal
    /// warp posterior (or chains stuck in different modes). NaN for degenerate samples.
    pub warp_bimodality: Vec<f64>,
    /// Curves flagged as multimodal: `warp_bimodality > 5/9`.
    pub multimodal_curves: Vec<bool>,
}

// ─── Validation ──────────────────────────────────────────────────────────────

fn invalid(parameter: &'static str, message: String) -> FdarError {
    FdarError::InvalidParameter { parameter, message }
}

fn validate_dims(data: &FdMatrix, argvals: &[f64]) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (data columns)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if n < 1 || m < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 1 curves and m >= 4 points".to_string(),
            actual: format!("{n} x {m}"),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) || argvals.iter().any(|v| !v.is_finite()) {
        return Err(invalid(
            "argvals",
            "must be finite and strictly increasing".to_string(),
        ));
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(invalid("data", "contains non-finite values".to_string()));
    }
    Ok(())
}

fn validate_chain_config(c: &BayesianGroupConfig) -> Result<(), FdarError> {
    if c.burn_in < 1 || c.burn_in >= c.n_iter {
        return Err(invalid(
            "burn_in",
            format!(
                "must satisfy 1 <= burn_in < n_iter, got burn_in={}, n_iter={}",
                c.burn_in, c.n_iter
            ),
        ));
    }
    if c.thin == 0 {
        return Err(invalid("thin", "must be >= 1".to_string()));
    }
    if c.n_chains == 0 {
        return Err(invalid("n_chains", "must be >= 1".to_string()));
    }
    if !(c.alpha0 > 0.0 && c.beta0 > 0.0) {
        return Err(invalid(
            "alpha0/beta0",
            format!("must be > 0, got {} / {}", c.alpha0, c.beta0),
        ));
    }
    if !(c.credible_level > 0.0 && c.credible_level < 1.0) {
        return Err(invalid(
            "credible_level",
            format!("must be in (0, 1), got {}", c.credible_level),
        ));
    }
    if c.sampler == BayesianSampler::InfHmc {
        validate_hmc_config(c)?;
    }
    Ok(())
}

fn validate_hmc_config(c: &BayesianGroupConfig) -> Result<(), FdarError> {
    if !(c.hmc_step_size > 0.0 && c.hmc_step_size <= std::f64::consts::FRAC_PI_2) {
        return Err(invalid(
            "hmc_step_size",
            format!("must be in (0, pi/2], got {}", c.hmc_step_size),
        ));
    }
    if c.hmc_max_leapfrog == 0 {
        return Err(invalid("hmc_max_leapfrog", "must be >= 1".to_string()));
    }
    if !(c.hmc_target_accept > 0.0 && c.hmc_target_accept < 1.0) {
        return Err(invalid(
            "hmc_target_accept",
            format!("must be in (0, 1), got {}", c.hmc_target_accept),
        ));
    }
    Ok(())
}

fn validate_proposal_config(c: &BayesianGroupConfig, t_len: usize) -> Result<(), FdarError> {
    validate_pcn_mixture(c)?;
    validate_bases(c, t_len)
}

fn validate_pcn_mixture(c: &BayesianGroupConfig) -> Result<(), FdarError> {
    if c.sampler != BayesianSampler::Pcn {
        return Ok(());
    }
    if c.pcn_betas.is_empty() || c.pcn_betas.len() != c.pcn_probs.len() {
        return Err(invalid(
            "pcn_betas/pcn_probs",
            format!(
                "must be non-empty and of equal length, got {} / {}",
                c.pcn_betas.len(),
                c.pcn_probs.len()
            ),
        ));
    }
    if c.pcn_betas.iter().any(|&b| !(b > 0.0 && b <= 1.0)) {
        return Err(invalid(
            "pcn_betas",
            "each beta must be in (0, 1]".to_string(),
        ));
    }
    let psum: f64 = c.pcn_probs.iter().sum();
    if c.pcn_probs.iter().any(|&p| !(p >= 0.0 && p.is_finite())) || psum.is_nan() || psum <= 0.0 {
        return Err(invalid(
            "pcn_probs",
            "must be non-negative with a positive sum".to_string(),
        ));
    }
    Ok(())
}

fn validate_bases(c: &BayesianGroupConfig, t_len: usize) -> Result<(), FdarError> {
    if !(c.warp_prior_sd > 0.0 && c.warp_prior_sd.is_finite()) {
        return Err(invalid(
            "warp_prior_sd",
            "must be finite and > 0".to_string(),
        ));
    }
    if c.n_warp_harmonics == 0 || 2 * c.n_warp_harmonics + 1 >= t_len {
        return Err(invalid(
            "n_warp_harmonics",
            format!(
                "must satisfy 1 <= M and 2M + 1 < T = {t_len}, got M = {}",
                c.n_warp_harmonics
            ),
        ));
    }
    if c.n_template_basis < 3 || c.n_template_basis > t_len {
        return Err(invalid(
            "n_template_basis",
            format!("must be in [3, T = {t_len}], got {}", c.n_template_basis),
        ));
    }
    Ok(())
}

// ─── Bases and numerical helpers ─────────────────────────────────────────────

/// Orthonormal Fourier basis of the tangent space at ψ = 1:
/// `√2 sin(2πjt)`, `√2 cos(2πjt)`, j = 1..M (2M vectors of length T).
fn warp_tangent_basis(t: &[f64], n_harm: usize) -> Vec<Vec<f64>> {
    let s2 = std::f64::consts::SQRT_2;
    let tau = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(2 * n_harm);
    for j in 1..=n_harm {
        let w = tau * j as f64;
        out.push(t.iter().map(|&x| s2 * (w * x).sin()).collect());
        out.push(t.iter().map(|&x| s2 * (w * x).cos()).collect());
    }
    out
}

/// fdasrsf `create_basismatrix(x, nbasis, norder)`: B-splines of order `order` on
/// [0, 1] with `nbasis − order + 2` equally spaced breakpoints and `order`-fold
/// boundary knots. Returned column-wise (`nbasis` vectors of length T).
fn template_bspline_basis(t: &[f64], nbasis: usize, order: usize) -> Vec<Vec<f64>> {
    let nbreaks = nbasis - order + 2;
    let mut knots = vec![0.0; order - 1];
    knots.extend((0..nbreaks).map(|i| i as f64 / (nbreaks - 1) as f64));
    knots.extend(std::iter::repeat(1.0).take(order - 1));
    let mut cols = vec![vec![0.0; t.len()]; nbasis];
    for (r, &x) in t.iter().enumerate() {
        let (span, vals) = bspline_nonzero(&knots, order, x);
        for (k, v) in vals.into_iter().enumerate() {
            cols[span + 1 - order + k][r] = v;
        }
    }
    cols
}

/// Non-zero B-spline values at `x` (Piegl & Tiller A2.2). Returns the knot span
/// `s` (`knots[s] <= x < knots[s+1]`, last non-degenerate span at the right end)
/// and the `order` values of basis functions `s+1-order ..= s`.
fn bspline_nonzero(knots: &[f64], order: usize, x: f64) -> (usize, Vec<f64>) {
    let p = order - 1;
    let n_basis = knots.len() - order;
    let mut span = p;
    while span + 1 < n_basis && x >= knots[span + 1] {
        span += 1;
    }
    let mut vals = vec![0.0; order];
    let mut left = vec![0.0; order];
    let mut right = vec![0.0; order];
    vals[0] = 1.0;
    for j in 1..=p {
        left[j] = x - knots[span + 1 - j];
        right[j] = knots[span + j] - x;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = vals[r] / (right[r + 1] + left[j - r]);
            vals[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        vals[j] = saved;
    }
    (span, vals)
}

/// Trapezoidal cumulative integral (starts at 0).
fn cumtrapz(y: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; y.len()];
    for j in 1..y.len() {
        out[j] = out[j - 1] + 0.5 * (y[j] + y[j - 1]) * (x[j] - x[j - 1]);
    }
    out
}

/// Linear-interpolation quantile of an ascending slice (numpy default).
fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// (lower, median, upper) of a sample at the given credible level.
fn sample_quantiles(values: &mut [f64], level: f64) -> (f64, f64, f64) {
    values.sort_by(f64::total_cmp);
    let a = 0.5 * (1.0 - level);
    (
        quantile_sorted(values, a),
        quantile_sorted(values, 0.5),
        quantile_sorted(values, 1.0 - a),
    )
}

/// Standard-normal vector scaled elementwise by `sd`.
fn gaussian_vec(rng: &mut StdRng, sd: &[f64]) -> Vec<f64> {
    sd.iter()
        .map(|&s| s * rng.sample::<f64, _>(StandardNormal))
        .collect()
}

fn sse(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Least-squares solver for the template coefficients (pseudo-inverse of B).
fn pseudo_inverse(cols: &[Vec<f64>]) -> Result<nalgebra::DMatrix<f64>, FdarError> {
    let t_len = cols[0].len();
    let b = nalgebra::DMatrix::from_fn(t_len, cols.len(), |r, c| cols[c][r]);
    b.pseudo_inverse(1e-12)
        .map_err(|e| FdarError::ComputationFailed {
            operation: "bayesian_align_group",
            detail: format!("template basis pseudo-inverse failed: {e}"),
        })
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// Shared, immutable sampling problem.
struct Problem {
    t: Vec<f64>,
    q: Vec<Vec<f64>>,
    warp_basis: Vec<Vec<f64>>,
    warp_sd: Vec<f64>,
    tmpl_basis: Vec<Vec<f64>>,
    tmpl_sd: Vec<f64>,
    tmpl_pinv: nalgebra::DMatrix<f64>,
    betas: Vec<f64>,
    cum_probs: Vec<f64>,
    /// Warp prior variances (infHMC preconditioner).
    warp_var: Vec<f64>,
    /// `BᵀB` of the template basis (conjugate template draw).
    btb: nalgebra::DMatrix<f64>,
    /// Restrict shooting vectors to the injectivity radius `‖v‖ < π`.
    injective: bool,
}

/// Current warp of one curve.
#[derive(Clone)]
struct Warp {
    coef: Vec<f64>,
    psi: Vec<f64>,
    gamma: Vec<f64>,
    /// `q_i(γ) · ψ`, the curve's SRSF warped onto the template.
    warped: Vec<f64>,
}

impl Problem {
    fn new(data: &FdMatrix, argvals: &[f64], cfg: &BayesianGroupConfig) -> Result<Self, FdarError> {
        let (n, m) = data.shape();
        let t_len = if cfg.n_points == 0 { m } else { cfg.n_points };
        if t_len < 4 {
            return Err(invalid(
                "n_points",
                format!("must be 0 or >= 4, got {t_len}"),
            ));
        }
        validate_proposal_config(cfg, t_len)?;
        let (a, b) = (argvals[0], argvals[m - 1]);
        let tn: Vec<f64> = argvals.iter().map(|&x| (x - a) / (b - a)).collect();
        let t: Vec<f64> = (0..t_len).map(|j| j as f64 / (t_len - 1) as f64).collect();
        let q: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let fi = data.row(i);
                let f0: Vec<f64> = t.iter().map(|&x| linear_interp(&tn, &fi, x)).collect();
                srsf_single(&f0, &t)
            })
            .collect();
        let mut all: Vec<f64> = q.iter().flatten().copied().collect();
        all.sort_by(f64::total_cmp);
        let sigma_q = 5.0 * (quantile_sorted(&all, 0.95) - quantile_sorted(&all, 0.05));
        let sigma_q = if sigma_q > 0.0 { sigma_q } else { 1.0 };
        let nh = cfg.n_warp_harmonics;
        let warp_sd: Vec<f64> = (0..2 * nh)
            .map(|k| cfg.warp_prior_sd / (k / 2 + 1) as f64)
            .collect();
        let nb = cfg.n_template_basis;
        let tmpl_sd = (0..nb).map(|k| sigma_q / (k / 2 + 1) as f64).collect();
        let tmpl_basis = template_bspline_basis(&t, nb, 3);
        let tmpl_pinv = pseudo_inverse(&tmpl_basis)?;
        let psum: f64 = cfg.pcn_probs.iter().sum();
        let mut acc = 0.0;
        let cum_probs = cfg
            .pcn_probs
            .iter()
            .map(|&p| {
                acc += p / psum;
                acc
            })
            .collect();
        let warp_var = warp_sd.iter().map(|s: &f64| s * s).collect();
        let btb = nalgebra::DMatrix::from_fn(nb, nb, |a, b| {
            tmpl_basis[a]
                .iter()
                .zip(&tmpl_basis[b])
                .map(|(x, y)| x * y)
                .sum()
        });
        Ok(Self {
            warp_var,
            btb,
            injective: cfg.injective_warps || cfg.sampler == BayesianSampler::InfHmc,
            warp_basis: warp_tangent_basis(&t, nh),
            t,
            q,
            warp_sd,
            tmpl_basis,
            tmpl_sd,
            tmpl_pinv,
            betas: cfg.pcn_betas.clone(),
            cum_probs,
        })
    }

    fn n_curves(&self) -> usize {
        self.q.len()
    }

    fn t_len(&self) -> usize {
        self.t.len()
    }

    /// Z-mixture pCN proposal `√(1−β²)·c + β·ξ`, `ξ ~ N(0, diag(sd²))`.
    fn propose(&self, coef: &[f64], sd: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let z: f64 = rng.gen();
        let idx = self
            .cum_probs
            .iter()
            .position(|&c| z <= c)
            .unwrap_or(self.betas.len() - 1);
        let beta = self.betas[idx];
        let shrink = (1.0 - beta * beta).sqrt();
        let xi = gaussian_vec(rng, sd);
        coef.iter()
            .zip(&xi)
            .map(|(&c, &x)| shrink * c + beta * x)
            .collect()
    }

    fn expand(basis: &[Vec<f64>], coef: &[f64], t_len: usize) -> Vec<f64> {
        let mut out = vec![0.0; t_len];
        for (col, &c) in basis.iter().zip(coef) {
            if c != 0.0 {
                for (o, &v) in out.iter_mut().zip(col) {
                    *o += c * v;
                }
            }
        }
        out
    }

    /// Warp of curve `i` from coefficients; `None` if ψ is negative somewhere.
    fn warp_from_coef(&self, i: usize, coef: Vec<f64>) -> Option<Warp> {
        let v = Self::expand(&self.warp_basis, &coef, self.t_len());
        let r = l2_norm_l2(&v, &self.t);
        if self.injective && r >= std::f64::consts::PI {
            return None;
        }
        let psi_raw: Vec<f64> = if r < 1e-12 {
            vec![1.0; self.t_len()]
        } else {
            let (c, s) = (r.cos(), r.sin() / r);
            v.iter().map(|&x| c + s * x).collect()
        };
        if psi_raw.iter().any(|&p| p < 0.0 || !p.is_finite()) {
            return None;
        }
        let sq: Vec<f64> = psi_raw.iter().map(|p| p * p).collect();
        let mut gamma = cumtrapz(&sq, &self.t);
        let total = gamma[self.t_len() - 1];
        if total <= 0.0 {
            return None;
        }
        gamma.iter_mut().for_each(|g| *g /= total);
        let scale = total.sqrt().recip();
        let psi: Vec<f64> = psi_raw.iter().map(|p| p * scale).collect();
        let qg = reparameterize_curve(&self.q[i], &self.t, &gamma);
        let warped = qg.iter().zip(&psi).map(|(&a, &b)| a * b).collect();
        Some(Warp {
            coef,
            psi,
            gamma,
            warped,
        })
    }

    /// Warp coefficients of a unit-grid warp: shooting vector `exp_1⁻¹(√γ')`
    /// projected (L², trapezoidal) onto the orthonormal warp basis.
    fn coef_from_gamma(&self, gamma: &[f64]) -> Vec<f64> {
        let ones = vec![1.0; self.t_len()];
        let v = inv_exp_map_sphere(&ones, &unit_psi(gamma, &self.t), &self.t);
        self.warp_basis
            .iter()
            .map(|phi| {
                let prod: Vec<f64> = phi.iter().zip(&v).map(|(&a, &b)| a * b).collect();
                trapz(&prod, &self.t)
            })
            .collect()
    }

    fn template_from_coef(&self, coef: &[f64]) -> Vec<f64> {
        Self::expand(&self.tmpl_basis, coef, self.t_len())
    }

    fn template_coef(&self, q_star: &[f64]) -> Vec<f64> {
        let v = nalgebra::DVector::from_column_slice(q_star);
        (&self.tmpl_pinv * v).iter().copied().collect()
    }

    fn log_likelihood(&self, sse_total: f64, sigma2: f64) -> f64 {
        let nt = (self.n_curves() * self.t_len()) as f64;
        -0.5 * nt * (2.0 * std::f64::consts::PI * sigma2).ln() - sse_total / (2.0 * sigma2)
    }
}

/// Inverse Karcher mean `γ_I` of a set of unit-grid warps (fdasrvf `SqrtMeanInverse`).
fn centering_warp(warps: &[Warp], t: &[f64]) -> Vec<f64> {
    let mut gm = FdMatrix::zeros(warps.len(), t.len());
    for (i, w) in warps.iter().enumerate() {
        for (j, &g) in w.gamma.iter().enumerate() {
            gm[(i, j)] = g;
        }
    }
    sqrt_mean_inverse(&gm, t)
}

/// Group action on an SRSF: `(q, γ) = q(γ) · √γ'` (unit grid).
fn warp_srsf(q: &[f64], t: &[f64], gamma: &[f64]) -> Vec<f64> {
    let h = 1.0 / (t.len() - 1) as f64;
    let dg = gradient_uniform(gamma, h);
    reparameterize_curve(q, t, gamma)
        .iter()
        .zip(&dg)
        .map(|(&v, &d)| v * d.max(0.0).sqrt())
        .collect()
}

/// `ψ = √γ'` normalised to unit L² norm (unit grid).
fn unit_psi(gamma: &[f64], t: &[f64]) -> Vec<f64> {
    let h = 1.0 / (t.len() - 1) as f64;
    let mut psi: Vec<f64> = gradient_uniform(gamma, h)
        .iter()
        .map(|&d| d.max(0.0).sqrt())
        .collect();
    let nrm = l2_norm_l2(&psi, t);
    if nrm > 0.0 {
        psi.iter_mut().for_each(|p| *p /= nrm);
    }
    psi
}

// ─── One chain ───────────────────────────────────────────────────────────────

/// One retained draw.
struct Draw {
    warp_coefs: Vec<Vec<f64>>,
    template: Vec<f64>,
    sigma2: f64,
    log_lik: f64,
}

struct ChainState {
    warps: Vec<Warp>,
    tmpl_coef: Vec<f64>,
    template: Vec<f64>,
    sigma2: f64,
    sse: Vec<f64>,
}

struct ChainRun {
    seed: u64,
    draws: Vec<Draw>,
    warp_accept: Vec<usize>,
    tmpl_accept: usize,
    n_updates: usize,
    step_sizes: Vec<f64>,
}

impl ChainState {
    fn init(prob: &Problem, init_dp: bool) -> Result<Self, FdarError> {
        let n = prob.n_curves();
        // Template start: the curve SRSF closest to the pointwise mean SRSF.
        let t_len = prob.t_len();
        let mean: Vec<f64> = (0..t_len)
            .map(|j| prob.q.iter().map(|q| q[j]).sum::<f64>() / n as f64)
            .collect();
        let medoid = (0..n)
            .min_by(|&a, &b| sse(&prob.q[a], &mean).total_cmp(&sse(&prob.q[b], &mean)))
            .unwrap_or(0);
        let tmpl_coef = prob.template_coef(&prob.q[medoid]);
        let template = prob.template_from_coef(&tmpl_coef);
        if template.iter().any(|v| !v.is_finite()) {
            return Err(FdarError::ComputationFailed {
                operation: "bayesian_align_group",
                detail: "non-finite initial template".to_string(),
            });
        }
        let identity = |i: usize| {
            prob.warp_from_coef(i, vec![0.0; prob.warp_basis.len()])
                .expect("identity warp is always valid")
        };
        let warps: Vec<Warp> = (0..n)
            .map(|i| {
                if !init_dp {
                    return identity(i);
                }
                // DP alignment to the initial template, projected onto the warp basis.
                let gamma = dp_alignment_core(&template, &prob.q[i], &prob.t, 0.0);
                prob.warp_from_coef(i, prob.coef_from_gamma(&gamma))
                    .unwrap_or_else(|| identity(i))
            })
            .collect();
        let sse = warps.iter().map(|w| sse(&template, &w.warped)).collect();
        let mut state = Self {
            warps,
            tmpl_coef,
            template,
            sigma2: 1.0,
            sse,
        };
        if init_dp {
            state.center(prob);
        }
        Ok(state)
    }

    fn update_warps(&mut self, prob: &Problem, rng: &mut StdRng, accepts: &mut [usize]) {
        for i in 0..prob.n_curves() {
            let coef = prob.propose(&self.warps[i].coef, &prob.warp_sd, rng);
            let cand = prob.warp_from_coef(i, coef);
            let u: f64 = rng.gen();
            let Some(w) = cand else { continue };
            let sse_prop = sse(&self.template, &w.warped);
            let log_ratio = -(sse_prop - self.sse[i]) / (2.0 * self.sigma2);
            if u.ln() <= log_ratio.min(0.0) {
                self.warps[i] = w;
                self.sse[i] = sse_prop;
                accepts[i] += 1;
            }
        }
    }

    fn update_template(&mut self, prob: &Problem, rng: &mut StdRng) -> bool {
        let coef = prob.propose(&self.tmpl_coef, &prob.tmpl_sd, rng);
        let tmpl = prob.template_from_coef(&coef);
        let sse_prop: Vec<f64> = self.warps.iter().map(|w| sse(&tmpl, &w.warped)).collect();
        let delta: f64 = sse_prop.iter().sum::<f64>() - self.sse.iter().sum::<f64>();
        let u: f64 = rng.gen();
        if u.ln() <= (-delta / (2.0 * self.sigma2)).min(0.0) {
            self.tmpl_coef = coef;
            self.template = tmpl;
            self.sse = sse_prop;
            return true;
        }
        false
    }

    fn update_sigma2(
        &mut self,
        prob: &Problem,
        cfg: &BayesianGroupConfig,
        rng: &mut StdRng,
    ) -> Result<(), FdarError> {
        let shape = (prob.n_curves() * prob.t_len()) as f64 / 2.0 + cfg.alpha0;
        let rate = 0.5 * self.sse.iter().sum::<f64>() + cfg.beta0;
        let g = Gamma::new(shape, 1.0 / rate).map_err(|e| FdarError::ComputationFailed {
            operation: "bayesian_align_group",
            detail: format!("inverse-gamma draw failed: {e}"),
        })?;
        let prec: f64 = rng.sample(g);
        self.sigma2 = 1.0 / prec.max(f64::MIN_POSITIVE);
        Ok(())
    }

    /// Re-centre: Karcher mean of the warps → identity, template warped to match
    /// (fdasrsf `ncenter` step). Skipped if a re-projected warp is invalid.
    fn center(&mut self, prob: &Problem) {
        let gam_i = centering_warp(&self.warps, &prob.t);
        let q_new = warp_srsf(&self.template, &prob.t, &gam_i);
        let mut new_warps = Vec::with_capacity(prob.n_curves());
        for (i, w) in self.warps.iter().enumerate() {
            let g_new = reparameterize_curve(&w.gamma, &prob.t, &gam_i);
            match prob.warp_from_coef(i, prob.coef_from_gamma(&g_new)) {
                Some(nw) => new_warps.push(nw),
                None => return,
            }
        }
        self.tmpl_coef = prob.template_coef(&q_new);
        self.template = prob.template_from_coef(&self.tmpl_coef);
        self.warps = new_warps;
        self.sse = self
            .warps
            .iter()
            .map(|w| sse(&self.template, &w.warped))
            .collect();
    }

    fn snapshot(&self, prob: &Problem) -> Draw {
        Draw {
            warp_coefs: self.warps.iter().map(|w| w.coef.clone()).collect(),
            template: self.template.clone(),
            sigma2: self.sigma2,
            log_lik: prob.log_likelihood(self.sse.iter().sum(), self.sigma2),
        }
    }
}

fn run_chain(prob: &Problem, cfg: &BayesianGroupConfig, seed: u64) -> Result<ChainRun, FdarError> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut state = ChainState::init(prob, cfg.init_dp)?;
    let mut warp_accept = vec![0usize; prob.n_curves()];
    let mut tmpl_accept = 0usize;
    let mut draws = Vec::new();
    let mut hmc = (cfg.sampler == BayesianSampler::InfHmc)
        .then(|| hmc::HmcWarpSampler::new(cfg, prob.n_curves()));
    // Iteration 0 is the initial state (as in the reference); updates run 1..n_iter.
    for it in 1..cfg.n_iter {
        let tmpl_moved = match hmc.as_mut() {
            None => {
                state.update_warps(prob, &mut rng, &mut warp_accept);
                state.update_template(prob, &mut rng)
            }
            Some(h) => {
                h.sweep(
                    prob,
                    &mut state,
                    it,
                    cfg.burn_in,
                    &mut rng,
                    &mut warp_accept,
                );
                state.gibbs_template(prob, &mut rng)?;
                true
            }
        };
        if tmpl_moved {
            tmpl_accept += 1;
        }
        state.update_sigma2(prob, cfg, &mut rng)?;
        if cfg.center_every != 0 && (it + 1) % cfg.center_every == 0 {
            state.center(prob);
        }
        if it >= cfg.burn_in && (it - cfg.burn_in) % cfg.thin == 0 {
            draws.push(state.snapshot(prob));
        }
    }
    Ok(ChainRun {
        seed,
        draws,
        warp_accept,
        tmpl_accept,
        n_updates: cfg.n_iter - 1,
        step_sizes: hmc.map(|h| h.eps).unwrap_or_default(),
    })
}

// ─── Posterior summaries ─────────────────────────────────────────────────────

/// Maps unit-grid quantities to the output (`argvals`) grid.
struct OutputGrid<'a> {
    argvals: &'a [f64],
    tn: Vec<f64>,
    a: f64,
    len: f64,
}

impl<'a> OutputGrid<'a> {
    fn new(argvals: &'a [f64]) -> Self {
        let m = argvals.len();
        let (a, b) = (argvals[0], argvals[m - 1]);
        Self {
            argvals,
            tn: argvals.iter().map(|&x| (x - a) / (b - a)).collect(),
            a,
            len: b - a,
        }
    }

    /// γ on the unit grid → γ on `argvals`, values on the `argvals` scale.
    fn gamma(&self, t: &[f64], gamma: &[f64]) -> Vec<f64> {
        self.tn
            .iter()
            .map(|&x| self.a + self.len * linear_interp(t, gamma, x))
            .collect()
    }

    /// Unit-domain SRSF → SRSF on `argvals` (scaled by `1/√L`).
    fn srsf(&self, t: &[f64], q: &[f64]) -> Vec<f64> {
        let s = self.len.sqrt().recip();
        self.tn
            .iter()
            .map(|&x| s * linear_interp(t, q, x))
            .collect()
    }

    /// `f0 + ∫ q|q|` on `argvals`.
    fn function(&self, q: &[f64], f0: f64) -> Vec<f64> {
        let qq: Vec<f64> = q.iter().map(|v| v * v.abs()).collect();
        cumtrapz(&qq, self.argvals).iter().map(|v| f0 + v).collect()
    }
}

/// γ (unit grid) from the normalised extrinsic mean of ψ draws.
fn psi_mean_gamma(psis: &[&[f64]], t: &[f64]) -> Vec<f64> {
    let t_len = t.len();
    let mut mean = vec![0.0; t_len];
    for p in psis {
        mean.iter_mut().zip(p.iter()).for_each(|(m, &v)| *m += v);
    }
    let nrm = l2_norm_l2(&mean, t).max(1e-300);
    let sq: Vec<f64> = mean.iter().map(|v| (v / nrm).powi(2)).collect();
    let mut g = cumtrapz(&sq, t);
    let total = g[t_len - 1].max(1e-300);
    g.iter_mut().for_each(|v| *v /= total);
    g
}

/// Split-R̂ (Gelman et al., BDA3) of a scalar over chains; each chain is halved.
fn split_rhat(chains: &[Vec<f64>]) -> f64 {
    let half = chains.iter().map(Vec::len).min().unwrap_or(0) / 2;
    if half < 2 {
        return f64::NAN;
    }
    let mut pieces: Vec<&[f64]> = Vec::with_capacity(2 * chains.len());
    for c in chains {
        let n = c.len();
        pieces.push(&c[..half]);
        pieces.push(&c[n - half..]);
    }
    let nf = half as f64;
    let means: Vec<f64> = pieces.iter().map(|p| p.iter().sum::<f64>() / nf).collect();
    let w = pieces
        .iter()
        .zip(&means)
        .map(|(p, &mu)| p.iter().map(|v| (v - mu).powi(2)).sum::<f64>() / (nf - 1.0))
        .sum::<f64>()
        / pieces.len() as f64;
    let grand = means.iter().sum::<f64>() / means.len() as f64;
    let b =
        nf * means.iter().map(|m| (m - grand).powi(2)).sum::<f64>() / (means.len() as f64 - 1.0);
    let scale = w.abs().max(grand.abs()).max(1.0);
    if w <= 1e-14 * scale * scale {
        return if b <= 1e-14 * scale * scale {
            1.0
        } else {
            f64::INFINITY
        };
    }
    (((nf - 1.0) / nf * w + b / nf) / w).sqrt()
}

/// Effective sample size of one chain: Geyer's initial monotone sequence estimator
/// `n / (−1 + 2 Σ_m Γ_m)`, `Γ_m = ρ_{2m} + ρ_{2m+1}` truncated at the first
/// non-positive pair and made monotone. NaN for fewer than 4 draws or a constant chain.
fn geyer_ess(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = x.iter().sum::<f64>() / nf;
    let d: Vec<f64> = x.iter().map(|v| v - mu).collect();
    let autocov = |k: usize| -> f64 {
        d[..n - k]
            .iter()
            .zip(&d[k..])
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / nf
    };
    let c0 = autocov(0);
    if c0 <= 1e-300 || !c0.is_finite() {
        return f64::NAN;
    }
    let mut tau = -1.0;
    let mut prev = f64::INFINITY;
    let mut k = 0;
    while k + 1 < n {
        let pair = (autocov(k) + autocov(k + 1)) / c0;
        if pair <= 0.0 {
            break;
        }
        let pair = pair.min(prev);
        tau += 2.0 * pair;
        prev = pair;
        k += 2;
    }
    nf / tau.max(1.0 / nf)
}

/// ESS summed over chains (NaN chains ignored; NaN if all are).
fn multi_chain_ess(chains: &[Vec<f64>]) -> f64 {
    let vals: Vec<f64> = chains
        .iter()
        .map(|c| geyer_ess(c))
        .filter(|v| v.is_finite())
        .collect();
    if vals.is_empty() {
        f64::NAN
    } else {
        vals.iter().sum()
    }
}

/// Sarle's bimodality coefficient with bias-corrected skewness and excess kurtosis
/// (as `scipy.stats.skew/kurtosis(bias=False)`). NaN for n < 4 or zero variance.
fn bimodality_coefficient(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mu = x.iter().sum::<f64>() / nf;
    let moment = |p: i32| x.iter().map(|v| (v - mu).powi(p)).sum::<f64>() / nf;
    let m2 = moment(2);
    if m2 <= 1e-300 * mu.abs().max(1.0) || !m2.is_finite() {
        return f64::NAN;
    }
    let g1 = moment(3) / m2.powf(1.5);
    let g2 = moment(4) / (m2 * m2) - 3.0;
    let skew = g1 * (nf * (nf - 1.0)).sqrt() / (nf - 2.0);
    let denom23 = (nf - 2.0) * (nf - 3.0);
    let kurt = (nf - 1.0) / denom23 * ((nf + 1.0) * g2 + 6.0);
    (skew * skew + 1.0) / (kurt + 3.0 * (nf - 1.0).powi(2) / denom23)
}

/// Warps and template of one retained draw on the unit grid, optionally
/// re-centred so that the Karcher mean of its warps is the identity.
struct RealDraw {
    gammas: Vec<Vec<f64>>,
    psis: Vec<Vec<f64>>,
    template: Vec<f64>,
}

fn realise(prob: &Problem, d: &Draw, center: bool) -> RealDraw {
    let warps: Vec<Warp> = d
        .warp_coefs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            prob.warp_from_coef(i, c.clone())
                .expect("retained draws are valid warps")
        })
        .collect();
    if !center {
        return RealDraw {
            psis: warps.iter().map(|w| w.psi.clone()).collect(),
            gammas: warps.into_iter().map(|w| w.gamma).collect(),
            template: d.template.clone(),
        };
    }
    let gam_i = centering_warp(&warps, &prob.t);
    let gammas: Vec<Vec<f64>> = warps
        .iter()
        .map(|w| reparameterize_curve(&w.gamma, &prob.t, &gam_i))
        .collect();
    RealDraw {
        psis: gammas.iter().map(|g| unit_psi(g, &prob.t)).collect(),
        gammas,
        template: warp_srsf(&d.template, &prob.t, &gam_i),
    }
}

fn summarize_chain(
    prob: &Problem,
    run: &ChainRun,
    real: &[RealDraw],
    out: &OutputGrid,
) -> BayesianGroupChain {
    let (n, m) = (prob.n_curves(), out.tn.len());
    let mode_idx = (0..run.draws.len())
        .max_by(|&a, &b| run.draws[a].log_lik.total_cmp(&run.draws[b].log_lik))
        .unwrap_or(0);
    let mut mean_g = FdMatrix::zeros(n, m);
    let mut mode_g = FdMatrix::zeros(n, m);
    for i in 0..n {
        let psis: Vec<&[f64]> = real.iter().map(|r| r.psis[i].as_slice()).collect();
        let gm = out.gamma(&prob.t, &psi_mean_gamma(&psis, &prob.t));
        let gmode = out.gamma(&prob.t, &real[mode_idx].gammas[i]);
        for j in 0..m {
            mean_g[(i, j)] = gm[j];
            mode_g[(i, j)] = gmode[j];
        }
    }
    let nd = real.len() as f64;
    let mut tmean = vec![0.0; prob.t_len()];
    for d in real {
        tmean
            .iter_mut()
            .zip(&d.template)
            .for_each(|(a, &b)| *a += b / nd);
    }
    let upd = run.n_updates.max(1) as f64;
    BayesianGroupChain {
        seed: run.seed,
        posterior_mean_gammas: mean_g,
        mode_gammas: mode_g,
        mode_log_likelihood: run.draws[mode_idx].log_lik,
        mode_template_srsf: out.srsf(&prob.t, &real[mode_idx].template),
        template_srsf_mean: out.srsf(&prob.t, &tmean),
        warp_acceptance: run.warp_accept.iter().map(|&a| a as f64 / upd).collect(),
        template_acceptance: run.tmpl_accept as f64 / upd,
        sigma2_trace: run.draws.iter().map(|d| d.sigma2).collect(),
        log_likelihood_trace: run.draws.iter().map(|d| d.log_lik).collect(),
        warp_step_sizes: run.step_sizes.clone(),
    }
}

/// Pointwise (lower, median, upper) over the rows of a draws matrix.
fn pointwise_bands(draws: &FdMatrix, level: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = draws.ncols();
    let (mut lower, mut median, mut upper) = (vec![0.0; m], vec![0.0; m], vec![0.0; m]);
    for j in 0..m {
        let mut col = draws.column(j).to_vec();
        let (lo, md, hi) = sample_quantiles(&mut col, level);
        lower[j] = lo;
        median[j] = md;
        upper[j] = hi;
    }
    (lower, median, upper)
}

/// Pooled warp summaries of one curve.
struct WarpSummary {
    mean: Vec<f64>,
    median: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    draws: FdMatrix,
    rhat: f64,
    ess: f64,
    bimodality: f64,
}

fn summarize_curve(
    prob: &Problem,
    real: &[Vec<RealDraw>],
    i: usize,
    out: &OutputGrid,
    level: f64,
) -> WarpSummary {
    let m = out.tn.len();
    let pooled: Vec<&RealDraw> = real.iter().flatten().collect();
    let psis: Vec<&[f64]> = pooled.iter().map(|r| r.psis[i].as_slice()).collect();
    let mean = out.gamma(&prob.t, &psi_mean_gamma(&psis, &prob.t));
    let mut draws = FdMatrix::zeros(pooled.len(), m);
    for (k, r) in pooled.iter().enumerate() {
        let g = out.gamma(&prob.t, &r.gammas[i]);
        for j in 0..m {
            draws[(k, j)] = g[j];
        }
    }
    let (lower, median, upper) = pointwise_bands(&draws, level);
    let area: Vec<Vec<f64>> = real
        .iter()
        .map(|chain| {
            chain
                .iter()
                .map(|r| {
                    let dev: Vec<f64> = r.gammas[i]
                        .iter()
                        .zip(&prob.t)
                        .map(|(g, t)| g - t)
                        .collect();
                    trapz(&dev, &prob.t)
                })
                .collect()
        })
        .collect();
    let pooled_area: Vec<f64> = area.iter().flatten().copied().collect();
    WarpSummary {
        mean,
        median,
        lower,
        upper,
        draws,
        rhat: split_rhat(&area),
        ess: multi_chain_ess(&area),
        bimodality: bimodality_coefficient(&pooled_area),
    }
}

/// Per-curve max pairwise L²\[0,1\] distance between chain posterior-mean warps.
fn mode_spread(chains: &[BayesianGroupChain], out: &OutputGrid, n: usize) -> Vec<f64> {
    let pair_dist = |a: &FdMatrix, b: &FdMatrix, i: usize| -> f64 {
        let d: Vec<f64> = (0..out.tn.len())
            .map(|j| ((a[(i, j)] - b[(i, j)]) / out.len).powi(2))
            .collect();
        trapz(&d, &out.tn).max(0.0).sqrt()
    };
    (0..n)
        .map(|i| {
            let mut best: f64 = 0.0;
            for a in 0..chains.len() {
                for b in (a + 1)..chains.len() {
                    let d = pair_dist(
                        &chains[a].posterior_mean_gammas,
                        &chains[b].posterior_mean_gammas,
                        i,
                    );
                    best = best.max(d);
                }
            }
            best
        })
        .collect()
}

/// Template SRSF / function summaries on the output grid.
struct TemplateSummary {
    srsf_draws: FdMatrix,
    srsf: Vec<f64>,
    srsf_lower: Vec<f64>,
    srsf_upper: Vec<f64>,
    func: Vec<f64>,
    func_lower: Vec<f64>,
    func_upper: Vec<f64>,
}

fn summarize_template(
    prob: &Problem,
    real: &[Vec<RealDraw>],
    out: &OutputGrid,
    f0: f64,
    level: f64,
) -> TemplateSummary {
    let m = out.tn.len();
    let pooled: Vec<&RealDraw> = real.iter().flatten().collect();
    let nd = pooled.len();
    let mut srsf_draws = FdMatrix::zeros(nd, m);
    let mut func_draws = FdMatrix::zeros(nd, m);
    for (k, d) in pooled.iter().enumerate() {
        let q = out.srsf(&prob.t, &d.template);
        let f = out.function(&q, f0);
        for j in 0..m {
            srsf_draws[(k, j)] = q[j];
            func_draws[(k, j)] = f[j];
        }
    }
    let srsf: Vec<f64> = (0..m)
        .map(|j| srsf_draws.column(j).iter().sum::<f64>() / nd as f64)
        .collect();
    let (srsf_lower, _, srsf_upper) = pointwise_bands(&srsf_draws, level);
    let (func_lower, _, func_upper) = pointwise_bands(&func_draws, level);
    TemplateSummary {
        func: out.function(&srsf, f0),
        srsf_draws,
        srsf,
        srsf_lower,
        srsf_upper,
        func_lower,
        func_upper,
    }
}

fn rows_to_matrix(rows: &[Vec<f64>], m: usize) -> FdMatrix {
    let mut mat = FdMatrix::zeros(rows.len(), m);
    for (i, r) in rows.iter().enumerate() {
        for j in 0..m {
            mat[(i, j)] = r[j];
        }
    }
    mat
}

fn assemble(
    prob: &Problem,
    runs: &[ChainRun],
    data: &FdMatrix,
    argvals: &[f64],
    cfg: &BayesianGroupConfig,
) -> BayesianGroupResult {
    let (n, m) = data.shape();
    let out = OutputGrid::new(argvals);
    let level = cfg.credible_level;
    let real: Vec<Vec<RealDraw>> = iter_maybe_parallel!(0..runs.len())
        .map(|c| {
            runs[c]
                .draws
                .iter()
                .map(|d| realise(prob, d, cfg.center_draws))
                .collect()
        })
        .collect();
    let chains: Vec<BayesianGroupChain> = runs
        .iter()
        .zip(&real)
        .map(|(r, rd)| summarize_chain(prob, r, rd, &out))
        .collect();
    let curves: Vec<WarpSummary> = (0..n)
        .map(|i| summarize_curve(prob, &real, i, &out, level))
        .collect();
    let f0 = (0..n).map(|i| data[(i, 0)]).sum::<f64>() / n as f64;
    let tmpl = summarize_template(prob, &real, &out, f0, level);
    let mean_rows: Vec<Vec<f64>> = curves.iter().map(|c| c.mean.clone()).collect();
    let aligned: Vec<Vec<f64>> = (0..n)
        .map(|i| reparameterize_curve(&data.row(i), argvals, &mean_rows[i]))
        .collect();
    let mut sigma2_draws: Vec<f64> = runs
        .iter()
        .flat_map(|r| r.draws.iter().map(|d| d.sigma2))
        .collect();
    let sigma2_mean = sigma2_draws.iter().sum::<f64>() / sigma2_draws.len() as f64;
    let mut sorted = sigma2_draws.clone();
    let (sigma2_lower, _, sigma2_upper) = sample_quantiles(&mut sorted, level);
    let nc = chains.len() as f64;
    let warp_acceptance = (0..n)
        .map(|i| chains.iter().map(|c| c.warp_acceptance[i]).sum::<f64>() / nc)
        .collect();
    let template_acceptance = chains.iter().map(|c| c.template_acceptance).sum::<f64>() / nc;
    let sig_traces: Vec<Vec<f64>> = chains.iter().map(|c| c.sigma2_trace.clone()).collect();
    let ll_traces: Vec<Vec<f64>> = chains
        .iter()
        .map(|c| c.log_likelihood_trace.clone())
        .collect();
    let spread = mode_spread(&chains, &out, n);
    sigma2_draws.shrink_to_fit();
    let warp_bimodality: Vec<f64> = curves.iter().map(|c| c.bimodality).collect();
    BayesianGroupResult {
        gammas: rows_to_matrix(&mean_rows, m),
        gammas_median: rows_to_matrix(
            &curves.iter().map(|c| c.median.clone()).collect::<Vec<_>>(),
            m,
        ),
        gammas_lower: rows_to_matrix(
            &curves.iter().map(|c| c.lower.clone()).collect::<Vec<_>>(),
            m,
        ),
        gammas_upper: rows_to_matrix(
            &curves.iter().map(|c| c.upper.clone()).collect::<Vec<_>>(),
            m,
        ),
        aligned_data: rows_to_matrix(&aligned, m),
        template_srsf: tmpl.srsf,
        template_srsf_lower: tmpl.srsf_lower,
        template_srsf_upper: tmpl.srsf_upper,
        template: tmpl.func,
        template_lower: tmpl.func_lower,
        template_upper: tmpl.func_upper,
        sigma2_mean,
        sigma2_lower,
        sigma2_upper,
        warp_acceptance,
        template_acceptance,
        rhat_sigma2: split_rhat(&sig_traces),
        rhat_log_likelihood: split_rhat(&ll_traces),
        rhat_warps: curves.iter().map(|c| c.rhat).collect(),
        ess_sigma2: multi_chain_ess(&sig_traces),
        ess_log_likelihood: multi_chain_ess(&ll_traces),
        ess_warps: curves.iter().map(|c| c.ess).collect(),
        multimodal_curves: warp_bimodality.iter().map(|&b| b > 5.0 / 9.0).collect(),
        warp_bimodality,
        mode_spread: spread,
        gamma_draws: curves.into_iter().map(|c| c.draws).collect(),
        template_srsf_draws: tmpl.srsf_draws,
        sigma2_draws,
        chains,
    }
}

// ─── Public entry point ──────────────────────────────────────────────────────

/// Bayesian group alignment: joint posterior draws of all warps, the template and
/// the noise variance via Z-mixture pCN MCMC, with multiple chains.
///
/// See the [module documentation](self) for the model. Each curve `f_i` (row of
/// `data`) gets a posterior over warps `γ_i` such that `f_i ∘ γ_i` is aligned to the
/// common template; the result reports the posterior-mean (ψ-mean) and median
/// warps, pointwise credible bands for the warps, the template SRSF and the
/// template function, σ², acceptance rates, per-chain modes and split-R̂.
///
/// Chains run in parallel under the `parallel` feature; chain `k` is seeded with
/// `config.seed + k`, so the result is identical with or without parallelism.
///
/// # Arguments
/// * `data` — n × m curves (rows), m ≥ 4
/// * `argvals` — strictly increasing evaluation points (length m)
/// * `config` — MCMC configuration
///
/// # Errors
/// [`FdarError::InvalidDimension`] for shape mismatches; [`FdarError::InvalidParameter`]
/// for out-of-range configuration values or non-finite data;
/// [`FdarError::ComputationFailed`] if a numerical step fails.
///
/// # Example
/// ```
/// use fdars_core::alignment::{bayesian_align_group, BayesianGroupConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 31;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let shifts = [-0.04, 0.0, 0.04];
/// let mut data = FdMatrix::zeros(3, m);
/// for (i, s) in shifts.iter().enumerate() {
///     for j in 0..m {
///         let x = t[j] + s;
///         data[(i, j)] = (-(x - 0.5).powi(2) / 0.02).exp();
///     }
/// }
/// let mut cfg = BayesianGroupConfig::default();
/// cfg.n_iter = 200;
/// cfg.burn_in = 100;
/// cfg.n_warp_harmonics = 3;
/// cfg.n_template_basis = 12;
/// cfg.n_chains = 2;
/// let res = bayesian_align_group(&data, &t, &cfg).unwrap();
/// assert_eq!(res.gammas.shape(), (3, m));
/// assert_eq!(res.chains.len(), 2);
/// // Every posterior-mean warp is a monotone map of [0, 1] onto itself.
/// for i in 0..3 {
///     assert!((res.gammas[(i, 0)] - 0.0).abs() < 1e-12);
///     assert!((res.gammas[(i, m - 1)] - 1.0).abs() < 1e-12);
///     assert!((1..m).all(|j| res.gammas[(i, j)] >= res.gammas[(i, j - 1)]));
/// }
/// ```
#[must_use = "expensive MCMC computation whose result should not be discarded"]
pub fn bayesian_align_group(
    data: &FdMatrix,
    argvals: &[f64],
    config: &BayesianGroupConfig,
) -> Result<BayesianGroupResult, FdarError> {
    validate_dims(data, argvals)?;
    validate_chain_config(config)?;
    let prob = Problem::new(data, argvals, config)?;
    let runs: Vec<ChainRun> = iter_maybe_parallel!(0..config.n_chains)
        .map(|k| run_chain(&prob, config, config.seed.wrapping_add(k as u64)))
        .collect::<Result<Vec<_>, FdarError>>()?;
    if runs.iter().any(|r| r.draws.is_empty()) {
        return Err(invalid(
            "n_iter/burn_in/thin",
            "no draws retained after burn-in".to_string(),
        ));
    }
    Ok(assemble(&prob, runs.as_slice(), data, argvals, config))
}

#[path = "bayesian_group_hmc.rs"]
mod hmc;

#[cfg(test)]
#[path = "bayesian_group_tests.rs"]
mod tests;
