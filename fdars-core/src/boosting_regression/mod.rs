//! Component-wise gradient boosting and Bayesian regression for functional responses.
//!
//! Implements REG-06: FDboost-style penalized functional base-learner boosting,
//! GAMLSS distributional boosting, conjugate Gibbs Bayesian FOSR, and stability selection.
//!
//! # Methods
//!
//! - [`boost_fosr`] — Component-wise boosted function-on-scalar regression (REG-06-01)
//! - [`boost_fofr`] — Component-wise boosted function-on-function regression (REG-06-02)
//! - [`gamlss_fosr`] — GAMLSS distributional boosting: location + scale (REG-06-03)
//! - [`bayesian_fosr`] — Bayesian FOSR via conjugate Gibbs sampler (REG-06-04)
//! - [`stability_selection`] — FDboost-style stability selection (REG-06-05)
//!
//! # References
//!
//! Hothorn et al. (2010). Model-Based Boosting. *Journal of Statistical Software*.
//! Hofner et al. (2016). gamboostLSS. *Journal of Statistical Software*, 74(1).
//! Jiang et al. (2025). arXiv:2505.05633 (Bayesian FoSR).
//! Meinshausen & Bühlmann (2010). Stability Selection. *JRSS-B*, 72(4).
//!
//! Divergences from R baselines (FDboost 1.1-4, refund, stabs) documented per function.

use crate::matrix::FdMatrix;

pub mod bayesian;
pub mod boost_fofr;
pub mod boost_fosr;
pub mod gamlss;
pub mod stability;

// ---------------------------------------------------------------------------
// Config structs
// ---------------------------------------------------------------------------

/// Configuration for component-wise gradient boosting (FOSR, FoFR, GAMLSS, stability).
///
/// All base-learners use the same `nbasis`, `order`, `lfd_order`, and `lambda`
/// to ensure equal effective degrees of freedom, preventing selection bias toward
/// more flexible learners (see Pitfall 4 in RESEARCH.md).
///
/// **Divergence from FDboost:** Fixed `nu` and `mstop` rather than CV-based early stopping;
/// the GCV path is tracked for diagnostic purposes but not used for stopping.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostingConfig {
    /// Number of boosting iterations (must be ≥ 1).
    pub mstop: usize,
    /// Learning rate ν ∈ (0, 1] (FDboost default: 0.1).
    pub nu: f64,
    /// Number of B-spline basis functions per base-learner (must be ≥ 4).
    ///
    /// The actual number of basis functions is `nknots + order` where
    /// `nknots = nbasis - order`. With `order = 4` (cubic), `nbasis = 10`
    /// gives 6 interior knots.
    pub nbasis: usize,
    /// B-spline order (typically 4 for cubic splines).
    pub order: usize,
    /// Penalty derivative order (typically 2 for roughness).
    pub lfd_order: usize,
    /// Smoothing parameter λ > 0 for penalized base-learners.
    pub lambda: f64,
    /// Number of predictor FPC components for FoFR base-learners (REG-06-02).
    pub ncomp_x: usize,
    /// RNG seed (used by stability selection and future extensions; unused in pure boosting).
    pub seed: u64,
}

impl Default for BoostingConfig {
    /// FDboost-convention defaults: `mstop = 100`, `nu = 0.1`, cubic (`order = 4`)
    /// B-spline base-learners with `nbasis = 10`, second-derivative penalty
    /// (`lfd_order = 2`), `lambda = 1.0`, `ncomp_x = 3`, `seed = 0`.
    fn default() -> Self {
        Self {
            mstop: 100,
            nu: 0.1,
            nbasis: 10,
            order: 4,
            lfd_order: 2,
            lambda: 1.0,
            ncomp_x: 3,
            seed: 0,
        }
    }
}

/// Configuration for the Bayesian FOSR Gibbs sampler (REG-06-04).
///
/// Uses conjugate Normal-Inverse-Gamma priors. Defaults match the weakly-informative
/// settings recommended by Jiang et al. (2025): `τ² = 100`, `IG(0.001, 0.001)`.
///
/// **Divergence from refund:** refund's Bayesian FOSR uses spline basis priors;
/// this implementation uses FPCA score compression via `fdata_to_pc_1d` for
/// simplicity and zero new dependencies.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianConfig {
    /// Number of FPC components for score compression (must be ≥ 1).
    pub ncomp: usize,
    /// Prior variance τ² on FPC-space coefficients (default: 100.0).
    ///
    /// Large `τ²` gives a weakly informative prior; very small values are dogmatic.
    pub tau2: f64,
    /// Inverse-Gamma prior shape a₀ (default: 0.001 — weakly informative).
    pub ig_a0: f64,
    /// Inverse-Gamma prior rate b₀ (default: 0.001 — weakly informative).
    pub ig_b0: f64,
    /// Number of Gibbs iterations retained after burn-in (must be ≥ 1).
    pub n_iter: usize,
    /// Burn-in iterations discarded (must be < `n_iter` iterations will run).
    pub burn_in: usize,
    /// Thinning interval — keep every `thin`-th draw (must be ≥ 1).
    pub thin: usize,
    /// RNG seed — chain is fully deterministic: `StdRng::seed_from_u64(seed)`.
    pub seed: u64,
}

impl Default for BayesianConfig {
    /// Weakly-informative defaults per Jiang et al. (2025): `tau2 = 100.0`,
    /// `IG(0.001, 0.001)`, with `ncomp = 4`, `n_iter = 400`, `burn_in = 200`,
    /// `thin = 1`, `seed = 0` (mirrors `bayesian::tests::default_config`).
    fn default() -> Self {
        Self {
            ncomp: 4,
            tau2: 100.0,
            ig_a0: 0.001,
            ig_b0: 0.001,
            n_iter: 400,
            burn_in: 200,
            thin: 1,
            seed: 0,
        }
    }
}

/// Configuration for FDboost-style stability selection (REG-06-05).
///
/// Implements the Meinshausen-Bühlmann subsampling scheme with ⌊n/2⌋ rows
/// per replicate. The PFER bound `E[V] ≤ q² / ((2·π_thr − 1)·p)` is reported
/// as an informational diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityConfig {
    /// Number of resamples B (must be ≥ 1; default: 100).
    pub n_resamples: usize,
    /// Selection threshold π ∈ (0.5, 1.0] (default: 0.9).
    ///
    /// Base-learner j is declared "stable" if its selection frequency ≥ `pi_thr`.
    pub pi_thr: f64,
    /// Base RNG seed; replicate `b` uses `seed.wrapping_add(b as u64)` for isolation.
    pub seed: u64,
}

impl Default for StabilityConfig {
    /// Meinshausen-Bühlmann defaults: `n_resamples = 100`, `pi_thr = 0.9`,
    /// `seed = 0`.
    fn default() -> Self {
        Self {
            n_resamples: 100,
            pi_thr: 0.9,
            seed: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Result structs
// ---------------------------------------------------------------------------

/// Result of component-wise boosted function-on-scalar regression (REG-06-01).
///
/// Follows the `FosrResult` field convention; adds the boosting path diagnostics
/// (`selected_learners`, `gcv_path`, `mstop`, `nu`).
///
/// **Divergence from FDboost:** fixed `mstop` / fixed `nu`; no CV-based early stopping.
/// GCV path is tracked for post-hoc diagnostics only (see `gcv_path`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoostFosrResult {
    /// Intercept function F₀(t) = Ȳ(t) (pointwise mean of Y, length m_t).
    pub intercept: Vec<f64>,
    /// Accumulated coefficient functions β_j(t) per predictor (p × m_t).
    ///
    /// Row j holds the total contribution of base-learner j across all iterations
    /// where it was selected, scaled by ν.
    pub beta: FdMatrix,
    /// Fitted functional values F̂(xᵢ, t) = Σ contributions (n × m_t).
    pub fitted: FdMatrix,
    /// Residual curves Y − F̂ (n × m_t).
    pub residuals: FdMatrix,
    /// Pointwise R²(t) at each response grid point (length m_t).
    pub r_squared_t: Vec<f64>,
    /// Integrated R² (scalar summary).
    pub r_squared: f64,
    /// Number of boosting iterations used.
    pub mstop: usize,
    /// Learning rate ν used.
    pub nu: f64,
    /// Index j* of the base-learner selected at each boosting iteration (length mstop).
    pub selected_learners: Vec<usize>,
    /// ‖residual‖_F² recorded after each iteration (length mstop).
    ///
    /// Should be non-increasing for L2 loss (use for path diagnostics / GCV).
    pub gcv_path: Vec<f64>,
}

/// Result of component-wise boosted function-on-function regression (REG-06-02).
///
/// Functional predictors are compressed via FPCA score projection; the boosting
/// core operates on the resulting scalar design matrices (bfpc variant).
///
/// **Divergence from FDboost:** uses FPC score compression (`fdata_to_pc_1d`) rather
/// than FDboost's `bsignal` B-spline joint expansion. Simpler and dependency-free.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoostFofrResult {
    /// Intercept function F₀(t) = Ȳ(t) (length m_y).
    pub intercept: Vec<f64>,
    /// Fitted response curves (n × m_y).
    pub fitted: FdMatrix,
    /// Residual curves (n × m_y).
    pub residuals: FdMatrix,
    /// Pointwise R²(t) at each response grid point (length m_y).
    pub r_squared_t: Vec<f64>,
    /// Overall R².
    pub r_squared: f64,
    /// FPCA result for each functional predictor (one per predictor).
    pub fpca_x: Vec<crate::regression::FpcaResult>,
    /// Accumulated FPC-space score coefficients per predictor (Vec[j] is K_j × m_y).
    pub score_coefs: Vec<FdMatrix>,
    /// Reconstructed coefficient surfaces β_j(s,t) per predictor (Vec[j] is m_x × m_y).
    pub beta_surfaces: Vec<FdMatrix>,
    /// Index j* of the base-learner selected at each boosting iteration (length mstop).
    pub selected_learners: Vec<usize>,
    /// ‖residual‖_F² per boosting iteration (length mstop).
    pub gcv_path: Vec<f64>,
    /// Number of boosting iterations used.
    pub mstop: usize,
    /// Learning rate ν used.
    pub nu: f64,
}

/// Result of GAMLSS-style distributional functional regression (REG-06-03).
///
/// Models a Gaussian functional response Y(t) with location μ(t) and scale σ(t).
/// Cyclic component-wise boosting alternates between boosting μ and log-σ.
///
/// **Divergence from gamboostLSS:** uses cyclic rather than noncyclic (non-cyclic
/// per-iteration selection is superior for variable selection but more complex).
/// Links: identity for μ, log for σ.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GamlssResult {
    /// Fitted location μ̂(t) per observation (n × m_t).
    pub mu_fitted: FdMatrix,
    /// Fitted scale σ̂(t) per observation (n × m_t); always positive.
    pub sigma_fitted: FdMatrix,
    /// Intercept for the μ model: F̂_μ,₀(t) = Ȳ(t) (length m_t).
    pub mu_intercept: Vec<f64>,
    /// Intercept for the log-σ model: 0 → exp(0) = 1 (length m_t).
    pub sigma_intercept: Vec<f64>,
    /// Accumulated μ coefficient functions (p × m_t).
    pub mu_beta: FdMatrix,
    /// Accumulated log-σ coefficient functions (p × m_t).
    pub sigma_beta: FdMatrix,
    /// Final Gaussian log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Log-likelihood per cyclic iteration (length mstop).
    pub ll_path: Vec<f64>,
    /// Number of cyclic boosting iterations used.
    pub mstop: usize,
    /// Learning rate ν used.
    pub nu: f64,
}

/// Result of Bayesian function-on-scalar regression via conjugate Gibbs (REG-06-04).
///
/// Posterior summaries are computed from thinned post-burn-in draws. Credible bands
/// are pointwise (not simultaneous) quantiles over the retained draws.
///
/// **Divergence from refund:** uses FPCA score compression via `fdata_to_pc_1d`
/// rather than spline basis priors. Pointwise credible bands only (no simultaneous bands).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BayesianFosrResult {
    /// Posterior mean coefficient functions β̄(t) (p × m_t).
    pub beta_mean: FdMatrix,
    /// Pointwise 2.5% credible band (p × m_t).
    pub beta_lower: FdMatrix,
    /// Pointwise 97.5% credible band (p × m_t).
    pub beta_upper: FdMatrix,
    /// Posterior-mean fitted values (n × m_t).
    pub fitted: FdMatrix,
    /// Posterior-mean residuals (n × m_t).
    pub residuals: FdMatrix,
    /// Posterior mean σ²(t) across the response grid (length m_t).
    pub sigma2_mean: Vec<f64>,
    /// Number of Gibbs iterations retained (after burn-in, before thinning).
    pub n_iter: usize,
    /// Burn-in iterations discarded.
    pub burn_in: usize,
    /// Thinning interval used.
    pub thin: usize,
    /// FPC components used for score compression.
    pub ncomp: usize,
}

/// Result of FDboost-style stability selection (REG-06-05).
///
/// Aggregates base-learner selection frequencies over B subsamples of size ⌊n/2⌋.
/// The PFER bound is informational: `E[V] ≤ q² / ((2·π_thr − 1)·p)`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StabilityResult {
    /// Selection frequency π̂[j] ∈ [0, 1] for each base-learner j = 0..p.
    pub selection_freq: Vec<f64>,
    /// Indices j where π̂[j] ≥ pi_thr (the "stable set").
    pub stable_set: Vec<usize>,
    /// Threshold π used.
    pub pi_thr: f64,
    /// PFER upper bound: `q² / ((2·pi_thr − 1)·p)` where q = mean per-subsample selection count.
    pub pfer_bound: f64,
    /// Number of resamples B used.
    pub n_resamples: usize,
}

// ---------------------------------------------------------------------------
// Barrel re-exports
// ---------------------------------------------------------------------------

pub use self::bayesian::bayesian_fosr;
pub use self::boost_fofr::boost_fofr;
pub use self::boost_fosr::boost_fosr;
pub use self::gamlss::gamlss_fosr;
pub use self::stability::stability_selection;
