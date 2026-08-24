---
phase: 43-boosting-bayesian-functional-regression
plan: "04"
subsystem: boosting_regression
tags: [functional-regression, bayesian, gibbs, fpca, credible-bands, REG-06-04]
requirements: [REG-06-04]
status: complete

dependency_graph:
  requires: [43-01-boosting-core-fosr]
  provides: [bayesian_fosr, BayesianFosrResult — conjugate Gibbs FOSR with pointwise credible bands]
  affects:
    - fdars-core/src/boosting_regression/bayesian.rs

tech_stack:
  added: []
  patterns:
    - "FPCA response compression via fdata_to_pc_1d → per-component score regression"
    - "Conjugate Normal/Inverse-Gamma Gibbs full-conditionals per FPC component"
    - "Multivariate-Normal precision draw via Cholesky (Rue 2001): b = mu_post + L^T^-1 z"
    - "IG draw as 1/Gamma(alpha, 1/rate) via rand_distr::Gamma (scale parameterization)"
    - "Coefficient reconstruction beta_j(t) = sum_k b_jk · phi_k(t) from FPCA rotation per draw"
    - "Pointwise credible bands = sorted-draw quantiles per (j,t); StdRng::seed_from_u64 determinism"

key_files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/bayesian.rs

decisions:
  - "FPCA score compression of the response (fdata_to_pc_1d) rather than refund's spline basis priors — zero new deps; documented divergence in rustdoc."
  - "Per-FPC-component conjugate regression of scores on mean-centered scalar predictors; response intercept mu(t) carried by the FPCA mean function."
  - "Pointwise (not simultaneous) credible bands — per project scope."
  - "Gaussian-Normal posterior draw sampled from the PRECISION Cholesky (A = X'X/sigma2 + I/tau2) via a back-substitution helper back_solve_lt, reusing cholesky_factor + cholesky_forward_back — no dense inverse formed."
  - "sigma2_mean(t) reported as posterior-mean residual variance in the response domain (interpretable, strictly positive)."

verification:
  module_tests: "6/6 pass — cargo test -p fdars-core --features linalg,parallel --lib boosting_regression::bayesian"
  tests:
    - bayesian_fosr_recovers_beta (posterior mean beta(t) correlates >0.9 with true sin(pi t))
    - bayesian_fosr_credible_bands_bracket_mean (lower <= mean <= upper, all finite, at every t)
    - bayesian_fosr_is_deterministic_under_seed (two runs, same seed → bit-identical beta_mean/lower/upper/sigma2_mean)
    - bayesian_fosr_sigma2_positive_and_shapes (fitted/residuals n×m_t; sigma2_mean length m_t, all >0 finite)
    - bayesian_fosr_errors_on_dimension_mismatch (FdarError, no panic)
    - bayesian_fosr_errors_on_invalid_params (FdarError on tau2<=0 and ncomp=0)

notes:
  - "Implemented inline by the orchestrator after repeated transient API (529 Overloaded) errors prevented executor-subagent dispatch. Algorithm follows RESEARCH.md Algorithm 4; the FPC-space model was made concrete as a per-component conjugate regression of response scores on the (centered) scalar predictors."
  - "Full crate-wide clippy + fmt + test gate deferred to the phase-end out-of-band gate."

commits:
  - "80086d14 feat(43-04): Bayesian function-on-scalar regression via conjugate Gibbs (REG-06-04)"
---

# Plan 43-04 — Bayesian FOSR via conjugate Gibbs sampler (REG-06-04)

Implemented `bayesian_fosr` in `fdars-core/src/boosting_regression/bayesian.rs`. The
functional response is compressed with FPCA; for each principal component the FPC scores
are regressed on the mean-centered scalar predictors via a conjugate Normal / Inverse-Gamma
Gibbs sampler. Each retained post-burn-in, thinned draw reconstructs the coefficient
functions `β_j(t) = Σ_k b_{jk}·φ_k(t)` from the FPCA rotation; posterior mean and pointwise
2.5%/97.5% credible bands follow directly.

**Requirement REG-06-04** — "fit a Bayesian function-on-scalar regression via a Gibbs
sampler and obtain coefficient posterior summaries (posterior mean + credible bands)" — is
satisfied.

## Numerical approach

The multivariate-Normal full conditional is sampled from the precision matrix
`A = X̃'X̃/σ²_k + I/τ²` via its Cholesky factor (Rue 2001): `b_k = μ_post + Lᵀ⁻¹z`,
reusing `cholesky_factor` + `cholesky_forward_back` plus a small `Lᵀ`-back-substitution
helper — no dense inverse. The Inverse-Gamma variance draw uses `rand_distr::Gamma`
(`1/Gamma(α, 1/rate)`). Adding `I/τ²` keeps `A` positive-definite, so the Cholesky never
fails. The chain is fully deterministic under `config.seed`.

## Verification

6/6 module tests pass, including coefficient recovery, credible-band bracketing, and
bit-identical determinism under a fixed seed. Crate-wide clippy/fmt/test gate runs at
phase end.
