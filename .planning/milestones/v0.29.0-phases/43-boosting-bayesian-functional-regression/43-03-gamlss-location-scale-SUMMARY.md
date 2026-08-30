---
phase: 43-boosting-bayesian-functional-regression
plan: "03"
subsystem: boosting_regression
tags: [functional-regression, gamlss, distributional, location-scale, boosting, REG-06-03]
requirements: [REG-06-03]
status: complete

dependency_graph:
  requires: [43-01-boosting-core-fosr]
  provides: [gamlss_fosr, GamlssResult — Gaussian location+scale distributional functional regression]
  affects:
    - fdars-core/src/boosting_regression/gamlss.rs
    - fdars-core/src/boosting_regression/boost_fosr.rs

tech_stack:
  added: []
  patterns:
    - "Cyclic gamboostLSS: alternate one boosting step per distributional parameter (μ then σ) per iteration"
    - "Gaussian negative gradients: μ-step u=(Y−μ)/σ² (identity link); σ-step u=−1+(Y−μ)²/σ² (log link)"
    - "Functional intercept in the σ-step (i-mean of the pseudo-response) + predictor learner on the centered remainder"
    - "Shared boosting base-learners (penalized B-spline via build_bspline_design_at + cholesky_factor) across μ and σ models"
    - "Data-adaptive log-σ clamp + marginal-scale μ-gradient-denominator floor for coupled-dynamics stability"

key_files:
  created: []
  modified:
    - fdars-core/src/boosting_regression/gamlss.rs
    - fdars-core/src/boosting_regression/boost_fosr.rs

decisions:
  - "Cyclic (not noncyclic) gamboostLSS parameter selection — per 43-CONTEXT.md v1 scope; documented divergence in rustdoc."
  - "Gaussian family only, location μ (identity link) + scale σ (log link)."
  - "Made build_bspline_design → pub(crate) build_bspline_design_at in boost_fosr.rs so gamlss reuses the base-learner design build."
  - "Numerical stabilization (all documented in rustdoc): (1) warm-start σ at the marginal residual scale σ₀=SD(Y−Ȳ(t)) rather than 1.0; (2) floor the σ used in the μ working-response denominator at σ₀ to prevent (Y−μ)/σ² blow-up when σ is locally small — the root cause of a μ-overflow / σ-saturation failure mode; (3) functional intercept in the σ-step so purely grid-varying heteroscedasticity is captured directly instead of being mis-located through a scalar-predictor learner; (4) data-adaptive clamp on the log-σ accumulator."

verification:
  module_tests: "6/6 pass — cargo test -p fdars-core --features linalg,parallel --lib boosting_regression::gamlss"
  tests:
    - gamlss_recovers_mean_and_scale (μ tracks data R²>0; fitted σ larger on the high-variance grid half — heteroscedasticity recovered in the correct direction)
    - gamlss_sigma_positive_everywhere (σ̂ > 0 at every point, log-link guard)
    - gamlss_loglik_non_decreasing (Gaussian log-likelihood improves along the boosting path)
    - gamlss_result_shapes (mu_fitted/sigma_fitted n×m_t; mu_beta/sigma_beta p×m_t; ll_path length mstop; sigma_intercept length m_t)
    - gamlss_errors_on_dimension_mismatch (FdarError, no panic)
    - gamlss_errors_on_invalid_params (FdarError, no panic)

notes:
  - "A prior executor authored the implementation + tests but died on transient API (529) errors before first successful compile; a second failure mode (σ-recovery direction) was diagnosed and fixed inline by the orchestrator. Two commits: initial partial (superseded within the same working tree) and the final feat(43-03) commit 6905f12f."
  - "Full crate-wide clippy + fmt + test gate deferred to the phase-end out-of-band gate (per repo 600s-watchdog convention)."

commits:
  - "6905f12f feat(43-03): GAMLSS location+scale distributional regression (REG-06-03)"
---

# Plan 43-03 — GAMLSS location+scale distributional regression (REG-06-03)

Implemented `gamlss_fosr` in `fdars-core/src/boosting_regression/gamlss.rs`: a Gaussian
location+scale distributional functional regression via cyclic gamboostLSS-style boosting.
Each iteration runs one component-wise boosting step for the location model μ(t) (identity
link) and one for the log-scale model σ(t) (log link), reusing the penalized B-spline
base-learners from the Plan 01 boosting core.

**Requirement REG-06-03** — "fit a GAMLSS-style distributional functional regression that
models more than one distributional parameter (location + scale)" — is satisfied:
`gamlss_fosr` returns fitted μ(t) and σ(t), per-parameter coefficient functions, the
functional σ intercept, and the Gaussian log-likelihood path.

## Numerical robustness

The coupled μ/σ dynamics are stiff: a locally small σ makes the μ working-response
`(Y−μ)/σ²` explode, which boosts μ into overflow and saturates σ at the wrong scale.
Fixed with (1) marginal-scale warm-start of σ, (2) a marginal-scale floor on the σ used in
the μ-gradient denominator, (3) a functional σ intercept so grid-varying heteroscedasticity
is captured directly, and (4) a data-adaptive clamp on the log-σ accumulator. All documented
in rustdoc. The recovery test confirms μ tracks the data and fitted σ is larger in the
high-variance region — heteroscedasticity recovered in the correct direction.

## Verification

6/6 module tests pass. Crate-wide clippy/fmt/test gate runs at phase end.
