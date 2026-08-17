---
phase: 25-functional-glm-exponential-family
plan: "01"
subsystem: scalar_on_function
tags: [glm, exponential-family, irls, functional-regression, poisson, gamma, gaussian, binomial]
status: complete

dependency_graph:
  requires: []
  provides:
    - functional_glm (pub fn, crate root re-export)
    - predict_functional_glm (pub fn, crate root re-export)
    - GlmFamily (pub enum, crate root re-export)
    - FunctionalGlmResult (pub struct, crate root re-export)
  affects:
    - fdars-core/src/scalar_on_function/mod.rs
    - fdars-core/src/lib.rs

tech_stack:
  added: []
  patterns:
    - Generic IRLS loop over FPC scores (generalisation of functional_logistic pattern)
    - GlmFamily enum dispatch for link/variance/deviance per family
    - Deviance-change convergence criterion
    - Gamma intercept initialisation (beta[0] = 1/mean(y))
    - Binomial parity via identical IRLS math to logistic.rs

key_files:
  created:
    - fdars-core/src/scalar_on_function/glm.rs
  modified:
    - fdars-core/src/scalar_on_function/mod.rs
    - fdars-core/src/lib.rs

decisions:
  - "GlmFamily and FunctionalGlmResult defined in mod.rs alongside other result types; glm.rs contains only the implementation (methods, helpers, entry points)"
  - "Poisson log(y!) computed exactly as sum(ln(k) for k=1..=y) — no statrs needed (statrs is absent from Cargo.toml despite RESEARCH.md claim)"
  - "Binomial link_deriv = 1/(mu*(1-mu)) gives z=eta+(y-mu)/(mu*(1-mu)) which matches logistic.rs irls_step verbatim — SC2 parity"
  - "Recovery tests use make_rich_data with 3 orthogonal sine-basis components and permuted scores to ensure X'WX is non-singular with ncomp=3"
  - "Convergence: deviance-change < tol (as specified in CONTEXT.md; deviates from logistic.rs coefficient-change but does not affect Binomial parity at convergence)"

metrics:
  duration_minutes: 16
  completed_date: "2026-08-17"
  tasks_completed: 4
  tasks_total: 4
  commits: 1

actuals:
  tokens: 9200
  tasks: 4
  commits: 1
---

# Phase 25 Plan 01: Functional GLM (Exponential Family) Summary

Implemented `functional_glm` — a scalar-on-function GLM covering four exponential-family
distributions (Binomial, Poisson, Gamma, Gaussian) via a generic IRLS loop over FPC scores,
reusing the existing logistic IRLS scaffold verbatim.

## What Was Built

New public symbols (all additive; no existing signature changed):

| Symbol | Kind | File |
|--------|------|------|
| `functional_glm` | `#[must_use]` pub fn | glm.rs |
| `predict_functional_glm` | pub fn | glm.rs |
| `GlmFamily` | `#[non_exhaustive]` pub enum (4 variants) | mod.rs |
| `FunctionalGlmResult` | `#[non_exhaustive]` pub struct (15 fields) | mod.rs |

### Architecture

```
functional_glm(data, y, family, scalar_covariates, ncomp, max_iter, tol)
  → validate_response (domain guard, before FPCA)
  → fdata_to_pc_1d (FPC scores)
  → build_design_matrix [1 | ξ₁…ξK | z₁…zP]
  → irls_loop_glm
      for iter in 0..max_iter:
        irls_step_glm: eta=Xβ; mu=inv_link(eta); w=irls_weight(mu)
                        z=eta+(y-mu)*link_deriv(mu)   ← link_deriv separate from irls_weight
                        (X'WX)β_new = X'Wz via cholesky_solve
        break if |dev_new - dev_old| < tol
  → build_glm_result: beta_t, beta_se, AIC, BIC, deviance, log_likelihood, family
```

### Key Technical Decisions

**Gamma sign safety:** `link_deriv(mu) = -1/mu²` (negative). Working response uses
`z = eta + (y-mu) * link_deriv`, NEVER `z = eta + (y-mu)/weight`. This is the Gamma
sign-safety invariant — weight is always positive but link_deriv carries the sign.

**Poisson log(y!):** `statrs` is not in `Cargo.toml` (RESEARCH.md claim was incorrect).
Computed exactly as `Σ_{k=1}^{y} ln(k)` — valid since Poisson y is validated to be a
non-negative integer before fitting. No new dependency added.

**Gamma init:** `beta[0] = 1/mean(y)` so `η₀ > 0` and `μ₀ = mean(y)`. Zero init for
Poisson/Binomial/Gaussian works because their inv_link is finite at η=0.

**Recovery test data:** Uses `make_rich_data` with 3 orthogonal sine-basis components
and index-permuted decorrelated scores so that X'WX is non-singular at ncomp=3. A
rank-1 signal causes Cholesky failure on the first Poisson IRLS step (discovered during
execution — auto-fixed under Rule 1).

## Success Criteria Verification

| SC | Description | Result |
|----|-------------|--------|
| SC1 | functional_glm + predict_functional_glm + GlmFamily + FunctionalGlmResult re-exported at crate root | PASS |
| SC2 | functional_glm(…, Binomial) reproduces functional_logistic within 1e-6; logistic signature unchanged | PASS |
| SC3 | Poisson/Gamma recover generative signals; all fitted_values finite and positive | PASS |
| SC4 | Out-of-domain responses + dimension mismatches return FdarError, never panic | PASS |
| SC5 | Full suite + clippy --all-targets green; no new dependency | PASS |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Poisson recovery test: rank-1 signal causes Cholesky failure**

- **Found during:** Task 3 (Poisson recovery test execution)
- **Issue:** `make_signal_data` generated a rank-1 functional dataset (all curves = score_i * sin(πt) + linear_trend). FPCA produces non-zero component 1 but near-zero components 2 and 3. The Cholesky solve on X'WX failed (singular) on the first IRLS step, leaving beta = [0,0,0,0] and all fitted_values = exp(0) = 1.0, giving Pearson corr = 0.
- **Fix:** Replaced `make_signal_data` with `make_rich_data`, generating 3 orthogonal sine-basis components (sin(πt), sin(2πt), sin(3πt)) with index-permuted decorrelated scores (s0=i/n, s1=(3i%n)/(n-1), s2=(7i%n)/(n-1)), ensuring all 3 FPCA components capture real variance and X'WX is non-singular.
- **Files modified:** `fdars-core/src/scalar_on_function/glm.rs` (tests::make_rich_data replaces make_signal_data; test_gamma_recovery updated similarly)
- **Commit:** cb839d52

**2. [Rule 3 - Blocking] statrs::function::gamma::ln_gamma missing from Cargo.toml**

- **Found during:** Task 1 (compilation)
- **Issue:** RESEARCH.md stated `statrs` was "already a dependency" but `fdars-core/Cargo.toml` has no `statrs` entry. `use statrs::function::gamma::ln_gamma` failed to compile.
- **Fix:** Replaced `ln_gamma` with exact integer factorial: `Σ_{k=1}^{y} ln(k)` (valid because Poisson y is validated to be a non-negative integer before fitting). Equivalent to `ln_gamma(y+1)` for integer y.
- **Files modified:** `fdars-core/src/scalar_on_function/glm.rs` (log_likelihood Poisson arm)
- **Commit:** cb839d52

## Self-Check

All files created:
- `fdars-core/src/scalar_on_function/glm.rs` — FOUND
- `GlmFamily` + `FunctionalGlmResult` in `mod.rs` — FOUND

Commit exists:
- `cb839d52` — FOUND (`feat(25-01): functional_glm — exponential-family GLM over FPC scores (REG-02)`)

Full test suite: 2079 tests passed; 0 failed.
Clippy `--all-targets`: clean.

## Self-Check: PASSED
