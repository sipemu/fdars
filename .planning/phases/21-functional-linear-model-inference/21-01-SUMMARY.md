---
phase: 21-functional-linear-model-inference
plan: 01
subsystem: inference
tags: [inference, flm, anova, f-test, goodness-of-fit, v-statistic, INF-02]
requires:
  - Phase-20 inference/ module (TestResult, chi-square SF)
  - scalar_on_function::fregre_lm / FregreLmResult
  - function_on_scalar::compute_group_means (group-mean logic)
  - helpers::simpsons_weights
provides:
  - inference::flm_f_test (overall-significance F-test on a fitted FLM)
  - inference::flm_gof_test (residual-based F-form lack-of-fit test)
  - inference::oneway_anova_vstat (asymptotic V-statistic ANOVA)
  - inference::dist (crate-internal home for chi_square_sf + f_sf)
affects:
  - fdars-core crate-root public API (three new re-exported fns)
tech-stack:
  added: []
  patterns:
    - self-contained F-distribution SF via regularized incomplete beta (Lentz betacf)
    - scaled-chi2 (Box/Satterthwaite) moment match for the fdANOVA V-null
key-files:
  created:
    - fdars-core/src/inference/dist.rs
    - fdars-core/src/inference/flm.rs
    - fdars-core/src/inference/anova.rs
  modified:
    - fdars-core/src/inference/hotelling.rs
    - fdars-core/src/inference/mod.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/function_on_scalar.rs
decisions:
  - "flm_gof_test null method = F-form Ramsey-RESET-style lack-of-fit: regress FLM residuals on a cubic polynomial of standardized fitted values, F-test the joint significance of the polynomial terms; reject = mis-specification."
  - "oneway_anova_vstat p-value via scaled-chi2 (Box/Satterthwaite) approximation of the weighted-sum-of-chi2 V-null, calibrated against the pooled within-group variance (diagonal covariance approximation)."
  - "No new crate dependency: f_sf implemented self-contained via the regularized incomplete beta function reusing the existing Lanczos ln_gamma."
metrics:
  duration: ~1h20m
  completed: 2026-08-16
actuals:
  tokens: 9264
  tasks: 4
  commits: 4
status: complete
---

# Phase 21 Plan 01: Functional-Linear-Model Inference Summary

Added formal FLM inference to the Phase-20 `inference/` module — `flm_f_test`, `flm_gof_test`, and `oneway_anova_vstat` — plus a self-contained F-distribution survival function, all crate-root re-exported with statistical-correctness tests and zero changes to any existing public signature. Closes INF-02, the final requirement of milestone v0.19.0.

## What was built

- **`inference/dist.rs`** (new, crate-internal): single home for the distribution tails. Relocated the gamma machinery (`gamma_p_series`, `gamma_q_cf`, `ln_gamma`, `chi_square_sf`) verbatim out of `hotelling.rs` (private refactor, zero behavior change) and added:
  - `f_sf(f, d1, d2)` — F upper-tail via the regularized incomplete beta `I_x(a,b)` (Numerical-Recipes Lentz `betacf`, symmetry swap for stability), using `SF_F(f) = I_{d2/(d2+d1·f)}(d2/2, d1/2)`; `f <= 0 → 1.0`. No new crate dependency (reuses `ln_gamma`).
  - `chi_square_sf_df(x, df)` — real-valued-df χ² tail for the scaled-χ² V-null.
- **`inference/flm.rs`** (new):
  - `flm_f_test(&FregreLmResult)` — overall-significance F-test, `F = (R²/p)/((1−R²)/(n−p−1))`, `p = ncomp`, p-value from `f_sf(F, p, n−p−1)`. Validates `ncomp == 0`, degenerate df, non-finite / `r_squared >= 1.0`.
  - `flm_gof_test(&FregreLmResult)` — residual-based F-form lack-of-fit (see chosen null below).
- **`inference/anova.rs`** (new): `oneway_anova_vstat(data, groups, argvals)` — Simpson-integrated between-group `V = ∫ Σ_g n_g (x̄_g(t) − x̄(t))² dt` with a scaled-χ² asymptotic p-value; reuses `compute_group_means` + `simpsons_weights`.
- Crate-root re-exports (`lib.rs`): `flm_f_test`, `flm_gof_test`, `oneway_anova_vstat`.
- `function_on_scalar::compute_group_means` widened to `pub(crate)` (visibility only; `fanova` untouched).

## flm_gof_test — chosen null method

**F-form residual lack-of-fit (Ramsey-RESET style).** H0: the linear FLM is well specified. The test regresses the fitted model's residuals `e_i` on a cubic polynomial of the standardized fitted values, `e_i = a0 + a1·ŷ_i + a2·ŷ_i² + a3·ŷ_i³ + u_i`, and reports the F-statistic for `H0: a1 = a2 = a3 = 0` (q = 3 restrictions), referenced to F(3, n−4). A small p-value rejects adequacy of the linear FLM: an unmodelled nonlinearity leaves curvature in the residual-vs-fitted relationship that the polynomial terms capture. Documented in the function's rustdoc (statistic, reference distribution, rejection direction).

## Tests added

13 new inline tests relative to the Phase-20 baseline:

- `dist::f_sf_matches_tabulated_quantiles` — `f_sf` vs tabulated F(1,10)/F(5,20)/F(3,30) quantiles, `f<=0→1`, monotone, bounded; plus a real-df χ² cross-check in `chi_square_sf_sane`.
- `flm` (6): `flm_f_test` rejects a genuine functional effect (p<0.05, SC-1a) and fails to reject a null-effect fit (p>0.20, SC-1b), guards degenerate df; `flm_gof_test` fails to reject a well-specified linear fit (p>0.10, SC-2a), flags a mis-specified quadratic fit (p<0.05, SC-2b), guards degenerate df.
- `anova` (4): `oneway_anova_vstat` rejects separated groups (agreeing in direction with permutation `fanova`, SC-3), fails to reject pooled groups (agreeing with `fanova`), deterministic statistic, input-validation Err.

All statistical-correctness tests (mandatory per CONTEXT §Specifics) pass with deterministic data construction.

## Verification

- `cargo test -p fdars-core --features linalg,parallel` — **green: 2039 lib tests + all integration/doc tests pass, 0 failed** (was 2010 lib pre-plan; +29 inference tests now in the module, 13 net-new for this plan).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — **clean**.
- Relocated `chi_square_sf` tests in `hotelling` still pass unchanged; existing `fanova` / `function_on_scalar` tests (24) unchanged and green.
- Pre-commit gate (fmt + clippy + doc) passed on every commit — no `--no-verify` needed; /tmp doctest-linking issue did not surface (TMPDIR redirected to the cache dir).

## Success criteria

- SC-1: `flm_f_test` rejects genuine effect (p<0.05) and fails to reject null-effect (p>0.20). ✓
- SC-2: `flm_gof_test` fails to reject well-specified, flags mis-specified; null method documented in rustdoc. ✓
- SC-3: `oneway_anova_vstat` agrees in reject/accept direction with permutation `fanova` on separated vs pooled groups and is deterministic. ✓
- SC-4: All three fns `Result<_, FdarError>`, input-validated, crate-root re-exported, inline `#[cfg(test)]`; **no existing public signature changed** (`fanova` and all others intact); no new crate dependency. ✓

## Deviations from Plan

None — plan executed as written. The only production-code adjustment outside the three new fns is the additive `pub(crate)` visibility widening of `compute_group_means` (explicitly sanctioned by the plan/CONTEXT). One rustdoc intra-doc link was demoted to plain text to satisfy the `-D warnings` doc gate (a public fn cannot link to the crate-internal `chi_square_sf_df`).

## Known Stubs

None.

## Self-Check: PASSED

- Created files verified on disk: `dist.rs`, `flm.rs`, `anova.rs`, `21-01-SUMMARY.md`.
- Commits verified in git log: `fd05c9da`, `7f9ebf00`, `8706417d`, `d4727c2d`.
