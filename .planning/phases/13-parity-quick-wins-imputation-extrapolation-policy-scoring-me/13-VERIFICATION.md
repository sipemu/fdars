---
phase: 13-parity-quick-wins-imputation-extrapolation-policy-scoring-me
verified: 2026-08-11T21:26:47Z
status: gaps_found
score: 7/8
behavior_unverified: 0
overrides_applied: 0
gaps:
  - truth: "ExtrapolationPolicy threads through spline_interpolate and the existing linear interpolation/evaluation path (ROADMAP SC #2 / FEAT-04)"
    status: partial
    reason: "fdata_interpolate_with_policy covers the linear+cubic-hermite path only. spline_interpolate still accepts no ExtrapolationPolicy argument and always errors on OOB (hardcoded Exception-equivalent). The ROADMAP SC says both paths must be covered; only one is."
    artifacts:
      - path: "fdars-core/src/helpers.rs"
        issue: "spline_interpolate (line 416) does not accept ExtrapolationPolicy. No spline_interpolate_with_policy exists. spline_interpolate hardcodes an OOB error (line 448-455), giving only Exception-equivalent behavior."
    missing:
      - "A spline_interpolate_with_policy(data, argvals, query_points, order, policy: ExtrapolationPolicy) -> Result<FdMatrix, FdarError> wrapper that dispatches OOB to the policy (Boundary=clamp, Exception=error, Fill=constant, Periodic=wrap) OR evidence that the ROADMAP SC was intentionally narrowed to exclude spline"
---

# Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics Verification Report

**Phase Goal:** Add three additive, non-breaking scikit-fda parity capabilities to fdars-core — in-grid NaN imputation (FEAT-03), a composable ExtrapolationPolicy for interpolation/evaluation (FEAT-04), and functional scoring metrics (FEAT-05). Every new public fn returns Result<_, FdarError>.
**Verified:** 2026-08-11T21:26:47Z
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | impute_missing_values(data, argvals, method) -> Result<FdMatrix, FdarError> exists in helpers.rs (FEAT-03) | VERIFIED | Line 862, fdars-core/src/helpers.rs; returns Result |
| 2 | ImputationMethod enum with Linear, Mean, Constant(f64) variants (FEAT-03) | VERIFIED | Lines 834-844, helpers.rs; all three variants present |
| 3 | impute_missing_values rejects all-NaN curves (InvalidParameter) and argvals mismatch (InvalidDimension) (FEAT-03) | VERIFIED | Lines 878-883, 868-873; test_impute_all_nan and test_impute_dim_mismatch pass |
| 4 | Boundary extension for leading/trailing NaN in Linear strategy (FEAT-03) | VERIFIED | Lines 924-925 ("boundary fill (leading/trailing NaN)"); test_impute_boundary_nan passes |
| 5 | ExtrapolationPolicy enum {Boundary, Exception, Fill(f64), Periodic} in helpers.rs (FEAT-04) | VERIFIED | Lines 715-734; all four variants with doc-comments |
| 6 | fdata_interpolate_with_policy applies all four policy variants; Exception -> FdarError (FEAT-04) | VERIFIED | Lines 756-821; all four branches implemented; test_extrapolation_exception passes with InvalidParameter{"new_argvals"} |
| 7 | Existing fdata_interpolate, linear_interp, spline_interpolate, cubic_hermite_interp signatures UNCHANGED (non-breaking) | VERIFIED | fdata_interpolate (line 366) and linear_interp (line 172) signatures unchanged; cubic_hermite_interp remains private fn (line 513); spline_interpolate (line 416) unchanged |
| 8 | ExtrapolationPolicy threads through spline_interpolate AND the existing linear path (ROADMAP SC #2) | FAILED | fdata_interpolate_with_policy covers linear+cubic-hermite only; spline_interpolate accepts no ExtrapolationPolicy and hardcodes OOB error (Exception-equivalent only). No spline_interpolate_with_policy exists. |
| 9 | Five functional scoring metrics: functional_mae/mse/mape/msle/explained_variance -> Result<f64, FdarError>, integrated over argvals (FEAT-05) | VERIFIED | scoring.rs lines 59-271; all five implemented; all return Result<f64, FdarError> |
| 10 | MAPE zero-denominator and MSLE domain violations return FdarError (FEAT-05) | VERIFIED | Lines 118-131 (MAPE pre-scan); lines 163-185 (MSLE pre-scan); test_functional_mape_zero_y_true and test_functional_msle_domain_y_true/y_pred pass |
| 11 | All five scoring functions re-exported at crate root (FEAT-05) | VERIFIED | lib.rs lines 443-445: pub use scoring::{functional_explained_variance, functional_mae, functional_mape, functional_mse, functional_msle} |
| 12 | cargo test passes (1984 tests, 0 failed) and clippy --all-targets -D warnings clean | VERIFIED | Confirmed by direct runs: 12 imputation tests + 15 scoring tests pass; clippy: "Finished dev profile" with no warnings |

**Score:** 7/8 must-haves verified (truth #8 FAILED — spline_interpolate_with_policy absent)

Note: ROADMAP SC #2 explicitly says both "spline_interpolate AND the existing linear interpolation/evaluation path." Eleven of twelve observable truths pass; only the spline thread-through is missing.

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/helpers.rs` | ImputationMethod enum, impute_missing_values fn, ExtrapolationPolicy enum, fdata_interpolate_with_policy fn + inline tests | VERIFIED | All four new items present; 12 inline tests (6 extrapolation + 6 imputation); file line count ~1638 |
| `fdars-core/src/scoring.rs` | Five functional_* metric fns + inline tests + validate_shapes helper | VERIFIED | 499 lines; all five fns + private validate_shapes + 15 inline tests |
| `fdars-core/src/lib.rs` | pub mod scoring; + pub use scoring::{...} + ExtrapolationPolicy, ImputationMethod, fdata_interpolate_with_policy, impute_missing_values in helpers re-export | VERIFIED | All items confirmed at lines 124, 175-178, 443-445 |

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| impute_missing_values | helpers::linear_interp | impute_row -> linear_interp([argvals[l], argvals[r]], [row[l], row[r]], argvals[j]) | VERIFIED | Line 922; Linear strategy delegates to existing linear_interp |
| fdata_interpolate_with_policy | linear_interp / cubic_hermite_interp | dispatch in in-range branch + Boundary/Periodic branches | VERIFIED | Lines 786-814; both interpolators called per-method |
| scoring module | helpers::simpsons_weights | use crate::helpers::{simpsons_weights, NUMERICAL_EPS}; | VERIFIED | Line 13 of scoring.rs; all five metrics call simpsons_weights(argvals) once |
| New items | crate root | pub use helpers::{...} + pub use scoring::{...} | VERIFIED | lib.rs lines 175-178, 443-445 |

## Data-Flow Trace (Level 4)

All five scoring metrics: data flows from FdMatrix elements to weighted sum via simpsons_weights; non-static. Imputation: NaN replaced by computed interpolated values from neighbors (Linear) or computed mean (Mean) or user-supplied constant (Constant(f64)) — not hardcoded zeros. All data flows are real and non-hollow.

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 6 ExtrapolationPolicy tests (all variants) | cargo test -- helpers::tests::test_extrapolation | 6 passed, 0 failed | PASS |
| 6 ImputationMethod tests (all strategies + error cases) | cargo test -- helpers::tests::test_impute | 6 passed, 0 failed | PASS |
| 15 scoring tests (all 5 metrics, error cases) | cargo test -- scoring:: | 15 passed, 0 failed | PASS |
| Full crate suite | cargo test -p fdars-core --features linalg | 1984 passed, 0 failed | PASS |
| Clippy strict | cargo clippy --features linalg --all-targets -- -D warnings | Clean (no warnings) | PASS |

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FEAT-03 | 13-01-PLAN.md | In-grid NaN imputation: ImputationMethod + impute_missing_values | SATISFIED | helpers.rs lines 827-933; 6 inline tests; rejects all-NaN curves |
| FEAT-04 | 13-01-PLAN.md | ExtrapolationPolicy: 4-variant enum + fdata_interpolate_with_policy | PARTIAL | Linear/cubic path covered; spline path NOT covered. ROADMAP SC 2 says "threads through spline_interpolate" — gap |
| FEAT-05 | 13-02-PLAN.md | 5 scoring metrics in scoring.rs, integrated over argvals | SATISFIED | scoring.rs 499 lines; 15 inline tests; all re-exported at crate root |

## Anti-Patterns Found

No TBD, FIXME, or XXX markers in helpers.rs, scoring.rs, or lib.rs. No placeholder returns. No hardcoded empty data. No TODO comments. No debt markers.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

## Human Verification Required

None — all checks are programmatic. The tests and clippy run confirm the implementation.

## Gaps Summary

**1 gap blocking full ROADMAP SC achievement:**

ROADMAP Success Criterion #2 states: "A composable `ExtrapolationPolicy` enum … threads through `spline_interpolate` **and** the existing linear interpolation/evaluation path to control behavior for query points outside `argvals`."

The implementation provides `fdata_interpolate_with_policy` which covers the linear + cubic hermite paths. However, `spline_interpolate` is NOT wrapped: it retains its original signature and always errors on OOB (hardcoded InvalidParameter, equivalent to Exception-only). A `spline_interpolate_with_policy` wrapper was not created.

The PLAN explicitly decided against threading through spline (RESEARCH.md: "The cleanest non-breaking extension is a new function `fdata_interpolate_with_policy`…"), and the CONTEXT.md LOCKED decision says "wrapper or new fn" was acceptable. But the CONTEXT.md "Claude's discretion" description for FEAT-04 says "thread the policy through the interpolation/evaluation path" without specifying spline is excluded. The ROADMAP SC says both paths explicitly.

**This deviation is intentional per planning artifacts, but it is a ROADMAP-SC-level gap, not just a plan deviation.** The developer must either:
1. Add `spline_interpolate_with_policy` to close the gap, OR
2. Accept this via an override (the spline path already enforces Exception-equivalent behavior, and the new wrapper covers the more common linear path)

If accepting via override, add to VERIFICATION.md frontmatter:

```yaml
overrides:
  - must_have: "ExtrapolationPolicy threads through spline_interpolate and the existing linear path"
    reason: "spline_interpolate already errors on OOB (Exception-equivalent hardcoded). The new fdata_interpolate_with_policy covers the dominant linear+cubic path. Adding a spline wrapper was descoped at research time as CONTEXT.md permits 'wrapper or new fn' approach."
    accepted_by: "{your name}"
    accepted_at: "{ISO timestamp}"
```

---

_Verified: 2026-08-11T21:26:47Z_
_Verifier: Claude (gsd-verifier)_
