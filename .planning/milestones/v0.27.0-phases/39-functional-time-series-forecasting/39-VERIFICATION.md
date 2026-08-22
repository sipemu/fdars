---
phase: 39-functional-time-series-forecasting
verified: 2026-08-22T16:30:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 39: Functional Time-Series Forecasting Verification Report

**Phase Goal:** A user can forecast future functional curves from a time-ordered curve series — the FPCA-based `ftsm`, FPC-score-regression forecasting, a functional PLS variant (`fplsr`), dynamic forecast updating, and iterative multi-step (h>1) forecasting — all in new `fdars-core/src/fts/forecast.rs`, reusing `fdata_to_pc_1d` + `scoring.rs` + `fts/acf.rs`, additive/non-breaking, no new crate dependency. R baseline: `ftsa`.
**Verified:** 2026-08-22T16:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | New Result-returning, crate-root-re-exported entry points in `fts/forecast.rs` consuming a column-major FdMatrix curve series: `ftsm` fit, `ftsm_forecast`, `fplsr`, `ftsm_update`, `ftsm_forecast_multistep`. | ✓ VERIFIED | All five symbols present in `forecast.rs` (lines 267, 357, 382, 449, 554); all exported via `fts/mod.rs` line 26 and `lib.rs` line 253. |
| 2 | `ftsm` decomposes via `fdata_to_pc_1d`, retains mean/loadings/score-series, reconstructs fitted curves recovering the input within documented tolerance (inline tests). | ✓ VERIFIED | `fdata_to_pc_1d` called at line 288; `fpca.reconstruct` at line 290; `ftsm_fitted_recovers_input` test asserts MSE < 1% of data variance; `ftsm_deterministic` asserts bit-identical repeat calls. 23/23 tests pass. |
| 3 | h-step forecast via AR models on FPC-score sequences; on a synthetic AR-score series the forecast recovers the AR one-step prediction within tolerance and beats a naive last-curve baseline (inline tests). | ✓ VERIFIED | `forecast_recovers_ar_one_step` (phi recovery within 0.12, score within 25% relative slack) and `forecast_beats_naive_baseline` (model MSE < naive MSE) both pass. `ArModel::fit` implements Yule-Walker + AIC; `ArModel::forecast` implements iterative plug-in. |
| 4 | `fplsr` PLS-score forecasts; dynamic-update agrees with a full refit within tolerance; iterative multi-step returns per-horizon curves whose h=1 matches the single-step forecast (inline tests). | ✓ VERIFIED | `fplsr_produces_finite_forecast`, `fplsr_no_worse_than_naive`, `update_agrees_with_refit` (< 1% relative-L2), `update_freezes_loadings`, `multistep_h1_equals_single_step` (< 1e-12 per point), `multistep_returns_h_rows` (h=5 yields 5 x m) all pass. `ftsm_forecast` delegates to `ftsm_forecast_multistep` (forecast.rs line 362). |
| 5 | All entry points reuse `fdata_to_pc_1d` + `scoring.rs` + `fts/acf.rs` (no new subsystem), no new crate dependency, invalid inputs return FdarError not panic; existing public signatures unchanged; full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green. | ✓ VERIFIED | `fdata_to_pc_1d` imported line 49, `scoring::functional_mse` used in tests; `validate_fts_input` re-implemented verbatim from acf.rs per plan spec (design intent). `Cargo.toml` unchanged across phase commits. Error-path tests: `ftsm_rejects_empty`, `ftsm_rejects_ncomp_ge_n`, `ftsm_rejects_argvals_mismatch`, `forecast_rejects_h_zero`, `update_rejects_bad_shape`, `fplsr_rejects_bad_input` all pass. Clippy: `Finished dev profile` with zero warnings (verified independently). |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/fts/forecast.rs` | New file with all five entry points + inline tests | ✓ VERIFIED | 1,044 lines; all five public fns present; 23 `#[cfg(test)]` tests |
| `FtsmResult` (declared in `fts/mod.rs`) | mean, rotation, scores, fitted, weights, ncomp, ar_models | ✓ VERIFIED | `mod.rs` lines 99–117; four-attribute derive chain (Debug/Clone/PartialEq, serde cfg_attr, non_exhaustive) |
| `FtsmForecastResult` (declared in `fts/mod.rs`) | forecast (h x m), h | ✓ VERIFIED | `mod.rs` lines 119–131 |
| `ArModelResult` (declared in `fts/mod.rs`) | order, phi, sigma2 | ✓ VERIFIED | `mod.rs` lines 79–92 |
| `FplsrResult` (declared in `fts/mod.rs`) | forecast (1 x m), fitted ((n-1) x m), ncomp | ✓ VERIFIED | `mod.rs` lines 133–148 |
| `mod forecast;` in `fts/mod.rs` | Module wiring | ✓ VERIFIED | `mod.rs` line 21 |
| Crate-root re-exports in `lib.rs` | All five fns + four result structs | ✓ VERIFIED | `lib.rs` lines 252–256: `fplsr, ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update, ArModelResult, FplsrResult, FtsmForecastResult, FtsmResult` |
| Private AR helpers: `scalar_acov`, `levinson_durbin_yw`, `ArModel` | In `forecast.rs` | ✓ VERIFIED | Lines 83, 107, 147 respectively |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `ftsm` | `regression::fdata_to_pc_1d` | `use crate::regression::fdata_to_pc_1d` (line 49); call at line 288 | ✓ WIRED | No custom SVD |
| `ftsm` | `FpcaResult::reconstruct` | `fpca.reconstruct(&fpca.scores, effective_ncomp)?` line 290 | ✓ WIRED | Reuses shipped reconstruction |
| `ftsm_forecast` | `ftsm_forecast_multistep` | Thin delegate at line 362: `ftsm_forecast_multistep(fit, h, argvals)` | ✓ WIRED | Guarantees h=1 bit-identity by construction |
| `ftsm_update` | frozen loadings | `fit.mean[j]`, `fit.rotation[(j,k)]`, `fit.weights[j]` used directly; no `fdata_to_pc_1d` call | ✓ WIRED | Proven by `update_freezes_loadings` test (bitwise equal) |
| `fplsr` | `scalar_on_function::fregre_pls` / `predict_fregre_pls` | `use crate::scalar_on_function::{fregre_pls, predict_fregre_pls}` (line 50); calls at lines 590, 594 | ✓ WIRED | No custom NIPALS; per-evaluation-point scalar PLS |
| `forecast.rs` tests | `crate::scoring::functional_mse` | `use crate::scoring::functional_mse` (line 608) | ✓ WIRED | Used in `ftsm_fitted_recovers_input`, `forecast_beats_naive_baseline`, `fplsr_no_worse_than_naive`, `update_agrees_with_refit` |
| `fts/mod.rs` | `forecast` module + result structs | `mod forecast;` line 21; `pub use forecast::{...}` line 26; struct declarations lines 79–148 | ✓ WIRED | All five fns + four structs re-exported |
| `lib.rs` | `fts::{...}` block | `pub use fts::{...}` lines 252–256 | ✓ WIRED | All five public fns + four result structs at crate root |

---

### Data-Flow Trace (Level 4)

All data flows from real inputs: `ftsm` receives a caller-supplied `FdMatrix` and delegates to `fdata_to_pc_1d` (which performs actual SVD) — no static returns. `ftsm_forecast_multistep` reconstructs from the scored AR forecasts. `fplsr` iterates over real data to produce fitted values. No static/hardcoded data in any public entry point.

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `ftsm` → `FtsmResult.fitted` | `fitted` | `fpca.reconstruct(&fpca.scores, ncomp)` — real FPCA scores from caller data | Yes | ✓ FLOWING |
| `ftsm_forecast_multistep` → `FtsmForecastResult.forecast` | `forecast` | AR forecasts from `ArModel::forecast(h)` seeded by actual score history | Yes | ✓ FLOWING |
| `ftsm_update` → extended `scores` | `ext_scores` | Old scores copied + new rows projected via real weight/rotation inner product | Yes | ✓ FLOWING |
| `fplsr` → `FplsrResult.forecast` | `forecast` | Per-point `predict_fregre_pls` on real lag-1 design derived from caller data | Yes | ✓ FLOWING |

---

### Behavioral Spot-Checks

Tests were enumerated and run independently (not relying on SUMMARY claims):

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 23 `fts::forecast` tests pass | `cargo test -p fdars-core --features linalg,parallel -- fts::forecast` | 23 passed, 0 failed, 2414 filtered out | ✓ PASS |
| Clippy clean with `--all-targets` | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | `Finished dev profile` — zero warnings | ✓ PASS |

Specific tests verified by enumeration (`--list` output confirms each by name):
- SC2: `ftsm_fitted_recovers_input`, `ftsm_deterministic`, `ftsm_rejects_ncomp_ge_n`, `ftsm_rejects_empty`, `ftsm_rejects_argvals_mismatch`
- SC3: `forecast_recovers_ar_one_step`, `forecast_beats_naive_baseline`, `forecast_rejects_h_zero`, `ar_model_fit_and_forecast_ar1`, `levinson_durbin_recovers_ar1`, `levinson_durbin_rejects_zero_variance`, `scalar_acov_variance_and_decay`
- SC4: `multistep_h1_equals_single_step`, `multistep_returns_h_rows`, `multistep_rejects_h_zero`, `update_agrees_with_refit`, `update_freezes_loadings`, `update_extends_scores`, `update_rejects_bad_shape`, `fplsr_produces_finite_forecast`, `fplsr_no_worse_than_naive`, `fplsr_deterministic`, `fplsr_rejects_bad_input`

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FTS-01-01 | 39-01 | `ftsm` FPCA-based model fit with score-series + fitted reconstruction | ✓ SATISFIED | `ftsm` in forecast.rs; `ftsm_fitted_recovers_input` passes; `ftsm_deterministic` passes |
| FTS-01-02 | 39-01 | `ftsm_forecast` h-step score-AR forecast | ✓ SATISFIED | `ftsm_forecast` (delegates to `ftsm_forecast_multistep`); `forecast_recovers_ar_one_step`, `forecast_beats_naive_baseline` pass |
| FTS-01-03 | 39-03 | `fplsr` functional PLS lag-1 forecasting variant | ✓ SATISFIED | `fplsr` + `FplsrResult` in forecast.rs/mod.rs; four fplsr_ tests pass |
| FTS-01-04 | 39-02 | `ftsm_update` dynamic update without FPCA refit | ✓ SATISFIED | `ftsm_update` in forecast.rs; four update_ tests pass; loadings frozen (bitwise equal) |
| FTS-01-05 | 39-02 | `ftsm_forecast_multistep` iterative h>1 forecast, h=1 bit-identical to single-step | ✓ SATISFIED | `ftsm_forecast_multistep` in forecast.rs; `multistep_h1_equals_single_step` (< 1e-12 per point) passes |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No `TODO`, `FIXME`, `TBD`, `XXX`, `HACK`, or placeholder markers found in `forecast.rs`. No `return null`, stub handlers, or hardcoded empty arrays in any public entry point. No new crate dependency added to `Cargo.toml` during this phase.

One design note (not a blocker): `validate_fts_input` is re-implemented locally in `forecast.rs` rather than imported from `fts/acf.rs` (the private `acf.rs` helper is not accessible cross-module). This is explicitly documented in the plan: "Re-implement `validate_fts_input` and `mean_curve` verbatim as private helpers in forecast.rs (do NOT modify acf.rs)." The 1/n normalization convention is shared, satisfying SC5's "no new algorithm subsystem" intent.

---

### Human Verification Required

None. All success criteria are mechanically verifiable and have been verified via code inspection + independently-run cargo commands.

---

### Gaps Summary

No gaps. All five success criteria are fully met:

1. All five public entry points exist, are substantive, and are crate-root re-exported.
2. `ftsm` uses `fdata_to_pc_1d` and `FpcaResult::reconstruct`; fitted recovery < 1% relative-L2 asserted by test.
3. AR score forecasting recovers AR(1) coefficient within tolerance and beats naive baseline; both asserted by tests.
4. `fplsr`, `ftsm_update`, `ftsm_forecast_multistep` all implemented and tested; h=1 consistency guaranteed by single arithmetic path.
5. No new dependency; no existing signature changed; clippy clean; all error paths return `FdarError`.

---

_Verified: 2026-08-22T16:30:00Z_
_Verifier: Claude (gsd-verifier)_
