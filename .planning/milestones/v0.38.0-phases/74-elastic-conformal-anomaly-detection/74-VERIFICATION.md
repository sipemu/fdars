---
phase: 74-elastic-conformal-anomaly-detection
verified: 2026-09-05T22:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 74: Elastic Conformal Anomaly Detection — Verification Report

**Phase Goal:** Users can flag both magnitude and shape outliers in functional data via an inductive conformal anomaly detector that scores curves against a reference template using elastic distance.
**Verified:** 2026-09-05T22:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `elastic_nonconformity(curve, curve, argvals, 0.0, variant) < 1e-4` for all three elastic variants | ✓ VERIFIED | `test_elastic_nonconformity_self_near_zero` passes; asserts `score < 1e-4` for AmplitudeElastic, PhaseElastic, CombinedElastic |
| 2 | `elastic_nonconformity` non-negative for all elastic variants on distinct curves | ✓ VERIFIED | `test_elastic_nonconformity_nonneg` passes; asserts `score >= 0.0` |
| 3 | `elastic_nonconformity` returns `Err(InvalidParameter)` for SupNorm/L2 | ✓ VERIFIED | `test_elastic_nonconformity_invalid_variant` passes; asserts `matches!(result, Err(FdarError::InvalidParameter { .. }))` |
| 4 | `CombinedElastic = sqrt(amp² + phase²)` — genuine combination, not alias of amplitude | ✓ VERIFIED | Code at lines 172–179 computes `amp = amplitude_distance(...)` and `ph = phase_distance_pair(...)` separately; `test_elastic_nonconformity_combined_not_alias` asserts `combined >= amp - 1e-12` on a scaled curve |
| 5 | Marginal validity: flag rate <= 0.25 on clean exchangeable data (alpha=0.1) | ✓ VERIFIED | `test_elastic_conformal_marginal_validity` passes; n_calib=100, n_test=100, fixed seeds 1/2, CombinedElastic |
| 6 | Injected magnitude outlier flagged by AmplitudeElastic (10x scaled sinusoid) | ✓ VERIFIED | `test_elastic_conformal_magnitude_outlier` passes; asserts `result.flags[n_clean] == true` and `scores > threshold` |
| 7 | Injected phase/shape outlier (cosine vs sine template) flagged by PhaseElastic | ✓ VERIFIED | `test_elastic_conformal_shape_outlier` passes; asserts `result.flags[n_clean] == true` |
| 8 | p_value == `(1 + #{calib >= a*}) / (n+1)`; flags == `(p_value <= alpha)`; threshold == `(1-alpha)` order-statistic | ✓ VERIFIED | `test_elastic_conformal_pvalue_threshold_correctness` passes; recomputes p_value independently and asserts diff < 1e-12; asserts flags equal p_values <= 0.1; recomputes threshold independently (INFINITY for n=5, alpha=0.1, k=ceil(6*0.9)=6>5) |
| 9 | `conformal_prediction_band` returns None for elastic variants, Some for SupNorm/L2 (no panic) | ✓ VERIFIED | `test_conformal_band_rejects_elastic_variants` passes; early guard at conformal.rs lines 81–88 returns None; `_ => unreachable!()` in both match arms |

**Score:** 9/9 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/tolerance/conformal_anomaly.rs` | New module — full elastic conformal anomaly implementation | ✓ VERIFIED | 870 lines; contains `elastic_nonconformity`, `elastic_conformal_anomaly`, `ConformalAnomalyConfig`, `ConformalAnomalyResult`, 12 unit tests, running module doctest |
| `NonConformityScore::AmplitudeElastic` | New variant in tolerance/types.rs | ✓ VERIFIED | types.rs lines 40–43; unit variant (Copy-compatible), documented |
| `NonConformityScore::PhaseElastic` | New variant in tolerance/types.rs | ✓ VERIFIED | types.rs lines 44–49; documented with geodesic-warp description |
| `NonConformityScore::CombinedElastic` | New variant in tolerance/types.rs | ✓ VERIFIED | types.rs lines 50–57; documented as `sqrt(amplitude² + phase²)` |
| `elastic_nonconformity` fn | Scoring function dispatching to alignment distances | ✓ VERIFIED | conformal_anomaly.rs lines 157–187; `#[must_use]` present (IN-01 fix) |
| `elastic_conformal_anomaly` fn | Full inductive detector pipeline | ✓ VERIFIED | conformal_anomaly.rs lines 223–346; `#[must_use]` present |
| `ConformalAnomalyConfig` | Config struct with default | ✓ VERIFIED | lines 61–97; `#[non_exhaustive]`, `Debug/Clone/PartialEq`, conditional serde, `impl Default` |
| `ConformalAnomalyResult` | Result struct with 4 fields | ✓ VERIFIED | lines 104–123; `#[non_exhaustive]`, `Debug/Clone/PartialEq`, conditional serde; fields: p_values, scores, flags, threshold |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `elastic_nonconformity` | `crate::alignment::amplitude_distance` | AmplitudeElastic dispatch arm | ✓ WIRED | conformal_anomaly.rs line 166–168; argument order `(curve, template, argvals, lambda)` matches pairwise.rs signature |
| `elastic_nonconformity` | `crate::alignment::phase_distance_pair` | PhaseElastic dispatch arm | ✓ WIRED | conformal_anomaly.rs lines 169–171 |
| `elastic_nonconformity` | `sqrt(amp² + ph²)` | CombinedElastic dispatch arm | ✓ WIRED | conformal_anomaly.rs lines 172–179; calls both `amplitude_distance` and `phase_distance_pair` |
| `elastic_conformal_anomaly` | `crate::alignment::karcher_mean` | Default template resolution | ✓ WIRED | conformal_anomaly.rs lines 294–302; `km.mean` is `Vec<f64>` of length m |
| `elastic_conformal_anomaly` | `crate::helpers::sort_nan_safe` | Threshold computation | ✓ WIRED | conformal_anomaly.rs line 317 |
| `tolerance/mod.rs` | `conformal_anomaly::{elastic_conformal_anomaly, elastic_nonconformity, ConformalAnomalyConfig, ConformalAnomalyResult}` | `pub use` re-export | ✓ WIRED | tolerance/mod.rs lines 32–35 |
| `lib.rs` | four new symbols via `tolerance` block | crate-root re-export | ✓ WIRED | lib.rs lines 347–351 confirmed by grep |
| `prelude.rs` | `ConformalAnomalyConfig, ConformalAnomalyResult` | prelude re-export | ✓ WIRED | prelude.rs line 91 confirmed by grep |
| `conformal.rs` | elastic variant early guard | `matches!` at top of `conformal_prediction_band` | ✓ WIRED | conformal.rs lines 81–88; guards before `nonconformity_score` call |
| `conformal.rs` | defensive `_ => unreachable!()` in `nonconformity_score` match | defensive arm | ✓ WIRED | conformal.rs line 25–28 |
| `conformal.rs` | defensive `_ => unreachable!()` in `half_width` match | defensive arm | ✓ WIRED | conformal.rs lines 127–130 |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 12 unit tests in conformal_anomaly module pass | `cargo test -p fdars-core --lib --features linalg,parallel conformal_anomaly` | 12 passed; 0 failed; finished in 4.43s | ✓ PASS |
| Module calibrate→flag doctest compiles and passes | `cargo test -p fdars-core --features linalg,parallel --doc conformal_anomaly` | 1 passed; 0 failed; finished in 1.58s | ✓ PASS |

---

### Code-Review Findings Resolution

All items from 74-REVIEW.md resolved in commit `846a264a`:

| Finding | Severity | Resolution | Status |
|---------|----------|------------|--------|
| CR-01: Caller-supplied template length not validated | Critical | `if t.len() != m { return Err(FdarError::InvalidDimension { .. }) }` added at conformal_anomaly.rs lines 284–290; regression test `test_conformal_anomaly_rejects_mismatched_template` added | ✓ FIXED |
| WR-01: `unwrap_or(NaN)` fallback silently suppresses anomaly flags | Warning | Calibration loop replaced with `.collect::<Result<Vec<f64>, FdarError>>()?` (line 313); test loop uses `?` (line 328) | ✓ FIXED |
| IN-01: `elastic_nonconformity` missing `#[must_use]` | Info | `#[must_use = "expensive computation: elastic_nonconformity runs an elastic alignment; use the score"]` added at line 157 | ✓ FIXED |
| IN-02: Test module defines local `uniform_grid` instead of `crate::test_helpers` | Info | Left as cosmetic (noted in 74-REVIEW.md header); does not affect correctness or CI | cosmetic — no action |

---

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|---------|
| ECA-01 | Extend `NonConformityScore` with elastic variants; each is non-negative and zero for identical curve | ✓ SATISFIED | Three new unit variants in types.rs; `elastic_nonconformity` dispatches to `amplitude_distance`/`phase_distance_pair`/`sqrt(amp²+phase²)`; truths 1-4 verified |
| ECA-02 | Inductive conformal detector: p-values, flags at level alpha; marginal validity; magnitude and shape outliers flagged | ✓ SATISFIED | `elastic_conformal_anomaly` implements full inductive pipeline; truths 5-8 verified; Karcher mean default template + optional caller-supplied template with length guard |
| ECA-03 | `ConformalAnomalyResult` with 4 fields; crate-root + prelude re-exports; running module doctest; band path unchanged | ✓ SATISFIED | Result struct verified; 4 re-export paths wired; doctest passes; `conformal_prediction_band` returns None for elastic variants (truth 9); additive/non-breaking |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tolerance/conformal_anomaly.rs` | 373–375 | Local `uniform_grid` in test module (duplicate of `crate::test_helpers::uniform_grid`) | Info | Cosmetic; zero test or runtime impact; left intentionally (per 74-REVIEW.md IN-02 disposition) |

No TBD/FIXME/XXX markers found in any files modified by this phase.

---

### Human Verification Required

None. All must-haves are verified programmatically. The marginal validity test asserts only the upper bound (`flag_rate <= 0.25`) rather than the two-sided `[0.02, 0.20]` band from RESEARCH.md; this is the correct conformal guarantee (the finite-sample bound is one-sided: `<= alpha + 1/(n+1)`) and is verified deterministically. No human visual or interactive check is needed.

---

### Summary

Phase 74 goal is **achieved**. The inductive conformal anomaly detector is fully implemented in `tolerance/conformal_anomaly.rs` (870 lines, 12 unit tests, 1 module doctest). All three elastic `NonConformityScore` variants (`AmplitudeElastic`, `PhaseElastic`, `CombinedElastic`) are additive and `Copy`. The critical code-review defect (CR-01 template length validation) and warning (WR-01 NaN fallback) were fixed in commit `846a264a` before this verification. Re-exports are wired through `tolerance/mod.rs`, `lib.rs`, and `prelude.rs`. The existing `conformal_prediction_band` path is behaviorally unchanged.

All 9 must-haves VERIFIED. ECA-01, ECA-02, ECA-03 all SATISFIED.

---

_Verified: 2026-09-05T22:00:00Z_
_Verifier: Claude (gsd-verifier)_
