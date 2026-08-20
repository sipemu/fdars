---
phase: 29-outlier-detector-suite
verified: 2026-08-19T22:15:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 29: Outlier-Detector Suite Verification Report

**Phase Goal:** A user can flag magnitude and shape outliers with the four fdaoutlier/roahd detectors fdars was missing — tvdmss, muod, sequential_transform_outliers, depthgram — as numeric outputs (no rendering), reusing DEPTH-01 depths + the existing MS-plot/outliergram machinery, without any existing outlier code changing.

**Verified:** 2026-08-19T22:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | New Result-returning public fns in outliers.rs (crate-root re-exported): tvdmss, muod, sequential_transform_outliers, depthgram — numeric outputs, no plotting | ✓ VERIFIED | All four fns present at outliers.rs lines 518, 668, 832, 931 with `Result<T, FdarError>` signatures; all four re-exported in lib.rs lines 435–439 |
| 2 | On synthetic data with injected magnitude + shape outliers, tvdmss flags both classes and the other detectors return expected outlier index sets (inline tests) | ✓ VERIFIED | Tests `tvdmss_flags_magnitude_outlier`, `tvdmss_flags_shape_outlier`, `muod_flags_magnitude_amplitude_shape`, `seq_transform_default_sequence_flags_outlier_and_union_is_flatten`, `depthgram_flags_magnitude_and_shape` all pass — 45 passed, 0 failed |
| 3 | tvdmss computes from DEPTH-01's total_variation_depth_1d + MSSI; suite reuses functional_boxplot / outliergram / MS-plot machinery; no new crate dependency | ✓ VERIFIED | outliers.rs line 528: `let depth = total_variation_depth_1d(data, data)?;`; lines 549, 850, 963 all call `functional_boxplot`; `git diff` against the pre-phase plan commit shows no change to fdars-core/Cargo.toml |
| 4 | Invalid inputs (empty / single-curve / mismatched dims / degenerate columns) return FdarError, never panic, at each entry point | ✓ VERIFIED | `tvdmss_rejects_empty_and_too_few`, `muod_rejects_bad_dims`, `seq_transform_error_paths` (D1 single-col → InvalidDimension; T2 zero-norm → ComputationFailed; n==1 → InvalidDimension), `depthgram_rejects_bad_dims` — all pass in the 45-test run |
| 5 | Existing detectors (magnitude_shape_outlyingness, outliergram) and DEPTH-01 depth fns keep public signatures unchanged; full suite + clippy green | ✓ VERIFIED | `outliergram` signature at line 280 and `magnitude_shape_outlyingness` at line 354 are unchanged from pre-phase; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits clean; 45 outlier tests pass in 0.14s |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/outliers.rs` | Four new public detector fns + result structs + inline tests | ✓ VERIFIED | 1781 lines; tvdmss (l. 518), muod (l. 668), sequential_transform_outliers (l. 832), depthgram (l. 931); 12 new inline tests for Phase 29 detectors (l. 1565–1779) |
| `fdars-core/src/lib.rs` (re-export block ~line 434) | All four fns + config/result types re-exported at crate root | ✓ VERIFIED | Lines 434–440: tvdmss, muod, sequential_transform_outliers, depthgram, TvdMssConfig, TvdMssOutliers, MuodConfig, MuodResult, SeqTransform, SeqTransformConfig, SeqTransformOutliers, DepthgramConfig, DepthgramResult all present |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `tvdmss` | `total_variation_depth_1d` | direct call (outliers.rs:528) | ✓ WIRED | `let depth = total_variation_depth_1d(data, data)?;` — no reimplementation |
| `tvdmss` stage 2 | `functional_boxplot` | call (outliers.rs:549) | ✓ WIRED | `functional_boxplot(&reduced, DepthMethod::ModifiedBand, config.emp_factor_tvd)?` |
| `sequential_transform_outliers` | `functional_boxplot` | call per step (outliers.rs:850) | ✓ WIRED | `functional_boxplot(&current, config.depth_method, config.emp_factor)?` |
| `depthgram` shape outliers | `iqr_fence` | private helper (outliers.rs:958) | ✓ WIRED | `let (_, upper) = iqr_fence(&dist, config.outliergram_factor);` |
| `depthgram` magnitude | `functional_boxplot` | call (outliers.rs:963) | ✓ WIRED | `functional_boxplot(&mbd_mat2, DepthMethod::ModifiedBand, config.boxplot_factor)?` |
| `depthgram` indices | `modified_band_1d` / `modified_epigraph_index_1d` | direct calls (outliers.rs:941–948) | ✓ WIRED | reuses existing depth machinery |
| `lib.rs` | `outliers` module | `pub use outliers::{...}` | ✓ WIRED | All 13 new public symbols present in re-export block |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 45 outlier tests pass | `cargo test -p fdars-core --features linalg,parallel --lib outliers::` | 45 passed, 0 failed, finished in 0.14s | ✓ PASS |
| Clippy clean across all targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished dev profile, no warnings | ✓ PASS |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | No debt markers (TBD/FIXME/XXX), no stub returns, no unresolved TODOs |

---

### Requirements Coverage

| Requirement | Plans | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| OUT-01 | 29-01, 29-02 | Four missing fdaoutlier/roahd detectors as numeric Result-returning fns | ✓ SATISFIED | All four detectors implemented, tested, re-exported; no new dependency; existing signatures preserved |

---

### Documented Divergences (Not Gaps)

The following divergences from the R baselines are documented in rustdoc and the SUMMARY frontmatter. They are intentional design decisions, not gaps:

1. **tvdmss stage-2 central region**: `fdaoutlier` scales the stage-2 central region by `n_orig/n_reduced`; fdars' `functional_boxplot` fixes it at the deepest 50%. `central_region_tvd` field is informational only — documented in the `TvdMssConfig` rustdoc.
2. **muod is Fast-MUOD variant**: regression vs pointwise mean (not R's pairwise C++ block); boxplot cutoff only (tangent deferred to backlog) — documented in `muod` rustdoc.
3. **depthgram univariate only (p=1)**: the three representations (`_d/_t/_t2`) are identical and cloned — documented in `DepthgramResult` rustdoc.
4. **SeqTransformConfig lacks serde**: holds a `DepthMethod` which does not derive serde — documented in `SeqTransformConfig` rustdoc.

---

### Human Verification Required

None. All success criteria are verifiable programmatically and the test suite confirms correct behavior on synthetic fixtures.

---

### Gaps Summary

No gaps. All five success criteria are fully verified by code inspection and passing tests.

---

_Verified: 2026-08-19T22:15:00Z_
_Verifier: Claude (gsd-verifier)_
