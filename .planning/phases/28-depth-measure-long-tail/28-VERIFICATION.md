---
phase: 28-depth-measure-long-tail
verified: 2026-08-19T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 28: Depth-Measure Long Tail Verification Report

**Phase Goal:** A user can compute every canonical batch univariate functional depth measure roahd/fdaoutlier expose but fdars was missing — HRD & MHRD, HI/MHI/EI, extremal, ERL, L∞, and TVD+MSSI — over a column-major FdMatrix, selectable through the existing DepthMethod dispatcher, without any existing depth code changing.

**Verified:** 2026-08-19
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Nine new Result-returning public fns exist in `fdars-core/src/depth/`, each returning `Vec<f64>` per curve, crate-root re-exported | ✓ VERIFIED | All 9 signatures confirmed in source; all 9 appear in `lib.rs` `pub use depth::{...}` block (line 422–431) |
| 2 | Each measure registered as a `DepthMethod` variant; dispatcher routes to it; pre-existing 4 variants unchanged | ✓ VERIFIED | 9 new variants enumerated + 9 dispatch match arms confirmed in `dispatch.rs`; `band.rs` / `fraiman_muniz.rs` show zero git diff across all phase-28 commits |
| 3 | Inline `#[cfg(test)]` tests assert central-deepest / outlier-among-most-extreme ordering with known synthetic data | ✓ VERIFIED | 59 `#[test]` items across the 7 new/modified files; per-measure ordering assertions confirmed; `cargo test --lib depth::` → 135 passed, 0 failed |
| 4 | Invalid inputs return `FdarError`, not panic | ✓ VERIFIED | Every new fn opens with dimension guards before any computation; `min_n_guards_return_err_without_panic` test covers all 8 guards in dispatcher; empty-matrix error paths tested in each module |
| 5 | No new crate dependency; existing depth signatures untouched; full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green | ✓ VERIFIED | `git diff HEAD~6..HEAD -- fdars-core/Cargo.toml` produces empty output; clippy exits clean; `cargo test --lib depth::` → 135 passed, 0 failed |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/depth/hypo_epi.rs` | `hypograph_index_1d`, `epigraph_index_1d`, `modified_hypograph_index_1d` → `Result<Vec<f64>, FdarError>` | ✓ VERIFIED | File exists, all 3 fns present with correct Result-returning signatures; 14 tests |
| `fdars-core/src/depth/half_region.rs` | `half_region_depth_1d`, `modified_half_region_depth_1d` → `Result<Vec<f64>, FdarError>` | ✓ VERIFIED | File exists, both fns present; 8 tests including `hrd_ranks_central_deepest_and_extremes_shallow` |
| `fdars-core/src/depth/extremal.rs` | `extremal_depth_1d` → `Result<Vec<f64>, FdarError>` | ✓ VERIFIED | File exists, fn present; 5 tests including `central_deepest_and_outlier_among_shallowest` |
| `fdars-core/src/depth/erl.rs` | `extreme_rank_length_depth_1d` → `Result<Vec<f64>, FdarError>` | ✓ VERIFIED | File exists, fn present; 4 tests including `central_deepest_and_outlier_among_shallowest` |
| `fdars-core/src/depth/linf.rs` | `linfinity_depth_1d` → `Result<Vec<f64>, FdarError>` | ✓ VERIFIED | File exists, fn present; 4 tests including `closest_to_sample_is_deepest_and_outlier_shallow` |
| `fdars-core/src/depth/tvd.rs` | `total_variation_depth_1d` → `Result<TvdMssResult, FdarError>` + `TvdMssResult { tvd, mss }` | ✓ VERIFIED | File exists, fn present returning `TvdMssResult`; dispatcher extracts `.tvd`; 6 tests |
| `fdars-core/src/depth/dispatch.rs` | 9 new `DepthMethod` variants + dispatch arms; original 4 variants unchanged | ✓ VERIFIED | All 9 variants enumerated in enum body; all 9 match arms present; `FraimanMuniz`, `Band`, `ModifiedBand`, `RandomProjection` untouched |
| `fdars-core/src/lib.rs` (lines 422–431) | All 9 new fns + `TvdMssResult` crate-root re-exported | ✓ VERIFIED | `pub use depth::{..., epigraph_index_1d, extremal_depth_1d, extreme_rank_length_depth_1d, half_region_depth_1d, hypograph_index_1d, linfinity_depth_1d, modified_half_region_depth_1d, modified_hypograph_index_1d, ..., total_variation_depth_1d, DepthMethod, ..., TvdMssResult}` confirmed |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `dispatch.rs::functional_depth` | `hypo_epi.rs` | `hypograph_index_1d(data, data)?`, `epigraph_index_1d(data, data)?`, `modified_hypograph_index_1d(data, data)?` | ✓ WIRED | All three import + call sites confirmed |
| `dispatch.rs::functional_depth` | `half_region.rs` | `half_region_depth_1d(data, data)?`, `modified_half_region_depth_1d(data, data)?` | ✓ WIRED | Both import + call sites confirmed |
| `dispatch.rs::functional_depth` | `extremal.rs` | `extremal_depth_1d(data, data)?` | ✓ WIRED | Import + call site confirmed |
| `dispatch.rs::functional_depth` | `erl.rs` | `extreme_rank_length_depth_1d(data, data)?` | ✓ WIRED | Import + call site confirmed |
| `dispatch.rs::functional_depth` | `linf.rs` | `linfinity_depth_1d(data, data)?` | ✓ WIRED | Import + call site confirmed |
| `dispatch.rs::functional_depth` | `tvd.rs` | `total_variation_depth_1d(data, data)?.tvd` | ✓ WIRED | Import + call site confirmed; `.tvd` projection correctly surfaces the TVD magnitude component |
| `depth/mod.rs` | all 6 new modules | `pub use` re-exports | ✓ WIRED | `mod.rs` declares all 6 modules and re-exports all 9 new public symbols + `TvdMssResult` |
| `lib.rs` | `depth::*` | `pub use depth::{...}` | ✓ WIRED | All 9 new fns + `DepthMethod` + `TvdMssResult` appear in crate-root `pub use` block |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 135 depth lib tests pass (including all 59 new test items) | `cargo test -p fdars-core --features linalg,parallel --lib depth::` | 135 passed, 0 failed | ✓ PASS |
| clippy with all-targets and `-D warnings` clean | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Finished with no warnings or errors | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source | Description | Status | Evidence |
|-------------|--------|-------------|--------|----------|
| DEPTH-01 | Phase goal | Add HRD, MHRD, HI, MHI, EI, extremal, ERL, L∞, TVD+MSSI depth measures | ✓ SATISFIED | All 9 measures implemented, dispatched, and tested |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tvd.rs` | 78 | `[ASSUMED]` tag on MSS shape-variation derivation | ℹ️ Info | Documents reconstruction from paper; not a debt marker (no TBD/FIXME/XXX); accepted annotation pattern in this codebase |

No blockers. No unresolved TBD / FIXME / XXX markers in any file modified by Phase 28. The `[ASSUMED]` annotation in `tvd.rs` is a documentation note on a reconstruction approach, not a code-quality debt item.

---

### Human Verification Required

None. All success criteria are verifiable through code inspection, test execution, and static analysis.

---

### Gaps Summary

No gaps. All five success criteria are fully met:

1. All 9 functions exist with correct Result-returning signatures in dedicated submodules under `fdars-core/src/depth/`, and all are crate-root re-exported.
2. All 9 `DepthMethod` variants are enumerated in the dispatcher enum with corresponding match arms; the original 4 variants (`FraimanMuniz`, `Band`, `ModifiedBand`, `RandomProjection`) and their dispatch arms are byte-identical to their pre-Phase-28 state.
3. Fifty-nine inline `#[cfg(test)]` tests cover ordering invariants (central-deepest, outlier-among-most-extreme), range properties, dispatcher round-trips, and error paths. The test binary confirms 135 depth tests pass.
4. Every new entry point validates dimensions and curve-count minimums, returning `FdarError::InvalidDimension` on invalid input; zero panics possible from input validation.
5. `Cargo.toml` is unchanged (zero diff). Clippy exits clean with `--all-targets --features linalg,parallel -D warnings`.

---

_Verified: 2026-08-19_
_Verifier: Claude (gsd-verifier)_
