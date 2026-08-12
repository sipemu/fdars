---
phase: 14-shift-registration
verified: 2026-08-12T14:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 14: Shift Registration — Verification Report

**Phase Goal:** A user can register a set of curves by simple rigid horizontal shift to the sample mean, and quantify how well any registration worked with the three standard scikit-fda diagnostics — closing the "simplest registration method" gap.
**Verified:** 2026-08-12T14:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 (SC1) | `least_squares_shift_registration(data, argvals, max_shift)` returns registered curves + per-curve δᵢ via golden-section L2-to-mean | ✓ VERIFIED | `shift.rs:178-252` — full implementation; golden-section in `golden_section_search` (100 iter, tol=1e-6); L2 objective in `l2_shift_objective` (Simpson-weighted); `ShiftRegistrationResult { registered_data, shifts }` returned |
| 2 (SC2) | Already-aligned set → δᵢ ≈ 0; injected offsets recovered within tolerance | ✓ VERIFIED | `test_shift_already_aligned` (FEAT-06-A, 12 passed) and `test_shift_recovers_injected_offset` (FEAT-06-B, tol 0.05) both green in live test run |
| 3 (SC3) | Three `Result`-returning score functions exist in `quality.rs` | ✓ VERIFIED | `least_squares_score` (line 279), `sobolev_least_squares_score` (line 357), `pairwise_correlation_score` (line 484) — all `Result<f64, FdarError>`, verified in quality.rs source |
| 4 (SC4) | Scores move in expected direction after registration (LS drops, correlation rises) | ✓ VERIFIED | `test_ls_score_drops_after_registration` (FEAT-07-B) and `test_pairwise_corr_rises_after_registration` (FEAT-07-C) both green in live test run |
| 5 (SC5) | All new public fns return `Result`, are re-exported at crate root, carry inline tests, no API breakage | ✓ VERIFIED | All 5 items in `lib.rs` lines 152-169; `alignment/mod.rs` lines 88-102; 12 inline tests green; clippy `--all-targets -D warnings` clean |

**Score:** 5/5 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/alignment/shift.rs` | `least_squares_shift_registration` + `ShiftRegistrationResult` | ✓ VERIFIED | 414 lines; fully substantive; re-exported via mod.rs and lib.rs |
| `fdars-core/src/alignment/quality.rs` | Three FEAT-07 score functions appended | ✓ VERIFIED | Lines 233-734; all three functions present, documented, and tested |
| `fdars-core/src/alignment/mod.rs` | `mod shift;` + `pub use shift::{...}` + extended `pub use quality::{...}` | ✓ VERIFIED | Lines 37 (`mod shift`), 88-90 (quality scores), 100-102 (shift items) |
| `fdars-core/src/lib.rs` | Five new items in flat alignment `pub use` block | ✓ VERIFIED | Lines 152, 153, 156, 169 — all five items present alphabetically |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `shift.rs::least_squares_shift_registration` | `alignment/mod.rs` | `pub use shift::{...}` line 100-102 | ✓ WIRED | Direct `pub use` confirmed in source |
| `quality.rs::{least_squares_score, sobolev_least_squares_score, pairwise_correlation_score}` | `alignment/mod.rs` | `pub use quality::{...}` lines 87-90 | ✓ WIRED | Direct `pub use` confirmed in source |
| `alignment::` (all 5 items) | `lib.rs` crate root | flat `pub use alignment::{...}` block | ✓ WIRED | grep confirmed all 5 names at lib.rs lines 152-169 |
| `quality.rs::pairwise_correlation_score` | `shift.rs::least_squares_shift_registration` | direction test in `test_pairwise_corr_rises_after_registration` calls `crate::alignment::shift::least_squares_shift_registration` | ✓ WIRED | Cross-module call wired and test passes |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `least_squares_shift_registration` | `mean` for objective | `crate::fdata::mean_1d(data)` — cross-sectional mean of input rows | Yes | ✓ FLOWING |
| `least_squares_shift_registration` | `weights` for integral | `helpers::simpsons_weights(argvals)` | Yes | ✓ FLOWING |
| `least_squares_score` | `mean` | `crate::fdata::mean_1d(registered)` | Yes | ✓ FLOWING |
| `pairwise_correlation_score` | `centred` curves | per-curve Simpson-weighted mean subtracted inline (lines 522-531) | Yes | ✓ FLOWING |
| `sobolev_least_squares_score` | derivative term | `gradient_uniform` on each centred row (line 431) | Yes | ✓ FLOWING |

No hollow props, no static returns, no disconnected data paths.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 12 named tests green (FEAT-06-A…E + FEAT-07-A…F + WR-02 guard) | `cargo test -p fdars-core --features linalg -- alignment::shift alignment::quality` | 12 passed, 0 failed (lib) + 1 doctest passed | ✓ PASS |
| Doctest for `least_squares_shift_registration` | included in above run | ok | ✓ PASS |
| Clippy CI gate (`--all-targets -D warnings`) | `cargo clippy --all-targets -p fdars-core --features linalg -- -D warnings` | Finished — 0 errors, 0 warnings | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FEAT-06 | 14-01-PLAN.md | Rigid-shift registration via golden-section L2-to-mean | ✓ SATISFIED | `shift.rs` fully implements; 5 named tests green; crate-root export confirmed |
| FEAT-07 | 14-02-PLAN.md | Three registration-quality scores in `quality.rs` | ✓ SATISFIED | `quality.rs` appended with `least_squares_score`, `sobolev_least_squares_score`, `pairwise_correlation_score`; 7 named tests green; crate-root exports confirmed |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | No debt markers (TBD/FIXME/XXX), no stubs, no empty implementations found | — | — |

Post-review state: CR-01 (unused test code → clippy hard error) fixed in commit `36c40c1a`. WR-01 (cosine vs Pearson mislabelling + implementation) fixed in `97ed3b2a`. WR-02 (missing m<2 guard on score functions) fixed in `c4ee0926`. WR-03 (non-uniform grid silently wrong in Sobolev) fixed in `fc21d3ae`. All four findings resolved before verification.

---

### CONTEXT Compliance

**Standalone-energy form (not silently reverted to ratio-based):** Confirmed at `quality.rs` lines 242-247 — comment block explicitly states scores "do NOT divide by the spread of the unregistered data" and "differs from scikit-fda's ratio-based scorers". Each function's rustdoc repeats this in the "Standalone-energy form" section.

**True Pearson centering (WR-01 fix in place):** Confirmed at `quality.rs` lines 519-531 — `centred` curves are computed by subtracting the Simpson-weighted functional mean `μᵢ = (Σⱼ fᵢ(tⱼ)·wⱼ) / (Σⱼ wⱼ)` before inner products. Comment: "centred curve (true Pearson, not cosine similarity)".

---

### Human Verification Required

None. All phase behaviors have automated verification; all 12 named tests are green and the clippy gate is clean.

---

### Gaps Summary

No gaps. All 5 success criteria verified against live codebase and confirmed by test execution. Phase goal achieved.

---

_Verified: 2026-08-12T14:00:00Z_
_Verifier: Claude (gsd-verifier)_
