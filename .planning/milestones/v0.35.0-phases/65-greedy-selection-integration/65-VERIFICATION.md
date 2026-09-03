---
phase: 65-greedy-selection-integration
verified: 2026-09-03T00:00:00Z
status: passed
score: 6/6 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: null
---

# Phase 65: Greedy Selection & Integration — Verification Report

**Phase Goal:** `optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>` — deterministic greedy sequential forward selection delegating to Phase 64's `design_criterion`; adds `OptDesConfig`/`OptDesResult`; full additive crate-root + prelude re-exports; module doctest; criterion benchmark. No new math. Two-stage read-only (no re-estimation).
**Verified:** 2026-09-03
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `optimal_design` is public, `#[must_use=...]`, exact signature, returns `Result<OptDesResult, FdarError>`; greedy forward selection delegates to `design_criterion`; model consumed read-only | ✓ VERIFIED | `optimal_design.rs:305-397` — `#[must_use = "expensive computation whose result should not be discarded"]`, exact signature `pub fn optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>`. Greedy loop calls `design_criterion(model, &trial, config.criterion.clone())?` at line 363. No mutation of `model`. |
| 2 | Determinism: parallel evaluate + SEQUENTIAL fold-based argmin (smallest-index tie-break); `min_by` absent from logic; two-call byte-identical; seq==parallel | ✓ VERIFIED | `optimal_design.rs:359-388` — `iter_maybe_parallel!(remaining)` collects scores then `.into_iter().fold(None, ...)` with strict `<` (no rayon `min_by` in code — only in comments). `test_determinism_two_calls` passes under both `--features linalg,parallel` and `--features linalg` (28/28 both runs). |
| 3 | `OptDesConfig` has single `criterion: DesignCriterion` field (no separate `OptimalityKind`), `Default`, NOT `#[non_exhaustive]`. `OptDesResult` is `#[non_exhaustive]` with `selected_indices`/`selected_argvals`/`criterion_trace`. | ✓ VERIFIED | `optimal_design.rs:204-244` — `OptDesConfig` struct at line 206: fields `candidate_grid`, `budget`, `criterion: DesignCriterion`; no `#[non_exhaustive]`; `Default` impl at line 217. `OptDesResult` at line 234 with `#[non_exhaustive]` at line 232. |
| 4 | Candidate→argvals index mapping with FP tolerance (`1e-9`); 5 validation guards (budget==0, budget>grid, off-grid, ncomp==0, sigma2<=0) → `InvalidParameter` | ✓ VERIFIED | `optimal_design.rs:256-274` — `map_candidates_to_indices` uses `(t - cand).abs() < 1e-9`. Lines 311-338 implement all 5 fast-fail guards. Tests `test_validation_budget_zero`, `test_validation_budget_exceeds_grid`, `test_validation_off_grid_candidate`, `test_validation_ncomp_zero`, `test_validation_sigma2_nonpositive` all pass. |
| 5 | 13 new module tests present and passing (incl. determinism two-call, duplicate-free, monotone trace, trajectory known-answer computed in-test, config default, prelude reexport) | ✓ VERIFIED | `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` → **28 passed, 0 failed**. All 13 named Phase 65 tests (lines 879-1088) pass. `test_trajectory_selects_informative_point` computes expected argmin in-test via sequential scan, not hardcoded (lines 1022-1036). |
| 6 | Additive full re-export (crate root + prelude), module doctest, registered criterion benchmark. No existing public signature changed. | ✓ VERIFIED | `lib.rs:593-595` — six-symbol `pub use optimal_design::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptDesResult, OptimalityKind}`. `prelude.rs:54-56` — identical six-symbol block. Doctest at `optimal_design.rs:29-73` runs fit-PACE→optimal_design→assert; `cargo test --doc` passes 1/1. `Cargo.toml:151` — `[[bench]] name = "optimal_design" harness = false`. Bench compiles clean. |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/optimal_design.rs` | Extended with `OptDesConfig`, `OptDesResult`, `optimal_design`, helper, 13 tests | ✓ VERIFIED | 1089 lines; all symbols present and substantive. Phase 64 code (15 tests) intact. |
| `fdars-core/src/lib.rs` | Extended `pub use optimal_design::{...}` to full six-symbol surface | ✓ VERIFIED | Lines 593-595; six symbols exported additively. |
| `fdars-core/src/prelude.rs` | FOptDes re-export block added | ✓ VERIFIED | Lines 54-56; six-symbol block present. |
| `fdars-core/benches/optimal_design.rs` | New criterion 0.5 benchmark | ✓ VERIFIED | 103-line file; four bench functions in one group; builds via real `pace_fpca` fit path. |
| `fdars-core/Cargo.toml` | `[[bench]] name = "optimal_design"` stanza | ✓ VERIFIED | Line 151-152; `harness = false`. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `optimal_design` greedy loop | Phase 64 `design_criterion` | `design_criterion(model, &trial, config.criterion.clone())?` at line 363 | ✓ WIRED | Each candidate at each greedy step evaluated through the Phase 64 function. |
| Candidate grid values | `model.argvals` indices | `map_candidates_to_indices` with `1e-9` FP tolerance (line 265) | ✓ WIRED | Private helper, one-time mapping, preserves candidate_grid order. |
| Parallel evaluation | Sequential argmin | `iter_maybe_parallel!(remaining)` collects to `Vec<(usize,f64)>` then `.into_iter().fold(...)` (lines 359-384) | ✓ WIRED | Strict `<` fold; no rayon `min_by` in logic. |
| Doctest | Crate root surface | `use fdars_core::{optimal_design, DesignCriterion, OptDesConfig}` in doctest | ✓ WIRED | Validated by `cargo test --doc` passing 1/1. |
| Benchmark | Crate root surface | `use fdars_core::{design_criterion, optimal_design, ...}` in `benches/optimal_design.rs:15` | ✓ WIRED | Bench compiles via `cargo build --benches --features linalg,parallel`. |

---

### Data-Flow Trace (Level 4)

Not applicable — this is a pure algorithmic library module. All outputs (`selected_indices`, `selected_argvals`, `criterion_trace`) flow directly from the greedy computation over the supplied model; no external data sources or rendering layer.

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 28 module tests pass (both feature sets) | `cargo test -p fdars-core --lib optimal_design --features linalg,parallel` | 28 passed, 0 failed | ✓ PASS |
| seq==parallel determinism | `cargo test -p fdars-core --lib optimal_design --features linalg` | 28 passed, 0 failed | ✓ PASS |
| Bench compiles | `cargo build --benches -p fdars-core --features linalg,parallel` | Finished, no errors | ✓ PASS |
| Module doctest passes | `cargo test -p fdars-core --doc --features linalg,parallel optimal_design` | 1 passed, 0 failed | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FOD-04 | 65-01 | Greedy sequential forward selection — deterministic, duplicate-free, monotone non-increasing trace, seq==parallel | ✓ SATISFIED | `optimal_design` function implements the full greedy loop; 9 of 13 new tests directly target FOD-04 properties; all pass. |
| FOD-05 | 65-01, 65-02 | Two-stage `&PaceFpcaResult` entry point with no re-estimation; `OptDesConfig`/`OptDesResult` types; full additive re-exports; doctest; criterion benchmark | ✓ SATISFIED | Entry point implemented read-only; types follow `PaceFpcaConfig`/`PaceFpcaResult` precedent; 6-symbol re-export in both `lib.rs` and `prelude.rs`; module doctest passes; bench registered and compiles. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None found | — | No `TBD`, `FIXME`, `XXX`, placeholder returns, or stub patterns in `optimal_design.rs`, `benches/optimal_design.rs`, or the modified `lib.rs`/`prelude.rs` lines. | — | — |

---

### Human Verification Required

None. All Phase 65 behaviors have automated verification. The module has no UI, no external service integration, and no visual output requiring human judgment.

---

### Gaps Summary

No gaps found. All 6 must-haves are verified, all 5 artifacts are substantive and wired, all key links confirmed in code, and all behavioral spot-checks pass.

**Pre-existing serde defect (NOT a gap):** `cargo build --features serde` fails due to a pre-existing issue in `shapelet/classifier.rs` (Phase 60). The FOptDes types in `optimal_design.rs` are serde-clean (correct `#[cfg_attr(feature = "serde", derive(...))]` gates). This defect predates Phase 65 and is not a Phase 65 regression.

---

_Verified: 2026-09-03_
_Verifier: Claude (gsd-verifier)_
