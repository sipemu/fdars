---
phase: 64-criterion-machinery-core
verified: 2026-09-02T23:15:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 64: Criterion Machinery Core — Verification Report

**Phase Goal:** A public `#[must_use] design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>` in new `src/optimal_design.rs` computing either the integrated trajectory-reconstruction BLUP-MSE (FOD-01) or the A-/D-optimal posterior score-covariance summary (FOD-02), dispatched via `DesignCriterion`/`OptimalityKind` enums (FOD-03), sharing a private `build_sigma_design`. NO greedy loop (Phase 65).

**Verified:** 2026-09-02T23:15:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `design_criterion` is public, `#[must_use]`, has the exact signature, returns `Result<f64, FdarError>` | ✓ VERIFIED | Line 77–82 of `optimal_design.rs`: `#[must_use = "expensive computation whose result should not be discarded"] pub fn design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>` |
| 2 | `DesignCriterion::{Trajectory, Score(OptimalityKind)}` and `OptimalityKind::{A, D}` enums exist and dispatch correctly | ✓ VERIFIED | Lines 36–55 of `optimal_design.rs` define both enums with `Debug, Clone, PartialEq` and serde-gated derives; dispatch at line 117–120 routes `Trajectory` → `trajectory_criterion`, `Score(kind)` → `score_criterion`; `test_enum_dispatch` passes |
| 3 | `build_sigma_design` assembles p×p Σ_d (p = `selected.len()`) with σ²I diagonal, ridge-retry, never panics | ✓ VERIFIED | Lines 128–147: row-major double loop over `selected`; `sigma_d[row * p + row] += model.sigma2` after inner col loop (not per-k); `factor_sigma_design_with_retry` (lines 153–166) retries once with 1e-8 ridge, returns `ComputationFailed` rather than panicking; `test_ridge_retry` passes with `sigma2=1e-12` |
| 4 | Trajectory branch uses `helpers::simpsons_weights(&model.argvals)` — NOT uniform weights | ✓ VERIFIED | `use crate::helpers::simpsons_weights;` at line 25; called at line 188 inside `trajectory_criterion`; `test_trajectory_grid_invariance` proves grid-invariance (m=21/51/101 agree to 1e-10), which is impossible with uniform weights |
| 5 | Known-answer gates present and passing: MSE(∅)=Σλ_k grid-invariant; A(∅)=Σλ_k; D(∅)=Σ log λ_k; Cov(ξ|∅)=diag(λ); monotone for all three; validation guards; ridge-retry | ✓ VERIFIED | All 14 optimal_design tests pass: `test result: ok. 14 passed; 0 failed` (confirmed by running `cargo test -p fdars-core --features linalg optimal_design`) — covers every named gate |
| 6 | `design_criterion` validates: out-of-range index / sigma2<=0 / ncomp==0 → `FdarError::InvalidParameter` | ✓ VERIFIED | Lines 85–114 of `optimal_design.rs`; also additional guard: `eigenvalues.len() < ncomp` → `InvalidParameter`; `test_validation_index_range`, `test_validation_sigma2`, `test_validation_ncomp` all pass |
| 7 | Monotone non-increasing for all three criteria: `criterion(S∪{t}) ≤ criterion(S) + 1e-12` | ✓ VERIFIED | `test_monotonicity_trajectory`, `test_monotonicity_a_opt`, `test_monotonicity_d_opt` all assert slack ≤ 1e-12 and pass |
| 8 | `lib.rs` additively declares `pub mod optimal_design;` and re-exports `design_criterion, DesignCriterion, OptimalityKind` only | ✓ VERIFIED | Line 109: `pub mod optimal_design;` (peer of `pub mod kshape;`); line 593: `pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};`; no greedy fn, no `OptDesConfig`, no `OptDesResult`, no prelude entry added |
| 9 | No Phase-65 items leaked: no greedy loop, no `OptDesConfig`/`OptDesResult`, no prelude change, no benchmark | ✓ VERIFIED | `grep` across `optimal_design.rs` and `lib.rs` returns no matches for `greedy`, `OptDesConfig`, `OptDesResult`; `prelude.rs` contains no `optimal_design` references |

**Score:** 9/9 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/optimal_design.rs` | New file with enums, `design_criterion`, `build_sigma_design`, trajectory + score branches | ✓ VERIFIED | 525-line file exists; all required items present and substantive |
| `fdars-core/src/lib.rs` | Additive `pub mod optimal_design;` + `pub use` of 3 items | ✓ VERIFIED | Lines 109 and 593; additive only, no existing lines removed |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `design_criterion` | `trajectory_criterion` | `match criterion { Trajectory => trajectory_criterion(...) }` at line 118 | ✓ WIRED | Live dispatch confirmed in source |
| `design_criterion` | `score_criterion` | `match criterion { Score(kind) => score_criterion(..., kind) }` at line 119 | ✓ WIRED | Live dispatch confirmed in source |
| `trajectory_criterion` | `helpers::simpsons_weights` | `let weights = simpsons_weights(&model.argvals)` at line 188 | ✓ WIRED | Import at line 25; called in non-empty and empty-set paths |
| `trajectory_criterion` / `score_criterion` | `build_sigma_design` + `factor_sigma_design_with_retry` | called for non-empty selected in both branches | ✓ WIRED | Lines 203, 263 |
| `score_criterion` | `linalg::log_det_from_cholesky` | D-opt path, line 312 | ✓ WIRED | Import at line 26 |
| `lib.rs` | `optimal_design::{design_criterion, DesignCriterion, OptimalityKind}` | `pub use` at line 593 | ✓ WIRED | Re-export present, makes symbols reachable at crate root |

---

### Data-Flow Trace (Level 4)

Not applicable. This is a pure numerical computation module — data flows from `PaceFpcaResult` fields (eigenvalues, eigenfunctions, argvals, sigma2) through the criterion math and into a returned `f64`. No rendering, no display layer, no static fallback. All three criterion variants route to real computation paths; no stub `Ok(0.0)` placeholder remains (the 64-01 placeholder was replaced in plan 64-02, confirmed by reading the full `score_criterion` implementation at lines 234–315).

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 14 optimal_design tests pass | `cargo test -p fdars-core --features linalg optimal_design` | `test result: ok. 14 passed; 0 failed` | ✓ PASS |
| All named tests present | enumerated from test output | All 14 test names match the VALIDATION.md contract exactly | ✓ PASS |
| `simpsons_weights` actually called (not commented) | `grep -n "simpsons_weights" optimal_design.rs` | Lines 25, 188, 335 (import + two call sites) | ✓ PASS |
| `#[must_use]` on `design_criterion` | `grep -n "#\[must_use" optimal_design.rs` | Line 77: `#[must_use = "expensive computation whose result should not be discarded"]` | ✓ PASS |
| lib.rs has ≥ 2 `optimal_design` references | `grep -c 'optimal_design' lib.rs` | Lines 109 (module decl) and 593 (re-export) | ✓ PASS |

---

### Probe Execution

No probes declared for this phase (pure Rust unit test harness; no `scripts/*/tests/probe-*.sh`).

---

### Requirements Coverage

| Requirement | Plans | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| FOD-01 | 64-01 | Integrated trajectory-reconstruction BLUP-MSE criterion | ✓ SATISFIED | `trajectory_criterion` with Simpson weights; known-answer + grid-invariance + monotonicity tests pass |
| FOD-02 | 64-02 | FPC-score A-/D-optimal posterior covariance criterion | ✓ SATISFIED | `score_criterion` implementing `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ`; A(∅)=Σλ_k and D(∅)=Σ log λ_k confirmed by test |
| FOD-03 | 64-01, 64-02 | Public enum dispatch, validation, lib.rs re-export, serde-gated derives | ✓ SATISFIED | `DesignCriterion`/`OptimalityKind` enums with serde gating; `design_criterion` public and re-exported; validation guards confirmed; serde failure is pre-existing `ClassifFit` defect (Phase 60, commit ea39c623) — not introduced by Phase 64 |

---

### Anti-Patterns Found

No blocking anti-patterns. Scan of `fdars-core/src/optimal_design.rs`:

- No `TBD`, `FIXME`, `XXX` markers found.
- No `TODO` or `HACK` markers found.
- No `placeholder`, `coming soon`, or `not yet implemented` strings found.
- No empty implementations: the Score branch placeholder (`Ok(0.0)`) documented in 64-01-SUMMARY.md was replaced in 64-02 with the real posterior-covariance math (confirmed by reading lines 234–315 of `optimal_design.rs`).
- `return null` / `return {}` / `return []` patterns not applicable (Rust).

One pre-existing tech debt item noted (not introduced by Phase 64): `cargo build -p fdars-core --features serde` fails due to `ShapeletTransformClassifier` embedding non-serde `ClassifFit` (Phase 60, commit ea39c623). Phase 64's new types (`DesignCriterion`, `OptimalityKind`) have correct serde-gated derives and are not the source of the failure. This is backlog material for a future phase, not a Phase 64 gap.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `shapelet/*.rs` (pre-existing) | — | `ClassifFit` missing serde derives | ℹ️ Info (pre-existing, Phase 60) | `--features serde` build broken crate-wide; not introduced by Phase 64 |

---

### Divergences from Plan (Both Accepted)

**Divergence 1 — `test_enum_dispatch` uses algebraic identity instead of "three mutually distinct" check.** The plan's original behavior spec said the three variants must yield "three mutually distinct" values. This is mathematically false for the synthetic orthonormal model: when eigenfunctions are orthonormal w.r.t. Simpson weights, `∫Var[x̂(t)]dt = trace(Cov(ξ))`, so Trajectory ≡ A-optimality is an exact algebraic identity. The test correctly asserts: (a) all three are `Ok` and finite, (b) `|traj − a| < 1e-9` (proving Trajectory runs the real integral, not a stub), and (c) `|d − a| > 1e-9` and `d < a` (proving D routes to its own distinct code path). This is a stronger contract, not a weakened one — ACCEPTED.

**Divergence 2 — Additional defensive guard `eigenvalues.len() < ncomp`.** Added beyond the plan spec. Returns `FdarError::InvalidParameter` cleanly when a model's eigenvalues vector is shorter than its declared `ncomp`, preventing an index panic. Purely defensive and additive — ACCEPTED.

---

### Human Verification Required

None. All phase behaviors have automated known-answer verification. The VALIDATION.md explicitly states: "All phase behaviors have automated verification." The behavioral spot-checks confirm this.

---

## Gaps Summary

No gaps. All 9 must-have truths are VERIFIED by direct source inspection and live test execution. The two plan divergences are mathematically correct improvements, not defects.

---

_Verified: 2026-09-02T23:15:00Z_
_Verifier: Claude (gsd-verifier)_
