---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 04
subsystem: autodiff
tags: [rust, autodiff, reverse-mode, vjp, wengert-list, finite-differences, validation]

# Dependency graph
requires:
  - phase: 94-03-vjp-hardening-agreement
    provides: "Tier-5 lifecycle/edge-case tests + Tier-3 reverse-vs-Dual agreement for every op — vjp proven correct before FD cross-check"
provides:
  - "Tier-4 FD cross-check tests: vjp through soft_dtw_distance_generic::<Var> and project_scores_generic::<Var>, both called UNCHANGED (RAD-03)"
  - "Tier-4 composed-objective vjp test: soft_dtw + λ·Σscores² FD-checked per component (RAD-03 capstone)"
  - "Running vjp doctest demonstrating 2-input gradient (value=14.0, grad=[6.0,1.0])"
  - "prelude.rs re-exports Var + vjp (already present from earlier plan, confirmed here); Tape stays opaque"
affects: [95-generic-scalar-hot-path, 99-end-to-end-flow, 100-release-prep]

actuals:
  tokens: 3219
  tasks: 3
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Tier-4 FD cross-check pattern: instantiate generic function at Var unchanged, compare each gradient component against central FD (h=1e-8 simple, h=1e-6 composed)"
    - "Composed-objective test: soft_dtw_distance_generic + lambda * project_scores_generic squared, one vjp call, FD-checked per component — mirrors forward.rs::grad_composed_objective_matches_finite_diff structure"
    - "Type disambiguation for Var in test closures: use <Var as Scalar>::zero() / from_f64() when Rust cannot infer the impl from context alone"
    - "clippy assign_op_pattern: acc = acc + x must be acc += x in all Var closures"

key-files:
  created: []
  modified:
    - fdars-core/src/autodiff/reverse.rs

key-decisions:
  - "Tier-4 FD cross-check proves RAD-03 without modifying the validation targets — soft_dtw.rs and regression.rs remain read-only integration points; the only proof needed is that ::<Var> instantiates and gradients match FD"
  - "Reference values in soft_dtw vjp closure are off-tape constants (<Var as Scalar>::from_f64(r)) — correct because the reference curve is fixed data, not a gradient input; the gradient flows only through the curve argument"
  - "Scalar::zero() / from_f64() require explicit <Var as Scalar>::... disambiguation in closures where Rust cannot infer the impl from the parameter type alone (the f: Fn(&[Var]) -> Var bound is not enough for free-standing trait calls)"
  - "vjp doctest updated to 2-input example with explicit Var type annotation, confirming the API shape and making the reverse-mode surface discoverable"

requirements-completed: [RAD-03]

coverage:
  - id: D1
    description: "vjp gradients through soft_dtw_distance_generic::<Var> match central FD (h=1e-8) within 1e-6 per component — Var: Scalar compile + correctness"
    requirement: RAD-03
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_soft_dtw_matches_finite_diff"
        status: pass
    human_judgment: false
  - id: D2
    description: "vjp gradients through project_scores_generic::<Var> (sum-of-squares objective) match central FD (h=1e-8) within 1e-6 per component"
    requirement: RAD-03
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_project_scores_matches_finite_diff"
        status: pass
    human_judgment: false
  - id: D3
    description: "Composed objective (soft_dtw + λ·Σscores²) differentiated in one vjp call matches central FD (h=1e-6) per component — capstone RAD-03 validation"
    requirement: RAD-03
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_composed_objective_matches_finite_diff"
        status: pass
    human_judgment: false
  - id: D4
    description: "Var + vjp discoverable via prelude; running vjp doctest demonstrates 2-input gradient API"
    requirement: RAD-03
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs - autodiff::reverse::vjp (line 485)"
        status: pass
    human_judgment: false

duration: 10min
completed: 2026-09-10
status: complete
---

# Phase 94 Plan 04: Validation & Prelude Summary

**Reverse-mode gradients (vjp) validated against central finite differences within 1e-6 on soft_dtw_distance_generic, project_scores_generic, and a composed objective — RAD-03 closed; 72 autodiff tests, all 3 Tier-4 FD cross-checks pass**

## Performance

- **Duration:** 10 min
- **Started:** 2026-09-10T20:52:04Z
- **Completed:** 2026-09-10T21:01:25Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Added 3 Tier-4 FD cross-check tests in `reverse.rs#[cfg(test)]`: `vjp_soft_dtw_matches_finite_diff` (m=16, h=1e-8, tol=1e-6), `vjp_project_scores_matches_finite_diff` (m=24, n=40, ncomp=3, h=1e-8, tol=1e-6), and `vjp_composed_objective_matches_finite_diff` (soft_dtw + λ·Σscores², m=24, h=1e-6, tol=1e-6, same seed 20260906 as the forward-mode reference test).
- Both validation targets (`soft_dtw_distance_generic`, `project_scores_generic`) called with ZERO call-site changes — `Var: Scalar` is a correct drop-in, proving RAD-03.
- Updated `vjp` doctest from single-input to 2-input demonstration with explicit `Var` type annotation.
- `prelude.rs` already had `Var` and `vjp` re-exported (landed in an earlier plan); confirmed `Tape` stays opaque.
- Full gate results: 72 autodiff tests, 2899 total tests, doctests (7), clippy `--all-targets --D warnings`, `--features serde` build — all green. `soft_dtw.rs` and `regression.rs` unmodified throughout.

## Task Commits

1. **Task 1: Tier-4 FD cross-check — vjp through soft_dtw + FPCA scores** - `3fef0f54` (test)
2. **Task 2: Composed-objective vjp FD cross-check** - `f2b2428f` (test)
3. **Task 3: Prelude re-exports + running vjp doctest** - `50989e38` (feat)
4. **Deviation fix: clippy assign_op_pattern** - `a8b8c897` (fix)

## Files Created/Modified

- `fdars-core/src/autodiff/reverse.rs` — Added 272 lines: Tier-4 tests (Tasks 1+2), updated vjp doctest (Task 3)

## Decisions Made

- Reference values in the `soft_dtw` vjp closure are off-tape constants (`<Var as Scalar>::from_f64(r)`) — the reference curve is fixed data, not a gradient input; gradient flows only through the `curve` argument.
- `Scalar::zero()` / `from_f64()` require explicit `<Var as Scalar>::...` disambiguation in closures where Rust cannot infer the impl from the parameter type alone.
- `prelude.rs` was already correct from Plan 01's setup — no changes needed for Task 3; only the doctest was updated.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] clippy assign_op_pattern violations in Tier-4 test closures**
- **Found during:** clippy gate after Task 3
- **Issue:** `acc = acc + *s * *s` pattern in two test closures; clippy `-D warnings` treats this as an error
- **Fix:** Replaced both occurrences with `acc += *s * *s`
- **Files modified:** `fdars-core/src/autodiff/reverse.rs`
- **Committed in:** `a8b8c897`

---

**Total deviations:** 1 auto-fixed (Rule 1 - clippy assign_op_pattern)
**Impact on plan:** No behavior change; clippy compliance is required by the project gate.

## Issues Encountered

- Type disambiguation: in the composed-objective closure (`|c: &[Var]| -> Var`), Rust could not infer the impl for free-standing `Scalar::zero()` / `from_f64()` calls. Fixed by using `<Var as Scalar>::zero()` and `<Var as Scalar>::from_f64(...)`. Not a deviation from the plan — the plan's code snippet showed `Scalar::zero()` which would have required an annotation anyway.

## Known Stubs

None.

## Next Phase Readiness

- RAD-03 is fully closed: reverse-mode gradients match central FD within 1e-6 on both existing differentiable-subset functions and their composed objective.
- All four RAD requirements (01, 02, 03) are satisfied — Phase 94 is complete.
- Phase 95 (GEN-01 — Generic Scalar Hot-Path Signatures) is the next milestone phase.

## Self-Check: PASSED

- `fdars-core/src/autodiff/reverse.rs` exists and has 1466 lines (72 tests).
- Commits `3fef0f54`, `f2b2428f`, `50989e38`, `a8b8c897` all present in `git log`.
- `cargo test -p fdars-core --features linalg,parallel autodiff`: 72 passed.
- `cargo test -p fdars-core --features linalg,parallel --doc autodiff`: 7 passed (includes `vjp` doctest).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean.
- `git diff fdars-core/src/metric/soft_dtw.rs fdars-core/src/regression.rs`: empty (validation targets unmodified).

---
*Phase: 94-reverse-mode-autodiff-core-vjp-tape*
*Completed: 2026-09-10*
