---
phase: 14-shift-registration
plan: "01"
subsystem: alignment
tags: [shift-registration, golden-section, functional-data, feat-06]
status: complete

dependency_graph:
  requires: []
  provides:
    - ShiftRegistrationResult
    - least_squares_shift_registration
    - DEFAULT_MAX_SHIFT_FRACTION
  affects:
    - fdars-core/src/alignment/mod.rs

tech_stack:
  added: []
  patterns:
    - golden-section 1D minimisation (new private helper)
    - parallel-collect-then-sequential-assemble (mirrors set.rs)
    - Simpson-weighted L2 shift objective

key_files:
  created:
    - fdars-core/src/alignment/shift.rs
  modified:
    - fdars-core/src/alignment/mod.rs

decisions:
  - "Sign convention: registered(t) = original(t - delta); positive delta shifts peak rightward in registered output"
  - "Rule 3 deviation: added pub use shift::{...} to alignment/mod.rs in this plan (not deferred to 14-02) because clippy -D warnings flags public items in private modules as dead_code; sequential execution has no merge-conflict risk"
  - "DEFAULT_MAX_SHIFT_FRACTION exported from alignment:: so the doctest can reference it without a magic literal"
  - "make_shifted_bumps helper retained in tests module even though unused by the two initial tests (available for future tests)"

metrics:
  duration: "~13 minutes"
  completed: "2026-08-12"
  tasks_completed: 3
  commits: 3

actuals:
  tokens: 5400
  tasks: 3
  commits: 3
---

# Phase 14 Plan 01: Least-Squares Shift Registration (FEAT-06) Summary

Delivered rigid horizontal shift registration via a new `fdars-core/src/alignment/shift.rs` module implementing `least_squares_shift_registration`. Each curve is aligned to the cross-sectional sample mean by minimising the Simpson-weighted L2 objective over a per-curve shift δᵢ found by golden-section search.

## What Was Built

**New file: `fdars-core/src/alignment/shift.rs`** (428 lines)

- `ShiftRegistrationResult { registered_data: FdMatrix, shifts: Vec<f64> }` — `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, serde-gated, per D-Area1 decision.
- `least_squares_shift_registration(data, argvals, max_shift) -> Result<ShiftRegistrationResult, FdarError>` — full V5/V7 input validation, golden-section search, parallel-collect-then-sequential-assemble pattern.
- `DEFAULT_MAX_SHIFT_FRACTION: f64 = 0.25` — public constant for caller guidance.
- Private `golden_section_search` (PHI = 1.618…, 100 max iterations, tol = 1e-6).
- Private `l2_shift_objective` — Simpson-weighted ‖fᵢ(t − δ) − mean(t)‖² via `linear_interp` (Boundary clamping built-in).
- Inline `#[cfg(test)] mod tests` with 5 tests (FEAT-06-A…E) plus doctest.

**Modified: `fdars-core/src/alignment/mod.rs`** (4 lines added)

- `mod shift;` declaration (alphabetical after `shape_ci`).
- `pub use shift::{least_squares_shift_registration, ShiftRegistrationResult, DEFAULT_MAX_SHIFT_FRACTION};`

## Tests

| Test | Req | Result |
|------|-----|--------|
| `test_shift_already_aligned` | FEAT-06-A | PASS — all δᵢ < 1e-3 on constant/aligned set |
| `test_shift_recovers_injected_offset` | FEAT-06-B | PASS — bumps at 0.4/0.5/0.6 recover offsets within 0.05 |
| `test_shift_registration_curve_values` | FEAT-06-C | PASS — spot-checks `registered_data[(i,j)] == linear_interp(argvals, row_i, argvals[j] - shifts[i])` within 1e-9 |
| `test_shift_registration_empty_data` | FEAT-06-D | PASS — n=0 and m=0 return `Err(FdarError::InvalidDimension)` |
| `test_shift_registration_argvals_mismatch` | FEAT-06-E | PASS — length mismatch returns `Err(FdarError::InvalidDimension)` |
| Doctest | SC1 | PASS — `cargo test --doc -- alignment::shift` green |

## Deviations from Plan

### Auto-fixed Issues

**[Rule 3 - Blocking] Added pub use shift::{...} to alignment/mod.rs in plan 14-01**

- **Found during:** Task 1 — clippy gate
- **Issue:** Plan deferred all `mod.rs` re-exports to plan 14-02 to avoid merge conflicts. However, in sequential (non-worktree) execution there is no parallel plan; without `pub use shift::`, clippy `-D warnings` flags every public item in the module as dead_code (`error: constant DEFAULT_MAX_SHIFT_FRACTION is never used`, etc.), blocking the commit.
- **Fix:** Added `mod shift;` + `pub use shift::{least_squares_shift_registration, ShiftRegistrationResult, DEFAULT_MAX_SHIFT_FRACTION};` to `alignment/mod.rs` in this plan.
- **Impact on 14-02:** Plan 14-02 will attempt to add the same `pub use shift` line. It must skip or handle that idempotently.
- **Files modified:** `fdars-core/src/alignment/mod.rs`
- **Commits:** `9de3bf0e`

**[Rule 1 - Bug] Fixed sign convention in FEAT-06-B test**

- **Found during:** Task 1 — first test run
- **Issue:** Initial test had `true_shifts = [0.0, -0.1, 0.1]` (expecting the shift to be the negative of the displacement from mean). The correct sign is `δᵢ = mean_centre - mu_i`: `registered(t) = original(t - δ)`, so to bring peak at mu=0.4 to t=0.5, δ=+0.1 (evaluate original at t-0.1=0.4).
- **Fix:** Corrected expected shifts to `[0.0, +0.1, -0.1]` with explanatory comments.
- **Commits:** `9de3bf0e`

## Re-export Note

`mod shift;` and `pub use shift::{...}` were added to `alignment/mod.rs` in this plan (deviation from plan note). Plan 14-02 (re-export consolidation) should skip the shift re-export if already present, or simply omit it from its scope. The `lib.rs` crate-root re-export remains for plan 14-02.

## Self-Check

### Files exist

- FOUND: `fdars-core/src/alignment/shift.rs`
- FOUND: `fdars-core/src/alignment/mod.rs` (modified)

### Commits exist

- FOUND: `9de3bf0e` — feat(14-01): implement least_squares_shift_registration (FEAT-06-A, B)
- FOUND: `6c09a695` — test(14-01): add registered-curve spot-check test (FEAT-06-C)
- FOUND: `c9adc9b6` — test(14-01): add input validation + error-path tests (FEAT-06-D, FEAT-06-E)

## Self-Check: PASSED
