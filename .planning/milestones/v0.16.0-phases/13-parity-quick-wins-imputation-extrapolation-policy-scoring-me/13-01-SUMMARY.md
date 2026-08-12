---
phase: 13-parity-quick-wins-imputation-extrapolation-policy-scoring-me
plan: 01
subsystem: helpers
tags: [feat, parity, imputation, extrapolation, fdars-core]
status: complete

dependency_graph:
  requires: []
  provides:
    - ExtrapolationPolicy enum (Boundary, Exception, Fill(f64), Periodic)
    - fdata_interpolate_with_policy fn
    - ImputationMethod enum (Linear, Mean, Constant(f64))
    - impute_missing_values fn
  affects:
    - fdars-core/src/helpers.rs
    - fdars-core/src/lib.rs

tech_stack:
  added: []
  patterns:
    - "*_with_policy wrapper (Phase-12 pattern): new fn wraps existing private helpers, no signature changes to fdata_interpolate/linear_interp/cubic_hermite_interp"
    - "Guarded modulo for Periodic wrap: ((t-t_min)%L+L)%L prevents negative remainder for t<t_min"
    - "Per-curve imputation via row() + column-major write-back"
    - "is_nan() for NaN detection (never == NaN)"

key_files:
  created: []
  modified:
    - fdars-core/src/helpers.rs
    - fdars-core/src/lib.rs

decisions:
  - "No #[must_use] on Result-returning fns: Result<T,E> is already #[must_use]; double annotation triggers clippy::double_must_use (deviation from plan note — auto-fixed per Rule 1)"
  - "No #[non_exhaustive] on new enums per CONTEXT.md LOCKED decision (FEAT-04 + FEAT-03 enums are small, closed, exhaustive-match intended)"
  - "ExtrapolationPolicy placed before impute_missing_values in helpers.rs to respect FEAT-04 (tracer) then FEAT-03 (auto) task order"

metrics:
  completed_date: "2026-08-11"
  duration_min: 30
  completed_tasks: 3
  total_tasks: 3
  commits: 2

actuals:
  tokens: 15000
  tasks: 3
  commits: 2
---

# Phase 13 Plan 01: FEAT-04 ExtrapolationPolicy + FEAT-03 Imputation Summary

Closes two additive scikit-fda parity gaps in `fdars-core/src/helpers.rs`: composable
`ExtrapolationPolicy` for out-of-range interpolation control (FEAT-04) and in-grid NaN
imputation with three strategies (FEAT-03).

## What Was Built

### FEAT-04: ExtrapolationPolicy + fdata_interpolate_with_policy

New `ExtrapolationPolicy` enum with four variants:
- `Boundary`: clamps out-of-range queries to `[t_min, t_max]`
- `Exception`: returns `Err(InvalidParameter)` for any OOB query
- `Fill(f64)`: writes a constant into OOB result cells
- `Periodic`: wraps via guarded-modulo `((t-t_min)%L+L)%L`

New `fdata_interpolate_with_policy` fn applies the policy per query point then delegates
in-range interpolation to the existing private `linear_interp` / `cubic_hermite_interp`.
Original `fdata_interpolate` is unchanged (Phase-12 `*_with_policy` pattern).

### FEAT-03: ImputationMethod + impute_missing_values

New `ImputationMethod` enum with three variants:
- `Linear`: interpolates between nearest non-NaN neighbors; boundary NaN filled by nearest valid
- `Mean`: replaces each NaN with the curve's non-NaN mean
- `Constant(f64)`: replaces each NaN with a user-supplied constant

New `impute_missing_values` fn validates input, rejects all-NaN curves, imputes per-curve,
writes result column-major, and returns a new `FdMatrix`.

### Crate-root re-exports (lib.rs)

Both enums and both fns added to the `pub use helpers::{...}` block.

## Verification

- `cargo test -p fdars-core --features linalg` — 1969 tests, all pass (0 failed)
- `cargo clippy --all-targets --all-features -D warnings` — clean
- `git diff` on committed range confirms additive-only changes (no edits inside
  `fdata_interpolate`, `linear_interp`, `spline_interpolate`, `cubic_hermite_interp`)
- 12 new inline tests: 6 extrapolation (boundary/exception/fill/periodic/in-range-equiv/dim-guard)
  + 6 imputation (linear/mean/constant/all-nan/boundary-nan/dim-guard)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed `#[must_use]` from `fdata_interpolate_with_policy`**
- **Found during:** clippy run before Task 1 commit
- **Issue:** `Result<FdMatrix, FdarError>` is itself `#[must_use]`; adding `#[must_use]` to the
  function triggers `clippy::double_must_use` under `-D warnings`
- **Fix:** Removed `#[must_use]` attribute from `fdata_interpolate_with_policy` (and similarly
  from `impute_missing_values` preemptively). The STATE.md convention note confirms this:
  "`#[must_use]` on expensive computations (note: `Result<T, E>` already carries must_use —
  do not double-annotate)"
- **Files modified:** `fdars-core/src/helpers.rs`
- **Commit:** 109dc103 (Task 1)

**2. [Rule 3 - Formatting] Applied cargo fmt after each task commit attempt**
- **Found during:** pre-commit hook running `cargo fmt --check`
- **Issue:** Rustfmt reformats multi-line expressions differently than written
- **Fix:** Ran `cargo fmt -p fdars-core` before re-staging and committing each task
- **Files modified:** `fdars-core/src/helpers.rs`, `fdars-core/src/lib.rs`

## Known Stubs

None. Both FEAT-03 and FEAT-04 are fully wired with real implementations and inline tests.

## Threat Flags

No new security-relevant surface beyond the mitigations in the plan's threat model:
- T-13-01: `is_nan()` used throughout imputation (not `== NaN`)
- T-13-02: Guarded-modulo in Periodic variant implemented correctly
- T-13-03: Exception policy returns `Err` instead of clamping
- T-13-04: Entry-point `InvalidDimension` guards on both public fns

## Self-Check

- [x] `fdars-core/src/helpers.rs` exists and contains both new enums and both new fns
- [x] `fdars-core/src/lib.rs` re-exports all four new items
- [x] Commit 109dc103 exists (FEAT-04)
- [x] Commit 39e050a8 exists (FEAT-03)
- [x] Full test suite green (1969 tests)
- [x] CI-parity clippy clean (`--all-targets --all-features -D warnings`)

## Self-Check: PASSED
