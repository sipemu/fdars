---
phase: 13-parity-quick-wins-imputation-extrapolation-policy-scoring-me
plan: "02"
subsystem: scoring
tags: [feat, scoring, functional-metrics, mae, mse, mape, msle, explained-variance, FEAT-05]
status: complete

dependency_graph:
  requires: [13-01]
  provides: [fdars-core/src/scoring.rs, functional_mae, functional_mse, functional_mape, functional_msle, functional_explained_variance]
  affects: [fdars-core/src/lib.rs]

tech_stack:
  added: []
  patterns:
    - Simpson's rule integration via simpsons_weights for all five metrics
    - Private validate_shapes helper for DRY shape validation across all metrics
    - Pre-scan domain validation before integration (MAPE zero-check, MSLE log-domain-check)
    - Per-curve EV computation then averaged (per RESEARCH Assumption A3)

key_files:
  created:
    - fdars-core/src/scoring.rs
  modified:
    - fdars-core/src/lib.rs

decisions:
  - All five functional scoring metrics implemented in a single tracer task (task 1) because they share validate_shapes + integration pattern — splitting across tasks would have been redundant
  - validate_shapes private helper returns (n, m) to avoid repeated destructuring across all five fns
  - MAPE pre-scans all y_true values before computing any integral (fail-fast, no partial NaN results)
  - MSLE pre-scans both y_true and y_pred for the -1 domain boundary
  - explained_variance: SS_tot near-zero guarded — returns 1.0 when both SS_res and SS_tot ~ 0 (trivial perfect fit), 0.0 when SS_tot ~ 0 but SS_res > 0
  - All five re-exported at crate root in a single pub use scoring block (lib.rs:443)

metrics:
  duration_minutes: 10
  completed: "2026-08-11"
  tasks_completed: 3
  tasks_planned: 3
  commits: 1

actuals:
  tokens: 12500
  tasks: 3
  commits: 1
---

# Phase 13 Plan 02: Scoring Metrics (FEAT-05) Summary

**One-liner:** Five functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance) integrated over argvals via Simpson's rule with per-curve averaging, domain validation, and crate-root re-export.

## What Was Built

New module `fdars-core/src/scoring.rs` implementing FEAT-05 (scikit-fda MISC-04 parity gap):

| Function | Description |
|----------|-------------|
| `functional_mae` | `(1/n) * sum_i ∫ |y_true_i(t) - y_pred_i(t)| dt` |
| `functional_mse` | `(1/n) * sum_i ∫ (y_true_i(t) - y_pred_i(t))^2 dt` |
| `functional_mape` | `(1/n) * sum_i ∫ |y_true_i(t) - y_pred_i(t)| / |y_true_i(t)| dt` |
| `functional_msle` | `(1/n) * sum_i ∫ (ln(1+y_true_i(t)) - ln(1+y_pred_i(t)))^2 dt` |
| `functional_explained_variance` | `(1/n) * sum_i (1 - SS_res_i / SS_tot_i)` where SS are integrated squared deviations |

All metrics:
- Accept `(y_true: &FdMatrix, y_pred: &FdMatrix, argvals: &[f64])`, return `Result<f64, FdarError>`
- Use `helpers::simpsons_weights(argvals)` for integration (uniform/non-uniform grids)
- Share private `validate_shapes` helper for dimension guards
- Are declared `pub mod scoring` and re-exported via `pub use scoring::{...}` at crate root

## Validation Guards

| Guard | Trigger | Error |
|-------|---------|-------|
| y_pred shape mismatch | `y_pred.shape() != y_true.shape()` | `InvalidDimension { parameter: "y_pred" }` |
| argvals length mismatch | `argvals.len() != y_true.ncols()` | `InvalidDimension { parameter: "argvals" }` |
| Degenerate input | `n == 0 || m < 2` | `InvalidDimension { parameter: "data" }` |
| MAPE near-zero denominator | `|y_true[(i,j)]| < NUMERICAL_EPS` | `InvalidParameter { parameter: "y_true" }` |
| MSLE y_true domain | `y_true[(i,j)] <= -1 + NUMERICAL_EPS` | `InvalidParameter { parameter: "y_true" }` |
| MSLE y_pred domain | `y_pred[(i,j)] <= -1 + NUMERICAL_EPS` | `InvalidParameter { parameter: "y_pred" }` |

## Inline Tests (15 total)

All pass under `cargo test -p fdars-core --features linalg`:

- `test_functional_mae_constant_error` — constant error c: integral over [0,1] = c (hand-computed)
- `test_functional_mae_multi_curve` — two curves with errors 1 and 2: average = 1.5
- `test_functional_mae_shape_mismatch_y_pred` — wrong ncols returns `Err(InvalidDimension{y_pred})`
- `test_functional_mae_shape_mismatch_argvals` — wrong argvals len returns `Err(InvalidDimension{argvals})`
- `test_functional_mse_constant_error` — constant error c: integral = c^2 (hand-computed)
- `test_functional_mse_zero_error` — perfect prediction returns MSE = 0
- `test_functional_mape_constant_error` — y_true=4, y_pred=5: MAPE = 0.25 (hand-computed)
- `test_functional_mape_zero_y_true` — near-zero y_true returns `Err(InvalidParameter{y_true})`
- `test_functional_msle_constant` — identical inputs: MSLE = 0
- `test_functional_msle_hand_computed` — y_true=3, y_pred=1: MSLE = ln(2)^2 ≈ 0.480453
- `test_functional_msle_domain_y_true` — y_true=-1.5 returns `Err(InvalidParameter{y_true})`
- `test_functional_msle_domain_y_pred` — y_pred=-2.0 returns `Err(InvalidParameter{y_pred})`
- `test_explained_variance_perfect` — y_pred == y_true: EV = 1.0
- `test_explained_variance_constant_true` — constant y_true and y_pred: EV = 1.0
- `test_explained_variance_shape_mismatch` — wrong y_pred shape returns `Err(InvalidDimension{y_pred})`

## Commits

| Hash | Message |
|------|---------|
| `ccffc099` | feat(13-02): add scoring.rs with functional_mae + functional_mse (FEAT-05) |

Note: All five functions (including mape/msle/explained_variance) were implemented in the tracer task (Task 1) to keep the module cohesive. The validate_shapes helper, integration pattern, and module structure made implementing all five simultaneously simpler than splitting across tasks.

## Deviations from Plan

### Implementation Approach

All five metrics implemented in Task 1 (tracer) rather than splitting across Tasks 1 and 2. The plan's Task 2 described adding mape/msle/explained_variance, but these were already implemented in the tracer commit because:
1. validate_shapes shared helper was needed by all five — factoring it once is cleaner
2. The integration pattern is identical across all five — adding them together avoids duplication
3. The tests for all five are in the same `#[cfg(test)]` block

Task 2 and Task 3 were therefore verification gates on the already-committed code, not separate implementation steps. This matches Rule 2 (auto-add missing critical functionality) — building all five consistently in one pass is more correct than partial implementation.

## Threat Mitigations Applied

| Threat ID | Mitigation |
|-----------|------------|
| T-13-05 | MAPE pre-scan: returns `InvalidParameter` if any `|y_true| < NUMERICAL_EPS` |
| T-13-06 | MSLE pre-scan: returns `InvalidParameter` if any value `<= -1 + NUMERICAL_EPS` |
| T-13-07 | All domain violations return `Err` — no silent NaN/Inf propagation |
| T-13-08 | `validate_shapes` at entry of every function — no silent truncation |

## Threat Flags

None. The scoring module adds no new network endpoints, auth paths, file access, or external-facing schema changes.

## Known Stubs

None. All five functions are fully implemented and produce correct numerical output.

## Self-Check: PASSED

- `fdars-core/src/scoring.rs` exists: FOUND
- `ccffc099` exists in git log: FOUND
- All 15 inline tests pass: PASSED
- `cargo test -p fdars-core --features linalg` full suite: PASSED (1984 tests ok)
- `cargo clippy --all-targets --all-features -D warnings ...`: CLEAN
- All five names reachable from crate root (`pub use scoring::{...}` at lib.rs:443): VERIFIED
