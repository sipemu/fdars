---
phase: 107-spm-complexity-cluster
plan: "01"
subsystem: spm
tags: [complexity-reduction, refactor, pmat, spm, amewma, iterative]
status: complete

dependency_graph:
  requires: []
  provides: [107-02, 107-03, 107-04]
  affects: [fdars-core/src/spm/amewma.rs, fdars-core/src/spm/iterative.rs]

tech_stack:
  added: []
  patterns:
    - private-helper-extraction
    - split-validate
    - verbatim-loop-body-extraction

key_files:
  modified:
    - fdars-core/src/spm/amewma.rs
    - fdars-core/src/spm/iterative.rs

decisions:
  - Split-validate applied to amewma_validate (7 guards): split into amewma_validate_lambda_range (4 lambda guards, cog 15) + amewma_validate_params (eta + clip, cog 10) to stay under threshold
  - amewma_ewma_step extracts only the EWMA inner body (S_t = lambda*z + (1-lambda)*prev) and is marked #[inline]; the outer for-i recurrence loop stays intact in parent
  - iterative_update_state uses &mut Vec<usize> refs to preserve the flagged_set + remaining_indices filter in exact original order
  - iterative_check_removal_budget returns bool (should_stop) covering both the fraction check and n_after<4 check; called before iterative_update_state to replicate the original break ordering

metrics:
  duration_seconds: 408
  completed_date: "2026-09-14"
  tasks: 2
  commits: 2
  plan_head_before: fca1db5a89241cdd0baa453d1397ae04cd1d8150

actuals:
  tokens: 12000
  tasks: 2
  commits: 2

requirements: [QUAL-02]
---

# Phase 107 Plan 01: SPM Tracer — amewma + iterative Summary

Proved the Phase-107 extract→test→pmat→commit loop on the two highest-scoring SPM
complexity hotspots: `spm_amewma_monitor` (cog 52, cyc 25) and `spm_phase1_iterative`
(cog 42, cyc 16). Both refactored to private helper extraction with numeric-identical,
behavior-identical output confirmed by the existing SPM test suite.

## One-liner

Private helper extraction drops spm_amewma_monitor (cog 52→11) and spm_phase1_iterative (cog 42→12) below pmat 3.18.0 thresholds with sequential recurrence order preserved.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | TRACER — decompose spm_amewma_monitor (cog 52→11) | e889c45e | fdars-core/src/spm/amewma.rs |
| 2 | decompose spm_phase1_iterative + confirm both clean (cog 42→12) | 58c678b1 | fdars-core/src/spm/iterative.rs |

## Complexity Results (pmat 3.18.0)

### spm_amewma_monitor — amewma.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `spm_amewma_monitor` | 52 / 25 | 11 / 7 | CLEAR |
| `amewma_validate` | (new) | 3 / 2 | CLEAR |
| `amewma_validate_lambda_range` | (new) | 15 / 8 | CLEAR |
| `amewma_validate_params` | (new) | 10 / 5 | CLEAR |
| `amewma_project_standardize` | (new) | 12 / 7 | CLEAR |
| `amewma_ewma_step` | (new) | 1 / 2 | CLEAR |

### spm_phase1_iterative — iterative.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `spm_phase1_iterative` | 42 / 16 | 12 / 5 | CLEAR |
| `iterative_validate` | (new) | 11 / 6 | CLEAR |
| `iterative_identify_flagged` | (new) | 9 / 6 | CLEAR |
| `iterative_check_removal_budget` | (new) | 3 / 2 | CLEAR |
| `iterative_update_state` | (new) | 0 / 1 | CLEAR |

No new helper is flagged by pmat. All functions below cognitive ≤ 25 and cyclomatic ≤ 30.

## Test Results

- SPM module tests: **156 passed, 0 failed**
- Full lib test suite: **2942 passed, 0 failed**
- Doctest suite: **215 passed, 0 failed**
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: **clean**
- `cargo fmt --check`: **clean**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing detail] Split-validate applied to amewma_validate**
- **Found during:** Task 1
- **Issue:** `amewma_validate` contained 7 guards (1 dimension + 4 lambda + 2 eta/clip). The 4 lambda guards alone produce sufficient nesting to risk pushing extracted validator above cog 15.
- **Fix:** Applied the Phase-106 proven split-validate pattern: top-level `amewma_validate` delegates to `amewma_validate_lambda_range` (4 lambda/lambda_init guards, cog 15) and `amewma_validate_params` (eta + error_clip, cog 10). Parent `amewma_validate` scores cog 3.
- **Files modified:** fdars-core/src/spm/amewma.rs
- **Commit:** e889c45e

### Successful Numeric Hazard Mitigation

**AMEWMA sequential recurrence (T-107-01):** The outer `for i in 0..n` loop is preserved
intact in `spm_amewma_monitor`. `amewma_ewma_step` receives `lambda_cur` and the previous
smoothed row as parameters — it does not write `lambda_prev`. The recurrence state
(`q_prev`, `lambda_prev`, `sum_lambda_sq_prod`) is updated in its original order after
the ewma_step call, inside the outer loop. Test coverage via `test_amewma_time_dependent_covariance`
and `test_amewma_lambda_increases_on_shift` confirms identical lambda_t and T² sequences.

**iterative state recurrence (T-107-02):** The outer `for _ in 0..config.max_iterations`
loop is preserved intact. `iterative_update_state` receives all mutable state as `&mut` refs
and performs the flagged_set construction → remaining_indices filter → rate/history update in
the exact original order.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced.
Pure private helper extraction within existing module files. No threat flags.

## Known Stubs

None.

## Self-Check: PASSED

- [x] fdars-core/src/spm/amewma.rs exists and modified
- [x] fdars-core/src/spm/iterative.rs exists and modified
- [x] Commit e889c45e exists (spm_amewma_monitor decomposition)
- [x] Commit 58c678b1 exists (spm_phase1_iterative decomposition)
- [x] pmat confirms both targets and all new helpers below thresholds
- [x] 2942 lib + 215 doctests green
- [x] No public API/signature/visibility change
