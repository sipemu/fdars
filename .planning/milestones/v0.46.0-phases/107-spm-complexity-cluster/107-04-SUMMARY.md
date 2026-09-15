---
phase: 107-spm-complexity-cluster
plan: "04"
subsystem: spm
tags: [complexity-reduction, refactor, pmat, spm, partial, frcc, profile]
status: complete

dependency_graph:
  requires: [107-01]
  provides: []
  affects:
    - fdars-core/src/spm/partial.rs
    - fdars-core/src/spm/frcc.rs
    - fdars-core/src/spm/profile.rs

tech_stack:
  added: []
  patterns:
    - private-helper-extraction
    - split-validate
    - tikhonov-retry-preservation

key_files:
  modified:
    - fdars-core/src/spm/partial.rs
    - fdars-core/src/spm/frcc.rs
    - fdars-core/src/spm/profile.rs

decisions:
  - cond_exp_build_m_matrix returns (m_matrix, diag_max) tuple where diag_max is captured from the ORIGINAL pre-regularization matrix, then passed into cond_exp_solve_with_retry to preserve the Tikhonov retry sequence byte-identically
  - frcc_fosr_lambda guard moved into frcc_validate (pre-split) so all guards consolidated before the split_indices call
  - profile_validate_windows extracted as a second-level split-validate to keep profile_validate under threshold (7 guards total)
  - rolling_betas (cog 21) and solve_spd (cog 21) left untouched — warning-level, out of scope for QUAL-02

metrics:
  duration_seconds: 1800
  completed_date: "2026-09-14"
  tasks: 3
  commits: 3
  plan_head_before: 8beb5eae11aeb4226effc9fe69b98044ddaa2541

actuals:
  tokens: 18000
  tasks: 3
  commits: 3

requirements: [QUAL-02]
---

# Phase 107 Plan 04: SPM Remaining Monitors Summary

Decomposed the final three SPM QUAL-02 error-level complexity hotspots: `conditional_expectation` (cog 29) in partial.rs, `frcc_phase1` (cog 29) in frcc.rs, and `profile_phase1` (cog 26) in profile.rs. All three refactored to private helper extraction with numeric-identical, behavior-identical output confirmed by the existing SPM test suite.

## One-liner

Private helper extraction drops conditional_expectation (29→3), frcc_phase1 (29→3), and profile_phase1 (26→3) below pmat 3.18.0 thresholds with the Tikhonov retry sequence preserved byte-identically.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | decompose conditional_expectation (cog 29→3) | 091ec1a0 | fdars-core/src/spm/partial.rs |
| 2 | decompose frcc_phase1 (cog 29→3) | 6df45fee | fdars-core/src/spm/frcc.rs |
| 3 | decompose profile_phase1 (cog 26→3); confirm all 3 targets clean | 759e3532 | fdars-core/src/spm/profile.rs |

## Complexity Results (pmat 3.18.0)

### conditional_expectation — partial.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `conditional_expectation` | 29 / 16 | 3 / 3 | CLEAR |
| `cond_exp_sigma2` | (new) | 3 / 3 | CLEAR |
| `cond_exp_build_m_matrix` | (new) | 10 / 8 | CLEAR |
| `cond_exp_solve_with_retry` | (new) | 13 / 5 | CLEAR |

`solve_spd` (cog 21, warning-level) left untouched — out of scope for QUAL-02.

### frcc_phase1 — frcc.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `frcc_phase1` | 29 / 13 | 3 / 2 | CLEAR |
| `frcc_validate` | (new) | 16 / 7 | CLEAR |
| `frcc_validate_split` | (new) | 6 / 3 | CLEAR |
| `frcc_compute_r_squared` | (new) | 4 / 4 | CLEAR |

### profile_phase1 — profile.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `profile_phase1` | 26 / 12 | 3 / 2 | CLEAR |
| `profile_validate` | (new) | 13 / 6 | CLEAR |
| `profile_validate_windows` | (new) | 6 / 3 | CLEAR |
| `profile_lag1_autocorr` | (new) | 4 / 3 | CLEAR |
| `profile_effective_n` | (new) | 1 / 2 | CLEAR |

`rolling_betas` (cog 21, warning-level) left untouched — out of scope for QUAL-02.

No new helper is flagged by pmat. All functions below cognitive ≤ 25 and cyclomatic ≤ 30.

## Test Results

- SPM module tests: **156 passed, 0 failed**
- Full lib test suite: **2942 passed, 0 failed**
- Doctest suite: **215 passed, 0 failed**
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: **clean**
- `cargo fmt -p fdars-core --check`: **clean**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing seam] split-validate applied to profile_validate**
- **Found during:** Task 3
- **Issue:** `profile_validate` with 7 guards (predictors rows, argvals len, step_size==0, window_size<3, window_size>n, expected_n_windows<4, n_windows check) risked exceeding cog 25 if all inline. Research assumption A4 anticipated this.
- **Fix:** Applied the Phase-106/107-01 proven split-validate pattern: `profile_validate` delegates window-size guards to `profile_validate_windows` (cog 6), keeping `profile_validate` at cog 13.
- **Files modified:** fdars-core/src/spm/profile.rs
- **Commit:** 759e3532

**2. [Rule 3 - Seam adjustment] frcc_fosr_lambda guard moved into frcc_validate**
- **Found during:** Task 2
- **Issue:** The original `frcc_phase1` placed the `fosr_lambda < 0.0` guard after the split/subset operations, breaking the clean pre-split / post-split extraction seam described in RESEARCH.md.
- **Fix:** Moved the `fosr_lambda` guard into `frcc_validate` (pre-split), consolidating all pre-split guards. Semantically equivalent — the guard fires before any FOSR work in both the original and refactored code. Behavior identical.
- **Files modified:** fdars-core/src/spm/frcc.rs
- **Commit:** 6df45fee

### Successful Numeric Hazard Mitigation

**Tikhonov retry sequence (T-107-10):** `cond_exp_build_m_matrix` computes `diag_max` from the ORIGINAL M matrix (before any regularization is applied) and returns it as a tuple element alongside the (optionally regularized) matrix. `cond_exp_solve_with_retry` receives `diag_max` as a parameter and starts the retry sequence with `diag_max * 1e-8`. The retry loop (reg_strength *= 10.0, up to 5 attempts) is extracted verbatim. The solve_spd call order is identical to the original. Full-suite test coverage via spm::partial tests confirms identical estimated scores + completed_curve.

**frcc_phase1 split determinism (T-107-11):** The `split_indices(n, tuning_fraction, seed)` call remains in the parent — its output depends only on the inputs, not on helper extraction. `frcc_compute_r_squared` receives `tune_y` and `tune_predicted` by reference; FP accumulation order is verbatim from the original. spm::frcc tests confirm identical eigenvalues / R² / control limits.

**profile_phase1 FP accumulation (T-107-11):** `profile_lag1_autocorr` extracts the lag-1 autocorrelation computation verbatim — the mean→var→cov accumulation order is preserved exactly. `profile_effective_n` extracts the Bartlett factor computation verbatim. spm::profile tests confirm identical lag1_autocorrelation and effective_n_windows.

## Phase 107 QUAL-02 Completion

Combined with plans 107-01/02/03, all 13 SPM error-level functions cleared:

| Plan | Functions | Before (max cog) | After (max cog) |
|------|-----------|-----------------|-----------------|
| 107-01 | spm_amewma_monitor, spm_phase1_iterative | 52, 42 | 11, 12 |
| 107-02 | mfpca, select_ncomp, t2_contributions_mfpca, spe_contributions | 41, 32, 30, 27 | all ≤ 14 |
| 107-03 | spm_ewma_monitor, spm_mewma_monitor, spe_multivariate, spm_monitor_from_fields | 27, 25, 27, 27 | all ≤ 12 |
| 107-04 | conditional_expectation, frcc_phase1, profile_phase1 | 29, 29, 26 | 3, 3, 3 |

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced. Pure private helper extraction within existing module files. No threat flags.

## Known Stubs

None.

## Self-Check: PASSED

- [x] fdars-core/src/spm/partial.rs exists and modified
- [x] fdars-core/src/spm/frcc.rs exists and modified
- [x] fdars-core/src/spm/profile.rs exists and modified
- [x] Commit 091ec1a0 exists (conditional_expectation decomposition)
- [x] Commit 6df45fee exists (frcc_phase1 decomposition)
- [x] Commit 759e3532 exists (profile_phase1 decomposition)
- [x] pmat confirms all 3 targets and all new helpers below thresholds
- [x] 2942 lib + 215 doctests green
- [x] clippy clean, fmt clean
- [x] No public API/signature/visibility change; solve_spd/rolling_betas left untouched
