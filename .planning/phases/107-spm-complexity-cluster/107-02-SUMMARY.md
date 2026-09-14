---
phase: 107-spm-complexity-cluster
plan: "02"
subsystem: spm
tags: [complexity-reduction, refactor, pmat, spm, mfpca, ncomp, contrib]
status: complete

dependency_graph:
  requires: [107-01]
  provides: [107-03, 107-04]
  affects:
    - fdars-core/src/spm/mfpca.rs
    - fdars-core/src/spm/ncomp.rs
    - fdars-core/src/spm/contrib.rs

tech_stack:
  added: []
  patterns:
    - private-helper-extraction
    - verbatim-loop-body-extraction
    - mut-slice-reference-for-in-place-mutation

key_files:
  modified:
    - fdars-core/src/spm/mfpca.rs
    - fdars-core/src/spm/ncomp.rs
    - fdars-core/src/spm/contrib.rs

decisions:
  - mfpca_build_stacked takes &mut [f64] (not &mut Vec<f64>) for scales — propagates scales[p]=1.0 mutation back to caller; clippy correctly flagged Vec variant as ptr_arg
  - mfpca_extract_components receives u and v_t as named parameters directly (not via _svd field access) to keep the function signature explicit; _svd is still passed for singular_values
  - t2_contributions_mfpca: t2_mfpca_validate returns (ncomp, p) tuple; ncomp from scores.shape() is used authoritatively inside the parent (ncomp_validated shadowed with let _ = )
  - spe_contribs_compute_var marked #[inline] — it is the innermost hot-path loop body (i,j iteration)

metrics:
  duration_seconds: 360
  completed_date: "2026-09-14"
  tasks: 3
  commits: 3
  plan_head_before: f5950f3db6c86deba0133fc00b6931368cb34cd0

actuals:
  tokens: 10000
  tasks: 3
  commits: 3

requirements: [QUAL-02]
---

# Phase 107 Plan 02: MFPCA Cluster Summary

Decomposed the MFPCA cluster in `fdars-core/src/spm/` — `mfpca` (cog 41), `select_ncomp`
(cog 32), `t2_contributions_mfpca` (cog 30), and `spe_contributions` (cog 27) — into private
helpers with numeric-identical, behavior-identical output confirmed by the full test suite.

## One-liner

Private helper extraction drops four MFPCA-cluster functions (mfpca cog 41→0, select_ncomp 32→14, t2_contributions_mfpca 30→13, spe_contributions 27→6) below pmat 3.18.0 thresholds with scales in-place mutation preserved.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 | Decompose mfpca (cog 41→0) — preserve the scales in-place mutation | c91b1fb2 | fdars-core/src/spm/mfpca.rs |
| 2 | Decompose select_ncomp (cog 32→14) | 4c4a3630 | fdars-core/src/spm/ncomp.rs |
| 3 | Decompose t2_contributions_mfpca (cog 30→13) + spe_contributions (cog 27→6); confirm all clean | 3c07f0d3 | fdars-core/src/spm/contrib.rs, mfpca.rs |

## Complexity Results (pmat 3.18.0)

### mfpca — mfpca.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `mfpca` | 41 / 21 | 0 / 1 | CLEAR |
| `mfpca_validate` | (new) | 12 / 5 | CLEAR |
| `mfpca_compute_means_scales` | (new) | 5 / 4 | CLEAR |
| `mfpca_build_stacked` | (new) | 12 / 7 | CLEAR |
| `mfpca_extract_components` | (new) | 12 / 8 | CLEAR |

### select_ncomp — ncomp.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `select_ncomp` | 32 / 10 | 14 / 5 | CLEAR |
| `ncomp_validate_eigenvalues` | (new) | 11 / 4 | CLEAR |
| `ncomp_validate_threshold` | (new) | 4 / 3 | CLEAR |

### t2_contributions_mfpca + spe_contributions — contrib.rs

| Function | Before cog/cyc | After cog/cyc | Status |
|----------|---------------|---------------|--------|
| `t2_contributions_mfpca` | 30 / 12 | 13 / 5 | CLEAR |
| `t2_mfpca_validate` | (new) | 6 / 3 | CLEAR |
| `t2_mfpca_compute_weights` | (new) | 11 / 6 | CLEAR |
| `spe_contributions` | 27 / 11 | 6 / 4 | CLEAR |
| `spe_contribs_validate` | (new) | 4 / 3 | CLEAR |
| `spe_contribs_var_validate` | (new) | 8 / 5 | CLEAR |
| `spe_contribs_compute_var` | (new) | 3 / 3 | CLEAR |

No new helper is flagged by pmat. All functions below cognitive ≤ 25 and cyclomatic ≤ 30.

## Test Results

- SPM module tests: **156 passed, 0 failed**
- Full lib test suite: **2942 passed, 0 failed**
- Doctest suite: **215 passed, 0 failed**
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: **clean**
- `cargo fmt --check`: **clean**

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy ptr_arg on mfpca_build_stacked scales parameter**
- **Found during:** Task 3 (clippy gate)
- **Issue:** `mfpca_build_stacked` used `scales: &mut Vec<f64>` — clippy correctly flagged this as `clippy::ptr_arg` because we only use indexed mutation (no push/pop); idiomatic Rust prefers `&mut [f64]` for slice mutation.
- **Fix:** Changed parameter to `scales: &mut [f64]`. This is still behavior-identical and the in-place mutation `scales[p] = 1.0` still propagates correctly to the caller.
- **Files modified:** fdars-core/src/spm/mfpca.rs
- **Commit:** 3c07f0d3

### Successful Numeric Hazard Mitigation

**MFPCA scales mutation (T-107-04):** The `scales[p] = 1.0` mutation (when `!weighted`) inside
`mfpca_build_stacked` is applied via `&mut [f64]` reference — the mutation propagates back to
the `scales` variable in `mfpca()`, which is then stored in `MfpcaResult.scales`. This means
`MfpcaResult::project()` and `MfpcaResult::reconstruct()` use the correctly mutated scales for
un-scaling. The 156 SPM tests (including projection-related tests) confirm identical output.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes introduced.
Pure private helper extraction within existing module files. No threat flags.

## Known Stubs

None.

## Self-Check: PASSED

- [x] fdars-core/src/spm/mfpca.rs exists and modified (c91b1fb2)
- [x] fdars-core/src/spm/ncomp.rs exists and modified (4c4a3630)
- [x] fdars-core/src/spm/contrib.rs exists and modified (3c07f0d3)
- [x] Commit c91b1fb2 exists (mfpca decomposition)
- [x] Commit 4c4a3630 exists (select_ncomp decomposition)
- [x] Commit 3c07f0d3 exists (contrib.rs + mfpca clippy fix)
- [x] pmat confirms all 4 targets and all new helpers below thresholds
- [x] 2942 lib + 215 doctests green
- [x] clippy --all-targets clean
- [x] No public API/signature/visibility change
