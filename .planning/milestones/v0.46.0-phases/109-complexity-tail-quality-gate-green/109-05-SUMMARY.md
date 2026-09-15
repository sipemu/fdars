---
phase: 109-complexity-tail-quality-gate-green
plan: "05"
subsystem: clustering
tags: [pmat, complexity, refactor, coclustering, kernel-kmeans, kshape]

requires: ["109-01"]
provides:
  - "coclustering.rs: m_step, e_row_step, e_col_step, col_kmeans_init below cog 23 / cyc 30"
  - "kernel_kmeans.rs: run_restart below cog 23 / cyc 30"
  - "kshape.rs: shape_extraction below cog 23 / cyc 30"
affects: []

actuals:
  tokens: 22000
  tasks: 3
  commits: 3
  plan_head_before: 1b1f2e7ddc5119fa85d2035c18912ac1adaea7be

tech-stack:
  added: []
  patterns:
    - "Multi-seam decomposition: coclustering m_step (cog 56) needed 3-seam split (block_mean_pass, block_var_pass, m_step_block_params)"
    - "Self-flagging helper caught mid-task: kmeans_iterate (cog 32) and m_step_block_params (cog 27) each needed further split before threshold cleared"
    - "ptr_arg clippy fix: &mut Vec<T> -> &mut [T] in kmeans_iterate (auto-fix Rule 1)"
    - "nalgebra SymmetricEigen extraction: shape_extraction split into build_aligned_members + build_xtx + apply_q_centering + compute_shape_matrix"

key-files:
  created: []
  modified:
    - fdars-core/src/coclustering.rs
    - fdars-core/src/kernel_kmeans.rs
    - fdars-core/src/kshape.rs

key-decisions:
  - "coclustering m_step (cog 56): three-seam split — block_mean_pass + block_var_pass extracted via m_step_block_params; m_step delegates per-block Gaussian computation to m_step_block_params"
  - "coclustering col_kmeans_init (cog 50): extracted kmeans_pp_select_centers (preserves &mut StdRng draw order) and kmeans_iterate -> recompute_centroids + reassign_to_centroids; ptr_arg fixed"
  - "coclustering e_row_step (cog 29): extracted row_cluster_score (the per-curve log-density accumulation)"
  - "coclustering e_col_step (cog 48): extracted build_point_scores (s buffer) + col_candidate_gain (per-l_cand gain with FP accumulation verbatim)"
  - "kernel_kmeans run_restart (cog 33): extracted compute_d2_matrix + argmin_reassign; RNG draw order preserved (gen_range then ensure_no_empty_random)"
  - "kshape shape_extraction (cog 32): two-seam then further split — build_aligned_members + (build_xtx + apply_q_centering via compute_shape_matrix); sign-fix and eigenvector loop stay in shape_extraction"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "m_step (coclustering.rs) cog 56->5, cyc 15->5; helpers m_step_block_params cog 1, block_mean_pass cog 11, block_var_pass cog 9"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "coclustering::tests (15 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "col_kmeans_init (coclustering.rs) cog 50->3, cyc 17->2; helpers kmeans_pp_select_centers cog 15, kmeans_iterate cog 6, recompute_centroids cog 9, reassign_to_centroids cog 8"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "coclustering::tests::test_determinism_under_seed,test_coclustering_recovers_block_structure"
        status: pass
    human_judgment: false
  - id: D3
    description: "e_row_step (coclustering.rs) cog 29->13, cyc 8->5; helper row_cluster_score cog 8"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "coclustering::tests (15 tests)"
        status: pass
    human_judgment: false
  - id: D4
    description: "e_col_step (coclustering.rs) cog 48->13, cyc 13->5; helpers build_point_scores cog 3, col_candidate_gain cog 16"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "coclustering::tests (15 tests)"
        status: pass
    human_judgment: false
  - id: D5
    description: "run_restart (kernel_kmeans.rs) cog 33->12, cyc 13->6; helpers compute_d2_matrix cog 8, argmin_reassign cog 6"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "kernel_kmeans::tests (10 tests)"
        status: pass
    human_judgment: false
  - id: D6
    description: "shape_extraction (kshape.rs) cog 32->5, cyc 18->5; helpers build_aligned_members cog 1, build_xtx cog 13, apply_q_centering cog 13, compute_shape_matrix cog 0"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "kshape::tests (9 tests)"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 05: Coclustering + Kernel-KMeans + KShape Complexity Tail Summary

**Six clustering violations eliminated via private helper extraction (coclustering.rs, kernel_kmeans.rs, kshape.rs) — all targets now cog <=13 / cyc <=6, all helpers cog <=16 / cyc <=13, 34/34 tests pass, clippy clean**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-09-14
- **Tasks:** 3
- **Commits:** 3
- **Files modified:** 3

## Before → After Scores

| Function | File | cog before | cog after | cyc before | cyc after |
|---|---|---|---|---|---|
| `m_step` | coclustering.rs | 56 | 5 | 15 | 5 |
| `col_kmeans_init` | coclustering.rs | 50 | 3 | 17 | 2 |
| `e_col_step` | coclustering.rs | 48 | 13 | 13 | 5 |
| `e_row_step` | coclustering.rs | 29 | 13 | 8 | 5 |
| `run_restart` | kernel_kmeans.rs | 33 | 12 | 13 | 6 |
| `shape_extraction` | kshape.rs | 32 | 5 | 18 | 5 |
| `m_step_block_params` (new) | coclustering.rs | — | 1 | — | 2 |
| `block_mean_pass` (new) | coclustering.rs | — | 11 | — | 6 |
| `block_var_pass` (new) | coclustering.rs | — | 9 | — | 5 |
| `row_cluster_score` (new) | coclustering.rs | — | 8 | — | 4 |
| `build_point_scores` (new) | coclustering.rs | — | 3 | — | 3 |
| `col_candidate_gain` (new) | coclustering.rs | — | 16 | — | 7 |
| `kmeans_pp_select_centers` (new) | coclustering.rs | — | 15 | — | 5 |
| `kmeans_iterate` (new) | coclustering.rs | — | 6 | — | 3 |
| `recompute_centroids` (new) | coclustering.rs | — | 9 | — | 6 |
| `reassign_to_centroids` (new) | coclustering.rs | — | 8 | — | 5 |
| `compute_d2_matrix` (new) | kernel_kmeans.rs | — | 8 | — | 5 |
| `argmin_reassign` (new) | kernel_kmeans.rs | — | 6 | — | 4 |
| `build_aligned_members` (new) | kshape.rs | — | 1 | — | 2 |
| `build_xtx` (new) | kshape.rs | — | 13 | — | 5 |
| `apply_q_centering` (new) | kshape.rs | — | 13 | — | 9 |
| `compute_shape_matrix` (new) | kshape.rs | — | 0 | — | 1 |

## Task Commits

1. **Task 1: coclustering.rs m_step + col_kmeans_init** — `cbbbbfac` (refactor)
2. **Task 2: coclustering.rs e_col_step + e_row_step** — `bdfd6538` (refactor)
3. **Task 3: kernel_kmeans.rs run_restart + kshape.rs shape_extraction + clippy gate** — `4271a4d1` (refactor)

## Files Created/Modified

- `fdars-core/src/coclustering.rs` — 10 new private helpers extracted
- `fdars-core/src/kernel_kmeans.rs` — compute_d2_matrix + argmin_reassign extracted
- `fdars-core/src/kshape.rs` — 4 new private helpers extracted

## Decisions Made

- **Three-seam for m_step (cog 56):** Gaussian estimation needed block_mean_pass + block_var_pass extracted via an intermediate m_step_block_params; single seam left m_step_block_params at cog 27 (over threshold), requiring a second pass
- **Self-flagging helpers:** Both kmeans_iterate (cog 32) and m_step_block_params (cog 27) self-flagged and needed further decomposition — recompute_centroids + reassign_to_centroids from kmeans_iterate; block_mean_pass + block_var_pass from m_step_block_params
- **RNG draw order preserved:** kmeans_pp_select_centers receives `&mut StdRng` and calls gen_range then gen::<f64>() in exactly the same order as the original inlined code; run_restart likewise
- **EM accumulation order preserved:** col_candidate_gain lifts the inner i-loop body verbatim; FP summation order unchanged
- **kshape shape_extraction:** needed three helpers (build_xtx, apply_q_centering, compute_shape_matrix) because the QSQ centering block alone (apply_q_centering) sat at cog 13 — within threshold; compute_shape_matrix became a trivial two-line delegator

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clippy ptr_arg lint in kmeans_iterate**
- **Found during:** Task 3 (clippy gate)
- **Issue:** `kmeans_iterate` used `&mut Vec<usize>` instead of `&mut [usize]`; clippy flags under `-D warnings`
- **Fix:** Changed parameter type to `&mut [usize]`; call site in `col_kmeans_init` unchanged (Rust coerces automatically)
- **Files modified:** `fdars-core/src/coclustering.rs`
- **Committed in:** `4271a4d1` (Task 3 commit)

**2. [Rule 2 - Self-flagging helpers] kmeans_iterate (cog 32) and m_step_block_params (cog 27) exceeded threshold**
- **Found during:** Task 1 (mid-extraction pmat check)
- **Issue:** First extraction left two intermediate helpers above cog 23
- **Fix:** Split kmeans_iterate into recompute_centroids + reassign_to_centroids; split m_step_block_params into block_mean_pass + block_var_pass
- **Impact:** Added 4 extra helpers beyond plan estimate; all below threshold

---

**Total deviations:** 2 (Rule 1 auto-fix, Rule 2 self-flagging split)

## Known Stubs

None.

## Self-Check: PASSED

- `fdars-core/src/coclustering.rs`: exists, modified
- `fdars-core/src/kernel_kmeans.rs`: exists, modified
- `fdars-core/src/kshape.rs`: exists, modified
- Commits `cbbbbfac`, `bdfd6538`, `4271a4d1`: verified in git log
- 34/34 module tests pass
- clippy `--all-targets --features linalg,parallel -D warnings` clean

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
