---
phase: "109"
plan: "02"
subsystem: alignment
tags: [complexity, refactor, clustering, nd, generative, quality-gate]
status: complete

depends_on: ["109-01"]
provides: ["clustering.rs clean", "nd.rs clean", "generative.rs clean"]
affects: ["alignment module complexity budget"]

tech_stack:
  patterns: ["private helper extraction", "seam decomposition", "validation helper split"]

key_files:
  modified:
    - fdars-core/src/alignment/clustering.rs
    - fdars-core/src/alignment/nd.rs
    - fdars-core/src/alignment/generative.rs

decisions:
  - "build_centered_stacked shared between karcher_covariance_nd and pca_nd — DRY and keeps both functions below threshold"
  - "update_medoids signature changed from &mut Vec<usize> to &mut [usize] to satisfy clippy ptr_arg (Rule 1 auto-fix)"
  - "reconstruct_curve_from_aug_srsf and phase_shooting_to_warp shared between gauss_model and joint_gauss_model — natural shared boundary"
  - "gauss_validate consolidates four identical guard checks across both generative functions"

metrics:
  duration_minutes: 11
  completed_date: "2026-09-14"
  tasks_completed: 3
  tasks_total: 3

actuals:
  tokens: 67000
  tasks: 3
  commits: 3
  plan_head_before: "435f0e46505dca965911196b570036a24b179969"
---

# Phase 109 Plan 02: Alignment Clustering/Nd/Generative Complexity Summary

Decomposed eight pmat complexity violations across `alignment/clustering.rs`, `alignment/nd.rs`, and `alignment/generative.rs` into private helpers. Every public function and every new helper is cognitive ≤18 / cyclomatic ≤10, well inside the cog ≤23 / cyc ≤30 threshold.

## Results by Target

### clustering.rs

| Function | Before cog | After cog | Before cyc | After cyc |
|---|---|---|---|---|
| kmeans_pp_init | 32 | 8 | 9 | 5 |
| kmedoids_from_distances | 32 | 16 | 11 | 7 |
| hierarchical_from_distances | 53 | 4 | 16 | 4 |

New private helpers (all ≤18 cog / ≤6 cyc):
- `kmeans_pp_fallback_center` (cog 6, cyc 3)
- `kmeans_pp_weighted_draw` (cog 6, cyc 3) — receives `&mut StdRng`, preserves draw order
- `update_medoids` (cog 11, cyc 5)
- `hierarchical_validate` (cog 6, cyc 3)
- `hierarchical_find_min_pair` (cog 18, cyc 6)
- `hierarchical_update_distances` (cog 12, cyc 6)

### nd.rs

| Function | Before cog | After cog | Before cyc | After cyc |
|---|---|---|---|---|
| karcher_mean_nd | 54 | 9 | 24 | 5 |
| karcher_covariance_nd | 30 | 18 | 15 | 8 |
| pca_nd | 44 | 12 | 25 | 8 |

New private helpers (all ≤18 cog / ≤10 cyc):
- `karcher_mean_nd_validate` (cog 16, cyc 8)
- `karcher_mean_nd_align_step` (cog 9, cyc 6)
- `karcher_mean_nd_store_gammas` (cog 3, cyc 3)
- `karcher_mean_nd_recompute_aligned` (cog 12, cyc 7)
- `build_centered_stacked` (cog 12, cyc 8) — shared by covariance and pca
- `compute_gram` (cog 6, cyc 4)
- `compute_pca_scores` (cog 3, cyc 3)
- `compute_pca_loadings` (cog 11, cyc 6)

### generative.rs

| Function | Before cog | After cog | Before cyc | After cyc |
|---|---|---|---|---|
| gauss_model | 28 | 13 | 13 | 7 |
| joint_gauss_model | 40 | 3 | 20 | 3 |

New private helpers (all ≤13 cog / ≤6 cyc):
- `gauss_validate` (cog 13, cyc 6) — shared validation for both public functions
- `reconstruct_curve_from_aug_srsf` (cog 1, cyc 2) — shared GH #34 midpoint logic
- `phase_shooting_to_warp` (cog 0, cyc 1) — shared sphere exp-map -> warp conversion
- `build_joint_scores` (cog 5, cyc 4)
- `joint_score_mean_var` (cog 6, cyc 5)
- `sample_joint_z` (cog 1, cyc 2) — consumes one StandardNormal draw per component
- `reconstruct_amplitude` (cog 3, cyc 3)
- `reconstruct_phase_shooting` (cog 5, cyc 4)

## Verification

- All 15 clustering tests pass
- All 5 nd tests pass
- All 5 generative tests pass
- Full suite: 2942 lib tests + 215 doc tests — 0 failures
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean
- `cargo fmt --check` — clean (fmt applied per task)
- pmat: all eight targets and all new helpers ≤ cog 18 / cyc 10

## Commits

| Hash | Description |
|---|---|
| 6a9c4491 | refactor(109-02): decompose clustering.rs complexity violations |
| 197bcb21 | refactor(109-02): decompose nd.rs complexity violations |
| 16d1fc88 | refactor(109-02): decompose generative.rs complexity violations + clippy fix |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] clippy ptr_arg in update_medoids**
- **Found during:** Task 3 (clippy gate)
- **Issue:** `update_medoids` declared `medoids: &mut Vec<usize>` but only mutated elements, not the Vec itself — clippy flags this as unnecessary indirection
- **Fix:** Changed to `&mut [usize]`; call site unchanged (Vec coerces to slice automatically)
- **Files modified:** `fdars-core/src/alignment/clustering.rs`
- **Commit:** 16d1fc88

## Known Stubs

None.

## Self-Check: PASSED

- [x] clustering.rs committed: 6a9c4491
- [x] nd.rs committed: 197bcb21
- [x] generative.rs committed: 16d1fc88
- [x] All 8 targets below cog 23 / cyc 30 (pmat confirmed)
- [x] All helpers below cog 23 / cyc 30 (max: cog 18)
- [x] Full test suite green (2942 lib + 215 doc)
- [x] No public API/signature/visibility change
