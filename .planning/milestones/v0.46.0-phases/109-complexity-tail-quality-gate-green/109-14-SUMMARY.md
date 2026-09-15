---
phase: 109
plan: 14
subsystem: fdars-core/complexity
tags: [complexity, refactor, pmat, quality-gate]
status: complete
completed: "2026-09-15"
depends_on: ["109-13"]
requirements: []

dependency_graph:
  requires: ["109-13"]
  provides: ["pmat-quality-gate-green"]
  affects: ["fdars-core/src/explain_generic", "fdars-core/src/frechet", "fdars-core/src/inference", "fdars-core/src/shapelet", "fdars-core/src/simulation.rs", "fdars-core/src/fem_smoothing.rs"]

tech_stack:
  added: []
  patterns:
    - "Private helper extraction — no visibility changes, no new dependencies"
    - "Split cognitive complexity by factoring validation, loop bodies, and post-processing blocks"

key_files:
  created: []
  modified:
    - fdars-core/src/explain_generic/pdp.rs
    - fdars-core/src/explain_generic/shap.rs
    - fdars-core/src/explain_generic/stability.rs
    - fdars-core/src/frechet/regression.rs
    - fdars-core/src/inference/anova.rs
    - fdars-core/src/inference/flm.rs
    - fdars-core/src/irreg_fdata/face.rs
    - fdars-core/src/optimal_design.rs
    - fdars-core/src/outliers.rs
    - fdars-core/src/pda.rs
    - fdars-core/src/regression.rs
    - fdars-core/src/shapelet/discovery.rs
    - fdars-core/src/shapelet/distance.rs
    - fdars-core/src/simulation.rs
    - fdars-core/src/fem_smoothing.rs

decisions:
  - "Extracted private helpers only — no pub fn, no struct, no new crate dependency"
  - "Preserved RNG draw order in fvarma_core: eps_t generation stays first in loop body; helpers are pure fn (no rng param)"
  - "Preserved RNG draw order in discover_shapelets: generate_candidates seeded draw order unchanged; greedy_select_shapelets is post-scoring"
  - "Used &mut [T] instead of &mut Vec<T> in all helper signatures (clippy ptr_arg rule)"
  - "update_histories takes &[f64] for x_new (borrow), clones internally — matches original insert(0, x_new.clone()) semantics"
  - "validate_and_resolve + remap_labels split needed to get validate_and_resolve from cog 24 to cog 21"
  - "entropy check intentionally excluded from pmat quality-gate per plan specification"

metrics:
  duration: "~2 sessions (resumed after context compaction)"
  completed: "2026-09-15"

actuals:
  tokens: 74000
  tasks: 4
  commits: 4
  plan_head_before: "de8e993e0ef475c11800c553cd0387fc0b3c3ce8"
---

# Phase 109 Plan 14: Complexity Tail Summary

Decomposed the 15 remaining high-complexity functions across 15 files into private helpers, driving every function and helper to cognitive complexity ≤ 23 and cyclomatic complexity ≤ 30. The final `pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation` reports **0 violations**. This is the FINAL plan of milestone v0.46.0.

## What Was Built

Private helper extraction across 15 files in 4 batches, reducing 15 flagged functions from their original cognitive complexity scores to safe levels while preserving all numeric outputs including seeded-RNG draw order:

| File | Function | Cog Before | Cog After | Helpers Added |
|------|----------|-----------|-----------|---------------|
| explain_generic/pdp.rs | generic_pdp | 26 | 13 | fill_ice_row |
| explain_generic/shap.rs | generic_shap_values | 25 | 16 | shap_row_for_obs, accumulate_shap_samples |
| explain_generic/stability.rs | generic_stability | 38 | 18 | boot_regression, boot_logistic |
| frechet/regression.rs | compute_global_weights | 24 | 12 | compute_covariance_chol |
| inference/anova.rs | oneway_anova_vstat | 26 | 20 | compute_between_ss, compute_within_cov_diag |
| inference/flm.rs | flm_gof_test | 27 | 21 | accumulate_normal_equations |
| irreg_fdata/face.rs | mface_covariance | 38 | 22 | fill_diag_blocks, fill_offdiag_blocks |
| optimal_design.rs | score_criterion | 35 | 17 | posterior_covariance, eval_optimality_criterion |
| outliers.rs | seq_transform_apply | 34 | 1 | apply_t1, apply_t2, apply_diff |
| pda.rs | principal_differential_analysis | 41 | 13 | compute_derivatives, solve_pointwise_pda |
| regression.rs | fdata_to_pc | 27 | 18 | scale_by_sqrt_weights, unscale_rotation_by_sqrt_weights |
| shapelet/discovery.rs | discover_shapelets | 40 | 3 | remap_labels, validate_and_resolve, greedy_select_shapelets |
| shapelet/distance.rs | shapelet_distance | 27 | 17 | window_sq_dist |
| simulation.rs | fvarma_core | 52 | 14 | apply_ar_term, apply_ma_term, update_histories |
| fem_smoothing.rs | fem_smooth | 41 | 15 | build_phi_matrices, compute_a_inv_edf |

## Gate Results

```
pmat quality-gate --checks complexity,satd,dead-code,duplicates --fail-on-violation
Quality Gate: PASSED
Total violations: 0
```

- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: PASSED
- `cargo test -p fdars-core --features linalg,parallel --lib`: 2942 tests, 0 failures

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy ptr_arg and redundant-ref warnings in helper signatures**
- **Found during:** Task 4 (clippy gate)
- **Issue:** Helpers introduced in Tasks 1 and 2 used `&mut Vec<T>` parameters (clippy flags as `ptr_arg`) and redundant `&` on already-slice variables (`&variables` where `variables: &[...]`). Additionally one `.as_slice()` call on `&mut [f64]` triggered an unstable `str_as_str` feature error on MSRV 1.81.
- **Fix:** Changed `&mut Vec<f64>` → `&mut [f64]` in pdp.rs, shap.rs, face.rs, pda.rs; removed redundant `&` on slice arguments in face.rs; replaced `.as_slice()` with `&*obs_z` in pdp.rs.
- **Files modified:** explain_generic/pdp.rs, explain_generic/shap.rs, irreg_fdata/face.rs, pda.rs
- **Commit:** 96e0ea56

**2. [Rule 2 - Missing] validate_and_resolve needed remap_labels split**
- **Found during:** Task 3 (pmat check after first extraction)
- **Issue:** After extracting `validate_and_resolve` from `discover_shapelets`, the helper measured cog 24 (just over the 23 threshold) because it still contained the label remapping closure + collection.
- **Fix:** Extracted `remap_labels` as a further private helper; `validate_and_resolve` dropped to cog 21.
- **Files modified:** shapelet/discovery.rs
- **Commit:** 551c62a3

## RNG Preservation Notes

Both RNG-sensitive functions were refactored safely:

- **fvarma_core** (`simulation.rs`): The `eps_t` draw (`rng.sample::<f64, _>(normal)`) is called once per step at the top of the loop body, before any helper calls. The three new helpers (`apply_ar_term`, `apply_ma_term`, `update_histories`) are pure — they take no `rng` argument and perform no draws. Draw order is byte-identical to the original.

- **discover_shapelets** (`shapelet/discovery.rs`): The seeded candidate generation is handled by `generate_candidates` (unchanged). The new helpers operate only on the already-generated candidate list (`greedy_select_shapelets`, `validate_and_resolve`, `remap_labels`). No RNG state change.

## Self-Check: PASSED

- All 15 source files exist and compile: confirmed (cargo build clean)
- 4 commits in range de8e993e..HEAD: confirmed (`git rev-list --count`)
- pmat gate: 0 violations confirmed
- 2942 lib tests pass: confirmed
- clippy --all-targets: 0 errors confirmed
