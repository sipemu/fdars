---
phase: 109
plan: 13
subsystem: complexity-refactor
tags: [complexity, refactor, pmat, cognitive-complexity]
status: complete

requires: [109-12-SUMMARY.md]
provides: [complexity-tail-clean-files]
affects: [fdars-core/src/boosting_regression/stability.rs, fdars-core/src/concurrent_regression.rs, fdars-core/src/conformal/elastic.rs, fdars-core/src/cv.rs, fdars-core/src/depth/half_region.rs, fdars-core/src/depth/tvd.rs, fdars-core/src/elastic_regression/logistic.rs, fdars-core/src/function_on_scalar_2d.rs]

tech-stack:
  added: []
  patterns: [private-helper-extraction, seam-decomposition, behavior-preserving-refactor]

key-files:
  modified:
    - fdars-core/src/boosting_regression/stability.rs
    - fdars-core/src/concurrent_regression.rs
    - fdars-core/src/conformal/elastic.rs
    - fdars-core/src/cv.rs
    - fdars-core/src/depth/half_region.rs
    - fdars-core/src/depth/tvd.rs
    - fdars-core/src/elastic_regression/logistic.rs
    - fdars-core/src/function_on_scalar_2d.rs

decisions:
  - Extracted validate_stability_inputs + one_resample from stability_selection (cog 30 → 6); Fisher-Yates RNG draw order preserved via seed+b parameters
  - Extracted check_predictor_shapes + validate_concurrent_inputs + compute_ols_column + compute_concurrent_fitted from concurrent_regression (cog 54 → 9); OLS column solve lifted verbatim
  - Extracted project_vert_pca + project_horiz_pca + project_joint_pca from rebuild_elastic_pcr_model (cog 28 → 1); pure match dispatch
  - Extracted run_one_fold from cv_fdata_with_metrics (cog 25 → 12); fold accumulation order preserved
  - Extracted containment_flags (#[inline]) from half_region_depth_1d (cog 31 → 13)
  - Extracted validate_tvd_inputs + mss_for_curve + build_rank_matrix + build_shape_arrays from total_variation_depth_1d (cog 29 → 2)
  - Extracted validate_contiguous_labels + row_normalise + row_argmax_classes from elastic_multinomial (cog 35 → 14); predict_elastic_multinomial also simplified
  - Extracted validate_fosr_2d_inputs + compute_ols_beta + eval_gcv_candidate + apply_2d_smoothing from fosr_2d/select_lambdas_gcv (cog 33 → 8, cog 49 → 10)
  - Fixed clippy ptr_arg lint in run_one_fold (Vec<f64> → &mut [f64])

metrics:
  duration_minutes: 60
  completed_date: "2026-09-15T18:13:08Z"
  tasks: 3
  commits: 3
  plan_head_before: "46196215d733e2a212da30fd64c3d0c3800c3ee2"

actuals:
  tokens: 32000
  tasks: 3
  commits: 3
---

# Phase 109 Plan 13: Complexity Tail Summary

Decomposed 9 flagged functions across 8 files using private helper extraction. All 9 plan targets and all new helpers now satisfy cognitive ≤23 and cyclomatic ≤30 under pmat 3.18.0. Refactoring is behavior-preserving (numeric-identical) with no public API/signature/visibility changes.

## Tasks Completed

| Task | Files | Commits | Key Reductions |
|------|-------|---------|---------------|
| 1 | stability.rs, concurrent_regression.rs, conformal/elastic.rs | 5049e155 | cog 30→6, 54→9, 28→1 |
| 2 | cv.rs, depth/half_region.rs, depth/tvd.rs | 835b68c7 | cog 25→12, 31→13, 29→2 |
| 3 | elastic_regression/logistic.rs, function_on_scalar_2d.rs | 73f43029 | cog 35→14, 49→10, 33→8 |

## Final pmat Scores (All Targets)

| Function | File | Before (cog) | After (cog) | After (cyc) |
|----------|------|-------------|-------------|-------------|
| stability_selection | boosting_regression/stability.rs | 30 | 6 | 4 |
| concurrent_regression | concurrent_regression.rs | 54 | 9 | 6 |
| rebuild_elastic_pcr_model | conformal/elastic.rs | 28 | 1 | 2 |
| cv_fdata_with_metrics | cv.rs | 25 | 12 | 7 |
| half_region_depth_1d | depth/half_region.rs | 31 | 13 | 8 |
| total_variation_depth_1d | depth/tvd.rs | 29 | 2 | 3 |
| elastic_multinomial | elastic_regression/logistic.rs | 35 | 14 | 9 |
| select_lambdas_gcv | function_on_scalar_2d.rs | 49 | 10 | 5 |
| fosr_2d | function_on_scalar_2d.rs | 33 | 8 | 8 |

## New Helpers (All ≤ cog 23, cyc 30)

| Helper | File | cog | cyc |
|--------|------|-----|-----|
| validate_stability_inputs | boosting_regression/stability.rs | 20 | 9 |
| one_resample | boosting_regression/stability.rs | 4 | 4 |
| check_predictor_shapes | concurrent_regression.rs | 11 | 4 |
| validate_concurrent_inputs | concurrent_regression.rs | 19 | 8 |
| compute_ols_column | concurrent_regression.rs | 9 | 6 |
| compute_concurrent_fitted | concurrent_regression.rs | 6 | 4 |
| project_vert_pca | conformal/elastic.rs | 6 | 4 |
| project_horiz_pca | conformal/elastic.rs | 6 | 4 |
| project_joint_pca | conformal/elastic.rs | 6 | 4 |
| run_one_fold | cv.rs | 8 | 6 |
| containment_flags | depth/half_region.rs | 12 | 6 |
| validate_tvd_inputs | depth/tvd.rs | 12 | 7 |
| mss_for_curve | depth/tvd.rs | 1 | 2 |
| build_rank_matrix | depth/tvd.rs | 5 | 4 |
| build_shape_arrays | depth/tvd.rs | 5 | 4 |
| validate_contiguous_labels | elastic_regression/logistic.rs | 9 | 4 |
| row_normalise | elastic_regression/logistic.rs | 9 | 5 |
| row_argmax_classes | elastic_regression/logistic.rs | 3 | 3 |
| validate_fosr_2d_inputs | function_on_scalar_2d.rs | 15 | 6 |
| compute_ols_beta | function_on_scalar_2d.rs | 3 | 3 |
| eval_gcv_candidate | function_on_scalar_2d.rs | 13 | 6 |
| apply_2d_smoothing | function_on_scalar_2d.rs | 3 | 3 |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical Functionality] Clippy ptr_arg lint in run_one_fold**
- **Found during:** Task 3 clippy gate
- **Issue:** `run_one_fold` parameter `oof_preds: &mut Vec<f64>` triggers clippy::ptr_arg — should be `&mut [f64]` since only slice operations are used inside
- **Fix:** Changed signature to `&mut [f64]`; call-site already passes a `&mut vec[..]` slice so no caller changes needed
- **Files modified:** fdars-core/src/cv.rs
- **Commit:** 73f43029

## Gates

| Gate | Result |
|------|--------|
| cargo fmt --check | PASS |
| cargo clippy --all-targets --features linalg,parallel -D warnings | PASS |
| cargo test --features linalg,parallel (lib) | PASS (2884+ lib tests) |
| cargo test --features linalg,parallel (integration) | PASS (34 integration tests) |
| pmat cognitive ≤23 for all targets | PASS |
| pmat cyclomatic ≤30 for all targets | PASS |

## Commits

| Hash | Description |
|------|-------------|
| 5049e155 | refactor(109-13): decompose stability_selection, concurrent_regression, rebuild_elastic_pcr_model |
| 835b68c7 | refactor(109-13): decompose cv_fdata_with_metrics, half_region_depth_1d, total_variation_depth_1d |
| 73f43029 | refactor(109-13): decompose cv/logistic/fosr_2d into private helpers (cog ≤14) |

## Self-Check: PASSED

- All 8 modified files confirmed present on disk
- All 3 task commits confirmed in git log
- All 9 plan targets confirmed ≤ cog 23 / cyc 30 via pmat
- All new helpers confirmed ≤ cog 23 / cyc 30 via pmat
- Full test suite: 2884+ lib tests + 215 doc tests + 34 integration tests, all passing
