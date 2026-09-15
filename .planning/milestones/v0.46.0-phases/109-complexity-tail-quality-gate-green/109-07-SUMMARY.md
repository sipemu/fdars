---
phase: 109-complexity-tail-quality-gate-green
plan: "07"
subsystem: scalar_on_function
tags: [complexity, pmat, refactor, cognitive-complexity, coordinate-descent, cross-validation, backfitting]

requires:
  - phase: 109-01
    provides: tracer decomposition patterns (two-seam, split-validate, inline)

provides:
  - Private-helper decomposition of all 7 scalar_on_function complexity violations (QUAL-03)
  - multi.rs: fregre_lm_multi/predict/cv below cog 23 (Task 1, prior commit e113e2f6)
  - additive.rs: fregre_gkam/select_group_lasso_lambda/group_lasso_cd below cog 23
  - fregre_lm.rs: fregre_cv below cog 23

affects: [109-08, 109-09, 110-pmat-ci-guardrail]

actuals:
  tokens: 26000
  tasks: 3
  commits: 2
  plan_head_before: e113e2f6

tech-stack:
  added: []
  patterns:
    - split-validate: gkam_validate_predictors extracted to bring gkam_validate below threshold
    - fold-helper extraction: eval_lambda_cv_fold / oof_fold_predict isolate per-fold logic from outer grid/selection loops
    - CD group helpers: cd_group_partial / cd_group_ols_soft / cd_scalar_partial_j decompose coordinate-descent sweep
    - ptr_arg auto-fix: gkam_backfit_sweep signature changed to &mut [Vec<f64>] per clippy::ptr_arg

key-files:
  created: []
  modified:
    - fdars-core/src/scalar_on_function/multi.rs (Task 1 — prior commit e113e2f6)
    - fdars-core/src/scalar_on_function/additive.rs (Tasks 2: gkam + group_lasso decomposition)
    - fdars-core/src/scalar_on_function/fregre_lm.rs (Task 3: fregre_cv decomposition)

key-decisions:
  - "gkam_validate split into gkam_validate_predictors (per-predictor loop, cog 11) + gkam_validate (top-level guards, cog 15) — split-validate pattern; per-predictor shape loop drove gkam_validate above threshold after initial extraction"
  - "gkam_backfit_sweep takes &mut [Vec<f64>] not &mut Vec<Vec<f64>> — clippy::ptr_arg auto-fixed (Rule 3)"
  - "eval_lambda_cv_fold extracts fold build + group_lasso_cd call + val prediction from select_group_lasso_lambda outer grid loop"
  - "oof_fold_predict extracts per-fold train/fit/predict from fregre_cv OOF accumulation loop; fold order preserved verbatim"
  - "CD sweep decomposed: cd_group_partial (partial residual), cd_group_ols_soft (OLS+ridge+soft-threshold), cd_scalar_partial_j (scalar OLS residual) — all lifted verbatim, FP accumulation order unchanged"

patterns-established:
  - "fold-helper pattern: extract train-split build + model fit + error accumulation into a single helper; outer function reduces to a loop calling the helper"
  - "CD helper pattern: separate partial-residual computation from OLS-solve-and-threshold; both helpers are pure (no mutable state except return)"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "multi.rs: fregre_lm_multi (cog 6), predict_fregre_lm_multi (cog 21), fregre_lm_multi_cv (cog 16) all below cog 23 / cyc 30"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "scalar_on_function::multi::tests — 5 tests"
        status: pass
      - kind: other
        ref: "pmat analyze complexity fdars-core/src/scalar_on_function/multi.rs — all targets reported"
        status: pass
    human_judgment: false
  - id: D2
    description: "additive.rs: fregre_gkam (cog 11), select_group_lasso_lambda (cog 8), group_lasso_cd (cog 17) all below cog 23 / cyc 30; all helpers below threshold"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "scalar_on_function::additive::tests — 23 tests"
        status: pass
      - kind: other
        ref: "pmat analyze complexity fdars-core/src/scalar_on_function/additive.rs — top function cog 21 (variable_selection)"
        status: pass
    human_judgment: false
  - id: D3
    description: "fregre_lm.rs: fregre_cv (cog 14), oof_fold_predict (cog 7) both below cog 23 / cyc 30; clippy clean"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "scalar_on_function::tests::test_fregre_cv_returns_result, test_fregre_cv_with_scalar_covariates"
        status: pass
      - kind: other
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings — clean"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 07: scalar_on_function Complexity Tail Summary

**Seven scalar_on_function complexity violations decomposed via private helpers — fregre_gkam (58→11), select_group_lasso_lambda (54→8), group_lasso_cd (67→17), fregre_cv (26→14) — all below pmat cog 23 / cyc 30, 2942 tests green, CD/CV order byte-identical**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-09-14T15:00:00Z
- **Completed:** 2026-09-14T15:30:00Z
- **Tasks:** 3 (Task 1 already committed before this session)
- **Files modified:** 3
- **Commits (this session):** 2 (ed8022c6, 0b8efccb)

## Accomplishments

- multi.rs (prior): fregre_lm_multi cog 50→6, predict_fregre_lm_multi cog 30→21, fregre_lm_multi_cv cog 38→16
- additive.rs: fregre_gkam cog 58→11 via gkam_validate_predictors/gkam_validate/gkam_scalar_dist_col/gkam_adjusted/gkam_nw_from_dists/gkam_max_delta/gkam_backfit_sweep (7 new helpers)
- additive.rs: select_group_lasso_lambda cog 54→8 via gl_build_train_scores/gl_build_train_sc/eval_lambda_cv_fold (3 new helpers)
- additive.rs: group_lasso_cd cog 67→17 via cd_group_partial/cd_group_ols_soft/cd_scalar_partial_j (3 new helpers)
- fregre_lm.rs: fregre_cv cog 26→14 via oof_fold_predict (1 new helper)
- Full 2942-test suite green; clippy --all-targets clean; fmt clean

## Task Commits

1. **Task 1: multi.rs** — `e113e2f6` (prior session, refactor)
2. **Task 2: additive.rs** — `ed8022c6` (refactor)
3. **Task 3: fregre_lm.rs + clippy fix** — `0b8efccb` (refactor)

## Files Created/Modified

- `fdars-core/src/scalar_on_function/multi.rs` — fregre_lm_multi/predict/cv decomposed (prior commit)
- `fdars-core/src/scalar_on_function/additive.rs` — gkam + group_lasso decomposed (13 new helpers)
- `fdars-core/src/scalar_on_function/fregre_lm.rs` — fregre_cv decomposed (1 new helper)

## Decisions Made

- gkam_validate split into gkam_validate_predictors (per-predictor loop) + gkam_validate (top-level guards) to keep both below cog 23 after initial extraction scored cog 26
- gkam_backfit_sweep signature changed to `&mut [Vec<f64>]` (clippy::ptr_arg auto-fix)
- CD sweep helpers lift loop bodies verbatim — no FP accumulation reorder
- oof_fold_predict extracts the per-fold train/fit/predict block from fregre_cv; fold order is verbatim
- eval_lambda_cv_fold (single fold for one lambda) reduces select_group_lasso_lambda's outer grid loop to a simple accumulate

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] clippy::ptr_arg in gkam_backfit_sweep**
- **Found during:** Task 3 clippy gate
- **Issue:** `&mut Vec<Vec<f64>>` instead of `&mut [Vec<f64>]` triggers `-D warnings`
- **Fix:** Changed parameter type to `&mut [Vec<f64>]` — no behavior change, slice deref is equivalent
- **Files modified:** fdars-core/src/scalar_on_function/additive.rs
- **Verification:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean
- **Committed in:** 0b8efccb (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 3 — blocking clippy warning)
**Impact on plan:** Auto-fix was a one-line signature change; no scope creep.

## Issues Encountered

- gkam_validate initially scored cog 26 after extracting the per-predictor loop body — needed a second split (split-validate pattern, consistent with Phase 109-01 TRACER decision); resolved with gkam_validate_predictors

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All 7 Plan-07 QUAL-03 targets below cog 23 / cyc 30
- Full test suite (2942 tests) green; clippy clean; fmt clean
- Ready for Phase 109-08 or remaining complexity tail work toward pmat quality-gate PASS

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
