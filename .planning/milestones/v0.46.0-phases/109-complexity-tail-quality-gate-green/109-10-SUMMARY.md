---
phase: 109-complexity-tail-quality-gate-green
plan: "10"
subsystem: gmm/density/conformal
tags: [pmat, complexity, refactor, gmm, density, conformal, em, wasserstein]

requires:
  - "109-01 (tracer — proven extract-pmat-commit loop)"

provides:
  - "gmm/subspace.rs: run_one_em + fun_hddc_cluster below cog 23 / cyc 30"
  - "density_fda.rs: inverse_lqd + wasserstein_barycenter below cog 23 / cyc 30"
  - "conformal/cv.rs: cv_conformal_regression + cv_conformal_classification below cog 23 / cyc 30"

affects: []

actuals:
  tokens: 38000
  tasks: 3
  commits: 3
  plan_head_before: 4d37e1a72f6eaf0963742800715a6d914cc9fcc9

tech-stack:
  added: []
  patterns:
    - "EM decomposition: init_em_params + m_step_means_weights + m_step_update_subspace — lifts EM body verbatim, seeded RNG draw order preserved"
    - "validation extraction: validate_hddc_inputs / validate_inverse_lqd_inputs / validate_cv_regression_inputs / validate_cv_classif_inputs — guard blocks lifted verbatim"
    - "loop extraction: compute_q_bar / run_cv_regression_fold — inner loops lifted verbatim, FP accumulation order preserved"
    - "too_many_arguments allow retained on run_cv_regression_fold (fold-runner needs all context)"

key-files:
  created: []
  modified:
    - fdars-core/src/gmm/subspace.rs
    - fdars-core/src/density_fda.rs
    - fdars-core/src/conformal/cv.rs

key-decisions:
  - "gmm/subspace.rs: three-seam decomposition for run_one_em (cog 68): init_em_params (init block), m_step_means_weights (M-step part 1), m_step_update_subspace (M-step part 2 per cluster); EM iteration structure and seeded RNG draw order preserved verbatim"
  - "gmm/subspace.rs: single-seam for fun_hddc_cluster (cog 27): validate_hddc_inputs lifts all 6 guard checks; main fn drops to cog 8"
  - "density_fda.rs: single-seam for inverse_lqd (cog 24): validate_inverse_lqd_inputs lifts 6 guards; computation chain left intact; FP accumulation order preserved"
  - "density_fda.rs: two-seam for wasserstein_barycenter (cog 49): resolve_barycenter_weights (weight resolution/validation) + compute_q_bar (quantile-average loop); Q-inversion + renormalization kept inline"
  - "conformal/cv.rs: two-seam for cv_conformal_regression (cog 34): validate_cv_regression_inputs (4 guards) + run_cv_regression_fold (per-fold fit+accumulate); CV fold order preserved"
  - "conformal/cv.rs: single-seam for cv_conformal_classification (cog 28): validate_cv_classif_inputs (5 guards + n_classes resolution); fold loop already small after prior extractions"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "run_one_em (gmm/subspace.rs) drops from cog 68 to cog 10, cyc 24->5"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/gmm/subspace.rs#test_funhddC_recovery,test_funhddC_bic_finite,test_funhddC_deterministic"
        status: pass
    human_judgment: false
  - id: D2
    description: "fun_hddc_cluster (gmm/subspace.rs) drops from cog 27 to cog 8, cyc 13->6"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/gmm/subspace.rs#test_funhddC_recovery,test_funhddC_bic_finite,test_funhddC_deterministic"
        status: pass
    human_judgment: false
  - id: D3
    description: "inverse_lqd (density_fda.rs) drops from cog 24 to cog 6, cyc 9->3"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/density_fda.rs#round_trip_lqd_density_within_tolerance,inverse_lqd_normalized_nonneg"
        status: pass
    human_judgment: false
  - id: D4
    description: "wasserstein_barycenter (density_fda.rs) drops from cog 49 to cog 18, cyc 16->7"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/density_fda.rs#barycenter_singleton_reduction,barycenter_two_density_midpoint,barycenter_normalized_nonneg"
        status: pass
    human_judgment: false
  - id: D5
    description: "cv_conformal_regression (conformal/cv.rs) drops from cog 34 to cog 6, cyc 14->4"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "conformal cv compilation + 4 doctests green"
        status: pass
    human_judgment: false
  - id: D6
    description: "cv_conformal_classification (conformal/cv.rs) drops from cog 28 to cog 15, cyc 12->7"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "conformal cv compilation + 4 doctests green"
        status: pass
    human_judgment: false

duration: 9min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 10: GMM / Density / Conformal Complexity Tail Summary

**Six complexity violations eliminated across three files via private helper extraction — all targets now cog ≤19 / cyc ≤10, all new helpers cog ≤19 / cyc ≤8, 2942/2942 lib tests pass, clippy clean**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-09-14T20:10:33Z
- **Completed:** 2026-09-14T20:19:20Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `run_one_em` cog 68->10, cyc 24->5 (extracted `init_em_params`, `m_step_means_weights`, `m_step_update_subspace`)
- `fun_hddc_cluster` cog 27->8, cyc 13->6 (extracted `validate_hddc_inputs`)
- `inverse_lqd` cog 24->6, cyc 9->3 (extracted `validate_inverse_lqd_inputs`)
- `wasserstein_barycenter` cog 49->18, cyc 16->7 (extracted `resolve_barycenter_weights`, `compute_q_bar`)
- `cv_conformal_regression` cog 34->6, cyc 14->4 (extracted `validate_cv_regression_inputs`, `run_cv_regression_fold`)
- `cv_conformal_classification` cog 28->15, cyc 12->7 (extracted `validate_cv_classif_inputs`)
- No new helper exceeds cog 19 / cyc 8 — all well below threshold
- Full 2942/2942 lib tests pass; behavior and numeric output identical
- clippy `--all-targets --features linalg,parallel -D warnings` clean

## Before → After Scores

| Function | File | cog before | cog after | cyc before | cyc after |
|---|---|---|---|---|---|
| `run_one_em` | gmm/subspace.rs | 68 | 10 | 24 | 5 |
| `fun_hddc_cluster` | gmm/subspace.rs | 27 | 8 | 13 | 6 |
| `inverse_lqd` | density_fda.rs | 24 | 6 | 9 | 3 |
| `wasserstein_barycenter` | density_fda.rs | 49 | 18 | 16 | 7 |
| `cv_conformal_regression` | conformal/cv.rs | 34 | 6 | 14 | 4 |
| `cv_conformal_classification` | conformal/cv.rs | 28 | 15 | 12 | 7 |
| `init_em_params` (new) | gmm/subspace.rs | — | 12 | — | 8 |
| `m_step_means_weights` (new) | gmm/subspace.rs | — | 12 | — | 7 |
| `m_step_update_subspace` (new) | gmm/subspace.rs | — | 12 | — | 7 |
| `validate_hddc_inputs` (new) | gmm/subspace.rs | — | 19 | — | 8 |
| `validate_inverse_lqd_inputs` (new) | density_fda.rs | — | 18 | — | 7 |
| `resolve_barycenter_weights` (new) | density_fda.rs | — | 18 | — | 6 |
| `compute_q_bar` (new) | density_fda.rs | — | 13 | — | 5 |
| `validate_cv_regression_inputs` (new) | conformal/cv.rs | — | 13 | — | 6 |
| `run_cv_regression_fold` (new) | conformal/cv.rs | — | 10 | — | 7 |
| `validate_cv_classif_inputs` (new) | conformal/cv.rs | — | 13 | — | 6 |

## Task Commits

1. **Task 1: gmm/subspace.rs decomposition** — `9fd8c51e` (refactor)
2. **Task 2: density_fda.rs decomposition** — `aff14321` (refactor)
3. **Task 3: conformal/cv.rs decomposition + clippy gate** — `90fa5fd5` (refactor)

## Files Created/Modified

- `fdars-core/src/gmm/subspace.rs` — `init_em_params`, `m_step_means_weights`, `m_step_update_subspace`, `validate_hddc_inputs` extracted
- `fdars-core/src/density_fda.rs` — `validate_inverse_lqd_inputs`, `resolve_barycenter_weights`, `compute_q_bar` extracted
- `fdars-core/src/conformal/cv.rs` — `validate_cv_regression_inputs`, `run_cv_regression_fold`, `validate_cv_classif_inputs` extracted

## Decisions Made

- **Three-seam for run_one_em (cog 68):** The EM loop body had three natural seams — initialization, M-step means/weights, and M-step per-cluster subspace update. Each extracted helper received exactly the data it needed; `&mut StdRng` init draw order preserved by keeping seeded init in the caller
- **Single-seam for fun_hddc_cluster (cog 27):** Six validation guards formed the only seam; extracted as `validate_hddc_inputs`; main fn dropped to cog 8 in one pass
- **Single-seam for inverse_lqd (cog 24):** Six input validation guards extracted as `validate_inverse_lqd_inputs`; the computation chain (steps 1-6) kept inline as it is a single linear flow
- **Two-seam for wasserstein_barycenter (cog 49):** Weight resolution (`resolve_barycenter_weights`) and quantile-average accumulation loop (`compute_q_bar`) are natural seams; Q-inversion + renorm kept inline
- **Two-seam for cv_conformal_regression (cog 34):** Validation block and per-fold execution are natural seams; fold order and CV residual accumulation index preserved
- **Single-seam for cv_conformal_classification (cog 28):** Validation + n_classes resolution extracted; fold-delegation to `conformal_classif_fold` already existed, reducing remaining complexity

## Deviations from Plan

None — plan executed exactly as written. All six targets dropped below threshold in first attempt; no helper self-flagged.

## Issues Encountered

None — all six targets dropped below threshold in first attempt.

## Known Stubs

None.

## Threat Flags

None — all changes are private implementation helpers; no public API surface added or modified.

## Self-Check: PASSED

- `fdars-core/src/gmm/subspace.rs` — exists, modified
- `fdars-core/src/density_fda.rs` — exists, modified
- `fdars-core/src/conformal/cv.rs` — exists, modified
- Task commits: `9fd8c51e`, `aff14321`, `90fa5fd5` — all present in git log
- 2942 lib tests pass
- clippy --all-targets clean
- All six targets and all 10 new helpers below cog 23 / cyc 30

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
