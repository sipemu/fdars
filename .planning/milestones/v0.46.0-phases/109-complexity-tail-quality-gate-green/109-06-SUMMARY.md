---
phase: 109-complexity-tail-quality-gate-green
plan: "06"
subsystem: helpers
tags: [pmat, complexity, refactor, interpolation, gradient, shared-infra]

requires: ["109-01"]
provides:
  - "helpers.rs: spline_interpolate below cog 23 / cyc 30 (19/9)"
  - "helpers.rs: spline_interpolate_with_policy below cog 23 / cyc 30 (16/9)"
  - "helpers.rs: gradient_nonuniform below cog 23 / cyc 30 (17/10)"
  - "helpers.rs: fdata_interpolate_with_policy below cog 23 / cyc 30 (13/8)"
affects: []

actuals:
  tokens: 18200
  tasks: 2
  commits: 2
  plan_head_before: dd36f006e08ae0ccce2ca428dbd76e58f9bfa55e

tech-stack:
  added: []
  patterns:
    - "inner-loop extraction: gradient_nonuniform_interior extracted verbatim from gradient_nonuniform"
    - "per-curve helper + policy-dispatch helper: fdata_interpolate_with_policy split into interp_curve_with_policy + interp_at_point"
    - "spline eval helper: spline_eval_curve (plain) + spline_eval_curve_masked (fill-aware)"
    - "query-mapping helper: spline_map_queries for OOB policy dispatch in spline_interpolate_with_policy"
    - "all helpers private fn; no public/pub(crate) surface change"

key-files:
  created: []
  modified:
    - fdars-core/src/helpers.rs

key-decisions:
  - "spline_interpolate (cog 24): extract spline_eval_curve — gathers curve, applies pinv, accumulates output; single seam drops cog 24->19"
  - "spline_interpolate_with_policy (cog 40): two seams — spline_map_queries (OOB policy dispatch loop, cog 12) + spline_eval_curve_masked (per-curve eval with fill support, cog 9); drops cog 40->16"
  - "gradient_nonuniform (cog 27): extract gradient_nonuniform_interior (interior Lagrange loop, cog 10); boundary handling stays in parent; drops cog 27->17"
  - "fdata_interpolate_with_policy (cog 44): two seams — interp_at_point (single-point in-range/OOB dispatch, cog 11) + interp_curve_with_policy (per-curve loop, cog 1); drops cog 44->13"
  - "FP-accumulation order verbatim throughout; loop bodies lifted exactly — no sums reordered"
  - "interp_curve_with_policy returns Result<(), FdarError> so the Exception path can early-return through it correctly"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "spline_interpolate drops from cog 24 to cog 19; helper spline_eval_curve cog 3"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#spline_interpolate_reproduces_argvals,spline_interpolate_cubic_offgrid,spline_interpolate_rejects_out_of_range,spline_interpolate_rejects_bad_order,spline_interpolate_rejects_dim_mismatch"
        status: pass
    human_judgment: false
  - id: D2
    description: "spline_interpolate_with_policy drops from cog 40 to cog 16; helpers spline_map_queries cog 12, spline_eval_curve_masked cog 9"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_spline_with_policy_*"
        status: pass
    human_judgment: false
  - id: D3
    description: "gradient_nonuniform drops from cog 27 to cog 17; helper gradient_nonuniform_interior cog 10"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_gradient_uniform_linear (shared infrastructure)"
        status: pass
    human_judgment: false
  - id: D4
    description: "fdata_interpolate_with_policy drops from cog 44 to cog 13; helpers interp_curve_with_policy cog 1, interp_at_point cog 11"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_extrapolation_*"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 06: helpers.rs Complexity Tail Summary

**All 4 helpers.rs complexity violations eliminated via private helper extraction — targets now cog <=19 / cyc <=10, all new helpers cog <=12 / cyc <=10, 2942 lib tests + 215 doctests pass, clippy clean**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-09-14T14:00:44Z
- **Completed:** 2026-09-14T14:12:xx Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- `spline_interpolate` cog 24 -> 19, cyc 11 -> 9 (extracted `spline_eval_curve`, cog 3)
- `spline_interpolate_with_policy` cog 40 -> 16, cyc 16 -> 9 (extracted `spline_map_queries` cog 12, `spline_eval_curve_masked` cog 9)
- `gradient_nonuniform` cog 27 -> 17, cyc 14 -> 10 (extracted `gradient_nonuniform_interior` cog 10)
- `fdata_interpolate_with_policy` cog 44 -> 13, cyc 15 -> 8 (extracted `interp_curve_with_policy` cog 1, `interp_at_point` cog 11)
- All new helpers private `fn`; no public/pub(crate) signature or visibility change
- FP-accumulation order verbatim in all extracted loop bodies (interpolation, gradient)
- Full lib suite: 2942 pass (0 failed); doctests: 215 pass; clippy --all-targets clean

## Before → After Scores

| Function | cog before | cog after | cyc before | cyc after |
|---|---|---|---|---|
| `spline_interpolate` | 24 | 19 | 11 | 9 |
| `spline_interpolate_with_policy` | 40 | 16 | 16 | 9 |
| `gradient_nonuniform` | 27 | 17 | 14 | 10 |
| `fdata_interpolate_with_policy` | 44 | 13 | 15 | 8 |
| `spline_eval_curve` (new) | — | 3 | — | 3 |
| `spline_map_queries` (new) | — | 12 | — | 5 |
| `spline_eval_curve_masked` (new) | — | 9 | — | 5 |
| `gradient_nonuniform_interior` (new) | — | 10 | — | 5 |
| `interp_curve_with_policy` (new) | — | 1 | — | 2 |
| `interp_at_point` (new) | — | 11 | — | 7 |

## Task Commits

1. **Task 1: spline_interpolate + spline_interpolate_with_policy** — `fc94de10`
2. **Task 2: gradient_nonuniform + fdata_interpolate_with_policy + full-suite gate** — `56d7f418`

## Decisions Made

- **spline_interpolate single-seam:** Extracting `spline_eval_curve` (gather curve, pinv*y, accumulate) reduced cog 24->19 in one extraction
- **spline_interpolate_with_policy two-seam:** Query-mapping loop extracted as `spline_map_queries` (returns `Result<(Vec<f64>, Vec<bool>), FdarError>`) and per-curve evaluation with fill-mask extracted as `spline_eval_curve_masked`; `fill_val` pre-computed once outside the per-curve loop
- **gradient_nonuniform single-seam:** Interior Lagrange loop extracted as `gradient_nonuniform_interior`; left/right boundary handling kept in parent (they're brief conditional blocks)
- **fdata_interpolate_with_policy two-seam:** `interp_at_point` dispatches a single point (all policy/method branches); `interp_curve_with_policy` is the thin per-curve loop that calls it. `interp_curve_with_policy` returns `Result<(), FdarError>` so Exception early-returns propagate correctly through `?`
- **No reuse of spline_eval_curve in with_policy path:** The masked variant needs fill-mask logic, so a separate `spline_eval_curve_masked` is cleaner

## Deviations from Plan

None — all four targets dropped below threshold in first attempt. No helper self-flagged. Full-suite confirmed 2942 + 215 with no failures.

## Self-Check: PASSED

- SUMMARY.md: FOUND at `.planning/phases/109-complexity-tail-quality-gate-green/109-06-SUMMARY.md`
- Commits: FOUND `fc94de10` (Task 1), `56d7f418` (Task 2)
- All 4 target functions at or below cog 19 / cyc 10
- No new helper exceeds cog 12 / cyc 5
- 2942 lib tests + 215 doctests: green
- clippy --all-targets: clean
