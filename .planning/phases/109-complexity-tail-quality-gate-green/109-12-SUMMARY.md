---
phase: 109-complexity-tail-quality-gate-green
plan: "12"
subsystem: alignment
tags: [pmat, complexity, refactor, elastic, geodesic, lambda-cv, dp-grid, multires, phase-boxplot, warp-stats]

requires:
  - "109-01: proven extract->pmat->commit loop for alignment module"

provides:
  - "alignment/geodesic.rs: curve_geodesic_nd below cog 23 / cyc 30"
  - "alignment/lambda_cv.rs: lambda_cv below cog 23 / cyc 30"
  - "alignment/mod.rs: dp_grid_solve_banded below cog 23 / cyc 30"
  - "alignment/multires.rs: elastic_align_pair_multires below cog 23 / cyc 30"
  - "alignment/phase_boxplot.rs: phase_boxplot below cog 23 / cyc 30"
  - "alignment/warp_stats.rs: warp_statistics below cog 23 / cyc 30"

affects: [109-13, 109-14]

actuals:
  tokens: 32000
  tasks: 2
  commits: 2
  plan_head_before: a0915bf530829a898ce23fcc9485335d74a18864

tech-stack:
  added: []
  patterns:
    - "single-seam decomposition: for cog 21-31, one well-chosen seam (inner loop or step body) drops below threshold"
    - "multi-seam decomposition: phase_boxplot cog 50 required four helpers (central_envelope, whisker_fences, find_outliers, trim_whiskers)"
    - "ptr_arg auto-fix: &mut Vec<T> -> &mut [T] for slice-style helpers (established in 109-01)"

key-files:
  created: []
  modified:
    - fdars-core/src/alignment/geodesic.rs
    - fdars-core/src/alignment/lambda_cv.rs
    - fdars-core/src/alignment/mod.rs
    - fdars-core/src/alignment/multires.rs
    - fdars-core/src/alignment/phase_boxplot.rs
    - fdars-core/src/alignment/warp_stats.rs

key-decisions:
  - "geodesic.rs: extract geodesic_nd_step (all per-point interpolation logic including phase geodesic + per-dim amplitude); FP accumulation order preserved exactly"
  - "lambda_cv.rs: extract lambda_cv_score (all per-lambda fold loop); CV fold order and karcher/elastic_distance call order preserved exactly"
  - "mod.rs: two-step split — dp_fill_grid extracts the nested loops, dp_skip_cell extracts the skip predicate; DP cell-relaxation order unchanged"
  - "multires.rs: refine_gamma + refine_gamma_step; gradient perturbation sequence (orig+h, orig-h) lifted verbatim; ptr_arg fixed as Rule 1"
  - "phase_boxplot.rs: four helpers each own one step (envelope, fences, outlier detection, whisker trim); all step outputs passed explicitly — no shared mutable state"
  - "warp_stats.rs: karcher_mean_psi_to_warp owns the iterative sphere Karcher mean; inv_exp_map_sphere/exp_map_sphere call sequence unchanged"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "curve_geodesic_nd (geodesic.rs) drops cog 28->21, cyc 12->9; helper geodesic_nd_step cog 4, cyc 4"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/geodesic.rs#geodesic_nd_dimensions,geodesic_endpoints_match,geodesic_distances_nonneg,geodesic_identical_curves"
        status: pass
    human_judgment: false
  - id: D2
    description: "lambda_cv (lambda_cv.rs) drops cog 28->14, cyc 11->7; helper lambda_cv_score cog 9, cyc 5"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/lambda_cv.rs#lambda_cv_default_config,lambda_cv_loo,lambda_cv_rejects_*"
        status: pass
    human_judgment: false
  - id: D3
    description: "dp_grid_solve_banded (mod.rs) drops cog 25->0; helpers dp_fill_grid cog 10, dp_skip_cell cog 10"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/tests (full alignment suite)"
        status: pass
    human_judgment: false
  - id: D4
    description: "elastic_align_pair_multires (multires.rs) drops cog 44->13, cyc 14->6; helpers refine_gamma cog 11, refine_gamma_step cog 8"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/multires.rs#multires_identity,multires_phase_shifted,multires_falls_back_short_curves,multires_rejects_bad_coarsen_factor"
        status: pass
    human_judgment: false
  - id: D5
    description: "phase_boxplot (phase_boxplot.rs) drops cog 50->10, cyc 21->5; helpers central_envelope cog 9, find_outliers cog 13, trim_whiskers cog 17, whisker_fences cog 1"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/phase_boxplot.rs (9 tests)"
        status: pass
    human_judgment: false
  - id: D6
    description: "warp_statistics (warp_stats.rs) drops cog 31->15, cyc 14->8; helper karcher_mean_psi_to_warp cog 16, cyc 7"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/warp_stats.rs (no inline tests — covered by integration)"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 12: Alignment Complexity Tail (geodesic/lambda_cv/dp_grid/multires/phase_boxplot/warp_stats) Summary

**Six alignment singleton violations eliminated via private helper extraction — all targets now cog <=21 / cyc <=9, all new helpers cog <=17 / cyc <=7, 2942 lib tests pass, clippy clean**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-09-14T20:42:54Z
- **Completed:** 2026-09-14T20:49:27Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `curve_geodesic_nd` cog 28->21, cyc 12->9 (extracted `geodesic_nd_step`)
- `lambda_cv` cog 28->14, cyc 11->7 (extracted `lambda_cv_score`)
- `dp_grid_solve_banded` cog 25->0 (extracted `dp_fill_grid` + `dp_skip_cell`)
- `elastic_align_pair_multires` cog 44->13, cyc 14->6 (extracted `refine_gamma` + `refine_gamma_step`)
- `phase_boxplot` cog 50->10, cyc 21->5 (extracted `central_envelope`, `whisker_fences`, `find_outliers`, `trim_whiskers`)
- `warp_statistics` cog 31->15, cyc 14->8 (extracted `karcher_mean_psi_to_warp`)
- No new helper exceeds cog 17 / cyc 7 — all well below threshold
- All 2942 lib tests pass; behavior and numeric output identical
- clippy `--all-targets --features linalg,parallel -D warnings` clean

## Before → After Scores

| Function | File | cog before | cog after | cyc before | cyc after |
|---|---|---|---|---|---|
| `curve_geodesic_nd` | geodesic.rs | 28 | 21 | 12 | 9 |
| `lambda_cv` | lambda_cv.rs | 28 | 14 | 11 | 7 |
| `dp_grid_solve_banded` | mod.rs | 25 | 0 | 7 | 1 |
| `elastic_align_pair_multires` | multires.rs | 44 | 13 | 14 | 6 |
| `phase_boxplot` | phase_boxplot.rs | 50 | 10 | 21 | 5 |
| `warp_statistics` | warp_stats.rs | 31 | 15 | 14 | 8 |
| `geodesic_nd_step` (new) | geodesic.rs | — | 4 | — | 4 |
| `lambda_cv_score` (new) | lambda_cv.rs | — | 9 | — | 5 |
| `dp_fill_grid` (new) | mod.rs | — | 10 | — | 4 |
| `dp_skip_cell` (new) | mod.rs | — | 10 | — | 5 |
| `refine_gamma` (new) | multires.rs | — | 11 | — | 5 |
| `refine_gamma_step` (new) | multires.rs | — | 8 | — | 5 |
| `central_envelope` (new) | phase_boxplot.rs | — | 9 | — | 5 |
| `whisker_fences` (new) | phase_boxplot.rs | — | 1 | — | 2 |
| `find_outliers` (new) | phase_boxplot.rs | — | 13 | — | 5 |
| `trim_whiskers` (new) | phase_boxplot.rs | — | 17 | — | 7 |
| `karcher_mean_psi_to_warp` (new) | warp_stats.rs | — | 16 | — | 7 |

## Task Commits

1. **Task 1: geodesic.rs, lambda_cv.rs, mod.rs** — `184a3d37` (refactor)
2. **Task 2: multires.rs, phase_boxplot.rs, warp_stats.rs + clippy gate** — `2b1bcafb` (refactor)

## Files Created/Modified

- `fdars-core/src/alignment/geodesic.rs` — `geodesic_nd_step` extracted
- `fdars-core/src/alignment/lambda_cv.rs` — `lambda_cv_score` extracted
- `fdars-core/src/alignment/mod.rs` — `dp_fill_grid` + `dp_skip_cell` extracted
- `fdars-core/src/alignment/multires.rs` — `refine_gamma` + `refine_gamma_step` extracted
- `fdars-core/src/alignment/phase_boxplot.rs` — `central_envelope`, `whisker_fences`, `find_outliers`, `trim_whiskers` extracted
- `fdars-core/src/alignment/warp_stats.rs` — `karcher_mean_psi_to_warp` extracted

## Decisions Made

- **geodesic_nd_step:** Takes all per-k loop locals as parameters (t_k, m, d, v, psi_id, time_01, t0, domain, argvals, q1, q2a, f1, f_aligned, weights, dim_curves, k); `#[allow(clippy::too_many_arguments)]` applied — function boundary justified by cog score
- **lambda_cv_score:** Receives `folds` slice and `k_max` computed in caller; identical call sequence to karcher_mean and elastic_distance
- **dp_grid_solve_banded two-step:** First extraction (dp_fill_grid) left cog 25 in the helper; second extraction (dp_skip_cell) split the skip predicate to bring dp_fill_grid to cog 10
- **refine_gamma ptr_arg fix:** Parameter changed from `&mut Vec<f64>` to `&mut [f64]` per clippy -D warnings rule (same fix as store_align_results in 109-01)
- **phase_boxplot four-helper split:** Single-seam insufficient for cog 50; each of the 4 innermost step bodies becomes a helper — step outputs passed as owned values, no shared mutable state
- **warp_statistics single seam:** Iterative sphere Karcher mean extracted; the outer warp-to-psi conversion loop left in caller (cog contribution small)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clippy ptr_arg lint in refine_gamma**
- **Found during:** Task 2 (clippy gate)
- **Issue:** `refine_gamma` used `&mut Vec<f64>` instead of `&mut [f64]`; clippy flags under `-D warnings`
- **Fix:** Changed parameter type to `&mut [f64]`; call site unchanged (Rust coerces automatically)
- **Files modified:** `fdars-core/src/alignment/multires.rs`
- **Committed in:** `2b1bcafb` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — lint/correctness)
**Impact on plan:** Required for `clippy --all-targets -D warnings` gate to pass. Zero scope creep.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes.

## Self-Check: PASSED

- `fdars-core/src/alignment/geodesic.rs`: FOUND
- `fdars-core/src/alignment/lambda_cv.rs`: FOUND
- `fdars-core/src/alignment/mod.rs`: FOUND
- `fdars-core/src/alignment/multires.rs`: FOUND
- `fdars-core/src/alignment/phase_boxplot.rs`: FOUND
- `fdars-core/src/alignment/warp_stats.rs`: FOUND
- Commit `184a3d37`: FOUND
- Commit `2b1bcafb`: FOUND
- Full lib suite: 2942/2942 PASSED
- clippy --all-targets -D warnings: CLEAN

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
