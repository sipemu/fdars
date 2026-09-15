---
phase: 109-complexity-tail-quality-gate-green
plan: "11"
subsystem: smooth_basis, metric/soft_dtw, fts/spectral, depth/dispatch
tags: [pmat, complexity, refactor, smooth-basis, soft-dtw, spectral, depth]

requires: [QUAL-03]
provides:
  - "smooth_basis.rs: smooth_monotone (cog 57->13) + evaluate_nbasis_cv (cog 32->12) below threshold"
  - "metric/soft_dtw.rs: soft_dtw_backward (cog 25->13) + lbfgs_refine_from_init (cog 27->17) below threshold"
  - "fts/spectral.rs: dpca (cog 26->10) + dpca_reconstruct (cog 53->0) below threshold"
  - "depth/dispatch.rs: functional_depth (cog 55->10) + functional_boxplot (cog 39->11) below threshold"
  - "All 8 targets + 12 new helpers: max cog 23 / max cyc 7"
affects: []

actuals:
  tokens: 28000
  tasks: 3
  commits: 3
  plan_head_before: 0004c5a5b1ad132ec53bc2d64ee7b353997b1aa6

tech-stack:
  added: []
  patterns:
    - "multi-seam decomposition: 4-seam for cog 57 (smooth_monotone: validate+accum+penalty+gn_step)"
    - "two-seam decomposition: cog 25-53 functions drop below threshold with 2-3 targeted seams"
    - "bug-preserving extraction: soft_dtw_backward endpoint-seed zero-gradient bug preserved verbatim"
    - "require_n inline helper: reuses n-constraint guard across 7 depth method arms"

key-files:
  created: []
  modified:
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/metric/soft_dtw.rs
    - fdars-core/src/fts/spectral.rs
    - fdars-core/src/depth/dispatch.rs

key-decisions:
  - "smooth_monotone (cog 57): four-seam split — validate, accum_obs, add_penalty, gn_step; all helpers cog <=16"
  - "evaluate_nbasis_cv (cog 32): single seam — eval_nbasis_cv_fold (cog 7) extracted from per-fold inner loop"
  - "soft_dtw_backward (cog 25): two seams — sdtw_backward_contrib (#[inline], cog 0) + sdtw_backward_cell (cog 4); endpoint-seed guard (i==n && j==m continue) PRESERVED for bug-preserving extraction"
  - "lbfgs_refine_from_init (cog 27): two seams — lbfgs_hessian_scale (#[inline], cog 4) + lbfgs_update_history (cog 3)"
  - "dpca (cog 26): two seams — dpca_build_filters (cog 6) + dpca_compute_scores (cog 10); inverse-FFT lag symmetry preserved"
  - "dpca_reconstruct (cog 53): three seams — validate (cog 13) + fitted (cog 23, at limit) + error (cog 17); main fn trivially wires them"
  - "functional_depth (cog 55): require_n + depth_dispatch_n2 sub-dispatch covers 7 n>=2 methods, simplifying main match to 6 explicit arms"
  - "functional_boxplot (cog 39): boxplot_central_region (cog 12) + boxplot_fences_and_outliers (cog 6); outlier flagging uses .any() for clarity"

requirements-completed: [QUAL-03]

duration: ~40min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 11: Complexity Tail (smooth_basis + soft_dtw + spectral + depth) Summary

**Eight high-complexity functions across four files decomposed into private helpers — all 8 targets and 12 new helpers are now cognitive <=23 / cyclomatic <=30; 2942 lib tests pass; clippy clean**

## Performance

- **Duration:** ~40 min
- **Completed:** 2026-09-14T20:39:55Z
- **Tasks:** 3
- **Commits:** 3
- **Files modified:** 4

## Before -> After Scores

| Function | File | cog before | cog after | cyc after |
|---|---|---|---|---|
| `smooth_monotone` | smooth_basis.rs | 57 | 13 | 9 |
| `evaluate_nbasis_cv` | smooth_basis.rs | 32 | 12 | 6 |
| `soft_dtw_backward` | metric/soft_dtw.rs | 25 | 13 | 5 |
| `lbfgs_refine_from_init` | metric/soft_dtw.rs | 27 | 17 | 7 |
| `dpca` | fts/spectral.rs | 26 | 10 | 6 |
| `dpca_reconstruct` | fts/spectral.rs | 53 | 0 | 1 |
| `functional_depth` | depth/dispatch.rs | 55 | 10 | 5 |
| `functional_boxplot` | depth/dispatch.rs | 39 | 11 | 6 |

**New helpers (all private, all <=cog 23):**

| Helper | File | cog | cyc |
|---|---|---|---|
| `smooth_monotone_validate` | smooth_basis.rs | 16 | 7 |
| `smooth_monotone_accum_obs` | smooth_basis.rs | 3 | 3 |
| `smooth_monotone_add_penalty` | smooth_basis.rs | 9 | 5 |
| `smooth_monotone_gn_step` | smooth_basis.rs | 4 | 5 |
| `eval_nbasis_cv_fold` | smooth_basis.rs | 7 | 5 |
| `sdtw_backward_contrib` | metric/soft_dtw.rs | 0 | 1 |
| `sdtw_backward_cell` | metric/soft_dtw.rs | 4 | 5 |
| `lbfgs_hessian_scale` | metric/soft_dtw.rs | 4 | 3 |
| `lbfgs_update_history` | metric/soft_dtw.rs | 3 | 3 |
| `dpca_build_filters` | fts/spectral.rs | 6 | 4 |
| `dpca_compute_scores` | fts/spectral.rs | 10 | 5 |
| `dpca_reconstruct_validate` | fts/spectral.rs | 13 | 6 |
| `dpca_reconstruct_fitted` | fts/spectral.rs | 23 | 7 |
| `dpca_reconstruct_error` | fts/spectral.rs | 17 | 7 |
| `require_n` | depth/dispatch.rs | 3 | 2 |
| `depth_dispatch_n2` | depth/dispatch.rs | 1 | 2 |
| `boxplot_central_region` | depth/dispatch.rs | 12 | 7 |
| `boxplot_fences_and_outliers` | depth/dispatch.rs | 6 | 5 |

## Task Commits

1. **Task 1: smooth_basis.rs + soft_dtw.rs** — `9ac706e3`
2. **Task 2: fts/spectral.rs** — `ab940b59`
3. **Task 3: depth/dispatch.rs + clippy gate** — `0be3e883`

## Files Created/Modified

- `fdars-core/src/smooth_basis.rs` — 5 helpers extracted from smooth_monotone + evaluate_nbasis_cv
- `fdars-core/src/metric/soft_dtw.rs` — 4 helpers extracted from soft_dtw_backward + lbfgs_refine_from_init
- `fdars-core/src/fts/spectral.rs` — 5 helpers extracted from dpca + dpca_reconstruct
- `fdars-core/src/depth/dispatch.rs` — 4 helpers extracted from functional_depth + functional_boxplot

## Decisions Made

- **Four-seam split for cog 57 (smooth_monotone):** validation block (cog 16), per-observation accumulation (cog 3), penalty addition (cog 9), Gauss-Newton step wrapper (cog 4). Each seam was independently verifiable and the function drops to cog 13.
- **Bug-preserving extraction of soft_dtw_backward:** The endpoint-seed guard `if i == n && j == m { continue; }` overwrites `E[n][m]=1` before the contribution loop can read it, causing the all-zero gradient bug. This is a known pre-existing bug (see project memory). The guard is preserved verbatim inside `sdtw_backward_cell`.
- **dpca_reconstruct_fitted lands at cog 23 (at limit):** The zero-padded boundary logic for short curves requires nested conditionals across score windows; further splitting would require sharing mutable output state across calls and was judged worse than landing at the limit.
- **require_n + depth_dispatch_n2 eliminates 7 repeated n-guard patterns** in `functional_depth`, dropping it from cog 55 to cog 10 while making the match arm intent clear.
- **No public API/signature change:** All 18 new helpers are `fn` (private). All 8 target functions retain identical signatures, return types, and calling conventions.

## Deviations from Plan

None — plan executed exactly as written. All 8 targets cleared in 3 task commits. No Rule 1/2/3/4 deviations needed.

## Verification

- `pmat analyze complexity` confirmed all 8 targets and all new helpers: max cog 23, max cyc 7
- `cargo test -p fdars-core --features linalg,parallel --lib`: **2942/2942 pass**
- `cargo test -p fdars-core --features linalg,parallel` (doctests): **215/215 pass**
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings`: **clean**

## Self-Check: PASSED

- `fdars-core/src/smooth_basis.rs`: exists, modified
- `fdars-core/src/metric/soft_dtw.rs`: exists, modified
- `fdars-core/src/fts/spectral.rs`: exists, modified
- `fdars-core/src/depth/dispatch.rs`: exists, modified
- Commits `9ac706e3`, `ab940b59`, `0be3e883`: all present in git log
