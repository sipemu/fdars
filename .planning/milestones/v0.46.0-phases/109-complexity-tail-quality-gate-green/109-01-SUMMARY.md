---
phase: 109-complexity-tail-quality-gate-green
plan: "01"
subsystem: alignment
tags: [pmat, complexity, refactor, elastic, bayesian, fpns, karcher-mean]

requires: []
provides:
  - "alignment/fpns.rs: top_singular_vector + horiz_fpns below cog 23 / cyc 30"
  - "alignment/closed.rs: karcher_mean_closed below cog 23 / cyc 30"
  - "alignment/bayesian.rs: bayesian_align_pair below cog 23 / cyc 30"
  - "Proven extract->pmat->commit loop for remaining 13 plans"
affects: [109-02, 109-03, 109-04, 109-05, 109-06, 109-07, 109-08, 109-09, 109-10, 109-11, 109-12, 109-13, 109-14]

actuals:
  tokens: 14800
  tasks: 3
  commits: 3
  plan_head_before: 57c9ed82b40ab59e58245898ca818274e9c91381

tech-stack:
  added: []
  patterns:
    - "extract->pmat-confirm->commit: verified end-to-end for alignment module refactors"
    - "private helper placement: helpers placed directly below their parent fn, private fn only"
    - "clippy ptr_arg: &mut Vec<T> -> &mut [T] for slice-style helpers"

key-files:
  created: []
  modified:
    - fdars-core/src/alignment/fpns.rs
    - fdars-core/src/alignment/closed.rs
    - fdars-core/src/alignment/bayesian.rs

key-decisions:
  - "fpns.rs: extract power_iteration_step (#[inline]) and project_to_subsphere from top_singular_vector / horiz_fpns respectively — lifts loop bodies verbatim, FP accumulation order preserved"
  - "closed.rs: extract karcher_iter_step (parallel alignment + SRSF accumulation) and store_align_results from karcher_mean_closed — parallel closure structure preserved exactly"
  - "bayesian.rs: extract validate_bayesian_inputs and compute_posterior_stats from bayesian_align_pair — MCMC RNG draw order and sampling sequence intact; cog 32->11 in one pass"
  - "clippy ptr_arg auto-fix: store_align_results signature changed from &mut Vec<usize> to &mut [usize] (Rule 2 — correctness; clippy -D warnings gate requires it)"

patterns-established:
  - "Two-seam decomposition: for cog 28-32 functions, one validation-block seam + one inner-loop/stats seam achieves the drop below 23 without over-fragmenting"
  - "Single-seam decomposition: for cog 24-25 functions, extracting the heaviest inner loop is sufficient"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "top_singular_vector (fpns.rs) drops from cog 25 to cog 8, cyc 11->5; helper power_iteration_step cog 10, cyc 7"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/fpns.rs#fpns_basic_dimensions,fpns_variance_decreasing,fpns_subsphere_means_on_sphere,fpns_ncomp_one_smoke"
        status: pass
    human_judgment: false
  - id: D2
    description: "horiz_fpns (fpns.rs) drops from cog 28 to cog 18, cyc 14->11; helper project_to_subsphere cog 4, cyc 4"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/fpns.rs#fpns_basic_dimensions,fpns_variance_decreasing,fpns_subsphere_means_on_sphere,fpns_ncomp_one_smoke"
        status: pass
    human_judgment: false
  - id: D3
    description: "karcher_mean_closed (closed.rs) drops from cog 25 to cog 18, cyc 11->8; helpers karcher_iter_step cog 4, store_align_results cog 3"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/closed.rs#closed_align_identity,closed_align_shifted,closed_distance_symmetric,closed_karcher_mean_smoke"
        status: pass
    human_judgment: false
  - id: D4
    description: "bayesian_align_pair (bayesian.rs) drops from cog 32 to cog 11, cyc 18->7; helpers validate_bayesian_inputs cog 14, compute_posterior_stats cog 7"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/bayesian.rs#bayesian_align_identical_curves,bayesian_align_credible_bands_order,bayesian_align_shifted_sine,bayesian_align_rejects_bad_config"
        status: pass
    human_judgment: false

duration: 7min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 01: Alignment Complexity Tail (TRACER) Summary

**Four alignment violations eliminated via private helper extraction (fpns.rs, closed.rs, bayesian.rs) — all targets now cog <=18 / cyc <=11, all new helpers cog <=14 / cyc <=7, 12/12 tests pass, clippy clean**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-09-14T12:16:33Z
- **Completed:** 2026-09-14T12:23:26Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `top_singular_vector` cog 25->8, cyc 11->5 (extracted `power_iteration_step`, `#[inline]`)
- `horiz_fpns` cog 28->18, cyc 14->11 (extracted `project_to_subsphere`)
- `karcher_mean_closed` cog 25->18, cyc 11->8 (extracted `karcher_iter_step` + `store_align_results`)
- `bayesian_align_pair` cog 32->11, cyc 18->7 (extracted `validate_bayesian_inputs` + `compute_posterior_stats`)
- No new helper exceeds cog 14 / cyc 7 — all well below threshold
- All 12 alignment tests pass; behavior and numeric output identical
- clippy `--all-targets --features linalg,parallel -D warnings` clean

## Before → After Scores

| Function | File | cog before | cog after | cyc before | cyc after |
|---|---|---|---|---|---|
| `top_singular_vector` | fpns.rs | 25 | 8 | 11 | 5 |
| `horiz_fpns` | fpns.rs | 28 | 18 | 14 | 11 |
| `karcher_mean_closed` | closed.rs | 25 | 18 | 11 | 8 |
| `bayesian_align_pair` | bayesian.rs | 32 | 11 | 18 | 7 |
| `power_iteration_step` (new) | fpns.rs | — | 10 | — | 7 |
| `project_to_subsphere` (new) | fpns.rs | — | 4 | — | 4 |
| `karcher_iter_step` (new) | closed.rs | — | 4 | — | 4 |
| `store_align_results` (new) | closed.rs | — | 3 | — | 3 |
| `validate_bayesian_inputs` (new) | bayesian.rs | — | 14 | — | 7 |
| `compute_posterior_stats` (new) | bayesian.rs | — | 7 | — | 6 |

## Task Commits

1. **Task 1: TRACER — fpns.rs decomposition** — `684e3371` (refactor)
2. **Task 2: closed.rs — karcher_mean_closed** — `47c4429a` (refactor)
3. **Task 3: bayesian.rs — bayesian_align_pair + clippy gate** — `402d4cd3` (refactor)

## Files Created/Modified

- `fdars-core/src/alignment/fpns.rs` — `power_iteration_step` + `project_to_subsphere` extracted
- `fdars-core/src/alignment/closed.rs` — `karcher_iter_step` + `store_align_results` extracted
- `fdars-core/src/alignment/bayesian.rs` — `validate_bayesian_inputs` + `compute_posterior_stats` extracted

## Decisions Made

- **Two-seam for cog 28-32:** `bayesian_align_pair` needed validation + stats seams; validation extracted as `validate_bayesian_inputs`, post-MCMC computation as `compute_posterior_stats`; MCMC loop body left intact (RNG draw order preserved)
- **Single-seam for cog 24-25:** `top_singular_vector` and `karcher_mean_closed` each needed one heavy inner loop extracted; one helper per function was sufficient
- **`#[inline]` on `power_iteration_step`:** hot-path numeric loop called 200 times per top-singular-vector call; matches existing `row_dot`/`row_l2_sq` convention in matrix.rs
- **Clippy `ptr_arg` fix included in Task 3 commit:** `store_align_results` signature `&mut Vec<usize>` → `&mut [usize]`; clippy -D warnings treats this as an error, so fixing it was a requirement for the gate

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed clippy ptr_arg lint in store_align_results**
- **Found during:** Task 3 (clippy gate)
- **Issue:** `store_align_results` used `&mut Vec<usize>` instead of `&mut [usize]`; clippy flags this as an error under `-D warnings`
- **Fix:** Changed parameter type to `&mut [usize]`; call site unchanged (Rust coerces `&mut Vec<usize>` to `&mut [usize]` automatically)
- **Files modified:** `fdars-core/src/alignment/closed.rs`
- **Committed in:** `402d4cd3` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — Bug/lint)
**Impact on plan:** Required for `clippy --all-targets -D warnings` gate to pass. Zero scope creep.

## Issues Encountered

None — all four targets dropped below threshold in first attempt; no helper self-flagged.

## Tracer Loop Proven

The extract → module-tests-green → pmat-confirms-drop (≤23/≤30) → `--no-verify` commit loop is
proven end-to-end. Key facts for the remaining 13 plans:

1. Two-seam decomposition handles cog 28-32 in one pass (bayesian proved it).
2. `pmat analyze complexity` counts immediately after editing — no build needed.
3. `cargo fmt` before `--no-verify` commit is mandatory (CI fmt-check).
4. Clippy should be run at least on the final task commit of each plan.
5. No helper self-flagged; pmat counts inside helpers but these stayed safely below 23.

## Next Phase Readiness

- Remaining 13 plans (109-02 through 109-14) can follow identical mechanical pattern
- All alignment targets cleared; subsequent plans address other module groups (peer.rs, clustering_advanced.rs, coclustering.rs, helpers.rs, etc.)

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
