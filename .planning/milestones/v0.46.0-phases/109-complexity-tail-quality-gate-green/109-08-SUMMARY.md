---
phase: 109-complexity-tail-quality-gate-green
plan: "08"
subsystem: fpca
tags: [rust, complexity, fpca, famm, pace, refactor, pmat]

requires:
  - phase: 109-01
    provides: tracer-confirmed complexity reduction pattern

provides:
  - fpca_variants.rs: dynamical_correlation, fsvd, ssvd below cog 23 / cyc 30 with private helpers
  - famm.rs: dense_flmm, multi_famm, fast_fmm below cog 23 / cyc 30 with private helpers
  - pace_fpca.rs: pace_fpca_validate_config residual (Phase 106) cleared to cog 12

affects: [pmat-quality-gate, QUAL-03]

actuals:
  tokens: 22000
  tasks: 3
  commits: 3
  plan_head_before: ffab30bd2a3a7040f0857412da5ce74a7548ff32

tech-stack:
  added: []
  patterns:
    - "Extract private fn helpers to reduce cognitive complexity without changing any public API"
    - "Combine gram-eigen + singular-vector recovery in one fn to avoid eigenvector recomputation"

key-files:
  created: []
  modified:
    - fdars-core/src/fpca_variants.rs
    - fdars-core/src/famm.rs
    - fdars-core/src/pace_fpca.rs

key-decisions:
  - "fsvd: combined gram_eigen_and_recover into one fn to avoid re-building eigen twice (would risk different eigenvector ordering)"
  - "fast_fmm: moved PointwiseResult struct to module level (renamed FastFmmPointResult) so extracted helpers can reference it"
  - "pace_fpca_validate_config: extracted only work_grid loops into validate_work_grid — one seam was sufficient to cross under cog 23"

requirements-completed: [QUAL-03]

coverage:
  - id: D1
    description: "dynamical_correlation (cog 34->4), fsvd (cog 64->3), ssvd (cog 33->18) in fpca_variants.rs all below cog 23 / cyc 30, helpers clean"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/fpca_variants.rs#tests (19 tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "dense_flmm (cog 27->17), multi_famm (cog 25->19), fast_fmm (cog 32->13) in famm.rs all below cog 23 / cyc 30, helpers clean"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/famm.rs#tests (41 tests)"
        status: pass
    human_judgment: false
  - id: D3
    description: "pace_fpca_validate_config residual (cog 24->12) in pace_fpca.rs below cog 23 / cyc 30"
    requirement: QUAL-03
    verification:
      - kind: unit
        ref: "fdars-core/src/pace_fpca.rs#tests (15 tests)"
        status: pass
    human_judgment: false

patterns-established:
  - "Never split a function whose helpers share an eigenvector result — combine into one fn to avoid re-computation"

duration: 30min
completed: 2026-09-14
status: complete
---

# Phase 109 Plan 08: FPCA/FAMM Complexity Tail Summary

**Seven FPCA/FAMM cognitive complexity violations cleared across fpca_variants.rs, famm.rs, and pace_fpca.rs via private helper extraction — all targets and helpers below cog 23 / cyc 30, 2942 tests green, numeric-identical.**

## Performance

- **Duration:** ~30 min
- **Started:** 2026-09-14
- **Completed:** 2026-09-14
- **Tasks:** 3
- **Files modified:** 3
- **Commits:** 3 (measured)

## Accomplishments

- `fpca_variants.rs`: decomposed `dynamical_correlation` (cog 34→4), `fsvd` (cog 64→3), `ssvd` (cog 33→18) via 10 private helpers (all helpers ≤ cog 21 / cyc 12)
- `famm.rs`: decomposed `dense_flmm` (cog 27→17), `multi_famm` (cog 25→19), `fast_fmm` (cog 32→13) via 5 private helpers (all helpers ≤ cog 10 / cyc 7); moved `PointwiseResult` to module level as `FastFmmPointResult`
- `pace_fpca.rs`: cleared Phase 106 residual `pace_fpca_validate_config` (cog 24→12) by extracting `pace_fpca_validate_work_grid` for the two work-grid loops
- Full library suite: 2942 tests pass; clippy --all-targets clean; fmt clean; SVD/EM iteration order preserved verbatim

## Task Commits

1. **Task 1: fpca_variants.rs** — `c7ed9a3b` (refactor)
2. **Task 2: famm.rs** — `855f48a2` (refactor)
3. **Task 3: pace_fpca.rs** — `84599a99` (refactor)

## Files Created/Modified

- `fdars-core/src/fpca_variants.rs` — added 10 private helpers; all 3 targets below threshold
- `fdars-core/src/famm.rs` — added 5 private helpers + `FastFmmPointResult` struct; all 3 targets below threshold
- `fdars-core/src/pace_fpca.rs` — added `pace_fpca_validate_work_grid`; residual cleared

## Decisions Made

- `fsvd`: the original plan considered splitting gram-eigen and singular-vector recovery into two functions, but the eigenvectors live inside nalgebra's eigen result — passing only `pairs: Vec<(f64, usize)>` would require recomputing eigen (different sort order risk). Combined into `fsvd_gram_eigen_and_recover` + per-component `fsvd_fill_singular_vecs`.
- `fast_fmm`: local struct `PointwiseResult` could not be referenced from module-level helpers; renamed to `FastFmmPointResult` and moved outside the function.
- `pace_fpca_validate_config`: a single seam (the work-grid loops) was sufficient to cross under cog 23 — extracting the scalar bandwidth/sigma2/alpha checks would also work but is unnecessary.

## Deviations from Plan

None — plan executed exactly as written. The fsvd refactor combined two planned helpers (gram_eigen + recover) into one to avoid eigenvector recomputation, which is a safe behavior-preserving choice within the plan's stated intent.

## Issues Encountered

None — all three tasks compiled cleanly on first attempt; no numeric divergence observed.

## Self-Check: PASSED

- `fdars-core/src/fpca_variants.rs` exists and is modified
- `fdars-core/src/famm.rs` exists and is modified
- `fdars-core/src/pace_fpca.rs` exists and is modified
- Commits c7ed9a3b, 855f48a2, 84599a99 verified in git log
- All targets below cog 23 / cyc 30 confirmed via pmat

## Next Phase Readiness

- These 7 functions are the last FPCA/FAMM concentration in the violation list
- Remaining plans in Phase 109 address other modules (alignment, clustering, etc.)
- pmat quality-gate progress continues toward 0 violations

---
*Phase: 109-complexity-tail-quality-gate-green*
*Completed: 2026-09-14*
