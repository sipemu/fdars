---
phase: 46-whole-crate-profiling-measurement
plan: 03
subsystem: infra
tags: [profiling, dedup, static-analysis, consolidation, cons]
requires:
  - phase: 46-01
    provides: (none — independent static-analysis plan)
provides:
  - PROF-02 ranked duplication/consolidation inventory (drives Phase 49)
affects: [49-code-consolidation-dedup]
actuals:
  tokens: 6000
  tasks: 2
  commits: 1
tech-stack:
  added: []
  patterns: [grep-verified file:line anchors, leverage = call-sites × drift-risk ranking]
key-files:
  created: [.planning/phases/46-whole-crate-profiling-measurement/PROF-02-dedup-inventory.md]
  modified: []
key-decisions:
  - "χ²/F survival is the #1 dedup target: two independent regularized-incomplete-gamma kernels (inference/dist.rs + spm/chi_squared.rs)."
  - "simpsons_weights, Cholesky, FPCA scoring are ALREADY consolidated — explicitly out of scope for Phase 49."
patterns-established:
  - "Dedup leverage ranking: (# call sites × complexity/drift-risk), not raw duplicate-LOC."
requirements-completed: [PROF-02]
coverage:
  - id: D1
    description: "PROF-02 ranked duplication inventory with grep-verified src/ file:line anchors and CONS-01/CONS-02 tags"
    requirement: PROF-02
    verification:
      - kind: other
        ref: "grep -Eo 'src/[a-z_/]+\\.rs:[0-9]+' PROF-02-dedup-inventory.md | sort -u | wc -l => 10 (>=8); grep -c CONS-0[12] => 11 (>=2)"
        status: pass
    human_judgment: true
    rationale: "Inventory completeness/usefulness for Phase 49 is a human judgement; automation only confirms anchor + tag presence."
duration: 15min
completed: 2026-08-30
status: complete
---

# Phase 46 / Plan 03: PROF-02 Duplication Inventory Summary

**Ranked, anchored duplication inventory: χ²/F survival kernels (2 impls) is the top dedup target; permutation loops, seeded-RNG, and SVD sign-fix follow; simpsons/Cholesky/FPCA-scoring confirmed already consolidated.**

## Performance
- **Duration:** ~15 min (ran in parallel with the 46-01 baseline suite)
- **Tasks:** 2
- **Files modified:** 1 (doc only)

## Accomplishments
- Re-verified all 7 RESEARCH duplication categories against the live tree with current file:line anchors and call-site counts.
- Confirmed the χ²/F survival duplication is real: `inference/dist.rs:99` (chi_square_sf + own gamma helpers) vs `spm/chi_squared.rs:164` (chi2_cdf + own regularized_gamma_p) — two independent numerical kernels → HIGH-leverage CONS-01, proposed new `src/distributions.rs`.
- Catalogued permutation loops (6 sites, 3 sequential / 3 parallel → CONS-02), per-thread seeded-RNG (10 thread-offset sites → CONS-02), SVD sign-fix (`regression.rs:180` canonical vs `pace_fpca.rs:219` inline mirror → CONS-01, correctness-critical).
- Confirmed `simpsons_weights` (161 hits), Cholesky (frechet reuses `crate::linalg`), and FPCA scoring (144/68 hits) are already consolidated — flagged No-Action so Phase 49 doesn't chase phantom dedup.

## Task Commits
1. **Task 1: grep-verify duplication candidates** — analysis (no file)
2. **Task 2: write PROF-02 inventory** — committed with plan docs

## Files Created/Modified
- `.planning/phases/46-whole-crate-profiling-measurement/PROF-02-dedup-inventory.md` — ranked inventory

## Decisions Made
- Ranked by leverage = call-sites × drift-risk (locked metric), placing the 2-kernel χ² duplication above the wider-but-simpler seeded-RNG pattern.

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None.

## Next Phase Readiness
- PROF-02 is migration-ready for Phase 49 (CONS-01/CONS-02): every target carries a live anchor + proposed `pub(crate)` signature.

---
*Phase: 46-whole-crate-profiling-measurement*
*Completed: 2026-08-30*
