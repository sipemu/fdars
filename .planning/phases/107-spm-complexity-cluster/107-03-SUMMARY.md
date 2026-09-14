---
phase: 107-spm-complexity-cluster
plan: "03"
subsystem: spm
tags: [rust, complexity, pmat, refactor, spm, ewma, mewma, stats, phase]

requires:
  - phase: 107-01
    provides: tracer pattern for SPM complexity decomposition

provides:
  - spm_ewma_monitor (cog 27→1) via private helpers in ewma.rs
  - spm_mewma_monitor (cog 25→7) via private helpers in mewma.rs
  - spe_multivariate (cog 27→6) via private helpers in stats.rs
  - spm_monitor_from_fields (cog 27→0) via private helpers in phase.rs

affects: [107-04, pmat quality-gate, QUAL-02]

actuals:
  tokens: 12525
  tasks: 3
  commits: 3
  plan_head_before: 06e1a7bb433408d67a37bbcda121c1b320e42d91

tech-stack:
  added: []
  patterns:
    - "validate-and-call: extract validation block returning Result<(dims...), FdarError>"
    - "verbatim loop-body extraction: lift inner loops to private fn preserving FP accumulation order"
    - "exact-covariance branch isolation: extract per-i formula loop as standalone fn (no cross-step state)"

key-files:
  created: []
  modified:
    - fdars-core/src/spm/ewma.rs
    - fdars-core/src/spm/mewma.rs
    - fdars-core/src/spm/stats.rs
    - fdars-core/src/spm/phase.rs

key-decisions:
  - "ewma_compute_spe receives raw (unsmoothed) scores + sequential_data separately to preserve the SPE-from-raw-scores invariant"
  - "mewma_validate returns effective ncomp so parent avoids recomputing chart.eigenvalues.len().min(config.ncomp)"
  - "spe_mv_compute_var returns Vec<f64> of per-obs contributions rather than a scalar so caller can accumulate in its outer loop"
  - "monitor_fields_validate consolidates all 5 guards in one fn (cog 15) — no split-validate needed (sequential if-returns, not nested)"

patterns-established:
  - "SPM validate-and-call: validated dimension tuple returned from _validate fn enables clean parent body"
  - "Exact-covariance branch extraction: direct per-i formula (no carried state) is safe to extract by value"

requirements-completed: [QUAL-02]

coverage:
  - id: D1
    description: "spm_ewma_monitor cognitive complexity drops from 27 to 1 with extracted private helpers (ewma_validate, ewma_truncate_scores, ewma_compute_t2_exact, ewma_compute_spe), numeric output unchanged"
    requirement: QUAL-02
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- spm: (2942 tests green)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/spm/ewma.rs: spm_ewma_monitor cog=1, all helpers cog<=10"
        status: pass
    human_judgment: false
  - id: D2
    description: "spm_mewma_monitor cognitive complexity drops from 25 to 7 (below warning threshold 23) with extracted private helpers (mewma_validate cog 13, mewma_compute_exact_t2 cog 3)"
    requirement: QUAL-02
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib: 2942 tests green"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/spm/mewma.rs: spm_mewma_monitor cog=7"
        status: pass
    human_judgment: false
  - id: D3
    description: "spe_multivariate cognitive complexity drops from 27 to 6 with extracted private helpers (spe_mv_validate, spe_mv_var_check, spe_mv_compute_var)"
    requirement: QUAL-02
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib: 2942 tests green"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/spm/stats.rs: spe_multivariate cog=6"
        status: pass
    human_judgment: false
  - id: D4
    description: "spm_monitor_from_fields cognitive complexity drops from 27 to 0 with extracted private helpers (monitor_fields_validate, monitor_fields_project, monitor_fields_recon)"
    requirement: QUAL-02
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib: 2942 tests green"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/spm/phase.rs: spm_monitor_from_fields cog=0"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-09-14
status: complete
---

# Phase 107 Plan 03: SPM Monitoring-Monitors Cluster Summary

**Private-helper extraction in ewma/mewma/stats/phase drops all four monitoring-cluster hotspots from cog 25-27 to cog 0-7 under pmat 3.18.0, 2942 lib tests numerically identical, clippy clean**

## Performance

- **Duration:** ~25 min
- **Completed:** 2026-09-14
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- `spm_ewma_monitor` (ewma.rs): cog 27 → 1 via 4 private helpers (`ewma_validate`, `ewma_truncate_scores`, `ewma_compute_t2_exact`, `ewma_compute_spe`)
- `spm_mewma_monitor` (mewma.rs): cog 25 → 7 (below the warning threshold 23) via 2 private helpers (`mewma_validate`, `mewma_compute_exact_t2`)
- `spe_multivariate` (stats.rs): cog 27 → 6 via 3 private helpers (`spe_mv_validate`, `spe_mv_var_check`, `spe_mv_compute_var`)
- `spm_monitor_from_fields` (phase.rs): cog 27 → 0 via 3 private helpers (`monitor_fields_validate`, `monitor_fields_project`, `monitor_fields_recon`)
- All public signatures and visibility unchanged; no new crate dependency; 2942 lib tests green

## pmat Before/After

| Function | File | Before (cog) | After (cog) | Max Helper (cog) |
|----------|------|-------------|-------------|-----------------|
| `spm_ewma_monitor` | ewma.rs | 27 | 1 | 10 (`ewma_compute_spe`) |
| `spm_mewma_monitor` | mewma.rs | 25 | 7 | 13 (`mewma_validate`) |
| `spe_multivariate` | stats.rs | 27 | 6 | 8 (`spe_mv_var_check`) |
| `spm_monitor_from_fields` | phase.rs | 27 | 0 | 15 (`monitor_fields_validate`) |

All extracted helpers are below the cog error threshold (25) and below the warning threshold (23) — the highest is `monitor_fields_validate` at cog 15.

## Task Commits

1. **Task 1: decompose spm_ewma_monitor + spm_mewma_monitor** - `72334e2a` (refactor)
2. **Task 2: decompose spe_multivariate** - `906038f3` (refactor)
3. **Task 3: decompose spm_monitor_from_fields** - `f7dedb87` (refactor)

## Files Created/Modified

- `fdars-core/src/spm/ewma.rs` — added `ewma_validate`, `ewma_truncate_scores`, `ewma_compute_t2_exact`, `ewma_compute_spe` below `spm_ewma_monitor`
- `fdars-core/src/spm/mewma.rs` — added `mewma_validate`, `mewma_compute_exact_t2` below `spm_mewma_monitor`
- `fdars-core/src/spm/stats.rs` — added `spe_mv_validate`, `spe_mv_var_check`, `spe_mv_compute_var` below `spe_multivariate`
- `fdars-core/src/spm/phase.rs` — added `monitor_fields_validate`, `monitor_fields_project`, `monitor_fields_recon` below `spm_monitor_from_fields`

## Decisions Made

- `ewma_compute_spe` receives both `scores` (raw, unsmoothed) and `sequential_data` separately to preserve the SPE-from-raw-scores invariant documented in the parent.
- `mewma_validate` returns the effective `ncomp` so the parent avoids recomputing `chart.eigenvalues.len().min(config.ncomp)` twice.
- `spe_mv_compute_var` returns `Vec<f64>` of per-observation contributions (not a scalar) so the outer `for v in 0..p` loop in the parent can accumulate them with `+=`.
- `monitor_fields_validate` consolidates all 5 guards into one function (cog 15) — the split-validate pattern was not needed because all 5 guards are sequential flat `if` returns (not nested).

## Deviations from Plan

None — plan executed exactly as written. All extraction seams from RESEARCH.md were applied verbatim. The `monitor_fields_validate` assumption (A3) that 5 guards might push to cog > 25 proved conservative; actual score is cog 15 (sequential non-nested guards).

## Issues Encountered

None.

## Next Phase Readiness

- Plan 107-03 targets cleared: all four functions below pmat threshold
- Plan 107-04 (`partial.rs`, `frcc.rs`, `profile.rs`) is independently executable — no file overlap with this plan
- QUAL-02 fully satisfied after 107-04 completes

---
*Phase: 107-spm-complexity-cluster*
*Completed: 2026-09-14*
