---
gsd_state_version: 1.0
milestone: v0.14.0
milestone_name: milestone
current_phase: 2
current_phase_name: Static Hot-Path Analysis
status: planning
stopped_at: Completed 01-02-PLAN.md (full phase baseline)
last_updated: "2026-08-07T19:35:31.451Z"
last_activity: 2026-08-07
last_activity_desc: Phase 01 complete, transitioned to Phase 2
progress:
  total_phases: 1
  completed_phases: 1
  total_plans: 2
  completed_plans: 2
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-07)

**Core value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.
**Current focus:** Phase 01 — measurement-discipline-baselines

## Current Position

Phase: 2 — Static Hot-Path Analysis
Plan: Not started
Status: Ready to plan
Last activity: 2026-08-07 — Phase 01 complete, transitioned to Phase 2

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 2
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01-measurement-discipline-baselines P01 | 12 | 1 tasks | 9 files |
| Phase 01 P02 | 35 | 3 tasks | 14 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Relevant to current work:

- Audit-only milestone — deliverables are a report + backlog, no production code changes to fdars-core
- scikit-fda 0.10.1 is the sole functionality-gap yardstick
- Performance measured via static analysis first, then real criterion benchmarks
- Backlog phrased as GSD-ready requirements/phases, ranked by user value not ease
- [Phase ?]: D-04 sentinel: karcher_mean substituted for fdata_to_pc_1d as 4-combo discriminator — FPCA center_columns is sequential, nalgebra SVD is always sequential; karcher_mean uses iter_maybe_parallel! (10x speedup with parallel feature confirmed)
- [Phase ?]: TMPDIR=/home/simonm/.cache/fdars-bench-tmp required for bench linking; /tmp tmpfs at 94% capacity causes doctest bus-errors (documented --no-verify exception)
- [Phase ?]: elastic_self_distance_matrix returns FdMatrix (not Result<Vec<f64>,_>); output wrapped in black_box directly — confirmed from src/alignment/pairwise.rs:194
- [Phase ?]: streaming sentinel tagged LOW CONFIDENCE: 11.1% two-run variance due to OS scheduler jitter at sub-ms scale; re-measure under taskset/cpupower in later phases

### Pending Todos

None yet.

### Blockers/Concerns

- Environment has known criterion/doctest linker bus-error flakiness — Phase 1 methodology must document infra-vs-code failure triage; all benchmark-running phases must apply it
- Phase 6 (SVD library comparison) is conditional — executes only if Phase 4 shows SVD is a significant runtime share and copy is not the dominant cost

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-08-07T19:24:23.968Z
Stopped at: Completed 01-02-PLAN.md (full phase baseline)
Resume file: None
