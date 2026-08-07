---
gsd_state_version: 1.0
milestone: v0.14.0
milestone_name: milestone
current_phase: 01
current_phase_name: measurement-discipline-baselines
status: executing
stopped_at: Completed 01-01-PLAN.md (audit apparatus tracer)
last_updated: "2026-08-07T19:09:42.471Z"
last_activity: 2026-08-07
last_activity_desc: Phase 01 execution started
progress:
  total_phases: 1
  completed_phases: 0
  total_plans: 2
  completed_plans: 1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-07)

**Core value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.
**Current focus:** Phase 01 — measurement-discipline-baselines

## Current Position

Phase: 01 (measurement-discipline-baselines) — EXECUTING
Plan: 2 of 2
Status: Ready to execute
Last activity: 2026-08-07 — Phase 01 execution started

Progress: [█████░░░░░] 50%

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: —

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01-measurement-discipline-baselines P01 | 12 | 1 tasks | 9 files |

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

Last session: 2026-08-07T19:09:42.462Z
Stopped at: Completed 01-01-PLAN.md (audit apparatus tracer)
Resume file: .planning/phases/01-measurement-discipline-baselines/01-02-PLAN.md
