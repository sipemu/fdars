---
gsd_state_version: 1.0
milestone: v0.14.0
milestone_name: milestone
current_phase: 02
current_phase_name: static-hot-path-analysis
status: verifying
stopped_at: Completed 02-02-PLAN.md (phase 2 static hot-path map complete)
last_updated: "2026-08-07T20:17:09.006Z"
last_activity: 2026-08-07
last_activity_desc: Phase 02 execution started
progress:
  total_phases: 2
  completed_phases: 2
  total_plans: 4
  completed_plans: 4
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-07)

**Core value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.
**Current focus:** Phase 02 — static-hot-path-analysis

## Current Position

Phase: 02 (static-hot-path-analysis) — EXECUTING
Plan: 2 of 2
Status: Phase complete — ready for verification
Last activity: 2026-08-07 — Phase 02 execution started

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
| Phase 02 P01 | 5 | 2 tasks | 1 files |
| Phase 02 P02 | 8 | 3 tasks | 1 files |

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
- [Phase ?]: elastic_fpca.rs:930 enclosing fn is optimize_balance_c_raw (inside eval_c closure) — source-verified, RESEARCH Open Question 2 resolved; site is called ≤20× per golden-section search
- [Phase ?]: Phase 2 three-table format (Complexity Table + Allocation Hotspot + Parallelism Gap List) proven end-to-end on elastic alignment tracer slice; Plan 02 expands by adding rows
- [Phase ?]: Open Question 1 resolved: fraiman_muniz_1d delegates to StreamingFraimanMuniz::depth_batch (iter_maybe_parallel!) — [parallel-gated], not a gap
- [Phase ?]: from_column_slice basis sites are a distinct allocation category from to_dmatrix() SVD copies — different optimization path per RESEARCH Pitfall 5
- [Phase ?]: regression.rs:291 weighted=centered.clone() is a zero-copy candidate; FpcaResult retains centered so pre-allocated buffer strategy needed

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

Last session: 2026-08-07T20:17:08.999Z
Stopped at: Completed 02-02-PLAN.md (phase 2 static hot-path map complete)
Resume file: None
