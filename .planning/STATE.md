---
gsd_state_version: 1.0
milestone: v0.14.0
milestone_name: milestone
current_phase: 6
current_phase_name: Conditional SVD Library Comparison
status: executing
stopped_at: Completed 05-03-PLAN.md
last_updated: "2026-08-08T21:09:59.570Z"
last_activity: 2026-08-08
last_activity_desc: Phase 05 complete, transitioned to Phase 6
progress:
  total_phases: 6
  completed_phases: 5
  total_plans: 13
  completed_plans: 12
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-07)

**Core value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.
**Current focus:** Phase 05 — parallelism-gap-assessment

## Current Position

Phase: 6 — Conditional SVD Library Comparison
Plan: Not started
Status: Ready to execute
Last activity: 2026-08-08 — Phase 05 complete, transitioned to Phase 6

Progress: [██████████] 100%

## Performance Metrics

**Velocity:**

- Total plans completed: 12
- Average duration: — min
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 | 2 | - | - |
| 02 | 2 | - | - |
| 03 | 2 | - | - |
| 04 | 3 | - | - |
| 05 | 3 | - | - |

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
| Phase 03 P01 | 6 | 3 tasks | 3 files |
| Phase 03 P02 | 115 | 4 tasks | 15 files |
| Phase 04 P01 | 6 | 2 tasks | 2 files |
| Phase 04 P02 | 5 | 3 tasks | 5 files |
| Phase 04 P03 | 16 | 3 tasks | 9 files |
| Phase 05 P01 | 22 | 2 tasks | 6 files |
| Phase 05 P02 | 16 | 3 tasks | 7 files |
| Phase 05 P03 | 2 | 3 tasks | 1 files |

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
- [Phase ?]: karcher_mean N=100 M=50 unbanded ~318 ms (D-06 params linalg,parallel) — Phase-3 tracer measured
- [Phase ?]: karcher_mean_banded import deferred to Plan 02 to keep wave warning-clean
- [Phase ?]: n500_m200 elastic_self/cross INFEASIBLE (O(N^2*m^2) ~384-700s/iter); documented as bottleneck evidence
- [Phase ?]: Observed banded reduction 4-6x at representative cells vs ~7x expected; karcher LOW CONFIDENCE due to OS jitter, cross-distance EXCELLENT confidence
- [Phase ?]: dhat 0.3.3 confirmed on crates.io; RESEARCH A1 was correct — dhat 0.3.x is current and compatible
- [Phase ?]: dhat::HeapStats uses max_bytes not peak_bytes (RESEARCH.md had wrong field; auto-fixed in alloc_audit_fpca.rs)
- [Phase ?]: All 3 dhat test cells added in Wave 0 per PATTERNS.md:195-198 (fdata_to_pc_1d, vert_fpca, joint_fpca); PATTERNS.md takes precedence over Task 2 minimal description
- [Phase ?]: no-verify commits used for tmp tmpfs exhaustion per MEMORY.md exception
- [Phase ?]: copy-share for to_dmatrix() at N=500,M=200 is ~0.17% of wall-clock -- copy is negligible; SVD dominates
- [Phase ?]: dhat baseline: fdata_to_pc_1d N=500,M=200 -- 23 total_blocks, 4,376,024 total_bytes, 4,332,792 peak_bytes
- [Phase ?]: Phase 6 is triggered (GO): SVD share ~99.8-99.9% of FPCA wall-clock, copy-share ~0.14-0.17% — both SC1 conditions met
- [Phase ?]: 6-cell FPCA grid stable (all variances < 0.64% OK confidence); FPCA parallel-invariant (D-04 formalized)
- [Phase ?]: Per-unit-work normalization: joint_fpca (320.96 bytes/n·m) > vert_fpca (57.16) > fdata_to_pc_1d (43.76)
- [Phase ?]: [Phase 5]: karcher_mean thread-scaling (N=100,M=50,linalg): 1t=1554ms, 2t=782ms(1.99x), 4t=405ms(3.84x LOW-CONF 11.4% spread), 8t=328ms(4.73x); curve flattens 4->8, not climbing at 8
- [Phase ?]: [Phase 5]: D-04 governor pinning FAILED (cpupower needs root, non-interactive sudo denied); taskset -c 0-7 + 3-run median+spread applied as backstop; karcher table carries governor-not-pinned LOW CONFIDENCE
- [Phase ?]: [Phase 5]: cargo bench rejects --release (bench profile already opt-3); release confirmed via /release/deps/ path instead
- [Phase ?]: SC1 payback-threshold N: karcher_mean ≤ 10 (heavy pays back at any N); StreamingFraimanMuniz::depth_batch ≈ 50 (light loses to single-thread below N_obj≈50) — the two sentinels bracket the crossover (D-01/D-02).
- [Phase ?]: PERF-05 complete: SC2 safe-to-parallelize list, SC3 unaccelerated-path cost (rayon-off ~10x + banding ~4-6x cited), SC4 GSD-ready parallelization backlog (P5-1..P5-4) added to AUDIT-REPORT.md ## Phase 5 section — static-argument-only, zero fdars-core/src edits (D-06).

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

Last session: 2026-08-08T19:00:50.050Z
Stopped at: Completed 05-03-PLAN.md
Resume file: None
