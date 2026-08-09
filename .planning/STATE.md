---
gsd_state_version: 1.0
milestone: v0.14.0
milestone_name: milestone
current_phase: 09
current_phase_name: consolidated-report-prioritized-backlog
status: executing
stopped_at: Completed 09-01-PLAN.md
last_updated: "2026-08-09T20:23:13.745Z"
last_activity: 2026-08-09
last_activity_desc: Phase 07 execution started
progress:
  total_phases: 9
  completed_phases: 8
  total_plans: 21
  completed_plans: 19
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-08-09)

**Core value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.
**Current focus:** Phase 09 — consolidated-report-prioritized-backlog

## Current Position

Phase: 09 (consolidated-report-prioritized-backlog) — EXECUTING
Plan: 2 of 3
Status: Ready to execute
Last activity: 2026-08-09 — Phase 09 execution started

Progress: [█████████░] 90%

## Performance Metrics

**Velocity:**

- Total plans completed: 18
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
| 06 | 1 | - | - |
| 07 | 2 | - | - |
| 08 | 3 | - | - |

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
| Phase 06 P01 | 180 | 3 tasks | 8 files |
| Phase 07 P01 | 9 | 3 tasks | 3 files |
| Phase 07 P02 | 5 | 3 tasks | 1 files |
| Phase 08 P1 | 8 | 3 tasks | 1 files |
| Phase 08-capability-parity-matrix-categorization P02 | 9 | 3 tasks | 1 files |
| Phase 08 P03 | 7 | 3 tasks | 1 files |
| Phase 09 P01 | 8 | 3 tasks | 2 files |

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
- [Phase ?]: svd_equivalence moved to tests/ integration test (harness=false bench issue); pattern matches alloc_audit_fpca.rs
- [Phase ?]: faer thin_svd measured 1.8-4.1x faster than nalgebra at fdars' real FPCA sizes; P6-1 backlog at P2/S-effort (borderline 1.8x at primary N=500,M=200)
- [Phase ?]: significant-values filter (1e-8*sigma_1) required in SVD equivalence test — near-zero values are noise in both backends
- [Phase ?]: D-01 RUNTIME path used: scikit-fda==0.10.1 installed in throwaway venv; skfda.__version__=0.10.1 confirmed at runtime on Python 3.14.5
- [Phase ?]: D-01a: 0.10.1 is both the agreed baseline and current latest PyPI release — no stale-baseline concern
- [Phase ?]: D-04 representation type-system ruling: FDataGrid/FDataBasis/FDataIrregular as type-system → Out-of-Scope; algorithmic capabilities (covariance, interpolation, basis conversion) → In-Scope Algorithm
- [Phase ?]: D-04: MSPlotOutlierDetector (algorithm) is In-Scope Algorithm; MagnitudeShapePlot (visualization) is Out-of-Scope (plotting) — distinct rows for the algorithm vs its plot counterpart
- [Phase ?]: D-04: fetch_* dataset loaders and DataFrame round-trips ruled Out-of-Scope (IO) per PROJECT.md; scoring metrics (r2_score etc.) ruled In-Scope API-Ergonomics
- [Phase ?]: Phase 7 scikit-fda enumeration complete: 125 in-scope + 35 out-of-scope = 160 total capabilities across 6 areas; Design-Goal Filter with explicit borderline rulings ready for Phase 8 consumption
- [Phase ?]: Phase 8 tracer (Plan 01): Preprocessing parity-mapped end-to-end (39 in-scope rows, 29→39 recount supersedes stale header); D-01 verdict + D-03 category rubrics established; schema user-approved for reuse across remaining 5 areas
- [Phase ?]: Misc area recounted 38→40 literal rows (Phase-7 compression documented; recount-supersedes-header convention applied)
- [Phase ?]: ec17d138 GMM accuracy flag cited in All-129 Coverage Check (GMM is fdars-exclusive, no scikit-fda ML row)
- [Phase ?]: Five remaining areas parity-mapped; aggregate 59 present / 19 partial / 63 absent across 141 rows
- [Phase ?]: Separated in-scope (82 gaps) from out-of-scope (32 plotting/IO/type-system) per Pitfall 14; 36 table-stakes + 46 differentiator actionable
- [Phase ?]: Reverse sweep found 30 fdars-exclusive capabilities (22 none / 8 partial-fdars-advantage): 4 SC3 headliners + 12 D-04 candidates + 14 additional from module-map walk
- [Phase ?]: Drafted 20 unranked backlog entries (PREP-01 through MISC-04) + ACC-01 D-02a comparative numerical-accuracy validation item for 4 fragile areas; ranking is Phase 9 (RPT-02)
- [Phase ?]: BACKLOG.md is standalone file separate from AUDIT-REPORT.md for direct /gsd-new-milestone consumption
- [Phase ?]: Ranking formula: score = value / sqrt(effort); value 1-5, effort S=1/M=3/L=9 (sqrt 1/1.732/3)
- [Phase ?]: P6-1 (faer SVD swap) scored as Value=3 / Effort=S => score=3.00 (P2 severity, 1.8x at primary cell)

### Pending Todos

None yet.

### Blockers/Concerns

- Environment has known criterion/doctest linker bus-error flakiness — Phase 1 methodology must document infra-vs-code failure triage; all benchmark-running phases must apply it
- ✓ Resolved [Phase 6]: SVD library comparison conditional gate — Phase 4 confirmed SVD ~99.8% share / copy ~0.15%; Phase 6 executed with GO verdict, faer 1.8–4.1× measured

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none)* | | | |

## Session Continuity

Last session: 2026-08-09T20:23:13.734Z
Stopped at: Completed 09-01-PLAN.md
Resume file: None
