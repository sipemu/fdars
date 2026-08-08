# Roadmap: fdars — Performance & Functionality Audit

**Created:** 2026-08-07
**Milestone:** AUDIT (v0.14.0)
**Core Value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized, GSD-ready backlog.
**Granularity:** fine
**Mode:** mvp (vertical slices — each phase delivers a self-contained chunk of audit analysis + its portion of the report/backlog so findings land incrementally)

> This is an **audit milestone**. Every phase produces *analysis artifacts* — complexity maps, benchmark tables, allocation profiles, parity matrices, and backlog entries — **not** production code changes to `fdars-core`. Implementing the findings is a future milestone seeded by the RPT-02 backlog.

## Phases

- [x] **Phase 1: Measurement Discipline & Baselines** - Lock in build-mode/feature-flag guardrails, define the N×M workload matrix, and record baseline benchmark numbers (completed 2026-08-07)
- [x] **Phase 2: Static Hot-Path Analysis** - Document per-module complexity, allocation hotspots, and parallelism gaps with zero runtime cost (completed 2026-08-07)
- [x] **Phase 3: Elastic Alignment Hot Path** - Benchmark and characterize the O(N²·M²) elastic alignment / Karcher / distance-matrix hot path (completed 2026-08-08)
- [x] **Phase 4: FPCA/SVD & Allocation Audit** - Benchmark FPCA and quantify the FdMatrix→DMatrix SVD-copy overhead with dhat (completed 2026-08-08)
- [ ] **Phase 5: Parallelism Gap Assessment** - Measure rayon thread scaling and flag safe-to-parallelize sequential loops
- [ ] **Phase 6: Conditional SVD Library Comparison** - Compare nalgebra vs faer SVD at real sizes only if FPCA benchmarks warrant it
- [ ] **Phase 7: scikit-fda Capability Enumeration** - Enumerate scikit-fda's public capability surface by area, pinning the compared version
- [ ] **Phase 8: Capability Parity Matrix & Categorization** - Map fdars vs scikit-fda by capability, categorize gaps, and document fdars strengths
- [ ] **Phase 9: Consolidated Report & Prioritized Backlog** - Combine all findings into a report with a value-ranked, GSD-ready backlog

## Phase Details

### Phase 1: Measurement Discipline & Baselines

**Goal**: Establish the measurement guardrails and workload definitions that make every downstream benchmark valid
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: PERF-02
**Success Criteria** (what must be TRUE):

  1. A representative workload matrix (N × M input sizes per hot-path module: elastic alignment, FPCA/SVD, depth & distance, CV loops, streaming depth, smoothing) exists in the report, with realistic sizes (N∈{100,500,1000}, M∈{50,200,500}) justified against Pitfall 4
  2. A benchmark methodology section documents the mandatory `--release` build check, the feature-flag matrix (`""`, `parallel`, `linalg`, `linalg,parallel`), `black_box` requirement, rustc version capture, and ±5% two-run variance threshold
  3. A baseline benchmark run for at least one target per hot-path module is recorded (release + `linalg,parallel`), with binary-path `/release/` confirmed and results saved under `.planning/research/bench/`
  4. The methodology section explicitly documents the criterion/doctest linker-flakiness issue and the "infrastructure failure vs code failure" triage rule so later phases classify signals (bus errors) as infra, not defects

**Plans:** 2/2 plans complete
**Wave 1**

- [x] 01-01-PLAN.md — Tracer: prove the audit apparatus end-to-end on the FPCA + 4-combo (karcher) sentinel (compile 4 combos, release run, raw artifact)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Expansion: remaining 5 module sentinels, release baselines (2 runs each), methodology + workload-matrix sections

### Phase 2: Static Hot-Path Analysis

**Goal**: Produce the zero-cost priority map of where fdars scales badly and why, before any expensive measurement
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: PERF-01
**Success Criteria** (what must be TRUE):

  1. A per-module bottleneck-candidate table exists in the report giving algorithmic complexity in N (curves) and M (grid points) for at least elastic alignment, FPCA/SVD, depth & distance matrices, CV loops, streaming depth, and smoothing
  2. An allocation-hotspot list enumerates every `to_dmatrix()` / `DMatrix::from_column_slice` / redundant-clone call site (including the 8 FdMatrix→DMatrix SVD-copy sites) as candidates for the later dhat audit
  3. A parallelism-gap list flags sequential loops that are candidates for parallelization (classification CV folds, streaming-depth `depth_batch`, elastic-FPCA inner N-loops) and notes where banding is opt-in rather than automatic
  4. Every finding annotates which code paths are feature-gated (`linalg`-only, `parallel`-only) so no path is mislabeled "sequential" when its hot loop is wrapped in `iter_maybe_parallel!`

**Plans:** 2/2 plans complete

**Wave 1**

- [x] 02-01-PLAN.md — Tracer: prove all 3 list formats + complexity-row format end-to-end on the elastic alignment module (worst case)

**Wave 2** *(blocked on Wave 1 completion — same append target)*

- [x] 02-02-PLAN.md — Expansion: remaining 5 module complexity rows, all 8 SVD sites + 14 basis sites + redundant clone, and the full parallelism-gap/already-parallel inventory

### Phase 3: Elastic Alignment Hot Path

**Goal**: Confirm with real numbers whether elastic alignment is fdars' top bottleneck and produce its report + backlog slice
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: PERF-03
**Success Criteria** (what must be TRUE):

  1. A criterion results table for `karcher_mean` and the elastic self/cross-distance matrices at N∈{100,500}×M∈{50,200}, release build with `linalg,parallel` and `black_box`, tagged with feature set and toolchain version, exists in the report
  2. Banded-vs-unbanded results are recorded at a fixed band fraction, quantifying the expected ~7× reduction and confirming `karcher_mean()` defaults to `band = None` (Anti-Pattern 2)
  3. Results are reproducible: raw criterion output is saved under `.planning/research/bench/` and each finding links to its artifact, with two-run variance within ±5% (results exceeding 10% variance marked LOW CONFIDENCE)
  4. This slice's backlog entries (elastic-alignment perf items) are drafted with function/current-cost/root-cause fields, ready for final ranking in Phase 9

**Plans:** 2/2 plans complete

**Wave 1**

- [x] 03-01-PLAN.md — Tracer: prove the measure→artifact→report→backlog pipeline end-to-end on one karcher_mean cell (N=100×M=50, D-06 params)

**Wave 2** *(blocked on Wave 1 — same append targets: audit_hotpaths.rs + AUDIT-REPORT.md)*

- [x] 03-02-PLAN.md — Expansion: full grid (3 targets × N∈{100,500}×M∈{50,200} × banded/unbanded), 2 runs each, results table + ~7× banded analysis + variance + GSD-ready backlog

### Phase 4: FPCA/SVD & Allocation Audit

**Goal**: Separate SVD compute cost from copy/allocation overhead in FPCA and produce its report + backlog slice
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: PERF-03, PERF-04
**Success Criteria** (what must be TRUE):

  1. A criterion results table for `fdata_to_pc_1d` (and elastic-FPCA SVD sites) at N∈{100,500,1000}×M∈{50,200}, release + `linalg,parallel` + `black_box`, tagged with feature set and toolchain, exists in the report
  2. A dhat allocation audit quantifies the documented `FdMatrix→DMatrix` SVD-copy overhead (bytes/allocations per FPCA call) and ranks other allocation hotspots, with a reproducible baseline saved under `.planning/research/bench/`
  3. The report states allocation cost as a share of wall-clock for the top FPCA path, so the SVD-compute vs copy split is explicit (addresses Pitfall 5) and directly informs the Phase 6 go/no-go trigger
  4. This slice's backlog entries (SVD-copy elimination, truncated-SVD candidates) are drafted with function/current-cost/root-cause fields

**Plans**: 3/3 plans executed

**Wave 0**

- [x] 04-01-PLAN.md — dhat wiring: add feature-gated dhat dev-dep + dhat-heap feature to Cargo.toml and create the alloc_audit_fpca.rs integration-test harness (PERF-04 gap)

**Wave 1** *(depends on 04-01)*

- [x] 04-02-PLAN.md — Tracer: prove the measure→dhat→report→copy-share pipeline end-to-end on one FPCA cell (N=500×M=200) + 04-COVERAGE.md stub

**Wave 2** *(depends on 04-02 — same append targets: audit_hotpaths.rs, alloc_audit_fpca.rs, AUDIT-REPORT.md)*

- [x] 04-03-PLAN.md — Expansion: full 6-cell N×M grid (2 runs + no-parallel invariance) + vert/joint elastic-FPCA cells + dhat baselines + completed report (hotspot ranking, SVD-vs-copy split, Phase-6 go/no-go verdict, GSD-ready backlog)

### Phase 5: Parallelism Gap Assessment

**Goal**: Measure how well fdars uses available cores and identify the safe, high-leverage parallelization gaps
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: PERF-05
**Success Criteria** (what must be TRUE):

  1. A rayon thread-scaling table (`RAYON_NUM_THREADS` sweep, e.g. 1/2/4/8) for representative parallel hot paths exists in the report, including the threshold N at which parallel overhead is paid back (Pitfall: rayon overhead on small N)
  2. Sequential loops confirmed safe to parallelize are listed with evidence (classification CV folds, streaming-depth `depth_batch` for FM/MBD, elastic-FPCA inner N-loops), each noting the thread-safe RNG-seeding pattern where relevant
  3. The assessment records where banding/parallelism is opt-in rather than automatic and the measured cost of the default (unaccelerated) path
  4. This slice's backlog entries (parallelization opportunities) are drafted with function/current-cost/root-cause fields

**Plans**: TBD

### Phase 6: Conditional SVD Library Comparison

**Goal**: Decide, on evidence, whether swapping nalgebra SVD for faer is warranted at fdars' real problem sizes
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: PERF-06
**Success Criteria** (what must be TRUE):

  1. A documented go/no-go decision references the Phase 4 evidence: the comparison is performed only if SVD is a significant share of FPCA runtime and copy is not the dominant cost; otherwise the report records "not warranted" with the supporting numbers
  2. If performed, a nalgebra-vs-faer SVD benchmark table at fdars' real sizes (from the Phase 1 workload matrix) records speedup, conversion cost, and crossover point, saved under `.planning/research/bench/`
  3. If performed, a faer adoption note assesses maintenance-burden / stability risk so the backlog item reflects integration ROI, not just raw speed
  4. This slice's outcome (a backlog item or an explicit "no action, with evidence" record) is drafted for Phase 9

**Plans**: TBD

### Phase 7: scikit-fda Capability Enumeration

**Goal**: Build the scikit-fda side of the comparison — a versioned, area-organized capability inventory
**Mode:** mvp
**Depends on**: Nothing (independent of the performance track)
**Requirements**: GAP-01
**Success Criteria** (what must be TRUE):

  1. scikit-fda's public capability surface is enumerated by area (representation, preprocessing, exploratory, ML, inference, misc) in the report
  2. The exact compared scikit-fda version is pinned and recorded (verified against PyPI `__version__`, baseline 0.10.1) in the methodology
  3. Enumeration is capability-oriented, not raw API-name counting: fit/predict/transform families are grouped by user task to avoid the 2–3× gap inflation of Pitfall 9
  4. A one-page design-goal filter is written (in-scope numeric algorithms vs out-of-scope plotting/IO/sklearn-pipeline) to be applied in Phase 8

**Plans**: TBD

### Phase 8: Capability Parity Matrix & Categorization

**Goal**: Map fdars against scikit-fda by capability and turn it into a categorized, strengths-aware gap picture
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: GAP-02, GAP-03, GAP-04
**Success Criteria** (what must be TRUE):

  1. A parity matrix marks each capability present / partial / absent, mapped by **capability not API name**, with an "fdars equivalent searched" note for every gap candidate (using `.planning/codebase/STRUCTURE.md` as the fdars side)
  2. Each gap is categorized table-stakes / differentiator / out-of-scope, with the design-goal filter applied and separate in-scope vs out-of-scope gap counts (plotting/IO excluded from the actionable count, per Pitfall 14)
  3. fdars capabilities that exceed scikit-fda are documented (model explainability, SPM/control charts, seasonal decomposition, streaming depth) so the audit reflects strengths, not only gaps
  4. Fragile/known-bug areas from CONCERNS.md (e.g. B-spline CV GH #33, elastic alignment) carry an "accuracy verified?" note rather than a bare ✓, and gap backlog entries are drafted with area/current-gap/root-cause fields

**Plans**: TBD

### Phase 9: Consolidated Report & Prioritized Backlog

**Goal**: Aggregate all performance and gap findings into one report with a value-ranked, promotion-ready backlog
**Mode:** mvp
**Depends on**: Phase 3, Phase 4, Phase 5, Phase 6, Phase 8
**Requirements**: RPT-01, RPT-02, RPT-03
**Success Criteria** (what must be TRUE):

  1. A consolidated audit report combines the performance findings and the gap findings, with reproducible evidence attached to each finding (benchmark numbers, allocation counts, or doc/source references under `.planning/research/bench/`)
  2. A prioritized backlog phrases each item as a candidate requirement/phase, ranked by **user value not ease** (heuristic `value / sqrt(effort)`), ready to promote via `/gsd-new-milestone`
  3. Every backlog item passes a completeness checklist: location/area, current cost or gap, root cause, proposed direction, severity (P1/P2/P3), effort estimate (S/M/L), and an evidence link
  4. At least one P1 item exists, no top-10 item is a cosmetic convenience-only entry, and the report's methodology section documents build-mode/feature-flag discipline and infra-vs-code failure triage

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Measurement Discipline & Baselines | 2/2 | Complete    | 2026-08-07 |
| 2. Static Hot-Path Analysis | 2/2 | Complete    | 2026-08-07 |
| 3. Elastic Alignment Hot Path | 2/2 | Complete    | 2026-08-08 |
| 4. FPCA/SVD & Allocation Audit | 3/3 | Complete    | 2026-08-08 |
| 5. Parallelism Gap Assessment | 0/0 | Not started | - |
| 6. Conditional SVD Library Comparison | 0/0 | Not started | - |
| 7. scikit-fda Capability Enumeration | 0/0 | Not started | - |
| 8. Capability Parity Matrix & Categorization | 0/0 | Not started | - |
| 9. Consolidated Report & Prioritized Backlog | 0/0 | Not started | - |

## Coverage

All 13 v1 requirements mapped to exactly one phase. No orphans, no duplicates.

| Phase | Requirements |
|-------|--------------|
| 1 | PERF-02 |
| 2 | PERF-01 |
| 3 | PERF-03 (elastic alignment share) |
| 4 | PERF-03 (FPCA share), PERF-04 |
| 5 | PERF-05 |
| 6 | PERF-06 |
| 7 | GAP-01 |
| 8 | GAP-02, GAP-03, GAP-04 |
| 9 | RPT-01, RPT-02, RPT-03 |

> PERF-03 (benchmark the top hot-path suspects) is delivered across Phases 3 and 4 by hot path — Phase 3 owns elastic alignment, Phase 4 owns FPCA/SVD. Each phase produces the release-build, feature-tagged criterion tables for its own suspects; no other requirement is split.

---
*Roadmap created: 2026-08-07*
