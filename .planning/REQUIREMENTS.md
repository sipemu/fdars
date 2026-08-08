# Requirements: fdars — Performance & Functionality Audit

**Defined:** 2026-08-07
**Core Value:** Produce an evidence-backed picture of where fdars is slow and what it is missing (vs scikit-fda), turned into a prioritized backlog.

> This is an **audit milestone**. Requirements describe analysis *deliverables* (reports, matrices, benchmarks, a backlog) — not production code changes to `fdars-core`. Implementing the findings is explicitly a future milestone.

## v1 Requirements

Requirements for this audit. Each maps to a roadmap phase.

### Performance Audit (PERF)

- [x] **PERF-01**: Static hot-path analysis documents bottleneck candidates per module with algorithmic complexity in N (curves) and M (grid points), covering at least elastic alignment, FPCA/SVD, depth & distance matrices, CV loops, streaming depth, and smoothing
- [x] **PERF-02**: A representative workload matrix (N × M input sizes) is defined per hot-path module so benchmarks reflect realistic usage, not toy inputs
- [x] **PERF-03**: Criterion benchmarks measure the top hot-path suspects in **release** build with correct feature flags (`linalg`, `parallel`) and `black_box`, producing a results table tagged with feature set and toolchain version
- [x] **PERF-04**: An allocation audit (dhat) quantifies the documented `FdMatrix→DMatrix` SVD-copy overhead (and other allocation hotspots) with a reproducible baseline
- [ ] **PERF-05**: A parallelism assessment measures rayon thread scaling (`RAYON_NUM_THREADS` sweep) and flags sequential loops that are safe to parallelize (e.g. classification CV folds, streaming-depth batch, elastic-FPCA inner loops) and where banding is opt-in rather than automatic
- [ ] **PERF-06**: A conditional nalgebra-vs-faer SVD comparison at fdars' real problem sizes, performed only if benchmarks show SVD to be a significant share of FPCA runtime (else recorded as "not warranted, with evidence")

### Functionality Gap Analysis (GAP)

- [ ] **GAP-01**: scikit-fda's public capability surface is enumerated by area (representation, preprocessing, exploratory, ML, inference, misc), with the exact compared scikit-fda version pinned and recorded
- [ ] **GAP-02**: fdars capabilities are mapped against scikit-fda by **capability** (not API name), producing a parity matrix marking each capability present / partial / absent
- [ ] **GAP-03**: Gaps are categorized as table-stakes vs differentiator vs out-of-scope, with a design-goal filter applied so plotting/IO features are excluded from the actionable gap count
- [ ] **GAP-04**: fdars capabilities that exceed scikit-fda are documented (e.g. model explainability, SPM/control charts, seasonal decomposition, streaming depth) so the audit reflects strengths, not only gaps

### Report & Backlog (RPT)

- [ ] **RPT-01**: A consolidated audit report combines the performance findings and the gap findings, with reproducible evidence attached to each finding (benchmark numbers, allocation counts, or doc references)
- [ ] **RPT-02**: A prioritized, GSD-ready backlog phrases each item as a candidate requirement/phase, ranked by **user value** (not ease), ready to promote via `/gsd-new-milestone`
- [ ] **RPT-03**: Each backlog item carries a completeness checklist — location/area, current cost or gap, root cause, proposed direction, severity, effort estimate, and evidence link

## v2 Requirements

Deferred — become their own future milestones, seeded by the RPT-02 backlog.

### Implementation (IMPL)

- **IMPL-01**: Implement the highest-value performance fixes surfaced by the audit
- **IMPL-02**: Close the highest-value scikit-fda functionality gaps surfaced by the audit

## Out of Scope

Explicitly excluded from this milestone. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Implementing perf fixes / missing features | Audit-only milestone; deliverables are a report + backlog. Fixes are future milestones (see v2) |
| R fda.usc / fda parity | scikit-fda chosen as the single comparison baseline |
| Plotting / visualization parity with scikit-fda | A numeric Rust crate does not need matplotlib-style output; excluded from the actionable gap count |
| Representation type-system refactor (FDataGrid/FDataBasis) | Deep architectural change; may be *noted* in the backlog but not designed or built here |
| Benchmarking fdars against Python scikit-fda at runtime | Gap analysis is capability comparison, not a cross-language speed contest |

## Traceability

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PERF-01 | Phase 2 | Complete |
| PERF-02 | Phase 1 | Complete |
| PERF-03 | Phase 3, Phase 4 | Complete |
| PERF-04 | Phase 4 | Complete |
| PERF-05 | Phase 5 | Pending |
| PERF-06 | Phase 6 | Pending |
| GAP-01 | Phase 7 | Pending |
| GAP-02 | Phase 8 | Pending |
| GAP-03 | Phase 8 | Pending |
| GAP-04 | Phase 8 | Pending |
| RPT-01 | Phase 9 | Pending |
| RPT-02 | Phase 9 | Pending |
| RPT-03 | Phase 9 | Pending |

> PERF-03 spans Phases 3 and 4 by hot path (Phase 3 = elastic alignment suspects, Phase 4 = FPCA/SVD suspects). It is the only requirement mapped to more than one phase, and the two phases own disjoint benchmark targets — no target is duplicated. All other requirements map to exactly one phase.

**Coverage:**

- v1 requirements: 13 total
- Mapped to phases: 13 (100%) ✓
- Unmapped: 0

---
*Requirements defined: 2026-08-07*
*Last updated: 2026-08-07 after roadmap creation (traceability filled, coverage 100%)*
