# fdars

## What This Is

fdars is a mature Rust functional-data-analysis (FDA) library (crate `fdars-core`, v0.14.0) with broad algorithm coverage — regression, classification, clustering, depth measures, elastic shape analysis, seasonal decomposition, statistical process monitoring, and model explainability — plus WASM/JS and R bindings. This milestone is an **audit**: proactively review execution performance and map functionality gaps against Python's scikit-fda, producing a report and a prioritized, GSD-ready backlog for future work.

## Core Value

Produce an evidence-backed picture of where fdars is slow and what it is missing (relative to scikit-fda), turned into a prioritized backlog — so future milestones target the highest-leverage performance and functionality work first.

## Requirements

### Validated

<!-- Inferred from existing codebase (see .planning/codebase/). These already ship and are relied upon. -->

- ✓ Scalar-on-function regression (linear `fregre_lm`, functional logistic) — existing
- ✓ Functional PCA (`FpcaResult`, 1D + 2D FOSR with tensor-product penalty) — existing
- ✓ Classification (LDA, QDA, kNN, kernel, DD) with FPC-space fitting — existing
- ✓ Depth measures (Fraiman-Muniz, modal, band, random projection) + streaming depth — existing
- ✓ Elastic curve registration & shape analysis (elastic regression, elastic FPCA, PDO curves) — existing
- ✓ Seasonal analysis (period detection, peak finding, STL decomposition) — existing
- ✓ Statistical process monitoring (control charts, outlier detection) — existing
- ✓ Model explainability (PDP, SHAP, LIME, ALE, importance) via generic `FpcPredictor` trait — existing
- ✓ Clustering (GMM) and irregular functional data support — existing
- ✓ Core infrastructure: column-major `FdMatrix`, `Result`-based error handling, feature-gated rayon parallelism, ~1,650 tests, 8 criterion benchmarks, 28 examples, WASM/JS + R bindings — existing
- ✓ **Static hot-path analysis** — zero-cost per-module bottleneck-candidate map (complexity in N/M, allocation hotspots incl. 8 `to_dmatrix()` SVD copies + 14 `from_column_slice` basis sites, parallelism gaps, feature-gate annotations) in `.planning/research/AUDIT-REPORT.md` — Validated in Phase 2 (PERF-01)
- ✓ **FPCA/SVD & allocation audit** — criterion 6-cell N×M grid + elastic-FPCA cells and a dhat allocation baseline (feature-gated `dhat-heap`) quantify the `FdMatrix→DMatrix` SVD-copy as ~0.14–0.17% of wall-clock: SVD compute dominates (~99.8%). Report carries the SVD-vs-copy split, a **Phase-6 GO** verdict (faer-vs-nalgebra comparison warranted), and a GSD-ready backlog in `.planning/research/AUDIT-REPORT.md` — Validated in Phase 4 (PERF-03, PERF-04)
- ✓ **Parallelism gap assessment** — criterion rayon thread-scaling (heavy `karcher_mean` + light `StreamingFraimanMuniz::depth_batch` sentinels, RAYON_NUM_THREADS ∈ {1,2,4,8}) plus payback-threshold N per target (karcher N≤10, streaming N≈50), a 5-candidate safe-to-parallelize list with source anchors, the default unaccelerated-path cost (rayon-off ~10×), and a GSD-ready backlog (P5-1..P5-4) in `.planning/research/AUDIT-REPORT.md` §Phase 5. Governor unpinned → multi-thread cells flagged LOW-CONFIDENCE. Zero `fdars-core/src/` edits (audit-only). — Validated in Phase 5 (PERF-05)
- ✓ **Conditional SVD library comparison** — criterion 7-cell N×M grid measuring faer `thin_svd` (zero-copy `MatRef` view) vs nalgebra `SVD::new` (clone-then-SVD) at fdars' real FPCA sizes, with a `svd_equivalence` integration test confirming agreement within 1e-10. faer measured **1.8–4.1× faster** (primary cell N=500,M=200: 1.8×), zero-copy conversion costs ~3.5–7.7 ns. Report §Phase 6 carries the GO verdict, comparison table, faer adoption/maintenance-risk note, and a GSD-ready backlog item **P6-1** (swap to faer `thin_svd`, P2/S-effort borderline with explicit downgrade condition). Zero `fdars-core/src/` edits (audit-only). — Validated in Phase 6 (PERF-06)
- ✓ **scikit-fda capability enumeration** — versioned (skfda **0.10.1**, runtime-verified `__version__`), area-organized capability inventory in `.planning/research/AUDIT-REPORT.md` §Phase 7: six areas (representation, preprocessing, exploratory, ML, inference, misc) enumerated capability-first (fit/predict/transform collapsed per row, Pitfall 9 avoided), plus a one-page Design-Goal Filter separating **129 in-scope** (numeric algorithms + API-ergonomics) from **32 out-of-scope** (plotting + IO) = **161 total**. This is the scikit-fda side of the parity comparison Phase 8 will map fdars against. Zero `fdars-core/src/` edits (audit-only). — Validated in Phase 7 (GAP-01)
- ✓ **Capability parity matrix & categorization** — the fdars-vs-scikit-fda parity comparison in `.planning/research/AUDIT-REPORT.md` §Phase 8: a single verdict rubric (present/partial/absent, D-01) and category rubric (table-stakes/differentiator/out-of-scope, D-03) driving **six area parity tables** (141 capability rows, 59 present / 19 partial / 63 absent), mapped by capability not API name with 83 "searched fdars for:" notes. Known-bug rows carry "present — accuracy NOT verified" flags citing fix commits (`2fb6d3c9` #33, `6ed62398` #34, `ec17d138` GMM). Separates **82 actionable in-scope gaps** from 32 out-of-scope (plotting/IO). A **reverse-parity strengths sweep** documents 30 fdars-only capabilities (explainability, SPM, seasonal, streaming depth + 26 more from a full module-map walk). A drafted **UNRANKED** gap backlog (21 entries + a D-02a numerical-accuracy validation item) feeds Phase 9 value ranking. Verified 9/9 must-haves; zero `fdars-core/src/` edits (audit-only). — Validated in Phase 8 (GAP-02, GAP-03, GAP-04)

### Active

<!-- This milestone's scope. Analysis/audit only — no production code changes required. -->

- [ ] **Benchmark confirmation** — extend/run criterion benchmarks on representative inputs to measure and confirm the top performance suspects with real numbers
- [ ] **Consolidated audit report** — a written report combining benchmark results and functionality-gap findings
- [ ] **GSD-ready prioritized backlog** — findings phrased as candidate requirements/phases, ranked by leverage, ready to promote via `/gsd-new-milestone`

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- Implementing the performance fixes or missing features found — deferred to future milestones; this milestone is audit-only to keep scope bounded and decisions evidence-driven
- Parity with R fda.usc / fda as the baseline — scikit-fda was chosen as the single comparison yardstick to avoid diluting the analysis
- Plotting/visualization parity with scikit-fda — a numeric Rust library does not need matplotlib-style output; treat as low-priority unless the audit surfaces a concrete need

## Context

- Brownfield: fdars is at v0.14.0 with a complete codebase map in `.planning/codebase/`. Architecture is a modular monolith with a layered design (public API → domain modules → shared infra → external deps).
- Existing benchmarking foundation: criterion 0.5 is already a dev-dependency with 8 benchmarks and HTML reports, so the "real benchmarks" work extends existing infrastructure rather than starting from scratch.
- Known perf anti-patterns already documented in `.planning/codebase/ARCHITECTURE.md`: dense matrix copy in SVD (FdMatrix ↔ nalgebra DMatrix round-trips), unvalidated slice access in hot loops, NaN-handling inconsistency.
- Stack: Rust 2021, MSRV 1.81 (1.84 for `linalg`), nalgebra 0.33, rayon 1.10, rustfft 6.2, faer 0.23 (optional), anofox-regression 0.4.
- Comparison target scikit-fda is a Python library — gap analysis is API/capability comparison, not a runtime benchmark against Python.

## Constraints

- **Scope**: Audit-only milestone — deliverables are a report + backlog, not code changes to `fdars-core`.
- **Baseline**: scikit-fda is the sole functionality-gap yardstick for this milestone.
- **Tech stack**: Benchmarks use the existing criterion 0.5 harness; performance reasoning must respect the column-major `FdMatrix` layout and feature-gated parallelism model.
- **Output**: Backlog items must be GSD-ready (phrased as candidate requirements/phases) so they can be promoted into future milestones.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| scikit-fda as the gap-analysis baseline (not R fda.usc/fda) | Single, modern Python reference keeps the comparison focused | — Pending |
| Audit-only milestone (report + backlog, no production code) | Bounds scope; makes future implementation decisions evidence-driven | — Pending |
| Performance measured via static analysis + real benchmarks | Static pass finds candidates cheaply; criterion numbers confirm the real bottlenecks | ✓ Validated — Phases 1–6 (SVD dominates FPCA ~99.8%, copy ~0.15%; faer 1.8–4.1× over nalgebra) |
| Gap-analysis breadth decided by findings ("start broad, deep-dive where warranted") | Avoids over-investing in low-value areas before knowing where the gaps are | — Pending |
| Backlog phrased as GSD-ready requirements/phases | Lets findings flow straight into `/gsd-new-milestone` without rework | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-08-09 — Phase 8 (Capability Parity Matrix & Categorization) complete: six area parity tables (141 rows, 59 present / 19 partial / 63 absent) mapping fdars vs scikit-fda 0.10.1 by capability, both rubrics (D-01 verdict, D-03 category), 82 actionable in-scope gaps separated from 32 out-of-scope, a 30-row reverse-parity strengths sweep, and a drafted UNRANKED gap backlog (21 entries + D-02a accuracy-validation item) feeding Phase 9; known-bug rows accuracy-flagged; verified 9/9; zero src edits (GAP-02/03/04). Prior: Phase 7 scikit-fda capability enumeration (GAP-01, skfda 0.10.1, 129 in-scope / 32 out-of-scope); Phase 6 conditional SVD comparison (PERF-06, faer 1.8–4.1× faster, P6-1); Phase 5 parallelism gap assessment (PERF-05); Phase 4 FPCA/SVD & allocation audit (PERF-03/04, Phase-6 GO); Phase 2 static hot-path map (PERF-01); Phase 1 benchmark apparatus + baselines (PERF-02).*
