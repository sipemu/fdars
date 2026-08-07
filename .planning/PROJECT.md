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

### Active

<!-- This milestone's scope. Analysis/audit only — no production code changes required. -->

- [ ] **Performance audit** — static hot-path analysis across the crate (complexity, allocations, SVD copies, parallelism gaps) to surface bottleneck candidates
- [ ] **Benchmark confirmation** — extend/run criterion benchmarks on representative inputs to measure and confirm the top performance suspects with real numbers
- [ ] **scikit-fda gap analysis** — map fdars against scikit-fda's public API (start broad, deep-dive where findings warrant) into a parity matrix + gap findings
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
| Performance measured via static analysis + real benchmarks | Static pass finds candidates cheaply; criterion numbers confirm the real bottlenecks | — Pending |
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
*Last updated: 2026-08-07 — Phase 1 (Measurement Discipline & Baselines) complete: benchmark apparatus + recorded baselines for 6 hot-path modules, workload matrix, and methodology established (PERF-02).*
