# Requirements: fdars — v0.30.0 Performance & Consolidation Pass

**Defined:** 2026-08-30
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps. With both parity backlogs (scikit-fda + R) exhausted, this milestone pivots from breadth to depth — profile the whole crate, then land real behavior-preserving improvements across performance, code duplication, additive API consistency, and benchmark coverage.

## v1 Requirements

Requirements for milestone v0.30.0. Each maps to a roadmap phase. **Measure-first:** the Profiling category (PROF) runs first and its ranked outputs drive the PERF / CONS / API categories.

### Profiling & Measurement (PROF)

- [ ] **PROF-01**: A whole-crate criterion + allocation profiling pass produces a ranked hot-path optimization target list (N×M-scaled where relevant), prioritizing the reuse-first v0.19–v0.29 subsystems (`inference`, `fts`, `frechet`, `density_fda`, `fpca_variants`, `face`, `boosting_regression`, `fem_smoothing`, `coclustering`).
- [ ] **PROF-02**: The profiling pass produces a duplication/consolidation inventory — machinery repeated across modules cataloged with source anchors (file:line), ranked by dedup leverage.
- [ ] **PROF-03**: The profiling pass produces an API-inconsistency inventory — config/result patterns and redundant public functions across modules that are candidates for additive unification, with the proposed canonical form noted per item.

### Hot-Path Performance (PERF)

- [ ] **PERF-01**: Each top-ranked hot path from PROF-01 is optimized behavior-preservingly, proven by a before/after criterion benchmark showing measurable improvement, with existing tests green (numeric outputs unchanged or provably-equivalent within documented tolerance).
- [ ] **PERF-02**: Allocation hotspots identified by PROF-01 (unnecessary `FdMatrix`↔`DMatrix` copies, per-iteration allocations in hot loops) are reduced, verified by an allocation profile (feature-gated `dhat-heap`) and equivalence tests.
- [ ] **PERF-03**: Parallelism gaps identified in the newer subsystems are closed with feature-gated rayon (via the existing `parallel.rs` macros), equivalence-tested vs the sequential path, with a payback-threshold N guard where a small-input regression is possible.

### Code Consolidation / Dedup (CONS)

- [ ] **CONS-01**: Duplicated numerical machinery cataloged in PROF-02 (e.g. FPCA scoring, Cholesky/ridge solves, Simpson/quadrature weights, χ² / F survival functions, SVD sign-fix) is factored into shared `pub(crate)` helpers; all call sites migrated; behavior unchanged (full suite green).
- [ ] **CONS-02**: Duplicated statistical-test scaffolding (permutation-test loops, per-thread seeded-RNG patterns) is consolidated into a reusable helper; call sites migrated; determinism/reproducibility preserved.

### API Surface Consolidation — additive only (API)

- [ ] **API-01**: Inconsistent config/result patterns identified in PROF-03 gain unified alternatives; the previous forms are marked `#[deprecated]` with a `note` pointing to the replacement; both the deprecated and unified paths continue to compile and pass tests.
- [ ] **API-02**: Redundant public functions gain a single canonical entry point; superseded functions are marked `#[deprecated]` (never removed); the crate-root re-export surface is tightened accordingly. No existing public signature is changed.
- [ ] **API-03**: `cargo build` / `cargo test`, the 28 examples, and the R/WASM binding call sites all still pass after the API work with deprecation warnings only — zero breakage to existing callers.

### Benchmark Coverage (BENCH)

- [ ] **BENCH-01**: New criterion benchmarks (registered as `[[bench]]` entries) cover the currently-unbenchmarked new modules: `fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`, `density_fda`, `inference`, `fpca_variants`, `face`.
- [ ] **BENCH-02**: The benchmarks used to prove PERF-01/02/03 wins are committed as permanent regression guards, with the before/after numbers documented, so future changes can detect regressions.

## Future Requirements

Deferred beyond v0.30.0. Tracked but not in this roadmap.

### API (breaking)

- **APIB-01**: Actually removing the functions/configs deprecated this milestone (a breaking `#[deprecated]`→removed sweep) — deferred to a future breaking release (likely a 1.0-readiness milestone).

### Release

- **REL-01**: Crate version bump + `cargo publish` + git tag folding in v0.29.0 + v0.30.0 changes — a deferred operator ship-time step (per prior-milestone convention).

## Out of Scope

Explicitly excluded for v0.30.0. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Breaking API changes (renaming/removing existing public signatures) | Milestone policy is additive-only: add unified alternative + `#[deprecated]`, never remove — protects R/WASM bindings, 28 examples, external callers |
| New external-feature / algorithm work | Both parity backlogs (scikit-fda + R) are exhausted; this is a depth (perf/consolidation) milestone, not a breadth milestone |
| Numeric-output changes to existing algorithms | Behavior-preserving milestone — optimizations must be equivalent within documented tolerance; changing results is out of scope |
| A fresh external gap-audit (MATLAB/`tidyfun`/Julia) | Candidate for the *next* milestone; this one is internally-driven by profiling evidence |
| New crate dependencies | Carried convention; profiling uses existing dev-deps only (criterion, feature-gated `dhat-heap`) |
| Crate release (version bump / publish / tag) | Deferred operator ship-time step (REL-01), consistent with v0.25–v0.29 precedent |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PROF-01 | Phase 46 | Pending |
| PROF-02 | Phase 46 | Pending |
| PROF-03 | Phase 46 | Pending |
| PERF-01 | Phase 47 | Pending |
| PERF-02 | Phase 47 | Pending |
| PERF-03 | Phase 48 | Pending |
| CONS-01 | Phase 49 | Pending |
| CONS-02 | Phase 49 | Pending |
| API-01 | Phase 50 | Pending |
| API-02 | Phase 50 | Pending |
| API-03 | Phase 50 | Pending |
| BENCH-01 | Phase 51 | Pending |
| BENCH-02 | Phase 51 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13 ✓ (Phases 46–51)
- Unmapped: 0 ✓

**Phase → Requirement summary:**
- Phase 46 (Profiling & Measurement): PROF-01, PROF-02, PROF-03
- Phase 47 (Hot-Path & Allocation Performance): PERF-01, PERF-02
- Phase 48 (Parallelism-Gap Closure): PERF-03
- Phase 49 (Code Consolidation / Dedup): CONS-01, CONS-02
- Phase 50 (Additive API-Surface Consolidation): API-01, API-02, API-03
- Phase 51 (Benchmark Coverage & Regression Guards): BENCH-01, BENCH-02

---
*Requirements defined: 2026-08-30*
*Last updated: 2026-08-30 after roadmap creation (Phases 46–51 mapped; 13/13 requirements covered)*
