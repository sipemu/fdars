# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 — Functional Inference Suite** — Phases 20–21 (shipped 2026-08-16) — [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 — Table-Stakes Quick Wins** — Phases 22–23 (shipped 2026-08-16) — [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 — Functional Regression Completeness** — Phases 24–25 (shipped 2026-08-17) — [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 — PACE Sparse FPCA & Elastic Multinomial** — Phases 26–27 (shipped 2026-08-19) — [archive](milestones/v0.22.0-ROADMAP.md)
- ✅ **v0.23.0 — Depth, Outliers & Interval Inference** — Phases 28–30 (shipped 2026-08-20) — [archive](milestones/v0.23.0-ROADMAP.md)
- ✅ **v0.24.0 — Functional Regression & Clustering Breadth** — Phases 31–33 (shipped 2026-08-20) — [archive](milestones/v0.24.0-ROADMAP.md)
- ✅ **v0.25.0 — Serial Dependence, Representation & Density Breadth** — Phases 34–36 (shipped 2026-08-21) — [archive](milestones/v0.25.0-ROADMAP.md)
- ✅ **v0.26.0 — FPCA Breadth & Sparse Covariance** — Phases 37–38 (shipped 2026-08-21) — [archive](milestones/v0.26.0-ROADMAP.md)
- ✅ **v0.27.0 — Functional Time Series & Fréchet Regression** — Phases 39–40 (shipped 2026-08-22) — [archive](milestones/v0.27.0-ROADMAP.md)
- ✅ **v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression** — Phases 41–42 (shipped 2026-08-23) — [archive](milestones/v0.28.0-ROADMAP.md)
- ✅ **v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering** — Phases 43–45 (shipped 2026-08-30) — [archive](milestones/v0.29.0-ROADMAP.md)
- 🚧 **v0.30.0 — Performance & Consolidation Pass** — Phases 46–51 (in progress)

## Overview

v0.30.0 is the **first internally-driven implementation milestone** now that both parity backlogs (scikit-fda v0.15–0.17, R v0.19–0.29) are exhausted. It pivots the crate from **breadth to depth**: profile the whole crate, then land real, **behavior-preserving** improvements across four fronts — hot-path performance, code duplication, additive API consistency, and benchmark coverage of the newer modules.

The milestone is **measure-first**: an opening whole-crate profiling/measurement phase (Phase 46) produces three ranked inventories — a hot-path optimization target list, a duplication/consolidation inventory, and an API-inconsistency inventory — and those rankings are **hard prerequisites** that drive every downstream implementation phase. No PERF/CONS/API work can be planned concretely before Phase 46 produces its evidence.

Every change is **behavior-preserving** (numeric outputs unchanged, or provably-equivalent within documented tolerance; proven by existing tests + before/after criterion benchmarks) and the public API stays **additive/non-breaking** (add a unified alternative + `#[deprecated]`, never remove — protecting R/WASM bindings, 28 examples, and external callers). The **no-new-crate-dependency** convention carries forward (profiling uses existing dev-deps only: criterion, feature-gated `dhat-heap`). Phase numbering continues from v0.29.0 (43/44/45) → **Phase 46 onward**.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (46.1, 46.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression (Phases 41–42) — SHIPPED 2026-08-23</summary>

- [x] Phase 41: Spectral Functional Time Series (FTS-03, 2 plans) — new `fts/spectral.rs` (`spectral_density`, `dpca`, `dpca_reconstruct`) + `simulation.rs` (`sim_fvarma`, `sim_farma`)
- [x] Phase 42: Object-Data Fréchet Regression (FRE-02, 3 plans) — new `frechet/spaces/` + generic `frechet_*_space` solvers

Full detail: [milestones/v0.28.0-ROADMAP.md](milestones/v0.28.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering (Phases 43–45) — SHIPPED 2026-08-30</summary>

- [x] Phase 43: Boosting / Bayesian Functional Regression (REG-06, 5 plans) — new `boosting_regression.rs`
- [x] Phase 44: FEM/PDE Smoothing on Irregular 2D Domains (REP-02) — new `fem_smoothing.rs` + additive `smooth_basis.rs` smoothers
- [x] Phase 45: Functional Co-Clustering (funLBM latent-block) (CLUS-02, 2 plans) — new `coclustering.rs`

Milestone audit PASSED 12/12 requirements. Full detail: [milestones/v0.29.0-ROADMAP.md](milestones/v0.29.0-ROADMAP.md)

</details>

### 🚧 v0.30.0 — Performance & Consolidation Pass (Phases 46–51) — IN PROGRESS

**Milestone Goal:** Profile the whole crate, then land behavior-preserving improvements on four fronts — hot-path performance, code dedup, additive API consolidation, and benchmark coverage — driven by same-milestone profiling evidence rather than an external gap-audit.

- [ ] **Phase 46: Whole-Crate Profiling & Measurement** - Produce the three ranked inventories (hot-path targets, duplication, API inconsistencies) that drive every downstream phase
- [ ] **Phase 47: Hot-Path & Allocation Performance** - Optimize the top-ranked compute-bound paths and allocation hotspots with before/after benchmark + equivalence proof
- [ ] **Phase 48: Parallelism-Gap Closure** - Close feature-gated rayon parallelism gaps in the newer subsystems, equivalence-tested with payback-threshold guards
- [ ] **Phase 49: Code Consolidation / Dedup** - Factor duplicated numerical + statistical-test machinery into shared `pub(crate)` helpers; migrate all call sites, behavior unchanged
- [ ] **Phase 50: Additive API-Surface Consolidation** - Unify inconsistent config/result patterns and redundant public functions via unified alternatives + `#[deprecated]`; zero breakage to existing callers
- [ ] **Phase 51: Benchmark Coverage & Regression Guards** - Add criterion benches for the currently-unbenchmarked new modules and commit the PERF-proof benches as permanent regression guards

**Execution order (dependency-driven):** Phase 46 is a **hard prerequisite** for 47–50 — its ranked outputs (PROF-01 → PERF, PROF-02 → CONS, PROF-03 → API) are what make those phases plannable. 47 → 48 are both PERF work and share the perf-benchmark harness (47 first for the highest-leverage compute/allocation wins, then 48's parallelism). 49 (CONS) and 50 (API) each depend only on Phase 46 and are otherwise independent of the PERF phases and of each other — they may be planned in either order after 46. Phase 51 runs last: BENCH-01 (new-module coverage) is largely independent, but BENCH-02 (regression guards for the PERF wins) depends on 47/48 landing first, so the whole benchmark-coverage phase is placed at the end.

## Phase Details

### Phase 46: Whole-Crate Profiling & Measurement
**Goal**: Produce the evidence base for the whole milestone — a ranked hot-path optimization target list, a duplication/consolidation inventory, and an API-inconsistency inventory — so every downstream implementation phase optimizes real bottlenecks, dedups real duplication, and unifies real inconsistencies rather than guessing.
**Depends on**: Nothing (first phase of the milestone)
**Requirements**: PROF-01, PROF-02, PROF-03
**Success Criteria** (what must be TRUE):
  1. A whole-crate criterion + allocation profiling pass exists that produces a **ranked hot-path optimization target list** (N×M-scaled where relevant), prioritizing the reuse-first v0.19–v0.29 subsystems (`inference`, `fts`, `frechet`, `density_fda`, `fpca_variants`, `face`, `boosting_regression`, `fem_smoothing`, `coclustering`), each target carrying a real criterion/allocation number and a source anchor.
  2. A **duplication/consolidation inventory** exists cataloging machinery repeated across modules with source anchors (file:line), ranked by dedup leverage — enough to drive Phase 49 concretely.
  3. An **API-inconsistency inventory** exists cataloging config/result patterns and redundant public functions that are candidates for additive unification, each with a proposed canonical form noted — enough to drive Phase 50 concretely.
  4. The profiling pass uses only existing dev-dependencies (criterion, feature-gated `dhat-heap`) — no new crate dependency — and makes zero behavior-changing edits to `fdars-core/src/` algorithms.
**Plans**: 5 plans
- [ ] 46-01-PLAN.md — Wave 0 setup + baseline-green + tracer probe-bench pipeline (fpca_variants end-to-end) + PROF-01 doc skeleton (PROF-01)
- [ ] 46-02-PLAN.md — Expand probe benches to remaining 8 subsystems + dhat alloc probes + complete PROF-01 ranked inventory + remove throwaway benches (PROF-01)
- [ ] 46-03-PLAN.md — PROF-02 duplication/consolidation inventory (static grep analysis, ranked by dedup leverage) (PROF-02)
- [ ] 46-04-PLAN.md — PROF-03 API-inconsistency inventory (static analysis, proposed canonical forms, additive-safe classification) (PROF-03)
- [ ] 46-05-PLAN.md — PROF-00 summary tying inventories together + final zero-behavior-change gate + validation sign-off (PROF-01/02/03)
**UI hint**: no

### Phase 47: Hot-Path & Allocation Performance
**Goal**: A user's compute-bound and allocation-heavy workloads run measurably faster while producing numerically-identical (or provably-equivalent within tolerance) results — the top-ranked hot paths and allocation hotspots from Phase 46 are optimized with benchmark proof.
**Depends on**: Phase 46 (consumes PROF-01's ranked hot-path + allocation target list)
**Requirements**: PERF-01, PERF-02
**Success Criteria** (what must be TRUE):
  1. Each top-ranked hot path from PROF-01 is optimized, proven by a **before/after criterion benchmark showing measurable improvement**, with the existing test suite green (numeric outputs unchanged or provably-equivalent within documented tolerance).
  2. Allocation hotspots identified by PROF-01 (unnecessary `FdMatrix`↔`DMatrix` copies, per-iteration allocations in hot loops) are reduced, verified by an **allocation profile** (feature-gated `dhat-heap`) showing fewer/smaller allocations plus equivalence tests confirming unchanged output.
  3. Every optimization is behavior-preserving and additive: no existing public signature changes, and any `linalg`/non-`linalg` split path keeps both branches producing equivalent results.
  4. No new crate dependency is introduced.
**Plans**: TBD
**UI hint**: no

### Phase 48: Parallelism-Gap Closure
**Goal**: A user with the `parallel` feature enabled gets multi-threaded speedups on the newer subsystems that previously ran only sequentially, with no small-input regression and bit-equivalent results versus the sequential path.
**Depends on**: Phase 46 (parallelism gaps surfaced by PROF-01), Phase 47 (shares the perf-benchmark harness; sequenced after the compute/allocation wins)
**Requirements**: PERF-03
**Success Criteria** (what must be TRUE):
  1. Parallelism gaps identified in the newer subsystems are closed with **feature-gated rayon** via the existing `parallel.rs` macros (`iter_maybe_parallel!` etc.), so the `parallel`-on path is faster on large inputs (criterion thread-scaling evidence).
  2. Each parallelized loop is **equivalence-tested against the sequential path** (bit-identical, tested with the `parallel` feature both ON and OFF).
  3. Where a small-input regression is possible, a **payback-threshold N guard** (outer-if) prevents parallel dispatch below the measured break-even size — matching the v0.17.0 `SCORES_PARALLEL_THRESHOLD` precedent.
  4. Per-thread RNG seeding determinism (`StdRng::seed_from_u64(seed + k)`) is preserved for any parallelized randomized loop; no new crate dependency.
**Plans**: TBD
**UI hint**: no

### Phase 49: Code Consolidation / Dedup
**Goal**: Duplicated numerical and statistical-test machinery scattered across the v0.19–v0.29 modules is factored into shared `pub(crate)` helpers with every call site migrated — reducing surface area and drift risk while leaving all observable behavior unchanged.
**Depends on**: Phase 46 (consumes PROF-02's ranked duplication/consolidation inventory)
**Requirements**: CONS-01, CONS-02
**Success Criteria** (what must be TRUE):
  1. Duplicated **numerical machinery** cataloged in PROF-02 (e.g. FPCA scoring, Cholesky/ridge solves, Simpson/quadrature weights, χ²/F survival functions, SVD sign-fix) is factored into shared `pub(crate)` helpers and **all call sites are migrated** to them.
  2. Duplicated **statistical-test scaffolding** (permutation-test loops, per-thread seeded-RNG patterns) is consolidated into a reusable helper, with call sites migrated and **determinism/reproducibility preserved** (seeded results unchanged).
  3. The full test suite is green after migration and behavior is unchanged — no public signature changed, no numeric output altered.
  4. No new crate dependency is introduced.
**Plans**: TBD
**UI hint**: no

### Phase 50: Additive API-Surface Consolidation
**Goal**: A user gets a single canonical, consistent entry point for previously-inconsistent config/result patterns and redundant public functions — with the old forms still compiling and passing (now emitting deprecation warnings), so R/WASM bindings, the 28 examples, and external callers all keep working with zero breakage.
**Depends on**: Phase 46 (consumes PROF-03's API-inconsistency inventory)
**Requirements**: API-01, API-02, API-03
**Success Criteria** (what must be TRUE):
  1. Inconsistent config/result patterns from PROF-03 gain **unified alternatives**; the previous forms are marked `#[deprecated]` with a `note` pointing to the replacement; both the deprecated and unified paths continue to compile and pass tests.
  2. Redundant public functions gain a **single canonical entry point**; superseded functions are marked `#[deprecated]` (never removed) and the crate-root re-export surface is tightened accordingly — **no existing public signature is changed**.
  3. `cargo build` / `cargo test`, the **28 examples**, and the R/WASM binding call sites all still pass with **deprecation warnings only** — zero breakage to existing callers.
  4. No new crate dependency is introduced.
**Plans**: TBD
**UI hint**: no

### Phase 51: Benchmark Coverage & Regression Guards
**Goal**: The criterion suite covers the previously-unbenchmarked new modules, and the benchmarks that proved the PERF wins are committed as permanent regression guards with documented before/after numbers — so future changes can detect both new bottlenecks and regressions of the wins landed this milestone.
**Depends on**: Phase 46 (module list), Phase 47 + Phase 48 (BENCH-02 guards the PERF-01/02/03 wins landed there)
**Requirements**: BENCH-01, BENCH-02
**Success Criteria** (what must be TRUE):
  1. New criterion benchmarks — registered as `[[bench]]` entries and runnable via `cargo bench` — cover the currently-unbenchmarked new modules: `fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`, `density_fda`, `inference`, `fpca_variants`, `face`.
  2. The benchmarks used to prove the PERF-01/02/03 wins are **committed as permanent regression guards**, with the before/after numbers documented so future changes can detect regressions.
  3. The full clippy gate (`cargo clippy --all-targets --features linalg,parallel -- -D warnings`, which lints bench code) stays green with the new bench entries; no new crate dependency.
**Plans**: TBD
**UI hint**: no

## Progress

**Execution Order:**
Phases execute in numeric order: 46 → 47 → 48 → 49 → 50 → 51. (46 gates all; 47 precedes 48; 49 and 50 each depend only on 46; 51 last so its BENCH-02 guards follow the PERF phases.)

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 46. Whole-Crate Profiling & Measurement | 0/5 | Planned | - |
| 47. Hot-Path & Allocation Performance | 0/TBD | Not started | - |
| 48. Parallelism-Gap Closure | 0/TBD | Not started | - |
| 49. Code Consolidation / Dedup | 0/TBD | Not started | - |
| 50. Additive API-Surface Consolidation | 0/TBD | Not started | - |
| 51. Benchmark Coverage & Regression Guards | 0/TBD | Not started | - |

All phases through v0.29.0 are shipped and archived under `milestones/`.
