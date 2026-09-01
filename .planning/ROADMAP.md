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
- ✅ **v0.30.0 — Performance & Consolidation Pass** — Phases 46–51 (shipped 2026-09-01) — [archive](milestones/v0.30.0-ROADMAP.md)

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

<details>
<summary>✅ v0.30.0 — Performance & Consolidation Pass (Phases 46–51) — SHIPPED 2026-09-01</summary>

First internally-driven milestone (both parity backlogs exhausted): measure-first, behavior-preserving depth work. Phase 46 profiling produced three ranked inventories driving 47–51.

- [x] Phase 46: Whole-Crate Profiling & Measurement (PROF-01/02/03, 5 plans) — ranked hot-path/dedup/API inventories
- [x] Phase 47: Hot-Path & Allocation Performance (PERF-01/02, 4 plans) — face_covariance −80.7% wall, dpca −54% alloc blocks; bit-identical
- [x] Phase 48: Parallelism-Gap Closure (PERF-03, 3 plans) — frechet_anova 9.9×, co_cluster 6.4× thread-scaling; payback guards
- [x] Phase 49: Code Consolidation / Dedup (CONS-01/02, 5 plans) — χ²/gamma → distributions.rs, seed_for_thread, permutation_pvalue, SVD sign-core; −358 LOC; bit-identical
- [x] Phase 50: Additive API-Surface Consolidation (API-01/02/03, 3 plans) — 3 Default impls, fanova_seeded, Dim + 5 dispatchers, 6 #[deprecated]; 28 examples + wasm compile
- [x] Phase 51: Benchmark Coverage & Regression Guards (BENCH-01/02, 4 plans) — 9 new module benches + BENCH-RESULTS.md ledger

Milestone audit: **tech_debt** (13/13 requirements satisfied, 6/6 phases verified passed, cross-phase integration SOUND; deferred: REL-01 version bump/publish, APIB-01 breaking removals, Nyquist validate-phase reconciliation for 49/50/51). Full detail: [milestones/v0.30.0-ROADMAP.md](milestones/v0.30.0-ROADMAP.md)

</details>

## Progress

All phases through **v0.30.0 are shipped and archived** under `milestones/`. The crate remains at version 0.29.0 — the v0.30.0 version bump + `cargo publish` + `v0.30.0` tag is the deferred operator ship step (REL-01), consciously kept out of the audit/perf milestone to avoid a phantom crates.io publish.

Next: `/gsd-new-milestone` (questioning → research → requirements → roadmap).
