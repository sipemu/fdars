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

## Overview

v0.28.0 draws the two now-unblocked score-1.00 (L-effort) items from the v0.18.0 `R-BACKLOG.md`: spectral/frequency-domain functional time series (FTS-03, rank 22) and object-data Fréchet regression across specific metric spaces (FRE-02, rank 23). Both dependencies (FTS-01, FRE-01) shipped in v0.27.0. The milestone is implementation, not audit — real `fdars-core/src/` code, additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports, **no new crate dependency** (FTS-03 reuses the existing `rustfft`; FRE-02 plugs into the shipped FRE-01 solver). Two phases, one requirement-category each, touching disjoint code areas (`fts/spectral.rs` + `simulation.rs` vs `frechet/` metric backends) — plannable/executable in any order or in parallel.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (41.1, 41.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.27.0 — Functional Time Series & Fréchet Regression (Phases 39–40) — SHIPPED 2026-08-22</summary>

- [x] Phase 39: Functional Time-Series Forecasting (FTS-01, 3 plans) — new `fts/forecast.rs`: `ftsm`, `ftsm_forecast`, `ftsm_forecast_multistep`, `ftsm_update`, `fplsr`
- [x] Phase 40: Fréchet / Object-Data Regression (FRE-01, 3 plans) — new `frechet/` module: `MetricSpace`/`WassersteinDensitySpace`, `frechet_mean`/`frechet_variance`, `wasserstein2_distance`, `frechet_global_reg`/`frechet_local_reg`, `frechet_anova`

Full detail: [milestones/v0.27.0-ROADMAP.md](milestones/v0.27.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression (Phases 41–42) — SHIPPED 2026-08-23</summary>

- [x] Phase 41: Spectral Functional Time Series (FTS-03, 2 plans) — new `fts/spectral.rs` (`spectral_density`, `dpca`, `dpca_reconstruct`) + `simulation.rs` (`sim_fvarma`, `sim_farma`)
- [x] Phase 42: Object-Data Fréchet Regression (FRE-02, 3 plans) — new `frechet/spaces/` (`SpdMatrixSpace`/`SpdMetric`, `CorrelationMatrixSpace`, `SphericalSpace`, `NetworkSpace`, `PointProcessSpace`) + generic `frechet_global_reg_space`/`frechet_local_reg_space`/`frechet_anova_space`

Milestone audit PASSED 12/12 requirements. Full detail: [milestones/v0.28.0-ROADMAP.md](milestones/v0.28.0-ROADMAP.md)

</details>

All phases through v0.28.0 are shipped and archived under `milestones/`.
