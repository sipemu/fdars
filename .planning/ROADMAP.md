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
- 🚧 **v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression** — Phases 41–42 (in progress)

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

All phases through v0.27.0 are shipped and archived under `milestones/`.

### 🚧 v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression (In Progress)

**Milestone Goal:** Draw the two now-unblocked score-1.00 (L-effort) `R-BACKLOG.md` items — spectral functional time series (FTS-03) and object-data Fréchet regression across specific metric spaces (FRE-02) — each by adding `fdars-core/src/` code additively and non-breaking, extending the v0.27.0 foundation.

- [x] **Phase 41: Spectral Functional Time Series** - Frequency-domain FTS — spectral density operator, DPCA (filters + scores + reconstruction), and functional VAR/VMA + FARMA process simulators (completed 2026-08-22)
- [ ] **Phase 42: Object-Data Fréchet Regression** - Non-density `MetricSpace` backends (SPD covariance / correlation / spherical / network / point-process) feeding the shipped FRE-01 regression + ANOVA solver

## Phase Details

### Phase 41: Spectral Functional Time Series

**Goal**: Users can analyze a functional time series in the frequency domain — estimate its spectral density operator, run dynamic FPCA, reconstruct curves from dynamic scores — and simulate functional VAR/VMA and FARMA processes.
**Depends on**: Nothing new in v0.28.0 (builds on the shipped FTS-01/FTS-02 — `fts/forecast.rs`, `fts/acf.rs`). Independent of Phase 42.
**Requirements**: FTS-03-01, FTS-03-02, FTS-03-03, FTS-03-04, FTS-03-05
**Success Criteria** (what must be TRUE):

  1. User can estimate the spectral density operator of a functional time series at a set of Fourier frequencies (the `rustfft`-transformed long-run covariance over lagged autocovariance operators) and receives a numeric per-frequency operator result.
  2. User can compute dynamic functional PCA from that spectral density — obtaining dynamic eigen-filters and dynamic scores as numeric outputs.
  3. User can reconstruct the original curve series from the DPCA dynamic scores via inverse dynamic filtering, and the reconstruction error decreases as more dynamic components are retained.
  4. User can simulate a functional VAR/VMA curve series from user-supplied operator kernels, producing a deterministic (seeded) numeric curve set.
  5. User can simulate a functional ARMA (FARMA) curve series combining AR and MA operator terms, producing a deterministic (seeded) numeric curve set.

**Plans**: 2 plans

Plans:

- [x] 41-01-PLAN.md — spectral density operator + DPCA filters/scores + reconstruction (`fts/spectral.rs`; FTS-03-01/02/03)
- [x] 41-02-PLAN.md — functional VAR/VMA + FARMA simulators (`simulation.rs`; FTS-03-04/05)

### Phase 42: Object-Data Fréchet Regression

**Goal**: Users can run global/local Fréchet regression and Fréchet-ANOVA over non-density object responses by selecting a `MetricSpace` backend — SPD covariance matrices (Frobenius / power / log-Cholesky), correlation matrices, spherical data, networks, or point processes.
**Depends on**: The shipped FRE-01 solver (`frechet/` — `MetricSpace` trait, `frechet_global_reg`/`frechet_local_reg`/`frechet_anova`). Independent of Phase 41.
**Requirements**: FRE-02-01, FRE-02-02, FRE-02-03, FRE-02-04, FRE-02-05, FRE-02-06, FRE-02-07
**Success Criteria** (what must be TRUE):

  1. User can select an SPD covariance-matrix response space (distance + weighted-Fréchet-mean under Frobenius, power, and log-Cholesky metrics) as a `MetricSpace` backend and get numeric distances and Fréchet means.
  2. User can select correlation-matrix, spherical (geodesic exp/log), network, and point-process response spaces — each a `MetricSpace` backend with a numeric distance and a weighted-Fréchet-mean solver.
  3. User can run global and local Fréchet regression over Euclidean predictors with at least one non-density object backend (e.g. SPD covariance matrices), reusing the generic FRE-01 solver, and receive a predicted object response at a new predictor value.
  4. User can run a Fréchet-ANOVA group-difference test over at least one non-density object space, reusing the generic `frechet_anova` machinery, and receive a numeric test statistic and (seeded-permutation) p-value.

**Plans**: TBD

Plans:

- [ ] 42-01: TBD (pin during planning)

## Progress

**Execution Order:**
Phases 41 and 42 are **independent** (disjoint code areas: `fts/spectral.rs` + `simulation.rs` vs `frechet/` metric backends), with no cross-phase hard dependency. They may be planned and executed in any order or in parallel.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 41. Spectral Functional Time Series | v0.28.0 | 2/2 | Complete    | 2026-08-22 |
| 42. Object-Data Fréchet Regression | v0.28.0 | 0/TBD | Not started | - |
