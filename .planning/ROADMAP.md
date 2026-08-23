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
- 🔵 **v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering** — Phases 43–45 (active) — the FINAL three `R-BACKLOG.md` items (all score 0.67, L-effort); **exhausts the backlog**

## Overview

v0.29.0 draws the **final three** items from the v0.18.0 `R-BACKLOG.md` — REG-06 (rank 24), REP-02 (rank 25), CLUS-02 (rank 26), all score 0.67, L-effort — **exhausting the R-parity backlog** (#1–23 shipped through v0.28.0). The milestone is implementation, not audit — real `fdars-core/src/` code, additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports, **zero changes to existing public signatures**. Three phases, one requirement-category each, touching disjoint code areas (`boosting_regression.rs` vs `fem_smoothing.rs`+`smooth_basis.rs` vs `coclustering.rs`) — plannable/executable in any order or in parallel.

**Unlike prior reuse-first milestones, all three are large standalone estimation subsystems — the heaviest milestone in the sequence.** Each phase likely needs a careful multi-plan decomposition. The **no-new-crate-dependency** convention carries forward, with one explicit caveat: **REP-02 (Phase 44)** is the phase where the planner MAY revisit the no-dependency constraint if an in-house triangulated-mesh/FEM implementation proves impractical at plan time. After v0.29.0 ships, `R-BACKLOG.md` is exhausted — the next milestone requires a fresh yardstick (a new gap-audit, a performance/consolidation pass, or a crate-release-hardening milestone), decided via `/gsd-new-milestone`.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (43.1, 43.2): Urgent insertions (marked with INSERTED)

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

<details open>
<summary>🔵 v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering (Phases 43–45) — ACTIVE</summary>

- [ ] **Phase 43: Boosting / Bayesian Functional Regression** (REG-06) — new `boosting_regression.rs`: component-wise gradient boosting with functional base-learners (boosted FOSR + boosted FoFR), GAMLSS distributional regression, Bayesian FOSR (Gibbs/VB), FDboost stability selection. R baseline `FDboost`/`refund`.
- [ ] **Phase 44: FEM/PDE Smoothing on Irregular 2D Domains** (REP-02) — new `fem_smoothing.rs`: linear finite-element basis over a triangulated 2D mesh + PDE (Laplacian) regularization; positive (log-domain) & Ramsay integral-of-exp monotone smoothers added additively to `smooth_basis.rs`. R baseline `fdaPDE`. Planner MAY revisit the no-dependency constraint here.
- [ ] **Phase 45: Functional Co-Clustering (funLBM latent-block)** (CLUS-02) — new `coclustering.rs`: block-wise-Gaussian EM on FPC scores with simultaneous row (curve) + column (argument) clustering, plus slope-heuristic model selection. R baseline `funLBM`/`funHDDC`.

**Execution order:** All three phases are **independent** — no cross-phase hard dependency, disjoint code areas. Plannable/executable in **any order or in parallel**. No forced sequence.

### Phase 43: Boosting / Bayesian Functional Regression
**Goal**: A user can fit gradient-boosting and Bayesian functional regression models that fdars previously lacked — boosted function-on-scalar / function-on-function base-learners, GAMLSS distributional regression, a Bayesian FOSR sampler with credible bands, and boosting stability selection.
**Depends on**: Nothing (independent — disjoint code area `boosting_regression.rs`)
**Requirements**: REG-06-01, REG-06-02, REG-06-03, REG-06-04, REG-06-05
**Success Criteria** (what must be TRUE):
  1. User can fit component-wise gradient-boosting functional regression with functional base-learners for a **function-on-scalar** response (boosted FOSR), one base-learner selected per boosting iteration — via a `Result`-returning public fn with inline `#[cfg(test)]` recovery + error-path tests.
  2. User can fit component-wise gradient-boosting functional regression for a **function-on-function** predictor/response (boosted FoFR base-learners) through the same boosting framework.
  3. User can fit a GAMLSS-style distributional functional regression that models more than one distributional parameter (e.g. location + scale) of the response.
  4. User can fit a Bayesian function-on-scalar regression via a Gibbs/VB sampler and obtain coefficient posterior summaries (posterior mean + credible bands).
  5. User can run FDboost-style stability selection over the boosting base-learners and obtain per-learner selection frequencies / a stable predictor set.
**Plans**: 5 plans
  - [ ] 43-01-boosting-core-fosr-PLAN.md — boosting core + boosted FOSR (REG-06-01): shared component-wise boosting loop, penalized B-spline base-learner fit, `BoostingConfig`/all config+result types, `boost_fosr`, `boost_fosr_one_step`, module barrel + skeletons, lib.rs/prelude registration (wave 1)
  - [ ] 43-02-boosted-fofr-PLAN.md — boosted FoFR (REG-06-02): FPC-score signal-compression base-learners through the boosting core, reconstructed β(s,t) surfaces (wave 2)
  - [ ] 43-03-gamlss-location-scale-PLAN.md — GAMLSS location+scale (REG-06-03): cyclic gamboostLSS boosting over μ (identity) and σ (log link), Gaussian negative gradients (wave 2)
  - [ ] 43-04-bayesian-fosr-gibbs-PLAN.md — Bayesian FOSR (REG-06-04): conjugate Normal/Inverse-Gamma Gibbs on FPC-score coefficients, posterior mean + pointwise credible bands, seeded determinism (wave 2)
  - [ ] 43-05-stability-selection-PLAN.md — stability selection (REG-06-05): B seeded ⌊n/2⌋ subsamples over the boosting path, per-learner selection frequencies, stable set at π=0.9, PFER bound (wave 2)

### Phase 44: FEM/PDE Smoothing on Irregular 2D Domains
**Goal**: A user can smooth scattered observations over an irregular 2D domain using a finite-element basis with PDE (Laplacian) regularization — plus obtain shape-constrained (positive, monotone) smoothers — capabilities absent from fdars' regular-grid 2D FOSR strength.
**Depends on**: Nothing (independent — disjoint code area `fem_smoothing.rs` + additive `smooth_basis.rs` smoothers)
**Requirements**: REP-02-01, REP-02-02, REP-02-03, REP-02-04
**Success Criteria** (what must be TRUE):
  1. User can construct a linear finite-element basis over a user-supplied triangulated 2D mesh (nodes + triangle connectivity), evaluate basis functions, and assemble the mass and stiffness matrices — via a `Result`-returning public fn with inline tests against hand-computed reference values + degenerate/invalid-mesh error paths.
  2. User can perform PDE-regularized (Laplacian-penalty) surface smoothing of scattered observations over an irregular 2D domain through the FE basis, receiving a fitted surface plus smoothing diagnostics.
  3. User can perform positive-valued smoothing that guarantees a nonnegative fitted function (log-domain fit).
  4. User can perform monotone smoothing via the Ramsay integral-of-exponential representation (guaranteed-monotone fitted function), added additively to `smooth_basis.rs`.
**Plans**: TBD (large standalone mesh/FEM subsystem — expect multi-plan decomposition; planner MAY revisit the no-new-crate-dependency constraint if an in-house triangulated-mesh/FEM implementation proves impractical, and must flag it at plan time)
**UI hint**: no

### Phase 45: Functional Co-Clustering (funLBM latent-block)
**Goal**: A user can co-cluster functional data — simultaneously grouping curves into row-clusters and argument points into column-clusters via a functional latent block model — and select the number of blocks automatically, a paradigm absent from fdars' curve-only clustering.
**Depends on**: Nothing (independent — disjoint code area `coclustering.rs`)
**Requirements**: CLUS-02-01, CLUS-02-02, CLUS-02-03
**Success Criteria** (what must be TRUE):
  1. User can fit a functional latent block model (funLBM) that **simultaneously** assigns curves to row-clusters and argument points to column-clusters via a block-wise-Gaussian EM on FPC scores, given a target (row, column) block count — via a `Result`-returning public fn with inline recovery + error-path tests.
  2. User can retrieve the co-clustering result — row labels, column labels, per-block parameters, and a converged log-likelihood / model criterion (e.g. ICL).
  3. User can select the number of blocks via the slope-heuristic criterion over a range of candidate (row, column) block counts.
**Plans**: TBD (substantial standalone latent-block EM estimator — expect multi-plan decomposition)

Full detail: this section (active milestone — archived to `milestones/v0.29.0-ROADMAP.md` on ship).

</details>

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 43. Boosting / Bayesian Functional Regression | 0/5 | Planned | - |
| 44. FEM/PDE Smoothing on Irregular 2D Domains | 0/TBD | Not started | - |
| 45. Functional Co-Clustering (funLBM latent-block) | 0/TBD | Not started | - |

All phases through v0.28.0 are shipped and archived under `milestones/`.
