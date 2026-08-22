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
- 🔨 **v0.27.0 — Functional Time Series & Fréchet Regression** — Phases 39–40 (in progress)

## Phases

- [ ] **Phase 39: Functional Time-Series Forecasting** (FTS-01) - FPCA-based `ftsm`, FPC-score-regression forecasting, functional PLS forecasting (`fplsr`), dynamic forecast updating, and iterative multi-step forecasting — new `fts/forecast.rs`, reusing `fdata_to_pc_1d` + `scoring.rs`, building on the shipped FTS-02 (`fts/acf.rs`)
- [ ] **Phase 40: Fréchet / Object-Data Regression** (FRE-01) - metric-space abstraction (distance + weighted-Fréchet-mean solver), Fréchet mean/variance, global & local Fréchet regression, 1D 2-Wasserstein distance, density-response Fréchet regression, and Fréchet ANOVA — new `frechet/` module, sharing DENS-01's (`density_fda.rs`) quantile machinery

## Phase Details

### Phase 39: Functional Time-Series Forecasting

**Goal**: A user can forecast future functional curves from a time-ordered curve series — the FPCA-based functional time-series model (`ftsm`), FPC-score-regression forecasting, a functional PLS forecasting variant (`fplsr`), dynamic forecast updating as new observations arrive, and iterative multi-step (h > 1) forecasting — all in a new `fdars-core/src/fts/forecast.rs`, reusing the existing dense FPCA (`fdata_to_pc_1d`) + forecast-error metrics (`scoring.rs`) and building on the shipped FTS-02 serial-dependence foundation (`fts/acf.rs`), without any existing code changing. Fills the largest single-area gap zone (Area 6, 2/25 present). R baseline: `ftsa`.
**Depends on**: Nothing hard (independent of Phase 40; disjoint modules — may run in any order or in parallel). Builds on the shipped FTS-02 (`fts/acf.rs`, Phase 34) for score-model order/inference and reuses shipped `fdata_to_pc_1d` + `scoring.rs`.
**Requirements**: FTS-01 (FTS-01-01, FTS-01-02, FTS-01-03, FTS-01-04, FTS-01-05)
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points (crate-root re-exported) in `fdars-core/src/fts/forecast.rs`, each consuming a time-ordered curve series in column-major `FdMatrix` form and returning structured numeric output: fit an FPCA-based functional time-series model (`ftsm` — mean + FPC loadings + the retained score-time-series, plus reconstructed fitted curves), forecast future curves, a functional PLS forecasting variant (`fplsr`), a dynamic-update entry point, and an iterative multi-step forecast entry point.
  2. Fitting the `ftsm` model over a curve series decomposes it via `fdata_to_pc_1d`, retains the mean, FPC loadings, and the score-time-series, and reconstructs fitted curves that recover the input series (up to the retained-component truncation) within a documented tolerance on synthetic data (inline `#[cfg(test)]` tests).
  3. User can forecast h-step-ahead curve(s) by fitting scalar (AR/ARIMA-style) time-series models to each FPC-score sequence and reconstructing the forecast curve(s) from the forecast scores, such that on a synthetic series whose scores follow a known AR process the forecast scores recover the AR one-step prediction within a documented tolerance, and forecast-error metrics from `scoring.rs` are lower than a naive last-curve baseline (inline `#[cfg(test)]` tests).
  4. The functional PLS forecasting variant (`fplsr`) produces PLS-score-based forecasts as an alternative to FPC-score regression; the dynamic-update path updates an existing forecast when new curve observation(s) arrive without refitting from scratch and agrees (within a documented tolerance) with a full refit that includes the same new observations; and the iterative multi-step path returns per-horizon forecast curves for h > 1 whose h = 1 curve matches the single-step forecast (inline `#[cfg(test)]` tests).
  5. All entry points reuse `fdata_to_pc_1d` + `scoring.rs` + the FTS-02 `fts/acf.rs` foundation rather than adding a new algorithm subsystem, add **no new crate dependency**, use per-thread seeded RNG for any stochastic path, and invalid inputs (empty/too-short series, fewer observations than requested components, non-monotone or mismatched argvals, `h < 1`, `ncomp` out of range, degenerate columns) return `FdarError` rather than panicking. Existing public signatures across `fdars-core` (including `fdata_to_pc_1d`, `scoring.rs`, and `fts/acf.rs`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 3 plans
- [ ] 39-01-PLAN.md — Tracer: module scaffold + wiring + Yule-Walker AR helpers + `ftsm` fit + `ftsm_forecast` one-step (FTS-01-01, FTS-01-02)
- [ ] 39-02-PLAN.md — `ftsm_forecast_multistep` (iterative h>1) + `ftsm_update` (dynamic update, no FPCA refit) (FTS-01-05, FTS-01-04)
- [ ] 39-03-PLAN.md — `fplsr` functional PLS lag-1 forecasting variant + `FplsrResult` (FTS-01-03)

### Phase 40: Fréchet / Object-Data Regression

**Goal**: A user can perform metric-space (object-data) regression and statistics that the R `frechet` package exposes but fdars was missing — a metric-space abstraction (distance + weighted-Fréchet-mean solver) with a 1D-Wasserstein (density-response) backend, the Fréchet mean and variance of a sample, global and local (kernel-weighted) Fréchet regression over Euclidean predictors, the 1D 2-Wasserstein distance between two distributions, density-response Fréchet regression, and a Fréchet ANOVA group-difference test — all in a new `fdars-core/src/frechet/` module, sharing DENS-01's (`density_fda.rs`) quantile/Wasserstein machinery, without any existing code changing. Opens the single largest all-absent zone (Area 7, 0/25 present). R baseline: `frechet`.
**Depends on**: Nothing hard (independent of Phase 39; disjoint modules — may run in any order or in parallel). Shares DENS-01's (`density_fda.rs`, Phase 36) 1D Wasserstein / quantile machinery for the density response space.
**Requirements**: FRE-01 (FRE-01-01, FRE-01-02, FRE-01-03, FRE-01-04, FRE-01-05, FRE-01-06, FRE-01-07, FRE-01-08)
**Success Criteria** (what must be TRUE):

  1. User can define a metric-space abstraction (a distance function + a weighted-Fréchet-mean solver) that the regression/statistics routines consume, with a 1D-Wasserstein (density-response) backend provided as the first concrete space; all new entry points are `Result`-returning and crate-root re-exported in the new `fdars-core/src/frechet/` module, consuming numeric inputs and returning structured numeric output.
  2. User can compute the Fréchet mean of a sample of metric-space objects (weighted-barycenter solver) and the Fréchet variance (mean squared distance to the Fréchet mean), such that in the 1D-Wasserstein density space the Fréchet mean recovers the quantile-average barycenter (agreeing with DENS-01's Wasserstein mean within a documented tolerance) and the variance is zero (within tolerance) for an identical-object sample and grows with dispersion (inline `#[cfg(test)]` tests).
  3. User can compute the 1D 2-Wasserstein distance between two distributions (quantile-based, reusing DENS-01's quantile machinery) — returning 0 within tolerance for identical distributions and matching a hand-computed reference (e.g. the shift/scale between two known distributions) on synthetic data (inline `#[cfg(test)]` tests).
  4. User can run global Fréchet regression with Euclidean predictors (predicting the conditional Fréchet-mean response object at new predictor values via the global/linear weight scheme) and local (local-linear / kernel-weighted) Fréchet regression, such that on synthetic data generated from a known predictor→object relationship the predicted response objects track the truth within a documented tolerance, and the density-response variant predicts a conditional density response from Euclidean predictors in 2-Wasserstein space (inline `#[cfg(test)]` tests).
  5. User can run a Fréchet ANOVA group-difference test on metric-space responses based on Fréchet means/variances, returning a numeric test statistic (and p-value) that flags a genuine between-group difference and does not flag a homogeneous sample on synthetic data. All entry points reuse `density_fda.rs`'s quantile/Wasserstein machinery rather than adding a new subsystem, add **no new crate dependency**, use per-thread seeded RNG for any stochastic (e.g. permutation) path, and invalid inputs (empty sample, mismatched predictor/response counts, non-monotone or mismatched grids, invalid bandwidth, fewer than two groups for ANOVA, degenerate objects) return `FdarError` rather than panicking. Existing public signatures across `fdars-core` (including `density_fda.rs`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 39. Functional Time-Series Forecasting | 0/3 | Not started | - |
| 40. Fréchet / Object-Data Regression | 0/TBD | Not started | - |

**Execution order:** Both phases are **independent** — FTS-01 (Phase 39) and FRE-01 (Phase 40) have **no cross-phase hard dependency** (as with prior implementation milestones), and each touches a disjoint area of the codebase (new `fts/forecast.rs` vs new `frechet/` module). They may be planned and executed in **any order or in parallel**. The two score-1.33 (L-effort) `R-BACKLOG.md` items (FTS-01 rank 20, FRE-01 rank 21), exhausting the 1.33 tier. Both are P2 differentiators opening the two largest gap zones (Area 6 functional time series, 2/25 present; Area 7 density/object data, 0/25 present). FTS-01 builds on the shipped FTS-02 (`fts/acf.rs`) foundation; FRE-01 shares DENS-01's (`density_fda.rs`) Wasserstein/quantile machinery. **Milestone constraints (apply to both phases):** additive/non-breaking (zero changes to existing public signatures), reuse-first (no new algorithm subsystem beyond the two new modules), all public functions `Result<T, FdarError>`-returning, inline `#[cfg(test)]` tests with hand-computed/reference checks + error paths, crate-root re-exports, **no new crate dependency**, numeric outputs only (plotting/rendering out of scope), `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean. R baselines matched by capability, not R's exact signatures. The next tier (score 1.00, L-effort) is FTS-03 (spectral FTS, depends on FTS-01) + FRE-02 (object-data Fréchet spaces, depends on FRE-01).

<details>
<summary>✅ v0.26.0 FPCA Breadth & Sparse Covariance (Phases 37–38) — SHIPPED 2026-08-21</summary>

The two remaining top-ranked `R-BACKLOG.md` items (score 1.73, M-effort), exhausting the 1.73 tier: FPCA-02 (Phase 37), SPARSE-01 (Phase 38). Both additive/non-breaking, reuse-first, no new crate dependency; whole-crate 2414 lib + doc tests green, `cargo clippy --all-targets` clean. Both phases independent (disjoint modules: `fpca_variants.rs` vs `irreg_fdata/face.rs`), each verified 5/5, milestone audit PASSED 8/8.

- [x] Phase 37: Specialized FPCA Variants (2/2 plans) — completed 2026-08-21 (FPCA-02) — `fpca_der`, `fsvd`, `cross_covariance`, `dynamical_correlation`, `ssvd`
- [x] Phase 38: Sparse Fast Covariance & Trajectory Bands (2/2 plans) — completed 2026-08-21 (SPARSE-01) — `face_covariance`, `mface_covariance`, `face_trajectory`

Full phase detail: [milestones/v0.26.0-ROADMAP.md](milestones/v0.26.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.25.0 Serial Dependence, Representation & Density Breadth (Phases 34–36) — SHIPPED 2026-08-21</summary>

The next three top-ranked `R-BACKLOG.md` items (score 1.73 each, M-effort): FTS-02 (Phase 34), REP-01 (Phase 35), DENS-01 (Phase 36). All additive/non-breaking, reuse-first, no new crate dependency; whole-crate 2385 lib + 166 doc tests green, `cargo clippy --all-targets` clean. All three phases independent (disjoint modules).

- [x] Phase 34: Functional Serial-Dependence Tooling (3/3 plans) — completed 2026-08-21 (FTS-02)
- [x] Phase 35: Basis-System Completions (4/4 plans) — completed 2026-08-21 (REP-01)
- [x] Phase 36: Density Object-Data FDA (3/3 plans) — completed 2026-08-21 (DENS-01)

Full phase detail: [milestones/v0.25.0-ROADMAP.md](milestones/v0.25.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.24.0 Functional Regression & Clustering Breadth (Phases 31–33) — SHIPPED 2026-08-20</summary>

The three top-ranked P2 differentiators from the R-ecosystem backlog (score 1.73 each), all additive/non-breaking to `fdars-core` with zero changes to existing public signatures and no new crate dependency. Milestone audit passed; whole-crate 2268-test suite + `cargo clippy --all-targets` green. All three phases independent (disjoint modules).

- [x] Phase 31: Additive Functional Regression & Variable Selection (2/2 plans) — completed 2026-08-20 (REG-04)
- [x] Phase 32: Flexible Mixed-Effects Regression (2/2 plans) — completed 2026-08-20 (REG-05)
- [x] Phase 33: Model-Based & Density Functional Clustering (3/3 plans) — completed 2026-08-20 (CLUS-01)

Full phase detail: [milestones/v0.24.0-ROADMAP.md](milestones/v0.24.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.23.0 Depth, Outliers & Interval Inference (Phases 28–30) — SHIPPED 2026-08-20</summary>

The top three P2 differentiator gaps from the R-ecosystem backlog (score 2.31 each), all additive/non-breaking. Milestone audit passed (3/3). DEPTH→OUT chain; INF-03 independent.

- [x] Phase 28: Depth-Measure Long Tail (3/3 plans) — completed 2026-08-20 (DEPTH-01)
- [x] Phase 29: Outlier-Detector Suite (2/2 plans) — completed 2026-08-20 (OUT-01)
- [x] Phase 30: Interval Testing Procedure Family (2/2 plans) — completed 2026-08-20 (INF-03)

Full phase detail: [milestones/v0.23.0-ROADMAP.md](milestones/v0.23.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.14.0–v0.22.0 (Phases 1–27) — SHIPPED</summary>

- ✅ **v0.14.0 Performance & scikit-fda Gap Audit** (Phases 1–9) — audit-only; `AUDIT-REPORT.md` + `BACKLOG.md`. [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 Top-Backlog Quick Wins** (Phases 10–11, PR #38) — FEAT-01/02, PERF-01/02. [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 Elastic Feasibility + Parity Quick Wins** (Phases 12–13, PR #40) — PERF-03, FEAT-03/04/05. [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 Registration Parity & Elastic-FPCA Performance** (Phases 14–15, PR #41) — FEAT-06/07, PERF-04. [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 R-Ecosystem Gap Audit** (Phases 16–19) — audit-only; `R-AUDIT-REPORT.md` + `R-BACKLOG.md`. [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 Functional Inference Suite** (Phases 20–21) — INF-01, INF-02; new `inference/` module. [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 Table-Stakes Quick Wins** (Phases 22–23) — T-01, T-02. [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 Functional Regression Completeness** (Phases 24–25) — REG-01, REG-02. [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 PACE Sparse FPCA & Elastic Multinomial** (Phases 26–27) — FPCA-01, REG-03. [archive](milestones/v0.22.0-ROADMAP.md)

</details>
