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
- 🔨 **v0.25.0 — Serial Dependence, Representation & Density Breadth** — Phases 34–36 (in progress)

## Phases

- [ ] **Phase 34: Functional Serial-Dependence Tooling** - Add L2-norm functional ACF/PACF with strong-white-noise confidence bands, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator in a new `fts/acf.rs`, reusing `helpers` quadrature + `covariance.rs` (FTS-02, independent)
- [ ] **Phase 35: Basis-System Completions** - Add `monomial_basis`/`exponential_basis`/`power_basis`/`polygonal_basis` factories (with penalty matrices) to `basis/`, a composable `MultiFunData` multi-domain container in new `multi_fdata.rs`, a composable `Lfd`/linear-differential-operator object, and a `principal_differential_analysis` estimator (REP-01, independent)
- [ ] **Phase 36: Density Object-Data FDA** - Add the log-quantile-density (LQD) transform + inverse, LQD-FPCA for probability densities (reuse `fdata_to_pc_1d` in LQD space), a 1D Wasserstein Fréchet-mean barycenter, and density normalization/regularization in a new `density_fda.rs` (DENS-01, independent)

## Phase Details

### Phase 34: Functional Serial-Dependence Tooling

**Goal**: A user can diagnose serial dependence in a time-ordered series of functional observations with the tooling `ftsa`/`fdaACF` expose but fdars was missing — L2-norm functional autocorrelation (fACF) and partial ACF (fPACF) with the strong-white-noise limiting distribution for confidence bands, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator — all in a new `fdars-core/src/fts/acf.rs`, reusing `helpers` quadrature and `covariance.rs`, without any existing code changing. Foundational for the deferred FTS-01/FTS-03 forecasting items (build this before FTS-01).
**Depends on**: Nothing (independent of Phases 35/36; may run in any order or in parallel)
**Requirements**: FTS-02
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points in `fdars-core/src/fts/acf.rs` (crate-root re-exported), each consuming a time-ordered series of curves in a column-major `FdMatrix` and returning structured numeric output: an L2-norm functional ACF (fACF) and partial ACF (fPACF) across a requested lag range, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator.
  2. The fACF/fPACF entry points return the L2-norm autocorrelation at each lag together with strong-white-noise confidence bands derived from the limiting distribution; on i.i.d. (white-noise) synthetic curves the ACF at nonzero lags falls inside the bands, and on a synthetic curve series with an injected lag-1 dependence structure the lag-1 fACF exceeds the band (inline `#[cfg(test)]` tests, seeded for reproducibility).
  3. The functional differencing operator produces a first-difference curve series whose length is one less than the input and round-trips against a cumulative-sum reconstruction within a documented tolerance, and the stationarity test rejects on an injected non-stationary (trended) series while not rejecting on a stationary one (inline `#[cfg(test)]` tests).
  4. The long-run-covariance kernel-sandwich estimator returns a symmetric operator/matrix that reduces to the lag-0 sample covariance when the bandwidth selects only lag 0, and the tooling reuses `helpers` quadrature + `covariance.rs` rather than adding a new subsystem — adding no new crate dependency; invalid inputs (empty matrix / fewer curves than the requested max lag / mismatched argvals vs values / degenerate columns / invalid bandwidth) return `FdarError` rather than panicking.
  5. Existing public signatures across `fdars-core` (including `covariance.rs` and `helpers`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 3 plans

Plans:
- [ ] 34-01-PLAN.md — Tracer: fts module skeleton + shared autocovariance helper + L2-norm fACF/fPACF with MC strong-white-noise band (functional_acf, functional_pacf, FacfResult), crate-root re-exported
- [ ] 34-02-PLAN.md — Expansion: functional_difference (cumulative-sum round-trip) + Monte-Carlo stationarity_test (StationarityResult)
- [ ] 34-03-PLAN.md — Expansion: Bartlett long_run_covariance (LongRunCovResult) + phase-wide error/determinism sweep + full-suite/clippy gate

### Phase 35: Basis-System Completions

**Goal**: A user can build functional-data representations with the basis systems and containers `fda`/`funData`/`tf` expose but fdars was missing — `monomial_basis`, `exponential_basis`, `power_basis`, and a named `polygonal_basis` (piecewise-linear) factory (each with a penalty matrix), a composable `MultiFunData` multivariate/multi-domain container, a composable `Lfd`/linear-differential-operator object, and a `principal_differential_analysis` (PDA, linear-ODE estimation) estimator — by extending `fdars-core/src/basis/` and adding a new `multi_fdata.rs`, without any existing basis code changing (the constant basis is already handled by T-01).
**Depends on**: Nothing (independent of Phases 34/36; may run in any order or in parallel)
**Requirements**: REP-01
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning `monomial_basis`, `exponential_basis`, `power_basis`, and `polygonal_basis` factories in `fdars-core/src/basis/` (crate-root re-exported), each producing a basis-evaluation matrix over supplied argvals plus a penalty matrix, and each evaluating to the expected closed-form basis functions on hand-computed reference points within a documented tolerance (inline `#[cfg(test)]` tests).
  2. A composable `MultiFunData` multivariate/multi-domain functional-data container lives in a new `fdars-core/src/multi_fdata.rs` (crate-root re-exported), holding several component `FdMatrix` blocks on possibly-different domains, with constructors and accessors that preserve each component's argvals and enforce a consistent number of observations across components.
  3. A composable `Lfd`/linear-differential-operator object can be constructed from coefficient functions and applied to functional data, and a `principal_differential_analysis` estimator recovers the coefficients of a known linear ODE from synthetic solution curves within a documented tolerance (inline `#[cfg(test)]` tests).
  4. The new bases and PDA reuse the existing `basis/` penalty/evaluation conventions and `helpers` quadrature rather than adding a new subsystem, add no new crate dependency, and invalid inputs (empty/mismatched argvals, non-monotone knots for the polygonal basis, invalid degree/rate parameters, mismatched `MultiFunData` observation counts, singular PDA design) return `FdarError` rather than panicking.
  5. Existing `basis/` public signatures (including the B-spline, Fourier, and constant bases) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: TBD

### Phase 36: Density Object-Data FDA

**Goal**: A user can do density-valued functional data analysis with the tooling `fdadensity` exposes but fdars was missing — the log-quantile-density (LQD) transform and its inverse (a compositional-geometry map between densities and LQD functions), LQD-FPCA for probability densities (reusing `fdata_to_pc_1d` in LQD space, with fraction-of-variance-explained), a 1D Wasserstein Fréchet mean (quantile-average barycenter) of densities, and density normalization/regularization — all numeric, in a new `fdars-core/src/density_fda.rs`, without any existing code changing. This is the tractable 1D-density subset of R-audit Area 7 (simpler than the general Fréchet items FRE-01/FRE-02).
**Depends on**: Nothing (independent of Phases 34/35; may run in any order or in parallel)
**Requirements**: DENS-01
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points in `fdars-core/src/density_fda.rs` (crate-root re-exported): a log-quantile-density (LQD) transform and its inverse, an LQD-FPCA estimator for a sample of probability densities, a 1D Wasserstein Fréchet-mean (quantile-average) barycenter, and a density normalization/regularization helper — each consuming/returning densities on a supplied grid as numeric output (no plotting/rendering).
  2. The LQD transform followed by its inverse round-trips a valid probability density back to itself within a documented tolerance, and the inverse always returns a normalized non-negative density (integrates to 1) for admissible LQD input (inline `#[cfg(test)]` tests).
  3. LQD-FPCA reuses `fdata_to_pc_1d` in LQD space and returns FPC components with a fraction-of-variance-explained (FVE) that is monotone non-decreasing in the number of components and reaches 1 at full rank; on a synthetic family of densities varying along a single mode of variation, the leading component captures (near-)all the variance (inline `#[cfg(test)]` tests).
  4. The 1D Wasserstein Fréchet mean equals the quantile-average barycenter — reducing to the input density on a single-density sample and lying quantile-between its inputs on a two-density sample — and the density-normalization helper turns a non-negative unnormalized curve into one that integrates to 1; the module reuses `fdata_to_pc_1d` + `helpers` quadrature rather than adding a new subsystem, adding no new crate dependency; invalid inputs (negative/all-zero density, non-monotone grid, mismatched argvals vs values, empty sample) return `FdarError` rather than panicking.
  5. Existing public signatures across `fdars-core` (including `fdata_to_pc_1d`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 34. Functional Serial-Dependence Tooling | 0/3 | Planned | - |
| 35. Basis-System Completions | 0/0 | Not started | - |
| 36. Density Object-Data FDA | 0/0 | Not started | - |

**Execution order:** All three phases are **independent** — FTS-02 (Phase 34), REP-01 (Phase 35), and DENS-01 (Phase 36) have **no cross-phase hard dependency** (as in v0.24.0's REG-04/REG-05/CLUS-01), so they may be planned and executed in **any order or in parallel**. Each extends a disjoint area of the codebase (new `fts/acf.rs` / extend `basis/` + new `multi_fdata.rs` / new `density_fda.rs`). The next three top-ranked `R-BACKLOG.md` items (FTS-02 rank 14, REP-01 rank 16, DENS-01 rank 17 — all score 1.73, M-effort). Additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports, **zero changes to existing public signatures**; reuse-first, **no new crate dependency**; numeric outputs only (plotting/rendering out of scope). R baselines matched by capability, not R's exact signatures.

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
