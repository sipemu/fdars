# Requirements: fdars — v0.25.0 Serial Dependence, Representation & Density Breadth

**Defined:** 2026-08-21
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the next three top-ranked items from the v0.18.0 `R-BACKLOG.md` (score 1.73 each), broadening fdars' functional-time-series diagnostics, representation layer, and density-FDA families.

Milestone-level conventions (carried from v0.19.0–v0.24.0, apply to every requirement below):

- Real `fdars-core/src/` code — **additive/non-breaking**, `Result<T, FdarError>`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.**
- **Reuse-first** — new `fts/acf.rs` reusing `helpers` quadrature + `covariance.rs`; extend `basis/` + new `multi_fdata.rs`; new `density_fda.rs` reusing `fdata_to_pc_1d`; no new algorithm subsystem, **no new crate dependency.**
- R baselines matched by **capability**, not R's exact signatures. Plotting/rendering is **out of scope** (numeric outputs only).
- All three requirements are **independent** — no cross-phase hard dependency — so the phases may be planned/executed in any order or in parallel.

## v1 Requirements

Requirements for milestone v0.25.0. Each maps to exactly one roadmap phase.

### Functional Time Series (differentiators)

- [x] **FTS-02**: Add functional serial-dependence tooling in a new `fts/acf.rs` module — L2-norm functional autocorrelation (fACF) and partial ACF (fPACF) with the strong-white-noise limiting distribution for confidence bands, a functional stationarity test, a long-run covariance kernel-sandwich estimator, and a functional differencing operator. Reuses `helpers` quadrature + `covariance.rs`. Additive/non-breaking; independent of REP-01/DENS-01. R baseline: `ftsa` (facf, T_stationary, long-run covariance) / `fdaACF` (L2-norm fACF, partial fACF, white-noise distribution). Notes: foundational for the deferred FTS-01/FTS-03 forecasting items — build this before FTS-01.

### Representation (differentiators)

- [ ] **REP-01**: Complete the basis-system family — add `monomial_basis`, `exponential_basis`, `power_basis`, and a named `polygonal_basis` (piecewise-linear) factory to `basis/`, each with penalty matrices; a composable `MultiFunData` multivariate/multi-domain functional-data container in a new `multi_fdata.rs`; a composable `Lfd`/linear-differential-operator object; and a `principal_differential_analysis` (PDA, linear-ODE estimation) estimator. Additive/non-breaking; the constant basis is already handled (T-01); independent of FTS-02/DENS-01. R baseline: `fda` (monomial, exponential, power, polygonal bases; Lfd/PDA) / `funData` (multiFunData multi-domain container) / `tf` (tidy multi-representation vector).

### Density FDA (differentiators)

- [ ] **DENS-01**: Add density-valued FDA in a new `density_fda.rs` module — the log-quantile-density (LQD) transform and its inverse (compositional-geometry map), LQD-FPCA for probability densities (reuse `fdata_to_pc_1d` in LQD space, with FVE for LQD-FPCA), a 1D Wasserstein Fréchet mean (quantile-average barycenter) of densities, and density normalization/regularization. Numeric outputs only. Additive/non-breaking; independent of FTS-02/REP-01. R baseline: `fdadensity` (0.1.4). Notes: the 1D-density subset of R-audit Area 7 — simpler than the general Fréchet items (FRE-01/FRE-02).

## v2 Requirements

Deferred to future milestones (from `.planning/research/R-BACKLOG.md`, ordered by score). Not in this roadmap.

### Specialized FPCA / Sparse (score 1.73, P3 differentiators)

- **FPCA-02**: Specialized FPCA variants (FPCAder, FSVD, cross-covariance, sandwich/ssvd) (score 1.73)
- **SPARSE-01**: Sparse/irregular fast covariance (FACE, mfaces) + trajectory bands (score 1.73)

### Larger items (score ≤ 1.33, L-effort)

- **FTS-01**, **FRE-01** (score 1.33, L); **FTS-03**, **FRE-02**, **REG-06**, **REP-02**, **CLUS-02** (score ≤ 1.00, L)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Plotting / rendering of fACF/PACF diagnostics, basis functions, or density curves | Numeric Rust library — numeric statistics are in scope, the plots are not (consistent with the R-audit plotting exclusion) |
| Full functional time series forecasting (ftsm, FPC-regression, fplsr, updating — FTS-01) | L-effort; FTS-02 builds the serial-dependence foundation it needs — forecasting deferred |
| Spectral/dynamic FTS methods (FTS-03) | L-effort separate cluster; deferred |
| General Fréchet regression / object-data statistics (FRE-01/FRE-02) | L-effort general metric-space machinery; DENS-01 covers only the tractable 1D-density subset of Area 7 |
| Multivariate density FPCA / general metric-space barycenters | DENS-01 is scoped to 1D densities (quantile geometry) — no general metric-space machinery |
| `tidyfun`-style tidy vector semantics beyond the `MultiFunData` container | REP-01 delivers the composable multi-domain container; a full tidy-vector API is a larger REP-02 item, deferred |
| New crate dependencies | All three items are reuse-first over existing infrastructure; no new dependency permitted |
| Changes to existing public signatures | Milestone is strictly additive/non-breaking |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FTS-02 | Phase 34 | Complete |
| REP-01 | Phase 35 | Pending |
| DENS-01 | Phase 36 | Pending |

**Coverage:**

- v1 requirements: 3 total
- Mapped to phases: 3
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-21*
*Last updated: 2026-08-21 after initial definition*
