# Requirements: fdars — Milestone v0.27.0 Functional Time Series & Fréchet Regression

**Defined:** 2026-08-22
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the two score-1.33 (L-effort) items from the v0.18.0 `R-BACKLOG.md`: functional time-series forecasting (FTS-01) and Fréchet / object-data regression (FRE-01).

**Source:** [`.planning/research/R-BACKLOG.md`](research/R-BACKLOG.md) items FTS-01 (rank 20) and FRE-01 (rank 21).

**Milestone constraints (apply to every requirement):** additive/non-breaking (zero changes to existing public signatures); reuse-first (no new algorithm subsystem beyond the two new modules); all public functions `Result<T, FdarError>`-returning; inline `#[cfg(test)]` tests with hand-computed/reference checks + error paths; crate-root re-exports; **no new crate dependency**; numeric outputs only (plotting/rendering out of scope); `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.

## v1 Requirements

Requirements for milestone v0.27.0. Each maps to a roadmap phase.

### FTS-01 — Functional time-series forecasting

R baseline: `ftsa` (ftsm, FPC-regression forecasting, fplsr, dynamic updating, iterative forecasting). New `fdars-core/src/fts/forecast.rs`, reusing `fdata_to_pc_1d` (FPCA decomposition) + `scoring.rs` (forecast-error metrics), building on the shipped FTS-02 ACF/long-run-covariance foundation (`fts/acf.rs`) for score-model order/inference.

- [x] **FTS-01-01**: User can fit an FPCA-based functional time-series model (`ftsm`) over a time-ordered curve series — decompose via `fdata_to_pc_1d`, retain mean + FPC loadings + the score-time-series, and reconstruct fitted curves.
- [x] **FTS-01-02**: User can forecast future curves by fitting scalar time-series (AR/ARIMA-style) models to each FPC-score sequence and reconstructing the h-step-ahead forecast curve(s) from the forecast scores.
- [x] **FTS-01-03**: User can produce a functional PLS forecasting variant (`fplsr`) — PLS-score-based forecasting as an alternative to FPC-score regression.
- [x] **FTS-01-04**: User can dynamically update an existing forecast as new curve observation(s) arrive, without refitting from scratch (dynamic-updating path).
- [x] **FTS-01-05**: User can obtain iterative multi-step (h > 1) forecasts with per-horizon forecast curves.

### FRE-01 — Fréchet / object-data regression + statistics

R baseline: `frechet` (global/local Fréchet regression, Fréchet mean/variance, Wasserstein, density-response regression, Fréchet ANOVA). New `fdars-core/src/frechet/` module — a metric-space abstraction (distance + weighted-Fréchet-mean solver) — starting from the density/2-Wasserstein response space that shares DENS-01's (`density_fda.rs`) quantile machinery.

- [x] **FRE-01-01**: User can define a metric-space abstraction (a distance + weighted-Fréchet-mean solver) that the regression/statistics routines consume, with a 1D-Wasserstein (density-response) backend as the first concrete space.
- [x] **FRE-01-02**: User can compute the Fréchet mean of a sample of metric-space objects (weighted-barycenter solver).
- [x] **FRE-01-03**: User can compute the Fréchet variance of a sample (mean squared distance to the Fréchet mean).
- [x] **FRE-01-04**: User can run global Fréchet regression with Euclidean predictors — predict the conditional Fréchet mean of the response object at new predictor values via the weighted global (linear) weight scheme.
- [x] **FRE-01-05**: User can run local (local-linear / kernel-weighted) Fréchet regression over Euclidean predictors.
- [x] **FRE-01-06**: User can compute the 1D 2-Wasserstein distance between two distributions (quantile-based), reusing DENS-01's quantile machinery.
- [x] **FRE-01-07**: User can run density-response Fréchet regression — predict a conditional density response from Euclidean predictors in 2-Wasserstein space.
- [x] **FRE-01-08**: User can run a Fréchet ANOVA — a group-difference test on metric-space responses based on Fréchet means/variances.

## Future Requirements

Deferred to future milestones (tracked in `R-BACKLOG.md`, not in this roadmap).

### Functional Time Series (1.00 L-effort tier)

- **FTS-03**: Spectral functional time series (DPCA, spectral density operator, functional VAR/VMA, FARMA simulation) — depends on FTS-01; reuses `rustfft`.

### Object Data (1.00 L-effort tier)

- **FRE-02**: Object-data Fréchet regression for specific spaces (covariance/correlation matrices, spherical data with geodesics, network, point-process) — depends on FRE-01's solver framework.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New crate dependency for time-series / metric-space machinery | Milestone constraint — reuse existing FPCA (`fdata_to_pc_1d`), `scoring.rs`, `density_fda.rs` quantile machinery, and self-contained scalar-TS modeling; adding a dep triggers a package-legitimacy review |
| Plotting/visualization of forecasts, prediction bands, or Fréchet fits | Numeric Rust library — renderer stays out of scope (consistent with the v0.14.0/v0.18.0 audit fence); only numeric outputs are delivered |
| Changes to existing public signatures (`fdata_to_pc_1d`, `fts/acf.rs`, `density_fda.rs`, …) | Additive/non-breaking constraint — new functions/modules only; existing paths preserved bit-for-bit |
| Spectral / frequency-domain FTS (DPCA, spectral density, VAR/VMA, FARMA) — FTS-03 | Separate 1.00-tier L-effort backlog item; this milestone is the forecasting core only |
| Object-space Fréchet backends beyond 1D density/Wasserstein (SPD matrices, spheres, networks, point processes) — FRE-02 | Separate 1.00-tier L-effort backlog item; this milestone delivers the core solver + density/Wasserstein backend only |
| Bayesian / boosting functional regression (FDboost, GAMLSS, Gibbs/VB) — REG-06 | Lower-ranked backlog item; not in this milestone |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FTS-01-01 | Phase 39 | Complete |
| FTS-01-02 | Phase 39 | Complete |
| FTS-01-03 | Phase 39 | Complete |
| FTS-01-04 | Phase 39 | Complete |
| FTS-01-05 | Phase 39 | Complete |
| FRE-01-01 | Phase 40 | Complete |
| FRE-01-02 | Phase 40 | Complete |
| FRE-01-03 | Phase 40 | Complete |
| FRE-01-04 | Phase 40 | Complete |
| FRE-01-05 | Phase 40 | Complete |
| FRE-01-06 | Phase 40 | Complete |
| FRE-01-07 | Phase 40 | Complete |
| FRE-01-08 | Phase 40 | Complete |

**Coverage:**
- v1 requirements: 13 total (FTS-01: 5, FRE-01: 8)
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-22*
*Last updated: 2026-08-22 after initial definition*
