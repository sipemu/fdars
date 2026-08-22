# Changelog

All notable changes to `fdars-core` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/). All entries below are **additive and
non-breaking** — no existing public signature changed and no new crate dependency
was added in this span.

## [0.27.0] - 2026-08-22

Published release covering the v0.25.0, v0.26.0, and v0.27.0 development
milestones (the 0.25.0 and 0.26.0 milestones shipped code but were never
published as separate crate versions; their additions are included here).

### Added — v0.27.0 (Functional Time Series & Fréchet Regression)

- **Functional time-series forecasting** (`fdars-core::fts`, new `fts/forecast.rs`):
  - `ftsm` — FPCA-based functional time-series model (decompose a time-ordered
    curve series via `fdata_to_pc_1d`, fit an independent Yule-Walker AR(p) with
    AIC order selection to each FPC-score sequence).
  - `ftsm_forecast` / `ftsm_forecast_multistep` — h-step-ahead FPC-score AR
    forecasts reconstructed into curves (iterative plug-in; `h = 1` is bit-identical
    across the two entry points).
  - `ftsm_update` — dynamic forecast update projecting new observation(s) onto the
    frozen FPC loadings and re-fitting the score AR models without refitting FPCA.
  - `fplsr` — functional PLS forecasting variant (lag-1 per-evaluation-point PLS).
  - Result types: `FtsmResult`, `FtsmForecastResult`, `ArModelResult`, `FplsrResult`.
  - R baseline: `ftsa`. Deterministic; reuses `fdata_to_pc_1d` + `scoring.rs` +
    `fts/acf.rs`.
- **Fréchet / object-data regression + statistics** (`fdars-core::frechet`, new
  `frechet/` module):
  - `MetricSpace` trait (distance + weighted-Fréchet-mean solver) with a
    `WassersteinDensitySpace` (1D-Wasserstein density) backend.
  - `wasserstein2_distance` — 1D 2-Wasserstein distance (quantile-L²).
  - `frechet_mean` / `frechet_variance` — sample Fréchet mean and variance.
  - `frechet_global_reg` (Petersen–Müller global linear weights) and
    `frechet_local_reg` (local-linear Gaussian-kernel weights) — conditional
    density-response regression over Euclidean predictors.
  - `frechet_anova` — Dubey–Müller group-difference test (seeded permutation
    p-value + asymptotic χ²(k−1)).
  - Result types: `FrechetGlobalRegResult`, `FrechetLocalRegResult`,
    `FrechetAnovaResult`.
  - R baseline: `frechet`. Reuses DENS-01's `density_fda.rs` quantile/Wasserstein
    machinery. Divergences (documented in rustdoc): signed-weight regression uses a
    sort-based isotonic projection instead of R's `osqp` QP; the Fréchet-ANOVA
    σ̂ₗ² variance estimator is `[ASSUMED]` (the permutation p-value is the primary,
    robust inference).

### Added — v0.26.0 (FPCA Breadth & Sparse Covariance)

- **Specialized FPCA variants** (`fpca_variants.rs`): `fpca_der`, `fsvd`
  (functional SVD / cross-FPCA via Gram-matrix eigendecomposition),
  `cross_covariance`, `dynamical_correlation`, `ssvd` (sandwich-smoother FPCA).
- **Sparse fast covariance & trajectory bands** (`irreg_fdata/face.rs`):
  `face_covariance` (FACE), `mface_covariance` (+ `MfaceCovResult`),
  `face_trajectory`.

### Added — v0.25.0 (Serial Dependence, Representation & Density Breadth)

- **Functional serial-dependence tooling** (`fts/acf.rs`): functional ACF/PACF with
  white-noise bands, a stationarity test, long-run covariance, and functional
  differencing.
- **Basis-system completions**: `monomial_basis` / `exponential_basis` /
  `power_basis` / `polygonal_basis` factories, a `MultiFunData` container, an
  `Lfd` linear-differential-operator object, and `principal_differential_analysis`.
- **Density object-data FDA** (`density_fda.rs`): log-quantile-density (LQD)
  transform + inverse, LQD-FPCA, 1D Wasserstein barycenter, density normalization.

### Notes

- Verified: whole-crate suite (2460 lib + 172 doc tests) green;
  `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean;
  `cargo fmt --check` clean.
- Per-milestone detail is archived under `.planning/milestones/`.

## [0.24.0] and earlier

See the git history and `.planning/milestones/` archives. Published crate versions
through 0.24.0 correspond to their matching `v0.X.0` git tags.
