# Changelog

All notable changes to `fdars-core` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/). All entries below are **additive and
non-breaking** — no existing public signature changed and no new crate dependency
was added in this span.

## [0.30.0] - 2026-09-01

Covers the v0.30.0 development milestone (Performance & Consolidation Pass) — the
first internally-driven, measure-first depth pass rather than an external gap-audit.
**Behavior-preserving** (numeric outputs unchanged or provably-equivalent within
documented tolerance, proven by existing tests + before/after criterion benchmarks)
and **additive/non-breaking** — no existing public signature was removed and no new
crate dependency was added. Folds in the v0.29.0 development work as well (the
0.29.0 tag was published without a corresponding root changelog entry).

### Performance (behavior-preserving)

- **`face_covariance` −80.7% wall-time** (983.8 → 189.8 ms): the sparse FACE
  covariance estimator now precomputes per-observation Gaussian kernel-weight tables
  once instead of recomputing them per `(s, t)` grid cell (~98% fewer `exp()` calls);
  byte-equivalent output.
- **`fts::dpca` −54% allocations** (17,739 → 8,139 blocks): dynamic-PCA eigenvector
  materialization now uses an index-sort instead of staging into intermediate `Vec`s;
  golden-equivalent within 1e-12.
- **`fsvd` / `ssvd` / `functional_acf`**: eigen matrices are now built via
  `DMatrix::from_fn` (no `Vec` staging, no `m×m` copy); `functional_acf` also
  precomputes `sqrt(w)`. Byte-equivalent (golden 1e-12).
- **`fem_smooth`**: `phi_t_phi` and the assembly matrix are built in a single pass,
  dropping an `N×N` clone; byte-equivalent. (The `O(N³)` Cholesky/GCV cost is
  documented and deferred.)
- **Thread-scaling**: `frechet_anova` and co-clustering initialization gained
  feature-gated rayon parallelism via the `parallel.rs` macros, equivalence-tested
  vs. sequential (ON/OFF) with payback-threshold guards.

### Changed — internal consolidation (no API change)

- Duplicated numerical/statistical machinery factored into shared `pub(crate)`
  helpers: χ²/F survival distributions (`distributions.rs`), per-thread RNG seeding
  (`seed_for_thread`), permutation-test p-value scaffolding (`permutation_pvalue`),
  and the SVD sign-fix core — with all prior call sites migrated. Behavior unchanged.

### Added — additive API consolidation

- `fanova_seeded` — a seedable variant of the permutation `fanova` (the original
  non-seedable `fanova` is retained).
- A `Dim` dimensionality parameter plus 5 unified dispatchers over previously
  separate `_1d` / `_2d` entry points.
- New criterion `[[bench]]` coverage for the previously-unbenchmarked modules
  (`fts`, `frechet`, `boosting_regression`, `coclustering`, `fem_smoothing`,
  `density_fda`, `inference`, `fpca_variants`, `face`), plus a `BENCH-RESULTS.md`
  regression-guard ledger.

### Deprecated

- 6 redundant public forms are now marked `#[deprecated]` in favor of the unified
  alternatives above. **All deprecated signatures still compile and work** — the
  breaking removal is deferred to a future 1.0-readiness release.

## [0.28.0] - 2026-08-23

Covers the v0.28.0 development milestone (Spectral Functional Time Series &
Object-Data Fréchet Regression). Additive and non-breaking — no existing public
signature changed, no new crate dependency.

### Added — v0.28.0 (Spectral Functional Time Series, FTS-03)

- **Spectral functional time series** (`fdars-core::fts`, new `fts/spectral.rs`):
  - `spectral_density` — the spectral density operator: a Bartlett-weighted DFT (via
    `rustfft`) across the lag index of the reused `fts/acf.rs` autocovariance
    operators, evaluated at the Fourier frequencies `θ_k = 2πk/N`; per-frequency
    Hermitian m×m operator (`SpectralDensityResult`).
  - `dpca` — dynamic functional PCA: per-frequency dynamic eigen-filters (inverse-FFT
    of Simpson-metric-scaled eigenvectors) + dynamic scores over the valid interior
    (`DpcaResult`).
  - `dpca_reconstruct` — inverse dynamic filtering with a monotone-non-increasing
    integrated-L2 reconstruction error (`DpcaReconstruction`).
  - R baseline: `freqdom` / `ftsa`. Documented divergences: real-part
    (`SymmetricEigen`) eigendecomposition, `1/2π` omission, score trimming.
- **Functional VAR/VMA + FARMA simulators** (`fdars-core::simulation`):
  - `sim_fvarma` — VAR/VMA from user-supplied m×m operator kernels with Gaussian
    innovations, burn-in, deterministic `seed` (`FvarmaResult`).
  - `sim_farma` — combined AR+MA (FARMA) simulator (`FarmaResult`).

### Added — v0.28.0 (Object-Data Fréchet Regression, FRE-02)

- **Non-density `MetricSpace` backends** (`fdars-core::frechet`, new `frechet/spaces/`):
  - `SpdMatrixSpace` + `SpdMetric { Frobenius, Power(f64), LogCholesky }` — SPD
    covariance-matrix responses.
  - `CorrelationMatrixSpace`, `SphericalSpace` (geodesic exp/log + intrinsic Karcher
    mean), `NetworkSpace` (graph Laplacian), `PointProcessSpace` (intensity L2).
- **Generic Fréchet regression + ANOVA** over any `MetricSpace` backend:
  - `frechet_global_reg_space` / `frechet_local_reg_space` — return a predicted object
    per query row.
  - `frechet_anova_space` — Dubey–Müller Tₙ group-difference test (seeded permutation)
    over object responses.
  - The existing density `frechet_global_reg` / `frechet_local_reg` / `frechet_anova`
    now delegate to shared `pub(crate)` weight/Tₙ helpers — output bit-identical
    (non-breaking). R baseline: `frechet` 0.3.0.

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
