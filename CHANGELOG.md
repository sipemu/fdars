# Changelog

All notable changes to `fdars-core` are documented here. This project adheres to
[Semantic Versioning](https://semver.org/). Under the SemVer 0.x rule, breaking
changes ship in MINOR bumps. **v0.41.0 is an explicitly breaking API-stabilization
release** — it removes long-deprecated forms, seals a couple of internals, and
tightens the public surface ahead of a future 1.0 (see [0.41.0] below; breaking is
API *shape* only — no numeric/behavioral change). Earlier entries in this span
(v0.40.0 and prior) remained additive and non-breaking with respect to public API
signatures and dependencies, with one intentional exception: the v0.40.0 soft-DTW
barycenter correction changes convergence behavior intentionally (see [0.40.0]
Changed below).

## [0.41.0] - 2026-09-07

**BREAKING — 1.0 API Stabilization Pass.** This is the first breaking release after a
long additive run. It settles the public API ahead of a future 1.0 by removing
long-deprecated dimensional forms, sealing internals that were unintentionally public,
marking forward-compatible enums/structs `#[non_exhaustive]`, and unifying naming.
**Breaking is API *shape* only — there is no numeric or behavioral change** to any
retained computation. **MSRV is unchanged** (1.81 for the crate, 1.84 for the `linalg`
feature). See `documentation/STABILITY.md` for the semver + MSRV policy and
`documentation/ROADMAP-TO-1.0.md` for the remaining 1.0 gap checklist.

### Removed

- **API-01 — deprecated dimensional forms removed.** The 6 long-deprecated `*_2d`/
  legacy forms and their crate-root/prelude re-exports are gone:
  - `mean_2d` → use `mean(…, Dim::Two)`
  - `fanova` → use `fanova_seeded(…, 42)` (pass an explicit seed for determinism)
  - `random_tukey_2d` → use `random_tukey(…, Dim::Two)`
  - `random_projection_2d` → use `random_projection(…, Dim::Two)`
  - `fraiman_muniz_2d` → use `fraiman_muniz(…, Dim::Two)`
  - `modal_2d` → use `modal(…, Dim::Two)`

### Changed

#### Surface sealing (API-02)

- `sort_nan_safe` and `solve_gaussian_pub` are now `pub(crate)` (they were
  unintentionally `pub`). These were internal numerical helpers with no intended
  external contract; there is no public replacement.

#### `#[non_exhaustive]` (API-03)

- `#[non_exhaustive]` added to 10 public enums — `PeerPenalty`, `LambdaChoice`,
  `LambdaMethod`, `DesignCriterion`, `OptimalityKind`, `ExtrapolationPolicy`,
  `ImputationMethod`, `SelectionCriterion`, `BasisType`, `BasisCriterion` — and to 2
  result structs — `OptimBandwidthResult`, `KnnCvResult`.
  - **Downstream impact:** exhaustive `match` on these enums now requires a wildcard
    (`_ => …`) arm, and struct-literal construction of the two result structs from
    outside the crate is no longer possible; use the constructing API and read fields.
    This allows future variants/fields to be added without a breaking bump.

#### Naming unification (API-04)

- Renamed for casing/consistency:
  - `funhddC_cluster` → `fun_hddc_cluster`
  - `FosrResult2d` → `Fosr2dResult`
  - `GmmResult` → `GmmFitResult`
  - **Migration:** update call sites / type references to the new names (signatures
    and semantics are unchanged).
- Collapsed dimensional pairs into single dispatchers over a domain enum:
  - `deriv_1d` / `deriv_2d` → `deriv(…, DerivDomain)` returning `DerivResult`.
    **Migration:** call `deriv(&data, DerivDomain::OneD { argvals, nderiv })` and match
    `DerivResult::OneD(m)` (or `DerivDomain::TwoD { … }` / `DerivResult::TwoD(…)`).
  - `lp_self_1d` / `lp_cross_1d` / `lp_self_2d` / `lp_cross_2d` → `lp_self` / `lp_cross`
    taking an `LpDomain`. **Migration:** call `lp_self(…, LpDomain::OneD { … })` (or
    `LpDomain::TwoD { … }`) instead of the dimension-suffixed variants.

### Added

- **Stability documentation (API-04 / STAB deliverables):**
  - `documentation/STABILITY.md` — the semver + MSRV policy (0.x breaking-in-minor
    rule, feature-gate MSRV, deprecation approach).
  - `documentation/ROADMAP-TO-1.0.md` — the remaining 1.0 gap checklist.
- **New public types backing the collapsed APIs:** the `DerivDomain` and `LpDomain`
  domain-selector enums and the `DerivResult` return type.

## [0.40.0] - 2026-09-07

Release Hardening & Ship v0.40.0 (Phase 80). Folds in the unpublished v0.39.0
forward-mode AD core and applies correctness / build fixes from Phases 78 and 79.

### Fixed

- **CORR-01 — soft-DTW backward endpoint-seed gradient bug**: `soft_dtw_backward`
  previously overwrote the required `E[n][m] = 1` endpoint seed in its reverse loop,
  producing an all-zero gradient for every call. All gradient-dependent code paths
  (`soft_dtw_barycenter`, the generic autodiff path) now receive the correct gradient.
  (Phase 78)
- **BUILD-01 — serde feature build**: `cargo build --features serde` now compiles
  cleanly. The build had been broken since Phase 60 due to `ShapeletTransformClassifier`
  embedding a non-serde `ClassifFit`; serde derives have been added to `ClassifFit` and
  all affected types. (Phase 79)

### Changed

- **soft-DTW barycenter convergence (intentional behavior change)**: `soft_dtw_barycenter`
  now genuinely converges to the barycenter via an inverse-curvature optimizer step,
  replacing the previous behavior of silently returning the pointwise mean on an all-zero
  gradient. Output values will differ from prior versions wherever `soft_dtw_barycenter`
  was called. (Phase 78, SDTW-O1 for a full L-BFGS optimizer is backlogged.)
- **CORR-02 — gradient-pass audit**: All hand-written gradient implementations were
  audited; no additional bugs were found beyond CORR-01. (Phase 78)

## [0.39.0] - 2026-09-07

Forward-Mode Automatic Differentiation Core (Phases 75–77). Code-complete but never
published to crates.io; folded into this release.

### Added

- **Scalar trait + Dual number substrate (DIF-01)**: New `src/autodiff.rs` module
  exports a generic `Scalar` trait and a `Dual<T>` value/tangent number type, enabling
  forward-mode AD over arbitrary scalar types. Includes known-answer tests, central
  finite-difference cross-checks, and f64-parity tests. (Phase 75)
- **Differentiable elastic distance + FPCA scores (DIF-02, DIF-03)**: `soft_dtw_distance_generic`
  is a fully generic soft-DTW distance instantiable with `Dual<f64>` for exact gradient
  computation; `amplitude_distance_at_warp` likewise. FPCA score projection is now
  differentiable via the generic `Scalar` interface — `∂score_k/∂curve[j]` equals the
  analytic closed form `rotation[j,k] · weights[j]` to ≤1e-12. All three validation tiers
  (oracle, finite-difference, f64-parity) are covered by inline tests. (Phase 76)
- **Gradient API + composition demo (DIF-04)**: `grad(f, x)` returns `(value, gradient)`
  for any scalar objective over `Vec<f64>`; `jacobian` provides the full Jacobian matrix.
  A composition demo wires two Phase-76 differentiable ops into a single objective and
  verifies the composed gradient against central finite differences (≤1e-6). Crate-root
  and `prelude::*` re-exports for `Scalar`, `Dual`, `diff`, `grad`, and the generic
  distance functions. Module doctest passes under `cargo test --doc`. (Phase 77)

## [0.38.0] - 2026-09-05

VEESA — Elastic Shape Explainability & Conformal Anomaly Detection (Phases 72–74).
Closes the gaps against the VEESA paper (Goode, Tucker & Ries), the `sandialabs/veesa`
R package, and arXiv 2504.01172 (elastic conformal anomaly detection). Reuse-first over
the shipped jfPCA / elastic-distance / conformal machinery; no new dependency.

### Added

- **jfPCA fit/transform seam** — `jfpca_fit` returns a reusable `JfpcaModel` (stores the
  trained Karcher-mean template, `mean_q`/`mean_psi`/`mean_srsf`, joint eigenvector
  components, `balance_c`, `argvals`, eigenvalues, and the embedded `JointFpcaResult`);
  training scores reproduce `joint_fpca` within 1e-8. `JfpcaModel::transform` projects new
  out-of-sample curves onto the trained basis (aligns to the trained template via
  `align_to_target`); `JfpcaModel::score_training` gives the exact training round-trip;
  `JfpcaTransform` result. (VEE-01, VEE-02)
- **Model-agnostic permutation feature importance** — `elastic_pfi` over jfPCA PC scores,
  generic over any predictor closure `Fn(&FdMatrix) -> Vec<f64>`; `PfiMetric`
  (Mse/Mae/Accuracy/Custom); deterministic under seed; `ElasticPfiResult`. (VEE-03)
- **Principal-direction reconstruction** — `JfpcaModel::principal_directions` reconstructs
  μ ± c·σⱼ split into amplitude and phase parts (`PrincipalDirections`); `c = 0` reproduces
  the jfPCA mean. (VEE-04)
- **VEESA pipeline** — `veesa_pipeline` ties fit → transform → PFI end-to-end
  (`VeesaPipelineResult`). (VEE-05)
- **Elastic conformal anomaly detection** — `NonConformityScore` extended with
  `AmplitudeElastic` / `PhaseElastic` / `CombinedElastic`; `elastic_nonconformity` (scores a
  curve against a reference template, non-negative, zero for an identical curve);
  `elastic_conformal_anomaly` inductive detector (per-curve conformal p-values, anomaly flags
  at level α, calibrated threshold — catches magnitude AND shape outliers);
  `ConformalAnomalyConfig` / `ConformalAnomalyResult`. The existing `conformal_prediction_band`
  path is unchanged. (ECA-01, ECA-02, ECA-03)

## [0.37.0] - 2026-09-04

WAV — Wavelet-Domain Functional Regression (Phases 69–71). Promotes GAP-07.

### Added

- **Discrete wavelet transform** — new `wavelet/` module: single-level orthonormal DWT
  (Haar + Daubechies db2–db10) with exact-adjoint analysis/synthesis under periodic and
  symmetric boundaries; multi-level Mallat pyramid (`WaveletCoeffs`, `decompose`,
  `reconstruct`) plus the `FdMatrix` batch path; perfect reconstruction ≤1e-10 across
  families, levels, boundary modes, and non-power-of-2 lengths.
- **`wcr`** — wavelet-domain scalar-on-function regression (PCR or PLS on per-curve
  concatenated wavelet-coefficient designs; β(t) recovered by inverse DWT); `WcrResult` with
  out-of-sample `predict` + coefficient/fitted accessors.
- **`wnet`** — wavelet-domain elastic-net scalar-on-function regression (per-coefficient
  L1+L2 coordinate descent, deterministic cross-validated λ); `WnetResult` with `predict`.
  Full crate-root + prelude re-exports and a running end-to-end module doctest.

## [0.36.0] - 2026-09-04

PEER — Structured-Penalty & Longitudinal Scalar-on-Function Regression (Phases 66–68).
Promotes GAP-06. (Crate versions 0.36.0 and 0.37.0 were code-complete but not published
separately — their code ships in the 0.38.0 crate release.)

### Added

- **`peer`** — structured-penalty scalar-on-function regression via the null-space/range-space
  decomposition of the penalty operator; `PeerPenalty` families (Ridge, 2nd-difference,
  caller-supplied Decree Q); `PeerResult` (β(t), intercept, fitted values, effective df,
  diagnostics).
- **Automatic λ selection** — `LambdaChoice` (Fixed / Gcv / Reml): deterministic GCV grid +
  self-contained REML EM.
- **`lpeer`** — longitudinal PEER with subject random effects (REML EM via `famm`);
  `LpeerResult` with variance components. Out-of-sample `predict` on both; full re-exports +
  doctest.

## [0.35.0] - 2026-09-03

Optimal Experimental Design for Sparse FDA (FOptDes, Phases 64–65). Promotes GAP-05.
Crate bumped 0.34.0 → 0.35.0.

### Added

- **`design_criterion`** — evaluates a candidate design under `DesignCriterion::Trajectory`
  (integrated BLUP-MSE) or `DesignCriterion::Score(A|D)` (FPC-score posterior covariance A-/
  D-optimality); `OptimalityKind` enum. New `optimal_design.rs`.
- **`optimal_design`** — deterministic greedy sequential forward selection over an estimated
  `PaceFpcaResult` (read-only, no re-estimation); `OptDesConfig` / `OptDesResult`. Full
  re-exports + doctest + criterion benchmark.

## [0.34.0] - 2026-09-02

k-Shape Clustering & Shape-Based Distance (Phases 61–63). Promotes GAP-03. Crate bumped
0.33.0 → 0.34.0.

### Added

- **Shape-based distance** — `src/metric/sbd.rs`: `sbd` (FFT normalized cross-correlation →
  distance + optimal shift, scale/offset-invariant) and `sbd_distance_matrix`.
- **k-Shape clustering** — `src/kshape.rs`: `kshape_fd` (iterative SBD assignment +
  shape-extraction centroids, `n_init` restarts, deterministic seeding) and
  `KShapeResult::predict`.
- **SBD k-medoids** — `sbd_kmedoids`. Full re-exports + criterion bench.

## [0.33.0] - 2026-09-02

Shapelet Transform & Classification (Phases 57–60). Promotes GAP-02 (discovery-based;
learning-shapelets deferred). Crate bumped 0.32.0 → 0.33.0.

### Added

- **Shapelet distance core** — `src/shapelet/distance.rs`: per-window z-normalized
  min-Euclidean with early-abandon; the `Shapelet` type.
- **Discovery & ranking** — `src/shapelet/discovery.rs`: candidate generation + `QualityMeasure`
  (information gain / F-statistic) + top-K selection with self-similarity pruning
  (`ShapeletSet`); byte-deterministic.
- **Shapelet transform** — `src/shapelet/transform.rs`: `shapelet_transform`,
  `shapelet_transform_fit`, and out-of-sample `ShapeletTransformFit::transform`.
- **Bundled classifier** — `src/shapelet/classifier.rs`: `shapelet_classifier_fit`
  (discover → transform → classify; kNN default, LDA optional) + `ShapeletClassifierFit::predict`.
  Criterion bench.

## [0.32.0] - 2026-09-02

Global Alignment Kernel & Kernel Clustering (Phases 54–56). Promotes GAP-01. First
implementation milestone after three audit/consolidation cycles; crate bumped 0.30.0 → 0.32.0.

### Added

- **GAK kernel** — `metric/gak.rs`: Cuturi Triangular Global Alignment Kernel (log-domain
  forward DP atop the soft-DTW lattice, PSD in [0,1] with unit diagonal); `gak_gram_matrix`;
  `sigma_gak` median-distance bandwidth heuristic.
- **Gram-matrix export** — `gak_gram_train` / `gak_gram_predict` (cross-normalized against the
  stored training diagonals) for external `SVC(kernel='precomputed')` handoff.
- **Kernel-k-means** — `kernel_kmeans.rs`: kernel-trick clustering on the GAK Gram,
  `n_init` restarts, deterministic seeding, out-of-sample `predict`.

## [0.31.0] - 2026-09-02

Multi-Ecosystem Gap Audit (Phases 52–53). **No `fdars-core` code changes** — an audit-only
milestone producing `.planning/research/GAP-AUDIT-REPORT.md` (MATLAB, Julia, tidyfun/refund,
and Python-beyond-scikit-fda capability surveys) and a value-ranked `GAP-BACKLOG.md` (7
net-new gaps, GAP-01…GAP-08). No crate version change and no git tag.

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
