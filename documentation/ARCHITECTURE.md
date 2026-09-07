<!-- generated-by: gsd-doc-writer -->
# fdars-core Architecture

`fdars-core` (v0.40.0) is a pure-Rust functional data analysis (FDA) library. It provides
algorithm implementations across regression, classification, clustering, depth measures, elastic
shape analysis, seasonal decomposition, statistical process monitoring (SPM), and model
explainability. All code is in a single crate at `fdars-core/` within a workspace whose
`Cargo.toml` declares only that one member.

---

## System Overview

The library takes collections of curves (functional data) as its primary inputs, represented as
column-major matrices, and returns structured result types. There are no runtime dependencies,
no configuration files, and no external services — all behavior is controlled via Cargo feature
flags and function parameters. The architecture is layered: a core infrastructure layer
(matrix type, error type, linear algebra helpers, parallelism macros) underlies a large set of
domain modules, all re-exported through a flat public API at the crate root and a convenience
`prelude` module.

---

## Component Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│  Public API Surface (src/lib.rs, src/prelude.rs)                         │
│  Flat re-exports + fdars_core::prelude::* convenience glob               │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ calls
┌──────────▼───────────────────────────────────────────────────────────────┐
│  Domain Modules (see full listing below)                                  │
│  alignment · classification · depth · seasonal · spm · scalar_on_function│
│  regression · explain · explain_generic · elastic_* · wavelet · tolerance│
│  fts · frechet · inference · gmm · basis · smoothing · irreg_fdata · ... │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ uses
┌──────────▼───────────────────────────────────────────────────────────────┐
│  Infrastructure Layer                                                      │
│  src/matrix.rs     FdMatrix + FdCurveSet (column-major storage)           │
│  src/error.rs      FdarError (4 variants)                                 │
│  src/linalg.rs     Cholesky / OLS helpers (pub(crate))                    │
│  src/parallel.rs   5 parallelism macros (feature-gated rayon)             │
│  src/helpers.rs    Numerical utilities (Simpson weights, interp, spline)  │
│  src/fdata.rs      Mean, centering, norms, derivatives                    │
└──────────┬───────────────────────────────────────────────────────────────┘
           │ optional dependencies
┌──────────▼───────────────────────────────────────────────────────────────┐
│  External Crates                                                           │
│  nalgebra 0.33  — DMatrix, SVD (always present)                           │
│  rayon 1.10     — parallel iterators (feature = "parallel")               │
│  faer 0.23      — thin SVD for FPCA (feature = "linalg", Rust >= 1.84)   │
│  anofox-regression 0.4 — ridge solve (feature = "linalg")                │
│  rustfft 6.2    — FFT for seasonal / spectral modules                     │
│  serde 1.0      — serialization (feature = "serde")                      │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Data Representation

### `FdMatrix` — column-major matrix (`src/matrix.rs`)

`FdMatrix` is the single concrete type used for all functional data. It wraps a flat `Vec<f64>`
in column-major (Fortran) order.

**Indexing invariant:** element `(row, col)` is at flat index `row + col * nrows`.

```
Observation × evaluation-point matrix (n=3 curves, m=4 points):

    col 0    col 1    col 2    col 3
  [ 1.0  |  4.0  |  7.0  | 10.0  ]   ← observation 0
  [ 2.0  |  5.0  |  8.0  | 11.0  ]   ← observation 1
  [ 3.0  |  6.0  |  9.0  | 12.0  ]   ← observation 2

Flat storage: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
```

- Rows = observations / curves (index `i`)
- Columns = evaluation points (index `j`)
- 2D surfaces (m1 × m2 grids) are flattened to `n × (m1 * m2)` matrices
- Zero-copy column access: `matrix.column(j)` returns a contiguous `&[f64]`
- Row operations without materializing: `row_to_buf`, `row_dot`, `row_l2_sq`
- `nalgebra` interop: `to_dmatrix()` / `from_dmatrix()` for SVD operations
- serde support when `feature = "serde"` is enabled

**Construction:**
```rust
let mat = FdMatrix::from_column_major(data, nrows, ncols)?;  // validates length
let mat = FdMatrix::from_slice(&slice, nrows, ncols)?;       // copies
let mat = FdMatrix::zeros(nrows, ncols);
```

### `FdCurveSet` — multidimensional curves (`src/matrix.rs`)

For d-dimensional curves (e.g., 3D trajectories), `FdCurveSet` holds one `FdMatrix` per
dimension. For d=1 it is equivalent to a single `FdMatrix`.

### `IrregFdata` — irregularly sampled data (`src/irreg_fdata/`)

For functional data where each observation is evaluated on its own grid (unequal sampling times).
Used by sparse FPCA (PACE estimator in `src/pace_fpca.rs`) and FACE covariance estimation.

---

## Error Handling (`src/error.rs`)

All public functions return `Result<T, FdarError>`. `FdarError` is `#[non_exhaustive]` with
four variants:

| Variant | When used |
|---------|-----------|
| `InvalidDimension { parameter, expected, actual }` | Matrix shape mismatch, empty input, length inconsistency |
| `InvalidParameter { parameter, message }` | Out-of-range value (e.g., `ncomp > min(n, m)`, bandwidth <= 0) |
| `ComputationFailed { operation, detail }` | SVD non-convergence, Cholesky on non-positive-definite matrix |
| `InvalidEnumValue { enum_name, value }` | Integer-to-enum conversion failure |

Dimension and parameter checks happen at function entry points. Internal helpers may return
`Option<T>`, bridged to `Result` via `.ok_or_else()` before being exposed to the public API.

---

## Feature Gates

Declared in `fdars-core/Cargo.toml`:

| Feature | Default | Description |
|---------|---------|-------------|
| `parallel` | **yes** | Enables rayon-based parallelism via the 5 macros in `src/parallel.rs` |
| `linalg` | no | Enables `faer 0.23` (thin SVD) and `anofox-regression 0.4` (ridge). Requires Rust >= 1.84. Not WASM-compatible. |
| `serde` | no | Adds `Serialize`/`Deserialize` to `FdMatrix`, `FpcaResult`, `SpmChart`, and other core types |
| `js` | no | Enables `getrandom/js` for WASM builds (`wasm32-unknown-unknown` target) |
| `dhat-heap` | no | Dev-only: activates `#[global_allocator]` for allocation profiling. Never use in production. |

The `linalg` feature is **not** in `default` because it requires Rust 1.84+, which is above
the MSRV of 1.81 set for CRAN Windows compatibility. The R package uses
`default-features = false`.

---

## Parallelism Model (`src/parallel.rs`)

Five macros provide a compile-time switch between rayon and sequential iteration:

| Macro | Sequential fallback | Parallel version |
|-------|---------------------|-----------------|
| `iter_maybe_parallel!(expr)` | `into_iter()` | `into_par_iter()` |
| `slice_maybe_parallel!(expr)` | `.iter()` | `.par_iter()` |
| `slice_maybe_parallel_mut!(expr)` | `.iter_mut()` | `.par_iter_mut()` |
| `maybe_par_chunks_mut!(slice, size, closure)` | `.chunks_mut().for_each` | `.par_chunks_mut().for_each` |
| `maybe_par_chunks_mut_enumerate!(slice, size, closure)` | `.chunks_mut().enumerate().for_each` | `.par_chunks_mut().enumerate().for_each` |

Per-thread RNG seeding convention for reproducible parallel computation:
```rust
StdRng::seed_from_u64(seed + k as u64)  // k = thread/chunk index
```

Tests always run in sequential mode (no `parallel` feature in test invocations without
`--features parallel`), ensuring deterministic test output.

---

## Linear Algebra Backends

### Always available — nalgebra (`nalgebra 0.33`)

- `FdMatrix::to_dmatrix()` converts to `nalgebra::DMatrix<f64>` for SVD operations
- `nalgebra::SVD` is the production SVD path when the `linalg` feature is disabled
- Used in `src/regression.rs` for `fdata_to_pc_1d` / FPCA

### `linalg` feature — faer + anofox-regression

- `faer 0.23`: thin SVD (`faer::linalg::solvers::Svd`) used for FPCA in `src/regression.rs`
  when `#[cfg(feature = "linalg")]` is active. Provides better performance for large matrices
  but requires Rust >= 1.84.
- `anofox-regression 0.4`: ridge regression via `RidgeRegressor` / `FittedRegressor` from
  the `argmin` solver framework.

The SVD implementation is dual-pathed with equivalence tests ensuring faer and nalgebra produce
identical results (up to sign conventions enforced by the codebase).

### Always available — `src/linalg.rs` (pub(crate))

Pure-Rust Cholesky factorization, forward/backward substitution, Mahalanobis distance, and
OLS helpers shared across `scalar_on_function`, `function_on_scalar`, `famm`, and related
modules. These are internal utilities; they do not depend on nalgebra or faer.

### FFT — `rustfft 6.2`

Used by `src/seasonal/` and `src/metric/fourier.rs` for frequency-domain operations. Always
present (not feature-gated).

---

## Domain Modules

All modules are under `fdars-core/src/`. Subdirectory modules have a `mod.rs` that re-exports
all public items. Single-file modules are standalone `.rs` files.

### Regression and Prediction

| Module | Path | Description |
|--------|------|-------------|
| `regression` | `src/regression.rs` | FPCA (`fdata_to_pc_1d`), PLS (`fregre_pls`), ridge (`RidgeResult`, `linalg`-gated), `FpcaResult` |
| `scalar_on_function` | `src/scalar_on_function/` | Scalar-on-function regression: OLS (`fregre_lm`), PLS (`fregre_pls`), logistic (`functional_logistic`), robust (`fregre_huber`/`fregre_l1`), nonparametric, GLM, multi-response, bootstrap CV |
| `function_on_scalar` | `src/function_on_scalar.rs` | Function-on-scalar regression (FOSR) |
| `function_on_scalar_2d` | `src/function_on_scalar_2d.rs` | 2D FOSR with tensor-product penalty (`Grid2d`, `FosrResult2d`) |
| `fof_regression` | `src/fof_regression.rs` | Function-on-function regression |
| `elastic_regression` | `src/elastic_regression/` | Elastic shape-based regression (PCR, logistic, scalar-on-shape) |
| `boosting_regression` | `src/boosting_regression/` | Gradient-boosted FOSR, Bayesian FOSR, GAMLSS, stability analysis |
| `concurrent_regression` | `src/concurrent_regression.rs` | Varying-coefficient functional regression |
| `famm` | `src/famm.rs` | Functional additive mixed models |

### Dimensionality Reduction and Variants

| Module | Path | Description |
|--------|------|-------------|
| `fpca_variants` | `src/fpca_variants.rs` | Derivative FPCA, weighted FPCA |
| `elastic_fpca` | `src/elastic_fpca.rs` | Elastic FPCA via SRVFs and Karcher mean |
| `pace_fpca` | `src/pace_fpca.rs` | Sparse/irregular FPCA (Yao–Müller–Wang PACE estimator) |
| `jfpca_model` | `src/jfpca_model.rs` | Joint FPCA model with fit/transform seam |
| `density_fda` | `src/density_fda.rs` | Log-quantile-density (LQD) FPCA for density-valued data |
| `frechet` | `src/frechet/` | Fréchet mean/regression in metric spaces |

### Classification

| Module | Path | Description |
|--------|------|-------------|
| `classification` | `src/classification/` | LDA, QDA, kNN, kernel, DD classifiers; CV; `ClassifFit` / `ClassifResult` |
| `shapelet` | `src/shapelet/` | Shapelet discovery, transform, and shapelet-based classification |

### Depth Measures

| Module | Path | Description |
|--------|------|-------------|
| `depth` | `src/depth/` | Fraiman-Muniz (1D/2D), modal, band, half-region, random projection, random Tukey, RPD, ERL, spatial, L-inf, extremal, hypo-epi depths |
| `streaming_depth` | `src/streaming_depth/` | Online/streaming depth measures (band, Fraiman-Muniz, MBD, rolling) |

### Clustering

| Module | Path | Description |
|--------|------|-------------|
| `clustering` | `src/clustering.rs` | k-means and fuzzy c-means |
| `clustering_advanced` | `src/clustering_advanced.rs` | Spectral and hierarchical clustering |
| `gmm` | `src/gmm/` | Gaussian Mixture Model EM clustering with basis-type projection |
| `kshape` | `src/kshape.rs` | k-Shape and SBD-backed k-medoids for time-series functional data |
| `kernel_kmeans` | `src/kernel_kmeans.rs` | Kernel k-means |
| `coclustering` | `src/coclustering.rs` | Functional co-clustering |

### Registration and Elastic Analysis

| Module | Path | Description |
|--------|------|-------------|
| `alignment` | `src/alignment/` | Elastic curve registration (SRVF), Karcher mean, pairwise alignment, NDimensional alignment, outliers, diagnostics, generative warping |
| `landmark` | `src/landmark.rs` | Landmark-based registration |
| `warping` | `src/warping.rs` | Warping function utilities |
| `elastic_changepoint` | `src/elastic_changepoint.rs` | Elastic changepoint detection |
| `elastic_explain` | `src/elastic_explain.rs` | Explainability for elastic shape models |
| `elastic_pfi` | `src/elastic_pfi.rs` | Permutation feature importance over jFPCA / VEESA pipeline |
| `conformal` | `src/conformal/` | Conformal prediction and anomaly detection |

### Seasonal and Time-Series

| Module | Path | Description |
|--------|------|-------------|
| `seasonal` | `src/seasonal/` | Period estimation (auto-period, Lomb-Scargle, SAZED), peak detection, seasonal strength, STL decomposition, Hilbert envelope, matrix profile |
| `detrend` | `src/detrend/` | Linear, polynomial, LOESS, STL, SSA, and automatic detrending |
| `fts` | `src/fts/` | Functional time-series: ACF, spectral analysis, FTSM forecasting |

### Statistical Process Monitoring (`src/spm/`)

Phase-1 control chart initialization, EWMA, MEWMA, AMEWMA, CUSUM, Hotelling T², FRCC,
bootstrap control limits, elastic SPM, iterative SPM, and multi-FPCA-based monitoring.

### Explainability

| Module | Path | Description |
|--------|------|-------------|
| `explain` | `src/explain/` | Model-specific explainability: PDP, SHAP, LIME, ALE, importance, sensitivity, counterfactual, advanced diagnostics |
| `explain_generic` | `src/explain_generic/` | Generic explainability via `FpcPredictor` trait (see Key Abstractions) |

### Inference and Statistics

| Module | Path | Description |
|--------|------|-------------|
| `inference` | `src/inference/` | Functional linear model inference, Hotelling T², permutation tests, simultaneous confidence bands (SCB), Fréchet ANOVA |
| `tolerance` | `src/tolerance/` | Functional tolerance bands (normal, conformal, elastic, FPCA-based, Degras) |
| `scoring` | `src/scoring.rs` | Proper scoring rules for functional forecasts |
| `peer` | `src/peer.rs` | PEER covariance estimation |
| `pda` | `src/pda.rs` | Penalized discriminant analysis |
| `permutation_test` | `src/permutation_test.rs` | Internal permutation test utilities (pub(crate)) |

### Basis Systems and Smoothing

| Module | Path | Description |
|--------|------|-------------|
| `basis` | `src/basis/` | B-spline, P-spline, Fourier, polynomial, power, monomial, exponential, polygonal, constant bases; automatic selection |
| `smoothing` | `src/smoothing.rs` | Nadaraya-Watson, local polynomial regression, bandwidth selection |
| `smooth_basis` | `src/smooth_basis.rs` | Basis-representation smoothing |
| `fem_smoothing` | `src/fem_smoothing.rs` | FEM-based smoothing with GCV bandwidth selection |

### Distance and Metric (`src/metric/`, `src/distance.rs`)

Lp norms, Hausdorff, DTW, soft-DTW, SBD, Fourier distance, GAK, KL divergence, PCA distance,
hierarchical shift metric. `pairwise_distance_matrix` is the primary entry point.

### Miscellaneous Infrastructure

| Module | Path | Description |
|--------|------|-------------|
| `fdata` | `src/fdata.rs` | Mean, centering, variance, norms, derivatives, interpolation |
| `helpers` | `src/helpers.rs` | Simpson weights, spline interpolation, numerical utilities (R²  , AIC, BIC) |
| `simulation` | `src/simulation.rs` | Synthetic functional data generation |
| `covariance` | `src/covariance.rs` | Covariance kernels and Gaussian process utilities |
| `dim` | `src/dim.rs` | Dimensionality helpers |
| `wire` | `src/wire.rs` | `FdaData` layered pipeline container; nodes read and add named layers |
| `multi_fdata` | `src/multi_fdata.rs` | Multi-group functional data containers |
| `utility` | `src/utility.rs` | General utility functions |
| `validation` | `src/validation.rs` | Shared input validation helpers |
| `andrews` | `src/andrews.rs` | Andrews functional plots |
| `optimal_design` | `src/optimal_design.rs` | Functional optimal experimental design |
| `outliers` | `src/outliers.rs` | Outlier detection (magnitude and shape) |

---

## Key Abstractions

### `FdMatrix` (`src/matrix.rs`)

Column-major matrix type. Prevents manual index arithmetic. Carries dimensions alongside data.
Used as the input and output type for nearly every public function in the library.

### `FpcaResult` (`src/regression.rs`)

Output of `fdata_to_pc_1d` (and related FPCA functions). Carries the canonical fields:

| Field | Meaning |
|-------|---------|
| `scores` | FPC scores, n × ncomp |
| `rotation` | Loadings (eigenfunctions), m × ncomp |
| `mean` | Mean function, length m |
| `weights` | Integration weights, length m |
| `singular_values` | Singular values (component scale) |
| `centered` | Mean-centered data, n × m |

Provides a `.project(&new_data)` method for out-of-sample projection. Embeds into
regression and classification result types for the generic explainability layer.

### `FpcPredictor` trait (`src/explain_generic/mod.rs`)

A `Send + Sync` trait abstracting over any FPC-based model (regression, logistic, classification)
for generic explainability:

```rust
pub trait FpcPredictor: Send + Sync {
    fn fpca_mean(&self) -> &[f64];
    fn fpca_rotation(&self) -> &FdMatrix;
    fn ncomp(&self) -> usize;
    fn training_scores(&self) -> &FdMatrix;
    fn task_type(&self) -> TaskType;
    fn fpca_weights(&self) -> &[f64];
    fn predict_from_scores(&self, scores: &[f64], ...) -> f64;
}
```

Implemented for `FregreLmResult`, `FunctionalLogisticResult`, and `ClassifFit`. Enables
14+ generic explainability functions (PDP, SHAP, LIME, ALE, Sobol, Friedman H, permutation
importance, saliency, anchors, counterfactuals, stability, VIF, prototype/criticism) to work
across all model types without per-model specialization.

`TaskType` distinguishes `Regression`, `BinaryClassification`, and
`MulticlassClassification(usize)`.

### Config Structs

Complex algorithms accept a configuration struct rather than a long parameter list:

| Struct | Used in |
|--------|---------|
| `ElasticConfig` | `alignment` |
| `StlConfig` | `seasonal/stl` |
| `GmmClusterConfig` | `gmm` |
| `ClassifCvConfig` | `classification/cv` |
| `ConformalConfig` | `conformal` |
| `KShapeConfig` | `kshape` |
| `ShapeletDiscoveryConfig`, `ShapeletClassifierConfig` | `shapelet` |
| `OptDesConfig` | `optimal_design` |

### Result Structs

All fitting functions return a dedicated immutable result struct. All result structs derive
`Debug, Clone, PartialEq` and carry `#[non_exhaustive]` for forward-compatible evolution.
Expensive computations are marked `#[must_use]`.

### `FdaData` pipeline container (`src/wire.rs`)

A composable, additive pipeline container. Nodes declare what layers they require and add new
named layers; layers are never destructively replaced. Replaces per-type wire enums with a
single structure that carries curves, argvals, grouping metadata, scalar covariates, and any
number of analysis result layers.

---

## Directory Structure

```
fdars-core/
├── Cargo.toml          Package manifest, feature declarations, 20+ bench targets, 28 examples
├── src/
│   ├── lib.rs          Crate root: module declarations + top-level re-exports
│   ├── prelude.rs      Convenience glob re-exports of the most common types
│   ├── error.rs        FdarError (4 variants)
│   ├── matrix.rs       FdMatrix + FdCurveSet (column-major storage core)
│   ├── parallel.rs     5 parallelism macros (rayon / sequential compile-time switch)
│   ├── linalg.rs       pub(crate) Cholesky / OLS helpers
│   ├── helpers.rs      Numerical utilities (Simpson weights, spline interp, gradients)
│   ├── fdata.rs        Functional data operations (mean, centering, norms, derivatives)
│   ├── regression.rs   FPCA, PLS, ridge (linalg-gated), FpcaResult
│   ├── alignment/      Elastic registration, Karcher mean, diagnostics (18 files)
│   ├── basis/          Basis systems: B-spline, Fourier, P-spline, etc. (13 files)
│   ├── boosting_regression/ Boosting + Bayesian FOSR (6 files)
│   ├── classification/ LDA, QDA, kNN, kernel, DD (9 files)
│   ├── conformal/      Conformal prediction and anomaly detection (9 files)
│   ├── depth/          Depth measures (14 files)
│   ├── detrend/        Detrending methods (9 files)
│   ├── elastic_regression/ Elastic shape regression (6 files)
│   ├── explain/        Model-specific explainability (12 files)
│   ├── explain_generic/ FpcPredictor trait + generic XAI (13 files)
│   ├── frechet/        Fréchet mean / regression (5 files)
│   ├── fts/            Functional time-series (4 files)
│   ├── gmm/            Gaussian mixture models (7 files)
│   ├── inference/      Hypothesis testing, SCB, ANOVA (7 files)
│   ├── irreg_fdata/    Irregularly sampled data (4 files)
│   ├── metric/         Distance functions (12 files)
│   ├── scalar_on_function/ Scalar-on-function regression suite (9 files)
│   ├── seasonal/       Seasonal analysis and decomposition (12 files)
│   ├── shapelet/       Shapelet discovery and classification (5 files)
│   ├── spm/            Statistical process monitoring (18 files)
│   ├── streaming_depth/ Online depth measures (7 files)
│   └── tolerance/      Functional tolerance bands (9 files)
├── benches/            Criterion benchmark suite (20+ bench targets)
├── examples/           28 runnable examples (01_simulation → 28_berkeley_growth)
└── tests/              Integration tests (cross-module, equivalence goldens)
```

---

## Architectural Invariants

1. **Column-major layout**: All matrices stored as flat `Vec<f64>` in column-major order.
   Index: `data[i + j * nrows]`. Column access (`data.column(j)`) is zero-copy and contiguous;
   row access (`row_to_buf`, `row_dot`, `row_l2_sq`) gathers without materializing.

2. **No panics on bad input**: Public functions always return `Result<T, FdarError>`. Panics
   are reserved for genuine programmer errors (e.g., `debug_assert!` invariants internal to
   an algorithm).

3. **No interior mutability**: Result types are immutable after construction. No `RefCell` or
   `Mutex` in the public API.

4. **Integration weights required**: All functional inner products and FPCA computations use
   Simpson's rule weights (`helpers::simpsons_weights`) or caller-supplied weights. Raw sums
   without quadrature weights are considered a bug.

5. **MSRV 1.81 for the crate**: The `linalg` feature alone requires Rust 1.84+ (faer 0.23+).
   All code outside `#[cfg(feature = "linalg")]` must compile on Rust 1.81. The R package
   (external CRAN package `fdars-r`) uses `default-features = false` to stay within MSRV.

6. **Deterministic seeding**: Parallel loops that require random numbers seed per-thread as
   `StdRng::seed_from_u64(seed + k as u64)`. This ensures reproducibility regardless of
   thread scheduling.

7. **SVD route duality**: FPCA SVD is dual-pathed in `src/regression.rs`. With `linalg`,
   `faer::linalg::solvers::Svd` is used; without it, `nalgebra::SVD` is used. Equivalence
   tests assert bit-identical output (up to sign normalization) between both paths.

---

## Binding Layers

### WASM (`wasm32-unknown-unknown`)

The crate compiles for WASM via the `js` feature, which enables `getrandom/js` for WASM-
compatible random seeding. The `parallel` and `linalg` features are incompatible with WASM
(rayon and faer are both native-only). CI runs a dedicated WASM build job:
```
cargo build --target wasm32-unknown-unknown --no-default-features
```
Higher-level WASM bindings (wasm-bindgen) are provided by a separate package not in this
workspace. <!-- VERIFY: separate wasm-bindgen wrapper package name and location -->

### R (`fdars-r`)

A separate CRAN package `fdars-r` wraps `fdars-core` via Rust-R FFI. It depends on
`fdars-core` with `default-features = false` to stay within the MSRV constraint of 1.81
required by CRAN's Windows Rust toolchain. This package is external to this workspace and
maintained separately. <!-- VERIFY: fdars-r repository URL and CRAN submission status -->

---

## CI and Build Targets

From `.github/workflows/rust-ci.yml`:

- **Full test matrix**: `cargo test --features linalg,parallel,serde` (multi-version: stable, beta, nightly)
- **Sequential mode**: `cargo test --no-default-features --features linalg`
- **Clippy**: `cargo clippy --all-targets --features linalg,parallel,serde -- -D warnings`
- **Docs**: `cargo doc --no-deps --features linalg,parallel,serde`
- **WASM build**: `cargo build --target wasm32-unknown-unknown --no-default-features`
- **Coverage**: Codecov (70% project target, 50% patch minimum; config at `codecov.yml`)

There are 20+ Criterion benchmark targets in `fdars-core/benches/` covering all major
algorithm domains. Benchmarks use `criterion 0.5` with HTML report generation.
