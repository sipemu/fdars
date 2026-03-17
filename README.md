# Functional Data Analysis (FDA)

[![Rust CI](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fdars-core.svg)](https://crates.io/crates/fdars-core)
[![Documentation](https://docs.rs/fdars-core/badge.svg)](https://docs.rs/fdars-core)
[![codecov](https://codecov.io/gh/sipemu/fdars/graph/badge.svg)](https://codecov.io/gh/sipemu/fdars)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Functional Data Analysis tools implemented in Rust with R bindings.

## Packages

| Package | Language | Registry | Folder | Status |
|---------|----------|----------|--------|--------|
| fdars | R | CRAN | [sipemu/fdars-r](https://github.com/sipemu/fdars-r) | [![CRAN](https://www.r-pkg.org/badges/version/fdars)](https://cran.r-project.org/package=fdars) |
| fdars-core | Rust | crates.io | `fdars-core/` | [![Crates.io](https://img.shields.io/crates/v/fdars-core.svg)](https://crates.io/crates/fdars-core) |

## Features

### Core

- **Simulation**: Karhunen-Loève expansion (Fourier/Legendre/Wiener), Gaussian process generation with 8 covariance kernels and kernel algebra
- **Functional Operations**: Mean, centering, derivatives, Lp norms, geometric median, Andrews curves
- **Smoothing & Basis**: Nadaraya-Watson, local linear/polynomial, k-NN; B-splines, Fourier, P-splines with GCV/AIC/BIC

### Descriptive

- **Depth**: Fraiman-Muniz, modal, band, modified band, random projection, random Tukey, RPD, functional spatial, kernel spatial, modified epigraph; streaming online depth
- **Distances**: Lp, Hausdorff, DTW, Soft-DTW, elastic (Fisher-Rao), amplitude/phase, Fourier/h-shift/PCA/derivative/basis-coefficient semimetrics, KL divergence
- **Outlier Detection**: LRT-based with bootstrap thresholding

### Regression & Classification

- **Scalar-on-Function**: FPC linear, nonparametric kernel, functional logistic, robust (L1/Huber), CV component selection
- **Function-on-Scalar**: Penalized OLS, FPC-based FOSR, 2D tensor-product FOSR, FANOVA with permutation tests
- **Other Regression**: FPCA, PLS, ridge; functional mixed model (FPCA + iterative GLS/REML)
- **Classification**: LDA, QDA, k-NN, kernel, DD-classifier with cross-validation; conformal prediction sets
- **Clustering**: K-means, fuzzy c-means, GMM (BIC/ICL); elastic k-means++ and hierarchical clustering

### Elastic Analysis

- **Alignment**: SRSF/DP alignment, Karcher mean (1-D/N-D), distance matrices, amplitude/phase decomposition, landmark registration, TSRVF, penalized alignment, lambda CV, peak persistence, multi-resolution, partial matching
- **Advanced Alignment**: Bayesian (pCN MCMC), closed curve rotation search, transfer alignment across populations, generative models (Gaussian amplitude/phase sampling), curve geodesics, horizontal FPNS
- **Robust & Diagnostic**: Karcher median (Weiszfeld), trimmed mean, SRVF outlier detection, elastic depth, shape confidence intervals, registration diagnostics, warp statistics, phase box plots
- **Shape Analysis**: Quotient space operations, orbit representatives, shape distances, shape means
- **Elastic Models**: Elastic FPCA (vertical/horizontal/joint), elastic regression/PCR/logistic, scalar-on-shape (ScoSh), changepoint detection, amplitude vs phase attribution

### Statistical Process Monitoring

- **Core Framework**: Hotelling T²/SPE, Phase I/II, multivariate FPCA, contribution diagnostics, automatic ncomp selection
- **Chart Types**: EWMA, MEWMA, CUSUM (Crosier's MCUSUM), adaptive EWMA (AMFEWMA), FRCC (covariate-adjusted), profile monitoring
- **Control Limits**: Parametric (chi-squared), empirical, bootstrap, KDE; ARL computation (Monte Carlo)
- **Advanced**: Partial-domain monitoring (BLUP/projection/zero-pad), phase-aware elastic SPM, iterative Phase I, Western Electric and Nelson rules

### Explainability

- **Model-Agnostic**: PDP/ICE, SHAP (exact linear + Kernel), ALE, LIME, permutation importance, Sobol indices, Friedman H-statistic, anchor explanations, counterfactual explanations
- **Diagnostics**: Bootstrap CIs for β(t), influence (Cook's, leverage, DFBETAS/DFFITS), VIF, LOO-CV/PRESS, regression depth, stability analysis
- **Calibration**: Brier score, log loss, Hosmer-Lemeshow, ECE/MCE/ACE, conformal prediction intervals
- **Other**: Functional saliency maps, domain selection, prototype/criticism (MMD), prediction intervals; unified `FpcPredictor` trait

### Inference

- **Tolerance Bands**: FPCA, conformal, Degras SCB, exponential family, elastic amplitude/phase bands
- **Conformal Prediction**: Split-conformal, Jackknife+, CV+, generic conformal with held-out calibration
- **Equivalence Testing**: Functional TOST (bootstrap, one/two-sample)

### Other

- **Seasonal Analysis**: FFT, ACF, Autoperiod, CFDAutoperiod, SAZED; Lomb-Scargle, matrix profile, SSA, peak detection, seasonal strength, amplitude modulation, change detection
- **Detrending**: Linear, polynomial, LOESS, differencing; classical, STL decomposition
- **Irregular Data**: CSR-compressed storage, kernel mean/covariance, Lp metric, grid regularization

## Installation

### R (fdars)

```r
install.packages("fdars")

# Development version from GitHub (requires Rust toolchain)
devtools::install_github("sipemu/fdars-r")
```

### Rust (fdars-core)

```toml
[dependencies]
fdars-core = "0.9"
```

Or install from the repository:

```toml
[dependencies]
fdars-core = { git = "https://github.com/sipemu/fdars" }
```

## Feature Flags

- `parallel` (default): Enable rayon-based parallel processing
- `linalg`: Enable linear algebra features (faer, ridge regression) — requires Rust 1.84+
- `js`: Enable WASM support with JS random number generation

For WASM builds, disable default features:

```toml
[dependencies]
fdars-core = { version = "0.9", default-features = false }
```

## Data Layout

Functional data is represented using the [`FdMatrix`](https://docs.rs/fdars-core/latest/fdars_core/matrix/struct.FdMatrix.html) type, a column-major matrix wrapping a flat `Vec<f64>` with safe `(i, j)` indexing and dimension tracking:
- For n observations with m evaluation points: `data[(i, j)]` gives observation i at point j
- Zero-copy column access via `data.column(j)`, row gather via `data.row(i)`
- nalgebra interop via `to_dmatrix()` / `from_dmatrix()` for SVD operations
- 2D surfaces (n observations, m1 x m2 grid): stored as n x (m1*m2) matrices

## Quick Start

```rust
use fdars_core::{FdMatrix, fdata, depth};

// Create sample functional data (3 observations, 10 points each)
let n = 3;
let m = 10;
let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64).sin()).collect();
let mat = FdMatrix::from_column_major(data, n, m).unwrap();
let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

// Compute mean function
let mean = fdata::mean_1d(&mat);

// Compute Fraiman-Muniz depth
let depths = depth::fraiman_muniz_1d(&mat, &mat, true);
```

## Examples

The [`fdars-core/examples/`](fdars-core/examples/) directory contains 26 runnable examples progressing from basic to advanced:

| # | Example | Command | Topics |
|---|---------|---------|--------|
| 01 | [Simulation](fdars-core/examples/01_simulation/) | `cargo run -p fdars-core --example simulation` | KL expansion, eigenfunctions, noise |
| 02 | [Functional Operations](fdars-core/examples/02_functional_operations/) | `cargo run -p fdars-core --example functional_operations` | Mean, centering, derivatives, norms, inner products |
| 03 | [Smoothing](fdars-core/examples/03_smoothing/) | `cargo run -p fdars-core --example smoothing` | Nadaraya-Watson, local linear/polynomial, k-NN |
| 04 | [Basis Representation](fdars-core/examples/04_basis_representation/) | `cargo run -p fdars-core --example basis_representation` | B-splines, Fourier, P-splines, GCV/AIC/BIC |
| 05 | [Depth Measures](fdars-core/examples/05_depth_measures/) | `cargo run -p fdars-core --example depth_measures` | 8 depth measures, outlier ranking |
| 06 | [Distances and Metrics](fdars-core/examples/06_distances_and_metrics/) | `cargo run -p fdars-core --example distances_and_metrics` | Lp, Hausdorff, DTW, Fourier, h-shift |
| 07 | [Clustering](fdars-core/examples/07_clustering/) | `cargo run -p fdars-core --example clustering` | K-means, fuzzy c-means, silhouette, CH index |
| 08 | [Regression](fdars-core/examples/08_regression/) | `cargo run -p fdars-core --example regression` | FPCA, PLS regression |
| 09 | [Outlier Detection](fdars-core/examples/09_outlier_detection/) | `cargo run -p fdars-core --example outlier_detection` | LRT bootstrap, depth confirmation |
| 10 | [Seasonal Analysis](fdars-core/examples/10_seasonal_analysis/) | `cargo run -p fdars-core --example seasonal_analysis` | FFT, ACF, Autoperiod, SAZED, peak detection |
| 11 | [Detrending](fdars-core/examples/11_detrending/) | `cargo run -p fdars-core --example detrending` | Linear/polynomial/LOESS, STL decomposition |
| 12 | [Streaming Depth](fdars-core/examples/12_streaming_depth/) | `cargo run -p fdars-core --example streaming_depth` | Online depth, rolling windows |
| 13 | [Irregular Data](fdars-core/examples/13_irregular_data/) | `cargo run -p fdars-core --example irregular_data` | CSR storage, regularization, kernel mean |
| 14 | [Complete Pipeline](fdars-core/examples/14_complete_pipeline/) | `cargo run -p fdars-core --example complete_pipeline` | End-to-end: simulate → smooth → outliers → FPCA → cluster |
| 15 | [Tolerance Bands](fdars-core/examples/15_tolerance_bands/) | `cargo run -p fdars-core --example tolerance_bands` | FPCA, conformal, Degras SCB, exponential family bands |
| 16 | [Elastic Alignment](fdars-core/examples/16_elastic_alignment/) | `cargo run -p fdars-core --example elastic_alignment` | SRSF, DP alignment, Karcher mean, elastic distances |
| 17 | [Equivalence Test](fdars-core/examples/17_equivalence_test/) | `cargo run -p fdars-core --example equivalence_test` | Functional TOST, bootstrap, one/two-sample tests |
| 18 | [Landmark Registration](fdars-core/examples/18_landmark_registration/) | `cargo run -p fdars-core --example landmark_registration` | Landmark detection, curve registration |
| 19 | [TSRVF](fdars-core/examples/19_tsrvf/) | `cargo run -p fdars-core --example tsrvf` | Transported SRVF, parallel transport |
| 20 | [Scalar-on-Function](fdars-core/examples/20_scalar_on_function/) | `cargo run -p fdars-core --features linalg --example scalar_on_function` | FPC linear model, kernel regression, logistic, CV |
| 21 | [Function-on-Scalar](fdars-core/examples/21_function_on_scalar/) | `cargo run -p fdars-core --features linalg --example function_on_scalar` | Penalized FOSR, FPC-based FOSR, FANOVA |
| 22 | [GMM Clustering](fdars-core/examples/22_gmm_clustering/) | `cargo run -p fdars-core --features linalg --example gmm_clustering` | GMM-EM, automatic K selection, BIC/ICL |
| 23 | [Classification](fdars-core/examples/23_classification/) | `cargo run -p fdars-core --features linalg --example classification` | LDA, QDA, k-NN, DD-classifier, cross-validation |
| 24 | [Mixed Effects](fdars-core/examples/24_mixed_effects/) | `cargo run -p fdars-core --features linalg --example mixed_effects` | FAMM, REML variance estimation, permutation tests |
| 25 | [Explainability](fdars-core/examples/25_explainability/) | `cargo run -p fdars-core --features linalg --example explainability` | Bootstrap CI, SHAP, ALE, PDP, VIF, DFBETAS, ECE, conformal, anchors |
| 26 | [Elastic Analysis](fdars-core/examples/26_elastic_analysis/) | `cargo run -p fdars-core --features linalg --example elastic_analysis` | Elastic FPCA, regression, PCR, logistic, changepoint detection |

## Performance

With the `parallel` feature (enabled by default), computationally intensive operations use `rayon` for multi-core performance. The library also supports WASM targets with sequential execution.

## Documentation

- **R Package**: [https://sipemu.github.io/fdars/](https://sipemu.github.io/fdars/)
- **Rust Crate**: [https://docs.rs/fdars-core](https://docs.rs/fdars-core)

## License

MIT
