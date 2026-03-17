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

| Area | Capabilities |
|------|-------------|
| **Core** | Simulation (KL expansion, GP with 8 kernels), functional operations, smoothing (NW, local polynomial, k-NN), basis representations (B-spline, Fourier, P-spline) |
| **Descriptive** | 10 depth measures + streaming online depth, 12 distance metrics (Lp, DTW, elastic, semimetrics, KL), LRT outlier detection |
| **Regression** | Scalar-on-function (FPC, kernel, logistic, robust), function-on-scalar (FOSR, 2D FOSR, FANOVA), FPCA, PLS, ridge, mixed effects |
| **Classification** | LDA, QDA, k-NN, kernel, DD-classifier, conformal prediction sets; k-means, fuzzy c-means, GMM |
| **Elastic Alignment** | SRSF/DP alignment, Karcher mean (1-D/N-D), TSRVF, Bayesian (pCN MCMC), closed curves, transfer alignment, partial matching, multi-resolution, generative models, geodesics, FPNS, lambda CV, peak persistence |
| **Elastic Robust** | Karcher median, trimmed mean, SRVF outlier detection, elastic depth, shape CIs, diagnostics, warp statistics, phase box plots, shape analysis |
| **Elastic Models** | Elastic FPCA, regression, PCR, logistic, scalar-on-shape (ScoSh), changepoint detection, elastic clustering |
| **SPM** | T²/SPE Phase I/II, EWMA, MEWMA, CUSUM, adaptive EWMA, FRCC, profile monitoring; bootstrap/KDE limits, ARL, partial-domain, elastic SPM, iterative Phase I, Western Electric/Nelson rules |
| **Explainability** | PDP/ICE, SHAP, ALE, LIME, Sobol, Friedman H, anchors, counterfactuals, prototype/criticism; influence diagnostics, VIF, calibration (ECE, Brier), saliency maps; `FpcPredictor` trait |
| **Inference** | Tolerance bands (FPCA, conformal, Degras, exponential, elastic), conformal prediction (split, Jackknife+, CV+), equivalence testing (TOST) |
| **Time Series** | Seasonal detection (FFT, ACF, Autoperiod, SAZED, Lomb-Scargle, SSA, matrix profile), detrending (polynomial, LOESS, STL) |
| **Specialized** | Streaming depth (online O(log N)), irregular data (CSR, kernel estimation) |

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

27 runnable examples in [`fdars-core/examples/`](fdars-core/examples/):

| # | Example | Topics |
|---|---------|--------|
| 01 | [Simulation](fdars-core/examples/01_simulation/) | KL expansion, GP generation |
| 02 | [Functional Operations](fdars-core/examples/02_functional_operations/) | Mean, derivatives, norms |
| 03 | [Smoothing](fdars-core/examples/03_smoothing/) | NW, local polynomial, k-NN |
| 04 | [Basis Representation](fdars-core/examples/04_basis_representation/) | B-splines, Fourier, P-splines |
| 05 | [Depth Measures](fdars-core/examples/05_depth_measures/) | 8 depth measures, outlier ranking |
| 06 | [Distances](fdars-core/examples/06_distances_and_metrics/) | Lp, DTW, elastic, semimetrics |
| 07 | [Clustering](fdars-core/examples/07_clustering/) | K-means, fuzzy c-means |
| 08 | [Regression](fdars-core/examples/08_regression/) | FPCA, PLS |
| 09 | [Outlier Detection](fdars-core/examples/09_outlier_detection/) | LRT bootstrap |
| 10 | [Seasonal Analysis](fdars-core/examples/10_seasonal_analysis/) | FFT, Autoperiod, SAZED |
| 11 | [Detrending](fdars-core/examples/11_detrending/) | Polynomial, LOESS, STL |
| 12 | [Streaming Depth](fdars-core/examples/12_streaming_depth/) | Online depth |
| 13 | [Irregular Data](fdars-core/examples/13_irregular_data/) | CSR storage, kernel estimation |
| 14 | [Complete Pipeline](fdars-core/examples/14_complete_pipeline/) | End-to-end workflow |
| 15 | [Tolerance Bands](fdars-core/examples/15_tolerance_bands/) | FPCA, conformal, Degras SCB |
| 16 | [Elastic Alignment](fdars-core/examples/16_elastic_alignment/) | SRSF, DP, Karcher mean |
| 17 | [Equivalence Test](fdars-core/examples/17_equivalence_test/) | Functional TOST |
| 18 | [Landmark Registration](fdars-core/examples/18_landmark_registration/) | Constrained alignment |
| 19 | [TSRVF](fdars-core/examples/19_tsrvf/) | Transported SRVF |
| 20 | [Scalar-on-Function](fdars-core/examples/20_scalar_on_function/)` *` | FPC linear, logistic, kernel |
| 21 | [Function-on-Scalar](fdars-core/examples/21_function_on_scalar/)` *` | FOSR, FANOVA |
| 22 | [GMM Clustering](fdars-core/examples/22_gmm_clustering/)` *` | GMM-EM, BIC/ICL |
| 23 | [Classification](fdars-core/examples/23_classification/)` *` | LDA, QDA, k-NN, DD |
| 24 | [Mixed Effects](fdars-core/examples/24_mixed_effects/)` *` | FAMM, REML |
| 25 | [Explainability](fdars-core/examples/25_explainability/)` *` | SHAP, ALE, PDP, anchors |
| 26 | [Elastic Analysis](fdars-core/examples/26_elastic_analysis/)` *` | Elastic FPCA, regression, PCR |
| 27 | [SPM](fdars-core/examples/27_spm/) | Phase I/II, EWMA, CUSUM, rules |

`*` requires `--features linalg`

Run with `cargo run -p fdars-core --example <name>` (add `--features linalg` where marked).

## Performance

With the `parallel` feature (enabled by default), computationally intensive operations use `rayon` for multi-core performance. The library also supports WASM targets with sequential execution.

## Documentation

- **R Package**: [https://sipemu.github.io/fdars/](https://sipemu.github.io/fdars/)
- **Rust Crate**: [https://docs.rs/fdars-core](https://docs.rs/fdars-core)

## License

MIT
