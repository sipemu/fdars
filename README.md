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

### Core Operations

- **Simulation**: Karhunen-Loève expansion with Fourier/Legendre/Wiener eigenfunctions, pointwise and curve-level noise; Gaussian process generation with 8 covariance kernels and kernel algebra
- **Functional Data Operations**: Mean, centering, derivatives, Lp norms, geometric median
- **Andrews Curves**: Fourier-based bijection mapping multivariate observations to functional curves; FPCA loading visualization
- **Covariance Kernels**: Gaussian, Exponential, Matérn, Brownian, Periodic, Linear, Polynomial, White Noise with Sum/Product kernel algebra
- **Smoothing**: Nadaraya-Watson, local linear, local polynomial, k-NN
- **Basis Representations**: B-splines, Fourier basis, P-splines with GCV/AIC/BIC selection

### Descriptive Analysis

- **Depth Measures**: Fraiman-Muniz, modal, band, modified band, random projection, random Tukey, RPD (random projection with derivatives), functional spatial, kernel functional spatial, modified epigraph index
- **Distance Metrics**: Lp distances, Hausdorff, DTW, Soft-DTW (with barycenter averaging), elastic (Fisher-Rao), amplitude/phase distances, Fourier-based semimetric, horizontal shift semimetric, PCA-based semimetric, derivative-based semimetric, basis coefficient semimetric, KL divergence
- **Outlier Detection**: LRT-based outlier detection with bootstrap thresholding

### Regression

- **Scalar-on-Function Regression**: FPC linear model, nonparametric kernel, functional logistic, robust (L1/Huber), CV component selection
- **Function-on-Scalar Regression**: Penalized pointwise OLS, FPC-based FOSR, 2D FOSR for surface-valued responses Y(s,t) with tensor-product penalty, functional ANOVA with permutation test
- **Regression**: Functional PCA, PLS, ridge regression
- **Mixed Effects Models**: Functional mixed model via FPCA + iterative GLS/REML, prediction, permutation hypothesis tests

### Classification & Clustering

- **Clustering**: K-means, fuzzy c-means with silhouette and Calinski-Harabasz validation; GMM with BIC/ICL model selection
- **Classification**: LDA, QDA, k-NN, kernel, DD-classifier with cross-validation; conformal prediction sets for classification

### Time Series & Alignment

- **Seasonal Analysis**: FFT, ACF, Autoperiod, CFDAutoperiod, SAZED period detection; Lomb-Scargle periodogram; matrix profile; SSA; peak detection; seasonal strength metrics; amplitude modulation detection; seasonality change detection
- **Elastic Alignment**: SRSF transform, dynamic programming alignment, Karcher mean, elastic distance matrices, amplitude/phase decomposition, landmark registration, transported SRVF (TSRVF)
- **Detrending**: Linear, polynomial, LOESS, differencing; classical, additive/multiplicative, and STL decomposition

### Explainability & Diagnostics

- **Bootstrap Confidence Intervals**: Pointwise and simultaneous CIs for β(t) via bootstrap
- **Beta Decomposition**: FPC-level variance proportions and coefficient attribution
- **PDP/ICE Curves**: Partial dependence and individual conditional expectation for FPC scores
- **Permutation Importance**: FPC-level importance via R² drop (linear and logistic)
- **Pointwise Variable Importance**: Per-grid-point contribution to prediction variance
- **Influence Diagnostics**: Cook's distance, leverage, studentized residuals
- **DFBETAS/DFFITS**: Leave-one-out influence measures with cutoff thresholds
- **VIF**: Variance inflation factors for multicollinearity detection
- **SHAP Values**: Exact linear SHAP and Kernel SHAP for logistic models
- **Prediction Intervals**: Confidence and prediction intervals with Cornish-Fisher correction
- **ALE Plots**: Accumulated local effects for FPC scores (linear and logistic)
- **Friedman H-statistic**: Pairwise FPC interaction strength
- **LOO-CV / PRESS**: Leave-one-out cross-validation diagnostics
- **Sobol Indices**: Global sensitivity analysis for FPC contributions
- **Calibration Diagnostics**: Brier score, log loss, Hosmer-Lemeshow test
- **Functional Saliency Maps**: Pointwise gradient-based importance
- **Domain Selection**: Interval importance for regression/classification
- **Conditional Permutation Importance**: Correlation-adjusted permutation importance
- **Counterfactual Explanations**: Minimal FPC score changes to reach a target prediction
- **Prototype/Criticism Selection**: MMD-based representative and outlier observations
- **LIME**: Local interpretable model-agnostic explanations in FPC score space
- **Expected Calibration Error**: ECE, MCE, and adaptive calibration error
- **Conformal Prediction**: Distribution-free split-conformal prediction intervals
- **Regression Depth**: Depth-based diagnostics for coefficients and observations
- **Stability Analysis**: Bootstrap robustness of β(t), coefficients, and importance rankings
- **Anchor Explanations**: Beam-search rule extraction in FPC score space
- **Generic Explainability**: `FpcPredictor` trait unifying regression, binary, and multiclass models with 15 model-agnostic functions (PDP, SHAP, ALE, LIME, permutation importance, Sobol, Friedman H, etc.)

### Elastic Analysis

- **Elastic FPCA**: Vertical, horizontal, and joint functional PCA after alignment
- **Elastic Regression**: Alignment-integrated scalar-on-function regression
- **Elastic PCR**: Principal component regression with elastic alignment
- **Elastic Logistic**: Binary classification with elastic alignment
- **Elastic Changepoint Detection**: Amplitude and phase changepoint tests with permutation
- **Elastic Attribution**: Amplitude vs phase importance decomposition

### Inference

- **Tolerance Bands**: FPCA, conformal prediction, Degras SCB, exponential family bands
- **Conformal Prediction**: Split-conformal regression intervals (`conformal_fregre_lm`, `conformal_fregre_np`, `conformal_elastic_regression`), Jackknife+ intervals, and CV+ intervals
- **Equivalence Testing**: Functional TOST with bootstrap, one-sample and two-sample tests

### Specialized

- **Streaming Depth**: Online FM, MBD, BD depth with sorted-reference O(log N) updates
- **Irregular Data**: CSR-compressed storage, kernel mean/covariance, Lp metric, grid regularization
- **Smooth Basis**: B-spline basis representation with smoothing penalty

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
fdars-core = "0.8"
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
fdars-core = { version = "0.8", default-features = false }
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
