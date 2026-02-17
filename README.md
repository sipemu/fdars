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
| fdars | R | CRAN | [sipemu/fdars-r](https://github.com/sipemu/fdars-r) | [![CRAN status](https://www.r-pkg.org/badges/version/fdars)](https://CRAN.R-project.org/package=fdars) |
| fdars-core | Rust | crates.io | `fdars-core/` | [![Crates.io](https://img.shields.io/crates/v/fdars-core.svg)](https://crates.io/crates/fdars-core) |

## Features

- **Functional Data Operations**: Mean, centering, derivatives, Lp norms, geometric median
- **Depth Measures**: Fraiman-Muniz, modal, band, modified band, random projection, random Tukey, functional spatial, kernel functional spatial, modified epigraph index
- **Distance Metrics**: Lp distances, Hausdorff, DTW, Fourier-based semimetric, horizontal shift semimetric
- **Basis Representations**: B-splines, Fourier basis, P-splines with GCV/AIC/BIC selection
- **Clustering**: K-means, fuzzy c-means with silhouette and Calinski-Harabasz validation
- **Smoothing**: Nadaraya-Watson, local linear, local polynomial, k-NN
- **Regression**: Functional PCA, PLS, ridge regression
- **Outlier Detection**: LRT-based outlier detection with bootstrap thresholding
- **Seasonal Analysis**: FFT, ACF, Autoperiod, CFDAutoperiod, SAZED period detection; seasonal strength metrics; amplitude modulation detection

## Installation

### R (fdars)

```r
# From GitHub (requires Rust toolchain)
devtools::install_github("sipemu/fdars-r")

# From binary release (no Rust required)
# Download from GitHub Releases, then:
install.packages("path/to/fdars_x.y.z.tgz", repos = NULL, type = "mac.binary")  # macOS
install.packages("path/to/fdars_x.y.z.zip", repos = NULL, type = "win.binary")  # Windows
```

### Rust (fdars-core)

```toml
[dependencies]
fdars-core = "0.3"
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
fdars-core = { version = "0.3", default-features = false }
```

## Data Layout

Functional data is represented as column-major matrices stored in flat `Vec<f64>`:
- For n observations with m evaluation points: `data[i + j * n]` gives observation i at point j
- 2D surfaces (n observations, m1 x m2 grid): stored as n x (m1*m2) matrices

## Quick Start

```rust
use fdars_core::{fdata, depth, helpers};

// Create sample functional data (3 observations, 10 points each)
let n = 3;
let m = 10;
let data: Vec<f64> = (0..(n * m)).map(|i| (i as f64).sin()).collect();
let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

// Compute mean function
let mean = fdata::mean_1d(&data, n, m);

// Compute Fraiman-Muniz depth
let depths = depth::fraiman_muniz_1d(&data, &data, n, n, m, true);
```

## Examples

The [`fdars-core/examples/`](fdars-core/examples/) directory contains 14 runnable examples progressing from basic to advanced:

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

## Performance

With the `parallel` feature (enabled by default), computationally intensive operations use `rayon` for multi-core performance. The library also supports WASM targets with sequential execution.

## Documentation

- **R Package**: [https://sipemu.github.io/fdars/](https://sipemu.github.io/fdars/)
- **Rust Crate**: [https://docs.rs/fdars-core](https://docs.rs/fdars-core)

## License

MIT
