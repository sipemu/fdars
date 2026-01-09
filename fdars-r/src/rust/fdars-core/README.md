# fdars-core

[![Rust CI](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fdars-core.svg)](https://crates.io/crates/fdars-core)
[![Documentation](https://docs.rs/fdars-core/badge.svg)](https://docs.rs/fdars-core)
[![codecov](https://codecov.io/gh/sipemu/fdars/branch/main/graph/badge.svg)](https://codecov.io/gh/sipemu/fdars)
[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

Pure Rust algorithms for Functional Data Analysis (FDA).

## Overview

`fdars-core` provides high-performance implementations of various FDA methods, designed to be used as a library in Rust projects or as the backend for R/Python bindings.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fdars-core = "0.3"
```

Or install from the repository:

```toml
[dependencies]
fdars-core = { git = "https://github.com/sipemu/fdars" }
```

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

## Feature Flags

- `parallel` (default): Enable rayon-based parallel processing
- `linalg` (default): Enable linear algebra features (faer, ridge regression)
- `js`: Enable WASM support with JS random number generation

For WASM builds, disable default features:

```toml
[dependencies]
fdars-core = { version = "0.2", default-features = false }
```

## Data Layout

Functional data is represented as column-major matrices stored in flat `Vec<f64>`:
- For n observations with m evaluation points: `data[i + j * n]` gives observation i at point j
- 2D surfaces (n observations, m1 x m2 grid): stored as n x (m1*m2) matrices

## Example

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

## Performance

With the `parallel` feature (enabled by default), computationally intensive operations use `rayon` for multi-core performance. The library also supports WASM targets with sequential execution.

## License

MIT
