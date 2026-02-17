# Functional Data Analysis (FDA)

[![Rust CI](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml/badge.svg)](https://github.com/sipemu/fdars/actions/workflows/rust-ci.yml)
[![Crates.io](https://img.shields.io/crates/v/fdars-core.svg)](https://crates.io/crates/fdars-core)
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
- **Seasonal Analysis**: Period estimation, peak detection, seasonal decomposition

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

## Documentation

- **R Package**: [https://sipemu.github.io/fdars/](https://sipemu.github.io/fdars/)
- **Rust Crate**: [https://docs.rs/fdars-core](https://docs.rs/fdars-core)

## License

MIT
