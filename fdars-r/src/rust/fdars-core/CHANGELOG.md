# Changelog

All notable changes to fdars-core will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-05

### Added

- **Seasonal Analysis Methods**
  - `autoperiod()`: Hybrid FFT + ACF period detection with gradient ascent refinement (Vlachos et al. 2005)
  - `cfd_autoperiod()`: Differencing-based detrending with density clustering (Puech et al. 2020)
  - `sazed()`: Parameter-free ensemble period detection combining 5 methods
  - All methods return confidence scores for detected periods

- **WASM Compatibility**
  - Conditional compilation for `wasm32-unknown-unknown` target
  - New feature flags: `parallel`, `linalg`, `js`
  - Sequential fallbacks when `parallel` feature is disabled
  - CI workflow for WASM build verification

### Changed

- `rayon` dependency is now optional (enabled by `parallel` feature, on by default)
- `faer` and `anofox-regression` dependencies are now optional (enabled by `linalg` feature)
- Parallel iteration now uses macros (`iter_maybe_parallel!`, `slice_maybe_parallel!`) for conditional compilation

### Fixed

- SAZED algorithm optimized to reduce false positive rate from 64% to 3%

## [0.2.0] - 2025-01-03

### Changed

- Improved test coverage to 84%+
- Added pre-commit hooks for cargo fmt and clippy
- Refactored seasonal analysis code to remove duplication

## [0.1.0] - 2024-12-01

### Added

- Initial release
- Functional data operations (mean, centering, derivatives, Lp norms, geometric median)
- Depth measures (Fraiman-Muniz, modal, band, modified band, random projection, etc.)
- Distance metrics (Lp, Hausdorff, DTW, Fourier semimetric)
- Basis representations (B-splines, Fourier, P-splines)
- Clustering (K-means, fuzzy c-means)
- Smoothing (Nadaraya-Watson, local linear, local polynomial, k-NN)
- Regression (functional PCA, PLS, ridge)
- Outlier detection (LRT-based)
- Seasonal analysis (FFT, ACF period detection, seasonal strength)
