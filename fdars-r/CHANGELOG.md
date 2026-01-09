# Changelog

All notable changes to fdars are documented in this file.

## [0.5.2] - 2025-01-09

### Changed
- **Breaking**: Renamed functions to avoid S3 method dispatch conflicts:
  - `fdata2basis.cv()` → `fdata2basis_cv()`
  - `fdata2basis.2d()` → `fdata2basis_2d()`
  - `basis2fdata.2d()` → `basis2fdata_2d()`

## [0.5.1] - 2024-12-16

### Added
- `normalize()` function to scale curves to unit Lp norm
- Vignette on basis representation and optimal basis selection

### Changed
- Renamed `norm.fdata()` to `norm()` for cleaner API
- Renamed covariance kernel functions from `cov.*` to `kernel_*` to avoid S3 dispatch conflicts

### Fixed
- Plot functions now display when called directly (`plot(fd)` shows plot, `p <- plot(fd)` does not)
- S3 method dispatch conflict with `cov()` generic function

## [0.5.0] - 2024-12-14

### Added
- **Basis representation module** with Rust backend:
  - `basis2fdata()` - reconstruct functional data from basis coefficients
  - `basis.gcv()`, `basis.aic()`, `basis.bic()` - goodness-of-fit metrics
  - `fdata2basis_cv()` - cross-validation for optimal nbasis selection
  - `pspline()` - P-spline smoothing with automatic lambda selection
  - `fdata2basis_2d()`, `basis2fdata_2d()` - 2D tensor product basis support
  - `pspline.2d()` - 2D P-spline with anisotropic penalties

### Changed
- Plot functions now return ggplot objects without auto-printing (use `print(p)` to display)
- Removed hardcoded `theme_minimal()` from all plots - respects `ggplot2::theme_set()`
- Fixed white lines in 2D surface plots when using facets

### Fixed
- Windows binary now built with R 4.2.2 for compatibility with older R versions

## [0.4.0] - 2024-12-13

### Added
- `id` and `metadata` slots in fdata objects for storing curve identifiers and associated data
- Outlier plot labeling: `plot(outliergram, label = "id")` or `label = "column_name"`
- `magnitudeshape()` labeling support (renamed from MS.plot)

### Changed
- Auto-reduce alpha when `show.mean = TRUE` in `plot.fdata()`

### Fixed
- Vignette error: extract data from mean fdata object correctly

## [0.3.4] - 2024-12-13

### Added
- `fregre.np.multi()` for regression with multiple functional predictors

### Fixed
- `plot.group.distance()` error handling
- Missing depth wrapper functions

### Changed
- Documented null hypothesis for `group.test()`

## [0.3.3] - 2024-12-12

### Added
- Enhanced `plot.fdata()` with group coloring, mean curves, and confidence intervals
- `group.distance()` for measuring distances between groups of curves
- `group.test()` permutation test for group differences

### Fixed
- Release workflow now generates documentation before building

## [0.3.2] - 2024-12-12

### Fixed
- `mean(fd)` now returns fdata object (was returning matrix)
- Missing `%||%` operator definition

## [0.3.1] - 2024-12-12

### Added
- `outliergram()` visualization (MEI vs MBD plot)
- `plot.fdata2pc()` for FPCA visualization (components, variance, scores)

### Changed
- Renamed `fdata.deriv()` to `deriv()` for consistency

## [0.3.0] - 2024-12-11

### Added
- Covariance kernel functions: `cov.Exponential()`, `cov.Matern()`, `cov.Gaussian()`, etc.
- `make_gaussian_process()` for simulating Gaussian process realizations
- 2D functional data support for most functions
- Unified API: `depth()`, `median()`, `trimmed()`, `trimvar()` with method parameter

### Changed
- Cleaned up API: removed backward compatibility shims
- Renamed functions for consistency (e.g., `fdata.mean` -> `mean.fdata`)
- All plots now use ggplot2 instead of base R graphics

### Added (from 0.2.x development)
- Band depth (`depth.BD`, `depth.MBD`, `depth.MEI`)
- Functional boxplot (`boxplot.fdata`)
- MS-plot for outlier detection
- Fuzzy c-means clustering (`cluster.fcm`)
- Geometric median (`gmed`)
- Curve registration (`register.fd`)
- Local averages feature extraction (`localavg.fdata`)
- Optimal k selection for k-means (`cluster.optim`)
- k-NN bandwidth selection for nonparametric regression

## [0.1.0] - 2024-12-10

### Initial Release
- **Core functional data structure**: `fdata()` for 1D and 2D functional data
- **Depth functions**: FM, mode, RP, RT, FSD, KFSD, RPD (all with Rust backend)
- **Statistics**: mean, variance, standard deviation, covariance
- **Distance metrics**: Lp, Hausdorff, DTW, KL divergence
- **Semimetrics**: basis projection, Fourier, horizontal shift, PCA, derivative
- **Regression**: `fregre.pc()`, `fregre.basis()`, `fregre.np()` with CV variants
- **Outlier detection**: `outliers.depth.pond()`, `outliers.depth.trim()`, `outliers.lrt()`
- **Smoothing**: Nadaraya-Watson, local linear, local polynomial, k-NN
- **Clustering**: functional k-means
- **Hypothesis testing**: `flm.test()`, `fmean.test.fdata()`
- **Utilities**: Simpson integration, inner product, derivatives
