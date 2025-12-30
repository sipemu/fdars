# Changelog

All notable changes to fdars are documented in this file.

## \[0.5.1\] - 2024-12-16

### Added

- [`normalize()`](https://sipemu.github.io/fdars/reference/normalize.md)
  function to scale curves to unit Lp norm
- Vignette on basis representation and optimal basis selection

### Changed

- Renamed
  [`norm.fdata()`](https://sipemu.github.io/fdars/reference/norm.md) to
  [`norm()`](https://sipemu.github.io/fdars/reference/norm.md) for
  cleaner API
- Renamed covariance kernel functions from `cov.*` to `kernel_*` to
  avoid S3 dispatch conflicts

### Fixed

- Plot functions now display when called directly (`plot(fd)` shows
  plot, `p <- plot(fd)` does not)
- S3 method dispatch conflict with
  [`cov()`](https://sipemu.github.io/fdars/reference/cov.md) generic
  function

## \[0.5.0\] - 2024-12-14

### Added

- **Basis representation module** with Rust backend:
  - [`basis2fdata()`](https://sipemu.github.io/fdars/reference/basis2fdata.md) -
    reconstruct functional data from basis coefficients
  - [`basis.gcv()`](https://sipemu.github.io/fdars/reference/basis.gcv.md),
    [`basis.aic()`](https://sipemu.github.io/fdars/reference/basis.aic.md),
    [`basis.bic()`](https://sipemu.github.io/fdars/reference/basis.bic.md) -
    goodness-of-fit metrics
  - [`fdata2basis.cv()`](https://sipemu.github.io/fdars/reference/fdata2basis.cv.md) -
    cross-validation for optimal nbasis selection
  - [`pspline()`](https://sipemu.github.io/fdars/reference/pspline.md) -
    P-spline smoothing with automatic lambda selection
  - [`fdata2basis.2d()`](https://sipemu.github.io/fdars/reference/fdata2basis.2d.md),
    [`basis2fdata.2d()`](https://sipemu.github.io/fdars/reference/basis2fdata.2d.md) -
    2D tensor product basis support
  - [`pspline.2d()`](https://sipemu.github.io/fdars/reference/pspline.2d.md) -
    2D P-spline with anisotropic penalties

### Changed

- Plot functions now return ggplot objects without auto-printing (use
  `print(p)` to display)
- Removed hardcoded `theme_minimal()` from all plots - respects
  [`ggplot2::theme_set()`](https://ggplot2.tidyverse.org/reference/get_theme.html)
- Fixed white lines in 2D surface plots when using facets

### Fixed

- Windows binary now built with R 4.2.2 for compatibility with older R
  versions

## \[0.4.0\] - 2024-12-13

### Added

- `id` and `metadata` slots in fdata objects for storing curve
  identifiers and associated data
- Outlier plot labeling: `plot(outliergram, label = "id")` or
  `label = "column_name"`
- [`magnitudeshape()`](https://sipemu.github.io/fdars/reference/magnitudeshape.md)
  labeling support (renamed from MS.plot)

### Changed

- Auto-reduce alpha when `show.mean = TRUE` in
  [`plot.fdata()`](https://sipemu.github.io/fdars/reference/plot.fdata.md)

### Fixed

- Vignette error: extract data from mean fdata object correctly

## \[0.3.4\] - 2024-12-13

### Added

- [`fregre.np.multi()`](https://sipemu.github.io/fdars/reference/fregre.np.multi.md)
  for regression with multiple functional predictors

### Fixed

- [`plot.group.distance()`](https://sipemu.github.io/fdars/reference/plot.group.distance.md)
  error handling
- Missing depth wrapper functions

### Changed

- Documented null hypothesis for
  [`group.test()`](https://sipemu.github.io/fdars/reference/group.test.md)

## \[0.3.3\] - 2024-12-12

### Added

- Enhanced
  [`plot.fdata()`](https://sipemu.github.io/fdars/reference/plot.fdata.md)
  with group coloring, mean curves, and confidence intervals
- [`group.distance()`](https://sipemu.github.io/fdars/reference/group.distance.md)
  for measuring distances between groups of curves
- [`group.test()`](https://sipemu.github.io/fdars/reference/group.test.md)
  permutation test for group differences

### Fixed

- Release workflow now generates documentation before building

## \[0.3.2\] - 2024-12-12

### Fixed

- `mean(fd)` now returns fdata object (was returning matrix)
- Missing `%||%` operator definition

## \[0.3.1\] - 2024-12-12

### Added

- [`outliergram()`](https://sipemu.github.io/fdars/reference/outliergram.md)
  visualization (MEI vs MBD plot)
- [`plot.fdata2pc()`](https://sipemu.github.io/fdars/reference/plot.fdata2pc.md)
  for FPCA visualization (components, variance, scores)

### Changed

- Renamed `fdata.deriv()` to
  [`deriv()`](https://sipemu.github.io/fdars/reference/deriv.md) for
  consistency

## \[0.3.0\] - 2024-12-11

### Added

- Covariance kernel functions: `cov.Exponential()`, `cov.Matern()`,
  `cov.Gaussian()`, etc.
- [`make_gaussian_process()`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)
  for simulating Gaussian process realizations
- 2D functional data support for most functions
- Unified API:
  [`depth()`](https://sipemu.github.io/fdars/reference/depth.md),
  [`median()`](https://sipemu.github.io/fdars/reference/median.md),
  [`trimmed()`](https://sipemu.github.io/fdars/reference/trimmed.md),
  [`trimvar()`](https://sipemu.github.io/fdars/reference/trimvar.md)
  with method parameter

### Changed

- Cleaned up API: removed backward compatibility shims
- Renamed functions for consistency (e.g., `fdata.mean` -\>
  `mean.fdata`)
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

## \[0.1.0\] - 2024-12-10

### Initial Release

- **Core functional data structure**:
  [`fdata()`](https://sipemu.github.io/fdars/reference/fdata.md) for 1D
  and 2D functional data
- **Depth functions**: FM, mode, RP, RT, FSD, KFSD, RPD (all with Rust
  backend)
- **Statistics**: mean, variance, standard deviation, covariance
- **Distance metrics**: Lp, Hausdorff, DTW, KL divergence
- **Semimetrics**: basis projection, Fourier, horizontal shift, PCA,
  derivative
- **Regression**:
  [`fregre.pc()`](https://sipemu.github.io/fdars/reference/fregre.pc.md),
  [`fregre.basis()`](https://sipemu.github.io/fdars/reference/fregre.basis.md),
  [`fregre.np()`](https://sipemu.github.io/fdars/reference/fregre.np.md)
  with CV variants
- **Outlier detection**:
  [`outliers.depth.pond()`](https://sipemu.github.io/fdars/reference/outliers.depth.pond.md),
  [`outliers.depth.trim()`](https://sipemu.github.io/fdars/reference/outliers.depth.trim.md),
  [`outliers.lrt()`](https://sipemu.github.io/fdars/reference/outliers.lrt.md)
- **Smoothing**: Nadaraya-Watson, local linear, local polynomial, k-NN
- **Clustering**: functional k-means
- **Hypothesis testing**:
  [`flm.test()`](https://sipemu.github.io/fdars/reference/flm.test.md),
  [`fmean.test.fdata()`](https://sipemu.github.io/fdars/reference/fmean.test.fdata.md)
- **Utilities**: Simpson integration, inner product, derivatives
