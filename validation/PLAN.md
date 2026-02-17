# Validation Plan: fdars-core against R and Python Reference Implementations

## Objective

Validate every public algorithm in fdars-core against established R and Python
packages. For each function we produce a test case that:

1. Generates or loads identical input data in both languages
2. Runs the fdars-core function (via Rust test harness or R bindings)
3. Runs the reference R/Python function
4. Compares outputs within documented tolerance

## Current State

- **Unit tests**: 331 tests, ~84% code coverage (correctness checks, edge cases)
- **Cross-validation**: None. No comparison against external implementations.
- **R bindings**: Separate repo ([sipemu/fdars-r](https://github.com/sipemu/fdars-r))

## Approach

### Data Exchange Format

Use JSON files as the interchange format:
- `validation/data/` -- shared input datasets (JSON)
- `validation/R/` -- R scripts that read inputs, run reference functions, write expected outputs
- `validation/expected/` -- reference outputs from R/Python (JSON)
- `validation/rust/` -- Rust integration tests that read inputs + expected outputs and compare

Each validation case follows the pattern:

```
validation/
  data/{module}_{case}.json          # input data
  R/{module}_validation.R            # R script generating expected/
  expected/{module}_{case}.json      # reference output
  rust/tests/{module}_validation.rs  # Rust test comparing against expected/
```

### Tolerance Levels

| Category | Default tolerance | Notes |
|----------|------------------|-------|
| Basis values | 1e-10 | Exact linear algebra |
| Depth values | 1e-6 | Numerical integration differences |
| Distances/metrics | 1e-8 | Floating point accumulation |
| Smoothing | 1e-4 | Kernel/bandwidth implementation differences |
| Decomposition | 1e-3 | Iterative algorithms (STL, LOESS) |
| Period detection | 5% relative | Discrete frequency resolution |
| Clustering labels | exact match | After canonical relabeling |
| Clustering centers | 1e-4 | Initialization-dependent |

### Priority Tiers

**Tier 1 (High)** -- Core FDA primitives used everywhere:
- Depth functions, basis representations, Lp metrics, FPCA

**Tier 2 (Medium)** -- Statistical methods with well-known references:
- Smoothing, clustering, regression, STL, outlier detection

**Tier 3 (Lower)** -- Novel/ensemble methods with fewer direct references:
- SAZED, Autoperiod, CFDAutoperiod, amplitude modulation, matrix profile

---

## Module-by-Module Validation

### 1. Depth (`depth.rs`) -- Tier 1

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `fraiman_muniz_1d` | `fda.usc::depth.FM(fdataobj, fdataori)` | `skfda.exploratory.depth.IntegratedDepth()` | High |
| `fraiman_muniz_2d` | `fda.usc::depth.FM()` | `skfda.exploratory.depth.IntegratedDepth()` | High |
| `band_1d` | `roahd::BD(Data)` | `skfda.exploratory.depth.BandDepth()` | High |
| `modified_band_1d` | `roahd::MBD(Data)` | `skfda.exploratory.depth.ModifiedBandDepth()` | High |
| `modal_1d` | `fda.usc::depth.mode(fdataobj, fdataori, h)` | -- | High |
| `modal_2d` | `fda.usc::depth.mode()` | -- | Medium |
| `random_projection_1d` | `ddalpha::depthf.RP1()$Half_FD` | -- | High |
| `random_projection_2d` | `ddalpha::depthf.RP2()` | -- | Medium |
| `random_tukey_1d` | `fda.usc::depth.RT(fdataobj, fdataori, nproj)` | -- | High |
| `random_tukey_2d` | `fda.usc::depth.RT()` | -- | Medium |
| `functional_spatial_1d` | `fda.usc::depth.FSD(fdataobj, fdataori)` | -- | High |
| `functional_spatial_2d` | `fda.usc::depth.FSD()` | -- | Medium |
| `kernel_functional_spatial_1d` | `fda.usc::depth.KFSD(fdataobj, fdataori, h)` | -- | Medium |
| `kernel_functional_spatial_2d` | `fda.usc::depth.KFSD()` | -- | Medium |
| `modified_epigraph_index_1d` | `roahd::MEI(Data)` | -- | High |

**Test data**: Generate 50 curves on 101 grid points using `roahd::generate_gauss_fdata()`.
Compute all depth measures in R, export as JSON, compare in Rust.

**Notes**:
- Random projection/Tukey depths depend on random projections. Fix seeds or
  compare distributional properties (rank correlation > 0.95 across methods).
- `ddalpha::depthf.RP1()` returns three components: `$Simpl_FD`, `$Half_FD`,
  `$RHalf_FD`. Our `random_projection_1d` corresponds to `$Half_FD`.

---

### 2. Basis Representations (`basis.rs`) -- Tier 1

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `bspline_basis` | `fda::eval.basis(t, create.bspline.basis(rangeval, nbasis, norder))` | `skfda.representation.basis.BSplineBasis()` | High |
| `fourier_basis` | `fda::eval.basis(t, create.fourier.basis(rangeval, nbasis, period))` | `skfda.representation.basis.FourierBasis()` | High |
| `fourier_basis_with_period` | `fda::eval.basis(t, create.fourier.basis(rangeval, nbasis, period))` | `skfda.representation.basis.FourierBasis()` | High |
| `difference_matrix` | `base::diff(diag(n), differences=order)` | `numpy.diff(numpy.eye(n), n=order, axis=0)` | High |
| `fdata_to_basis_1d` | `fda::smooth.basis(argvals, y, fdParobj)$fd$coefs` | `skfda.preprocessing.smoothing.BasisSmoother()` | High |
| `basis_to_fdata_1d` | `fda::eval.fd(argvals, fdobj)` | -- | High |
| `pspline_fit_1d` | `fda::smooth.basis(argvals, y, fdPar(basisobj, 2, lambda))` | -- | High |
| `fourier_fit_1d` | `fda::smooth.basis(argvals, y, fdPar(create.fourier.basis()))` | -- | Medium |
| `select_fourier_nbasis_gcv` | `fda::smooth.basis()` grid search over nbasis, compare `$gcv` | -- | Low |
| `select_basis_auto_1d` | Manual comparison of B-spline vs Fourier GCV scores | -- | Low |

**Test data**: Evaluate B-spline and Fourier bases on a regular grid of 101 points.
For smoothing, use a known smooth function (e.g., `sin(2*pi*x) + 0.1*noise`).

**Key comparison**:
- Basis matrix values should match to 1e-10 (same mathematical definition).
- P-spline coefficients may differ slightly if penalty matrix construction differs.
  Compare fitted values (not coefficients) to 1e-6.

---

### 3. Metrics (`metric.rs`) -- Tier 1

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `lp_self_1d` / `lp_cross_1d` | `fda.usc::metric.lp(fdata1, fdata2, lp=2)` | `skfda.misc.metrics.LpDistance(p=2)` | High |
| `lp_self_2d` / `lp_cross_2d` | `fda.usc::metric.lp()` | `skfda.misc.metrics.LpDistance(p=2)` | Medium |
| `hausdorff_self_1d` / `hausdorff_cross_1d` | `pracma::hausdorff_dist(A, B)` | `scipy.spatial.distance.directed_hausdorff(u, v)` | Medium |
| `dtw_self_1d` / `dtw_cross_1d` | `dtw::dtw(x, y, window.size=w)$distance` | `dtaidistance.dtw.distance(s1, s2)` | High |
| `fourier_self_1d` / `fourier_cross_1d` | `fda.usc::semimetric.fourier(fdata1, fdata2, nbasis)` | -- | Medium |
| `hshift_self_1d` / `hshift_cross_1d` | `fda.usc::semimetric.hshift(fdata1, fdata2)` | -- | Medium |

**Test data**: Reuse the 50-curve dataset from depth validation.

**Notes**:
- `fda.usc::metric.lp` uses trapezoidal integration; fdars uses Simpson's rule.
  Small numerical differences expected -- compare to 1e-6.
- DTW: match Sakoe-Chiba window constraint. Compare to 1e-10.
- Hausdorff: `pracma::hausdorff_dist` operates on point sets; fdars treats each
  curve as a point set in (t, y) space. Ensure identical curve representation.

---

### 4. Smoothing (`smoothing.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `nadaraya_watson` | `KernSmooth::locpoly(x, y, bandwidth=h, degree=0)` | `statsmodels.nonparametric.kernel_regression.KernelReg(reg_type='lc')` | Medium |
| `local_linear` | `KernSmooth::locpoly(x, y, bandwidth=h, degree=1)` | `statsmodels.nonparametric.kernel_regression.KernelReg(reg_type='ll')` | Medium |
| `local_polynomial` | `locpol::locpol(y ~ x, bw=h, deg=d)` | -- | Medium |
| `knn_smoother` | `FNN::knn.reg(train=x, test=x_new, y=y, k=k)$pred` | `sklearn.neighbors.KNeighborsRegressor(n_neighbors=k)` | Medium |
| `smoothing_matrix_nw` | `fda.usc::S.NW(tt=x, h=bandwidth)` | -- | Low |

**Test data**: Noisy sine on 201 points. Compare fitted values.

**Notes**:
- Kernel implementations differ (boundary handling, normalization). Use interior
  points only for comparison. Tolerance: 1e-4.
- Ensure identical kernel function (Gaussian, Epanechnikov) and bandwidth.

---

### 5. Clustering (`clustering.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `kmeans_fd` | `fda.usc::kmeans.fd(fdataobj, ncl=k)` | `skfda.ml.clustering.KMeans(n_clusters=k)` | High |
| `fuzzy_cmeans_fd` | `e1071::cmeans(x, centers=k, m=fuzziness)` | `skfuzzy.cluster.cmeans(data, c=k, m=fuzziness)` | Medium |
| `silhouette_score` | `cluster::silhouette(cluster, dist)` | `sklearn.metrics.silhouette_samples(X, labels)` | Medium |
| `calinski_harabasz` | `fpc::calinhara(x, clustering)` | `sklearn.metrics.calinski_harabasz_score(X, labels)` | Medium |

**Test data**: 3 well-separated clusters of 20 curves each (Gaussian functional data
with different mean functions).

**Notes**:
- K-means is initialization-dependent. Use identical initial centers or compare
  final within-cluster SS rather than exact labels.
- For fuzzy c-means, compare membership matrices after column permutation alignment.
- Silhouette and CH indices should match to 1e-6 given identical labels and distances.

---

### 6. Regression (`regression.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `fdata_to_pc_1d` | `fda::pca.fd(fdobj, nharm=ncomp)` | `skfda.preprocessing.dim_reduction.FPCA(n_components)` | High |
| `fdata_to_pls_1d` | `pls::plsr(y ~ X, ncomp, method="oscorespls")` | `sklearn.cross_decomposition.PLSRegression(n_components)` | Medium |
| `ridge_regression_fit` | `glmnet::glmnet(x, y, alpha=0, lambda)` | `sklearn.linear_model.Ridge(alpha=lambda)` | Medium |

**Test data**: 30 curves on 51 grid points with a scalar response.

**Notes**:
- FPCA: eigenvalues should match to 1e-8. Eigenvectors may have sign flips --
  compare absolute values or align signs.
- PLS: NIPALS implementations vary in convergence. Compare scores to 1e-4.
- Ridge: compare coefficients to 1e-8 (closed-form solution).

---

### 7. Outlier Detection (`outliers.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `outliers_threshold_lrt` | `fda.usc::outliers.depth.trim(fdataobj, nb=nb, smo=smo, trim=trim)` | -- | Medium |
| `detect_outliers_lrt` | `fda.usc::outliers.depth.trim()$outliers` | -- | Medium |

**Test data**: 50 curves with 3 known outliers (shifted by 3 standard deviations).

**Notes**:
- Bootstrap thresholds are stochastic. Fix seeds or compare that the same
  outliers are detected across multiple runs (consensus).
- The depth function used for outlier detection must match (Fraiman-Muniz).

---

### 8. Detrending & Decomposition (`detrend.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `detrend_linear` | `stats::lm(y ~ t)` then `residuals()` | `scipy.signal.detrend(data, type='linear')` | Medium |
| `detrend_polynomial` | `stats::lm(y ~ poly(t, degree))` then `residuals()` | -- | Medium |
| `detrend_diff` | `base::diff(x, differences=order)` | `numpy.diff(x, n=order)` | High |
| `detrend_loess` | `stats::loess(y ~ t, span=span)` | `statsmodels.nonparametric.smoothers_lowess.lowess()` | Medium |
| `stl_decompose` | `stats::stl(ts(x, frequency=period), s.window="periodic")` | `statsmodels.tsa.seasonal.STL(endog, period).fit()` | High |
| `stl_fdata` | `stats::stl()` per curve | -- | Medium |
| `decompose_additive` | `stats::decompose(ts(x, freq=p), type="additive")` | `statsmodels.tsa.seasonal.seasonal_decompose(model='additive')` | Medium |
| `decompose_multiplicative` | `stats::decompose(ts(x, freq=p), type="multiplicative")` | `statsmodels.tsa.seasonal.seasonal_decompose(model='multiplicative')` | Medium |
| `auto_detrend` | Compare AIC of `lm()` degree 1/2/3 and `loess()` | -- | Low |

**Test data**: `sin(2*pi*t/12) + 0.5*t + noise` (trend + seasonal + noise).

**Notes**:
- `stl()` in R vs fdars may differ in LOESS implementation details (number of
  robustness iterations, degree). Compare trend component to 1e-3.
- `detrend_diff` should match exactly (1e-15).
- LOESS bandwidth definition may differ (fraction of data vs absolute). Document
  the mapping.

---

### 9. Seasonal Analysis (`seasonal.rs`) -- Tiers 2-3

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `estimate_period_fft` | `stats::spectrum(x, method="pgram")` | `scipy.signal.periodogram(x, fs)` | High |
| `estimate_period_acf` | `stats::acf(x, lag.max)` + peak finding | `statsmodels.tsa.stattools.acf(x, nlags)` | High |
| `lomb_scargle` | `lomb::lsp(x, times)` | `astropy.timeseries.LombScargle(t, y).autopower()` | Medium |
| `detect_peaks` | `pracma::findpeaks(x, minpeakheight, minpeakdistance)` | `scipy.signal.find_peaks(x, height, distance)` | Medium |
| `hilbert_transform` | `signal::hilbert(x)` (R signal package) | `scipy.signal.hilbert(x)` | Medium |
| `ssa` | `Rssa::ssa(x, L)` + `Rssa::reconstruct()` | -- | Medium |
| `matrix_profile` | `tsmp::tsmp(x, window_size)` | `stumpy.stump(x, m=window_size)` | Medium |
| `autoperiod` | No direct equivalent (combine FFT + ACF) | -- | Low |
| `cfd_autoperiod` | No direct equivalent | -- | Low |
| `sazed` | No direct equivalent (ensemble method) | -- | Low |
| `seasonal_strength_variance` | `1 - Var(remainder)/Var(deseasonalized)` from `stl()` | Same formula from `STL().fit()` | Medium |
| `detect_amplitude_modulation` | No direct equivalent | -- | Low |
| `detect_amplitude_modulation_wavelet` | No direct equivalent | -- | Low |
| `instantaneous_period` | `signal::hilbert(x)` + `diff(unwrap(angle))` | `scipy.signal.hilbert()` | Low |

**Test data**:
- Pure sine: `sin(2*pi*t/20)` on 200 points (known period = 20)
- Noisy sine: same + Gaussian noise (SNR ~10)
- Multi-period: `sin(2*pi*t/20) + 0.5*sin(2*pi*t/7)` (periods 20 and 7)

**Notes**:
- Period detection methods should agree on the dominant period within 5% relative
  tolerance.
- For Tier 3 methods (Autoperiod, SAZED, CFDAutoperiod), validate against the
  known ground-truth period rather than another package.
- SSA: compare reconstructed trend + seasonal components against Rssa. Expect
  differences in automatic grouping -- compare with explicit grouping indices.

---

### 10. Simulation (`simulation.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `fourier_eigenfunctions` | `fda::eval.basis(t, create.fourier.basis())` | `skfda.representation.basis.FourierBasis().evaluate(t)` | Medium |
| `wiener_eigenfunctions` | `sqrt(2) * sin((k-0.5)*pi*t)` (analytical) | Same formula | Medium |
| `legendre_eigenfunctions` | `orthopolynom::legendre.polynomials(m)` | `numpy.polynomial.legendre` | Medium |
| `eigenvalues_*` | Analytical formulas: `1/k`, `exp(-k)`, `1/((k-0.5)*pi)^2` | Same formulas | High |
| `sim_kl` | Sample from KL expansion using `fda` eigenfunctions + `rnorm` scores | -- | Medium |

**Notes**:
- Eigenfunctions and eigenvalues have closed-form expressions. Compare to 1e-12.
- `sim_kl` is stochastic. Validate that the covariance structure of generated
  samples matches the theoretical covariance to within sampling error.

---

### 11. Functional Data Operations (`fdata.rs`) -- Tier 1

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `mean_1d` | `fda.usc::func.mean(fdataobj)` | `skfda.exploratory.stats.mean(fdatagrid)` | High |
| `center_1d` | `fda.usc::fdata.cen(fdataobj)` | -- | High |
| `norm_lp_1d` | `fda.usc::norm.fdata(fdataobj, lp=2)` | `skfda.misc.metrics.LpNorm(p=2)` | High |
| `deriv_1d` | `fda::deriv.fd(fdobj, 1)` or finite differences | -- | Medium |
| `geometric_median_1d` | Custom Weiszfeld algorithm comparison | `skfda.exploratory.stats.geometric_median()` | Medium |

**Test data**: Reuse the 50-curve dataset.

---

### 12. Utility (`utility.rs`, `helpers.rs`) -- Tier 1

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `simpsons_weights` | `pracma::simpsons(f, a, b)` | `scipy.integrate.simpson(y, x)` | High |
| `inner_product` | `fda.usc::inprod.fdata(fdata1, fdata2)` | -- | Medium |
| `inner_product_matrix` | `fda::inprod(fdobj1, fdobj2)` | -- | Medium |

**Notes**:
- Simpson's weights are deterministic. Compare to 1e-15.
- Inner products should match `fda::inprod()` to 1e-8.

---

### 13. Irregular Functional Data (`irreg_fdata.rs`) -- Tier 2

| fdars function | R reference | Python reference | Priority |
|---|---|---|---|
| `mean_irreg` | `fdapace::GetMeanCurve(Ly, Lt)` | `skfda.preprocessing.smoothing` on irregular grid | Medium |
| `cov_irreg` | `fdapace::GetCovSurface(Ly, Lt)` | -- | Medium |
| `to_regular_grid` | `fdapace::ConvertSupport()` or linear interpolation | `scipy.interpolate.interp1d()` | Medium |
| `integrate_irreg` | Trapezoidal rule on irregular grid | `scipy.integrate.trapezoid(y, x)` | Medium |
| `metric_lp_irreg` | `fda.usc::metric.lp()` after interpolation | -- | Low |

---

### 14. Streaming Depth (`streaming_depth.rs`) -- Tier 3

| fdars function | R reference | Priority |
|---|---|---|
| `StreamingFraimanMuniz` | Compare against batch `fda.usc::depth.FM()` on sliding window | Low |
| `StreamingBd` / `StreamingMbd` | Compare against batch `roahd::BD()` / `roahd::MBD()` | Low |

**Notes**: Validate streaming implementations against batch computation on the
same window of data. Results should be identical.

---

## Implementation Phases

### Phase 1: Infrastructure + Tier 1 Core (Depth, Basis, Metrics, fdata)

**Effort**: ~3 days

1. Set up `validation/R/install_deps.R` to install all required R packages
2. Create shared test data generator (`validation/R/generate_test_data.R`)
3. Implement depth validation (highest coverage impact)
4. Implement basis matrix validation (exact comparison possible)
5. Implement Lp metric validation
6. Implement fdata operations validation (mean, center, norm)

**R packages needed**: `fda`, `fda.usc`, `roahd`, `ddalpha`, `jsonlite`

### Phase 2: Tier 2 Statistical Methods

**Effort**: ~3 days

7. Smoothing validation (NW, local linear, local polynomial)
8. Clustering validation (k-means, fuzzy c-means, silhouette, CH)
9. Regression validation (FPCA, PLS, ridge)
10. Detrending/decomposition validation (STL, LOESS, polynomial)
11. Outlier detection validation
12. Simulation validation (eigenfunctions, eigenvalues)

**Additional R packages**: `KernSmooth`, `locpol`, `FNN`, `pls`, `glmnet`,
`e1071`, `cluster`, `fpc`

### Phase 3: Tier 3 Specialized Methods

**Effort**: ~2 days

13. Seasonal analysis (FFT, ACF, Lomb-Scargle, SSA, matrix profile)
14. Novel methods validated against ground truth (Autoperiod, SAZED, CFDAutoperiod)
15. Irregular functional data operations
16. Streaming depth (batch equivalence tests)

**Additional R packages**: `lomb`, `Rssa`, `tsmp`, `dtw`, `pracma`

---

## R Package Summary

All validation scripts require these R packages:

```r
# Core FDA
install.packages(c("fda", "fda.usc", "roahd", "ddalpha", "fdapace"))

# Statistics / Regression
install.packages(c("pls", "glmnet", "MASS"))

# Smoothing
install.packages(c("KernSmooth", "locpol", "FNN", "np"))

# Clustering
install.packages(c("e1071", "cluster", "fpc"))

# Time Series / Seasonal
install.packages(c("lomb", "Rssa", "tsmp", "dtw", "signal"))

# Utilities
install.packages(c("pracma", "jsonlite", "orthopolynom"))
```

## Python Package Summary (optional, secondary reference)

```
pip install scikit-fda scikit-learn statsmodels scipy numpy
pip install dtaidistance scikit-fuzzy astropy stumpy
```

---

## Success Criteria

- All Tier 1 validations pass within documented tolerances
- All Tier 2 validations pass (or deviations are documented with explanation)
- Tier 3 methods validated against ground truth where no reference exists
- Validation scripts are reproducible (`R --vanilla < validation/R/run_all.R`)
- Results are committed as JSON fixtures for CI regression testing
