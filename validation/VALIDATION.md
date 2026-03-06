# Validation Report

Numerical accuracy of `fdars-core` is validated against reference implementations
in **R** (primary) and **Python** (Soft-DTW, TSRVF, Landmark Registration).
Every reference value is generated with fixed random seeds and stored as JSON
fixtures so that validation is fully reproducible.

## Architecture

| Component | Path | Purpose |
|-----------|------|---------|
| R reference scripts | `validation/R/validate_*.R` | Generate expected values with `set.seed(123)` |
| Python reference script | `validation/generate_new_features.py` | Soft-DTW (tslearn), TSRVF (fdasrsf), Landmark (scipy) |
| Test data | `validation/data/*.json` | Shared input curves (50 x 101 grid, seasonal, irregular, etc.) |
| Expected values | `validation/expected/*.json` | Reference outputs from R / Python |
| Rust integration tests | `fdars-core/tests/validate_against_r.rs` | 153 tests comparing Rust ↔ R/Python |
| Rust module tests | `fdars-core/tests/validate_new_modules.rs` | 42 tests (property + fixture) for newer modules |

## Reference Packages

### R

| Package | Used For |
|---------|----------|
| **fda.usc** | Depth (FM, modal, FSD, KFSD), Lp distances, Fourier/hshift semimetric, FPCA, smoothing, classification |
| **fdasrvf** | SRSF transform, elastic alignment, Karcher mean, elastic distance, elastic tolerance bands |
| **roahd** | Band depth (BD), modified band depth (MBD), modified epigraph index (MEI) |
| **fda** | Fourier & B-spline basis, eigenfunction reference |
| **KernSmooth** | Nadaraya-Watson, local linear/polynomial smoothing, Degras SCB |
| **dtw** | Dynamic Time Warping distances |
| **Rssa** | Singular Spectrum Analysis (SSA) reconstruction |
| **lomb** | Lomb-Scargle periodogram (optional) |
| **pracma** | Peak detection (`findpeaks`) |
| **stats** | Linear/polynomial detrend, LOESS, STL, ACF, spectrum, decompose |
| **cluster** | Silhouette score |
| **fpc** | Calinski-Harabasz index |
| **mclust** | GMM clustering (Mclust) |
| **lme4** | Functional mixed models (lmer / REML) |
| **class** | k-NN classification |

### Python

| Package | Used For |
|---------|----------|
| **tslearn** | Soft-DTW distance, divergence, barycenter |
| **fdasrsf** | TSRVF sphere geometry, tangent vector transport |
| **scipy** | PCHIP monotone interpolation for landmark registration |

## Test Coverage Summary

**195 total tests** (153 in `validate_against_r.rs` + 42 in `validate_new_modules.rs`).
All 195 tests pass. Observed errors are measured from the most recent test run.

| Category | Tests | Tolerance | Observed Error | Reference |
|----------|------:|-----------|----------------|-----------|
| Functional Data (mean, center, norm) | 6 | 1e-12 – 1e-2 | ≤ 5.55e-16 abs | fda.usc |
| Depth Measures (FM, BD, MBD, MEI, modal, FSD, KFSD, RP, RT) | 12 | 1e-6 – 0.02 | ≤ 1.00e-2 abs, ρ ≥ 0.93 | fda.usc, roahd |
| Basis Expansion (B-spline, Fourier, P-spline, difference matrix) | 10 | 1e-15 – 0.1 | ≤ 4.94e-15 abs | fda, fda.usc |
| Metrics: Lp distances | 4 | 1e-2 – 0.05 | ≤ 4.27e-16 rel | fda.usc |
| Metrics: DTW | 3 | 0.1 – 10x | order-of-magnitude | dtw |
| Metrics: Fourier & hshift semimetric | 4 | rank ρ > 0.9 | ρ = 0.98, 1.00 | fda.usc |
| Metrics: Hausdorff | 3 | rank ρ > 0.9 | ρ = 0.98 | manual sup-norm |
| Metrics: Soft-DTW | 9 | 1e-6 rel | ≤ 2.21e-15 rel | tslearn (Python) |
| Smoothing (NW, local linear, local polynomial, KNN, NW matrix) | 5 | 1e-4 – 0.15 | ≤ 7.87e-4 abs | KernSmooth |
| Detrending (linear, polynomial, LOESS, differencing) | 6 | 1e-6 – rank ρ > 0.75 | ≤ 6.66e-15 abs, ρ = 0.79 | stats |
| Decomposition (STL, additive, multiplicative) | 5 | 1e-10 – 1e-3 | ≤ 2.22e-16 abs | stats |
| Seasonal (FFT period, Lomb-Scargle, SSA, Hilbert, peaks, strength) | 7 | 0.05 rel – rank | ≤ 1.22e-15 abs, ρ = 1.00 | stats, Rssa, pracma |
| Elastic Alignment (SRSF, pair align, distance matrix) | 6 | 0.05 – 0.15 rel | ≤ 5.55e-15 abs, ≤ 2.28e-15 rel | fdasrvf |
| Karcher Mean | 1 | ρ > 0.99, L2 < 0.05 | ρ > 0.99 | fdasrvf |
| Elastic Decomposition & Cross Distance | 2 | 0.2 rel | ≤ 1.33e-15 rel | fdasrvf |
| Landmark Registration | 4 | 0.01 – 0.02 | ≤ 0.02 abs | scipy PCHIP |
| TSRVF (sphere geometry, transport, smoothing) | 6 | 1e-10 – 1e-6 rel | ≤ 2.00e-15 abs, ≤ 4.31e-16 rel | fdasrsf (Python) |
| Tolerance Bands (FPCA, Degras SCB, conformal, elastic) | 7 | 1e-6 – order-of-magnitude | ρ ≥ 0.99 | fda.usc, KernSmooth, fdasrvf |
| Equivalence Testing | 3 | 1e-10 – 0.5 rel | ≤ 5.55e-16 abs, 3.4e-2 rel | custom |
| Streaming Depth | 5 | 0.15 – rank ρ > 0.9 | ρ = 1.00 | property-based |
| Outlier Detection | 3 | 1e-12 | exact | property-based |
| Clustering (k-means, fuzzy c-means, silhouette, CH) | 6 | 0.01 – 0.05 rel | ≤ 4.58e-4 rel | cluster, fpc |
| Regression (FPCA/SVD, ridge, PLS) | 3 | 1e-6 – 0.5 | ≤ 5.82e-6 abs (R²) | fda.usc, pls |
| Simulation (eigenfunctions, eigenvalues, KL, fundata) | 11 | 1e-15 – 0.1 | ≤ 0.00 (exact) | fda |
| Integration & Inner Products | 4 | 1e-15 – 1e-2 | ≤ 3.58e-3 abs | internal |
| Irregular FData (integrate, norm, mean, interpolate, distance) | 5 | 1e-4 – rank ρ > 0.9 | ≤ 1.11e-16 abs, ρ = 0.99 | manual R |
| Scalar-on-Function Regression | 3 | 0.01 – 0.3 | ≤ 1.77e-3 abs | fda.usc |
| Function-on-Scalar Regression (FOSR, FANOVA) | 6 | 1e-8 – ρ > 0.8 | ≤ 9.74e-5 rel | property-based |
| GMM Clustering | 10 | 1e-10 – 0.05 | 0.00 (exact match) | mclust |
| Classification (LDA, k-NN, DD) | 3 | 0.05 – 0.1 | 0.00 (exact match) | fda.usc, class |
| Functional Mixed Models (FAMM) | 4 | σ² ratios, R² > 0.9 | σ² rel ≤ 0.14 | lme4 |
| **Total** | **195** | | **all pass** | |

## Tolerance Explanations

### Tight (1e-15 – 1e-6) — Closed-form / deterministic

Functions with analytical solutions or identical algorithms produce near-identical
results between R and Rust. Examples: eigenvalues, difference matrices, mean/center
computations, band/modified band depth.

### Moderate (1e-4 – 0.05) — Minor implementation differences

Algorithms agree conceptually but differ in numerical details:

- **Integration rule**: Rust uses trapezoidal; R's `fda.usc` uses Simpson's 1/3.
  This causes ~1% differences in Lp norms and inner products.
- **SRSF gradient**: Rust uses centered finite differences; R uses `gradient()`.
  Produces ~5% differences in SRSF values.
- **Elastic alignment**: Dynamic programming grid resolution and smoothing
  post-processing differ, causing ~15% relative differences in elastic distances.
- **Smoothing kernels**: Rust and R may use different default kernels or bandwidth
  interpretations, producing moderate differences in fitted values.

### Loose (0.1 – rank correlation only) — Algorithmic differences

Some functions use fundamentally different algorithms or depend on RNG:

- **Random depth** (projection, Tukey): Rust and R use different PRNGs, so exact
  values cannot match. Validated via rank correlation (ρ > 0.6–0.75) and
  statistical properties (mean, range).
- **Modal depth**: Different kernel choices between R and Rust; validated by
  rank correlation (ρ > 0.9).
- **LOESS detrend**: R uses tri-cube kernel; Rust uses Gaussian + local polynomial.
  Validated by rank correlation (ρ > 0.75) and variance reduction.
- **Hausdorff distance**: R computes pointwise sup-norm max|f₁(t)−f₂(t)|; Rust
  computes 2D Hausdorff on (t, f(t)) point sets. These are related but distinct
  metrics; validated by rank correlation of distance orderings.
- **DTW**: Step pattern differences (symmetric2 vs custom) produce distances that
  agree to within an order of magnitude.
- **Bootstrap quantities** (Degras critical values, conformal quantiles): PRNG-dependent;
  validated by order-of-magnitude agreement.
- **Irregular mean**: R uses linear interpolation to common grid then averages; Rust
  uses kernel smoothing. Shape correlation validated (ρ > 0.9).

## Known Convention Differences

| Convention | Rust | R | Impact |
|------------|------|---|--------|
| Integration | Trapezoidal | Simpson's 1/3 | ~1% on Lp norms/distances |
| SRSF gradient | Centered finite difference | `gradient()` | ~5% on SRSF values |
| Fourier basis | Standard normalization | √2 scaling on non-constant terms | Corrected in test |
| FPCA scores | Unscaled | Scaled by √eigenvalue (in some packages) | Compared as absolute values |
| Eigenvalue indexing | exp(−k), k = 0, 1, ... | exp(−(k−1)), k = 1, 2, ... | Offset corrected |
| Karcher mean | DP + post-smoothing | fdasrvf `time_warping` | Shape correlation, not pointwise |
| Hausdorff distance | 2D point-set metric | 1D sup-norm | Different metrics; rank validated |
| Random depth | `rand` crate PRNG | R's Mersenne Twister | Rank correlation only |
| LOESS | Gaussian kernel, local polynomial | Tri-cube kernel, local polynomial | Rank correlation only |

## Observed Results (Detailed)

Measured from the most recent test run (593 individual metric comparisons across
153 integration tests). For array-valued metrics, the maximum error across all
elements is reported.

### Exact Agreement (error < 1e-14)

These functions produce bit-identical or near-identical results between Rust and R.

| Metric | Max Observed Error | Tolerance |
|--------|-------------------:|----------:|
| `fdata_mean` | 3.33e-16 | 1e-10 |
| `fdata_center` | 5.55e-16 | 1e-10 |
| `band_depth` | 0.00 | 1e-6 |
| `modified_band_depth` | 1.11e-16 | 1e-6 |
| `fraiman_muniz` | 7.77e-16 | 2e-2 |
| `diff_matrix_order1` | 0.00 | 1e-15 |
| `diff_matrix_order2` | 0.00 | 1e-15 |
| `eigenvalues_exponential` | 0.00 | 1e-10 |
| `eigenvalues_linear` | 0.00 | 1e-10 |
| `eval_exponential` | 0.00 | 1e-15 |
| `eval_linear` | 0.00 | 1e-15 |
| `eval_wiener` | 0.00 | 1e-10 |
| `fourier_matrix` | 4.94e-15 | 1e-6 |
| `fourier_fit_fitted` | 1.55e-15 | 1e-1 |
| `fpca_center` | 3.33e-16 | 1e-6 |
| `fpca_mean` | 8.33e-17 | 1e-8 |
| `stl_reconstruction` | 2.22e-16 | 1e-10 |
| `additive_decomp_recon` | 2.22e-16 | 1e-6 |
| `linear_detrended` | 3.33e-15 | 1e-6 |
| `linear_trend` | 3.33e-15 | 1e-6 |
| `poly_detrended` | 6.66e-15 | 1e-4 |
| `poly_trend` | 6.66e-15 | 1e-4 |
| `lp_l2_distance` | 2.22e-16 | 1e-2 |
| `l2_norms` | 4.44e-16 | 1e-2 |
| `srsf_row0` | 2.66e-15 | 5e-2 |
| `srsf_row1` | 5.55e-15 | 5e-2 |
| `srsf_roundtrip_vs_r` | 1.11e-15 | 5e-2 |
| `elastic_distance_01` | 5.94e-16 rel | 1.5e-1 |
| `amplitude_distance` | 5.94e-16 rel | 2e-1 |
| `soft_dtw distance(0,1)` | 0.00 rel | 1e-6 |
| `soft_dtw divergence(0,1)` | 1.19e-15 rel | 1e-6 |
| `soft_dtw self(0,0)` | 1.69e-16 rel | 1e-6 |
| `sphere_theta` | 5.27e-16 | 1e-10 |
| `tangent_vector` (max) | 2.00e-15 | 1e-6 |
| `aligned_SRSF_norm` (max) | 4.31e-16 rel | 1e-6 |
| `hilbert_amplitude` | 1.22e-15 | 5e-2 |
| `fft_period` | 2.21e-16 rel | 5e-2 |
| `lomb_peak_vs_true` | 0.00 rel | 1e-1 |
| `intercept` | 2.22e-16 | 1e-6 |
| `slope` | 4.44e-16 | 1e-6 |
| `pair_gamma_start` | 0.00 | 1e-10 |
| `pair_gamma_end` | 0.00 | 1e-10 |
| `irreg_integrate` | 1.11e-16 | 1e-4 |
| `irreg_norm_l2` | 1.11e-16 | 1e-4 |
| `trapezoidal_weights` | 8.88e-16 | 1e-14 |

### Close Agreement (error 1e-4 – 1e-2)

Minor numerical differences from integration rules or finite-difference schemes.

| Metric | Max Observed Error | Tolerance |
|--------|-------------------:|----------:|
| `nadaraya_watson` | 4.34e-5 abs | 1e-4 |
| `local_linear` | 1.47e-4 abs | 5e-4 |
| `local_polynomial_degree2` | 7.87e-4 abs | 1.5e-1 |
| `smoothing_matrix_nw_row100` | 4.86e-17 abs | 1e-6 |
| `fpca_singular_values` | 1.78e-14 abs | 1e-4 |
| `knn_k5` | 4.44e-16 abs | 1e-4 |
| `kernel_functional_spatial` | 1.11e-8 abs | 1e-4 |
| `functional_spatial` | 6.53e-3 abs | 1e-2 |
| `inner_product_12` | 1.70e-3 abs | 1e-2 |
| `inner_product_matrix` | 3.58e-3 abs | 1e-2 |
| `modified_epigraph` | 1.00e-2 abs | 2e-2 |
| `FPC_beta_vs_R` | 5.06e-3 abs | 1e-1 |
| `srsf_roundtrip_vs_original` | 3.84e-2 abs | 1e-1 |
| `Fitted values (scalar-on-function)` | 1.77e-3 abs | 3e-1 |
| `Residuals (scalar-on-function)` | 1.77e-3 abs | 3e-1 |
| `Residual_SS` | 2.01e-3 rel | 5e-2 |
| `R² (fregre.pc)` | 5.82e-6 abs | 1e-2 |
| `calinski_harabasz` | 4.58e-4 rel | 5e-2 |
| `equivalence_critical_value` | 3.43e-2 rel | 5e-1 |
| `FOSR mean residual L²` | 9.74e-5 rel | 3e-1 |

### Rank Correlation Comparisons

For algorithms with fundamentally different implementations (different PRNG,
kernel, or metric definition), rank correlation validates that orderings agree.

| Metric | Observed ρ | Threshold |
|--------|----------:|----------:|
| `hshift_semimetric` | 1.0000 | 0.90 |
| `ssa_component_1_2` | 1.0000 | 0.90 |
| `streaming_vs_batch_mbd` | 1.0000 | 0.90 |
| `conformal_center` | 0.9998 | 0.90 |
| `modal_depth_ranking` | 0.9995 | 0.90 |
| `elastic_tolerance_center` | 0.9947 | 0.90 |
| `irreg_mean_curve_shape` | 0.9862 | 0.90 |
| `degras_center_shape` | 0.9857 | 0.90 |
| `fourier_semimetric` | 0.9829 | 0.90 |
| `rp_vs_fm_depth` | 0.9782 | 0.90 |
| `hausdorff_rank_order` | 0.9758 | 0.90 |
| `random_projection_ranks` | 0.9283 | 0.75 |
| `loess_trend_shape` | 0.7874 | 0.75 |

### Classification & Clustering (Exact Agreement)

| Metric | Rust | R | Difference |
|--------|-----:|--:|----------:|
| `GMM accuracy` | 1.00 | 1.00 | 0.00 |
| `GMM weights` (sorted) | exact | exact | 0.00 |
| `LDA accuracy` | 1.00 | 1.00 | 0.00 |
| `k-NN accuracy` | 1.00 | 1.00 | 0.00 |
| `DD accuracy` | 1.00 | 1.00 | 0.00 |
| `DD depths` (per class) | exact | exact | ≤ 2.22e-16 |
| `avg_silhouette` | exact | exact | 0.00 |

### FAMM Variance Components

| Metric | Observed Rel. Error | Tolerance |
|--------|-------------------:|----------:|
| `sigma2_u` (max component) | 0.137 | 3.0 |

## Reproducibility

### Regenerate R references
```bash
cd validation && Rscript R/run_all.R
```

### Regenerate Python references
```bash
cd validation && python generate_new_features.py
```

### Run all validation tests
```bash
cargo test -p fdars-core --features linalg
```

### Run specific test suites
```bash
cargo test --test validate_against_r --features linalg    # R/Python validation
cargo test --test validate_new_modules --features linalg   # newer modules
```

### Install R dependencies
```bash
Rscript validation/R/install_deps.R
```

### Install Python dependencies
```bash
cd validation && pip install tslearn fdasrsf scipy numpy
```
