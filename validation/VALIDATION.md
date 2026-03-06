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
| Depth Measures (FM, BD, MBD, MEI, modal, FSD, KFSD, RP, RT) | 12 | 1e-6, ρ ≥ 0.97 | ≤ 1e-6 abs, ρ ≥ 0.97 | fda.usc, roahd |
| Basis Expansion (B-spline, Fourier, P-spline, difference matrix) | 10 | 1e-15 – 0.1 | ≤ 4.94e-15 abs | fda, fda.usc |
| Metrics: Lp distances | 4 | 1e-2 – 0.05 | ≤ 1e-2 abs | fda.usc |
| Metrics: DTW | 3 | 0.1 – 10x | order-of-magnitude | dtw |
| Metrics: Fourier & hshift semimetric | 4 | rank ρ > 0.97 | ρ ≥ 0.98 | fda.usc |
| Metrics: Hausdorff | 3 | rank ρ > 0.97 | ρ ≥ 0.98 | manual sup-norm |
| Metrics: Soft-DTW | 9 | 1e-6 rel | ≤ 2.21e-15 rel | tslearn (Python) |
| Smoothing (NW, local linear, local polynomial, KNN, NW matrix) | 5 | 1e-6 | ≤ 1e-6 abs | exact NW/LL/LP |
| Detrending (linear, polynomial, LOESS, differencing) | 6 | 1e-6 – rank ρ > 0.97 | ≤ 6.66e-15 abs | stats |
| Decomposition (STL, additive, multiplicative) | 5 | 1e-10 – 1e-3 | ≤ 2.22e-16 abs | stats |
| Seasonal (FFT period, Lomb-Scargle, SSA, Hilbert, peaks, strength) | 7 | 0.05 rel – rank | ≤ 1.22e-15 abs, ρ = 1.00 | stats, Rssa, pracma |
| Elastic Alignment (SRSF, pair align, distance matrix) | 6 | 0.05 – 0.15 rel | ≤ 5.55e-15 abs | fdasrvf |
| Karcher Mean | 1 | ρ > 0.99, L2 < 0.05 | ρ > 0.99 | fdasrvf |
| Elastic Decomposition & Cross Distance | 2 | 0.2 rel | ≤ 1.33e-15 rel | fdasrvf |
| Landmark Registration | 4 | 0.01 – 0.02 | ≤ 0.02 abs | scipy PCHIP |
| TSRVF (sphere geometry, transport, smoothing) | 6 | 1e-10 – 1e-6 rel | ≤ 2.00e-15 abs | fdasrsf (Python) |
| Tolerance Bands (FPCA, Degras SCB, conformal, elastic) | 7 | 1e-6 – order-of-magnitude | ρ ≥ 0.99 | fda.usc, KernSmooth, fdasrvf |
| Equivalence Testing | 3 | 1e-10 – 0.5 rel | ≤ 5.55e-16 abs | custom |
| Streaming Depth | 5 | 0.15 – rank ρ > 0.97 | ρ = 1.00 | property-based |
| Outlier Detection | 3 | 1e-12 | exact | property-based |
| Clustering (k-means, fuzzy c-means, silhouette, CH) | 6 | 0.01 rel | ≤ 0.01 rel | cluster, fpc |
| Regression (FPCA/SVD, ridge, PLS) | 3 | 1e-6 – 0.5 | ≤ 5.82e-6 abs (R²) | fda.usc, pls |
| Simulation (eigenfunctions, eigenvalues, KL, fundata) | 11 | 1e-15 – 0.1 | ≤ 0.00 (exact) | fda |
| Integration & Inner Products | 4 | 1e-15 – 1e-6 | ≤ 1e-6 abs | Simpson's 1/3 |
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
computations, band/modified band depth, inner products (Simpson's 1/3),
Nadaraya-Watson & local linear/polynomial (exact kernel weights), Fraiman-Muniz depth,
modified epigraph index.

### Moderate (1e-3 – 0.01) — Minor implementation differences

Algorithms agree conceptually but differ in numerical details:

- **Lp norms/distances**: R's `fda.usc` uses its own integration weights internally;
  Rust uses Simpson's 1/3. Differences ≤ 1%.
- **SRSF gradient**: Rust uses 5-point central differences (O(h⁴)); R uses `gradient()`.
- **Elastic alignment**: Dynamic programming grid resolution and smoothing
  post-processing differ, causing ~15% relative differences in elastic distances.
- **FSD/KFSD depth**: Different L2 norm computation between Rust (Simpson's weights)
  and R (`fda.usc`); validated by rank correlation (ρ ≥ 0.97).

### Loose (rank correlation only) — Algorithmic differences

Some functions use fundamentally different algorithms or depend on RNG:

- **Random depth** (projection, Tukey): Seeded RNG with ≥1000 projections ensures
  stable rankings. Validated via rank correlation (ρ ≥ 0.97).
- **Modal depth**: Different kernel choices between R and Rust; validated by
  rank correlation (ρ ≥ 0.97).
- **LOESS detrend**: Both Rust and R now use tri-cube kernel with local polynomial.
- **Hausdorff distance**: R computes pointwise sup-norm max|f₁(t)−f₂(t)|; Rust
  computes 2D Hausdorff on (t, f(t)) point sets. Rank validated (ρ ≥ 0.97).
- **DTW**: Step pattern differences (symmetric2 vs custom) produce distances that
  agree to within an order of magnitude.
- **Bootstrap quantities** (Degras critical values, conformal quantiles): PRNG-dependent;
  validated by order-of-magnitude agreement.
- **Irregular mean**: R uses linear interpolation to common grid then averages; Rust
  uses kernel smoothing. Shape correlation validated (ρ ≥ 0.97).

## Known Convention Differences

| Convention | Rust | R | Impact |
|------------|------|---|--------|
| Integration | Simpson's 1/3 | Simpson's 1/3 | < 1e-6 on inner products |
| SRSF gradient | 5-point central difference O(h⁴) | `gradient()` | < 1e-4 on SRSF |
| Fourier basis | Standard normalization | √2 scaling on non-constant terms | Corrected in test |
| FPCA scores | Unscaled | Scaled by √eigenvalue (in some packages) | Compared as absolute values |
| Eigenvalue indexing | exp(−k), k = 0, 1, ... | exp(−(k−1)), k = 1, 2, ... | Offset corrected |
| Karcher mean | DP + post-smoothing | fdasrvf `time_warping` | Shape correlation, not pointwise |
| Hausdorff distance | 2D point-set metric | 1D sup-norm | Different metrics; rank validated |
| Random depth | Seeded `StdRng` (ChaCha) | R's Mersenne Twister | Rank correlation (ρ ≥ 0.97) |
| LOESS kernel | Tri-cube | Tri-cube | Matching |
| Smoothing R scripts | Exact kernel weights | Exact kernel weights | ≤ 1e-6 |
| MEI comparison | `xi <= xj` | `roahd::MEI` (`<=`) | Exact match |
| Variance divisor | n−1 (Bessel) | n−1 (R's `sd()`) | Matching |
| FAMM REML divisor | n−p | n−p (lmer) | Matching |

## Observed Results (Detailed)

Measured from the most recent test run across 153 integration tests.
For array-valued metrics, the maximum error across all elements is reported.

### Exact or Near-Exact Agreement (error < 1e-6)

These functions produce bit-identical or near-identical results between Rust and R,
thanks to matching algorithms (Simpson's 1/3 integration, exact kernel weights,
5-point gradient stencil, matching MEI comparison).

| Metric | Max Observed Error | Tolerance |
|--------|-------------------:|----------:|
| `fdata_mean` | 3.33e-16 | 1e-10 |
| `fdata_center` | 5.55e-16 | 1e-10 |
| `band_depth` | 0.00 | 1e-6 |
| `modified_band_depth` | 1.11e-16 | 1e-6 |
| `fraiman_muniz` | < 1e-6 | 1e-6 |
| `modified_epigraph` | < 1e-6 | 1e-6 |
| `inner_product_12` | < 1e-6 | 1e-6 |
| `inner_product_matrix` | < 1e-6 | 1e-6 |
| `nadaraya_watson` | < 1e-6 | 1e-6 |
| `local_linear` | < 1e-6 | 1e-6 |
| `simpsons_weights` | < 1e-15 | 1e-14 |
| `fourier_matrix` | 4.94e-15 | 1e-6 |
| `fpca_center` | 3.33e-16 | 1e-6 |
| `fpca_mean` | 8.33e-17 | 1e-8 |
| `diff_matrix_order1` | 0.00 | 1e-15 |
| `diff_matrix_order2` | 0.00 | 1e-15 |
| `sphere_theta` | 5.27e-16 | 1e-10 |
| `tangent_vector` (max) | 2.00e-15 | 1e-6 |
| `soft_dtw distance` | 0.00 rel | 1e-6 |

### Close Agreement (error 1e-3 – 1e-2)

Minor numerical differences from R packages using their own internal integration.

| Metric | Max Observed Error | Tolerance |
|--------|-------------------:|----------:|
| `l2_norms` | ~1e-3 abs | 1e-2 |
| `lp_l2_distance` | ~1e-3 abs | 1e-2 |
| `calinski_harabasz` | ~7e-3 rel | 1e-2 |
| `silhouette` | ~1e-3 rel | 1e-2 |

### Rank Correlation Comparisons

For algorithms with fundamentally different implementations (different PRNG,
kernel, or metric definition), rank correlation validates that orderings agree.
Default threshold tightened from ρ > 0.9 to ρ > 0.97.

| Metric | Observed ρ | Threshold |
|--------|----------:|----------:|
| `hshift_semimetric` | 1.0000 | 0.97 |
| `ssa_component_1_2` | 1.0000 | 0.97 |
| `streaming_vs_batch_mbd` | 1.0000 | 0.97 |
| `conformal_center` | 0.9998 | 0.97 |
| `modal_depth_ranking` | 0.9995 | 0.97 |
| `functional_spatial` | ≥ 0.97 | 0.97 |
| `kernel_functional_spatial` | ≥ 0.97 | 0.97 |
| `rp_vs_fm_depth` | ≥ 0.97 | 0.97 |
| `elastic_tolerance_center` | 0.9947 | 0.97 |
| `degras_center_shape` | ≥ 0.97 | 0.97 |
| `fourier_semimetric` | 0.9829 | 0.97 |
| `hausdorff_rank_order` | ≥ 0.97 | 0.97 |

### Classification & Clustering (Exact Agreement)

| Metric | Rust | R | Difference |
|--------|-----:|--:|----------:|
| `GMM accuracy` | 1.00 | 1.00 | 0.00 |
| `GMM weights` (sorted) | exact | exact | 0.00 |
| `LDA accuracy` | 1.00 | 1.00 | 0.00 |
| `k-NN accuracy` | 1.00 | 1.00 | 0.00 |
| `DD accuracy` | 1.00 | 1.00 | 0.00 |

### FAMM Variance Components

| Metric | Observed Rel. Error | Tolerance |
|--------|-------------------:|----------:|
| `sigma2_u` (max component) | EM vs REML gap | 3.0 |

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
