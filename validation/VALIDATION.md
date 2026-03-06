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

| Category | Tests | Tolerance | Reference |
|----------|------:|-----------|-----------|
| Functional Data (mean, center, norm) | 6 | 1e-12 – 1e-2 | fda.usc |
| Depth Measures (FM, BD, MBD, MEI, modal, FSD, KFSD, RP, RT) | 12 | 1e-6 – 0.02 | fda.usc, roahd |
| Basis Expansion (B-spline, Fourier, P-spline, difference matrix) | 10 | 1e-15 – 0.1 | fda, fda.usc |
| Metrics: Lp distances | 4 | 1e-2 – 0.05 | fda.usc |
| Metrics: DTW | 3 | 0.1 – 10x | dtw |
| Metrics: Fourier & hshift semimetric | 4 | rank ρ > 0.9 | fda.usc |
| Metrics: Hausdorff | 3 | rank ρ > 0.9 | manual sup-norm |
| Metrics: Soft-DTW | 9 | 1e-6 rel | tslearn (Python) |
| Smoothing (NW, local linear, local polynomial, KNN, NW matrix) | 5 | 1e-4 – 0.15 | KernSmooth |
| Detrending (linear, polynomial, LOESS, differencing) | 6 | 1e-6 – rank ρ > 0.75 | stats |
| Decomposition (STL, additive, multiplicative) | 5 | 1e-10 – 1e-3 | stats |
| Seasonal (FFT period, Lomb-Scargle, SSA, Hilbert, peaks, strength) | 7 | 0.05 rel – rank | stats, Rssa, pracma |
| Elastic Alignment (SRSF, pair align, distance matrix) | 6 | 0.05 – 0.15 rel | fdasrvf |
| Karcher Mean | 1 | ρ > 0.99, L2 < 0.05 | fdasrvf |
| Elastic Decomposition & Cross Distance | 2 | 0.2 rel | fdasrvf |
| Landmark Registration | 4 | 0.01 – 0.02 | scipy PCHIP |
| TSRVF (sphere geometry, transport, smoothing) | 6 | 1e-10 – 1e-6 rel | fdasrsf (Python) |
| Tolerance Bands (FPCA, Degras SCB, conformal, elastic) | 7 | 1e-6 – order-of-magnitude | fda.usc, KernSmooth, fdasrvf |
| Equivalence Testing | 3 | 1e-10 – 0.5 rel | custom |
| Streaming Depth | 5 | 0.15 – rank ρ > 0.9 | property-based |
| Outlier Detection | 3 | 1e-12 | property-based |
| Clustering (k-means, fuzzy c-means, silhouette, CH) | 6 | 0.01 – 0.05 rel | cluster, fpc |
| Regression (FPCA/SVD, ridge, PLS) | 3 | 1e-6 – 0.5 | fda.usc, pls |
| Simulation (eigenfunctions, eigenvalues, KL, fundata) | 11 | 1e-15 – 0.1 | fda |
| Integration & Inner Products | 4 | 1e-15 – 1e-2 | internal |
| Irregular FData (integrate, norm, mean, interpolate, distance) | 5 | 1e-4 – rank ρ > 0.9 | manual R |
| Scalar-on-Function Regression | 3 | 0.01 – 0.3 | fda.usc |
| Function-on-Scalar Regression (FOSR, FANOVA) | 6 | 1e-8 – ρ > 0.8 | property-based |
| GMM Clustering | 10 | 1e-10 – 0.05 | mclust |
| Classification (LDA, k-NN, DD) | 3 | 0.05 – 0.1 | fda.usc, class |
| Functional Mixed Models (FAMM) | 4 | σ² ratios, R² > 0.9 | lme4 |
| **Total** | **195** | | |

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
