# Seasonal Detection Simulation Study

This folder contains simulation studies comparing methods for detecting seasonality in functional time series data.

## Summary & Recommendation

**Winner: Wavelet Strength** — achieves the highest F1 score (97.8%) with excellent robustness to trends and amplitude modulation.

| Method | F1 Score | FPR | Robustness to Trends |
|--------|----------|-----|----------------------|
| **Wavelet Strength** | **97.8%** | 4% | Excellent |
| Variance Strength | 97.3% | **2%** | Excellent |
| SAZED (Ensemble) | 87.5% | 3% | Excellent (parameter-free) |
| Spectral Strength | 95.3% | 10% | Good |
| FFT Confidence | 94.8% | 4% | Good |
| Autoperiod | 93.4% | 10% | Good |
| AIC Comparison | 91.5% | 18% | Moderate |
| CFDAutoperiod | 89.5% | 22% | Excellent (detrending) |
| ACF Confidence | 85.4% | 10% | Moderate |

### Method Selection Guide

| Use Case | Recommended Method |
|----------|-------------------|
| **Unknown signal characteristics** | SAZED (parameter-free ensemble) |
| **Lowest false positive rate** | Variance Strength (2% FPR) |
| **Best overall accuracy** | Wavelet Strength (97.8% F1) |
| **Strong trends present** | CFDAutoperiod (automatic detrending) |
| **Period estimation needed** | Autoperiod (hybrid FFT+ACF) |
| **Amplitude modulation** | Wavelet Strength (detects time-varying) |

### Recommended Usage

```r
# Option 1: Parameter-free detection (recommended for unknown signals)
result <- sazed(fd)
is_seasonal <- result$detected
confidence <- result$confidence

# Option 2: Highest accuracy (when period is known)
period <- 0.2  # Period in argvals units
strength <- seasonal.strength(fd, period = period, method = "wavelet")
is_seasonal <- strength > 0.26

# Option 3: Lowest false positive rate
strength <- seasonal.strength(fd, period = period, method = "variance")
is_seasonal <- strength > 0.2
```

### When Period is Unknown

```r
# Option A: Use SAZED (no period required, ensemble approach)
result <- sazed(fd)
period <- result$period
is_seasonal <- result$detected

# Option B: Use Autoperiod (hybrid FFT+ACF with refinement)
result <- autoperiod(fd)
period <- result$period
confidence <- result$confidence

# Option C: Use CFDAutoperiod (robust to trends)
result <- cfd.autoperiod(fd)
period <- result$period
```

### Critical Notes

1. **Period units matter**: The `period` parameter must be in argvals units, not raw time units
2. **SAZED for unknown signals**: When signal characteristics are unknown, SAZED provides robust parameter-free detection
3. **Thresholds are calibrated**: Default thresholds target ~5% false positive rate on pure noise
4. **Red noise caution**: FFT has high FPR (up to 100%) with AR(1) noise; use Variance/Wavelet Strength instead

## Quick Start

```bash
# Run simulations in order of complexity
Rscript seasonal_basis_comparison.R           # Study 1: Fourier vs P-spline AIC
Rscript seasonality_detection_comparison.R    # Study 2: All 9 methods comparison
Rscript seasonality_detection_with_trend.R    # Study 3: Non-linear trend robustness
Rscript seasonality_detection_trend_types.R   # Study 4: 8 different trend types
Rscript seasonality_robustness_tests.R        # Study 5: Robustness tests
```

## Simulation Studies

### Study 1: Fourier vs P-spline AIC Comparison
**Script**: `seasonal_basis_comparison.R`

Compares AIC between Fourier basis and P-splines to determine which fits seasonal data better.

- **Finding**: Fourier wins above seasonal strength ~0.17 (r = 0.96 correlation)
- **Output**: `plots/seasonal_basis_comparison.pdf`

### Study 2: Multi-Method Detection Comparison
**Script**: `seasonality_detection_comparison.R`

Compares 9 detection methods across varying seasonal strengths:
1. AIC Comparison (Fourier vs P-spline)
2. FFT Confidence
3. ACF Confidence
4. Variance Strength
5. Spectral Strength
6. Wavelet Strength
7. SAZED (Ensemble)
8. Autoperiod
9. CFDAutoperiod

- **Scenario**: 11 seasonal strengths × 50 curves each
- **Output**: `plots/seasonality_detection_comparison.pdf`, `plots/seasonality_detection_details.pdf`

### Study 3: Non-linear Trend Robustness
**Script**: `seasonality_detection_with_trend.R`

Tests how detection methods perform when non-linear trends are added.

- **Scenario**: 6 seasonal strengths × 6 trend strengths × 30 curves
- **Output**: `plots/seasonality_detection_trend_*.pdf`

### Study 4: Multiple Trend Types
**Script**: `seasonality_detection_trend_types.R`

Identifies which trend types cause false positives for each method.

Trend types tested: none, linear, quadratic, cubic, exponential, logarithmic, sigmoid, slow_sine

- **Finding**: slow_sine causes highest FPR (70% for AIC)
- **Output**: `plots/seasonality_detection_trend_types_*.pdf`

### Study 5: Robustness Tests
**Script**: `seasonality_robustness_tests.R`

Tests method robustness to real-world challenges:
- **Red Noise (AR(1))**: FFT fails at high autocorrelation; Variance/Wavelet robust
- **Multiple Seasonalities**: All methods detect primary when dominant
- **Amplitude Modulation**: Wavelet best at detecting time-varying seasonality
- **Outliers**: Pre-filtering recommended; Wavelet most robust

- **Output**: `plots/robustness_*.pdf`

## Documentation

| File | Description |
|------|-------------|
| `seasonality_detection_report.qmd` | Quarto report with full analysis |
| `seasonality_detection_report.pdf` | Compiled PDF report |

## Detection Methods Summary

| Method | Needs Period? | Key Advantage |
|--------|---------------|---------------|
| SAZED | No | Parameter-free ensemble, robust |
| Autoperiod | No | Hybrid FFT+ACF with refinement |
| CFDAutoperiod | No | Robust to trends via differencing |
| FFT Confidence | No | Fast, estimates period |
| ACF Confidence | No | Good for noisy data |
| Variance Strength | **Yes** | Lowest FPR (2%) |
| Spectral Strength | **Yes** | Good balance |
| Wavelet Strength | **Yes** | Best F1, detects modulation |
| AIC Comparison | No | Basis selection approach |

## Detection Thresholds

```r
# Calibrated to ~5% FPR on pure noise
thresholds <- list(
  variance_strength = 0.2,    # Variance ratio
  spectral_strength = 0.3,    # Spectral power
  wavelet_strength = 0.26,    # Wavelet energy
  fft_confidence = 6.0,       # Power ratio
  acf_confidence = 0.25,      # ACF peak
  aic_comparison = 0,         # Fourier AIC < P-spline AIC
  sazed = 3,                  # >=3 of 5 components agree
  autoperiod_acf = 0.3,       # ACF correlation at period
  cfd_autoperiod = 0.25       # ACF validation threshold
)
```

## Requirements

```r
library(fdars)      # Main package
library(ggplot2)    # Plotting
library(ggdist)     # Uncertainty visualization
library(tidyr)      # Data reshaping
library(dplyr)      # Data manipulation
library(gridExtra)  # Multi-panel plots
```

## File Structure

```
seasonal_simulation/
├── README.md                                  # This file
├── seasonal_basis_comparison.R                # Study 1: Fourier vs P-spline
├── seasonality_detection_comparison.R         # Study 2: Multi-method comparison
├── seasonality_detection_with_trend.R         # Study 3: Non-linear trends
├── seasonality_detection_trend_types.R        # Study 4: Multiple trend types
├── seasonality_robustness_tests.R             # Study 5: Robustness tests
├── generate_example_plots.R                   # Example visualization
├── verify_wavelet_implementation.R            # Wavelet verification
├── seasonality_detection_report.qmd           # Full Quarto report
├── seasonality_detection_report.pdf           # Compiled report
├── plots/                                     # Output plots
│   ├── seasonal_basis_comparison.pdf
│   ├── seasonality_detection_comparison.pdf
│   ├── seasonality_detection_details.pdf
│   ├── seasonality_detection_trend_*.pdf
│   ├── seasonality_detection_trend_types_*.pdf
│   ├── robustness_*.pdf
│   └── wavelet_verification_*.pdf
└── *.rds                                      # Saved R data objects
```
