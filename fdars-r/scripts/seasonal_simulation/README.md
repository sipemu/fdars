# Seasonal Detection Simulation Study

This folder contains simulation studies comparing methods for detecting seasonality in functional time series data.

## Summary & Recommendation

**Winner: Variance Strength** — achieves the highest accuracy (97.3% F1) with the lowest false positive rate (2%) and is most robust to non-linear trends.

| Method | F1 Score | FPR | Robustness to Trends |
|--------|----------|-----|----------------------|
| **Variance Strength** | **97.3%** | **2%** | Excellent (0.4% F1 drop) |
| Spectral Strength | 95.3% | 10% | Good (3.9% F1 drop) |
| FFT Confidence | 94.8% | 4% | Good (2.0% F1 drop) |
| AIC Comparison | 91.5% | 18% | Moderate (5.7% F1 drop) |
| ACF Confidence | 85.4% | 10% | Moderate (4.5% F1 drop) |

### Recommended Usage

```r
# Primary recommendation: Variance Strength
period <- 0.2  # Period in argvals units (e.g., 1/5 for 5 cycles in [0,1])
strength <- seasonal_strength(fd, period = period, method = "variance", detrend = "linear")
is_seasonal <- strength > 0.2
```

### When Period is Unknown

```r
# Step 1: Estimate period using FFT (no period required)
result <- estimate_period(fd, method = "fft", detrend = "linear")
estimated_period <- result$period

# Step 2: Measure strength with estimated period
strength <- seasonal_strength(fd, period = estimated_period, method = "variance")
is_seasonal <- strength > 0.2
```

### Critical Notes

1. **Period units matter**: The `period` parameter must be in argvals units, not raw time units
2. **Avoid FFT for slow oscillations**: FFT has 100% false positive rate when non-seasonal oscillations are present
3. **Thresholds are calibrated**: All thresholds target ~5% false positive rate on pure noise

## Quick Start

```bash
# Run simulations in order of complexity
Rscript seasonal_basis_comparison.R           # Study 1: Fourier vs P-spline AIC
Rscript seasonality_detection_comparison.R    # Study 2: All 5 methods, varying strength
Rscript seasonality_detection_with_trend.R    # Study 3: Add non-linear trends
Rscript seasonality_detection_trend_types.R   # Study 4: 8 different trend types
```

## Simulation Studies

### Study 1: Fourier vs P-spline AIC Comparison
**Script**: `seasonal_basis_comparison.R`

Compares AIC between Fourier basis and P-splines to determine which fits seasonal data better.

- **Complexity**: Basic
- **Scenario**: Varying seasonal strength (0-1), no trends
- **Output**: `plots/seasonal_basis_comparison.pdf`

### Study 2: Multi-Method Detection Comparison
**Script**: `seasonality_detection_comparison.R`

Compares 5 detection methods across varying seasonal strengths:
1. AIC Comparison (Fourier vs P-spline)
2. FFT Confidence
3. ACF Confidence
4. Variance Strength
5. Spectral Strength

- **Complexity**: Moderate
- **Scenario**: 11 seasonal strengths × 50 curves each, no trends
- **Output**: `plots/seasonality_detection_comparison.pdf`, `plots/seasonality_detection_details.pdf`

### Study 3: Non-linear Trend Robustness
**Script**: `seasonality_detection_with_trend.R`

Tests how detection methods perform when non-linear trends (quadratic + cubic + sigmoid) are added.

- **Complexity**: High
- **Scenario**: 6 seasonal strengths × 6 trend strengths × 30 curves
- **Output**: `plots/seasonality_detection_trend_*.pdf`

### Study 4: Multiple Trend Types
**Script**: `seasonality_detection_trend_types.R`

Identifies which trend types cause false positives for each method.

Trend types tested: none, linear, quadratic, cubic, exponential, logarithmic, sigmoid, slow_sine

- **Complexity**: Highest
- **Scenario**: 8 trend types × 5 seasonal strengths × 4 trend strengths × 20 curves
- **Output**: `plots/seasonality_detection_trend_types_*.pdf`

## Documentation

| File | Description |
|------|-------------|
| `seasonality_detection_report.qmd` | Quarto report with full analysis |
| `seasonality_detection_report.pdf` | Compiled PDF report |

## Period Parameter Requirements

| Method | Needs Period? | Notes |
|--------|---------------|-------|
| AIC Comparison | No | Compares basis types, period-agnostic |
| FFT Confidence | No | Estimates period automatically |
| ACF Confidence | No | Estimates period automatically |
| Variance Strength | **Yes** | Requires `period` in argvals units |
| Spectral Strength | **Yes** | Requires `period` in argvals units |

## Detection Thresholds

```r
# Calibrated to ~5% FPR on pure noise
thresholds <- list(
  aic_comparison = 0,        # Fourier AIC < P-spline AIC
  fft_confidence = 6.0,      # Power ratio
  acf_confidence = 0.25,     # ACF peak
  strength_variance = 0.2,   # Variance ratio
  strength_spectral = 0.3    # Spectral power
)
```

## Requirements

```r
library(fdars)      # Main package
library(ggplot2)    # Plotting
library(ggdist)     # Uncertainty visualization (stat_halfeye)
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
├── seasonality_detection_report.qmd           # Full Quarto report
├── seasonality_detection_report.pdf           # Compiled report
├── plots/                                     # Output plots
│   ├── seasonal_basis_comparison.pdf
│   ├── seasonality_detection_comparison.pdf
│   ├── seasonality_detection_details.pdf
│   ├── seasonality_detection_trend_*.pdf
│   └── seasonality_detection_trend_types_*.pdf
└── *.rds                                      # Saved R data objects
```
