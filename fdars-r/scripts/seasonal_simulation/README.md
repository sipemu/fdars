# Seasonal Detection Simulation Study

This folder contains simulation studies comparing methods for detecting seasonality in functional time series data.

## Quick Start

```bash
# Run simulations in order of complexity
Rscript seasonal_basis_comparison.R           # Study 1: Fourier vs P-spline AIC
Rscript seasonality_detection_comparison.R    # Study 2: All 5 methods, varying strength
Rscript seasonality_detection_with_trend.R    # Study 3: Add non-linear trends
Rscript seasonality_detection_trend_types.R   # Study 4: 8 different trend types
```

## Simulation Studies (Low to High Complexity)

### Study 1: Fourier vs P-spline AIC Comparison
**Script**: `seasonal_basis_comparison.R`

Compares AIC between Fourier basis and P-splines to determine which fits seasonal data better.

- **Complexity**: Basic
- **Scenario**: Varying seasonal strength (0-1), no trends
- **Output**: `plots/seasonal_basis_comparison.pdf`, `seasonal_basis_results.rds`

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

Trend types tested:
- none, linear, quadratic, cubic
- exponential, logarithmic, sigmoid
- slow_sine (problematic for FFT)

- **Complexity**: Highest
- **Scenario**: 8 trend types × 5 seasonal strengths × 4 trend strengths × 20 curves
- **Output**: `plots/seasonality_detection_trend_types_*.pdf`

## Documentation

| File | Description |
|------|-------------|
| `seasonality_detection_report.qmd` | Quarto report with full analysis |
| `seasonality_detection_report.pdf` | Compiled PDF report |

## Key Results

| Method | F1 Score | FPR | Best For |
|--------|----------|-----|----------|
| Variance Strength | 97.3% | 2% | General use when period is known |
| Spectral Strength | 95.3% | 10% | Robust to unknown trends |
| FFT Confidence | 94.8% | 4% | Quick screening (avoid with slow oscillations) |
| AIC Comparison | 91.5% | 18% | Interpretable model comparison |
| ACF Confidence | 85.4% | 10% | Conservative detection |

## Period Parameter Requirements

Methods differ in whether they require the seasonal period as input:

| Method | Needs Period? | Notes |
|--------|---------------|-------|
| AIC Comparison | No | Compares basis types, period-agnostic |
| FFT Confidence | No | Estimates period automatically |
| ACF Confidence | No | Estimates period automatically |
| Variance Strength | **Yes** | Requires `period` in argvals units |
| Spectral Strength | **Yes** | Requires `period` in argvals units |

**Typical workflow**:
1. Use period-free methods (FFT, ACF, AIC) to detect IF seasonality exists
2. Use `estimate_period()` to find the period(s)
3. Use period-based methods (Variance, Spectral) for precise strength measurement

```r
# Step 1: Quick detection (no period needed)
fft_result <- estimate_period(fd, method = "fft", detrend = "linear")
is_seasonal <- fft_result$confidence > 6.0

# Step 2: Get period estimate
estimated_period <- fft_result$period

# Step 3: Precise measurement (period required)
strength <- seasonal_strength(fd, period = estimated_period,
                              method = "variance", detrend = "linear")
```

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

## Important Notes

1. **Period parameter**: Must be in argvals units (e.g., 0.2 for 5 cycles in [0,1])
2. **FFT vulnerability**: Slow oscillations (e.g., 1 cycle over entire series) cause 100% FPR
3. **Recommended method**: Variance Strength with `period = 0.2` and `detrend = "linear"`

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
