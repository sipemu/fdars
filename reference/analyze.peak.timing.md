# Analyze Peak Timing Variability

For short series (e.g., 3-5 years of yearly data), this function detects
one peak per cycle and analyzes how peak timing varies between cycles.
Uses Fourier basis smoothing for peak detection.

## Usage

``` r
analyze.peak.timing(fdataobj, period, smooth_nbasis = NULL)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known period (e.g., 365 for daily data with yearly seasonality).

- smooth_nbasis:

  Number of Fourier basis functions for smoothing. If NULL, uses GCV for
  automatic selection (range 5-25). Default: NULL.

## Value

A list with components:

- peak_times:

  Vector of peak times

- peak_values:

  Vector of peak values

- normalized_timing:

  Position within cycle (0-1 scale)

- mean_timing:

  Mean normalized timing

- std_timing:

  Standard deviation of normalized timing

- range_timing:

  Range of normalized timing (max - min)

- variability_score:

  Variability score (0 = stable, 1 = highly variable)

- timing_trend:

  Linear trend in timing (positive = peaks getting later)

- cycle_indices:

  Cycle indices (1-indexed)

## Details

The variability score is computed as std_timing / 0.1, capped at 1. A
score \> 0.5 suggests peaks are shifting substantially between cycles.
The timing_trend indicates if peaks are systematically moving earlier or
later over time.

Fourier basis smoothing is ideal for seasonal signals because it
naturally captures periodic patterns.

## Examples

``` r
# 5 years of yearly data where peak shifts
t <- seq(0, 5, length.out = 365 * 5)
periods <- c(1, 1, 1, 1, 1)  # 5 complete years
# Peaks shift: March (0.2), April (0.3), May (0.4), April (0.3), March (0.2)
peak_phases <- c(0.2, 0.3, 0.4, 0.3, 0.2)
X <- sin(2 * pi * t + rep(peak_phases, each = 365))
fd <- fdata(matrix(X, nrow = 1), argvals = t)

result <- analyze.peak.timing(fd, period = 1)
print(result$variability_score)  # Shows timing variability
#> [1] 0.1233248
```
