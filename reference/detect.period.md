# Seasonal Analysis Functions for Functional Data

Functions for analyzing seasonal patterns in functional data including
period estimation, peak detection, seasonal strength measurement, and
detection of seasonality changes. Detect Period with Multiple Methods

## Usage

``` r
detect.period(
  fdataobj,
  method = c("sazed", "autoperiod", "cfd", "fft", "acf"),
  ...
)
```

## Arguments

- fdataobj:

  An fdata object.

- method:

  Detection method to use:

  "sazed"

  :   SAZED ensemble (default) - parameter-free, most robust

  "autoperiod"

  :   Autoperiod - hybrid FFT + ACF with gradient ascent

  "cfd"

  :   CFDAutoperiod - differencing + clustering + ACF validation

  "fft"

  :   Simple FFT periodogram peak

  "acf"

  :   Simple ACF peak detection

- ...:

  Additional arguments passed to the underlying method:

  tolerance

  :   For SAZED: relative tolerance for voting (default 0.1)

  n_candidates

  :   For Autoperiod: max FFT peaks to consider (default 5)

  gradient_steps

  :   For Autoperiod: refinement steps (default 10)

  cluster_tolerance

  :   For CFD: clustering tolerance (default 0.1)

  min_cluster_size

  :   For CFD: minimum cluster size (default 1)

  max_lag

  :   For ACF: maximum lag to search

  detrend_method

  :   For FFT/ACF: "none", "linear", or "auto"

## Value

A period detection result. The exact class depends on the method:

- sazed_result:

  For SAZED method

- autoperiod_result:

  For Autoperiod method

- cfd_autoperiod_result:

  For CFD method

- period_estimate:

  For FFT and ACF methods

## Details

Unified interface for period detection that dispatches to specialized
algorithms based on the chosen method. Provides a single entry point for
all period detection functionality.

**Method selection guidance:**

- **sazed**: Best general-purpose choice. Combines 5 methods and uses
  voting - robust across signal types with no tuning needed.

- **autoperiod**: Good when you need candidate details and
  gradient-refined estimates. Slightly more precise than SAZED.

- **cfd**: Best for signals with strong polynomial trends. Also detects
  multiple concurrent periods.

- **fft**: Fastest, but sensitive to noise and trends.

- **acf**: Simple but effective for clean periodic signals.

## See also

[`sazed`](https://sipemu.github.io/fdars/reference/sazed.md),
[`autoperiod`](https://sipemu.github.io/fdars/reference/autoperiod.md),
[`cfd.autoperiod`](https://sipemu.github.io/fdars/reference/cfd.autoperiod.md),
[`estimate.period`](https://sipemu.github.io/fdars/reference/estimate.period.md),
[`detect.periods`](https://sipemu.github.io/fdars/reference/detect.periods.md)

## Examples

``` r
# Generate seasonal data
t <- seq(0, 20, length.out = 400)
X <- matrix(sin(2 * pi * t / 2) + 0.1 * rnorm(400), nrow = 1)
fd <- fdata(X, argvals = t)

# Default (SAZED) - most robust
result <- detect.period(fd)
print(result$period)
#> [1] 2.003616

# Autoperiod with custom settings
result <- detect.period(fd, method = "autoperiod", n_candidates = 10)

# CFDAutoperiod for trended data
X_trend <- matrix(0.3 * t + sin(2 * pi * t / 2), nrow = 1)
fd_trend <- fdata(X_trend, argvals = t)
result <- detect.period(fd_trend, method = "cfd")

# Simple FFT for speed
result <- detect.period(fd, method = "fft")
```
