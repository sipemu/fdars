# Seasonal Analysis Functions for Functional Data

Functions for analyzing seasonal patterns in functional data including
period estimation, peak detection, seasonal strength measurement, and
detection of seasonality changes. Estimate Seasonal Period using FFT

## Usage

``` r
estimate_period(fdataobj, method = c("fft", "acf"), max_lag = NULL)
```

## Arguments

- fdataobj:

  An fdata object.

## Value

A list with components:

- period:

  Estimated period

- frequency:

  Dominant frequency (1/period)

- power:

  Power at the dominant frequency

- confidence:

  Confidence measure (ratio of peak power to mean power)

## Details

Estimates the dominant period in functional data using Fast Fourier
Transform and periodogram analysis.

The function computes the periodogram of the mean curve and finds the
frequency with maximum power. The confidence measure indicates how
pronounced the dominant frequency is relative to the background.

## Examples

``` r
# Generate seasonal data with period = 2
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2) + rnorm(200, sd = 0.1), nrow = 1)
fd <- fdata(X, argvals = t)

# Estimate period
result <- estimate_period(fd, method = "fft")
print(result$period)  # Should be close to 2
#> [1] 2.01005
```
