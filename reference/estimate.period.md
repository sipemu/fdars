# Estimate Seasonal Period using FFT

Estimates the dominant period in functional data using Fast Fourier
Transform and periodogram analysis.

## Usage

``` r
estimate.period(
  fdataobj,
  method = c("fft", "acf"),
  max_lag = NULL,
  detrend_method = c("none", "linear", "auto")
)
```

## Arguments

- fdataobj:

  An fdata object.

- method:

  Method for period estimation: "fft" (Fast Fourier Transform, default)
  or "acf" (autocorrelation function).

- max_lag:

  Maximum lag for ACF method. Default: half the series length.

- detrend_method:

  Detrending method to apply before period estimation:

  "none"

  :   No detrending (default)

  "linear"

  :   Remove linear trend

  "auto"

  :   Automatic AIC-based selection of detrending method

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

The function computes the periodogram of the mean curve and finds the
frequency with maximum power. The confidence measure indicates how
pronounced the dominant frequency is relative to the background.

For data with trends, the detrend_method parameter can significantly
improve period estimation accuracy. Strong trends can mask the true
seasonal period.

## Examples

``` r
# Generate seasonal data with period = 2
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2) + rnorm(200, sd = 0.1), nrow = 1)
fd <- fdata(X, argvals = t)

# Estimate period
result <- estimate.period(fd, method = "fft")
print(result$period)  # Should be close to 2
#> [1] 2.01005

# With trend - detrending improves estimation
X_trend <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
fd_trend <- fdata(X_trend, argvals = t)
result <- estimate.period(fd_trend, detrend_method = "linear")
```
