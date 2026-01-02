# Seasonal-Trend Decomposition

Decomposes functional data into trend, seasonal, and remainder
components. Similar to STL (Seasonal-Trend decomposition using LOESS).

## Usage

``` r
decompose(
  fdataobj,
  period = NULL,
  method = c("additive", "multiplicative"),
  trend_method = c("loess", "spline"),
  bandwidth = 0.3,
  n_harmonics = 3
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Seasonal period. If NULL, estimated automatically using FFT.

- method:

  Decomposition method:

  "additive"

  :   data = trend + seasonal + remainder (default)

  "multiplicative"

  :   data = trend \* seasonal \* remainder

- trend_method:

  Method for trend extraction: "loess" or "spline". Default: "loess".

- bandwidth:

  Bandwidth for trend extraction (fraction of range). Default: 0.3.

- n_harmonics:

  Number of Fourier harmonics for seasonal component. Default: 3.

## Value

A list with components:

- trend:

  fdata object with trend component

- seasonal:

  fdata object with seasonal component

- remainder:

  fdata object with remainder/residual

- period:

  Period used for decomposition

- method:

  Decomposition method ("additive" or "multiplicative")

## Details

For additive decomposition: data = trend + seasonal + remainder. The
trend is extracted using LOESS or spline smoothing, then the seasonal
component is estimated by fitting Fourier harmonics to the detrended
data.

For multiplicative decomposition: data = trend \* seasonal \* remainder.
This is achieved by log-transforming the data, applying additive
decomposition, and back-transforming. Use this when the seasonal
amplitude grows with the trend level.

## See also

[`detrend`](https://sipemu.github.io/fdars/reference/detrend.md) for
simple trend removal,
[`seasonal.strength`](https://sipemu.github.io/fdars/reference/seasonal.strength.md)
for measuring seasonality

## Examples

``` r
# Additive seasonal pattern
t <- seq(0, 20, length.out = 400)
X <- matrix(2 + 0.3 * t + sin(2 * pi * t / 2.5), nrow = 1)
fd <- fdata(X, argvals = t)

result <- decompose(fd, period = 2.5, method = "additive")
# plot(result$trend)      # Linear trend
# plot(result$seasonal)   # Sinusoidal seasonal
# plot(result$remainder)  # Residual noise

# Multiplicative pattern (amplitude grows with level)
X_mult <- matrix((2 + 0.3 * t) * (1 + 0.3 * sin(2 * pi * t / 2.5)), nrow = 1)
fd_mult <- fdata(X_mult, argvals = t)

result_mult <- decompose(fd_mult, period = 2.5, method = "multiplicative")
```
