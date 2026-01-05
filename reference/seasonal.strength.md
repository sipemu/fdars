# Measure Seasonal Strength

Computes the strength of seasonality in functional data. Values range
from 0 (no seasonality) to 1 (pure seasonal signal).

## Usage

``` r
seasonal.strength(
  fdataobj,
  period = NULL,
  method = c("variance", "spectral", "wavelet"),
  n_harmonics = 3,
  detrend_method = c("none", "linear", "auto")
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known or estimated period. If NULL, period is estimated automatically
  using FFT.

- method:

  Method for computing strength:

  "variance"

  :   Variance decomposition: Var(seasonal) / Var(total)

  "spectral"

  :   Spectral: power at seasonal frequencies / total power

  "wavelet"

  :   Wavelet: Morlet wavelet power at seasonal period / total variance

- n_harmonics:

  Number of Fourier harmonics to use (for variance method). Default: 3.

- detrend_method:

  Detrending method to apply before computing strength:

  "none"

  :   No detrending (default)

  "linear"

  :   Remove linear trend

  "auto"

  :   Automatic AIC-based selection of detrending method

## Value

A numeric value between 0 and 1 representing seasonal strength.

## Details

The variance method decomposes the signal into a seasonal component
(using Fourier basis with the specified period) and computes the
proportion of variance explained by the seasonal component.

The spectral method computes the proportion of total spectral power that
falls at the seasonal frequency and its harmonics.

The wavelet method uses a Morlet wavelet to measure power at the target
period. It provides time-localized frequency information and is robust
to non-stationary signals.

Trends can artificially lower the seasonal strength measure by
contributing non-seasonal variance. Use detrend_method to remove trends
before computing strength.

## Examples

``` r
# Pure seasonal signal
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2), nrow = 1)
fd_seasonal <- fdata(X, argvals = t)
seasonal.strength(fd_seasonal, period = 2)  # Should be close to 1
#> [1] 1

# Pure noise
X_noise <- matrix(rnorm(200), nrow = 1)
fd_noise <- fdata(X_noise, argvals = t)
seasonal.strength(fd_noise, period = 2)  # Should be close to 0
#> [1] 0.02096016

# Trending data - detrending improves strength estimate
X_trend <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
fd_trend <- fdata(X_trend, argvals = t)
seasonal.strength(fd_trend, period = 2, detrend_method = "linear")
#> [1] 0.9763512
```
