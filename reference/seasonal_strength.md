# Measure Seasonal Strength

Computes the strength of seasonality in functional data. Values range
from 0 (no seasonality) to 1 (pure seasonal signal).

## Usage

``` r
seasonal_strength(
  fdataobj,
  period = NULL,
  method = c("variance", "spectral"),
  n_harmonics = 3
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

- n_harmonics:

  Number of Fourier harmonics to use (for variance method). Default: 3.

## Value

A numeric value between 0 and 1 representing seasonal strength.

## Details

The variance method decomposes the signal into a seasonal component
(using Fourier basis with the specified period) and computes the
proportion of variance explained by the seasonal component.

The spectral method computes the proportion of total spectral power that
falls at the seasonal frequency and its harmonics.

## Examples

``` r
# Pure seasonal signal
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2), nrow = 1)
fd_seasonal <- fdata(X, argvals = t)
seasonal_strength(fd_seasonal, period = 2)  # Should be close to 1
#> [1] 1

# Pure noise
X_noise <- matrix(rnorm(200), nrow = 1)
fd_noise <- fdata(X_noise, argvals = t)
seasonal_strength(fd_noise, period = 2)  # Should be close to 0
#> [1] 0.02783907
```
