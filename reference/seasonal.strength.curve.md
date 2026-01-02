# Time-Varying Seasonal Strength

Computes seasonal strength at each time point using a sliding window,
allowing detection of how seasonality changes over time.

## Usage

``` r
seasonal.strength.curve(
  fdataobj,
  period,
  window_size = NULL,
  method = c("variance", "spectral")
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known or estimated period.

- window_size:

  Width of the sliding window. Recommended: 2 \* period.

- method:

  Method for computing strength: "variance" or "spectral".

## Value

An fdata object containing the time-varying seasonal strength curve.

## Examples

``` r
# Signal that transitions from seasonal to non-seasonal
t <- seq(0, 20, length.out = 400)
X <- ifelse(t < 10, sin(2 * pi * t / 2), rnorm(length(t[t >= 10]), sd = 0.5))
X <- matrix(X, nrow = 1)
fd <- fdata(X, argvals = t)

# Compute time-varying strength
ss <- seasonal.strength.curve(fd, period = 2, window_size = 4)
# plot(ss)  # Shows strength declining around t = 10
```
