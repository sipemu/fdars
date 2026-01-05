# Detect Changes in Seasonality

Detects points in time where seasonality starts (onset) or stops
(cessation) by monitoring time-varying seasonal strength.

## Usage

``` r
detect.seasonality.changes(
  fdataobj,
  period,
  threshold = 0.3,
  window_size = NULL,
  min_duration = NULL
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known or estimated period.

- threshold:

  Seasonal strength threshold for classification (0-1). Above threshold
  = seasonal, below = non-seasonal. Default: 0.3.

- window_size:

  Width of sliding window for strength estimation. Default: 2 \* period.

- min_duration:

  Minimum duration to confirm a change. Prevents detection of spurious
  short-term fluctuations. Default: period.

## Value

A list with components:

- change_points:

  Data frame with columns: time, type ("onset" or "cessation"),
  strength_before, strength_after

- strength_curve:

  Time-varying seasonal strength used for detection

## Examples

``` r
# Signal that starts non-seasonal, becomes seasonal, then stops
t <- seq(0, 30, length.out = 600)
X <- ifelse(t < 10, rnorm(sum(t < 10), sd = 0.3),
            ifelse(t < 20, sin(2 * pi * t[t >= 10 & t < 20] / 2),
                   rnorm(sum(t >= 20), sd = 0.3)))
X <- matrix(X, nrow = 1)
fd <- fdata(X, argvals = t)

# Detect changes
changes <- detect.seasonality.changes(fd, period = 2)
print(changes$change_points)  # Should show onset ~10, cessation ~20
#>        time      type strength_before strength_after
#> 1  8.664441     onset       0.2830366      0.3494510
#> 2 20.183639 cessation       0.3052235      0.2953254
```
