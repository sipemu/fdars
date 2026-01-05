# Detect Seasonality Changes with Automatic Threshold

Detects points where seasonality starts or stops, using automatic
threshold selection instead of a fixed value.

## Usage

``` r
detect.seasonality.changes.auto(
  fdataobj,
  period,
  threshold_method = "otsu",
  threshold_value = NULL,
  window_size = NULL,
  min_duration = NULL
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known seasonal period.

- threshold_method:

  Method for threshold selection:

  "fixed"

  :   Use threshold_value as fixed threshold

  "percentile"

  :   Use threshold_value as percentile of strength distribution

  "otsu"

  :   Use Otsu's method for bimodal separation (default)

- threshold_value:

  Value for "fixed" or "percentile" methods.

- window_size:

  Width of sliding window for strength estimation. Default: 2 \* period.

- min_duration:

  Minimum duration to confirm a change. Default: period.

## Value

A list with components:

- change_points:

  Data frame with time, type, strength_before, strength_after

- strength_curve:

  Time-varying seasonal strength used for detection

- computed_threshold:

  The threshold that was computed/used

## Details

Otsu's method automatically finds the optimal threshold for separating
seasonal from non-seasonal regions based on the strength distribution.
This is particularly useful when you don't know the appropriate
threshold for your data.

## Examples

``` r
# Signal that transitions from seasonal to non-seasonal
t <- seq(0, 20, length.out = 400)
X <- ifelse(t < 10, sin(2 * pi * t / 2), rnorm(sum(t >= 10), sd = 0.3))
X <- matrix(X, nrow = 1)
fd <- fdata(X, argvals = t)

# Detect changes with Otsu threshold
changes <- detect.seasonality.changes.auto(fd, period = 2)
print(changes$computed_threshold)
#> [1] 0.516327
```
