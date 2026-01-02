# Classify Seasonality Type

Classifies the type of seasonality in functional data. Particularly
useful for short series (3-5 years) to identify stable vs variable
timing patterns.

## Usage

``` r
classify.seasonality(
  fdataobj,
  period,
  strength_threshold = NULL,
  timing_threshold = NULL
)
```

## Arguments

- fdataobj:

  An fdata object.

- period:

  Known seasonal period.

- strength_threshold:

  Threshold for seasonal/non-seasonal (default: 0.3).

- timing_threshold:

  Max std of normalized timing for "stable" (default: 0.05).

## Value

A list with components:

- is_seasonal:

  Logical: is the series seasonal overall?

- has_stable_timing:

  Logical: is peak timing stable across cycles?

- timing_variability:

  Timing variability score (0-1)

- seasonal_strength:

  Overall seasonal strength

- cycle_strengths:

  Per-cycle seasonal strength

- weak_seasons:

  Indices of weak/missing seasons (0-indexed)

- classification:

  One of: "StableSeasonal", "VariableTiming", "IntermittentSeasonal",
  "NonSeasonal"

- peak_timing:

  Peak timing analysis (if peaks detected)

## Details

Classification types:

- StableSeasonal: Regular peaks with consistent timing

- VariableTiming: Regular peaks but timing shifts between cycles

- IntermittentSeasonal: Some cycles seasonal, some not

- NonSeasonal: No clear seasonality

## Examples

``` r
# Pure seasonal signal
t <- seq(0, 10, length.out = 500)
X <- matrix(sin(2 * pi * t / 2), nrow = 1)
fd <- fdata(X, argvals = t)

result <- classify.seasonality(fd, period = 2)
print(result$classification)  # "StableSeasonal"
#> [1] "StableSeasonal"
```
