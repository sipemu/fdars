# Detect Multiple Concurrent Periods

Detects multiple periodicities in functional data using iterative
residual subtraction. At each iteration, the dominant period is detected
using FFT, its sinusoidal component is subtracted, and the process
repeats on the residual until stopping criteria are met.

## Usage

``` r
detect.periods(
  fdataobj,
  max_periods = 3,
  min_confidence = 0.5,
  min_strength = 0.2,
  detrend_method = c("auto", "none", "linear")
)
```

## Arguments

- fdataobj:

  An fdata object.

- max_periods:

  Maximum number of periods to detect. Default: 3.

- min_confidence:

  Minimum FFT confidence to continue detection. Default: 0.5.

- min_strength:

  Minimum seasonal strength to continue detection. Default: 0.2.

- detrend_method:

  Detrending method to apply before period detection:

  "auto"

  :   Automatic AIC-based selection of detrending method (default)

  "none"

  :   No detrending

  "linear"

  :   Remove linear trend

## Value

A list with components:

- periods:

  Numeric vector of detected periods

- confidence:

  FFT confidence for each period

- strength:

  Seasonal strength for each period

- amplitude:

  Amplitude of the sinusoidal component

- phase:

  Phase of the sinusoidal component (radians)

- n_periods:

  Number of periods detected

## Details

The function uses two stopping criteria:

- FFT confidence: How prominent the dominant frequency is

- Seasonal strength: How much variance is explained by the periodicity

Both must exceed their thresholds for detection to continue. Higher
thresholds result in fewer (but more reliable) detected periods.

Periods are detected in order of amplitude (FFT power), not period
length. A weak yearly cycle will be detected after a strong weekly
cycle.

Trends can interfere with period detection. The default "auto"
detrending automatically selects an appropriate method to remove trends.

## See also

[`estimate.period`](https://sipemu.github.io/fdars/reference/estimate.period.md)
for single period estimation,
[`detrend`](https://sipemu.github.io/fdars/reference/detrend.md) for
standalone detrending

## Examples

``` r
# Signal with two periods: 2 and 7
t <- seq(0, 20, length.out = 400)
X <- sin(2 * pi * t / 2) + 0.6 * sin(2 * pi * t / 7)
fd <- fdata(matrix(X, nrow = 1), argvals = t)

# Detect multiple periods
result <- detect.periods(fd, max_periods = 3)
print(result$periods)  # Should find approximately 2 and 7
#> [1]  2.005013  6.683375 10.025063
print(result$n_periods)  # Should be 2
#> [1] 3
```
