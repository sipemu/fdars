# SAZED: Spectral-ACF Zero-crossing Ensemble Detection

A parameter-free ensemble method for robust period detection that
combines five different detection approaches and uses majority voting to
determine the consensus period.

## Usage

``` r
sazed(fdataobj, tolerance = 0.1, detrend_method = c("none", "linear", "auto"))
```

## Arguments

- fdataobj:

  An fdata object.

- tolerance:

  Relative tolerance for considering periods equal when voting. Default:
  0.1 (10% relative difference). Use smaller values for stricter
  matching, larger values for more lenient matching.

- detrend_method:

  Detrending method to apply before period estimation:

  "none"

  :   No detrending (default)

  "linear"

  :   Remove linear trend

  "auto"

  :   Automatic AIC-based selection of detrending method

## Value

A list of class "sazed_result" with components:

- period:

  Consensus period (average of agreeing components)

- confidence:

  Confidence score (0-1, proportion of agreeing components)

- agreeing_components:

  Number of components that agreed on the period

- components:

  List of individual component estimates:

## Details

SAZED combines five detection methods:

1.  **Spectral**: Finds peaks in the FFT periodogram above the noise
    floor

2.  **ACF Peak**: Identifies the first significant peak in the
    autocorrelation function

3.  **ACF Average**: Computes a weighted mean of ACF peak locations

4.  **Zero-crossing**: Estimates period from ACF zero-crossing intervals

5.  **Spectral Diff**: Applies FFT to first-differenced signal (trend
    removal)

The final period is chosen by majority voting: periods within the
tolerance are grouped together, and the group with the most members
determines the consensus. The returned period is the average of the
agreeing estimates.

This method is particularly robust because:

- It requires no tuning parameters (tolerance has sensible defaults)

- Multiple methods must agree for high confidence

- Differencing component handles trends automatically

- Works well across different signal types

## See also

[`estimate.period`](https://sipemu.github.io/fdars/reference/estimate.period.md)
for single-method estimation, `detect.multiple.periods` for detecting
multiple concurrent periods

## Examples

``` r
# Generate seasonal data with period = 2
t <- seq(0, 20, length.out = 400)
X <- matrix(sin(2 * pi * t / 2) + 0.1 * rnorm(400), nrow = 1)
fd <- fdata(X, argvals = t)

# Detect period using SAZED
result <- sazed(fd)
print(result)  # Shows consensus period and component details
#> SAZED Period Detection
#> ----------------------
#> Period:     2.0029
#> Confidence: 1.00 (5/5 components agree)
#> 
#> Component estimates:
#>   Spectral:      2.0050
#>   ACF Peak:      2.0050
#>   ACF Average:   2.0036
#>   Zero-crossing: 2.0010
#>   Spectral Diff: 2.0000

# With trend - SAZED's spectral_diff component handles this
X_trend <- matrix(0.3 * t + sin(2 * pi * t / 2), nrow = 1)
fd_trend <- fdata(X_trend, argvals = t)
result <- sazed(fd_trend)
```
