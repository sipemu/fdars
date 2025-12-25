# Remove Trend from Functional Data

Removes trend from functional data using various methods. This is useful
for preprocessing data before seasonal analysis when the data has a
significant trend component.

## Usage

``` r
detrend(
  fdataobj,
  method = c("linear", "polynomial", "diff1", "diff2", "loess", "auto"),
  degree = 2,
  bandwidth = 0.3,
  return_trend = FALSE
)
```

## Arguments

- fdataobj:

  An fdata object.

- method:

  Detrending method:

  "linear"

  :   Least squares linear fit (default)

  "polynomial"

  :   Polynomial regression of specified degree

  "diff1"

  :   First-order differencing

  "diff2"

  :   Second-order differencing

  "loess"

  :   Local polynomial regression (LOESS)

  "auto"

  :   Automatic selection via AIC

- degree:

  Polynomial degree for "polynomial" method. Default: 2.

- bandwidth:

  Bandwidth as fraction of data range for "loess" method. Default: 0.3.

- return_trend:

  Logical. If TRUE, return both trend and detrended data. Default:
  FALSE.

## Value

If return_trend = FALSE, an fdata object with detrended data. If
return_trend = TRUE, a list with components:

- detrended:

  fdata object with detrended data

- trend:

  fdata object with estimated trend

- method:

  Method used for detrending

- rss:

  Residual sum of squares per curve

## Details

For series with polynomial trends, "linear" or "polynomial" methods are
appropriate. For more complex trends, "loess" provides flexibility. The
"auto" method compares linear, polynomial (degree 2 and 3), and LOESS,
selecting the method with lowest AIC.

Differencing methods ("diff1", "diff2") reduce the series length by 1 or
2 points respectively. The resulting fdata has correspondingly shorter
argvals.

## See also

[`decompose`](https://sipemu.github.io/fdars/reference/decompose.md) for
full seasonal decomposition

## Examples

``` r
# Generate data with linear trend and seasonal component
t <- seq(0, 10, length.out = 200)
X <- matrix(2 + 0.5 * t + sin(2 * pi * t / 2), nrow = 1)
fd <- fdata(X, argvals = t)

# Detrend with linear method
fd_detrended <- detrend(fd, method = "linear")

# Now estimate period on detrended data
period <- estimate_period(fd_detrended)
print(period$period)  # Should be close to 2
#> [1] 2.01005

# Get both trend and detrended data
result <- detrend(fd, method = "linear", return_trend = TRUE)
# plot(result$trend)  # Shows the linear trend
```
