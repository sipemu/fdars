# Outlier Detection for Functional Data

Functions for detecting outliers in functional data using depth
measures. Outlier Detection using Weighted Depth

## Usage

``` r
outliers.depth.pond(
  fdataobj,
  nb = 200,
  dfunc = depth.mode,
  threshold_method = c("quantile", "mad", "iqr"),
  quan = 0.05,
  k = NULL,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nb:

  Number of bootstrap samples. Default is 200.

- dfunc:

  Depth function to use. Default is depth.mode.

- threshold_method:

  Method for computing the outlier threshold. Options:

  "quantile"

  :   Use quantile of weighted depths (default). Curves with depth below
      this quantile are flagged as outliers.

  "mad"

  :   Use median - k \* MAD of weighted depths. More robust to existing
      outliers in the data.

  "iqr"

  :   Use Q1 - k \* IQR, similar to boxplot whiskers.

- quan:

  Quantile for outlier cutoff when `threshold_method = "quantile"`.
  Default is 0.05, meaning curves with depth in the bottom 5% are
  flagged (95th percentile threshold). Lower values detect fewer
  outliers.

- k:

  Multiplier for MAD or IQR methods. Default is 2.5 for MAD and 1.5 for
  IQR. Higher values detect fewer outliers.

- ...:

  Additional arguments passed to depth function.

## Value

A list of class 'outliers.fdata' with components:

- outliers:

  Indices of detected outliers

- depths:

  Depth values for all curves

- weighted_depths:

  Bootstrap-weighted depth values

- cutoff:

  Depth cutoff used

- threshold_method:

  Method used for threshold computation

- fdataobj:

  Original fdata object

## Details

Detects outliers based on depth with bootstrap resampling. The threshold
for outlier detection can be computed using different methods.

The function first computes depth values for all curves, then uses
bootstrap resampling to obtain weighted depths that are more robust to
sampling variability.

**Threshold Methods:**

- **quantile**: Flags curves with depth below the specified quantile.
  With `quan = 0.1`, approximately 10% of curves would be flagged under
  the null hypothesis of no outliers. Suitable when you expect a
  specific proportion of outliers.

- **mad**: Uses `median(depths) - k * MAD(depths)` as threshold. More
  robust because MAD is not influenced by extreme values. With k = 2.5,
  this corresponds roughly to a 1-2% false positive rate under
  normality.

- **iqr**: Uses `Q1 - k * IQR` as threshold, similar to boxplot outlier
  detection. With k = 1.5, corresponds to the standard boxplot fence.

## Examples

``` r
# Create data with outliers
set.seed(42)
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:28) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
X[29, ] <- sin(2*pi*t) + 3  # outlier
X[30, ] <- -sin(2*pi*t)     # outlier
fd <- fdata(X, argvals = t)

# Default: quantile method with 95th percentile (bottom 5%)
out1 <- outliers.depth.pond(fd, nb = 50)

# More permissive: bottom 10%
out1b <- outliers.depth.pond(fd, nb = 50, quan = 0.1)

# MAD method (more robust)
out2 <- outliers.depth.pond(fd, nb = 50, threshold_method = "mad", k = 2.5)

# IQR method (boxplot-like)
out3 <- outliers.depth.pond(fd, nb = 50, threshold_method = "iqr", k = 1.5)
```
