# LRT Outlier Detection Threshold

Computes the bootstrap threshold for LRT-based outlier detection. This
is a highly parallelized Rust implementation providing significant
speedup over pure R implementations.

## Usage

``` r
outliers.thres.lrt(
  fdataobj,
  nb = 200,
  smo = 0.05,
  trim = 0.1,
  seed = NULL,
  percentile = 0.99
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nb:

  Number of bootstrap replications (default 200).

- smo:

  Smoothing parameter for bootstrap noise (default 0.05).

- trim:

  Proportion of curves to trim for robust estimation (default 0.1).

- seed:

  Random seed for reproducibility.

- percentile:

  Percentile of bootstrap distribution to use as threshold (default
  0.99, meaning 99th percentile). Lower values make detection more
  sensitive (detect more outliers).

## Value

The threshold value at the specified percentile.

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:30) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)
thresh <- outliers.thres.lrt(fd, nb = 100)

# More sensitive detection (95th percentile)
thresh_sensitive <- outliers.thres.lrt(fd, nb = 100, percentile = 0.95)
```
