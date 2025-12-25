# LRT-based Outlier Detection for Functional Data

Detects outliers using the Likelihood Ratio Test approach based on
Febrero-Bande et al. Uses bootstrap to estimate a threshold and
iteratively removes curves exceeding this threshold. Implemented in Rust
for high performance with parallelized bootstrap.

## Usage

``` r
outliers.lrt(
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

  Number of bootstrap replications for threshold estimation (default
  200).

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

A list of class 'outliers.fdata' with components:

- outliers:

  Indices of detected outliers

- distances:

  Normalized distances for all curves

- threshold:

  Bootstrap threshold used

- percentile:

  Percentile used for threshold

- fdataobj:

  Original fdata object

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:30) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
# Add an outlier
X[1, ] <- X[1, ] + 3
fd <- fdata(X, argvals = t)
out <- outliers.lrt(fd, nb = 100)

# More sensitive detection
out_sensitive <- outliers.lrt(fd, nb = 100, percentile = 0.95)
```
