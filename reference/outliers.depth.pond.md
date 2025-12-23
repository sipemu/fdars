# Outlier Detection for Functional Data

Functions for detecting outliers in functional data using depth
measures. Outlier Detection using Weighted Depth

## Usage

``` r
outliers.depth.pond(fdataobj, nb = 200, dfunc = depth.mode, quan = 0.5, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nb:

  Number of bootstrap samples. Default is 200.

- dfunc:

  Depth function to use. Default is depth.mode.

- quan:

  Quantile for outlier cutoff. Default is 0.5.

- ...:

  Additional arguments passed to depth function.

## Value

A list of class 'outliers.fdata' with components:

- outliers:

  Indices of detected outliers

- depths:

  Depth values for all curves

- cutoff:

  Depth cutoff used

## Details

Detects outliers based on depth with bootstrap resampling.

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))
out <- outliers.depth.pond(fd, nb = 50)
```
