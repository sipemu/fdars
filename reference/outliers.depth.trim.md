# Outlier Detection using Trimmed Depth

Detects outliers based on depth trimming.

## Usage

``` r
outliers.depth.trim(fdataobj, trim = 0.1, dfunc = depth.mode, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- trim:

  Proportion of curves to consider as potential outliers. Default is 0.1
  (curves with depth in bottom 10%).

- dfunc:

  Depth function to use. Default is depth.mode.

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

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))
out <- outliers.depth.trim(fd, trim = 0.1)
```
