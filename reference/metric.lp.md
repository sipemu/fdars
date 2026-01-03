# Lp Metric for Functional Data

Computes the Lp distance between functional data objects using numerical
integration (Simpson's rule). Works with both regular `fdata` and
irregular `irregFdata` objects.

## Usage

``` r
# S3 method for class 'irregFdata'
metric.lp(x, p = 2, ...)

metric.lp(x, ...)

# S3 method for class 'fdata'
metric.lp(x, y = NULL, p = 2, w = 1, ...)
```

## Arguments

- x:

  A functional data object (`fdata` or `irregFdata`).

- p:

  The order of the Lp metric (default 2 for L2 distance).

- ...:

  Additional arguments passed to methods.

- y:

  An object of class 'fdata'. If NULL, computes self-distances for x
  (more efficient symmetric computation). Only supported for `fdata`.

- w:

  Optional weight vector of length equal to number of evaluation points.
  Default is uniform weighting. Only supported for `fdata`.

## Value

A distance matrix.

## Examples

``` r
# Regular fdata
fd <- fdata(matrix(rnorm(100), 10, 10))
D <- metric.lp(fd)  # Self-distances

# Irregular fdata
ifd <- sparsify(fd, minObs = 3, maxObs = 7, seed = 42)
D_irreg <- metric.lp(ifd)
```
