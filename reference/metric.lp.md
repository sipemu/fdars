# Lp Metric for Functional Data

Computes the Lp distance between functional data objects using numerical
integration (Simpson's rule).

## Usage

``` r
metric.lp(fdata1, fdata2 = NULL, lp = 2, w = 1, ...)
```

## Arguments

- fdata1:

  An object of class 'fdata'.

- fdata2:

  An object of class 'fdata'. If NULL, computes self-distances for
  fdata1 (more efficient symmetric computation).

- lp:

  The p in Lp metric. Default is 2 (L2 distance).

- w:

  Optional weight vector of length equal to number of evaluation points.
  Default is uniform weighting.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix of dimensions n1 x n2 (or n x n if fdata2 is NULL).

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
D <- metric.lp(fd)  # Self-distances

fd2 <- fdata(matrix(rnorm(50), 5, 10))
D2 <- metric.lp(fd, fd2)  # Cross-distances
```
