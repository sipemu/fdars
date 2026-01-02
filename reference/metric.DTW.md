# Dynamic Time Warping for Functional Data

Computes the Dynamic Time Warping distance between functional data. DTW
allows for non-linear alignment of curves.

## Usage

``` r
metric.DTW(fdataobj, fdataref = NULL, p = 2, w = NULL, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, computes self-distances.

- p:

  The p in Lp distance (default 2 for L2/Euclidean).

- w:

  Sakoe-Chiba window constraint. Default is min(ncol(fdataobj),
  ncol(fdataref)). Use -1 for no window constraint.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
D <- metric.DTW(fd)
```
