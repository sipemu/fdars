# Hausdorff Metric for Functional Data

Computes the Hausdorff distance between functional data objects. The
Hausdorff distance treats each curve as a set of points (t, f(t)) in 2D
space and computes the maximum of the minimum distances.

## Usage

``` r
metric.hausdorff(fdata1, fdata2 = NULL, ...)
```

## Arguments

- fdata1:

  An object of class 'fdata'.

- fdata2:

  An object of class 'fdata'. If NULL, uses fdata1.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
D <- metric.hausdorff(fd)
```
