# Hausdorff Metric for Functional Data

Computes the Hausdorff distance between functional data objects. The
Hausdorff distance treats each curve as a set of points (t, f(t)) in 2D
space and computes the maximum of the minimum distances.

## Usage

``` r
metric.hausdorff(fdataobj, fdataref = NULL, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, uses fdataobj.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
D <- metric.hausdorff(fd)
```
