# Default Bandwidth

Compute a default bandwidth as the 15th percentile of pairwise
distances.

## Usage

``` r
h.default(fdataobj, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata', or a numeric vector of evaluation points.

- ...:

  Additional arguments (ignored).

## Value

A scalar bandwidth value.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
h <- h.default(tt)
```
