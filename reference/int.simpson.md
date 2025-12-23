# Utility Functions for Functional Data Analysis

Various utility functions including integration, inner products, random
process generation, and prediction metrics. Simpson's Rule Integration

## Usage

``` r
int.simpson(fdataobj)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

## Value

A numeric vector of integrals, one per curve.

## Details

Integrate functional data over its domain using Simpson's rule
(composite trapezoidal rule for non-uniform grids).

## Examples

``` r
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(2*pi*t)
fd <- fdata(X, argvals = t)
integrals <- int.simpson(fd)  # Should be approximately 0
```
