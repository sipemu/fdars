# Utility Functions for Functional Data Analysis

Various utility functions including integration, inner products, random
process generation, and prediction metrics. Simpson's Rule Integration

## Usage

``` r
# S3 method for class 'irregFdata'
int.simpson(x, ...)

int.simpson(x, ...)

# S3 method for class 'fdata'
int.simpson(x, ...)
```

## Arguments

- x:

  A functional data object (`fdata` or `irregFdata`).

- ...:

  Additional arguments passed to methods.

## Value

A numeric vector of integrals, one per curve.

## Details

Integrate functional data over its domain using Simpson's rule
(composite trapezoidal rule for non-uniform grids). Works with both
regular `fdata` and irregular `irregFdata` objects.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(2*pi*t)
fd <- fdata(X, argvals = t)
integrals <- int.simpson(fd)  # Should be approximately 0

# Also works with irregular data
ifd <- sparsify(fd, minObs = 20, maxObs = 50, seed = 123)
integrals_irreg <- int.simpson(ifd)
```
