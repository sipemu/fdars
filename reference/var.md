# Functional Variance

Computes the pointwise variance function of functional data.

## Usage

``` r
var(fdataobj, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ...:

  Additional arguments (currently ignored).

## Value

An fdata object containing the variance function (1D or 2D).

## Examples

``` r
# 1D functional data
fd <- fdata(matrix(rnorm(100), 10, 10))
v <- var(fd)

# 2D functional data
X <- array(rnorm(500), dim = c(5, 10, 10))
fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
v2d <- var(fd2d)
```
