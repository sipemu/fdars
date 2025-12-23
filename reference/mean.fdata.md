# Compute functional mean

Computes the pointwise mean function across all observations. This is an
S3 method for the generic `mean` function.

## Usage

``` r
# S3 method for class 'fdata'
mean(x, ...)
```

## Arguments

- x:

  An object of class 'fdata'.

- ...:

  Additional arguments (currently ignored).

## Value

For 1D fdata: a numeric vector containing the mean function values. For
2D fdata: an fdata object containing the mean surface.

## Examples

``` r
# 1D functional data
fd <- fdata(matrix(rnorm(100), 10, 10))
fm <- mean(fd)

# 2D functional data
X <- array(rnorm(500), dim = c(5, 10, 10))
fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
fm2d <- mean(fd2d)
```
