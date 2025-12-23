# Functional Standard Deviation

Computes the pointwise standard deviation function of functional data.

## Usage

``` r
sd(fdataobj, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ...:

  Additional arguments (currently ignored).

## Value

An fdata object containing the standard deviation function (1D or 2D).

## Examples

``` r
# 1D functional data
fd <- fdata(matrix(rnorm(100), 10, 10))
s <- sd(fd)

# 2D functional data
X <- array(rnorm(500), dim = c(5, 10, 10))
fd2d <- fdata(X, argvals = list(1:10, 1:10), fdata2d = TRUE)
s2d <- sd(fd2d)
```
