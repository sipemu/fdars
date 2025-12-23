# Functional Covariance Function

Computes the covariance function (surface) for functional data. For 1D:
`Cov(s, t) = E[(X(s) - mu(s))(X(t) - mu(t))]` For 2D: Covariance across
the flattened domain.

## Usage

``` r
cov(fdataobj, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ...:

  Additional arguments (currently ignored).

## Value

A list with components:

- cov:

  The covariance matrix (m x m for 1D, (m1*m2) x (m1*m2) for 2D)

- argvals:

  The evaluation points (same as input)

- mean:

  The mean function

## Examples

``` r
# 1D functional data
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
fd <- fdata(X, argvals = t)
cov_result <- cov(fd)
image(cov_result$cov, main = "Covariance Surface")
```
