# Convert Functional Data to Basis Coefficients

Project functional data onto a basis system and return coefficients.
Supports B-spline and Fourier basis.

## Usage

``` r
fdata2basis(fdataobj, nbasis = 10, type = c("bspline", "fourier"))
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nbasis:

  Number of basis functions (default 10).

- type:

  Type of basis: "bspline" (default) or "fourier".

## Value

A matrix of coefficients (n x nbasis).

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)
coefs <- fdata2basis(fd, nbasis = 10, type = "bspline")
```
