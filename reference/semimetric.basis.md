# Semi-metric based on Basis Expansion

Computes a semi-metric based on the L2 distance of basis expansion
coefficients. Supports B-spline and Fourier basis.

## Usage

``` r
semimetric.basis(
  fdataobj,
  fdataref = NULL,
  nbasis = 5,
  basis = "bspline",
  nderiv = 0,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, uses fdataobj.

- nbasis:

  Number of basis functions. Default is 5.

- basis:

  Type of basis: "bspline" (default) or "fourier".

- nderiv:

  Derivative order to compute distance on (default 0).

- ...:

  Additional arguments (ignored).

## Value

A distance matrix based on basis coefficients.

## Examples

``` r
# Create curves
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(2*pi*t + i/5) + rnorm(100, sd = 0.1)
fd <- fdata(X, argvals = t)

# Compute distance based on B-spline coefficients
D <- semimetric.basis(fd, nbasis = 7)
```
