# Basis Representation Functions for Functional Data

Functions for representing functional data using basis expansions,
including B-splines, Fourier bases, and P-splines with penalization.
Reconstruct Functional Data from Basis Coefficients

## Usage

``` r
basis2fdata(
  coefs,
  argvals,
  nbasis = NULL,
  type = c("bspline", "fourier"),
  rangeval = NULL
)
```

## Arguments

- coefs:

  Coefficient matrix `[n x nbasis]` where n is the number of curves and
  nbasis is the number of basis functions. Can also be a vector for a
  single curve.

- argvals:

  Numeric vector of evaluation points for reconstruction.

- nbasis:

  Number of basis functions. If NULL, inferred from ncol(coefs).

- type:

  Basis type: "bspline" (default) or "fourier".

- rangeval:

  Range of argvals. Default: `range(argvals)`.

## Value

An fdata object with reconstructed curves.

## Details

Given basis coefficients, reconstruct the functional data by evaluating
the basis expansion at specified argument values.

The reconstruction computes `X(t) = sum(coef_k * B_k(t))` where `B_k`
are the basis functions evaluated at `argvals`.

## Examples

``` r
# Create some functional data
t <- seq(0, 1, length.out = 100)
X <- matrix(sin(2 * pi * t), nrow = 1)
fd <- fdata(X, argvals = t)

# Project to basis and reconstruct
coefs <- fdata2basis(fd, nbasis = 15, type = "fourier")
fd_recon <- basis2fdata(coefs, argvals = t, type = "fourier")
```
