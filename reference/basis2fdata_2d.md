# Reconstruct 2D Functional Data from Tensor Product Basis Coefficients

Reconstructs 2D surfaces from tensor product basis coefficients.

## Usage

``` r
basis2fdata_2d(
  coefs,
  argvals,
  nbasis.s,
  nbasis.t,
  type = c("bspline", "fourier")
)
```

## Arguments

- coefs:

  Coefficient matrix `[n x (nbasis.s * nbasis.t)]`.

- argvals:

  List with two numeric vectors for s and t coordinates.

- nbasis.s:

  Number of basis functions in s direction.

- nbasis.t:

  Number of basis functions in t direction.

- type:

  Basis type: "bspline" (default) or "fourier".

## Value

A 2D fdata object.
