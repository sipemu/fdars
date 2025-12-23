# BIC for Basis Representation

Computes the Bayesian Information Criterion for a basis representation.
BIC penalizes complexity more strongly than AIC for larger samples.

## Usage

``` r
basis.bic(fdataobj, nbasis, type = c("bspline", "fourier"), lambda = 0)
```

## Arguments

- fdataobj:

  An fdata object.

- nbasis:

  Number of basis functions.

- type:

  Basis type: "bspline" (default) or "fourier".

- lambda:

  Smoothing/penalty parameter (default 0).

## Value

The BIC value (scalar).

## Details

BIC is computed as: \$\$BIC = n \log(RSS/n) + \log(n) \cdot edf\$\$
