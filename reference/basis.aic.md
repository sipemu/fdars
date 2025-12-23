# AIC for Basis Representation

Computes the Akaike Information Criterion for a basis representation.
Lower AIC indicates better model (balancing fit and complexity).

## Usage

``` r
basis.aic(fdataobj, nbasis, type = c("bspline", "fourier"), lambda = 0)
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

The AIC value (scalar).

## Details

AIC is computed as: \$\$AIC = n \log(RSS/n) + 2 \cdot edf\$\$
