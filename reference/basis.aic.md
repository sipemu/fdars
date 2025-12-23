# AIC for Basis Representation

Computes the Akaike Information Criterion for a basis representation.
Lower AIC indicates better model (balancing fit and complexity).

## Usage

``` r
basis.aic(
  fdataobj,
  nbasis,
  type = c("bspline", "fourier"),
  lambda = 0,
  pooled = TRUE
)
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

- pooled:

  Logical. If TRUE (default), compute a single AIC across all curves. If
  FALSE, compute AIC for each curve and return the mean.

## Value

The AIC value (scalar).

## Details

AIC is computed as: \$\$AIC = n \log(RSS/n) + 2 \cdot edf\$\$

When `pooled = TRUE`, the criterion uses total observations and total
effective degrees of freedom (n_curves \* edf). When `pooled = FALSE`,
the criterion is computed for each curve separately and the mean is
returned.
