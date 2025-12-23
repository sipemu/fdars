# BIC for Basis Representation

Computes the Bayesian Information Criterion for a basis representation.
BIC penalizes complexity more strongly than AIC for larger samples.

## Usage

``` r
basis.bic(
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

  Logical. If TRUE (default), compute a single BIC across all curves. If
  FALSE, compute BIC for each curve and return the mean.

## Value

The BIC value (scalar).

## Details

BIC is computed as: \$\$BIC = n \log(RSS/n) + \log(n) \cdot edf\$\$

When `pooled = TRUE`, the criterion uses total observations and total
effective degrees of freedom (n_curves \* edf). When `pooled = FALSE`,
the criterion is computed for each curve separately and the mean is
returned.
