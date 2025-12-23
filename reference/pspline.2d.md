# P-spline Smoothing for 2D Functional Data

Fits 2D P-splines with anisotropic penalties in both directions.

## Usage

``` r
pspline.2d(
  fdataobj,
  nbasis.s = 10,
  nbasis.t = 10,
  lambda.s = 1,
  lambda.t = 1,
  order = 2,
  lambda.select = FALSE,
  criterion = c("GCV", "AIC", "BIC")
)
```

## Arguments

- fdataobj:

  A 2D fdata object.

- nbasis.s:

  Number of B-spline basis functions in s direction.

- nbasis.t:

  Number of B-spline basis functions in t direction.

- lambda.s:

  Smoothing parameter in s direction.

- lambda.t:

  Smoothing parameter in t direction.

- order:

  Order of the difference penalty (default 2).

- lambda.select:

  Logical. If TRUE, select lambdas automatically.

- criterion:

  Criterion for selection: "GCV", "AIC", or "BIC".

## Value

A list of class "pspline.2d" similar to
[`pspline()`](https://sipemu.github.io/fdars/reference/pspline.md).

## Details

The 2D penalty uses Kronecker product structure: \$\$P = \lambda_s (I_t
\otimes P_s) + \lambda_t (P_t \otimes I_s)\$\$
