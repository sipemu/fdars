# Functional Basis Regression

Fits a functional linear model using basis expansion (ridge regression).
Uses the anofox-regression Rust backend for efficient L2-regularized
regression.

## Usage

``` r
fregre.basis(fdataobj, y, basis.x = NULL, basis.b = NULL, lambda = 0, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- basis.x:

  Basis for the functional covariate (currently ignored).

- basis.b:

  Basis for the coefficient function (currently ignored).

- lambda:

  Smoothing/regularization parameter (L2 penalty).

- ...:

  Additional arguments.

## Value

A fitted regression object of class 'fregre.fd' with components:

- coefficients:

  Beta coefficient function values

- intercept:

  Intercept term

- fitted.values:

  Fitted values

- residuals:

  Residuals

- lambda:

  Regularization parameter used

- r.squared:

  R-squared (coefficient of determination)

- mean.X:

  Mean of functional covariate (for prediction)

- mean.y:

  Mean of response (for prediction)

- sr2:

  Residual variance

- fdataobj:

  Original functional data

- y:

  Response vector

- call:

  The function call
