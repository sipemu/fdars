# Functional Regression

Functions for functional regression models. Functional Principal
Component Regression

## Usage

``` r
fregre.pc(fdataobj, y, ncomp = NULL, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- ncomp:

  Number of principal components to use.

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

- ncomp:

  Number of components used

- mean.X:

  Mean of functional covariate (for prediction)

- mean.y:

  Mean of response (for prediction)

- rotation:

  PC loadings (for prediction)

- l:

  Indices of selected components

- lm:

  Underlying linear model

- sr2:

  Residual variance

- fdataobj:

  Original functional data

- y:

  Response vector

- call:

  The function call

## Details

Fits a functional linear model using principal component regression.
