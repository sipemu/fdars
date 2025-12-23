# Cross-Validation for Functional Basis Regression

Performs k-fold cross-validation to select the optimal regularization
parameter (lambda) for functional basis regression.

## Usage

``` r
fregre.basis.cv(fdataobj, y, kfold = 10, lambda.range = NULL, seed = NULL, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- kfold:

  Number of folds for cross-validation (default 10).

- lambda.range:

  Range of lambda values to try. Default is 10^seq(-4, 4, length.out =
  20).

- seed:

  Random seed for fold assignment.

- ...:

  Additional arguments passed to fregre.basis.

## Value

A list with components:

- optimal.lambda:

  Optimal regularization parameter

- cv.errors:

  Mean squared prediction error for each lambda

- cv.se:

  Standard error of cv.errors

- model:

  Fitted model with optimal lambda
