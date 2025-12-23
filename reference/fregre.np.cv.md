# Cross-Validation for Nonparametric Functional Regression

Performs k-fold cross-validation to select the optimal bandwidth
parameter (h) for nonparametric functional regression.

## Usage

``` r
fregre.np.cv(
  fdataobj,
  y,
  kfold = 10,
  h.range = NULL,
  metric = metric.lp,
  seed = NULL,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- kfold:

  Number of folds for cross-validation (default 10).

- h.range:

  Range of bandwidth values to try. If NULL, automatically determined
  from the distance matrix.

- metric:

  Distance metric function. Default is metric.lp.

- seed:

  Random seed for fold assignment.

- ...:

  Additional arguments passed to the metric function.

## Value

A list with components:

- optimal.h:

  Optimal bandwidth parameter

- cv.errors:

  Mean squared prediction error for each h

- cv.se:

  Standard error of cv.errors

- model:

  Fitted model with optimal h
