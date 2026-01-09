# Cross-Validation for Basis Function Number Selection

Selects the optimal number of basis functions using k-fold
cross-validation or generalized cross-validation.

## Usage

``` r
fdata2basis_cv(
  fdataobj,
  nbasis.range = 4:20,
  type = c("bspline", "fourier"),
  criterion = c("GCV", "CV", "AIC", "BIC"),
  kfold = 10,
  lambda = 0
)
```

## Arguments

- fdataobj:

  An fdata object.

- nbasis.range:

  Vector of nbasis values to evaluate (default: 4:20).

- type:

  Basis type: "bspline" (default) or "fourier".

- criterion:

  Selection criterion: "GCV" (default), "CV", "AIC", or "BIC".

- kfold:

  Number of folds for k-fold CV (default 10). Ignored if criterion is
  "GCV", "AIC", or "BIC".

- lambda:

  Smoothing parameter (default 0).

## Value

A list with:

- optimal.nbasis:

  Optimal number of basis functions

- scores:

  Score for each nbasis value

- nbasis.range:

  The tested nbasis values

- criterion:

  The criterion used

- fitted:

  fdata object fitted with optimal nbasis

## Examples

``` r
set.seed(42)
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(4 * pi * t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)

# Find optimal nbasis
cv_result <- fdata2basis_cv(fd, nbasis.range = 5:15, type = "fourier")
print(cv_result$optimal.nbasis)
#> [1] 5
```
