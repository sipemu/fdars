# Cross-Validation for Functional PC Regression

Performs k-fold cross-validation to select the optimal number of
principal components for functional PC regression.

## Usage

``` r
fregre.pc.cv(fdataobj, y, kfold = 10, ncomp.range = NULL, seed = NULL, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- kfold:

  Number of folds for cross-validation (default 10).

- ncomp.range:

  Range of number of components to try. Default is 1 to min(n-1,
  ncol(data)).

- seed:

  Random seed for fold assignment.

- ...:

  Additional arguments passed to fregre.pc.

## Value

A list with components:

- optimal.ncomp:

  Optimal number of components

- cv.errors:

  Mean squared prediction error for each ncomp

- cv.se:

  Standard error of cv.errors

- model:

  Fitted model with optimal ncomp

## Examples

``` r
# Create functional data with a linear relationship
set.seed(42)
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 100, 50)
for (i in 1:100) X[i, ] <- sin(2*pi*t) * i/100 + rnorm(50, sd = 0.1)
beta_true <- cos(2*pi*t)
y <- X %*% beta_true + rnorm(100, sd = 0.5)
fd <- fdata(X, argvals = t)

# Cross-validate to find optimal number of PCs
cv_result <- fregre.pc.cv(fd, y, ncomp.range = 1:10)
```
