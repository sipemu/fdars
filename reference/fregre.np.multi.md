# Nonparametric Regression with Multiple Functional Predictors

Fits a nonparametric regression model with multiple functional
predictors using a weighted combination of distance matrices.

## Usage

``` r
fregre.np.multi(
  fdataobj.list,
  y,
  weights = NULL,
  h = NULL,
  knn = NULL,
  type.S = c("S.NW", "kNN.gCV", "kNN.lCV"),
  Ker = "norm",
  metric = metric.lp,
  cv.grid = NULL,
  cv.folds = 5,
  ...
)
```

## Arguments

- fdataobj.list:

  A list of fdata objects (functional predictors). All must have the
  same number of observations.

- y:

  Response vector (scalar).

- weights:

  Weights for combining distances. Can be:

  - NULL: Equal weights (1/p for each predictor)

  - Numeric vector: Fixed weights (will be normalized to sum to 1)

  - "cv": Cross-validate to find optimal weights

- h:

  Bandwidth for Nadaraya-Watson kernel (optional).

- knn:

  Maximum k for k-NN methods.

- type.S:

  Smoother type: "S.NW", "kNN.gCV", or "kNN.lCV".

- Ker:

  Kernel type (default "norm" for Gaussian).

- metric:

  Distance metric function (default metric.lp).

- cv.grid:

  Grid of weight values for CV (only used if weights = "cv"). Default is
  seq(0, 1, by = 0.1) for 2 predictors.

- cv.folds:

  Number of folds for weight CV (default 5).

- ...:

  Additional arguments passed to metric function.

## Value

An object of class 'fregre.np.multi' containing:

- fdataobj.list:

  List of functional predictors

- y:

  Response vector

- weights:

  Weights used (or optimized)

- weights.cv:

  CV results if weights = "cv"

- fitted.values:

  Fitted values

- residuals:

  Residuals

- D.list:

  List of distance matrices

- D.combined:

  Combined distance matrix

## Examples

``` r
# Create two functional predictors
set.seed(42)
n <- 50
m <- 30
t_grid <- seq(0, 1, length.out = m)

X1 <- matrix(0, n, m)
X2 <- matrix(0, n, m)
for (i in 1:n) {
  X1[i, ] <- sin(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
  X2[i, ] <- cos(2 * pi * t_grid) * i/n + rnorm(m, sd = 0.1)
}
y <- rowMeans(X1) + 0.5 * rowMeans(X2) + rnorm(n, sd = 0.1)

fd1 <- fdata(X1, argvals = t_grid)
fd2 <- fdata(X2, argvals = t_grid)

# Fit with equal weights
fit1 <- fregre.np.multi(list(fd1, fd2), y)

# Fit with cross-validated weights
fit2 <- fregre.np.multi(list(fd1, fd2), y, weights = "cv")
```
