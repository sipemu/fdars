# Predict Method for Nonparametric Functional Regression (fregre.np)

Predictions from a fitted nonparametric functional regression model.

## Usage

``` r
# S3 method for class 'fregre.np'
predict(object, newdata = NULL, ...)
```

## Arguments

- object:

  A fitted model object of class 'fregre.np'.

- newdata:

  An fdata object containing new functional data for prediction. If
  NULL, returns fitted values from training data.

- ...:

  Additional arguments (ignored).

## Value

A numeric vector of predicted values.

## Examples

``` r
# Create functional data
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:30) X[i, ] <- sin(2*pi*t) * i/30 + rnorm(50, sd = 0.1)
y <- rowMeans(X) + rnorm(30, sd = 0.1)
fd <- fdata(X, argvals = t)

# Fit model
fit <- fregre.np(fd, y)

# Predict on new data
X_new <- matrix(sin(2*pi*t) * 0.5, nrow = 1)
fd_new <- fdata(X_new, argvals = t)
predict(fit, fd_new)
#> [1] -0.005759715
```
