# Predict Method for Functional Regression (fregre.fd)

Predictions from a fitted functional regression model (fregre.pc or
fregre.basis).

## Usage

``` r
# S3 method for class 'fregre.fd'
predict(object, newdata = NULL, ...)
```

## Arguments

- object:

  A fitted model object of class 'fregre.fd'.

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
fit <- fregre.pc(fd, y, ncomp = 3)

# Predict on new data
X_new <- matrix(sin(2*pi*t) * 0.5, nrow = 1)
fd_new <- fdata(X_new, argvals = t)
predict(fit, fd_new)
#> [1] 0.008787298
```
