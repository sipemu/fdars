# Mean Squared Error

Compute the Mean Squared Error between predicted and actual values.

## Usage

``` r
pred.MSE(y_true, y_pred)
```

## Arguments

- y_true:

  Actual values.

- y_pred:

  Predicted values.

## Value

The mean squared error.

## Examples

``` r
y_true <- c(1, 2, 3, 4, 5)
y_pred <- c(1.1, 2.2, 2.9, 4.1, 4.8)
pred.MSE(y_true, y_pred)
#> [1] 0.022
```
