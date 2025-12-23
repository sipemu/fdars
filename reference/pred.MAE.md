# Mean Absolute Error

Compute the Mean Absolute Error between predicted and actual values.

## Usage

``` r
pred.MAE(y_true, y_pred)
```

## Arguments

- y_true:

  Actual values.

- y_pred:

  Predicted values.

## Value

The mean absolute error.

## Examples

``` r
y_true <- c(1, 2, 3, 4, 5)
y_pred <- c(1.1, 2.2, 2.9, 4.1, 4.8)
pred.MAE(y_true, y_pred)
#> [1] 0.14
```
