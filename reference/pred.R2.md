# R-Squared (Coefficient of Determination)

Compute the R-squared value between predicted and actual values.

## Usage

``` r
pred.R2(y_true, y_pred)
```

## Arguments

- y_true:

  Actual values.

- y_pred:

  Predicted values.

## Value

The R-squared value.

## Examples

``` r
y_true <- c(1, 2, 3, 4, 5)
y_pred <- c(1.1, 2.2, 2.9, 4.1, 4.8)
pred.R2(y_true, y_pred)
#> [1] 0.989
```
