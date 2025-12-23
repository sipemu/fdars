# Local Cubic Regression Smoother Matrix

Convenience function for Local Polynomial Regression with degree 3.

## Usage

``` r
S.LCR(tt, h, Ker = "norm", w = NULL, cv = FALSE)
```

## Arguments

- tt:

  Evaluation points (numeric vector).

- h:

  Bandwidth parameter.

- Ker:

  Kernel function or name.

- w:

  Optional weights vector.

- cv:

  Logical. If TRUE, compute leave-one-out cross-validation matrix.

## Value

An n x n smoother matrix S.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
S <- S.LCR(tt, h = 0.15)
```
