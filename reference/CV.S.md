# Cross-Validation for Smoother Selection

Compute leave-one-out cross-validation criterion for a smoother.

## Usage

``` r
CV.S(S.type, tt, h, y, Ker = "norm", w = NULL)
```

## Arguments

- S.type:

  Function to compute smoother matrix (e.g., S.NW, S.LLR).

- tt:

  Evaluation points.

- h:

  Bandwidth parameter.

- y:

  Response vector to smooth.

- Ker:

  Kernel type.

- w:

  Optional weights.

## Value

The cross-validation score (mean squared prediction error).

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
y <- sin(2 * pi * tt) + rnorm(50, sd = 0.1)
cv_score <- CV.S(S.NW, tt, h = 0.1, y = y)
```
