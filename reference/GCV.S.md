# Generalized Cross-Validation for Smoother Selection

Compute GCV criterion: RSS / (1 - tr(S)/n)^2

## Usage

``` r
GCV.S(S.type, tt, h, y, Ker = "norm", w = NULL)
```

## Arguments

- S.type:

  Function to compute smoother matrix.

- tt:

  Evaluation points.

- h:

  Bandwidth parameter.

- y:

  Response vector.

- Ker:

  Kernel type.

- w:

  Optional weights.

## Value

The GCV score.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
y <- sin(2 * pi * tt) + rnorm(50, sd = 0.1)
gcv_score <- GCV.S(S.NW, tt, h = 0.1, y = y)
```
