# Local Polynomial Regression Smoother Matrix

Compute the Local Polynomial Regression smoother matrix of degree p.
Special cases: p=0 is Nadaraya-Watson, p=1 is Local Linear Regression.

## Usage

``` r
S.LPR(tt, h, p = 1, Ker = "norm", w = NULL, cv = FALSE)
```

## Arguments

- tt:

  Evaluation points (numeric vector).

- h:

  Bandwidth parameter.

- p:

  Polynomial degree (default 1 for local linear).

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
S <- S.LPR(tt, h = 0.1, p = 2)  # Local quadratic regression
```
