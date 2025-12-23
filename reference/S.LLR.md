# Local Linear Regression Smoother Matrix

Compute the Local Linear Regression (LLR) smoother matrix. LLR has
better boundary bias properties than Nadaraya-Watson.

## Usage

``` r
S.LLR(tt, h, Ker = "norm", w = NULL, cv = FALSE)
```

## Arguments

- tt:

  Evaluation points (numeric vector).

- h:

  Bandwidth parameter.

- Ker:

  Kernel function or name. One of "norm", "epa", "tri", "quar", "cos",
  "unif".

- w:

  Optional weights vector of length n.

- cv:

  Logical. If TRUE, compute leave-one-out cross-validation matrix.

## Value

An n x n smoother matrix S.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
S <- S.LLR(tt, h = 0.1)
```
