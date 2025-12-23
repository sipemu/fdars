# Smoothing Functions for Functional Data

Functions for computing smoothing matrices and applying kernel smoothing
to functional data. Nadaraya-Watson Kernel Smoother Matrix

## Usage

``` r
S.NW(tt, h, Ker = "norm", w = NULL, cv = FALSE)
```

## Arguments

- tt:

  Evaluation points (numeric vector).

- h:

  Bandwidth parameter.

- Ker:

  Kernel function or name. One of "norm", "epa", "tri", "quar", "cos",
  "unif", or a custom function.

- w:

  Optional weights vector of length n.

- cv:

  Logical. If TRUE, compute leave-one-out cross-validation matrix
  (diagonal is zero).

## Value

An n x n smoother matrix S such that smooth(y) = S %\*% y.

## Details

Compute the Nadaraya-Watson kernel smoother matrix.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
S <- S.NW(tt, h = 0.1)
dim(S)  # 50 x 50
#> [1] 50 50
```
