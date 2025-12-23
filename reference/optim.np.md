# Optimize Bandwidth Using Cross-Validation

Find the optimal bandwidth by minimizing CV or GCV criterion.

## Usage

``` r
optim.np(
  fdataobj,
  S.type,
  h.range = NULL,
  criterion = "GCV",
  Ker = "norm",
  ...
)
```

## Arguments

- fdataobj:

  An fdata object.

- S.type:

  Smoother function (S.NW, S.LLR, etc.).

- h.range:

  Range of bandwidths to search (default: data-driven).

- criterion:

  "CV" or "GCV".

- Ker:

  Kernel type.

- ...:

  Additional arguments passed to optimizer.

## Value

A list with optimal bandwidth and CV/GCV score.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
y <- sin(2 * pi * tt) + rnorm(50, sd = 0.1)
fd <- fdata(matrix(y, nrow = 1), argvals = tt)
result <- optim.np(fd, S.NW)
```
