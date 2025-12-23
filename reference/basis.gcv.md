# GCV Score for Basis Representation

Computes the Generalized Cross-Validation score for a basis
representation. Lower GCV indicates better fit with appropriate
complexity.

## Usage

``` r
basis.gcv(
  fdataobj,
  nbasis,
  type = c("bspline", "fourier"),
  lambda = 0,
  pooled = TRUE
)
```

## Arguments

- fdataobj:

  An fdata object.

- nbasis:

  Number of basis functions.

- type:

  Basis type: "bspline" (default) or "fourier".

- lambda:

  Smoothing/penalty parameter (default 0, no penalty).

- pooled:

  Logical. If TRUE (default), compute a single GCV across all curves. If
  FALSE, compute GCV for each curve and return the mean.

## Value

The GCV score (scalar).

## Details

GCV is computed as: \$\$GCV = \frac{RSS/n}{(1 - edf/n)^2}\$\$ where RSS
is the residual sum of squares and edf is the effective degrees of
freedom (trace of the hat matrix).

When `pooled = TRUE`, the criterion is computed globally across all
curves. When `pooled = FALSE`, the criterion is computed for each curve
separately and the mean is returned. Use `pooled = FALSE` when curves
have heterogeneous noise levels.

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(sin(2 * pi * t) + rnorm(50, sd = 0.1), nrow = 1)
fd <- fdata(X, argvals = t)

# Compare GCV for different nbasis
gcv_5 <- basis.gcv(fd, nbasis = 5)
gcv_10 <- basis.gcv(fd, nbasis = 10)
gcv_20 <- basis.gcv(fd, nbasis = 20)
```
