# Kullback-Leibler Divergence Metric for Functional Data

Computes the symmetric Kullback-Leibler divergence between functional
data treated as probability distributions. Curves are first normalized
to be valid probability density functions.

## Usage

``` r
metric.kl(fdataobj, fdataref = NULL, eps = 1e-10, normalize = TRUE, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, computes self-distances.

- eps:

  Small value for numerical stability (default 1e-10).

- normalize:

  Logical. If TRUE (default), curves are shifted to be non-negative and
  normalized to integrate to 1.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix based on symmetric KL divergence.

## Details

The symmetric KL divergence is computed as: \$\$D\_{KL}(f, g) =
\frac{1}{2}\[KL(f\|\|g) + KL(g\|\|f)\]\$\$ where \$\$KL(f\|\|g) = \int
f(t) \log\frac{f(t)}{g(t)} dt\$\$

When `normalize = TRUE`, curves are first shifted to be non-negative (by
subtracting the minimum and adding eps), then normalized to integrate
to 1. This makes them valid probability density functions.

The symmetric KL divergence is always non-negative and equals zero only
when the two distributions are identical. However, it does not satisfy
the triangle inequality.

## Examples

``` r
# Create curves that look like probability densities
t <- seq(0, 1, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) {
  # Shifted Gaussian-like curves
  X[i, ] <- exp(-(t - 0.3 - i/50)^2 / 0.02) + rnorm(100, sd = 0.01)
}
fd <- fdata(X, argvals = t)

# Compute KL divergence
D <- metric.kl(fd)
```
