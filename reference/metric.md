# Distance Metrics for Functional Data

Functions for computing various distance metrics between functional
data. Compute Distance Metric for Functional Data

## Usage

``` r
metric(fdataobj, fdataref = NULL, method = "lp", ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, computes self-distances.

- method:

  Distance method to use. One of:

  - "lp" - Lp metric (default)

  - "hausdorff" - Hausdorff distance

  - "dtw" - Dynamic Time Warping

  - "pca" - Semi-metric based on PCA scores

  - "deriv" - Semi-metric based on derivatives

  - "basis" - Semi-metric based on basis coefficients

  - "fourier" - Semi-metric based on FFT coefficients

  - "hshift" - Semi-metric with horizontal shift

  - "kl" - Symmetric Kullback-Leibler divergence

- ...:

  Additional arguments passed to the specific distance function.

## Value

A distance matrix.

## Details

Unified interface for computing various distance metrics between
functional data objects. This function dispatches to the appropriate
specialized distance function based on the method parameter.

This function provides a convenient unified interface for all distance
computations in fdars. The additional arguments in `...` are passed to
the underlying distance function:

- lp: lp, w

- hausdorff: (none)

- dtw: p, w

- pca: ncomp

- deriv: nderiv, lp

- basis: nbasis, basis, nderiv

- fourier: nfreq

- hshift: max_shift

- kl: eps, normalize

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))

# Using different distance methods
D_lp <- metric(fd, method = "lp")
D_hausdorff <- metric(fd, method = "hausdorff")
D_pca <- metric(fd, method = "pca", ncomp = 3)

# Cross-distances
fd2 <- fdata(matrix(rnorm(100), 10, 10))
D_cross <- metric(fd, fd2, method = "lp")
```
