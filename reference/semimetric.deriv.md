# Semi-metric based on Derivatives

Computes a semi-metric based on the Lp distance of the nderiv-th
derivative of functional data.

## Usage

``` r
semimetric.deriv(fdataobj, fdataref = NULL, nderiv = 1, lp = 2, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, uses fdataobj.

- nderiv:

  Derivative order (1, 2, ...). Default is 1.

- lp:

  The p in Lp metric. Default is 2 (L2 distance).

- ...:

  Additional arguments passed to deriv.

## Value

A distance matrix based on derivative distances.

## Examples

``` r
# Create smooth curves
t <- seq(0, 2*pi, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(t + i/5)
fd <- fdata(X, argvals = t)

# Compute distance based on first derivative
D <- semimetric.deriv(fd, nderiv = 1)
```
