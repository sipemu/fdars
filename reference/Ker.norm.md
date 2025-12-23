# Kernel Functions

Symmetric, asymmetric, and integrated kernel functions for nonparametric
smoothing and density estimation. Normal (Gaussian) Kernel

## Usage

``` r
Ker.norm(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u.

## Examples

``` r
u <- seq(-3, 3, length.out = 100)
plot(u, Ker.norm(u), type = "l", main = "Normal Kernel")
```
