# Unified Symmetric Kernel Interface

Evaluates a symmetric kernel function by name.

## Usage

``` r
Kernel(u, type.Ker = "norm")
```

## Arguments

- u:

  Numeric vector of evaluation points.

- type.Ker:

  Kernel type: "norm", "epa", "tri", "quar", "cos", or "unif".

## Value

Kernel values at u.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Kernel(u, "epa"), type = "l")
lines(u, Kernel(u, "norm"), col = "red")
```
