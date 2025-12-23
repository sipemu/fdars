# Unified Asymmetric Kernel Interface

Evaluates an asymmetric kernel function by name.

## Usage

``` r
Kernel.asymmetric(u, type.Ker = "norm")
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
u <- seq(-0.5, 1.5, length.out = 100)
plot(u, Kernel.asymmetric(u, "epa"), type = "l")
```
