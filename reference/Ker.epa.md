# Epanechnikov Kernel

Epanechnikov Kernel

## Usage

``` r
Ker.epa(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[-1, 1\]).

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Ker.epa(u), type = "l", main = "Epanechnikov Kernel")
```
