# Asymmetric Epanechnikov Kernel

Asymmetric Epanechnikov Kernel

## Usage

``` r
AKer.epa(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[0, 1\]).

## Examples

``` r
u <- seq(-0.5, 1.5, length.out = 100)
plot(u, AKer.epa(u), type = "l", main = "Asymmetric Epanechnikov Kernel")
```
