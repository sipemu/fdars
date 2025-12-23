# Integrated Epanechnikov Kernel

Integral of Ker.epa from -1 to u.

## Usage

``` r
IKer.epa(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, IKer.epa(u), type = "l", main = "Integrated Epanechnikov Kernel")
```
