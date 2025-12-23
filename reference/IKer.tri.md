# Integrated Triweight Kernel

Integral of Ker.tri from -1 to u.

## Usage

``` r
IKer.tri(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, IKer.tri(u), type = "l", main = "Integrated Triweight Kernel")
```
