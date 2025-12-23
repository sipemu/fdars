# Integrated Quartic Kernel

Integral of Ker.quar from -1 to u.

## Usage

``` r
IKer.quar(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, IKer.quar(u), type = "l", main = "Integrated Quartic Kernel")
```
