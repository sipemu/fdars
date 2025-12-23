# Integrated Cosine Kernel

Integral of Ker.cos from -1 to u.

## Usage

``` r
IKer.cos(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, IKer.cos(u), type = "l", main = "Integrated Cosine Kernel")
```
