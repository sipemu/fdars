# Integrated Uniform Kernel

Integral of Ker.unif from -1 to u.

## Usage

``` r
IKer.unif(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, IKer.unif(u), type = "l", main = "Integrated Uniform Kernel")
```
