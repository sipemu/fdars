# Integrated Normal Kernel

Integrated Normal Kernel

## Usage

``` r
IKer.norm(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Cumulative integral of normal kernel from -Inf to u.

## Examples

``` r
u <- seq(-3, 3, length.out = 100)
plot(u, IKer.norm(u), type = "l", main = "Integrated Normal Kernel")
```
