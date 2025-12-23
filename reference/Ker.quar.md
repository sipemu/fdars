# Quartic (Biweight) Kernel

Quartic (Biweight) Kernel

## Usage

``` r
Ker.quar(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[-1, 1\]).

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Ker.quar(u), type = "l", main = "Quartic Kernel")
```
