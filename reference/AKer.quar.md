# Asymmetric Quartic Kernel

Asymmetric Quartic Kernel

## Usage

``` r
AKer.quar(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[0, 1\]).

## Examples

``` r
u <- seq(-0.5, 1.5, length.out = 100)
plot(u, AKer.quar(u), type = "l", main = "Asymmetric Quartic Kernel")
```
