# Asymmetric Triweight Kernel

Asymmetric Triweight Kernel

## Usage

``` r
AKer.tri(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[0, 1\]).

## Examples

``` r
u <- seq(-0.5, 1.5, length.out = 100)
plot(u, AKer.tri(u), type = "l", main = "Asymmetric Triweight Kernel")
```
