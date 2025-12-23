# Triweight Kernel

Triweight Kernel

## Usage

``` r
Ker.tri(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[-1, 1\]).

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Ker.tri(u), type = "l", main = "Triweight Kernel")
```
