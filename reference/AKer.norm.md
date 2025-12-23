# Asymmetric Normal Kernel

Asymmetric Normal Kernel

## Usage

``` r
AKer.norm(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 for u \< 0).

## Examples

``` r
u <- seq(-0.5, 3, length.out = 100)
plot(u, AKer.norm(u), type = "l", main = "Asymmetric Normal Kernel")
```
