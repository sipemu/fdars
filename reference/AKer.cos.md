# Asymmetric Cosine Kernel

Asymmetric Cosine Kernel

## Usage

``` r
AKer.cos(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 for u \< 0).

## Examples

``` r
u <- seq(-0.5, 1.5, length.out = 100)
plot(u, AKer.cos(u), type = "l", main = "Asymmetric Cosine Kernel")
```
