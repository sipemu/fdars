# Cosine Kernel

Cosine Kernel

## Usage

``` r
Ker.cos(u)
```

## Arguments

- u:

  Numeric vector of evaluation points.

## Value

Kernel values at u (0 outside \[-1, 1\]).

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Ker.cos(u), type = "l", main = "Cosine Kernel")
```
