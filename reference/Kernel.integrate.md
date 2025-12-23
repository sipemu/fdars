# Unified Integrated Kernel Interface

Evaluates an integrated kernel function by name.

## Usage

``` r
Kernel.integrate(u, Ker = "norm", a = -1)
```

## Arguments

- u:

  Numeric vector of evaluation points.

- Ker:

  Kernel type: "norm", "epa", "tri", "quar", "cos", or "unif".

- a:

  Lower integration bound (default -1 for symmetric kernels). Not
  currently used (always integrates from -1 or -Inf).

## Value

Cumulative integral values.

## Examples

``` r
u <- seq(-1.5, 1.5, length.out = 100)
plot(u, Kernel.integrate(u, "epa"), type = "l")
```
