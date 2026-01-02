# Multiply Covariance Functions

Combines two covariance functions by multiplication.

## Usage

``` r
kernel.mult(kernel1, kernel2)
```

## Arguments

- kernel1:

  First covariance function.

- kernel2:

  Second covariance function.

## Value

A combined covariance function.

## Examples

``` r
# Multiply periodic with Gaussian for locally periodic behavior
k_periodic <- kernel.periodic(period = 0.3)
k_gaussian <- kernel.gaussian(length_scale = 0.5)
k_prod <- kernel.mult(k_periodic, k_gaussian)
```
