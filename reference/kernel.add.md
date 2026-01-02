# Add Covariance Functions

Combines two covariance functions by addition.

## Usage

``` r
kernel.add(kernel1, kernel2)
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
# Combine Gaussian with white noise
k_signal <- kernel.gaussian(variance = 1, length_scale = 0.2)
k_noise <- kernel.whitenoise(variance = 0.1)
k_total <- kernel.add(k_signal, k_noise)

t <- seq(0, 1, length.out = 50)
fd <- make.gaussian.process(n = 5, t = t, cov = k_total)
```
