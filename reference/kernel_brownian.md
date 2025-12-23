# Brownian Motion Covariance Function

Computes the Brownian motion (Wiener process) covariance function:
\$\$k(s, t) = \sigma^2 \min(s, t)\$\$

## Usage

``` r
kernel_brownian(variance = 1)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

## Value

A covariance function object of class 'kernel_brownian'.

## Details

The Brownian motion covariance produces sample paths that start at 0 and
have independent increments. The covariance between two points equals
the variance times the minimum of their positions.

This covariance is only defined for 1D domains starting at 0.

## See also

[`kernel_gaussian`](https://sipemu.github.io/fdars/reference/kernel_gaussian.md),
[`make_gaussian_process`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)

## Examples

``` r
# Generate Brownian motion paths
cov_func <- kernel_brownian(variance = 1)
t <- seq(0, 1, length.out = 100)
fd <- make_gaussian_process(n = 10, t = t, cov = cov_func)
plot(fd)
```
