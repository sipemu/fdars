# Periodic Covariance Function

Computes the periodic covariance function: \$\$k(s, t) = \sigma^2
\exp\left(-\frac{2\sin^2(\pi\|s-t\|/p)}{\ell^2}\right)\$\$

## Usage

``` r
kernel_periodic(variance = 1, length_scale = 1, period = 1)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

- length_scale:

  Length scale parameter \\\ell\\ (default 1).

- period:

  Period parameter \\p\\ (default 1).

## Value

A covariance function object of class 'kernel_periodic'.

## Details

The periodic covariance function produces sample paths that are periodic
with the specified period. It is useful for modeling seasonal or
cyclical patterns in functional data.

## See also

[`kernel_gaussian`](https://sipemu.github.io/fdars/reference/kernel_gaussian.md),
[`make_gaussian_process`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)

## Examples

``` r
# Generate periodic function samples
cov_func <- kernel_periodic(period = 0.5, length_scale = 0.5)
t <- seq(0, 2, length.out = 100)
fd <- make_gaussian_process(n = 5, t = t, cov = cov_func)
plot(fd)
```
