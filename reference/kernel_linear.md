# Linear Covariance Function

Computes the linear covariance function: \$\$k(s, t) = \sigma^2 (s -
c)(t - c)\$\$

## Usage

``` r
kernel_linear(variance = 1, offset = 0)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

- offset:

  Offset parameter \\c\\ (default 0).

## Value

A covariance function object of class 'kernel_linear'.

## Details

The linear covariance function produces sample paths that are linear
functions. It is useful when the underlying process is expected to have
a linear trend.

## See also

[`kernel_polynomial`](https://sipemu.github.io/fdars/reference/kernel_polynomial.md),
[`make_gaussian_process`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)

## Examples

``` r
# Generate linear function samples
cov_func <- kernel_linear(variance = 1)
t <- seq(0, 1, length.out = 50)
fd <- make_gaussian_process(n = 10, t = t, cov = cov_func)
plot(fd)
```
