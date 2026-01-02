# Linear Covariance Function

Computes the linear covariance function: \$\$k(s, t) = \sigma^2 (s -
c)(t - c)\$\$

## Usage

``` r
kernel.linear(variance = 1, offset = 0)
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

[`kernel.polynomial`](https://sipemu.github.io/fdars/reference/kernel.polynomial.md),
[`make.gaussian.process`](https://sipemu.github.io/fdars/reference/make.gaussian.process.md)

## Examples

``` r
# Generate linear function samples
cov_func <- kernel.linear(variance = 1)
t <- seq(0, 1, length.out = 50)
fd <- make.gaussian.process(n = 10, t = t, cov = cov_func)
plot(fd)
```
