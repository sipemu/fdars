# Polynomial Covariance Function

Computes the polynomial covariance function: \$\$k(s, t) = \sigma^2 (s
\cdot t + c)^p\$\$

## Usage

``` r
kernel.polynomial(variance = 1, offset = 0, degree = 2)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

- offset:

  Offset parameter \\c\\ (default 0).

- degree:

  Polynomial degree \\p\\ (default 2).

## Value

A covariance function object of class 'kernel_polynomial'.

## Details

The polynomial covariance function produces sample paths that are
polynomial functions of degree at most `degree`. Setting `degree = 1`
and `offset = 0` gives the linear kernel.

## See also

[`kernel.linear`](https://sipemu.github.io/fdars/reference/kernel.linear.md),
[`make.gaussian.process`](https://sipemu.github.io/fdars/reference/make.gaussian.process.md)

## Examples

``` r
# Generate quadratic function samples
cov_func <- kernel.polynomial(degree = 2, offset = 1)
t <- seq(0, 1, length.out = 50)
fd <- make.gaussian.process(n = 10, t = t, cov = cov_func)
plot(fd)
```
