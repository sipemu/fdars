# Exponential Covariance Function

Computes the exponential covariance function: \$\$k(s, t) = \sigma^2
\exp\left(-\frac{\|s-t\|}{\ell}\right)\$\$

## Usage

``` r
kernel.exponential(variance = 1, length_scale = 1)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

- length_scale:

  Length scale parameter \\\ell\\ (default 1).

## Value

A covariance function object of class 'kernel_exponential'.

## Details

This is equivalent to the Matern covariance with \\\nu = 0.5\\. Sample
paths are continuous but not differentiable (rough).

The exponential covariance function produces sample paths that are
continuous but nowhere differentiable, resulting in rough-looking
curves. It is a special case of the Matern family with \\\nu = 0.5\\.

## See also

[`kernel.gaussian`](https://sipemu.github.io/fdars/reference/kernel.gaussian.md),
[`kernel.matern`](https://sipemu.github.io/fdars/reference/kernel.matern.md),
[`make.gaussian.process`](https://sipemu.github.io/fdars/reference/make.gaussian.process.md)

## Examples

``` r
# Create an exponential covariance function
cov_func <- kernel.exponential(variance = 1, length_scale = 0.2)

# Evaluate covariance matrix
t <- seq(0, 1, length.out = 50)
K <- cov_func(t)

# Generate rough GP samples
fd <- make.gaussian.process(n = 10, t = t, cov = cov_func)
plot(fd)
```
