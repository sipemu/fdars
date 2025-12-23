# Matern Covariance Function

Computes the Matern covariance function with smoothness parameter
\\\nu\\: \$\$k(s, t) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)}
\left(\sqrt{2\nu}\frac{\|s-t\|}{\ell}\right)^\nu
K\_\nu\left(\sqrt{2\nu}\frac{\|s-t\|}{\ell}\right)\$\$

## Usage

``` r
kernel_matern(variance = 1, length_scale = 1, nu = 1.5)
```

## Arguments

- variance:

  Variance parameter \\\sigma^2\\ (default 1).

- length_scale:

  Length scale parameter \\\ell\\ (default 1).

- nu:

  Smoothness parameter \\\nu\\ (default 1.5). Common values:

  - `nu = 0.5`: Exponential (continuous, not differentiable)

  - `nu = 1.5`: Once differentiable

  - `nu = 2.5`: Twice differentiable

  - `nu = Inf`: Gaussian/squared exponential (infinitely differentiable)

## Value

A covariance function object of class 'kernel_matern'.

## Details

where \\K\_\nu\\ is the modified Bessel function of the second kind.

The Matern family of covariance functions provides flexible control over
the smoothness of sample paths through the \\\nu\\ parameter. As \\\nu\\
increases, sample paths become smoother. The Matern family includes the
exponential (\\\nu = 0.5\\) and approaches the Gaussian kernel as \\\nu
\to \infty\\.

For computational efficiency, special cases \\\nu \in \\0.5, 1.5, 2.5,
\infty\\\\ use simplified closed-form expressions.

## See also

[`kernel_gaussian`](https://sipemu.github.io/fdars/reference/kernel_gaussian.md),
[`kernel_exponential`](https://sipemu.github.io/fdars/reference/kernel_exponential.md),
[`make_gaussian_process`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)

## Examples

``` r
# Create Matern covariance functions with different smoothness
cov_rough <- kernel_matern(nu = 0.5)    # Equivalent to exponential
cov_smooth <- kernel_matern(nu = 2.5)   # Twice differentiable

t <- seq(0, 1, length.out = 50)

# Compare sample paths
fd_rough <- make_gaussian_process(n = 5, t = t, cov = cov_rough, seed = 42)
fd_smooth <- make_gaussian_process(n = 5, t = t, cov = cov_smooth, seed = 42)
```
