# Covariance Functions and Gaussian Process Generation

## Introduction

Covariance functions (also called kernels) are fundamental building
blocks for Gaussian processes. They define the correlation structure
between function values at different points, controlling properties like
smoothness and periodicity.

**fdars** provides a comprehensive set of covariance functions for
generating synthetic functional data from Gaussian processes.

``` r
library(fdars)
#> 
#> Attaching package: 'fdars'
#> The following objects are masked from 'package:stats':
#> 
#>     cov, deriv, median, sd, var
#> The following object is masked from 'package:base':
#> 
#>     norm
library(ggplot2)
```

## Available Covariance Functions

### Gaussian (Squared Exponential)

The most common choice, producing infinitely differentiable (very
smooth) sample paths:
$$k(s,t) = \sigma^{2}\exp\left( - \frac{(s - t)^{2}}{2\ell^{2}} \right)$$

``` r
t <- seq(0, 1, length.out = 100)

# Create Gaussian covariance function
cov_gauss <- kernel_gaussian(variance = 1, length_scale = 0.2)
print(cov_gauss)
#> Covariance Kernel: gaussian 
#> Parameters:
#>    variance = 1 
#>    length_scale = 0.2

# Generate smooth GP samples
fd_gauss <- make_gaussian_process(n = 10, t = t, cov = cov_gauss, seed = 42)
plot(fd_gauss, main = "Gaussian Covariance (smooth)")
```

![](covariance-functions_files/figure-html/cov-gaussian-1.png)

The `length_scale` parameter controls the correlation distance - smaller
values produce more rapidly varying functions:

``` r
# Generate data for different length scales
ls_values <- c(0.05, 0.2, 0.5)
df_ls <- do.call(rbind, lapply(ls_values, function(ls) {
  fd <- make_gaussian_process(n = 5, t = t,
                              cov = kernel_gaussian(length_scale = ls),
                              seed = 42)
  data.frame(
    t = rep(t, 5),
    value = as.vector(t(fd$data)),
    curve = rep(1:5, each = length(t)),
    length_scale = paste("length_scale =", ls)
  )
}))
df_ls$length_scale <- factor(df_ls$length_scale,
                              levels = paste("length_scale =", ls_values))

ggplot(df_ls, aes(x = t, y = value, group = curve, color = factor(curve))) +
  geom_line(alpha = 0.8) +
  facet_wrap(~ length_scale) +
  labs(x = "t", y = "X(t)", title = "Effect of Length Scale") +
  theme_minimal() +
  theme(legend.position = "none")
```

![](covariance-functions_files/figure-html/length-scale-1.png)

### Exponential

Produces rougher paths than Gaussian (continuous but not
differentiable):
$$k(s,t) = \sigma^{2}\exp\left( - \frac{|s - t|}{\ell} \right)$$

``` r
cov_exp <- kernel_exponential(variance = 1, length_scale = 0.2)
fd_exp <- make_gaussian_process(n = 10, t = t, cov = cov_exp, seed = 42)
plot(fd_exp, main = "Exponential Covariance (rough)")
```

![](covariance-functions_files/figure-html/cov-exponential-1.png)

### Matern Family

The Matern covariance is parameterized by smoothness parameter $\nu$. It
interpolates between Exponential ($\nu = 0.5$) and Gaussian
($\left. \nu\rightarrow\infty \right.$):

``` r
# Generate data for different nu values
nu_values <- c(0.5, 1.5, 2.5, Inf)
nu_labels <- c("0.5", "1.5", "2.5", "Inf")
df_matern <- do.call(rbind, lapply(seq_along(nu_values), function(i) {
  fd <- make_gaussian_process(n = 5, t = t,
                              cov = kernel_matern(length_scale = 0.2, nu = nu_values[i]),
                              seed = 42)
  data.frame(
    t = rep(t, 5),
    value = as.vector(t(fd$data)),
    curve = rep(1:5, each = length(t)),
    nu = paste("Matern nu =", nu_labels[i])
  )
}))
df_matern$nu <- factor(df_matern$nu,
                        levels = paste("Matern nu =", nu_labels))

ggplot(df_matern, aes(x = t, y = value, group = curve, color = factor(curve))) +
  geom_line(alpha = 0.8) +
  facet_wrap(~ nu, ncol = 2) +
  labs(x = "t", y = "X(t)", title = "Matern Covariance with Different Smoothness") +
  theme_minimal() +
  theme(legend.position = "none")
```

![](covariance-functions_files/figure-html/cov-matern-1.png)

Common choices are: - $\nu = 0.5$: Equivalent to Exponential (rough) -
$\nu = 1.5$: Once differentiable - $\nu = 2.5$: Twice differentiable -
$\nu = \infty$: Equivalent to Gaussian (infinitely smooth)

### Brownian Motion

Standard Brownian motion covariance: $$k(s,t) = \sigma^{2}\min(s,t)$$

``` r
cov_brown <- kernel_brownian(variance = 1)
fd_brown <- make_gaussian_process(n = 10, t = t, cov = cov_brown, seed = 42)
plot(fd_brown, main = "Brownian Motion")
```

![](covariance-functions_files/figure-html/cov-brownian-1.png)

Note: Brownian covariance is only defined for 1D domains.

### Periodic

For data with periodic structure:
$$k(s,t) = \sigma^{2}\exp\left( - \frac{2\sin^{2}\left( \pi|s - t|/p \right)}{\ell^{2}} \right)$$

``` r
t_long <- seq(0, 3, length.out = 200)
cov_per <- kernel_periodic(variance = 1, length_scale = 0.5, period = 1)
fd_per <- make_gaussian_process(n = 5, t = t_long, cov = cov_per, seed = 42)
plot(fd_per, main = "Periodic Covariance (period = 1)")
```

![](covariance-functions_files/figure-html/cov-periodic-1.png)

### Linear

Linear covariance produces functions that are linear combinations of a
constant and a linear function: $$k(s,t) = \sigma^{2}(s \cdot t + c)$$

``` r
cov_lin <- kernel_linear(variance = 1, offset = 0)
fd_lin <- make_gaussian_process(n = 10, t = t, cov = cov_lin, seed = 42)
plot(fd_lin, main = "Linear Covariance")
```

![](covariance-functions_files/figure-html/cov-linear-1.png)

### Polynomial

Generalization of linear to polynomial basis functions:
$$k(s,t) = \sigma^{2}(s \cdot t + c)^{d}$$

``` r
cov_poly <- kernel_polynomial(variance = 1, offset = 1, degree = 3)
fd_poly <- make_gaussian_process(n = 10, t = t, cov = cov_poly, seed = 42)
plot(fd_poly, main = "Polynomial Covariance (degree 3)")
```

![](covariance-functions_files/figure-html/cov-polynomial-1.png)

### White Noise

Diagonal covariance representing independent noise:
$$k(s,t) = \sigma^{2}\mathbf{1}_{s = t}$$

``` r
cov_white <- kernel_whitenoise(variance = 0.5)
fd_white <- make_gaussian_process(n = 5, t = t, cov = cov_white, seed = 42)
plot(fd_white, main = "White Noise")
```

![](covariance-functions_files/figure-html/cov-white-1.png)

## Combining Covariance Functions

### Addition (kernel_add)

Sum of covariance functions models independent components:

``` r
# Signal + noise model
cov_signal <- kernel_gaussian(variance = 1, length_scale = 0.2)
cov_noise <- kernel_whitenoise(variance = 0.1)
cov_total <- kernel_add(cov_signal, cov_noise)

fd_noisy <- make_gaussian_process(n = 5, t = t, cov = cov_total, seed = 42)
plot(fd_noisy, main = "Smooth signal + noise")
```

![](covariance-functions_files/figure-html/cov-add-1.png)

### Multiplication (kernel_mult)

Product of covariance functions:

``` r
# Locally periodic: smooth envelope modulating periodic behavior
cov_envelope <- kernel_gaussian(variance = 1, length_scale = 0.5)
cov_periodic <- kernel_periodic(period = 0.2)
cov_local_per <- kernel_mult(cov_envelope, cov_periodic)

fd_local_per <- make_gaussian_process(n = 5, t = t, cov = cov_local_per, seed = 42)
plot(fd_local_per, main = "Locally periodic")
```

![](covariance-functions_files/figure-html/cov-mult-1.png)

## Mean Functions

Gaussian processes can have non-zero mean functions:

``` r
# Scalar mean
fd_mean5 <- make_gaussian_process(n = 10, t = t,
                                   cov = kernel_gaussian(variance = 0.1),
                                   mean = 5, seed = 42)
plot(fd_mean5, main = "Constant mean = 5")
```

![](covariance-functions_files/figure-html/mean-func-1.png)

``` r

# Function mean
mean_func <- function(t) sin(2 * pi * t)
fd_sinmean <- make_gaussian_process(n = 10, t = t,
                                     cov = kernel_gaussian(variance = 0.1),
                                     mean = mean_func, seed = 42)
plot(fd_sinmean, main = "Sinusoidal mean function")
```

![](covariance-functions_files/figure-html/mean-func-2.png)

## 2D Functional Data (Surfaces)

Covariance functions can generate 2D functional data (surfaces):

``` r
s <- seq(0, 1, length.out = 30)
t2d <- seq(0, 1, length.out = 30)

# Generate 2D GP samples
fd2d <- make_gaussian_process(n = 4, t = list(s, t2d),
                               cov = kernel_gaussian(length_scale = 0.3),
                               seed = 42)
plot(fd2d)
```

![](covariance-functions_files/figure-html/2d-gp-1.png)

Note:
[`kernel_brownian()`](https://sipemu.github.io/fdars/reference/kernel_brownian.md)
and
[`kernel_periodic()`](https://sipemu.github.io/fdars/reference/kernel_periodic.md)
only support 1D domains.

## Reproducibility

Use the `seed` parameter for reproducible samples:

``` r
fd1 <- make_gaussian_process(n = 3, t = t, cov = kernel_gaussian(), seed = 123)
fd2 <- make_gaussian_process(n = 3, t = t, cov = kernel_gaussian(), seed = 123)
all.equal(fd1$data, fd2$data)  # TRUE
#> [1] TRUE
```

## Comparison of Smoothness

``` r
# Generate data for comparison
kernels <- list(
  list(name = "Gaussian (very smooth)", cov = kernel_gaussian()),
  list(name = "Matern 5/2", cov = kernel_matern(nu = 2.5)),
  list(name = "Matern 3/2", cov = kernel_matern(nu = 1.5)),
  list(name = "Exponential (rough)", cov = kernel_exponential())
)

df_smooth_comp <- do.call(rbind, lapply(kernels, function(k) {
  fd <- make_gaussian_process(n = 3, t = t, cov = k$cov, seed = 1)
  data.frame(
    t = rep(t, 3),
    value = as.vector(t(fd$data)),
    curve = rep(1:3, each = length(t)),
    kernel = k$name
  )
}))
df_smooth_comp$kernel <- factor(df_smooth_comp$kernel,
                                 levels = sapply(kernels, function(k) k$name))

ggplot(df_smooth_comp, aes(x = t, y = value, group = curve, color = factor(curve))) +
  geom_line(alpha = 0.8) +
  facet_wrap(~ kernel, ncol = 2) +
  labs(x = "t", y = "X(t)", title = "Comparison of Smoothness") +
  theme_minimal() +
  theme(legend.position = "none")
```

![](covariance-functions_files/figure-html/smoothness-comparison-1.png)

## Summary Table

| Covariance  | Parameters                     | Smoothness                  | Notes                     |
|-------------|--------------------------------|-----------------------------|---------------------------|
| Gaussian    | variance, length_scale         | $C^{\infty}$                | Most common, very smooth  |
| Exponential | variance, length_scale         | $C^{0}$                     | Rough, non-differentiable |
| Matern      | variance, length_scale, nu     | $C^{\lbrack\nu - 1\rbrack}$ | Flexible smoothness       |
| Brownian    | variance                       | $C^{0}$                     | 1D only, non-stationary   |
| Linear      | variance, offset               | \-                          | Linear functions          |
| Polynomial  | variance, offset, degree       | \-                          | Polynomial functions      |
| WhiteNoise  | variance                       | \-                          | Independent noise         |
| Periodic    | variance, length_scale, period | $C^{\infty}$                | 1D only, periodic         |

## References

- Rasmussen, C.E. and Williams, C.K.I. (2006). *Gaussian Processes for
  Machine Learning*. MIT Press.
- Ramsay, J.O. and Silverman, B.W. (2005). *Functional Data Analysis*.
  Springer.
