# White Noise Covariance Function

Computes the white noise covariance function: \$\$k(s, t) = \sigma^2
\mathbf{1}\_{s = t}\$\$

## Usage

``` r
kernel_whitenoise(variance = 1)
```

## Arguments

- variance:

  Variance (noise level) parameter \\\sigma^2\\ (default 1).

## Value

A covariance function object of class 'kernel_whitenoise'.

## Details

where \\\mathbf{1}\_{s = t}\\ is 1 if \\s = t\\ and 0 otherwise.

The white noise covariance function represents independent noise at each
point. It can be added to other covariance functions to model
observation noise.

## See also

[`kernel_gaussian`](https://sipemu.github.io/fdars/reference/kernel_gaussian.md),
[`make_gaussian_process`](https://sipemu.github.io/fdars/reference/make_gaussian_process.md)

## Examples

``` r
# White noise covariance produces independent samples at each point
cov_func <- kernel_whitenoise(variance = 0.1)
t <- seq(0, 1, length.out = 50)
K <- cov_func(t)
# K is diagonal
```
