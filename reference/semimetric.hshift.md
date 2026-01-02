# Semi-metric based on Horizontal Shift (Time Warping)

Computes a semi-metric based on the minimum L2 distance after optimal
horizontal shifting of curves. This is useful for comparing curves that
may have phase differences.

## Usage

``` r
semimetric.hshift(fdataobj, fdataref = NULL, max_shift = -1, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, uses fdataobj.

- max_shift:

  Maximum shift in number of grid points. Default is m/4 where m is the
  number of evaluation points. Use -1 for automatic.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix based on minimum L2 distance after shift.

## Details

For each pair of curves, this function computes: \$\$d(f, g) =
\min\_{\|h\| \le h\_{max}} \|\|f(t) - g(t+h)\|\|\$\$ where h is the
horizontal shift in discrete units.

This semi-metric is useful when comparing curves with phase shifts, such
as ECG signals with different heart rates or periodic signals with
different phases.

## Examples

``` r
# Create curves with phase shifts
t <- seq(0, 2*pi, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(t + i*0.2) + rnorm(100, sd = 0.1)
fd <- fdata(X, argvals = t)

# Compute distance accounting for phase shifts
D <- semimetric.hshift(fd, max_shift = 10)
```
