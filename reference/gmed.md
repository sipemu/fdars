# Geometric Median of Functional Data

Computes the geometric median (L1 median) of functional data using
Weiszfeld's iterative algorithm. The geometric median minimizes the sum
of L2 distances to all curves/surfaces, making it robust to outliers.

## Usage

``` r
gmed(fdataobj, max.iter = 100, tol = 1e-06)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- max.iter:

  Maximum number of iterations (default 100).

- tol:

  Convergence tolerance (default 1e-6).

## Value

An fdata object containing the geometric median function (1D or 2D).

## Details

The geometric median y minimizes: \$\$\sum\_{i=1}^n \|\|X_i -
y\|\|\_{L2}\$\$

Unlike the mean (L2 center), the geometric median is robust to outliers
because extreme values have bounded influence (influence function is
bounded).

The Weiszfeld algorithm is an iteratively reweighted least squares
method that converges to the geometric median.

## See also

[`mean.fdata`](https://sipemu.github.io/fdars/reference/mean.fdata.md)
for the (non-robust) mean function,
[`median`](https://sipemu.github.io/fdars/reference/median.md) for
depth-based median

## Examples

``` r
# Create functional data with an outlier
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:19) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
X[20, ] <- sin(2*pi*t) + 5  # Large outlier
fd <- fdata(X, argvals = t)

# Compare mean vs geometric median
mean_curve <- mean(fd)
gmed_curve <- gmed(fd)

# The geometric median is less affected by the outlier
```
