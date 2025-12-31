# Convert Functional Data to Basis Coefficients

Project functional data onto a basis system and return coefficients.
Supports B-spline and Fourier basis. Works with both regular `fdata` and
irregular `irregFdata` objects.

## Usage

``` r
fdata2basis(x, nbasis = 10, type = c("bspline", "fourier"), ...)

# S3 method for class 'fdata'
fdata2basis(x, nbasis = 10, type = c("bspline", "fourier"), ...)

# S3 method for class 'irregFdata'
fdata2basis(x, nbasis = 10, type = c("bspline", "fourier"), ...)
```

## Arguments

- x:

  An object of class 'fdata' or 'irregFdata'.

- nbasis:

  Number of basis functions (default 10).

- type:

  Type of basis: "bspline" (default) or "fourier".

- ...:

  Additional arguments (currently unused).

## Value

A matrix of coefficients (n x nbasis).

## Details

For regular `fdata` objects, all curves are projected onto the same
basis evaluated at the common grid points.

For irregular `irregFdata` objects, each curve is individually fitted to
the basis using least squares at its own observation points. This is the
preferred approach for sparse/irregularly sampled data as it avoids
interpolation artifacts.

## Examples

``` r
# Regular fdata
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)
coefs <- fdata2basis(fd, nbasis = 10, type = "bspline")

# Irregular fdata (sparsified)
ifd <- sparsify(fd, minObs = 10, maxObs = 20, seed = 42)
coefs_irreg <- fdata2basis(ifd, nbasis = 10, type = "bspline")
```
