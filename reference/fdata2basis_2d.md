# Convert 2D Functional Data to Tensor Product Basis Coefficients

Projects 2D functional data (surfaces) onto a tensor product basis,
which is the Kronecker product of two 1D bases.

## Usage

``` r
fdata2basis_2d(
  fdataobj,
  nbasis.s = 10,
  nbasis.t = 10,
  type = c("bspline", "fourier")
)
```

## Arguments

- fdataobj:

  A 2D fdata object (surfaces).

- nbasis.s:

  Number of basis functions in the s (first) direction.

- nbasis.t:

  Number of basis functions in the t (second) direction.

- type:

  Basis type: "bspline" (default) or "fourier".

## Value

A matrix of coefficients `[n x (nbasis.s * nbasis.t)]`.

## Details

The tensor product basis is defined as: \$\$B\_{2d}(s, t) = B_s(s)
\otimes B_t(t)\$\$

## Examples

``` r
# Create 2D surface data
s <- seq(0, 1, length.out = 20)
t <- seq(0, 1, length.out = 20)
surface <- outer(sin(2*pi*s), cos(2*pi*t))
fd2d <- fdata(array(surface, dim = c(1, 20, 20)))

# Project to tensor product basis
coefs <- fdata2basis_2d(fd2d, nbasis.s = 7, nbasis.t = 7, type = "fourier")
```
