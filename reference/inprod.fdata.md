# Inner Product of Functional Data

Compute the inner product of two functional data objects. \<f, g\> =
integral(f(t) \* g(t) dt)

## Usage

``` r
inprod.fdata(fdata1, fdata2 = NULL)
```

## Arguments

- fdata1:

  First functional data object.

- fdata2:

  Second functional data object. If NULL, computes inner products of
  fdata1 with itself.

## Value

A matrix of inner products. If fdata1 has n1 curves and fdata2 has n2
curves, returns an n1 x n2 matrix.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
X1 <- matrix(sin(2*pi*t), nrow = 1)
X2 <- matrix(cos(2*pi*t), nrow = 1)
fd1 <- fdata(X1, argvals = t)
fd2 <- fdata(X2, argvals = t)
# Inner product of sin and cos over [0,1] should be 0
inprod.fdata(fd1, fd2)
#>              [,1]
#> [1,] 2.218175e-17
```
