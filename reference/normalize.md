# Normalize functional data

Scales each curve to have Lp norm equal to 1.

## Usage

``` r
normalize(fdataobj, p = 2)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- p:

  The order of the norm (default 2 for L2 norm).

## Value

A normalized 'fdata' object where each curve has unit norm.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10), argvals = seq(0, 1, length.out = 10))
fd_norm <- normalize(fd)
norm(fd_norm)  # All values should be 1
#>  [1] 1 1 1 1 1 1 1 1 1 1
```
