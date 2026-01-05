# Standardize functional data (z-score normalization)

Transforms each curve to have mean 0 and standard deviation 1. This is
useful for comparing curve shapes regardless of their level or scale.

## Usage

``` r
standardize(fdataobj)

# S3 method for class 'fdata'
standardize(fdataobj)

# S3 method for class 'irregFdata'
standardize(fdataobj)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

## Value

A standardized 'fdata' object where each curve has mean 0 and sd 1.

## Examples

``` r
fd <- fdata(matrix(rnorm(100) * 10 + 50, 10, 10), argvals = seq(0, 1, length.out = 10))
fd_std <- standardize(fd)
# Check: each curve now has mean ~0 and sd ~1
rowMeans(fd_std$data)
#>  [1] -3.996803e-16  4.773959e-16  2.442491e-16  3.441691e-16  2.997602e-16
#>  [6]  7.299716e-16  3.330669e-17  3.941292e-16  1.776357e-16  2.664535e-16
apply(fd_std$data, 1, sd)
#>  [1] 1 1 1 1 1 1 1 1 1 1
```
