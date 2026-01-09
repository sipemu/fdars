# Min-Max scaling for functional data

Scales each curve to the range \\\[0, 1\]\\ (or custom range). This
preserves the shape while normalizing the range.

## Usage

``` r
scale_minmax(fdataobj, min = 0, max = 1)

# S3 method for class 'fdata'
scale_minmax(fdataobj, min = 0, max = 1)

# S3 method for class 'irregFdata'
scale_minmax(fdataobj, min = 0, max = 1)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- min:

  Target minimum value (default 0).

- max:

  Target maximum value (default 1).

## Value

A scaled 'fdata' object where each curve is in the specified range.

## Examples

``` r
fd <- fdata(matrix(rnorm(100) * 10 + 50, 10, 10), argvals = seq(0, 1, length.out = 10))
fd_scaled <- scale_minmax(fd)
# Check: each curve now in [0, 1]
apply(fd_scaled$data, 1, range)
#>      [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#> [1,]    0    0    0    0    0    0    0    0    0     0
#> [2,]    1    1    1    1    1    1    1    1    1     1
```
