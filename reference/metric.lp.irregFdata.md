# Compute Lp Distance Matrix for Irregular Data

Compute Lp Distance Matrix for Irregular Data

## Usage

``` r
metric.lp.irregFdata(x, p = 2)
```

## Arguments

- x:

  An object of class `irregFdata`.

- p:

  The order of the Lp distance (default 2).

## Value

A symmetric distance matrix.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 10, argvals = t, M = 5, seed = 42)
ifd <- sparsify(fd, minObs = 20, maxObs = 50, seed = 123)

D <- metric.lp.irregFdata(ifd)
print(round(D, 2))
#>       [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
#>  [1,] 0.00 1.39 1.81 2.58 1.84 2.04 2.05 2.07 2.37  2.53
#>  [2,] 1.39 0.00 0.52 1.49 1.09 1.14 0.85 1.44 1.88  2.57
#>  [3,] 1.81 0.52 0.00 1.18 1.11 1.22 0.66 1.50 2.15  2.81
#>  [4,] 2.58 1.49 1.18 0.00 1.31 1.84 1.65 2.17 2.63  3.06
#>  [5,] 1.84 1.09 1.11 1.31 0.00 1.10 1.22 2.24 1.79  2.03
#>  [6,] 2.04 1.14 1.22 1.84 1.10 0.00 0.89 1.91 1.41  2.48
#>  [7,] 2.05 0.85 0.66 1.65 1.22 0.89 0.00 1.66 1.90  2.81
#>  [8,] 2.07 1.44 1.50 2.17 2.24 1.91 1.66 0.00 2.93  3.92
#>  [9,] 2.37 1.88 2.15 2.63 1.79 1.41 1.90 2.93 0.00  1.70
#> [10,] 2.53 2.57 2.81 3.06 2.03 2.48 2.81 3.92 1.70  0.00
```
