# Compute Lp Norm for Irregular Functional Data

Compute Lp Norm for Irregular Functional Data

## Usage

``` r
norm.irregFdata(x, p = 2)
```

## Arguments

- x:

  An object of class `irregFdata`.

- p:

  The order of the norm (default 2 for L2).

## Value

A numeric vector of norms, one per curve.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 10, argvals = t, M = 5, seed = 42)
ifd <- sparsify(fd, minObs = 20, maxObs = 50, seed = 123)

l2_norms <- norm.irregFdata(ifd, p = 2)
print(l2_norms)
#>  [1] 1.4253704 0.3836368 0.6444795 1.6637410 0.8791651 0.8964032 0.7804063
#>  [8] 1.5604639 1.7617871 2.4448937
```
