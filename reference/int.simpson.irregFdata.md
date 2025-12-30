# Integrate Irregular Functional Data

Compute the integral of each curve using trapezoidal rule.

## Usage

``` r
int.simpson.irregFdata(x)
```

## Arguments

- x:

  An object of class `irregFdata`.

## Value

A numeric vector of integrals, one per curve.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 10, argvals = t, M = 5, seed = 42)
ifd <- sparsify(fd, minObs = 20, maxObs = 50, seed = 123)

integrals <- int.simpson.irregFdata(ifd)
print(integrals)
#>  [1]  0.004278081  0.113386144  0.240958656 -0.029098983 -0.545397334
#>  [6] -0.211763487  0.171711645  1.461696458 -1.314386209 -2.348170607
```
