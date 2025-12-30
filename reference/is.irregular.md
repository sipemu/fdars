# Check if an Object is Irregular Functional Data

Check if an Object is Irregular Functional Data

## Usage

``` r
is.irregular(x)
```

## Arguments

- x:

  Any R object.

## Value

`TRUE` if `x` is of class `irregFdata`, `FALSE` otherwise.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
is.irregular(fd)  # FALSE
#> [1] FALSE

ifd <- sparsify(fd, minObs = 3, maxObs = 7)
is.irregular(ifd)  # TRUE
#> [1] TRUE
```
