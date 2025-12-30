# Convert Regular Functional Data to Irregular by Subsampling

Creates an `irregFdata` object from regular `fdata` by randomly
selecting a subset of observation points for each curve.

## Usage

``` r
sparsify(fdataobj, minObs = 5, maxObs = NULL, prob = NULL, seed = NULL)
```

## Arguments

- fdataobj:

  An object of class `fdata`.

- minObs:

  Minimum number of observations to keep per curve.

- maxObs:

  Maximum number of observations to keep per curve. If `NULL`, uses the
  total number of points.

- prob:

  Sampling probability function. If `NULL`, uniform sampling is used.
  Otherwise, a function that takes `argvals` and returns sampling
  weights (not necessarily normalized).

- seed:

  Optional integer random seed for reproducibility.

## Value

An object of class `irregFdata`.

## Details

For each curve, the function:

1.  Draws a random number of points to keep between `minObs` and
    `maxObs`

2.  Samples that many points (without replacement) from the grid

3.  If `prob` is provided, sampling is weighted accordingly

Common probability functions:

- Uniform: `NULL` (default)

- More points in middle: `function(t) dnorm(t, mean = 0.5, sd = 0.2)`

- More points at ends: `function(t) 1 - dnorm(t, mean = 0.5, sd = 0.2)`

## See also

[`irregFdata`](https://sipemu.github.io/fdars/reference/irregFdata.md),
[`as.fdata.irregFdata`](https://sipemu.github.io/fdars/reference/as.fdata.irregFdata.md),
[`addError`](https://sipemu.github.io/fdars/reference/addError.md)

## Examples

``` r
# Create regular functional data
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 20, argvals = t, M = 5, seed = 42)

# Uniform sparsification
ifd <- sparsify(fd, minObs = 10, maxObs = 30, seed = 123)
print(ifd)
#> Irregular Functional Data Object
#> =================================
#>   Number of observations: 20 
#>   Points per curve:
#>     Min: 10 
#>     Median: 24 
#>     Max: 29 
#>     Total: 441 
#>   Domain: [ 0 , 1 ]
plot(ifd)


# Non-uniform: more observations in the middle
prob_middle <- function(t) dnorm(t, mean = 0.5, sd = 0.2)
ifd_middle <- sparsify(fd, minObs = 15, maxObs = 25, prob = prob_middle, seed = 123)
plot(ifd_middle, main = "More Observations in Middle")
```
