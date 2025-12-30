# Estimate Mean Function for Irregular Data

Uses kernel smoothing to estimate the mean function from irregularly
sampled functional data.

## Usage

``` r
# S3 method for class 'irregFdata'
mean(
  x,
  argvals = NULL,
  bandwidth = NULL,
  kernel = c("epanechnikov", "gaussian")
)
```

## Arguments

- x:

  An object of class `irregFdata`.

- argvals:

  Target grid for mean estimation. If `NULL`, uses a regular grid of 100
  points.

- bandwidth:

  Kernel bandwidth. If `NULL`, uses a default based on the data.

- kernel:

  Kernel type: `"epanechnikov"` (default) or `"gaussian"`.

## Value

An `fdata` object containing the estimated mean function.

## Examples

``` r
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 50, argvals = t, M = 5, seed = 42)
ifd <- sparsify(fd, minObs = 10, maxObs = 30, seed = 123)

mean_fd <- mean.irregFdata(ifd)
#> Error in mean.irregFdata(ifd): could not find function "mean.irregFdata"
plot(mean_fd, main = "Estimated Mean Function")
#> Error: object 'mean_fd' not found
```
