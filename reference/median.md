# Compute Functional Median

Returns the curve with maximum depth using the specified depth method.

## Usage

``` r
median(
  fdataobj,
  method = c("FM", "mode", "RP", "RT", "BD", "MBD", "FSD", "KFSD", "RPD"),
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- method:

  Depth method to use. One of "FM", "mode", "RP", "RT", "BD", "MBD",
  "FSD", "KFSD", or "RPD". Default is "FM".

- ...:

  Additional arguments passed to the depth function.

## Value

The curve (as fdata object) with maximum depth.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
med <- median(fd)
med_mode <- median(fd, method = "mode")
```
