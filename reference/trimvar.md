# Compute Functional Trimmed Variance

Computes the trimmed variance by excluding curves with lowest depth.

## Usage

``` r
trimvar(
  fdataobj,
  trim = 0.1,
  method = c("FM", "mode", "RP", "RT", "BD", "MBD", "FSD", "KFSD", "RPD"),
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- trim:

  Proportion of curves to trim (default 0.1).

- method:

  Depth method to use. One of "FM", "mode", "RP", "RT", "BD", "MBD",
  "FSD", "KFSD", or "RPD". Default is "FM".

- ...:

  Additional arguments passed to the depth function.

## Value

An fdata object containing the trimmed variance function.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
tv <- trimvar(fd, trim = 0.2)
tv_mode <- trimvar(fd, trim = 0.2, method = "mode")
```
