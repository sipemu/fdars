# Center functional data

Subtract the mean function from each curve.

## Usage

``` r
fdata.cen(fdataobj)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

## Value

A centered 'fdata' object.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
fd_centered <- fdata.cen(fd)
```
