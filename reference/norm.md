# Compute Lp norm of functional data

Compute Lp norm of functional data

## Usage

``` r
norm(fdataobj, lp = 2)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- lp:

  The p in Lp norm (default 2 for L2 norm).

## Value

A numeric vector of norms, one per curve.

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))
norms <- norm(fd)
```
