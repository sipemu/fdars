# Compute Lp Norm of Functional Data

Generic function to compute Lp norms for functional data objects. Works
with both regular `fdata` and irregular `irregFdata` objects.

## Usage

``` r
norm(x, p = 2, ...)

# S3 method for class 'fdata'
norm(x, p = 2, ...)

# S3 method for class 'irregFdata'
norm(x, p = 2, ...)
```

## Arguments

- x:

  A functional data object (`fdata` or `irregFdata`).

- p:

  The order of the norm (default 2 for L2 norm).

- ...:

  Additional arguments passed to methods.

## Value

A numeric vector of norms, one per curve.

## Examples

``` r
# Regular fdata
fd <- fdata(matrix(rnorm(100), 10, 10))
norms <- norm(fd)

# Irregular fdata
ifd <- sparsify(fd, minObs = 3, maxObs = 7, seed = 42)
norms_irreg <- norm(ifd)
```
