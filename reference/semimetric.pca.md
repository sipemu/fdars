# Semi-metric based on Principal Components

Computes a semi-metric based on the first ncomp principal component
scores.

## Usage

``` r
semimetric.pca(fdataobj, fdataref = NULL, ncomp = 2, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- fdataref:

  An object of class 'fdata'. If NULL, uses fdataobj.

- ncomp:

  Number of principal components to use.

- ...:

  Additional arguments (ignored).

## Value

A distance matrix based on PC scores.

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))
D <- semimetric.pca(fd, ncomp = 3)
```
