# Semi-metric based on Principal Components

Computes a semi-metric based on the first ncomp principal component
scores.

## Usage

``` r
semimetric.pca(fdata1, fdata2 = NULL, ncomp = 2, ...)
```

## Arguments

- fdata1:

  An object of class 'fdata'.

- fdata2:

  An object of class 'fdata'. If NULL, uses fdata1.

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
