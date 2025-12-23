# Convert Functional Data to Principal Component Scores

Performs functional PCA and returns principal component scores for
functional data. Uses SVD on centered data.

## Usage

``` r
fdata2pc(fdataobj, ncomp = 2, lambda = 0, norm = TRUE)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ncomp:

  Number of principal components to extract (default 2).

- lambda:

  Regularization parameter (default 0, not currently used).

- norm:

  Logical. If TRUE (default), normalize the scores.

## Value

A list with components:

- d:

  Singular values (proportional to sqrt of eigenvalues)

- rotation:

  fdata object containing PC loadings

- x:

  Matrix of PC scores (n x ncomp)

- mean:

  Mean function (numeric vector)

- fdataobj.cen:

  Centered fdata object

- call:

  The function call

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)
pc <- fdata2pc(fd, ncomp = 3)
```
