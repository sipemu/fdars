# Convert Functional Data to PLS Scores

Performs Partial Least Squares regression and returns component scores
for functional data using the NIPALS algorithm.

## Usage

``` r
fdata2pls(fdataobj, y, ncomp = 2, lambda = 0, norm = TRUE)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- y:

  Response vector (numeric).

- ncomp:

  Number of PLS components to extract (default 2).

- lambda:

  Regularization parameter (default 0, not currently used).

- norm:

  Logical. If TRUE (default), normalize the scores.

## Value

A list with components:

- weights:

  Matrix of PLS weights (m x ncomp)

- scores:

  Matrix of PLS scores (n x ncomp)

- loadings:

  Matrix of PLS loadings (m x ncomp)

- call:

  The function call

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
y <- rowMeans(X) + rnorm(20, sd = 0.1)
fd <- fdata(X, argvals = t)
pls <- fdata2pls(fd, y, ncomp = 3)
```
