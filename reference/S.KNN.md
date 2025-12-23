# K-Nearest Neighbors Smoother Matrix

Compute a smoother matrix using adaptive bandwidth based on the k
nearest neighbors. The bandwidth at each point is the distance to the
k-th nearest neighbor.

## Usage

``` r
S.KNN(tt, knn, Ker = "norm", w = NULL, cv = FALSE)
```

## Arguments

- tt:

  Evaluation points (numeric vector).

- knn:

  Number of nearest neighbors.

- Ker:

  Kernel function or name.

- w:

  Optional weights vector.

- cv:

  Logical. If TRUE, compute leave-one-out cross-validation matrix.

## Value

An n x n smoother matrix S.

## Examples

``` r
tt <- seq(0, 1, length.out = 50)
S <- S.KNN(tt, knn = 10)
```
