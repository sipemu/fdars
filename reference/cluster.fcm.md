# Fuzzy C-Means Clustering for Functional Data

Performs fuzzy c-means clustering on functional data, where each curve
has a membership degree to each cluster rather than a hard assignment.

## Usage

``` r
cluster.fcm(fdataobj, ncl, m = 2, max.iter = 100, tol = 1e-06, seed = NULL)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ncl:

  Number of clusters.

- m:

  Fuzziness parameter (default 2). Must be \> 1. Higher values give
  softer cluster boundaries.

- max.iter:

  Maximum number of iterations (default 100).

- tol:

  Convergence tolerance (default 1e-6).

- seed:

  Optional random seed for reproducibility.

## Value

A list of class 'fuzzycmeans.fd' with components:

- membership:

  Matrix of membership degrees (n x ncl). Each row sums to 1.

- cluster:

  Hard cluster assignments (argmax of membership).

- centers:

  An fdata object containing the cluster centers.

- objective:

  Final value of the objective function.

- fdataobj:

  The input functional data object.

## Details

Fuzzy c-means minimizes the objective function: \$\$J = \sum\_{i=1}^n
\sum\_{c=1}^k u\_{ic}^m \|\|X_i - v_c\|\|^2\$\$ where u_ic is the
membership of curve i in cluster c, v_c is the cluster center, and m is
the fuzziness parameter.

The membership degrees are updated as: \$\$u\_{ic} = 1 / \sum\_{j=1}^k
(d\_{ic}/d\_{ij})^{2/(m-1)}\$\$

When m approaches 1, FCM becomes equivalent to hard k-means. As m
increases, the clusters become softer (more overlap). m = 2 is the most
common choice.

## See also

[`cluster.kmeans`](https://sipemu.github.io/fdars/reference/cluster.kmeans.md)
for hard clustering

## Examples

``` r
# Create functional data with THREE groups - one genuinely overlapping
set.seed(42)
t <- seq(0, 1, length.out = 50)
n <- 45
X <- matrix(0, n, 50)

# Group 1: Sine waves centered at 0
for (i in 1:15) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
# Group 2: Sine waves centered at 1.5 (clearly separated from group 1)
for (i in 16:30) X[i, ] <- sin(2*pi*t) + 1.5 + rnorm(50, sd = 0.2)
# Group 3: Between groups 1 and 2 (true overlap - ambiguous membership)
for (i in 31:45) X[i, ] <- sin(2*pi*t) + 0.75 + rnorm(50, sd = 0.3)

fd <- fdata(X, argvals = t)

# Fuzzy clustering reveals the overlap
fcm <- cluster.fcm(fd, ncl = 3, seed = 123)

# Curves in group 3 (31-45) have split membership - this is the key benefit!
cat("Membership for curves 31-35 (overlap region):\n")
#> Membership for curves 31-35 (overlap region):
print(round(fcm$membership[31:35, ], 2))
#>      [,1] [,2] [,3]
#> [1,] 0.02 0.02 0.96
#> [2,] 0.01 0.01 0.97
#> [3,] 0.01 0.02 0.96
#> [4,] 0.02 0.04 0.94
#> [5,] 0.02 0.01 0.97

# Compare to hard clustering which forces a decision
km <- cluster.kmeans(fd, ncl = 3, seed = 123)
cat("\nHard vs Fuzzy assignment for curve 35:\n")
#> 
#> Hard vs Fuzzy assignment for curve 35:
cat("K-means cluster:", km$cluster[35], "\n")
#> K-means cluster: 3 
cat("FCM memberships:", round(fcm$membership[35, ], 2), "\n")
#> FCM memberships: 0.02 0.01 0.97 
```
