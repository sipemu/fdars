# Clustering Functions for Functional Data

Functions for clustering functional data, including k-means and related
algorithms. Functional K-Means Clustering

## Usage

``` r
cluster.kmeans(
  fdataobj,
  ncl,
  metric = "L2",
  max.iter = 100,
  nstart = 10,
  seed = NULL,
  draw = FALSE,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ncl:

  Number of clusters.

- metric:

  Either a string ("L2", "L1", "Linf") for fast Rust-based distance
  computation, or a metric/semimetric function (e.g., `metric.lp`,
  `metric.hausdorff`, `semimetric.pca`). Using a function provides
  flexibility but may be slower for semimetrics computed in R.

- max.iter:

  Maximum number of iterations (default 100).

- nstart:

  Number of random starts (default 10). The best result (lowest
  within-cluster sum of squares) is returned.

- seed:

  Optional random seed for reproducibility.

- draw:

  Logical. If TRUE, plot the clustered curves (not yet implemented).

- ...:

  Additional arguments passed to the metric function.

## Value

A list of class 'cluster.kmeans' with components:

- cluster:

  Integer vector of cluster assignments (1 to ncl).

- centers:

  An fdata object containing the cluster centers.

- withinss:

  Within-cluster sum of squares for each cluster.

- tot.withinss:

  Total within-cluster sum of squares.

- size:

  Number of observations in each cluster.

- fdataobj:

  The input functional data object.

## Details

Performs k-means clustering on functional data using the specified
metric. Uses k-means++ initialization for better initial centers.

When `metric` is a string ("L2", "L1", "Linf"), the entire k-means
algorithm runs in Rust with parallel processing, providing 50-200x
speedup.

When `metric` is a function, distances are computed using that function.
Functions like `metric.lp`, `metric.hausdorff`, and `metric.DTW` have
Rust backends and remain fast. Semimetric functions (`semimetric.*`) are
computed in R and will be slower for large datasets.

## Examples

``` r
# Create functional data with two groups
t <- seq(0, 1, length.out = 50)
n <- 30
X <- matrix(0, n, 50)
true_cluster <- rep(1:2, each = 15)
for (i in 1:n) {
  if (true_cluster[i] == 1) {
    X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
  } else {
    X[i, ] <- cos(2*pi*t) + rnorm(50, sd = 0.1)
  }
}
fd <- fdata(X, argvals = t)

# Cluster with string metric (fast Rust path)
result <- cluster.kmeans(fd, ncl = 2, metric = "L2")
table(result$cluster, true_cluster)
#>    true_cluster
#>      1  2
#>   1 15  0
#>   2  0 15

# Cluster with metric function (also fast - Rust backend)
result2 <- cluster.kmeans(fd, ncl = 2, metric = metric.lp)

# Cluster with semimetric (flexible but slower)
result3 <- cluster.kmeans(fd, ncl = 2, metric = semimetric.pca, ncomp = 3)
```
