# K-Means++ Center Initialization

Initialize cluster centers using the k-means++ algorithm, which selects
centers with probability proportional to squared distance from existing
centers.

## Usage

``` r
cluster.init(fdataobj, ncl, metric = "L2", seed = NULL)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- ncl:

  Number of clusters.

- metric:

  Metric to use. One of "L2", "L1", or "Linf".

- seed:

  Optional random seed.

## Value

An fdata object containing the initial cluster centers.

## Examples

``` r
t <- seq(0, 1, length.out = 50)
X <- matrix(rnorm(30 * 50), 30, 50)
fd <- fdata(X, argvals = t)
init_centers <- cluster.init(fd, ncl = 3)
```
