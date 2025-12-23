# Compute Distance/Similarity Between Groups of Functional Data

Computes various distance and similarity measures between pre-defined
groups of functional curves.

## Usage

``` r
group.distance(
  fdataobj,
  groups,
  method = c("centroid", "hausdorff", "depth", "all"),
  metric = "lp",
  p = 2,
  depth.method = "FM",
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- groups:

  A factor or character vector specifying group membership for each
  curve. Must have length equal to the number of curves.

- method:

  Distance/similarity method:

  - "centroid": L2 distance between group mean curves

  - "hausdorff": Hausdorff-style distance between groups

  - "depth": Depth-based overlap (similarity, not distance)

  - "all": Compute all methods

- metric:

  Distance metric for centroid method (default "lp").

- p:

  Power for Lp metric (default 2 for L2).

- depth.method:

  Depth method for depth-based overlap (default "FM").

- ...:

  Additional arguments passed to metric functions.

## Value

An object of class 'group.distance' containing:

- centroid:

  Centroid distance matrix (if method includes centroid)

- hausdorff:

  Hausdorff distance matrix (if method includes hausdorff)

- depth:

  Depth-based similarity matrix (if method includes depth)

- groups:

  Unique group labels

- group.sizes:

  Number of curves per group

- method:

  Methods used

## Examples

``` r
# Create grouped functional data
set.seed(42)
n <- 30
m <- 50
t_grid <- seq(0, 1, length.out = m)
X <- matrix(0, n, m)
for (i in 1:15) X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
for (i in 16:30) X[i, ] <- cos(2 * pi * t_grid) + rnorm(m, sd = 0.1)
fd <- fdata(X, argvals = t_grid)
groups <- factor(rep(c("A", "B"), each = 15))

# Compute all distance measures
gd <- group.distance(fd, groups, method = "all")
print(gd)
#> Group Distance Analysis
#> =======================
#> Groups: A, B 
#> Group sizes: A=15, B=15 
#> 
#> Centroid Distance (L2 between group means):
#>       A     B
#> A 0.000 1.002
#> B 1.002 0.000
#> 
#> Hausdorff Distance (worst-case between groups):
#>       A     B
#> A 0.000 1.018
#> B 1.018 0.000
#> 
#> Depth Overlap (similarity, higher = more similar):
#>       A     B
#> A 1.000 0.029
#> B 0.027 1.000
#> 
```
