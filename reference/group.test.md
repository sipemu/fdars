# Permutation Test for Group Differences

Tests whether groups of functional data are significantly different
using permutation testing.

## Usage

``` r
group.test(
  fdataobj,
  groups,
  n.perm = 1000,
  statistic = c("centroid", "ratio"),
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- groups:

  A factor or character vector specifying group membership.

- n.perm:

  Number of permutations (default 1000).

- statistic:

  Test statistic: "centroid" (distance between group means) or "ratio"
  (between/within group variance ratio).

- ...:

  Additional arguments passed to distance functions.

## Value

An object of class 'group.test' containing:

- statistic:

  Observed test statistic

- p.value:

  Permutation p-value

- perm.dist:

  Permutation distribution of test statistic

- n.perm:

  Number of permutations used

## Details

**Null Hypothesis (H0):** All groups come from the same distribution.
That is, the group labels are exchangeable and there is no systematic
difference between the functional curves in different groups.

**Alternative Hypothesis (H1):** At least one group differs from the
others in terms of location (mean function) or dispersion.

The test works by:

1.  Computing a test statistic on the observed data

2.  Repeatedly permuting the group labels and recomputing the statistic

3.  Calculating the p-value as the proportion of permuted statistics \>=
    observed

Two test statistics are available:

- `"centroid"`: Sum of pairwise L2 distances between group mean
  functions. Sensitive to differences in group locations (means).

- `"ratio"`: Ratio of between-group to within-group variance, similar to
  an F-statistic. Sensitive to both location and dispersion.

A small p-value (e.g., \< 0.05) indicates evidence against H0,
suggesting that the groups are significantly different.

## Examples

``` r
if (FALSE) { # \dontrun{
set.seed(42)
n <- 30
m <- 50
t_grid <- seq(0, 1, length.out = m)
X <- matrix(0, n, m)
for (i in 1:15) X[i, ] <- sin(2 * pi * t_grid) + rnorm(m, sd = 0.1)
for (i in 16:30) X[i, ] <- cos(2 * pi * t_grid) + rnorm(m, sd = 0.1)
fd <- fdata(X, argvals = t_grid)
groups <- factor(rep(c("A", "B"), each = 15))

# Test for significant difference
gt <- group.test(fd, groups, n.perm = 500)
print(gt)
} # }
```
