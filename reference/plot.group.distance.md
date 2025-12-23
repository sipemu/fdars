# Plot method for group.distance

Plot method for group.distance

## Usage

``` r
# S3 method for class 'group.distance'
plot(x, type = c("heatmap", "dendrogram"), which = NULL, ...)
```

## Arguments

- x:

  An object of class 'group.distance'.

- type:

  Plot type: "heatmap" or "dendrogram".

- which:

  Which distance matrix to plot. If NULL (default), uses the first
  available matrix from the group.distance object.

- ...:

  Additional arguments.

## Value

A ggplot object (for heatmap) or NULL (for dendrogram, uses base
graphics).
