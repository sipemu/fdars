# Plot method for fdata objects

Displays a plot of functional data. For 1D functional data, plots curves
as lines with optional coloring. For 2D functional data, plots surfaces
as heatmaps with contour lines.

## Usage

``` r
# S3 method for class 'fdata'
plot(
  x,
  color = NULL,
  alpha = NULL,
  show.mean = FALSE,
  show.ci = FALSE,
  ci.level = 0.9,
  palette = NULL,
  ...
)
```

## Arguments

- x:

  An object of class 'fdata'.

- color:

  Optional vector for coloring curves. Can be:

  - Numeric vector: curves colored by continuous scale (viridis)

  - Factor/character: curves colored by discrete groups

  Must have length equal to number of curves.

- alpha:

  Transparency of individual curve lines. Default is 0.7 for basic
  plots, but automatically reduced to 0.3 when `show.mean = TRUE` or
  `show.ci = TRUE` to reduce visual clutter and allow mean curves to
  stand out. Can be explicitly set to override the default.

- show.mean:

  Logical. If TRUE and color is categorical, overlay group mean curves
  with thicker lines (default FALSE).

- show.ci:

  Logical. If TRUE and color is categorical, show pointwise confidence
  interval ribbons per group (default FALSE).

- ci.level:

  Confidence level for CI ribbons (default 0.90 for 90 percent).

- palette:

  Optional named vector of colors for categorical coloring, e.g., c("A"
  = "blue", "B" = "red").

- ...:

  Additional arguments (currently ignored).

## Value

The ggplot object (invisibly).

## Details

This function displays the plot immediately. To get the ggplot object
without displaying (e.g., for customization), use
[`autoplot.fdata`](https://sipemu.github.io/fdars/reference/autoplot.fdata.md).

## Examples

``` r
library(ggplot2)
# Display plot immediately
fd <- fdata(matrix(rnorm(200), 20, 10))
plot(fd)


# To get ggplot object without displaying, use autoplot:
p <- autoplot(fd)
```
