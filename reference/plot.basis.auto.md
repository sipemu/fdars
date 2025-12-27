# Plot method for basis.auto objects

Plot method for basis.auto objects

## Usage

``` r
# S3 method for class 'basis.auto'
plot(
  x,
  which = c("all", "fourier", "pspline"),
  show.original = TRUE,
  max.curves = 20,
  ...
)
```

## Arguments

- x:

  A basis.auto object.

- which:

  Which curves to plot: "all" (default), "fourier", or "pspline".

- show.original:

  Logical. If TRUE (default), overlay original data.

- max.curves:

  Maximum number of curves to plot (default 20).

- ...:

  Additional arguments passed to ggplot.
