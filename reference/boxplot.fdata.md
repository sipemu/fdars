# Functional Boxplot

Creates a functional boxplot for visualizing the distribution of
functional data. The boxplot shows the median curve, central 50 percent
envelope, fence (equivalent to whiskers), and outliers.

## Usage

``` r
# S3 method for class 'fdata'
boxplot(
  x,
  prob = 0.5,
  factor = 1.5,
  depth.func = depth.MBD,
  show.outliers = TRUE,
  col.median = "black",
  col.envelope = "magenta",
  col.fence = "pink",
  col.outliers = "red",
  ...
)
```

## Arguments

- x:

  An object of class 'fdata'.

- prob:

  Proportion of curves for the central region (default 0.5 for 50
  percent).

- factor:

  Factor for fence calculation (default 1.5, as in standard boxplots).

- depth.func:

  Depth function to use. Default is depth.MBD.

- show.outliers:

  Logical. If TRUE (default), show outlier curves.

- col.median:

  Color for median curve (default "black").

- col.envelope:

  Color for central envelope (default "magenta").

- col.fence:

  Color for fence region (default "pink").

- col.outliers:

  Color for outlier curves (default "red").

- ...:

  Additional arguments passed to depth function.

## Value

A list of class 'fbplot' with components:

- median:

  Index of the median curve

- central:

  Indices of curves in the central region

- outliers:

  Indices of outlier curves

- depth:

  Depth values for all curves

- plot:

  The ggplot object

## Details

The functional boxplot (Sun & Genton, 2011) generalizes the standard
boxplot to functional data using depth ordering:

- **Median**: The curve with maximum depth

- **Central region**: Envelope of curves with top 50 percent depth

- **Fence**: 1.5 times the envelope width beyond the central region

- **Outliers**: Curves that exceed the fence at any point

## References

Sun, Y. and Genton, M.G. (2011). Functional boxplots. *Journal of
Computational and Graphical Statistics*, 20(2), 316-334.

## See also

[`depth.MBD`](https://sipemu.github.io/fdars/reference/depth.MBD.md) for
the default depth function,
[`outliers.boxplot`](https://sipemu.github.io/fdars/reference/outliers.boxplot.md)
for outlier detection using functional boxplots

## Examples

``` r
# Create functional data with outliers
set.seed(42)
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:28) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
X[29, ] <- sin(2*pi*t) + 2  # Magnitude outlier
X[30, ] <- cos(2*pi*t)       # Shape outlier
fd <- fdata(X, argvals = t)

# Create functional boxplot
fbp <- boxplot(fd)
```
