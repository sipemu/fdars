# Outlier Detection using Functional Boxplot

Detects outliers based on the functional boxplot method. Curves that
exceed the fence (1.5 times the central envelope width) at any point are
flagged as outliers.

## Usage

``` r
outliers.boxplot(
  fdataobj,
  prob = 0.5,
  factor = 1.5,
  depth.func = depth.MBD,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- prob:

  Proportion of curves for the central region (default 0.5).

- factor:

  Factor for fence calculation (default 1.5).

- depth.func:

  Depth function to use. Default is depth.MBD.

- ...:

  Additional arguments passed to depth function.

## Value

A list of class 'outliers.fdata' with components:

- outliers:

  Indices of detected outliers

- depths:

  Depth values for all curves

- cutoff:

  Not used (for compatibility)

- fdataobj:

  Original fdata object

## See also

[`boxplot.fdata`](https://sipemu.github.io/fdars/reference/boxplot.fdata.md)
for functional boxplot visualization

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

# Detect outliers
out <- outliers.boxplot(fd)
print(out)
#> Functional data outlier detection
#>   Number of observations: 30 
#>   Number of outliers: 2 
#>   Outlier indices: 29 30
#>   Depth cutoff: NA 
```
