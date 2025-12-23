# Magnitude-Shape Outlier Detection for Functional Data

Performs Magnitude-Shape (MS) outlier detection for functional data.
Each curve is represented as a point in 2D space where the x-axis
represents magnitude outlyingness and the y-axis represents shape
outlyingness.

## Usage

``` r
magnitudeshape(
  fdataobj,
  depth.func = depth.MBD,
  cutoff.quantile = 0.993,
  col.normal = "black",
  col.outliers = "red",
  label = "index",
  label_all = FALSE,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- depth.func:

  Depth function to use for computing outlyingness. Default is
  depth.MBD.

- cutoff.quantile:

  Quantile for outlier cutoff (default 0.993).

- col.normal:

  Color for normal curves (default "black").

- col.outliers:

  Color for outlier curves (default "red").

- label:

  What to use for labeling outlier points. Options:

  - `"index"`: Use numeric indices (default)

  - `"id"`: Use observation IDs from the fdata object

  - A column name from the fdata metadata (e.g., `"patient_id"`)

  - `NULL`: No labels

- label_all:

  Logical. If TRUE, label all points, not just outliers. Default FALSE.

- ...:

  Additional arguments passed to depth function.

## Value

A list of class 'magnitudeshape' with components:

- MO:

  Magnitude outlyingness values

- VO:

  Shape (variability) outlyingness values

- outliers:

  Indices of detected outliers

- cutoff:

  Chi-squared cutoff value used

- plot:

  The ggplot object

## Details

The MS plot (Dai & Genton, 2019) decomposes functional outlyingness
into:

- **Magnitude Outlyingness (MO)**: Based on pointwise median of
  directional outlyingness - captures shift outliers

- **Shape Outlyingness (VO)**: Based on variability of directional
  outlyingness - captures shape outliers

Outliers are detected using the chi-squared distribution with cutoff at
the specified quantile.

## References

Dai, W. and Genton, M.G. (2019). Directional outlyingness for
multivariate functional data. *Computational Statistics & Data
Analysis*, 131, 50-65.

## Examples

``` r
# Create functional data with outliers
set.seed(42)
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 30, 50)
for (i in 1:28) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.2)
X[29, ] <- sin(2*pi*t) + 2  # Magnitude outlier
X[30, ] <- sin(4*pi*t)       # Shape outlier
fd <- fdata(X, argvals = t)

# Create MS plot
ms <- magnitudeshape(fd)

# With IDs and metadata
fd <- fdata(X, argvals = t, id = paste0("curve_", 1:30))
ms <- magnitudeshape(fd, label = "id")
```
