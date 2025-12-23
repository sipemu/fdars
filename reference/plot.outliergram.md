# Plot Method for Outliergram Objects

Creates a scatter plot of MEI vs MBD with the parabolic boundary and
identified outliers highlighted.

## Usage

``` r
# S3 method for class 'outliergram'
plot(
  x,
  col_normal = "gray60",
  col_outlier = "red",
  color_by_type = FALSE,
  show_parabola = TRUE,
  show_threshold = TRUE,
  label = "index",
  label_all = FALSE,
  ...
)
```

## Arguments

- x:

  An object of class 'outliergram'.

- col_normal:

  Color for normal observations. Default is "gray60".

- col_outlier:

  Color for outliers (used when `color_by_type = FALSE`). Default is
  "red".

- color_by_type:

  Logical. If TRUE, color outliers by their type (shape, magnitude_high,
  magnitude_low, mixed). Default is FALSE.

- show_parabola:

  Logical. If TRUE, draw the theoretical parabola. Default TRUE.

- show_threshold:

  Logical. If TRUE, draw the adjusted threshold parabola. Default TRUE.

- label:

  What to use for labeling outlier points. Options:

  - `"index"`: Use numeric indices (default)

  - `"id"`: Use observation IDs from the fdata object

  - A column name from the fdata metadata (e.g., `"patient_id"`)

- label_all:

  Logical. If TRUE, label all points, not just outliers. Default FALSE.

- ...:

  Additional arguments passed to plotting functions.
