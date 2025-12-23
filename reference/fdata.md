# Create a functional data object

Creates an fdata object for 1D functional data (curves) or 2D functional
data (surfaces). For 2D data, the internal storage uses a flattened
matrix format `[n, m1*m2]` where each row represents a surface stored in
row-major order.

## Usage

``` r
fdata(
  mdata,
  argvals = NULL,
  rangeval = NULL,
  names = NULL,
  fdata2d = FALSE,
  id = NULL,
  metadata = NULL
)
```

## Arguments

- mdata:

  Input data. Can be:

  - For 1D: A matrix `[n, m]` where n is number of curves, m is number
    of evaluation points

  - For 2D: A 3D array `[n, m1, m2]` where n is number of surfaces, m1 x
    m2 is the grid size. Automatically detected and converted to
    flattened storage.

  - For 2D: A matrix `[n, m1*m2]` (already flattened) with argvals
    specifying grid dimensions

  - For 2D: A single surface matrix `[m1, m2]` with argvals specifying
    grid dimensions

- argvals:

  Evaluation points. For 1D: a numeric vector. For 2D: a list with two
  numeric vectors specifying the s and t coordinates.

- rangeval:

  Range of the argument values. For 1D: a numeric vector of length 2.
  For 2D: a list with two numeric vectors of length 2.

- names:

  List with components 'main', 'xlab', 'ylab' for plot titles. For 2D,
  also 'zlab' for the surface value label.

- fdata2d:

  Logical. If TRUE, create 2D functional data (surface). Automatically
  set to TRUE if mdata is a 3D array.

- id:

  Optional character vector of identifiers for each observation. If
  NULL, uses row names of mdata or generates "obs_1", "obs_2", etc.

- metadata:

  Optional data.frame with additional covariates (one row per
  observation). If metadata has an "id" column or non-default row names,
  they must match the `id` parameter.

## Value

An object of class 'fdata' containing:

- data:

  The data matrix. For 2D: flattened `[n, m1*m2]` format

- argvals:

  Evaluation points

- rangeval:

  Range of arguments

- names:

  Plot labels

- fdata2d:

  Logical indicating if 2D

- dims:

  For 2D only: `c(m1, m2)` grid dimensions

- id:

  Character vector of observation identifiers

- metadata:

  Data frame of additional covariates (or NULL)

## Details

For 2D functional data, surfaces are stored internally as a flattened
matrix where each row is a surface in row-major order. To extract a
single surface as a matrix, use subsetting with `drop = TRUE` or reshape
manually:

    # Extract surface i as matrix
    surface_i <- fd[i, drop = TRUE]
    # Or manually:
    surface_i <- matrix(fd$data[i, ], nrow = fd$dims[1], ncol = fd$dims[2])

## Examples

``` r
# Create 1D functional data (curves)
x <- matrix(rnorm(100), nrow = 10, ncol = 10)
fd <- fdata(x, argvals = seq(0, 1, length.out = 10))

# Create with identifiers and metadata
meta <- data.frame(group = rep(c("A", "B"), 5), endpoint = rnorm(10))
fd <- fdata(x, id = paste0("patient_", 1:10), metadata = meta)

# Access metadata
fd$id
#>  [1] "patient_1"  "patient_2"  "patient_3"  "patient_4"  "patient_5" 
#>  [6] "patient_6"  "patient_7"  "patient_8"  "patient_9"  "patient_10"
fd$metadata$group
#>  [1] "A" "B" "A" "B" "A" "B" "A" "B" "A" "B"

# Create 2D functional data from 3D array [n, m1, m2]
surfaces <- array(rnorm(500), dim = c(5, 10, 10))
fd2d <- fdata(surfaces)

# Access individual surface as matrix
surface_1 <- fd2d[1, drop = TRUE]
```
