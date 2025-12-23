# Convert DataFrame to 2D functional data

Converts a data frame in long format to a 2D fdata object (surfaces).
The expected format is: one identifier column, one column for the
s-dimension index, and multiple columns for the t-dimension values.

## Usage

``` r
df_to_fdata2d(
  df,
  id_col = 1,
  s_col = 2,
  t_cols = NULL,
  names = NULL,
  metadata = NULL
)
```

## Arguments

- df:

  A data frame with the structure described below.

- id_col:

  Name or index of the identifier column (default: 1).

- s_col:

  Name or index of the s-dimension column (default: 2).

- t_cols:

  Names or indices of the t-dimension value columns. If NULL (default),
  uses all columns after `s_col`.

- names:

  Optional list with 'main', 'xlab', 'ylab', 'zlab' for labels.

- metadata:

  Optional data.frame with additional covariates (one row per surface).
  If metadata has an "id" column or non-default row names, they must
  match the surface identifiers from `id_col`.

## Value

An object of class 'fdata' with 2D functional data.

## Details

The expected data frame structure is:

- Column 1 (id_col): Surface identifier (e.g., "surface_1", "surface_2")

- Column 2 (s_col): Index for the s-dimension (row index of surface)

- Columns 3+ (t_cols): Values for each t-dimension point (columns of
  surface)

Each unique identifier represents one surface. For each surface, there
should be m1 rows (one per s-value), and m2 t-columns, resulting in an
m1 x m2 surface.

## Examples

``` r
# Create example data frame
df <- data.frame(
  id = rep(c("surf1", "surf2"), each = 5),
  s = rep(1:5, 2),
  t1 = rnorm(10),
  t2 = rnorm(10),
  t3 = rnorm(10)
)
fd <- df_to_fdata2d(df)
print(fd)
#> Functional data object
#>   Type: 2D (surface) 
#>   Number of observations: 2 
#>   Grid dimensions: 5 x 3 
#>   Range s: 1 - 5 
#>   Range t: 1 - 3 

# With metadata
meta <- data.frame(group = c("A", "B"), value = c(1.5, 2.3))
fd <- df_to_fdata2d(df, metadata = meta)
```
