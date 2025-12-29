# Depth Functions for Functional Data

Functions for computing various depth measures for functional data.
Compute Functional Data Depth

## Usage

``` r
depth(
  fdataobj,
  fdataori = NULL,
  method = c("FM", "mode", "RP", "RT", "BD", "MBD", "MEI", "FSD", "KFSD", "RPD"),
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata' to compute depth for.

- fdataori:

  An object of class 'fdata' as reference sample. If NULL, uses fdataobj
  as reference.

- method:

  Depth method to use. One of "FM" (Fraiman-Muniz), "mode" (modal), "RP"
  (random projection), "RT" (random Tukey), "BD" (band depth), "MBD"
  (modified band depth), "MEI" (modified epigraph index), "FSD"
  (functional spatial depth), "KFSD" (kernel functional spatial depth),
  or "RPD" (random projection with derivatives). Default is "FM".

- ...:

  Additional arguments passed to the specific depth function.

## Value

A numeric vector of depth values, one per curve in fdataobj.

## Details

Unified interface for computing various depth measures for functional
data.

Available methods:

- FM:

  Fraiman-Muniz depth - integrates univariate depths over domain

- mode:

  Modal depth - based on kernel density estimation

- RP:

  Random projection depth - projects to random directions

- RT:

  Random Tukey depth - halfspace depth via random projections

- BD:

  Band depth - proportion of bands containing the curve (1D only)

- MBD:

  Modified band depth - allows partial containment (1D only)

- MEI:

  Modified epigraph index - proportion of time below other curves (1D
  only)

- FSD:

  Functional spatial depth - based on spatial signs

- KFSD:

  Kernel functional spatial depth - smoothed FSD

- RPD:

  Random projection with derivatives - includes curve derivatives

## Examples

``` r
fd <- fdata(matrix(rnorm(100), 10, 10))

# Different depth methods
depth(fd, method = "FM")
#>  [1] 0.48 0.46 0.40 0.66 0.52 0.36 0.52 0.66 0.52 0.42
depth(fd, method = "mode")
#>  [1] 0.2714079 0.1717654 0.2102241 0.2917229 0.2032817 0.1923078 0.2250097
#>  [8] 0.2894317 0.2493039 0.1962121
depth(fd, method = "RP")
#>  [1] 0.2890909 0.2400000 0.2981818 0.3145455 0.2436364 0.2800000 0.2581818
#>  [8] 0.2818182 0.2581818 0.2636364
```
