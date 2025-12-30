# Simulate Multivariate Functional Data

Generates multivariate (vector-valued) functional data where each
component is simulated via Karhunen-Loeve expansion with potentially
different eigenfunctions, eigenvalues, and domains.

## Usage

``` r
simMultiFunData(
  n,
  argvals,
  M,
  eFun.type = "Fourier",
  eVal.type = "linear",
  mean = NULL,
  seed = NULL
)
```

## Arguments

- n:

  Number of multivariate curves to generate.

- argvals:

  List of numeric vectors, one per component.

- M:

  Integer or integer vector. Number of basis functions per component. If
  a single integer, used for all components.

- eFun.type:

  Character or character vector specifying eigenfunction type for each
  component. See
  [`eFun`](https://sipemu.github.io/fdars/reference/eFun.md) for
  options.

- eVal.type:

  Character or character vector specifying eigenvalue decay for each
  component. See
  [`eVal`](https://sipemu.github.io/fdars/reference/eVal.md) for
  options.

- mean:

  List of mean functions (one per component), or `NULL`.

- seed:

  Optional integer random seed.

## Value

A list of class `multiFunData` containing:

- components:

  List of `fdata` objects, one per component

- n:

  Number of observations

- p:

  Number of components

## See also

[`simFunData`](https://sipemu.github.io/fdars/reference/simFunData.md),
[`eFun`](https://sipemu.github.io/fdars/reference/eFun.md),
[`eVal`](https://sipemu.github.io/fdars/reference/eVal.md)

## Examples

``` r
# Two-component multivariate functional data
t1 <- seq(0, 1, length.out = 100)
t2 <- seq(0, 0.5, length.out = 50)

mfd <- simMultiFunData(
  n = 20,
  argvals = list(t1, t2),
  M = c(5, 3),
  eFun.type = c("Fourier", "Wiener"),
  eVal.type = c("exponential", "linear")
)

# Plot first component
plot(mfd$components[[1]], main = "Component 1")
```
