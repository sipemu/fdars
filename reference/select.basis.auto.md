# Automatic Per-Curve Basis Type and Number Selection

Selects the optimal basis type (Fourier or P-spline) and number of basis
functions for each curve individually using model selection criteria.
This is useful when working with mixed datasets containing both seasonal
and non-seasonal curves.

## Usage

``` r
select.basis.auto(
  fdataobj,
  criterion = c("GCV", "AIC", "BIC"),
  nbasis.range = NULL,
  lambda.pspline = NULL,
  use.seasonal.hint = TRUE
)
```

## Arguments

- fdataobj:

  An fdata object.

- criterion:

  Model selection criterion: "GCV" (default), "AIC", or "BIC".

- nbasis.range:

  Optional numeric vector of length 2 specifying `c(min, max)` for
  nbasis search range. If NULL, automatic ranges are used: Fourier 3-25,
  P-spline 6-40 (or limited by data length).

- lambda.pspline:

  Smoothing parameter for P-splines. If NULL (default), lambda is
  automatically selected from a grid for each curve.

- use.seasonal.hint:

  Logical. If TRUE (default), uses FFT-based seasonality detection to
  inform basis preference. Seasonal curves start Fourier search from 5
  basis functions.

## Value

A list of class "basis.auto" with:

- basis.type:

  Character vector ("pspline" or "fourier") for each curve

- nbasis:

  Integer vector of selected nbasis per curve

- score:

  Numeric vector of best criterion scores

- coefficients:

  List of coefficient vectors for each curve

- fitted:

  fdata object with fitted values

- edf:

  Numeric vector of effective degrees of freedom

- seasonal.detected:

  Logical vector indicating detected seasonality

- lambda:

  Numeric vector of lambda values (NA for Fourier curves)

- criterion:

  Character string of criterion used

- original:

  Original fdata object

## Details

For each curve, the function searches over:

- Fourier basis: odd nbasis values from 3 (or 5 if seasonal) to min(m/3,
  25)

- P-spline basis: nbasis from 6 to min(m/2, 40), with lambda from grid
  {0.001, 0.01, 0.1, 1, 10, 100} if lambda.pspline is NULL

The function uses parallel processing (via Rust/rayon) for efficiency
when processing multiple curves.

## See also

[`fdata2basis.cv`](https://sipemu.github.io/fdars/reference/fdata2basis.cv.md)
for global basis selection,
[`pspline`](https://sipemu.github.io/fdars/reference/pspline.md) for
P-spline fitting

## Examples

``` r
# Generate mixed data: some seasonal, some polynomial
set.seed(42)
t <- seq(0, 10, length.out = 100)

# 3 seasonal curves
X_seasonal <- matrix(0, 3, 100)
for (i in 1:3) {
  X_seasonal[i, ] <- sin(2 * pi * t / 2.5) + rnorm(100, sd = 0.2)
}

# 3 polynomial curves
X_poly <- matrix(0, 3, 100)
for (i in 1:3) {
  X_poly[i, ] <- 0.1 * t^2 - t + rnorm(100, sd = 0.5)
}

fd <- fdata(rbind(X_seasonal, X_poly), argvals = t)

# Auto-select optimal basis for each curve
result <- select.basis.auto(fd)
print(result)
#> Automatic Basis Selection Results
#> ==================================
#> Curves: 6 
#> Criterion: GCV 
#> 
#> Basis type distribution:
#>   Fourier: 3 (50%)
#>   P-spline:3 (50%)
#> 
#> Mean GCV score: 0.1484 
#> Mean EDF: 6.86 
#> Seasonal detected: 6 curves

# Should detect: first 3 as Fourier, last 3 as P-spline
table(result$basis.type)
#> 
#> fourier pspline 
#>       3       3 
```
