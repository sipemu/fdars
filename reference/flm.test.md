# Statistical Tests for Functional Data

Functions for hypothesis testing with functional data. Test for
Functional Linear Model

## Usage

``` r
flm.test(fdataobj, y, B = 500, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata' (functional covariate).

- y:

  Response vector.

- B:

  Number of bootstrap samples for p-value computation.

- ...:

  Additional arguments.

## Value

A list of class 'htest' with components:

- statistic:

  The test statistic

- p.value:

  Bootstrap p-value

- method:

  Name of the test

## Details

Tests the goodness-of-fit for a functional linear model using the
projected Cramer-von Mises statistic.

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))
y <- rnorm(20)
# test_result <- flm.test(fd, y, B = 100)
```
