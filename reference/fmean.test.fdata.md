# Test for Equality of Functional Means

Tests whether the mean function equals a specified value.

## Usage

``` r
fmean.test.fdata(fdataobj, mu0 = NULL, B = 500, ...)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- mu0:

  Hypothesized mean function (vector). If NULL, tests against zero.

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

## Examples

``` r
fd <- fdata(matrix(rnorm(200), 20, 10))
# test_result <- fmean.test.fdata(fd, B = 100)
```
