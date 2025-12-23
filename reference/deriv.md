# Compute functional derivative

Compute the numerical derivative of functional data. Uses finite
differences for fast computation via Rust.

## Usage

``` r
deriv(
  fdataobj,
  nderiv = 1,
  method = "diff",
  class.out = "fdata",
  nbasis = NULL,
  ...
)
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nderiv:

  Derivative order (1, 2, ...). Default is 1. For 2D data, only
  first-order derivatives are currently supported.

- method:

  Method for computing derivatives. Currently only "diff" (finite
  differences) is supported.

- class.out:

  Output class, either "fdata" or "fd". Default is "fdata".

- nbasis:

  Not used (for compatibility with fda.usc).

- ...:

  Additional arguments (ignored).

## Value

For 1D data: an 'fdata' object containing the derivative values. For 2D
data: a list with components `ds`, `dt`, and `dsdt`, each an 'fdata'
object containing the respective partial derivative.

## Details

For 1D functional data (curves), computes the nth derivative. For 2D
functional data (surfaces), computes partial derivatives:

- `ds`: partial derivative with respect to s (first argument)

- `dt`: partial derivative with respect to t (second argument)

- `dsdt`: mixed partial derivative

## Examples

``` r
# Create smooth curves
t <- seq(0, 2*pi, length.out = 100)
X <- matrix(0, 10, 100)
for (i in 1:10) X[i, ] <- sin(t + i/5)
fd <- fdata(X, argvals = t)

# First derivative (should be approximately cos)
fd_deriv <- deriv(fd, nderiv = 1)
```
