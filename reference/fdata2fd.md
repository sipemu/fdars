# Convert Functional Data to fd class

Converts an fdata object to an fd object from the fda package. Requires
the fda package to be installed.

## Usage

``` r
fdata2fd(fdataobj, nbasis = 10, type = c("bspline", "fourier"))
```

## Arguments

- fdataobj:

  An object of class 'fdata'.

- nbasis:

  Number of basis functions (default 10).

- type:

  Type of basis: "bspline" (default) or "fourier".

## Value

An object of class 'fd' from the fda package.

## Examples

``` r
if (FALSE) { # \dontrun{
t <- seq(0, 1, length.out = 50)
X <- matrix(0, 20, 50)
for (i in 1:20) X[i, ] <- sin(2*pi*t) + rnorm(50, sd = 0.1)
fd <- fdata(X, argvals = t)
fd_obj <- fdata2fd(fd, nbasis = 10)
} # }
```
