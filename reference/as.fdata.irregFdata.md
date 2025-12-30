# Convert Irregular Functional Data to Regular Grid

Creates a regular `fdata` object from an `irregFdata` object by
interpolating or placing `NA` at unobserved points.

## Usage

``` r
# S3 method for class 'irregFdata'
as.fdata(x, argvals = NULL, method = c("na", "linear", "nearest"), ...)

as.fdata(x, ...)
```

## Arguments

- x:

  An object of class `irregFdata`.

- argvals:

  Target regular grid. If `NULL`, uses the union of all observation
  points.

- method:

  Interpolation method:

  na

  :   (Default) Only fill exact matches; other points are `NA`

  linear

  :   Linear interpolation between observed points

  nearest

  :   Nearest neighbor interpolation

- ...:

  Additional arguments (ignored).

## Value

An `fdata` object with `NA` for unobserved points (unless interpolated).

## See also

[`sparsify`](https://sipemu.github.io/fdars/reference/sparsify.md),
[`irregFdata`](https://sipemu.github.io/fdars/reference/irregFdata.md)

## Examples

``` r
# Create sparse data
t <- seq(0, 1, length.out = 100)
fd <- simFunData(n = 10, argvals = t, M = 5, seed = 42)
ifd <- sparsify(fd, minObs = 10, maxObs = 30, seed = 123)

# Convert back to regular grid with NA
fd_na <- as.fdata(ifd)

# Convert with linear interpolation
fd_interp <- as.fdata(ifd, method = "linear")
```
