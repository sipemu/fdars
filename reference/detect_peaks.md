# Detect Peaks in Functional Data

Detects local maxima (peaks) in functional data using derivative
zero-crossings. Returns peak times, values, and prominence measures.

## Usage

``` r
detect_peaks(
  fdataobj,
  min_distance = NULL,
  min_prominence = NULL,
  smooth_first = FALSE,
  smooth_lambda = 10
)
```

## Arguments

- fdataobj:

  An fdata object.

- min_distance:

  Minimum time between peaks. Default: NULL (no constraint).

- min_prominence:

  Minimum prominence for a peak (0-1 scale). Peaks with lower prominence
  are filtered out. Default: NULL (no filter).

- smooth_first:

  Logical. If TRUE, apply P-spline smoothing before peak detection.
  Default: FALSE.

- smooth_lambda:

  Smoothing parameter for P-splines. Default: 10.

## Value

A list with components:

- peaks:

  List of data frames, one per curve, with columns: time, value,
  prominence

- inter_peak_distances:

  List of numeric vectors with distances between consecutive peaks

- mean_period:

  Mean inter-peak distance across all curves (estimates period)

## Details

Peak prominence measures how much a peak stands out from its
surroundings. It is computed as the height difference between the peak
and the highest of the two minimum values on either side, normalized by
the data range.

## Examples

``` r
# Generate data with clear peaks
t <- seq(0, 10, length.out = 200)
X <- matrix(sin(2 * pi * t / 2), nrow = 1)
fd <- fdata(X, argvals = t)

# Detect peaks
peaks <- detect_peaks(fd, min_distance = 1.5)
print(peaks$mean_period)  # Should be close to 2
#> [1] 2.01005
```
