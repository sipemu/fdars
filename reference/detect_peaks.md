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
  smooth_nbasis = NULL
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

  Logical. If TRUE, apply Fourier basis smoothing before peak detection.
  Recommended for noisy data. Default: FALSE.

- smooth_nbasis:

  Number of Fourier basis functions for smoothing. If NULL and
  smooth_first=TRUE, uses GCV to automatically select optimal nbasis
  (range 5-25). Default: NULL (auto).

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

Fourier basis smoothing is ideal for seasonal signals because it
naturally captures periodic patterns without introducing boundary
artifacts.

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

# With automatic Fourier smoothing (GCV selects nbasis)
peaks_smooth <- detect_peaks(fd, min_distance = 1.5, smooth_first = TRUE)
```
