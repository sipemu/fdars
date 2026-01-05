# CFDAutoperiod: Clustered Filtered Detrended Autoperiod

Implements the CFDAutoperiod algorithm (Puech et al. 2020) which applies
first-order differencing for detrending, then uses density-based
clustering of spectral peaks and ACF validation on the original signal.

## Usage

``` r
cfd.autoperiod(fdataobj, cluster_tolerance = 0.1, min_cluster_size = 1)
```

## Arguments

- fdataobj:

  An fdata object.

- cluster_tolerance:

  Relative tolerance for clustering nearby period candidates. Default:
  0.1 (10% relative difference). Candidates within this tolerance are
  grouped into clusters.

- min_cluster_size:

  Minimum number of candidates required to form a valid cluster.
  Default: 1.

## Value

A list of class "cfd_autoperiod_result" with components:

- period:

  Primary detected period (best validated cluster center)

- confidence:

  Combined confidence (ACF validation \* spectral power)

- acf_validation:

  ACF validation score for the primary period

- n_periods:

  Number of validated period clusters

- periods:

  All detected period cluster centers

- confidences:

  Confidence scores for each detected period

## Details

The CFDAutoperiod algorithm works in five stages:

1.  **Differencing**: First-order differencing removes polynomial trends

2.  **FFT**: Computes periodogram on the detrended signal

3.  **Peak Detection**: Finds all peaks above the noise floor

4.  **Clustering**: Groups nearby period candidates into clusters

5.  **ACF Validation**: Validates cluster centers using the ACF of the
    original (non-differenced) signal

This method is particularly effective for:

- Signals with strong polynomial trends

- Multiple concurrent periodicities

- Signals where spectral leakage creates nearby spurious peaks

## References

Puech, T., Boussard, M., D'Amato, A., & Millerand, G. (2020). A fully
automated periodicity detection in time series. In Advanced Analytics
and Learning on Temporal Data (pp. 43-54). Springer.

## See also

[`autoperiod`](https://sipemu.github.io/fdars/reference/autoperiod.md)
for the original Autoperiod algorithm,
[`sazed`](https://sipemu.github.io/fdars/reference/sazed.md) for an
ensemble method

## Examples

``` r
# Generate data with trend
t <- seq(0, 20, length.out = 400)
X <- matrix(0.2 * t + sin(2 * pi * t / 2), nrow = 1)
fd <- fdata(X, argvals = t)

# CFDAutoperiod handles trends via differencing
result <- cfd.autoperiod(fd)
print(result)
#> CFDAutoperiod Detection
#> -----------------------
#> Primary Period: 2.0000
#> Confidence:     0.8939
#> ACF Validation: 0.8939
#> Periods Found:  1

# Multiple periods detected
X2 <- matrix(sin(2 * pi * t / 2) + 0.5 * sin(2 * pi * t / 5), nrow = 1)
fd2 <- fdata(X2, argvals = t)
result2 <- cfd.autoperiod(fd2)
print(result2$periods)  # All detected periods
#> [1] 2
```
