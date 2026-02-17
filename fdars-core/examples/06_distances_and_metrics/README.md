# Example 06: Distances and Metrics

## What this demonstrates

Computing pairwise distance matrices between functional observations using various metrics. Different metrics capture different aspects of similarity — Lp distances measure pointwise closeness, Hausdorff measures maximum deviation, DTW accounts for time warping, and Fourier/h-shift metrics focus on frequency content or phase alignment.

## API functions used

- `metric::lp_self_1d()` — pairwise Lp distances within a dataset
- `metric::lp_cross_1d()` — Lp distances between two datasets
- `metric::hausdorff_self_1d()` — pairwise Hausdorff distances
- `metric::dtw_self_1d()` — Dynamic Time Warping distances
- `metric::fourier_self_1d()` — Fourier-based semimetric
- `metric::hshift_self_1d()` — horizontal shift semimetric

## How to run

```bash
cargo run --example distances_and_metrics
```

## Expected output

Distance matrices (upper triangle) for each metric, a cross-distance matrix, and a comparison of all metrics for the same pair of curves.

## Key concepts

- **Lp distance**: d_p(f,g) = (∫|f(t)-g(t)|^p dt)^(1/p)
- **Hausdorff distance**: maximum over all points of the minimum distance to the other curve
- **DTW**: aligns curves by warping the time axis to minimize total distance
- **Fourier semimetric**: distance based on first k Fourier coefficients
- **Horizontal shift**: minimum Lp distance over horizontal translations
