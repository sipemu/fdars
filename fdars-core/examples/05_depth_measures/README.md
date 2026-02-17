# Example 05: Functional Depth Measures

## What this demonstrates

Depth measures assign a centrality score to each curve in a dataset. Central curves have high depth; outlying curves have low depth. This example computes eight different depth measures and shows how they rank curves, with particular attention to identifying injected outliers.

Different depth measures capture different notions of centrality — band-based depths focus on pointwise containment, while projection-based depths consider the curve as a whole.

## API functions used

- `depth::fraiman_muniz_1d()` — integrated univariate depth
- `depth::band_1d()` — band depth (proportion of bands containing the curve)
- `depth::modified_band_1d()` — proportion-of-time within bands
- `depth::modal_1d()` — kernel-based density depth
- `depth::random_projection_1d()` — depth via random linear projections
- `depth::random_tukey_1d()` — halfspace depth via random projections
- `depth::functional_spatial_1d()` — spatial (geometric) depth
- `depth::modified_epigraph_index_1d()` — modified epigraph index

## How to run

```bash
cargo run --example depth_measures
```

## Expected output

For each depth measure: the 3 deepest and 3 shallowest curves. A summary showing which methods successfully rank injected outliers at the bottom.

## Key concepts

- **Band depth**: counts how often a curve lies within the band formed by J other curves
- **Modified band depth**: fraction of the domain where the curve is inside the band
- **Fraiman-Muniz**: integrates univariate depths over the domain
- **Halfspace/Tukey depth**: minimum fraction of data on one side of any hyperplane through the curve
