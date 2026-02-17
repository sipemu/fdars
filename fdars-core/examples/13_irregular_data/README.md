# Example 13: Irregular Functional Data

## What this demonstrates

Working with functional data observed at different time points for each curve (irregular sampling). The `IrregFdata` struct uses CSR-like (Compressed Sparse Row) storage for efficient access. This example shows construction, querying, integration, norms, cross-curve mean estimation, pairwise distances, and regularization to a common grid.

Irregular data is common in practice — clinical visits, sensor readings at irregular intervals, or longitudinal studies where follow-up times vary.

## API functions used

- `irreg_fdata::IrregFdata::from_lists()` — construct from per-curve vectors
- `irreg_fdata::integrate_irreg()` — trapezoidal integration per curve
- `irreg_fdata::norm_lp_irreg()` — Lp norms for irregular data
- `irreg_fdata::mean_irreg()` — kernel-smoothed cross-curve mean
- `irreg_fdata::metric_lp_irreg()` — pairwise Lp distances
- `irreg_fdata::to_regular_grid()` — interpolate to a common regular grid

## How to run

```bash
cargo run --example irregular_data
```

## Expected output

Observation counts per curve, CSR storage details, integrals, norms, estimated mean function, pairwise distances, and regularized data on a common grid.

## Key concepts

- **CSR-like storage**: offsets array + flat argvals/values arrays for cache-friendly access
- **Kernel mean estimation**: weighted average across curves using kernel weights at each target point
- **Regularization**: interpolate irregular observations to a common grid for standard FDA methods
- **Range validation**: `IrregFdata` tracks the domain range for consistent analysis
