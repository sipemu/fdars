# Plan 36-02 Summary — Wasserstein barycenter

**Requirement:** DENS-01 · **Wave:** 2 · **Status:** complete

## What was built

Added to `fdars-core/src/density_fda.rs`:
- `wasserstein_barycenter(density_matrix, argvals, weights) -> Result<Vec<f64>, FdarError>` — the 1D Wasserstein Fréchet mean = pointwise (weighted) quantile average Q̄(t)=Σ wᵢ Qᵢ(t), inverted back to a normalized density. Per-density: `normalize_density` → CDF via `cumulative_trapz` → quantile via `linear_interp`, averaged with uniform-or-supplied weights, then inverted (dedup + `linear_interp` + renormalize).
- Crate-root re-export of `wasserstein_barycenter`.

## Verification

- Inline tests: singleton sample reduces to the input density (L∞ small); two-density sample lies quantile-between the inputs; weighted case shifts toward the heavier density; output normalized + non-negative; error paths (empty sample, weights length mismatch / zero-sum, non-positive density, argvals mismatch).

## Notes / deviations

- Implemented in the same executor run as 36-01/36-03; reuses the 36-01 normalize/CDF/quantile spine. No new dependency. Full-crate gate result recorded in 36-03-SUMMARY.
