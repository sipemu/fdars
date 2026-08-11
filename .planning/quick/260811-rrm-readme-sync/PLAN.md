---
quick_id: 260811-rrm
slug: readme-sync
date: 2026-08-11
---

# Quick Task: Sync README.md with current codebase

Bring the top-level `README.md` back in line with the code after the v0.15.0 milestone.

## Tasks
1. Bump install version `0.9` → `0.14` in both dependency snippets (crate is at 0.14.0).
2. Fix example count `27` → `28` and add the missing row for example 28 (`28_berkeley_growth`).
3. Add v0.15.0 public API to the Features table: spline interpolation (`spline_interpolate`, FEAT-01) → Core row; functional summary statistics (`functional_variance/std/covariance`, `depth_based_median`, `trim_mean`, FEAT-02) → Descriptive row.
4. Reconcile the "10 depth measures" vs "8 depth measures" discrepancy — verified 8 distinct 1D depth methods in `src/depth/`, so the Features row's "10" was wrong; corrected to "8".
