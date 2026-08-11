---
quick_id: 260811-rrm
slug: readme-sync
date: 2026-08-11
status: complete
---

# Summary: Sync README.md with current codebase

Updated `README.md` to match the post-v0.15.0 codebase. All changes verified against source.

## Changes
- **Version:** `fdars-core = "0.9"` → `"0.14"` in both install snippets (matches `fdars-core/Cargo.toml` `version = "0.14.0"`).
- **Examples:** count `27` → `28`; added the `28 | Berkeley Growth` table row (growth-curve CV case study — P-spline smoothing, GCV/AIC/BIC selection, 10-fold CV, PLS). No `*` marker (example needs no `required-features`).
- **Features / Core:** added "spline interpolation" (FEAT-01, `spline_interpolate`).
- **Features / Descriptive:** added "functional summary statistics (variance, std, covariance, depth-based median, trimmed mean)" (FEAT-02); corrected "10 depth measures" → "8" (verified 8 distinct 1D depth methods: band, fraiman_muniz, functional_spatial, kernel_functional_spatial, modal, modified_band, random_projection, rpd — matches example 05's count).

## Verification
- Version cross-checked against `Cargo.toml`; example count against `ls examples/` (28 dirs); new API names confirmed exported at crate root in `src/lib.rs`; depth count enumerated from `src/depth/`.
- Docs-only change; no source or tests touched.

## Self-Check: PASSED
