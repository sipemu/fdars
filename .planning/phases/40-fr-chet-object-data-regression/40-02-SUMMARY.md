---
phase: 40-fr-chet-object-data-regression
plan: 02
subsystem: frechet
tags: [frechet-regression, wasserstein, global, local, kernel, object-data]

requires:
  - phase: 40-01
    provides: MetricSpace, WassersteinDensitySpace, signed_quantile_average, wasserstein2_distance
provides:
  - "frechet_global_reg (Petersen-Müller global linear weights → signed quantile average)"
  - "frechet_local_reg (local-linear product-Gaussian-kernel weights → signed quantile average)"
  - "density-response regression (the two fns specialized to the WassersteinDensitySpace density outputs)"
  - "FrechetGlobalRegResult / FrechetLocalRegResult"
affects: [40-03]

actuals:
  tokens: 15000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Signed regression weights (Petersen-Müller global-linear / local-linear correction) routed through signed_quantile_average, never wasserstein_barycenter"

key-files:
  created:
    - fdars-core/src/frechet/regression.rs
  modified:
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/frechet/space.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Global weights sᵢ(x) = 1 + (Xᵢ−X̄)ᵀΣ̂⁻¹(x−X̄) via linalg Cholesky with 1e-6 ridge; local weights sᵢ = Kᵢ(1 − (Xᵢ−x₀)ᵀμ₂⁻¹μ₁) with a product Gaussian kernel + 1e-6 ridge on μ₂."
  - "Both regressions use signed_quantile_average (sort-based isotonic projection, no osqp/QP dependency); documented divergence from R's osqp approach in rustdoc."
  - "FIX during execution: signed_quantile_average originally copied wasserstein_barycenter's rescale-to-full-support step, which stretched a narrow/spread barycenter across the whole grid and produced a large systematic W₂ bias (~0.5). Changed to invert the averaged quantile directly in x-units (with an [lb,ub] clamp) — removes the bias; regression now tracks a known predictor→density relationship symmetrically at ~0.13–0.16 W₂ (the inherent quantile→density inversion floor)."
  - "Density-response variant (FRE-01-07) is delivered by the same entry points specialized to the density space — no separate function."

patterns-established:
  - "Realistic test data must have densities that adequately fill the grid (σ≈1 on [-6,6]); narrow densities on a wide grid produce quantile-tail artifacts in the density round-trip."

requirements-completed: [FRE-01-04, FRE-01-05, FRE-01-07]

coverage:
  - id: D1
    description: "Global Fréchet regression tracks a known predictor→density relationship (density-response); signed weights never reach wasserstein_barycenter."
    requirement: "FRE-01-04, FRE-01-07"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/regression.rs::tests::global_tracks_known_relationship, global_accepts_negative_weights, global_rejects_bad_input"
        status: pass
    human_judgment: false
  - id: D2
    description: "Local (local-linear kernel-weighted) Fréchet regression tracks the truth; negative local weights accepted; bandwidth validated."
    requirement: "FRE-01-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/regression.rs::tests::local_tracks_known_relationship, local_accepts_negative_weights, local_rejects_bad_bandwidth"
        status: pass
    human_judgment: false

duration: 55min
completed: 2026-08-22
status: complete
---

# Phase 40 Plan 02: Global + Local Fréchet Regression Summary

**`frechet_global_reg` (Petersen–Müller global linear weights) and `frechet_local_reg` (local-linear Gaussian-kernel weights) predict conditional density responses at new Euclidean predictor values, both routing signed weights through the sort-based `signed_quantile_average` — never `wasserstein_barycenter`.**

## Performance

- **Duration:** ~55 min (incl. diagnosing + fixing a barycenter-inversion bias)
- **Tasks:** 3/3
- **Tests:** 6 new inline tests (18 total in module), all passing

## Accomplishments

- New `frechet/regression.rs`; `frechet_global_reg` / `frechet_local_reg` + result structs, crate-root re-exported.
- Reused `linalg` Cholesky (Σ̂⁻¹, μ₂⁻¹) with ridge regularization + `helpers::gaussian_kernel`; no new dependency.
- **Corrected `signed_quantile_average`** to invert the averaged quantile directly in x-units (dropped the rescale-to-full-support step that biased predictions ~0.5 in W₂). Regression now tracks symmetrically at ~0.13–0.16 W₂ (documented inversion floor).

## Verification

- `cargo test -p fdars-core --features linalg,parallel frechet` → 18 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → clean.
- No existing signature changed; no new dependency.

## Notes for Wave 3

- ANOVA (`anova.rs`) reuses `frechet_mean`/`frechet_variance` + the in-crate `chi_square_sf` (widen `inference::dist::mod` to `pub(crate)` — additive) + seeded permutation p-value.
