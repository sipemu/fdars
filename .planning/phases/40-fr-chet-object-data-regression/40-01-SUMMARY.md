---
phase: 40-fr-chet-object-data-regression
plan: 01
subsystem: frechet
tags: [metric-space, wasserstein, frechet-mean, object-data, density]

requires:
  - phase: 36-density-fda
    provides: density_fda wasserstein_barycenter + density→quantile→density back-map helpers
provides:
  - "MetricSpace trait (distance + weighted_frechet_mean) — Send+Sync, generic backend for regression/ANOVA"
  - "WassersteinDensitySpace backend (densities on a shared grid)"
  - "public wasserstein2_distance (quantile-L2 1D 2-Wasserstein distance)"
  - "generic frechet_mean + frechet_variance"
  - "pub(crate) signed_quantile_average (signed-weight sort-based isotonic projection, no osqp) — for Wave 2 regression"
affects: [40-02, 40-03]

actuals:
  tokens: 16000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "MetricSpace trait abstraction generic over object-data spaces"
    - "Signed-weight quantile average + sort-based isotonic projection (zero-dependency alternative to R's osqp QP)"

key-files:
  created:
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/frechet/space.rs
    - fdars-core/src/frechet/mean.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/density_fda.rs

key-decisions:
  - "WassersteinDensitySpace::weighted_frechet_mean delegates to density_fda::wasserstein_barycenter (non-negative weights); it never calls signed_quantile_average."
  - "wasserstein2_distance = L2 distance of quantile functions, reusing the density→CDF→quantile step from density_fda."
  - "signed_quantile_average (pub(crate)) computes a SIGNED weighted quantile average then sorts (isotonic projection) — the no-osqp path for regression weights that can be negative. Widened density_fda's dedup_adjacent + quantile_density_from_q to pub(crate) (additive) to reuse the back-map."
  - "Documented tolerance: the barycenter density→quantile→density round-trip has an inherent ~0.09 W2 floor; identical-object recovery is asserted within 0.15 W2 and identical-sample variance within 0.02 (both far below dispersion-scale variances). frechet_mean agrees EXACTLY with wasserstein_barycenter (same call)."

patterns-established:
  - "Generic Fréchet statistics (mean/variance) over any MetricSpace; density backend is the first impl (FRE-02 will add more)."

requirements-completed: [FRE-01-01, FRE-01-02, FRE-01-03, FRE-01-06]

coverage:
  - id: D1
    description: "MetricSpace trait + WassersteinDensitySpace backend; public crate-root re-exported entry points."
    requirement: "FRE-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/space.rs::tests::space_new_validates_grid, distance_delegates_to_w2"
        status: pass
    human_judgment: false
  - id: D2
    description: "Fréchet mean (agrees exactly with wasserstein_barycenter) + Fréchet variance (≈0 for identical sample, grows with dispersion)."
    requirement: "FRE-01-02, FRE-01-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/mean.rs::tests::mean_of_identical_recovers_object, variance_zero_for_identical_sample, variance_grows_with_dispersion; space.rs::weighted_frechet_mean_of_identical_recovers_object"
        status: pass
    human_judgment: false
  - id: D3
    description: "Public 1D 2-Wasserstein distance (0 for identical, matches location shift), and signed_quantile_average accepts negative weights."
    requirement: "FRE-01-06"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/space.rs::tests::w2_identical_is_zero, w2_matches_location_shift, w2_rejects_length_mismatch, signed_quantile_average_accepts_negative_weights, signed_quantile_average_uniform_weights_matches_barycenter"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-08-22
status: complete
---

# Phase 40 Plan 01: Fréchet Metric-Space Tracer Summary

**The `frechet` module now provides a generic `MetricSpace` abstraction with a 1D-Wasserstein density backend, the public `wasserstein2_distance`, and the sample `frechet_mean`/`frechet_variance` — the end-to-end tracer proving trait → density backend → DENS-01 reuse → generic statistics, plus the signed-weight quantile-average helper Wave 2 regression depends on.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 3/3
- **Tests:** 12 inline tests, all passing

## Accomplishments

- New `fdars-core/src/frechet/` module (`mod.rs`, `space.rs`, `mean.rs`), crate-root re-exported.
- `MetricSpace` trait + `WassersteinDensitySpace` backend; `wasserstein2_distance`; `frechet_mean`/`frechet_variance`.
- `signed_quantile_average` (pub(crate), sort-based isotonic projection, no new dependency) ready for Wave 2.
- Reused DENS-01's `wasserstein_barycenter` + back-map helpers (widened two to `pub(crate)` — additive/non-breaking).

## Verification

- `cargo test -p fdars-core --features linalg,parallel frechet` → 12 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → clean.

## Notes for Wave 2/3

- Wave 2 (`regression.rs`) calls `signed_quantile_average` for global (Petersen–Müller signed linear weights) + local (kernel local-linear) Fréchet regression + density-response variant — this removes the temporary `#[allow(dead_code)]` on that helper.
- Wave 3 (`anova.rs`) uses `frechet_mean`/`frechet_variance` + the in-crate `chi_square_sf` + seeded permutation p-value.
- Documented tolerance: barycenter reconstruction floor ~0.09 W₂ — keep this in mind for regression/ANOVA tolerances.
