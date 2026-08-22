---
phase: 42-object-data-fr-chet-regression
plan: 03
subsystem: frechet
tags: [frechet-regression, frechet-anova, metric-space, generic, petersen-muller, dubey-muller, object-data]

requires:
  - phase: 42-object-data-fr-chet-regression (plans 42-01, 42-02)
    provides: five MetricSpace backends (SPD Frobenius used as the demo/validation backend)
  - phase: 39-40 (FRE-01, shipped)
    provides: density frechet_global_reg/frechet_local_reg/frechet_anova + MetricSpace trait
provides:
  - frechet_global_reg_space / frechet_local_reg_space (generic regression, FRE-02-06)
  - frechet_anova_space (generic ANOVA, FRE-02-07)
  - pub(crate) compute_global_weights / compute_local_weights / compute_tn_generic
affects: [frechet]

actuals:
  tokens: 21000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Internal helper extraction: pub(crate) weight/Tn helpers shared by density + generic paths (density fns delegate, byte-identical)"
    - "Generic entry points over S: MetricSpace with S::Object: Clone, returning Vec<S::Object> / FrechetAnovaResult"

key-files:
  created: []
  modified:
    - fdars-core/src/frechet/regression.rs
    - fdars-core/src/frechet/anova.rs
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Extracted compute_global_weights/compute_local_weights (no formula change); density frechet_global_reg/frechet_local_reg delegate → byte-identical output, existing tests are the regression gate"
  - "compute_tn renamed to pub(crate) compute_tn_generic<S: MetricSpace> (body unchanged — already called generic frechet_mean/variance); frechet_anova delegates"
  - "S::Object: Clone bound (all backends use Vec<f64>); signed Petersen–Müller weights pass through natively for linear-combination spaces; spherical signed-weight caveat documented"

patterns-established:
  - "Non-breaking generification: keep the concrete public fn, extract a shared pub(crate) core, add a generic sibling"

requirements-completed: [FRE-02-06, FRE-02-07]

coverage:
  - id: D1
    description: "Generic global + local Fréchet regression over Euclidean predictors with a non-density (SPD) backend"
    requirement: FRE-02-06
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/regression.rs#spd_global_reg_constant_response_predicts_constant, spd_local_reg_returns_object_per_xout, spd_global_reg_rejects_response_count_mismatch, spd_local_reg_rejects_nonpositive_bandwidth"
        status: pass
  - id: D2
    description: "Generic Fréchet-ANOVA over a non-density (SPD) object space: significant/non-significant + seed-reproducible"
    requirement: FRE-02-07
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/anova.rs#spd_anova_separated_significant, spd_anova_homogeneous_not_significant, spd_anova_seed_reproducible"
        status: pass
  - id: D3
    description: "Existing density Fréchet regression + ANOVA tests still pass after the pub(crate) helper extraction (non-breaking regression gate)"
    requirement: FRE-02-06
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/{regression,anova}.rs#global_tracks_known_relationship, local_tracks_known_relationship, anova_flags_shifted_groups, anova_ignores_homogeneous_sample (+ others)"
        status: pass
---

# Plan 42-03 Summary: Generic Fréchet Regression + ANOVA over MetricSpace

## Accomplishments

- Extracted the Petersen–Müller global-linear weights and the local-linear kernel weights into `pub(crate) compute_global_weights` / `compute_local_weights` (no formula change). `frechet_global_reg` / `frechet_local_reg` now delegate — **byte-identical output**, their existing density tests are the regression gate.
- Renamed `compute_tn` → `pub(crate) compute_tn_generic<S: MetricSpace>` (body unchanged; it already called the generic `frechet_mean`/`frechet_variance`). `frechet_anova` delegates.
- Added the generic entry points (FRE-02-06/07):
  - `frechet_global_reg_space<S: MetricSpace>` / `frechet_local_reg_space<S: MetricSpace>` → `Vec<S::Object>` (one predicted object per `xout` row).
  - `frechet_anova_space<S: MetricSpace>` → `FrechetAnovaResult`, reusing the seeded-permutation Tₙ machinery.
- Validated over the SPD Frobenius backend: constant-response ⇒ exact constant prediction; separated 2-group sample significant (p<0.05); homogeneous sample non-significant (p>0.05); seed-reproducible.
- **No existing public signature changed, no new dependency.** All new symbols re-exported at the crate root.

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib frechet` — 58/58 pass (density regression gate + generic tests).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` — clean.

## Notes

- `S::Object: Clone` is required on the generic ANOVA path (all five backends use `Vec<f64>`, which is Clone).
- Signed Petersen–Müller weights are correct for linear-combination spaces; `SphericalSpace` (Karcher mean) should keep `xout` in-range or clip negatives — documented in the generic-regression rustdoc.
