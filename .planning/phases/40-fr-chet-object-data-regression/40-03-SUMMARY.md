---
phase: 40-fr-chet-object-data-regression
plan: 03
subsystem: frechet
tags: [frechet-anova, dubey-muller, permutation-test, object-data]

requires:
  - phase: 40-01
    provides: frechet_mean, frechet_variance, WassersteinDensitySpace
  - phase: (inference)
    provides: inference::dist::chi_square_sf (in-crate chi-square survival function)
provides:
  - "frechet_anova: Dubey-Müller Tn statistic + seeded-permutation p-value (primary) + asymptotic chi2(k-1) p-value (secondary)"
  - "FrechetAnovaResult"

actuals:
  tokens: 12000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Seeded permutation test (per-iteration StdRng::seed_from_u64(seed+k), 999 default) over metric-space Fréchet means/variances"

key-files:
  created:
    - fdars-core/src/frechet/anova.rs
  modified:
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/inference/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Reused frechet_mean/frechet_variance for per-group + pooled Fréchet variances; Tn = n·Un/Σ(λ/σ²) + n·Fn²/Σ(λ²σ²) (Dubey-Müller)."
  - "Asymptotic p-value uses the in-crate inference::dist::chi_square_sf (widened inference's `mod dist` to `pub(crate)` — additive/non-breaking); NO statrs, NO osqp."
  - "σ̂ₗ² estimator = (1/nₗ)Σ[d²(Yᵢ,μ̂ₗ)−V̂ₗ]² carries a rustdoc [ASSUMED] note (Dubey & Müller 2019, Biometrika 106(4); R frechet::DenANOVA provenance); the seeded permutation p-value is the primary, assumption-robust inference."
  - "group_labels must be contiguous 0..k; <2 distinct groups → InvalidParameter."

patterns-established:
  - "Metric-space Fréchet ANOVA via seeded label-permutation, thread-count-independent (per-iteration seeding)."

requirements-completed: [FRE-01-08]

coverage:
  - id: D1
    description: "Fréchet ANOVA flags a genuine between-group shift (perm p<0.05) and does not flag a homogeneous sample (perm p>0.05); seed-reproducible."
    requirement: "FRE-01-08"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/anova.rs::tests::anova_flags_shifted_groups, anova_ignores_homogeneous_sample, anova_permutation_is_seed_reproducible"
        status: pass
    human_judgment: false
  - id: D2
    description: "Invalid inputs (<2 groups, label/response mismatch) return FdarError."
    requirement: "FRE-01-08"
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/anova.rs::tests::anova_rejects_too_few_groups, anova_rejects_label_mismatch"
        status: pass
    human_judgment: false

duration: 25min
completed: 2026-08-22
status: complete
---

# Phase 40 Plan 03: Fréchet ANOVA Summary

**`frechet_anova` completes FRE-01: a Dubey–Müller `Tₙ` group-difference test over the Wasserstein density space with a primary seeded-permutation p-value and a secondary asymptotic χ²(k−1) p-value, reusing the Wave-1 Fréchet mean/variance machinery and the in-crate chi-square survival function — no new dependency.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 2/2
- **Tests:** 5 new inline tests (23 total in the frechet module), all passing

## Accomplishments

- New `frechet/anova.rs` with `frechet_anova` + `FrechetAnovaResult`, crate-root re-exported.
- Widened `inference::dist` to `pub(crate)` (additive) to reuse `chi_square_sf`; no `statrs`, no `osqp`.
- Seeded permutation p-value (per-iteration `StdRng::seed_from_u64(seed+k)`, 999 default), reproducible; σ̂ₗ² flagged `[ASSUMED]` in rustdoc.

## Verification

- `cargo test -p fdars-core --features linalg,parallel frechet` → 23 passed.
- Whole crate: 2460 lib tests + integration + 172 doctests green; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean; fmt clean.
- No existing public signature changed; no new crate dependency.

## Phase Close

All 8 FRE-01 requirements (FRE-01-01..08) implemented across Plans 01–03. The `frechet/` module is complete: metric-space abstraction, Wasserstein density backend, Fréchet mean/variance, W₂ distance, global/local Fréchet regression (density-response), and Fréchet ANOVA.
