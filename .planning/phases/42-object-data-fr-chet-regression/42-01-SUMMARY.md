---
phase: 42-object-data-fr-chet-regression
plan: 01
subsystem: frechet
tags: [metric-space, spd, covariance, correlation, log-cholesky, power-metric, nalgebra, object-data]

requires:
  - phase: 39-40 (FRE-01, shipped)
    provides: MetricSpace trait, frechet regression/ANOVA solver framework
provides:
  - SpdMatrixSpace + SpdMetric{Frobenius,Power(f64),LogCholesky} (FRE-02-01)
  - CorrelationMatrixSpace (FRE-02-02)
  - frechet/spaces/ submodule + crate-root re-exports
affects: [phase-42-02, phase-42-03, frechet]

actuals:
  tokens: 20000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Object = Vec<f64> flat column-major matrix backends mirroring WassersteinDensitySpace"
    - "SPD power-α via nalgebra::SymmetricEigen (V diag(λ^α) Vᵀ), eigenvalues clamped ≥0"
    - "Log-Cholesky coords: strictly-lower Cholesky entries + log-diagonal, reused crate::linalg::cholesky_factor"

key-files:
  created:
    - fdars-core/src/frechet/spaces/mod.rs
    - fdars-core/src/frechet/spaces/spd.rs
    - fdars-core/src/frechet/spaces/correlation.rs
  modified:
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "SPD metrics selected via SpdMetric enum on SpdMatrixSpace::new; Power(α) validates α>0"
  - "Correlation mean = weighted average renormalized to unit diagonal, positive-diagonal guarded (documented divergence from R affine-invariant geometry)"
  - "frechet/spaces/mod.rs owns the re-export block that 42-02/42-03 append to"

patterns-established:
  - "Metric-space backend template: new()-validated struct + impl MetricSpace{distance,weighted_frechet_mean} with entry validation, no panics"

requirements-completed: [FRE-02-01, FRE-02-02]

coverage:
  - id: D1
    description: "SPD covariance-matrix space: Frobenius/power-α/log-Cholesky distances + weighted Fréchet means"
    requirement: FRE-02-01
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/spaces/spd.rs#spd_power_alpha_one_equals_frobenius, spd_log_cholesky_mean_identity_and_4i_is_2i, spd_frobenius_mean_of_identical_recovers_matrix (+ 6 more)"
        status: pass
  - id: D2
    description: "Correlation-matrix space: Frobenius distance + unit-diagonal-renormalized weighted mean"
    requirement: FRE-02-02
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/spaces/correlation.rs#correlation_mean_has_unit_diagonal, correlation_mean_of_identical_recovers, correlation_rejects_non_positive_diagonal (+ 2 more)"
        status: pass
---

# Plan 42-01 Summary: SPD + Correlation Matrix Backends

## Accomplishments

- New `frechet/spaces/` submodule with the object-data backend template.
- **`SpdMatrixSpace`** (FRE-02-01) with `SpdMetric { Frobenius, Power(f64), LogCholesky }`:
  - Frobenius: element-wise distance, weighted-average mean.
  - Power-α: `‖Aᵅ−Bᵅ‖_F/α` via `nalgebra::SymmetricEigen`; mean `(Σwᵢaᵢᵅ/Σwᵢ)^{1/α}`; α=1 reproduces Frobenius exactly.
  - Log-Cholesky: distance/mean in log-Cholesky coordinates (reusing `crate::linalg::cholesky_factor`); mean(I,4I)=2I.
- **`CorrelationMatrixSpace`** (FRE-02-02): Frobenius distance + weighted-average mean renormalized to unit diagonal, with a positive-diagonal guard.
- Both `impl MetricSpace` over `Vec<f64>`; re-exported from `frechet/mod.rs` and the crate root. **No existing signature changed, no new dependency.**

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib frechet::spaces` — 14/14 pass.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` — clean.

## Notes for 42-02 / 42-03

- `frechet/spaces/mod.rs` re-export block and `frechet/mod.rs` / `lib.rs` `pub use` are the shared blocks that 42-02 (spherical/network/point-process) and 42-03 (generic solvers) append to.
- The metric-space backend template (entry validation → distance/weighted_frechet_mean, never panic) is the pattern the remaining three backends mirror.
