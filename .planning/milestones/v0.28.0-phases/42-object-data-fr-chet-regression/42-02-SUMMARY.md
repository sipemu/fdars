---
phase: 42-object-data-fr-chet-regression
plan: 02
subsystem: frechet
tags: [metric-space, spherical, karcher-mean, network, graph-laplacian, point-process, object-data]

requires:
  - phase: 42-object-data-fr-chet-regression (plan 42-01)
    provides: frechet/spaces/ module + MetricSpace backend template + re-export blocks
provides:
  - SphericalSpace (geodesic + Karcher mean, FRE-02-03)
  - NetworkSpace (graph-Laplacian Frobenius, FRE-02-04)
  - PointProcessSpace (intensity L2, FRE-02-05)
affects: [phase-42-03, frechet]

actuals:
  tokens: 17000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Intrinsic Karcher mean via Riemannian gradient descent (extrinsic init, exp/log maps, iter-capped)"
    - "Linear-combination MetricSpace backends (Frobenius/L2 distance + weighted-average mean)"

key-files:
  created:
    - fdars-core/src/frechet/spaces/spherical.rs
    - fdars-core/src/frechet/spaces/network.rs
    - fdars-core/src/frechet/spaces/point_process.rs
  modified:
    - fdars-core/src/frechet/spaces/mod.rs
    - fdars-core/src/frechet/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Spherical Karcher mean: extrinsic-mean init, max_iter=50, tol=1e-8; antipodal log / balanced init / non-convergence → ComputationFailed (never panic)"
  - "Network + point-process are linear-combination spaces (Frobenius / L2 distance, weighted-average mean); structure/non-negativity not re-validated per call (documented)"

patterns-established:
  - "Iterative-mean backend: guard division-by-sin(theta) at antipode, clamp dot to [-1,1] before acos, hard iter cap"

requirements-completed: [FRE-02-03, FRE-02-04, FRE-02-05]

coverage:
  - id: D1
    description: "Spherical space: geodesic distance + exp/log maps + iterative Karcher mean"
    requirement: FRE-02-03
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/spaces/spherical.rs#spherical_geodesic_antipodal_is_pi, spherical_karcher_midpoint, spherical_karcher_of_identical_recovers, spherical_karcher_antipodal_balanced_fails (+2)"
        status: pass
  - id: D2
    description: "Network space: graph-Laplacian Frobenius distance + Laplacian-preserving weighted mean"
    requirement: FRE-02-04
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/spaces/network.rs#network_mean_preserves_row_sums, network_mean_of_identical_recovers (+2)"
        status: pass
  - id: D3
    description: "Point-process space: intensity L2 distance + weighted-average mean"
    requirement: FRE-02-05
    verification:
      - kind: unit
        ref: "fdars-core/src/frechet/spaces/point_process.rs#point_process_distance_orthonormal_is_sqrt2, point_process_mean_of_identical_recovers (+2)"
        status: pass
---

# Plan 42-02 Summary: Spherical, Network & Point-Process Backends

## Accomplishments

- **`SphericalSpace`** (FRE-02-03): geodesic distance `arccos(clamp(⟨a,b⟩,−1,1))`, exp/log maps with an antipodal division-by-`sin θ` guard, and the intrinsic Karcher mean via Riemannian gradient descent (extrinsic-mean init, `max_iter=50`, `tol=1e-8`). Antipodally-balanced init, antipodal log, and non-convergence all return `ComputationFailed` — never a panic.
- **`NetworkSpace`** (FRE-02-04): graph-Laplacian Frobenius distance + weighted-average mean (stays a valid Laplacian on the convex cone; row-sums preserved).
- **`PointProcessSpace`** (FRE-02-05): intensity/count L2 distance + weighted-average mean.
- All three `impl MetricSpace` over `Vec<f64>`; append-only re-exports in `frechet/spaces/mod.rs`, `frechet/mod.rs`, and the crate root (42-01's additions untouched). **No existing signature changed, no new dependency.**

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib frechet::spaces` — 28/28 pass (14 new + 14 from 42-01).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` — clean.

## Notes for 42-03

- All five backends are now available; 42-03 wires the generic `frechet_*_reg_space` / `frechet_anova_space` solvers and validates them over the SPD Frobenius backend (which accepts the signed Petersen–Müller regression weights natively).
