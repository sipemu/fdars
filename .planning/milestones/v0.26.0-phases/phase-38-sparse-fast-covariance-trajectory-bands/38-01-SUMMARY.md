---
phase: 38-sparse-fast-covariance-trajectory-bands
plan: 01
subsystem: irreg_fdata
tags: [face, sparse-covariance, sandwich-smoother, psd, cov_irreg]

requires:
  - phase: (shipped) irreg_fdata::cov_irreg
    provides: kernel-smoothed sparse covariance
  - phase: 37
    provides: gaussian_smooth_cov separable sandwich smoother
provides:
  - New irreg_fdata/face.rs module (crate-root re-exported)
  - face_covariance(ifd, grid, bandwidth) -> Result<FdMatrix> — FACE kernel-sandwich covariance surface (symmetric + PSD)
  - gaussian_smooth_cov bumped to pub(crate) for intra-crate reuse
affects: [38-02, mface_covariance]

actuals:
  tokens: 34000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "FACE = cov_irreg raw covariance -> Gaussian sandwich -> symmetric_eigen PSD projection; reuses Phase 37 sandwich"

key-files:
  created:
    - fdars-core/src/irreg_fdata/face.rs
  modified:
    - fdars-core/src/irreg_fdata/mod.rs
    - fdars-core/src/fpca_variants.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "face_covariance validates inputs BEFORE cov_irreg (which panics via .expect() on bad dims); reuses cov_irreg + gaussian_smooth_cov (pub(crate)) + PSD projection via W^{1/2}CovW^{1/2} symmetric_eigen clipping negative eigenvalues"
  - "Documented divergence from refund::face: kernel-FACE (K-FACE) sandwich over cov_irreg rather than penalized tensor-product spline P-FACE — capability match, dependency-free"
  - "Dense-limit tolerance 0.30 (OU exp(-|s-t|) ridge is non-differentiable at s=t; kernel smoothing rounds it — bias-dominated, RESEARCH A2 ~0.3); trimmed test to m=31/n=200 for ~4s runtime"

patterns-established:
  - "Sparse-covariance FACE path in irreg_fdata/face.rs reusing cov_irreg + Phase 37 sandwich + symmetric_eigen PSD projection"

requirements-completed: [SPARSE-01-01]

coverage:
  - id: D1
    description: "face_covariance returns a symmetric PSD covariance surface for sparse/irregular data on a requested grid"
    requirement: "SPARSE-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_face_covariance_shape"
        status: pass
    human_judgment: false
  - id: D2
    description: "face_covariance recovers a known OU covariance surface on dense-limit synthetic data within documented tolerance (0.30)"
    requirement: "SPARSE-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_face_covariance_dense_limit"
        status: pass
    human_judgment: false
  - id: D3
    description: "Invalid inputs (empty sample, short/non-monotone grid, non-finite/non-positive bandwidth) return FdarError; face_covariance reachable at crate root"
    requirement: "SPARSE-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_face_covariance_errors"
        status: pass
    human_judgment: false

duration: 22min
completed: 2026-08-21
status: complete
---

# Phase 38 Plan 01: face_covariance FACE Sandwich Covariance Summary

**Landed the FACE fast-sandwich sparse-covariance estimator as the phase tracer — a new `irreg_fdata/face.rs` reusing `cov_irreg` + the Phase-37 sandwich smoother, producing a symmetric PSD covariance surface.**

## Performance

- **Duration:** ~22 min
- **Tasks:** 2/2
- **Commits:** 1 (`7f38c328`)

## Accomplishments

- New `fdars-core/src/irreg_fdata/face.rs` with module docs documenting the K-FACE vs `refund::face` P-FACE divergence; wired via `irreg_fdata/mod.rs` (`pub mod face;` + `pub use face::face_covariance;`) and crate-root `pub use irreg_fdata::face_covariance;`.
- `face_covariance(ifd, grid, bandwidth)`: validates inputs up front (empty sample, short/non-monotone grid, non-finite/non-positive bandwidth → `FdarError`, before `cov_irreg` which would panic), then `cov_irreg` raw → `gaussian_smooth_cov` sandwich → PSD projection (`W^{1/2}·Cov·W^{1/2}` symmetric_eigen, clip negatives, reconstruct). Symmetric + PSD by construction.
- `gaussian_smooth_cov` bumped from private to `pub(crate)` in `fpca_variants.rs` (intra-crate reuse, DRY; not a public-API change).
- Dense-limit test recovers the OU covariance `exp(-|s-t|)` from 200 dense curves within 0.30 (bias-dominated by the non-differentiable diagonal ridge).

## Verification

- 3 face tests green (shape/symmetry/PSD, dense-limit recovery, error paths).
- Full gate: `cargo fmt --check` clean; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean; `cargo test -p fdars-core --features linalg,parallel` = 2407 lib + all integration + doctests green.
- Additive/non-breaking; no new crate dependency; existing public signatures (`cov_irreg`, `pace_fpca`) unchanged.

## Notes

- Executed inline per repo operational memory (worktree base divergence + executor cargo-build stalls); committed `--no-verify` after gates run out-of-band.
