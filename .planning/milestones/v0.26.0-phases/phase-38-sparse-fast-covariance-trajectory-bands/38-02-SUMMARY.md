---
phase: 38-sparse-fast-covariance-trajectory-bands
plan: 02
subsystem: irreg_fdata
tags: [mfaces, block-covariance, trajectory-bands, pace, cross-covariance]

requires:
  - phase: 38-01
    provides: face_covariance, face.rs module, gaussian_smooth_cov pub(crate)
  - phase: (shipped) pace_fpca
    provides: BLUP fitted trajectories + pointwise bands
provides:
  - mface_covariance(&[IrregFdata], &[grid], bandwidth) -> MfaceCovResult — multivariate block covariance
  - MfaceCovResult struct (block_cov, grids, offsets) + block(p,q) accessor
  - face_trajectory(data, config) -> PaceFpcaResult — fitted trajectories + pointwise bands (pace_fpca delegation)
  - crate-root re-export of all four FACE symbols
affects: [multivariate sparse FDA]

actuals:
  tokens: 38000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Multivariate block covariance: diagonal = face_covariance, off-diagonal = kernel-smoothed cross-variable covariance, symmetric by transpose"
    - "Semantic thin-wrapper entry point delegating to shipped pace_fpca for trajectory bands"

key-files:
  created: []
  modified:
    - fdars-core/src/irreg_fdata/face.rs
    - fdars-core/src/irreg_fdata/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "mface_covariance validates >=2 variables, matching n_obs across variables, per-variable grids, and bandwidth BEFORE any accumulation (RESEARCH Pitfall 6); G_total^2 checked_mul overflow guard"
  - "Off-diagonal (q,p) block set as the transpose of (p,q) by construction (not computed independently) — no floating-point asymmetry"
  - "cross_cov_surface: new private helper pairing each subject's variable-p points with the SAME subject's variable-q points (outer loop over subjects), mirroring accumulate_cov_at_point"
  - "face_trajectory is a pure pace_fpca delegation returning PaceFpcaResult (RESEARCH Finding 2 / Open Q3); delegation proven by == equality test"
  - "mface known-cross-structure tolerance 0.4 (RESEARCH A3); trajectory-band coverage threshold 0.85 (pointwise 95% BLUP bands undercover in finite samples from kernel-smoothing bias)"

patterns-established:
  - "MfaceCovResult block layout + block(p,q) accessor for multivariate sparse covariance"

requirements-completed: [SPARSE-01-02, SPARSE-01-03]

coverage:
  - id: D1
    description: "mface_covariance returns a symmetric (G_total x G_total) block covariance; diagonal = face_covariance, off-diagonal = cross-variable; block(p,q) accessor; block(1,0)==block(0,1)ᵀ"
    requirement: "SPARSE-01-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_mface_shape"
        status: pass
    human_judgment: false
  - id: D2
    description: "mface off-diagonal block recovers a known rank-1 cross-structure within documented tolerance (0.4)"
    requirement: "SPARSE-01-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_mface_known_structure"
        status: pass
    human_judgment: false
  - id: D3
    description: "mface invalid inputs (<2 vars, mismatched n_obs/grids, short grid, invalid bandwidth) return FdarError"
    requirement: "SPARSE-01-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_mface_errors"
        status: pass
    human_judgment: false
  - id: D4
    description: "face_trajectory returns fitted trajectories + pointwise bands via pace_fpca (delegation exact); dense-curve fitted tracks truth within its bands (>=0.85 coverage)"
    requirement: "SPARSE-01-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_face_trajectory_delegation, test_face_trajectory_bands"
        status: pass
    human_judgment: false
  - id: D5
    description: "All four FACE symbols (face_covariance, mface_covariance, MfaceCovResult, face_trajectory) reachable from the crate root"
    requirement: "SPARSE-01-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/irreg_fdata/face.rs#test_reexports"
        status: pass
    human_judgment: false

duration: 45min
completed: 2026-08-21
status: complete
---

# Phase 38 Plan 02: mface_covariance + face_trajectory Summary

**Completed SPARSE-01 with the multivariate FACE block covariance (`mface_covariance` + `MfaceCovResult`) and the sparse trajectory-band entry point (`face_trajectory`, a thin PACE delegation) — all four FACE symbols crate-root reachable.**

## Performance

- **Duration:** ~45 min
- **Tasks:** 3/3
- **Commits:** 1 (`e6c9a950`)

## Accomplishments

- `mface_covariance(&[IrregFdata], &[grid], bandwidth) -> MfaceCovResult`: assembles a symmetric `(G_total × G_total)` block covariance — diagonal blocks reuse `face_covariance`, off-diagonal blocks are a new kernel-smoothed cross-variable covariance (`cross_cov_surface`, pairing each subject's variable-p points with the same subject's variable-q points), set symmetrically by transpose. Validates ≥2 variables, matching `n_obs`, per-variable grids, and bandwidth before accumulation; guards `G_total²` overflow. Recovers a known rank-1 cross-structure within tolerance 0.4.
- `MfaceCovResult` (`#[non_exhaustive]`, standard derives + conditional serde): `block_cov`/`grids`/`offsets` + `block(p, q)` accessor extracting the `G_p × G_q` sub-block.
- `face_trajectory(data, config) -> PaceFpcaResult`: thin, semantically-named delegation to `pace_fpca` (fitted trajectories + pointwise Gaussian BLUP bands). Delegation proven by `==` equality; dense-curve fitted tracks truth within its bands (≥85% pointwise coverage).
- Crate-root re-exports extended to `{face_covariance, face_trajectory, mface_covariance, MfaceCovResult}`; `test_reexports` asserts reachability.

## Verification

- 6 new inline tests (mface ×3, trajectory ×2, reexport ×1); 9 face tests total green.
- Full gate: `cargo fmt --check` clean; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean; `cargo test -p fdars-core --features linalg,parallel` = 2413 lib + all integration + doctests green.
- Additive/non-breaking; no new crate dependency; existing public signatures (`cov_irreg`, `pace_fpca`) unchanged.

## Notes

- Executed inline per repo operational memory (worktree base divergence + executor cargo-build stalls); committed `--no-verify` after gates run out-of-band.
- Minor test-code fixes during execution: `StandardNormal.sample` needed an explicit `let z: f64` binding (turbofish on the wrong generic); `test_reexports` references symbols by call rather than fn-pointer casts with `-> _`.
