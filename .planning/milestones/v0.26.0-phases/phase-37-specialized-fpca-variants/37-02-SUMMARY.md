---
phase: 37-specialized-fpca-variants
plan: 02
subsystem: fpca
tags: [fpca, fsvd, dynamical-correlation, sandwich-smoother, svd, nalgebra]

requires:
  - phase: 37-01
    provides: fpca_variants module, FsvdResult struct, cross_covariance
provides:
  - dynamical_correlation(x, y, argvals) — Dubin-Muller scalar functional correlation in [-1,1]
  - fsvd(x, argvals_x, y, argvals_y, ncomp) — functional SVD / cross-FPCA populating FsvdResult
  - ssvd(data, ncomp, argvals, bandwidth) — sandwich-smoother FPCA path returning FpcaResult
  - crate-root re-export of all five variants + FsvdResult
affects: [fpca, cross-covariance analysis]

actuals:
  tokens: 39000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Robust SVD of rank-deficient cross-covariance via symmetric eigendecomposition of the smaller Gram matrix (nalgebra general SVD fails to converge on near-rank-1 inputs)"
    - "Sandwich-smoother FPCA reusing the pace_fpca W^{1/2}CovW^{1/2} eigendecompose pattern (inlined)"

key-files:
  created: []
  modified:
    - fdars-core/src/fpca_variants.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "dynamical_correlation implements the exact 4-step Dubin-Muller / fdapace DynCorr, dividing by domain_length in BOTH the L2 norm and the inner product; shared argvals grid required; NaN-guarded L2 normalization"
  - "fsvd computes the SVD via symmetric eigendecomposition of the smaller Gram matrix (Cw^T Cw or Cw Cw^T) rather than nalgebra's general SVD — the latter's recompose() failed to reproduce a near-rank-1 weighted cross-covariance (err 0.10). Feature-agnostic (nalgebra DMatrix, no faer/linalg dependency), robust, and consistent with the pace_fpca eigendecompose approach"
  - "ssvd treats bandwidth <= 1e-10 as an identity smoother (empirical covariance passed straight through) because gaussian_kernel returns 0 at zero bandwidth; singular_value = sqrt(eigenvalue*(n-1)) to match fdata_to_pc_1d's SVD convention in the dense limit"

patterns-established:
  - "Rank-deficient weighted-SVD via Gram-matrix symmetric_eigen — reusable where a cross-covariance SVD is needed"

requirements-completed: [FPCA-02-02, FPCA-02-04, FPCA-02-05]

coverage:
  - id: D1
    description: "dynamical_correlation returns scalar in [-1,1]; =1 for x==x, =-1 for x==-x, in-range for random data; requires shared grid; rejects mismatched dims"
    requirement: "FPCA-02-04"
    verification:
      - kind: unit
        ref: "fdars-core/src/fpca_variants.rs#test_dyncorr_identical, test_dyncorr_negated, test_dyncorr_range, test_dyncorr_errors"
        status: pass
    human_judgment: false
  - id: D2
    description: "fsvd returns unit-L2 paired singular functions + singular values + per-sample scores; recovers a known rank-1 cross-covariance; deterministic paired sign convention; rejects invalid inputs"
    requirement: "FPCA-02-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/fpca_variants.rs#test_fsvd_unit_norm, test_fsvd_rank1, test_fsvd_errors"
        status: pass
    human_judgment: false
  - id: D3
    description: "ssvd sandwich-smoothed FPCA; W-orthonormal eigenfunctions; dense/near-zero-bandwidth limit matches fdata_to_pc_1d within relative 1e-4; rejects invalid inputs"
    requirement: "FPCA-02-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/fpca_variants.rs#test_ssvd_dense_limit, test_ssvd_orthonormality, test_ssvd_errors"
        status: pass
    human_judgment: false
  - id: D4
    description: "All five variants (cross_covariance, fpca_der, dynamical_correlation, fsvd, ssvd) + FsvdResult reachable from the crate root"
    requirement: "FPCA-02-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/fpca_variants.rs#smoke_reexports"
        status: pass
    human_judgment: false

duration: 50min
completed: 2026-08-21
status: complete
---

# Phase 37 Plan 02: dynamical_correlation + fsvd + ssvd Summary

**Completed FPCA-02 with the three normalization-sensitive variants — the Dubin–Müller dynamical correlation, the functional cross-SVD, and the sandwich-smoother FPCA path — all crate-root reachable.**

## Performance

- **Duration:** ~50 min (incl. debugging the nalgebra SVD convergence issue)
- **Tasks:** 3/3
- **Commits:** 1 (`43108e57`)

## Accomplishments

- `dynamical_correlation`: exact 4-step Dubin–Müller / `fdapace::DynCorr` (per-curve integrated-mean centering → population centering → L2-norm standardization → integrated inner product / domain_length averaged over subjects). Scalar in [−1,1]; `=1` for `x==x`, `=−1` for `x==−x`, in-range for random data. NaN-guarded normalization; shared-grid requirement enforced.
- `fsvd`: Simpson-weighted empirical cross-covariance decomposed and returned as unit-L2 paired singular functions, singular values, and per-sample scores in `FsvdResult`, with a deterministic paired sign convention. Recovers a known rank-1 cross-covariance structure.
- `ssvd`: separable-Gaussian sandwich-smoothed covariance + inlined `pace_fpca` `W^{1/2}·Cov·W^{1/2}` eigendecompose, returning an `FpcaResult`. `bandwidth <= 1e-10` is the no-smoothing dense limit that matches `fdata_to_pc_1d` within relative 1e-4; W-orthonormal eigenfunctions.
- Crate-root re-export extended to all five variants + `FsvdResult`; `smoke_reexports` asserts reachability.

## Deviations & Discoveries

- **nalgebra general SVD is unreliable for near-rank-deficient matrices.** The planned `SVD::new(Cw.to_dmatrix(), ...)` produced a decomposition whose own `recompose()` failed to reproduce `Cw` (max err ~0.10) on the rank-1 test cross-covariance, so the singular functions were wrong. Replaced with a **symmetric eigendecomposition of the smaller Gram matrix** (`Cwᵀ·Cw` or `Cw·Cwᵀ`), recovering σ = √λ, one singular vector from the eigenvector and the other via `Cw·v/σ`. This is feature-agnostic (no faer/`linalg` dependency), robust, and consistent with the `pace_fpca` eigendecompose approach. Plan intent (weighted cross-cov → unit-L2 functions → paired sign fix → scores) is unchanged; only the numerical SVD engine differs.

## Verification

- 10 new inline tests (dyncorr ×4, fsvd ×3, ssvd ×3) + extended `smoke_reexports`; 18 fpca_variants tests total green.
- Full gate: `cargo fmt --check` clean; `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean; `cargo test -p fdars-core --features linalg,parallel` = 2403 lib + all integration + 5 doctests green.
- Additive/non-breaking; no new crate dependency.

## Notes

- Executed inline per repo operational memory (worktree base divergence + executor cargo-build stalls); committed `--no-verify` after running gates out-of-band.
