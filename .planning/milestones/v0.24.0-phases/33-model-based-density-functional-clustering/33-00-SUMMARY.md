---
phase: 33-model-based-density-functional-clustering
plan: "00"
subsystem: gmm
tags: [clustering, functional-data, subspace-model, em-algorithm, fpca, ari]
dependency_graph:
  requires: []
  provides:
    - funhddC_cluster (AkBk per-group subspace EM)
    - adjusted_rand_index (test helper)
  affects:
    - fdars-core crate root (new re-exports)
tech_stack:
  added: []
  patterns:
    - per-group SVD M-step via nalgebra (same pattern as regression::fdata_to_pc_1d)
    - reuse gmm/em.rs helpers (compute_bic, compute_icl, hard_assignments, resp_to_membership)
    - reuse gmm/init.rs k-means++ (kmeans_init_assignments)
    - reuse gmm/covariance.rs floor (data_scaled_reg)
key_files:
  created:
    - fdars-core/src/gmm/subspace.rs
  modified:
    - fdars-core/src/test_helpers.rs
    - fdars-core/src/gmm/mod.rs
    - fdars-core/src/lib.rs
decisions:
  - "AkBk single-model implementation (not the 6-model R funHDDC family): documented in rustdoc"
  - "module-level #![allow(non_snake_case)] in subspace.rs preserves public API name funhddC_cluster"
  - "nalgebra SVD for per-group thin SVD (no faer for subspace.rs: keeps MSRV 1.81)"
  - "weighted centered rows passed to per_group_svd (sqrt(resp_i) * (x_i - mu_k))"
  - "#[allow(non_snake_case)] on lib.rs gmm pub use block for funhddC_cluster re-export"
metrics:
  duration_minutes: 7
  completed_date: "2026-08-20"
  tasks_completed: 3
  tasks_total: 3
  commits: 3
status: complete
actuals:
  tokens: 19500
  tasks: 3
  commits: 3
---

# Phase 33 Plan 00: funHDDC Tracer + ARI Helper Summary

**One-liner:** funHDDC AkBk per-group subspace EM (FPCA init + nalgebra SVD M-step + BIC/ICL) plus `adjusted_rand_index` test helper in `gmm/subspace.rs`, wired to crate root.

## What Was Built

### Task 1 — adjusted_rand_index test helper

Added `pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64` to `fdars-core/src/test_helpers.rs` (the module is already `#[cfg(test)]`-gated at the lib.rs declaration). Implements the Hubert & Arabie (1985) contingency-table formula:
- Dense-relabels both input vectors to 0..k_a / 0..k_b
- Builds k_a × k_b contingency table, computes S = Σ C(n_ij,2)
- Handles degenerate cases (n=0, C(n,2)=0, denom≈0) by returning 1.0
- 4 unit tests: identical=1.0, permutation=1.0, 3-cluster permutation=1.0, unrelated~0

### Task 2 — funHDDC per-group subspace EM (tracer)

New `fdars-core/src/gmm/subspace.rs` (with `#![allow(non_snake_case)]` for the `funhddC_cluster` public API name):

**Public surface:**
- `FunHddcConfig` (`#[non_exhaustive]`, `Debug+Clone+PartialEq+serde(cfg_attr)`): k, d_k, max_iter, tol, n_init, seed, ncomp_init
- `FunHddcResult` (`#[non_exhaustive]`, `Debug+Clone`): cluster, membership, subspaces, within_vars, noise_vars, means, weights, log_likelihood, bic, icl, iterations, converged, k
- `funhddC_cluster(data, argvals, config) -> Result<FunHddcResult, FdarError>` (`#[must_use]`)

**Algorithm:** Multi-restart EM with k-means++ init on global FPCA scores. M-step uses per-group thin SVD of sqrt(resp)-weighted centered data rows to get leading d_k_eff eigenvectors. E-step applies the AkBk log-density formula (within-subspace Gaussian + isotropic noise on complement). BIC/ICL via reused `gmm/em.rs` helpers. Empty-cluster fallback to identity-like component.

**Module doc** explicitly records deliberate divergence from R `funHDDC` 6-model family.

**8 inline tests:**
- `test_funhddC_recovery`: ARI ≥ 0.90 on 2 vertically-separated sin-wave groups
- `test_funhddC_bic_finite`: BIC, ICL, log-likelihood all finite on recovery data
- `test_funhddC_deterministic`: same seed → identical cluster assignments
- `test_funhddC_invalid_empty`, `_k_zero`, `_k_exceeds_n`, `_dk_ge_m`, `_argvals_mismatch`: each returns `Err`, no panic

### Task 3 — Wire into gmm/mod.rs and lib.rs

- `gmm/mod.rs`: `pub mod subspace;` + `pub use subspace::{funhddC_cluster, FunHddcConfig, FunHddcResult};`
- `lib.rs`: extended GMM re-export block with the three new items + `#[allow(non_snake_case)]` attribute

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written.

### Intentional Implementation Notes

**1. nalgebra SVD (not faer) in subspace.rs:**
- The plan says "thin-SVD the centered member slice (nalgebra SVD, same conversion pattern as fdata_to_pc_1d)".
- `fdata_to_pc_1d` uses faer under the `linalg` feature; nalgebra is its non-`linalg` fallback.
- `gmm/subspace.rs` uses `nalgebra::SVD` directly to keep `MSRV 1.81` compliance (faer requires 1.84+) and avoid the `#[cfg(feature = "linalg")]` split that would complicate the module significantly.
- This matches the plan's parenthetical "same conversion pattern" and keeps `subspace.rs` feature-agnostic.

**2. pub(super) visibility of gmm/em.rs helpers:**
- `hard_assignments`, `resp_to_membership`, `compute_bic`, `compute_icl` are `pub(super)` in `gmm/em.rs`.
- As a sibling module under `gmm/`, `gmm/subspace.rs` can access `pub(super)` items via `use super::em::{...}`. No visibility widening was needed.

**3. #![allow(non_snake_case)] module-level:**
- The plan API specifies `funhddC_cluster` (camelCase C in middle) as the public function name.
- This does not follow Rust naming conventions but matches R's `funHDDC` origin.
- Applied via `#![allow(non_snake_case)]` in subspace.rs and `#[allow(non_snake_case)]` on the lib.rs re-export block — no existing code was changed.

## Known Stubs

None — all implementations are production-quality.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundary crossings.

## Self-Check: PASSED

Files verified:
- `fdars-core/src/gmm/subspace.rs` — FOUND
- `fdars-core/src/test_helpers.rs` — FOUND (adjusted_rand_index present)
- `fdars-core/src/gmm/mod.rs` — FOUND (pub mod subspace + pub use subspace)
- `fdars-core/src/lib.rs` — FOUND (funhddC_cluster in gmm pub use)

Commits verified:
- `bb097c31` — test(33-00): add adjusted_rand_index helper to test_helpers
- `0e92d4ab` — feat(33-00): add funHDDC per-group subspace EM in gmm/subspace.rs
- `6ebea1f3` — feat(33-00): wire funHDDC into gmm/mod.rs and crate-root lib.rs

Tests: all 12 tests pass (4 ARI + 8 funHDDC).
Build: `cargo build -p fdars-core --features linalg,parallel --lib` — clean (0 warnings).
