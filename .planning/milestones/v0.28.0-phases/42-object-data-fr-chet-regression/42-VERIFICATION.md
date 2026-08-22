---
phase: 42-object-data-fr-chet-regression
verified: 2026-08-23T10:30:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0

requirements_verdict:
  FRE-02-01: SATISFIED
  FRE-02-02: SATISFIED
  FRE-02-03: SATISFIED
  FRE-02-04: SATISFIED
  FRE-02-05: SATISFIED
  FRE-02-06: SATISFIED
  FRE-02-07: SATISFIED
---

# Phase 42: Object-Data Fréchet Regression Verification Report

**Phase Goal:** Users can run global/local Fréchet regression and Fréchet-ANOVA over
non-density object responses by selecting a `MetricSpace` backend — SPD covariance matrices
(Frobenius / power / log-Cholesky), correlation matrices, spherical data, networks, or point
processes.

**Verified:** 2026-08-23T10:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | SPD covariance-matrix MetricSpace backend (Frobenius, power, log-Cholesky) delivers numeric distances + Fréchet means | ✓ VERIFIED | `SpdMatrixSpace` + `SpdMetric` enum in `frechet/spaces/spd.rs`; `impl MetricSpace` at line 223; `SymmetricEigen` (line 161) + `cholesky_factor` (line 182) wired; oracle tests pass: `spd_power_alpha_one_equals_frobenius`, `spd_log_cholesky_mean_identity_and_4i_is_2i` |
| 2 | Correlation, spherical, network, and point-process MetricSpace backends each deliver numeric distance + weighted-Fréchet-mean | ✓ VERIFIED | Four files in `frechet/spaces/`: `correlation.rs` (line 58), `spherical.rs` (line 111), `network.rs` (line 56), `point_process.rs` (line 55) all `impl MetricSpace`; oracle tests for antipodal-π, Karcher midpoint, row-sums-0, √2 all pass in the 58-test run |
| 3 | Global and local Fréchet regression over Euclidean predictors with SPD backend reusing FRE-01 solver → predicted object per xout row | ✓ VERIFIED | `frechet_global_reg_space` (line 278) + `frechet_local_reg_space` (line 363) in `regression.rs`; `compute_global_weights` (line 28) + `compute_local_weights` (line 105) extracted; constant-response oracle `spd_global_reg_constant_response_predicts_constant` passes; density gates `global_tracks_known_relationship` + `local_tracks_known_relationship` pass |
| 4 | Fréchet-ANOVA group-difference test over SPD backend reusing generic frechet_anova machinery → numeric statistic + seeded p-value | ✓ VERIFIED | `frechet_anova_space` (line 212) in `anova.rs`; `compute_tn_generic<S: MetricSpace>` (line 32); oracles pass: `spd_anova_homogeneous_not_significant` (p>0.05), `spd_anova_separated_significant` (p<0.05), `spd_anova_seed_reproducible` (bit-exact equality); density gate `anova_flags_shifted_groups` passes |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/frechet/spaces/spd.rs` | SpdMatrixSpace + SpdMetric impl MetricSpace | ✓ VERIFIED | Present, 400+ lines, SymmetricEigen + cholesky_factor wired |
| `fdars-core/src/frechet/spaces/correlation.rs` | CorrelationMatrixSpace impl MetricSpace | ✓ VERIFIED | Present, unit-diagonal renormalization implemented |
| `fdars-core/src/frechet/spaces/spherical.rs` | SphericalSpace (geodesic + Karcher) impl MetricSpace | ✓ VERIFIED | Present, exp/log maps + Karcher gradient descent, antipodal guard |
| `fdars-core/src/frechet/spaces/network.rs` | NetworkSpace impl MetricSpace | ✓ VERIFIED | Present, Frobenius distance + weighted-average mean |
| `fdars-core/src/frechet/spaces/point_process.rs` | PointProcessSpace impl MetricSpace | ✓ VERIFIED | Present, L2 distance + weighted-average mean |
| `fdars-core/src/frechet/regression.rs` | compute_global_weights, compute_local_weights, frechet_global_reg_space, frechet_local_reg_space | ✓ VERIFIED | All four present; pub(crate) helpers at lines 28 and 105 |
| `fdars-core/src/frechet/anova.rs` | compute_tn_generic, frechet_anova_space | ✓ VERIFIED | Both present at lines 32 and 212 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `frechet/spaces/mod.rs` | five space backends | `mod spd/correlation/spherical/network/point_process` + explicit `pub use` | ✓ WIRED | All five mods declared, all five structs re-exported explicitly |
| `frechet/mod.rs` | `spaces/` + regression + anova | `mod spaces;` at line 38; `pub use anova::{frechet_anova, frechet_anova_space}` at line 40; `pub use regression::{..., frechet_global_reg_space, frechet_local_reg_space}` at line 43 | ✓ WIRED | All three generic entry points + all five spaces in re-export block |
| `src/lib.rs` | `frechet::*` | `pub use frechet::{..., SpdMatrixSpace, SpdMetric, CorrelationMatrixSpace, SphericalSpace, NetworkSpace, PointProcessSpace, frechet_global_reg_space, frechet_local_reg_space, frechet_anova_space, ...}` at lines 153-157 | ✓ WIRED | Confirmed by grep: all eight new symbols present in the pub use block |
| `frechet_global_reg` / `frechet_local_reg` | `compute_global_weights` / `compute_local_weights` | Delegate call (no formula change) | ✓ WIRED | Pre-phase signatures byte-identical; density tests pass |
| `frechet_anova` | `compute_tn_generic` | Delegate call (`compute_tn` renamed to generic) | ✓ WIRED | Pre-phase signature byte-identical; density tests pass |

---

### Non-Breaking Additive Gate (Signature Preservation)

| Item | Pre-phase Signature | Post-phase Signature | Status |
|------|-------------------|---------------------|--------|
| `frechet_global_reg` | `(predictors: &FdMatrix, responses: &FdMatrix, argvals: &[f64], xout: &FdMatrix) -> Result<FrechetGlobalRegResult, FdarError>` | Identical | ✓ UNCHANGED |
| `frechet_local_reg` | `(predictors: &FdMatrix, responses: &FdMatrix, argvals: &[f64], xout: &FdMatrix, bandwidth: f64) -> Result<FrechetLocalRegResult, FdarError>` | Identical | ✓ UNCHANGED |
| `frechet_anova` | `(responses: &FdMatrix, argvals: &[f64], group_labels: &[usize], n_perm: usize, seed: u64) -> Result<FrechetAnovaResult, FdarError>` | Identical | ✓ UNCHANGED |
| `MetricSpace` trait | Lines 24-50 in `space.rs` | Identical | ✓ UNCHANGED |
| `WassersteinDensitySpace` | Lines 53+ in `space.rs` | Identical | ✓ UNCHANGED |
| `Cargo.toml` dependencies | No new dependencies | `git diff HEAD~4 HEAD -- fdars-core/Cargo.toml` empty | ✓ UNCHANGED |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 58 frechet lib tests (density + generic) | `cargo test -p fdars-core --features linalg,parallel --lib frechet` | 58 passed; 0 failed | ✓ PASS |
| Clippy across all targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Clean (0 warnings) | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| FRE-02-01 | 42-01 | SPD covariance-matrix space: Frobenius/power-α/log-Cholesky distance + weighted Fréchet mean | ✓ SATISFIED | `spd_power_alpha_one_equals_frobenius`, `spd_log_cholesky_mean_identity_and_4i_is_2i`, `spd_frobenius_mean_of_identical_recovers_matrix` pass |
| FRE-02-02 | 42-01 | Correlation-matrix space: Frobenius distance + unit-diagonal-renormalized weighted mean | ✓ SATISFIED | `correlation_mean_has_unit_diagonal`, `correlation_mean_of_identical_recovers`, `correlation_rejects_non_positive_diagonal` pass |
| FRE-02-03 | 42-02 | Spherical space: geodesic distance + exp/log maps + iterative Karcher mean | ✓ SATISFIED | `spherical_geodesic_antipodal_is_pi`, `spherical_karcher_midpoint`, `spherical_karcher_antipodal_balanced_fails` pass |
| FRE-02-04 | 42-02 | Network space: graph-Laplacian Frobenius distance + Laplacian-preserving weighted mean | ✓ SATISFIED | `network_mean_preserves_row_sums`, `network_distance_of_identical_is_zero` pass |
| FRE-02-05 | 42-02 | Point-process space: intensity L2 distance + weighted-average mean | ✓ SATISFIED | `point_process_distance_orthonormal_is_sqrt2`, `point_process_mean_of_identical_recovers` pass |
| FRE-02-06 | 42-03 | Generic global + local Fréchet regression over Euclidean predictors with non-density (SPD) backend | ✓ SATISFIED | `spd_global_reg_constant_response_predicts_constant`, `spd_local_reg_returns_object_per_xout`; density gates pass |
| FRE-02-07 | 42-03 | Generic Fréchet-ANOVA over non-density (SPD) object space: significant/non-significant + seed-reproducible | ✓ SATISFIED | `spd_anova_homogeneous_not_significant` (p>0.05), `spd_anova_separated_significant` (p<0.05), `spd_anova_seed_reproducible` (bit-exact); density gate passes |

---

### Anti-Patterns Found

No TBD, FIXME, or XXX markers in any file modified by this phase.
No `unimplemented!` or `todo!` calls in any of the five space backends.
No stub patterns (all `impl MetricSpace` blocks contain substantive computation).

---

### Human Verification Required

None. All success criteria are mechanically verifiable and all tests pass.

---

## Summary

Phase 42 achieves its goal in full. All five `MetricSpace` backends exist with substantive
implementations and are wired into `frechet/mod.rs` and the crate root. The three generic
entry points (`frechet_global_reg_space`, `frechet_local_reg_space`, `frechet_anova_space`)
delegate to extracted `pub(crate)` helpers that share the Petersen–Müller and Tₙ machinery
with the density path. The refactor is provably non-breaking: pre/post signatures are
byte-identical, no new Cargo dependency was added, and the full 58-test frechet suite — which
includes the density regression gate (`global_tracks_known_relationship`,
`local_tracks_known_relationship`, `anova_flags_shifted_groups`) alongside every new oracle
test — passes clean. Clippy is clean across all targets.

---

_Verified: 2026-08-23T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
