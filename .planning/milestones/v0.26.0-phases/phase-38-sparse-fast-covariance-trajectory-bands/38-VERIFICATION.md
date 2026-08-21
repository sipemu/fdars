---
phase: 38-sparse-fast-covariance-trajectory-bands
verified: 2026-08-21T20:32:47Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 38: Sparse Fast Covariance & Trajectory Bands — Verification Report

**Phase Goal:** A user can estimate sparse/irregular functional covariance with FACE/mfaces and trajectory bands — `face_covariance`, `mface_covariance`, and `face_trajectory` — all in `fdars-core/src/irreg_fdata/`, building on `cov_irreg` and `pace_fpca`, without any existing code changing and with no new crate dependency.
**Verified:** 2026-08-21T20:32:47Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All three entry points exist, are `Result`-returning, consume sparse/irregular data, and are crate-root re-exported (`face_covariance`, `mface_covariance`, `MfaceCovResult`, `face_trajectory`) | VERIFIED | `lib.rs:232` `pub use irreg_fdata::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult};`; `irreg_fdata/mod.rs:31` full re-export; `test_reexports` passes (compiles + calls all four through crate root) |
| 2 | `face_covariance` returns a symmetric PSD covariance surface that recovers a known covariance on dense-limit synthetic data within tolerance, reusing `cov_irreg` + `gaussian_smooth_cov` (not a new subsystem) | VERIFIED | `face.rs:128–151` calls `cov_irreg` → `gaussian_smooth_cov` → `psd_project`; `test_face_covariance_shape` (symmetry + PSD), `test_face_covariance_dense_limit` (OU recovery < 0.30 tolerance) both PASS |
| 3 | `mface_covariance` returns a `(G_total×G_total)` block covariance (diagonal = per-variable `face_covariance`, off-diagonal = cross-variable), recovers a known cross-structure within tolerance, with `MfaceCovResult` + `block()` accessor | VERIFIED | `face.rs:263–388` implements block assembly; `test_mface_shape` (block shape, symmetry, diagonal == standalone `face_covariance`, `block(1,0) == block(0,1)ᵀ`), `test_mface_known_structure` (rank-1 cross-structure < 0.40 tolerance), `test_mface_three_vars` (P=3, mixed grid sizes) all PASS |
| 4 | `face_trajectory` returns fitted trajectories + pointwise bands via `pace_fpca`; fitted tracks the true curve within its bands on dense data; invalid inputs return `FdarError`; NO new crate dependency | VERIFIED | `face.rs:407–411` is a one-line `pace_fpca(data, config)` delegation; `test_face_trajectory_delegation` (result == `pace_fpca` exactly), `test_face_trajectory_bands` (>= 85% pointwise coverage) PASS; `test_face_covariance_errors` + `test_mface_errors` validate all error paths; `fdars-core/Cargo.toml` unchanged (diff shows zero lines added) |
| 5 | Existing public signatures unchanged (additive/non-breaking); only change to an existing file is `gaussian_smooth_cov` bumped from private `fn` to `pub(crate) fn` in `fpca_variants.rs` (not a public-API change); full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` green | VERIFIED | Git diff shows zero removed public signatures; `fpca_variants.rs` diff is `fn` → `pub(crate) fn` + doc comment only; clippy exits 0; full suite: 2414 lib + all integration + doctests green (0 failures) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/irreg_fdata/face.rs` | New file — all three functions + `MfaceCovResult` + inline tests | VERIFIED | 797 lines; module doc; all 9 inline tests; `#[must_use]` on all three public fns |
| `pub mod face;` + `pub use face::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult}` in `irreg_fdata/mod.rs` | Module wiring + full re-export | VERIFIED | `mod.rs:23` + `mod.rs:31` match exactly |
| `pub use irreg_fdata::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult}` in `lib.rs` | Crate-root re-export | VERIFIED | `lib.rs:232` matches |
| `gaussian_smooth_cov` → `pub(crate) fn gaussian_smooth_cov` in `fpca_variants.rs` | Visibility bump (intra-crate reuse, not a public-API change) | VERIFIED | `fpca_variants.rs:592` matches |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `face_covariance` | `cov_irreg` | Direct call `face.rs:148` | VERIFIED | `let raw_cov = cov_irreg(ifd, grid, grid, bandwidth);` |
| `face_covariance` | `gaussian_smooth_cov` | Direct call `face.rs:149` | VERIFIED | `let smooth_cov = gaussian_smooth_cov(&raw_cov, grid, bandwidth);` |
| `face_covariance` | `psd_project` | Direct call `face.rs:150` | VERIFIED | `psd_project(&smooth_cov, grid)` |
| `mface_covariance` | `face_covariance` | Diagonal block loop `face.rs:338` | VERIFIED | `let diag = face_covariance(&variables[p], &grids[p], bandwidth)?;` |
| `mface_covariance` | `cross_cov_surface` | Off-diagonal loop `face.rs:351` | VERIFIED | Private helper paired with transpose write |
| `face_trajectory` | `pace_fpca` | Direct delegation `face.rs:411` | VERIFIED | One-line body: `pace_fpca(data, config)` |
| `irreg_fdata/mod.rs` | `lib.rs` | Re-export chain | VERIFIED | Both lines present; `test_reexports` compiles and calls all four symbols |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `face_covariance` | `raw_cov` | `cov_irreg(ifd, ...)` — accumulates kernel-weighted cross-products from actual observation data | Yes — `ifd` is caller-supplied sparse data | FLOWING |
| `face_trajectory` | `PaceFpcaResult` | `pace_fpca(data, config)` — BLUP fitted + bands from `data` | Yes — full delegation; no static returns | FLOWING |
| `mface_covariance` | `block_data` | Diagonal from `face_covariance`, off-diagonal from `cross_cov_surface` (kernel accumulation over subjects) | Yes — real accumulation loop | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 10 named Phase 38 tests pass | `cargo test ... -- test_face_covariance_shape test_face_covariance_dense_limit test_face_covariance_errors test_mface_shape test_mface_known_structure test_mface_errors test_mface_three_vars test_face_trajectory_delegation test_face_trajectory_bands test_reexports` | 10 passed; 0 failed | PASS |
| Full suite green (no regressions) | `cargo test -p fdars-core --features linalg,parallel` | 2414 lib tests passed; 0 failed; integration + doctests all green | PASS |
| Clippy clean on all targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Exit 0; no warnings | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| SPARSE-01-01 | 38-01-PLAN.md | `face_covariance` fast-sandwich covariance surface | SATISFIED | `face_covariance` exists, validates inputs, calls `cov_irreg` + `gaussian_smooth_cov`, PSD projects; shape, dense-limit, and error tests pass |
| SPARSE-01-02 | 38-02-PLAN.md | `mface_covariance` + `MfaceCovResult` block covariance | SATISFIED | `mface_covariance` assembles symmetric block matrix; `MfaceCovResult` with `block()` accessor; shape, known-structure, error, and 3-var tests pass |
| SPARSE-01-03 | 38-02-PLAN.md | `face_trajectory` fitted trajectories + pointwise bands | SATISFIED | Thin delegation to `pace_fpca`; delegation equality test + bands coverage test (>= 85%) pass |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No `TBD`, `FIXME`, `XXX`, placeholder returns, or stub patterns found in any modified file | — | — |

No debt markers or stub patterns found in `face.rs`, `fpca_variants.rs`, `irreg_fdata/mod.rs`, or `lib.rs`.

### Human Verification Required

(none — all must-haves verified programmatically)

---

## Gaps Summary

No gaps. All five ROADMAP success criteria are satisfied by the codebase evidence:

1. All four FACE symbols (`face_covariance`, `mface_covariance`, `MfaceCovResult`, `face_trajectory`) are `Result`-returning, wired from `irreg_fdata/face.rs` through `irreg_fdata/mod.rs` to `lib.rs`, and verified to resolve at the crate root by `test_reexports`.
2. `face_covariance` reuses `cov_irreg` + `gaussian_smooth_cov` (no new subsystem); recovers OU kernel within 0.30 tolerance on 200 dense curves.
3. `mface_covariance` assembles a symmetric `(G_total×G_total)` block matrix; diagonal blocks equal standalone `face_covariance`; off-diagonal recovers rank-1 cross-structure within 0.40 tolerance; `MfaceCovResult.block(p,q)` accessor verified.
4. `face_trajectory` is a proven one-line delegation to `pace_fpca`; no new crate dependency (Cargo.toml unchanged); all invalid-input paths return `FdarError`.
5. Existing public signatures are untouched (diff confirms zero removed public items); the only existing-file change is the `fn` → `pub(crate) fn` visibility bump on `gaussian_smooth_cov` (intra-crate, not public API); 2414-test suite + clippy clean.

---

_Verified: 2026-08-21T20:32:47Z_
_Verifier: Claude (gsd-verifier)_
