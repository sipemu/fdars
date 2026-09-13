---
phase: 102-deterministic-multi-restart-wrapper
verified: 2026-09-13T10:30:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/102-deterministic-multi-restart-wrapper/102-01-PLAN.md
  - .planning/phases/102-deterministic-multi-restart-wrapper/102-01-SUMMARY.md
  - fdars-core/src/metric/soft_dtw.rs
covered_digest: "v1:sha256:80e27281de6f39bb608a5d472823df57d53dabf960957e84a0ae597a30fc5801"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 102: Deterministic Multi-Restart Wrapper Verification Report

**Phase Goal:** A multi-restart wrapper runs the L-BFGS local optimizer from multiple initializations (pointwise mean + deterministically-seeded perturbed inits) and returns the barycenter with the lowest total soft-DTW cost across restarts, reproducibly for a fixed seed.
**Verified:** 2026-09-13T10:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A private multi-restart driver runs the L-BFGS refiner from restart 0 (unperturbed pointwise mean) plus n_restarts-1 seeded Gaussian-perturbed inits and returns the barycenter with the lowest total soft-DTW cost (SDTW-02) | VERIFIED | `soft_dtw_barycenter_multirestart` (lines 740-808): restart 0 uses `mean.clone()`, restarts k>=1 use `perturbed_init(&mean, &col_std, seed, k as u64)`, winner selected by strictly-less total cost |
| 2 | Winner selection is total-cost argmin with ties broken by lowest restart index | VERIFIED | Line 791: `if total < best_total` (strictly less — ties keep earlier restart); confirmed by `lowest_cost_is_argmin` and `winner_le_each_restart` tests — 4/4 pass |
| 3 | Same seed + same inputs + same params produce a bit-identical barycenter (verified via f64::to_bits) — deterministic | VERIFIED | `barycenter_multirestart_tests::deterministic` test asserts `.to_bits()` equality element-by-element — passes |
| 4 | Multi-restart is never worse than single-init: restart 0 is the unperturbed mean, so the returned total cost <= the single-init (4-arg) result's total cost | VERIFIED | `barycenter_multirestart_tests::never_worse_than_single_init` compares multi-restart cost vs `soft_dtw_barycenter` cost with 1e-9 slack — passes; invariant holds because restart 0 of the driver runs the same `lbfgs_refine_from_init(&rows, mean, ...)` as the 4-arg entry |
| 5 | The public 4-arg soft_dtw_barycenter(data, gamma, max_iter, tol) signature and SoftDtwBarycenterResult struct shape are unchanged; the crate still compiles on wasm32-unknown-unknown and default features with no new dependency | VERIFIED | `soft_dtw_barycenter` at lines 833-871 retains the identical 4-arg pub signature and delegates to `lbfgs_refine_from_init`; `SoftDtwBarycenterResult` struct at lines 22-29 is `#[non_exhaustive]` with fields `barycenter`, `n_iter`, `converged` — unchanged; `rand`/`rand_distr` were pre-existing deps (no new entry in Cargo.toml — diff of Cargo.toml from milestone-start commit 6c125b90 shows zero changes); seeding uses `StdRng::seed_from_u64` which is wasm32-safe (no OS entropy) |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/metric/soft_dtw.rs` — `lbfgs_refine_from_init` | Private single-init L-BFGS descent extracted from soft_dtw_barycenter | VERIFIED | Lines 575-682: full descent body with all Phase 101 safeguards (curvature guard `sᵀy > 1e-20`, non-descent fallback, Armijo minimum step) |
| `fdars-core/src/metric/soft_dtw.rs` — `soft_dtw_barycenter_multirestart` | Private deterministic multi-restart driver | VERIFIED | Lines 740-808: `#[allow(dead_code)]` private fn with n==0/m==0 guard, non-finite gamma guard (returns mean, converged:false), restarts loop, strictly-less tiebreak argmin, all-NaN fallback to mean |
| `fdars-core/src/metric/soft_dtw.rs` — `per_col_std` / `perturbed_init` | Per-column std helper and seeded perturbation helper | VERIFIED | Lines 690-707 (`per_col_std`, `#[allow(dead_code)]`): population std, n<=1 guard returns zeros. Lines 716-723 (`perturbed_init`, `#[allow(dead_code)]`): `StdRng::seed_from_u64(seed.wrapping_add(k))` + `Normal::new(0.0, 1.0)`, single left-to-right pass |
| `fdars-core/src/metric/soft_dtw.rs` — `barycenter_multirestart_tests` module | SDTW-02 unit tests | VERIFIED | Lines 1376-1570: 4 tests — `lowest_cost_is_argmin`, `winner_le_each_restart`, `never_worse_than_single_init`, `deterministic` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `soft_dtw_barycenter` (4-arg public) | `lbfgs_refine_from_init` | Line 870: `lbfgs_refine_from_init(&rows, mean, gamma, max_iter, tol)` | WIRED | One descent implementation, two callers — confirmed |
| `soft_dtw_barycenter_multirestart` | `lbfgs_refine_from_init` | Line 782: `lbfgs_refine_from_init(&rows, init, gamma, max_iter, tol)` | WIRED | Called once per restart k in 0..n_runs |
| Restart k>=1 seeding | `perturbed_init` | Line 779: `perturbed_init(&mean, &col_std, seed, k as u64)` | WIRED | Seeding convention: `seed.wrapping_add(k)` (safe wrapping equivalent of `seed + k`) |
| Winner selection | `soft_dtw_distance` | Lines 785-788: `rows.iter().map(|r| soft_dtw_distance(&result.barycenter, r, gamma)).sum()` | WIRED | Total cost matches the objective `lbfgs_refine_from_init` minimizes |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 4 SDTW-02 multirestart tests pass | `cargo test -p fdars-core --features linalg,parallel barycenter_multirestart_tests` | 4 passed, 0 failed | PASS |
| All 28 soft_dtw tests pass (incl. regression guard) | `cargo test -p fdars-core --features linalg,parallel soft_dtw` | 28 passed, 0 failed; `test_soft_dtw_barycenter_vs_tslearn` OK | PASS |
| `lbfgs_refine_from_init` exists as private fn | grep | Line 575 confirmed | PASS |
| `soft_dtw_barycenter_multirestart` is private (no `pub`) | grep | No `pub fn soft_dtw_barycenter_multirestart` found | PASS |
| No new crate dependency | `git diff 6c125b90 HEAD -- fdars-core/Cargo.toml` | Empty diff | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| SDTW-02 | 102-01-PLAN.md | Multi-restart wrapper with deterministic seeding, lowest-cost winner selection, reproducible for fixed seed | SATISFIED | `soft_dtw_barycenter_multirestart` implements all required properties; 4 SDTW-02 unit tests pass; `requirements-completed: [SDTW-02]` in SUMMARY |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `fdars-core/src/metric/soft_dtw.rs` | 689, 715, 739 | `#[allow(dead_code)]` on 3 private helpers | Info | Intentional — Phase 103 will wire them through the public config API; codebase uses this pattern elsewhere (e.g. `wire.rs`, `seasonal/mod.rs`); noted explicitly in SUMMARY key-decisions |

No `TBD`, `FIXME`, or `XXX` markers found in modified files.

### Human Verification Required

None. All must-haves are verifiable programmatically and confirmed by passing tests.

### Gaps Summary

No gaps. All 5 must-have truths are VERIFIED, all artifacts exist and are wired, all 4 SDTW-02 tests pass, the existing regression guard (`test_soft_dtw_barycenter_vs_tslearn`) passes, no new dependency was added, and the public API signature is unchanged.

---

_Verified: 2026-09-13T10:30:00Z_
_Verifier: Claude (gsd-verifier)_
