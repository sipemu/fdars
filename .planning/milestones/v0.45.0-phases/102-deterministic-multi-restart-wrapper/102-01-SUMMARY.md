---
phase: 102-deterministic-multi-restart-wrapper
plan: 01
subsystem: metric
tags: [soft-dtw, barycenter, multi-restart, determinism, rng, optimization, lbfgs]

# Dependency graph
requires:
  - phase: 101-in-crate-l-bfgs-barycenter-optimizer
    provides: single-init L-BFGS descent in soft_dtw_barycenter (SDTW-01)
provides:
  - lbfgs_refine_from_init: private single-init L-BFGS refiner (extracted from soft_dtw_barycenter)
  - per_col_std: private per-evaluation-point population std helper
  - perturbed_init: private seeded Gaussian-perturbed restart init
  - soft_dtw_barycenter_multirestart: private deterministic multi-restart driver (SDTW-02)
  - barycenter_multirestart_tests: 4 SDTW-02 unit tests (lowest-cost, winner<=restarts, never-worse, determinism)
affects: [103-config-api, 104-quality-evidence, 105-release-prep]

actuals:
  tokens: 5025       # 20098 chars / 4 over the realized diff
  tasks: 3
  commits: 2
  plan_head_before: f70e12302562f0b575bcf5470065542bae7aed8f

tech-stack:
  added: []           # no new crate dependency — rand/rand_distr already exist
  patterns:
    - "Extract-and-delegate: lbfgs_refine_from_init extracted from soft_dtw_barycenter; public 4-arg entry delegates unchanged"
    - "StdRng::seed_from_u64(seed + k) per-restart seeding (project convention)"
    - "rand_distr::Normal for Gaussian perturbations scaled by per-column population std"
    - "Strictly-less argmin tiebreak: earlier restart index wins on tied cost (determinism)"
    - "#[allow(dead_code)] on private helpers not yet called from production code (Phase 103 will wire them)"

key-files:
  created: []
  modified:
    - fdars-core/src/metric/soft_dtw.rs

key-decisions:
  - "lbfgs_refine_from_init takes owned init Vec<f64> and rows &[Vec<f64>] — one descent impl, two callers (4-arg entry and multi-restart driver)"
  - "per_col_std uses population std (not sample std): guard n<=1 yields 0.0, zero-std coordinates unperturbed"
  - "perturbed_init draws noise in single left-to-right pass over j — RNG consumption order is fixed for determinism"
  - "#[allow(dead_code)] on three private helpers (per_col_std, perturbed_init, soft_dtw_barycenter_multirestart) until Phase 103 exposes the public API"

requirements-completed: [SDTW-02]

coverage:
  - id: D1
    description: "lbfgs_refine_from_init extracted; public 4-arg soft_dtw_barycenter delegates to it"
    requirement: SDTW-02
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_lbfgs_tests: all 5 pass (regression guard)"
        status: pass
      - kind: integration
        ref: "fdars-core/tests/validate_against_r.rs#test_soft_dtw_barycenter_vs_tslearn"
        status: pass
    human_judgment: false
  - id: D2
    description: "Multi-restart driver: lowest-cost argmin with lowest-index tiebreak"
    requirement: SDTW-02
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_multirestart_tests::lowest_cost_is_argmin"
        status: pass
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_multirestart_tests::winner_le_each_restart"
        status: pass
    human_judgment: false
  - id: D3
    description: "Never-worse than single-init: multi-restart cost <= 4-arg cost + 1e-9"
    requirement: SDTW-02
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_multirestart_tests::never_worse_than_single_init"
        status: pass
    human_judgment: false
  - id: D4
    description: "Bit-identical determinism: same seed + inputs -> same barycenter (f64::to_bits)"
    requirement: SDTW-02
    verification:
      - kind: unit
        ref: "fdars-core/src/metric/soft_dtw.rs#barycenter_multirestart_tests::deterministic"
        status: pass
    human_judgment: false
  - id: D5
    description: "WASM-safe, serde-safe, no new crate dependency"
    requirement: SDTW-02
    verification:
      - kind: other
        ref: "cargo check -p fdars-core --target wasm32-unknown-unknown: pass"
        status: pass
      - kind: other
        ref: "cargo check -p fdars-core (default features): pass"
        status: pass
      - kind: other
        ref: "cargo build -p fdars-core --features serde: pass"
        status: pass
      - kind: other
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings: pass"
        status: pass
    human_judgment: false

duration: 5min
completed: 2026-09-13
status: complete
---

# Phase 102 Plan 01: Deterministic Multi-Restart Wrapper Summary

**Extracted Phase 101's single-init L-BFGS descent into a reusable private helper and wrapped it with a deterministic multi-restart driver (SDTW-02) — no new dependency, wasm32-safe, 4 SDTW-02 unit tests including lowest-cost argmin, winner<=restarts, never-worse-than-single-init, and bit-identical determinism**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-09-13T00:41:28Z
- **Completed:** 2026-09-13T00:46:00Z
- **Tasks:** 3 (Task 3 was verification-only, no code changes)
- **Files modified:** 1

## Accomplishments

- Extracted the Phase 101 L-BFGS descent body into `lbfgs_refine_from_init(rows, init, gamma, max_iter, tol) -> SoftDtwBarycenterResult` — a reusable private helper carrying all safeguards (curvature guard, non-descent fallback, Armijo minimum step)
- Rewired the public 4-arg `soft_dtw_barycenter` to delegate to `lbfgs_refine_from_init` — signature and `SoftDtwBarycenterResult` struct shape byte-compatible, all Phase 101 tests pass as regression guards
- Added `per_col_std(rows, m)`: per-evaluation-point population std helper (n<=1 guard yields 0.0)
- Added `perturbed_init(mean, col_std, seed, k)`: seeded Gaussian perturbation using `StdRng::seed_from_u64(seed + k)` + `rand_distr::Normal`, single left-to-right pass for determinism
- Added `soft_dtw_barycenter_multirestart(data, gamma, max_iter, tol, n_restarts, seed)`: deterministic multi-restart driver — restart 0 = unperturbed mean, restarts k>=1 = perturbed inits; winner = strictly-lowest total soft-DTW cost with lowest-index tiebreak
- Added 4 SDTW-02 unit tests in `barycenter_multirestart_tests` module
- All 3 build guards pass: wasm32, default features, `--features serde`; clippy `--all-targets` clean; no new dependency

## Task Commits

1. **Task 1: End-to-end multi-restart driver** - `ab89d40e` (feat)
2. **Task 2: SDTW-02 unit tests** - `e34a22cd` (test)
3. **Task 3: Build guards (verification-only)** - no commit (all guards passed without code changes)

## Files Created/Modified

- `fdars-core/src/metric/soft_dtw.rs` — `lbfgs_refine_from_init`, `per_col_std`, `perturbed_init`, `soft_dtw_barycenter_multirestart`, `barycenter_multirestart_tests` module (+398 / -51 lines net)

## Decisions Made

- `lbfgs_refine_from_init` takes owned `init: Vec<f64>` (caller moves the init in) and `rows: &[Vec<f64>]` (borrowed) — no extra allocation vs. the pre-refactor code
- `per_col_std` uses population std (denominator n, not n-1): zero-std coordinates yield 0.0 so restart 0 (unperturbed mean) already covers them; guard for n<=1 returns all-zeros (no perturbation possible with one row)
- `perturbed_init` draws `N(0, col_std[j])` noise via `StdRng::seed_from_u64(seed + k)` in a single left-to-right `j` loop — this exact RNG consumption order is the determinism invariant; future code must not change the draw order
- `#[allow(dead_code)]` on the three private helpers: they are not yet called from production code (Phase 103 will wire `soft_dtw_barycenter_multirestart` through the public config API); the codebase uses `#[allow(dead_code)]` at item level elsewhere (e.g. `wire.rs`, `seasonal/mod.rs`)
- Task 3 required no code fixes — wasm32, default-feature, serde, and clippy all passed cleanly on the first attempt

## Deviations from Plan

### Auto-fixed Issues

None — plan executed exactly as written. The only non-obvious choice was adding `#[allow(dead_code)]` to the three private helpers (anticipated in the plan's "Keep everything private" note since Phase 103 is the caller).

---

**Total deviations:** 0

## Issues Encountered

None — all 3 tasks executed cleanly. Build guards passed without fixes.

## Known Stubs

None. All new code is complete and tested. The private helpers are intentionally not yet called from production code; Phase 103 will wire them through the public config API.

## Self-Check

### Files

- [x] `fdars-core/src/metric/soft_dtw.rs` — FOUND (lbfgs_refine_from_init, per_col_std, perturbed_init, soft_dtw_barycenter_multirestart, barycenter_multirestart_tests)

### Commits

- [x] `ab89d40e` — feat(102-01): extract lbfgs_refine_from_init + add multi-restart driver
- [x] `e34a22cd` — test(102-01): SDTW-02 unit tests

### Tests

- [x] 4/4 SDTW-02 unit tests pass (lowest_cost_is_argmin, winner_le_each_restart, never_worse_than_single_init, deterministic)
- [x] 5/5 Phase 101 barycenter_lbfgs_tests pass (regression guard)
- [x] test_soft_dtw_barycenter_vs_tslearn pass (integration regression guard)
- [x] 28 total soft_dtw tests pass

## Self-Check: PASSED

## Next Phase Readiness

- Phase 103 (Public Config API) can consume `soft_dtw_barycenter_multirestart` directly — the function signature `(data, gamma, max_iter, tol, n_restarts, seed)` is ready; Phase 103 adds `SoftDtwBarycenterConfig` with those fields and a `Result`-returning entry point
- The `#[allow(dead_code)]` annotations on the three private helpers should be removed in Phase 103 once the public API calls them
- No blockers; all gates green

---
*Phase: 102-deterministic-multi-restart-wrapper*
*Completed: 2026-09-13*
