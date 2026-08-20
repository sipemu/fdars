---
phase: 30-interval-testing-procedure-family
plan: "02"
subsystem: inference/itp
status: complete
tags: [inference, itp, functional-testing, permutation]

dependency_graph:
  requires: ["30-01"]
  provides: [itp_two_pop, itp_flm, component_t_stat, pool_coefficients_itp, shuffle_itp]
  affects: [inference/itp.rs, inference/mod.rs, lib.rs]

tech_stack:
  added: []
  patterns:
    - "Pool + relabel permutation (Fisher-Yates inline copy of permutation.rs pattern)"
    - "Response-permutation null for FLM coefficient test"
    - "OLS t-statistic with degenerate-guard (sxx<1e-30, se2<=0.0 return 0.0)"
    - "Localized regression fixture: linear ramp curves with y=mean(X on subinterval)"

key_files:
  created: []
  modified:
    - fdars-core/src/inference/itp.rs
    - fdars-core/src/inference/mod.rs
    - fdars-core/src/lib.rs

decisions:
  - "Reused plan-01 rank_transform/build_pval_matrix/pval_correct/fisher_cf unchanged per plan spec"
  - "itp_flm uses response-permutation (shuffle y) — Assumption A2 simplification vs R ITPlmbspline partial-residual method; documented in rustdoc"
  - "component_t_stat guards: sxx<1e-30 -> 0.0; se2<=0.0 -> 0.0 (no NaN/divide-by-zero per T-30-04)"
  - "shuffle_itp is an inline private copy of permutation.rs shuffle_labels (that fn is private, not importable)"
  - "pool_coefficients_itp pools coefficient rows (n_a+n_b, p) analogous to pool_two_samples over coefficient space"
  - "flm_effect fixture changed from sine-amplitude correlation to linear-ramp-mean: stronger, more localized signal"

metrics:
  duration: "~20 minutes"
  completed: "2026-08-20"
  tasks_completed: 2
  commits: 2

actuals:
  tokens: 18000
  tasks: 2
  commits: 2
---

# Phase 30 Plan 02: itp_two_pop + itp_flm Summary

Two new public entry points added to `inference/itp.rs`, completing the ITP family (INF-03). Both reuse plan-01's closure-adjustment helpers (`rank_transform`, `build_pval_matrix`, `pval_correct`, `fisher_cf`) unchanged.

## What Was Built

### itp_two_pop — Two-population pool + relabel ITP

- Validates two groups via `validate_two_samples_itp` (n_a<2||n_b<2, m_a!=m_b, argvals mismatch, nbasis<2, n_perm==0)
- Projects each group independently via `fdata_to_basis`; uses `proj_a.n_basis` as `p` (B-spline clamp-safe)
- Pools coefficient rows `(n_a+n_b, p)` via `pool_coefficients_itp`
- Observed stat per component k: `|colMean(coeff_a[:,k]) - colMean(coeff_b[:,k])|`
- Single sequential `StdRng::seed_from_u64(seed)` loop; Fisher-Yates relabeling via `shuffle_itp`
- Raw p = `(n_ge+1)/(n_perm+1)`; then plan-01 closure pipeline
- 4 inline tests: `two_population_localized` / `null` / `deterministic` / `error_paths`

### itp_flm — Interval-wise FLM coefficient test (response permutation)

- Validates data/y/argvals dims (n<2, y.len()!=n, argvals mismatch, nbasis<2, n_perm==0)
- Projects X once via `fdata_to_basis`; per-component `component_t_stat` computes `|beta/se|`
- Guards in `component_t_stat`: `sxx<1e-30` -> 0.0; `se2<=0.0` -> 0.0 (T-30-04)
- Response-permutation null: shuffle `y` via `shuffle_itp` per permutation, recompute all component t-stats
- Rustdoc documents Assumption A2 divergence from R's `ITPlmbspline` partial-residual method
- 3 inline tests: `flm_effect` / `flm_null` / `flm_error_paths`

### Re-exports

- `inference/mod.rs`: `pub use itp::{itp_flm, itp_one_pop, itp_two_pop, ItpResult};`
- `lib.rs`: inference block extended with `itp_flm, itp_two_pop`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Compile error: `|&r|` pattern on `usize` in permutation loop**
- **Found during:** Task 1 compilation
- **Issue:** `(0..n_a).map(|&r| pooled[(r, k)])` — range yields `usize` directly, not a reference
- **Fix:** Changed `|&r|` to `|r|`
- **Files modified:** `fdars-core/src/inference/itp.rs`
- **Commit:** 4ec7a260

**2. [Rule 1 - Bug] flm_effect fixture too weak — sine-amplitude correlation insufficient**
- **Found during:** Task 2 test run (flm_effect panicked: min adjusted p = 0.964)
- **Issue:** `y = amplitude_of_sine + noise` produces a weak, non-localized correlation across all components; the ITP closure adjustment pushes all p-values toward 1.0 in a diffuse signal
- **Fix:** Replaced fixture with linear ramp curves (`v = scale*t + offset`) where `y = mean(X on [0.3, 0.7])` — strong, localized linear signal produces clear regression in the target sub-interval
- **Files modified:** `fdars-core/src/inference/itp.rs`
- **Commit:** 3f2a9c6c

## Gate Results

| Gate | Status |
|------|--------|
| `cargo fmt -p fdars-core` | Green (both tasks) |
| `cargo test --lib inference::itp::tests::two_population` | Green: 4/4 |
| `cargo test --lib inference::itp` | Green: 13/13 |
| `cargo build --features serde` | Green |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Green (both tasks) |

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, or trust boundaries introduced.

## Self-Check: PASSED

- `fdars-core/src/inference/itp.rs` exists and contains `itp_two_pop`, `itp_flm`, `component_t_stat`, `pool_coefficients_itp`, `shuffle_itp`
- `fdars-core/src/inference/mod.rs` exports `itp_flm, itp_one_pop, itp_two_pop, ItpResult`
- `fdars-core/src/lib.rs` re-exports `itp_flm, itp_two_pop` at crate root
- Commits verified: 4ec7a260 (task 1), 3f2a9c6c (task 2)
- All 13 itp tests pass; serde build clean; clippy clean
