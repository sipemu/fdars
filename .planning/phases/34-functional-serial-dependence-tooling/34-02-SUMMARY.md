---
phase: 34-functional-serial-dependence-tooling
plan: "02"
subsystem: fts
tags: [functional-time-series, stationarity, differencing, kpss, permutation, monte-carlo]
requirements: [FTS-02]

dependency_graph:
  requires:
    - fdars_core::fts::functional_acf  (plan 34-01 — reuses validate_fts_input, mean_curve, simpsons_weights)
    - fdars_core::fts::StationarityResult (plan 34-01 — struct declared in mod.rs, populated here)
  provides:
    - fdars_core::fts::functional_difference
    - fdars_core::fts::stationarity_test
    - fdars_core::fts::StationarityResult  (now producing function delivered)
  affects:
    - fdars-core/src/fts/acf.rs   (new public functions + 6 new inline tests)
    - fdars-core/src/fts/mod.rs   (pub use extended)
    - fdars-core/src/lib.rs       (pub use fts extended)

tech_stack:
  added: []
  patterns:
    - KPSS-style partial-sum functional statistic (T = (1/N²) Σ_k ‖S_k‖²_L2, Simpson-weighted)
    - Single StdRng::seed_from_u64(seed) for all n_perm Fisher-Yates shuffles (matches inference/permutation.rs)
    - (n_ge + 1) / (n_perm + 1) MC p-value formula (project convention)
    - Cumulative-sum round-trip verification for first-difference operator

key_files:
  created: []
  modified:
    - fdars-core/src/fts/acf.rs
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/lib.rs

decisions:
  - "Use unnormalized KPSS partial-sum statistic with pure permutation p-value — valid regardless of HKR 2014 long-run-variance normalization constant (not pinned from documentation alone); documented in rustdoc DIVERGENCE note as a future-precision item"
  - "Centered-curve buffer stored as flat row-major Vec<f64> for efficient index-permutation in the stationarity test loop (avoids per-iteration FdMatrix row access)"
  - "Both functional_difference and stationarity_test committed atomically in a single commit (06e20920) because all edits were staged together before the first git add; stationarity_test code was complete when staged"

metrics:
  duration: "20m"
  completed: "2026-08-21"
  tasks_completed: 2
  tasks_total: 2

status: complete

actuals:
  tokens: 14000
  tasks: 2
  commits: 1
---

# Phase 34 Plan 02: Functional Difference + Stationarity Test Summary

Expanded `fts/acf.rs` with the two tools that build directly on the plan-34-01 centering/L2 spine: the functional first-difference operator (with cumulative-sum round-trip) and the Monte-Carlo functional stationarity test (KPSS-style partial-sum statistic with seeded permutation p-value).

## What Was Built

| Artifact | Description |
|----------|-------------|
| `functional_difference` | First-order difference D[i,j] = data[(i+1,j)] - data[(i,j)], output (N-1)×m; InvalidDimension when N<2; round-trips within 1e-10 via cumulative sum |
| `stationarity_test` | KPSS-style T = (1/N²) Σ_k ‖S_k‖²_L2 with permutation p-value; single StdRng::seed_from_u64(seed); rejects trended, accepts stationary |
| `fdars-core/src/fts/acf.rs` | Both functions + 6 new inline tests (17 total in module) |
| `fdars-core/src/fts/mod.rs` | Extended pub use: functional_difference, stationarity_test |
| `fdars-core/src/lib.rs` | Extended pub use fts: functional_difference, stationarity_test, StationarityResult |

## Algorithm Correctness Notes

- **functional_difference:** Exact first-order finite difference. Round-trip via cumulative sum is exact up to floating-point rounding (~1e-16 for typical f64 inputs); test tolerance set conservatively at 1e-10.
- **stationarity_test:** Centered curves buffered flat as `centered[i * m + j]` for cache-friendly permutation-loop access. Partial sums accumulated incrementally (O(N·m) per permutation, not O(N²·m)). The unnormalized statistic grows with dataset variance; the permutation p-value remains calibrated because it compares against the same dataset's permutations.
- **p-value formula:** `(n_ge + 1) / (n_perm + 1)` — identical to the pattern in `inference/permutation.rs:183`, ensuring project-wide consistency.

## Tests Delivered and Results

All 17 tests in `fts::acf::tests` pass (seeded, reproducible):

| Test | Purpose | Result |
|------|---------|--------|
| `diff_roundtrip` | 8×15 analytic sin data, round-trip < 1e-10 | PASS |
| `diff_too_few_rows` | InvalidDimension for 0-row and 1-row matrices | PASS |
| `stat_test_stationary` | White-noise GP n=60: p-value > 0.05 (no rejection) | PASS |
| `stat_test_nonstationary` | i*t trended n=50: p-value <= 0.05 (rejects) | PASS |
| `stat_test_deterministic` | Same seed gives bit-identical StationarityResult | PASS |
| `stat_test_invalid` | n_perm==0 → InvalidParameter; empty matrix → InvalidDimension; argvals mismatch → InvalidDimension | PASS |
| (11 from plan 34-01) | All prior fACF/fPACF/band tests | PASS |

## Deviations from Plan

### Commit Granularity

Both `functional_difference` and `stationarity_test` were committed in a single atomic commit (06e20920) rather than two separate task commits. This occurred because all edits to `acf.rs`, `mod.rs`, and `lib.rs` were complete before the first `git add` was issued. The plan called for per-task commits; the actual outcome is one commit covering both tasks. All code is correct and all tests pass.

### No Other Deviations

Plan executed exactly as written. All functions, error variants, test names, and rustdoc notes match the plan specification.

## Threat Mitigations Applied

| ID | Threat | Mitigation |
|----|--------|-----------|
| T-34-04 | difference index `i+1` on <2-row matrix | `n < 2` returns `FdarError::InvalidDimension` before any index |
| T-34-05 | n_perm==0 degenerate p-value | `n_perm == 0` returns `FdarError::InvalidParameter`; permutation loop is O(n_perm · N · m) and caller-bounded |

## Known Stubs

None — `StationarityResult` fields (statistic, p_value, n_perm) are all populated.

`LongRunCovResult` and `long_run_covariance` remain unimplemented (deferred to plan 34-03 per phase planning).

## Threat Flags

None — no new network, filesystem, or deserialization surface introduced.

## Self-Check

- `functional_difference` callable at crate root: CONFIRMED (pub use fts::functional_difference in lib.rs)
- `stationarity_test` callable at crate root: CONFIRMED (pub use fts::stationarity_test in lib.rs)
- `StationarityResult` callable at crate root: CONFIRMED (pub use fts::StationarityResult in lib.rs)
- Commit 06e20920 exists: CONFIRMED
- All 17 fts::acf tests pass: CONFIRMED
- `git diff --exit-code fdars-core/Cargo.toml`: clean

## Self-Check: PASSED
