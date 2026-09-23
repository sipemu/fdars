---
phase: 115-svd-fpca-covariance-optimization
plan: "01"
subsystem: irreg_fdata / face_covariance
tags: [benchmark, golden-fixture, before-baseline, covariance, opt-02]
status: complete

dependency_graph:
  requires: []
  provides:
    - "BEFORE-baseline criterion (perf_face_covariance/n200_m30): 268,166,197.56 ns mean, CI [260.1 ms, 275.9 ms], SHA 328215e7"
    - "Golden-value fixture: fdars-core/tests/equivalence_phase115.rs (n=10, m=11, tol 1e-10)"
  affects:
    - "Plan 115-02: sym_cov_irreg optimization — compares AFTER benchmark against this baseline; verifies numeric identity against this golden fixture"

tech_stack:
  added: []
  patterns:
    - "Committed BEFORE-baseline pattern: criterion estimates.json + BASELINE.md in .planning/ before any src edit"
    - "Golden-value fixture pattern: assert_abs_close helper, make_test_ifd generator, const-per-checkpoint layout (from equivalence_phase114)"

key_files:
  created:
    - "fdars-core/tests/equivalence_phase115.rs — golden-value numeric-identity fixture (face_covariance, n=10 m=11, tol 1e-10, symmetry invariant)"
    - ".planning/phases/115-svd-fpca-covariance-optimization/115-baseline-criterion/perf_face_covariance__n200_m30/estimates.json"
    - ".planning/phases/115-svd-fpca-covariance-optimization/115-baseline-criterion/perf_face_covariance__n200_m30/benchmark.json"
    - ".planning/phases/115-svd-fpca-covariance-optimization/115-baseline-criterion/perf_face_covariance__n200_m30/BASELINE.md"
  modified: []

decisions:
  - "Used criterion slope estimate (~260 ms) as the conservative BEFORE number for speedup ratio; plan 02 should compare mean-to-mean for fairness"
  - "Separate constants for (0,5) and (5,0) mirror cells: pre-opt code computes both halves independently (differ by ~8e-18); post-opt (5,0) is a bit-copy of (0,5). Using assert_abs_close at 1e-10 for both ensures the fixture is green on both code versions"
  - "No assert_eq! for symmetry invariant — pre-opt symmetric within ~1e-17, post-opt bit-exact; 1e-10 tolerance covers both correctly"

metrics:
  duration: "75 minutes"
  completed: "2026-09-23"
  tasks_completed: 2
  commits: 2
  plan_head_before: "328215e762f7b37f6727762af5a528164eeaa767"

estimate:
  tokens: 55000
  tasks: 2
  confidence: med

actuals:
  tokens: 12000
  tasks: 2
  commits: 2
---

# Phase 115 Plan 01: Tracer — BEFORE-Baseline + Golden Fixture Summary

Captured the committed BEFORE-optimization evidence for the `face_covariance` covariance-surface
optimization (OPT-02) **before any `fdars-core/src` change**: (1) a committed criterion baseline on
the reproducible `perf_face_covariance/n200_m30` cell, and (2) a golden-value test locking the
current (un-optimized) `face_covariance` output to literal `f64` constants at tolerance 1e-10.

## BEFORE-Baseline Criterion (Task 1)

| Metric | Value |
|--------|-------|
| Cell | `perf_face_covariance/n200_m30` |
| Bench binary | `perf_hotpaths` (features: `linalg,parallel`) |
| **mean point_estimate** | **268,166,197.56 ns (~268 ms)** |
| mean CI lower (95%) | 260,142,509.90 ns (~260 ms) |
| mean CI upper (95%) | 275,883,662.23 ns (~276 ms) |
| slope point_estimate | 260,026,606.58 ns (~260 ms) |
| slope CI | [250.2 ms, 272.1 ms] |
| median | 268,850,616.50 ns (~269 ms) |
| sample_size | 15 |
| Git SHA | `328215e762f7b37f6727762af5a528164eeaa767` |
| Machine | 13th Gen Intel i9-13900H, Linux 6.18.45-1-MANJARO |

**Commit:** `df84b586`

Plan 02 re-runs the identical command and computes `after_mean_ns / 268_166_197.56`. The hard gate
is `after < before` (any positive speedup). Expected speedup for the upper-triangle+mirror refactor:
~1.5–2x (OPT-02 eliminates ~50% of the `cov_irreg` inner loop work).

Note on orphaned bench group: the on-disk `target/criterion/face_covariance/` directory is an
orphaned Phase-113 probe with no source bench. The reproducible cell is `perf_face_covariance`
(source in `fdars-core/benches/perf_hotpaths.rs`).

## Golden-Value Fixture (Task 2)

File: `fdars-core/tests/equivalence_phase115.rs`
**Commit:** `9fd46576`

Test: `golden_face_covariance_n10_m11` — `make_test_ifd(10, 11)`, bandwidth 0.3

### Captured Constants

| Cell | Constant | Value |
|------|----------|-------|
| `cov[(0,0)]` | `COV_0_0` | `2.01522424817674774e-2` |
| `cov[(5,5)]` | `COV_5_5` | `2.26413976377977649e-3` |
| `cov[(10,10)]` | `COV_10_10` | `1.47740963472988169e-2` |
| `cov[(0,5)]` | `COV_0_5` | `-3.31275058129442681e-3` |
| `cov[(5,0)]` | `COV_5_0` | `-3.31275058129442768e-3` |
| `cov[(2,7)]` | `COV_2_7` | `-5.92895218221224912e-3` |
| `cov[(7,2)]` | `COV_7_2` | `-5.92895218221224998e-3` |

All assertions use `assert_abs_close(cell, CONST, 1e-10)`. The (5,0) and (7,2) mirror constants
differ from their upper-triangle partners by ~8e-18 (pre-opt code computes both halves
independently); post-opt they will be bit-exact copies. Using separate constants for each ensures
the fixture is verifiable on both code versions.

Symmetry invariant: `assert_abs_close(cov[(i,j)], cov[(j,i)], 1e-10)` for all `i > j` in
`0..11`. Green on pre-opt (~1e-17 asymmetry) and will be green on post-opt (bit-exact).

## Verification Results

All automated verifications pass:
- `estimates.json` present with positive mean + valid CI (mean=268ms, lo=260ms, hi=276ms, hi>lo)
- Task 1 commit contains no `fdars-core/src/` files
- `cargo test --test equivalence_phase115`: `test result: ok. 1 passed`
- Task 2 commit contains `equivalence_phase115.rs` and no `fdars-core/src/` files

## Deviations from Plan

None — plan executed exactly as written.

## Threat Flags

None found. This plan adds only a test file and committed benchmark output; no network I/O, no
public API change, no new dependencies, no untrusted data.

## Known Stubs

None.

## Self-Check: PASSED

- `fdars-core/tests/equivalence_phase115.rs` — FOUND
- `.planning/phases/115-svd-fpca-covariance-optimization/115-baseline-criterion/perf_face_covariance__n200_m30/estimates.json` — FOUND
- Task 1 commit `df84b586` — verified in git log
- Task 2 commit `9fd46576` — verified in git log
- `COMMITS_ACTUAL=2` matches 2 tasks committed
