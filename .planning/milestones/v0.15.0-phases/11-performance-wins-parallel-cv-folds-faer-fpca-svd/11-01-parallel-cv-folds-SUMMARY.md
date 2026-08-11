---
phase: 11-performance-wins-parallel-cv-folds-faer-fpca-svd
plan: "01"
subsystem: classification/cv
status: complete
tags: [performance, parallelism, cv, classification, rayon]

dependency_graph:
  requires: []
  provides: [parallel-cv-folds]
  affects: [fdars-core/src/classification/cv.rs]

tech_stack:
  added: []
  patterns:
    - iter_maybe_parallel! macro for feature-gated parallel fold iteration
    - Rayon collect-in-order for deterministic parallel Vec<f64> assembly

key_files:
  modified:
    - fdars-core/src/classification/cv.rs

decisions:
  - Use iter_maybe_parallel!(0..nfold).map(...).collect() — follows the karcher.rs canonical pattern, no direct rayon import for the iteration itself
  - ParallelIterator trait imported under #[cfg(feature = "parallel")] — required for .collect() to resolve on the parallel iterator type
  - Equivalence test calls fclassif_cv twice with the same seed; asserts bit-for-bit fold_errors and error_rate equality (not approximate)

metrics:
  duration: "7m"
  completed: "2026-08-10"
  tasks_completed: 2
  commits: 2
  files_changed: 1

actuals:
  tokens: 9500
  tasks: 2
  commits: 2
---

# Phase 11 Plan 01: Parallel CV Folds Summary

## One-liner

Replaced the sequential `for fold in 0..nfold { fold_errors.push(...) }` loop in `fclassif_cv` with `iter_maybe_parallel!(0..nfold).map(...).collect()`, enabling parallel fold execution under the `parallel` feature with bit-for-bit identical results.

## What Was Built

**Task 1 (tracer): Parallelize the fclassif_cv fold loop**

Modified `fdars-core/src/classification/cv.rs`:
- Added `use crate::iter_maybe_parallel;` and `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;` to the import block
- Replaced `let mut fold_errors = Vec::with_capacity(nfold);` + `for fold in 0..nfold { ... fold_errors.push(errors); }` with `let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold).map(|fold| { ... }).collect();`
- The per-fold body is moved verbatim into the closure; the trailing `errors` value is the closure's return expression
- Sequential path (default features) produces identical output to the prior implementation

**Task 2 (auto): Add sequential-vs-parallel equivalence test**

Added `#[cfg(test)] mod tests` block to `fdars-core/src/classification/cv.rs`:
- `make_test_data(n, m)` helper: n=20 observations, m=10 evaluation points, 2 well-separated Gaussian-bump classes
- `test_fclassif_cv_parallel_matches_sequential`: calls `fclassif_cv` twice with identical arguments and seed=42; asserts element-wise bit-for-bit equality of `fold_errors` and `error_rate`
- Compiled and passes under both default features and `--features parallel`

## Verification Results

| Check | Result |
|-------|--------|
| `cargo build -p fdars-core` (default) | PASS |
| `cargo build -p fdars-core --features parallel` | PASS |
| `cargo test -p fdars-core test_fclassif_cv_parallel_matches_sequential` | PASS |
| `cargo test -p fdars-core --features parallel test_fclassif_cv_parallel_matches_sequential` | PASS |
| `cargo clippy -p fdars-core --features parallel -- -D warnings` | PASS |
| `grep 'iter_maybe_parallel!(0..nfold)'` | 1 match |
| `grep 'fold_errors.push'` | 0 matches (removed) |
| `grep 'let mut fold_errors'` | 0 matches (no longer mut) |
| `cargo test -p fdars-core` (full suite, 1941 tests) | PASS |
| `cargo test -p fdars-core --features parallel` (1941 tests) | PASS |

## Commits

| Hash | Message |
|------|---------|
| 23118b0b | feat(11-01): parallelize fclassif_cv fold loop via iter_maybe_parallel! |
| 30832954 | test(11-01): add sequential-vs-parallel equivalence test for fclassif_cv |

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None.

## Threat Surface Scan

No new trust boundaries, network endpoints, or auth paths introduced. The change is an internal refactor of a private fold loop. T-11-01-01 (Tampering/result correctness under parallelism) is mitigated by the bit-for-bit equivalence test.

## Self-Check: PASSED

- [x] `fdars-core/src/classification/cv.rs` modified and contains `iter_maybe_parallel!(0..nfold)` and `test_fclassif_cv_parallel_matches_sequential`
- [x] Commit 23118b0b exists (feat — fold loop parallel)
- [x] Commit 30832954 exists (test — equivalence test)
- [x] All 1941 tests pass under both default and parallel features
