---
phase: 15-elastic-fpca-performance
plan: "01"
subsystem: elastic_fpca
tags: [performance, parallelism, iter_maybe_parallel, elastic-fpca]
status: complete
completed: "2026-08-12"

dependency_graph:
  requires: []
  provides:
    - "PERF-04: three per-curve loops in elastic_fpca.rs parallelized"
    - "SCORES_PARALLEL_THRESHOLD named constant (= 50)"
    - "six PERF-04-A..F equivalence tests"
  affects:
    - fdars-core/src/elastic_fpca.rs

tech_stack:
  added: []
  patterns:
    - "collect-then-assign parallelism (mirrors alignment/set.rs align_to_target)"
    - "outer-if N>=50 threshold guard before parallel dispatch on light body"
    - "iter_maybe_parallel! feature-gated macro (existing primitive, first use in elastic_fpca.rs)"

key_files:
  modified:
    - fdars-core/src/elastic_fpca.rs

decisions:
  - "Collect-then-assign pattern (not parallel-write into column-major FdMatrix) — mirrors align_to_target; avoids data-race on shared buffer"
  - "SCORES_PARALLEL_THRESHOLD = 50 outer-if at function level (not per-k inside loop) — one branch decision, not repeated check"
  - "Test strategy: exact equality for pure-write loops (bit-identical), determinism re-run for SVD-based entry points"

metrics:
  duration_minutes: 12
  tasks_completed: 4
  commits: 4
  files_modified: 1

actuals:
  tokens: 18000
  tasks: 4
  commits: 4
---

# Phase 15 Plan 01: Elastic-FPCA Parallelization Summary

Parallelized the three per-curve loops on the elastic-FPCA critical path via `iter_maybe_parallel!`, producing numerically equivalent output to the sequential path. Change is additive and non-breaking: no public signatures changed, no new dependencies, and with the `parallel` feature off the macro compiles to sequential `into_iter` with no behavior change.

## What Was Built

Three internal loops in `fdars-core/src/elastic_fpca.rs` now use `iter_maybe_parallel!` under the `parallel` feature:

1. **`shooting_vectors_from_psis` (PERF-04-A)** — the heavy body (`inv_exp_map_sphere` per curve) uses the collect-then-assign pattern: `iter_maybe_parallel!(0..n).map(|i| inv_exp_map_sphere(...)).collect::<Vec<Vec<f64>>>()`, then sequential row-assign into the column-major `FdMatrix`.

2. **`build_augmented_srsfs` (PERF-04-B)** — the medium body (copy SRSF row + augmented column) uses the same collect-then-assign pattern: each `i` produces an owned `Vec<f64>` of length `m+1`, collected in parallel, then assigned sequentially.

3. **`svd_scores_and_eigenvalues` (PERF-04-C)** — the light body (single multiply `u[(i,k)] * sv`) is guarded by an outer-if on the named constant `SCORES_PARALLEL_THRESHOLD = 50`: at N >= 50 uses `iter_maybe_parallel!(0..n).map(...).collect()` then sequential fill; below 50 uses the original sequential nested loops.

A module-level doc comment on the constant explains the streaming-sentinel payback rationale from the audit.

Six inline `#[cfg(test)]` tests added (all green):
- `test_shooting_vectors_parallel_equiv` — bit-identical exact equality vs inline sequential reference at N=51 (PERF-04-A)
- `test_augmented_srsfs_parallel_equiv` — bit-identical exact equality at N=51 (PERF-04-B)
- `test_scores_threshold` — both N=10 (sequential branch) and N=51 (parallel branch) match `u[(i,k)]*sv[k]` exactly (PERF-04-C)
- `test_vert_fpca_parallel_equiv` — `vert_fpca` at N=51 deterministic across two calls (PERF-04-D)
- `test_joint_fpca_parallel_equiv` — `joint_fpca` at N=51 deterministic across two calls with fixed `balance_c=1.0` (PERF-04-E)
- All 20 elastic_fpca tests pass with `--features linalg` (parallel OFF) (PERF-04-F)

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test ... --features linalg,parallel -- elastic_fpca` | 20/20 passed |
| `cargo test ... --features linalg -- elastic_fpca` | 20/20 passed |
| `cargo clippy --all-targets ... --features linalg,parallel -- -D warnings` | clean |
| `cargo clippy --all-targets ... --features linalg -- -D warnings` | clean |
| `grep iter_maybe_parallel` in shooting_vectors_from_psis | confirmed (line 715) |
| `grep iter_maybe_parallel` in build_augmented_srsfs | confirmed (line 738) |
| `grep iter_maybe_parallel` in svd_scores_and_eigenvalues | confirmed (line 800) |
| `const SCORES_PARALLEL_THRESHOLD: usize = 50` | confirmed (line 28) |

## Commits

| Hash | Description |
|------|-------------|
| 9d5618d | feat(15-01): parallelize shooting_vectors_from_psis via iter_maybe_parallel! (PERF-04-A) |
| 5e413f9a | feat(15-01): parallelize build_augmented_srsfs via iter_maybe_parallel! (PERF-04-B) |
| 849be577 | feat(15-01): guard svd_scores_and_eigenvalues with N>=50 threshold (PERF-04-C) |
| 57ea8556 | test(15-01): add vert_fpca and joint_fpca end-to-end equivalence tests (PERF-04-D/E/F) |

## Deviations from Plan

None — plan executed exactly as written.

The PLAN_CHECKER_CLARIFICATION was honored: `SCORES_PARALLEL_THRESHOLD` is checked as a single outer-if at function level (not inside the per-k loop), matching the cleaner architectural intent.

## Known Stubs

None. All loops are fully parallelized with real implementations; no placeholders or deferred wiring.

## Threat Flags

None. This change is a pure internal compute refactor of three `pub(crate)`/private functions with no I/O, no new public surface, and no change to input-validation logic. Collect-then-assign avoids any data race.

## Self-Check: PASSED

- [x] `fdars-core/src/elastic_fpca.rs` modified and committed
- [x] `9d5618d`, `5e413f9a`, `849be577`, `57ea8556` all present in git log
- [x] All six named PERF-04 tests present and green under both feature configurations
- [x] `SCORES_PARALLEL_THRESHOLD` constant exists at line 28
- [x] Three functions confirmed using `iter_maybe_parallel!` (lines 715, 738, 800)
