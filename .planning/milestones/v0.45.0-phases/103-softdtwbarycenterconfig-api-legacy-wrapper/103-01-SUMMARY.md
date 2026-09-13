---
phase: 103-softdtwbarycenterconfig-api-legacy-wrapper
plan: "01"
subsystem: metric/soft_dtw
tags: [soft-dtw, barycenter, config-api, non-breaking, sdtw-03]
status: complete

dependency_graph:
  requires: [102-01]
  provides: [SoftDtwBarycenterConfig, soft_dtw_barycenter_with_config]
  affects: [metric/soft_dtw.rs, metric/mod.rs, lib.rs, prelude.rs]

tech_stack:
  added: []
  patterns:
    - "Sealed config struct (#[non_exhaustive] + manual Default) — SEAL-02 convention"
    - "Manual impl Default returning non-zero numeric defaults (gamma=1.0, n_restarts=10)"
    - "Three-site additive re-export chain: metric/mod.rs → lib.rs → prelude.rs"
    - "Running doctest verifying config-path API compiles and runs end-to-end"

key_files:
  created: []
  modified:
    - fdars-core/src/metric/soft_dtw.rs
    - fdars-core/src/metric/mod.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

decisions:
  - "Manual impl Default (not #[derive(Default)]) for SoftDtwBarycenterConfig — derived Default gives gamma=0.0 which is rejected by the optimizer guard"
  - "soft_dtw_barycenter_multirestart made pub(crate) instead of private; all three #[allow(dead_code)] attributes removed since the function is now reachable"
  - "Reachability tests use crate:: paths (not fdars_core:: externally) since tests live inside the crate; this is compile-time proof of the re-export chain"
  - "Tasks 1-3 committed atomically because the doctest on soft_dtw_barycenter_with_config requires the crate-root re-export to be in place before cargo test --doc can resolve fdars_core::SoftDtwBarycenterConfig"

metrics:
  duration_seconds: 31217
  completed_date: "2026-09-13"
  tasks_completed: 3
  tasks_total: 3
  commits: 1
  files_changed: 4
  tokens: 4127

actuals:
  tokens: 4127
  tasks: 3
  commits: 1

plan_head_before: c8c41005e29253cb9fbac91cf71e255d1609aa37
---

# Phase 103 Plan 01: SoftDtwBarycenterConfig API & Legacy Wrapper Summary

**One-liner:** Sealed `SoftDtwBarycenterConfig` struct + `soft_dtw_barycenter_with_config` multi-restart entry point with manual Default (gamma=1.0/n_restarts=10), parity-tested legacy wrapper, and additive re-exports at crate root + prelude (SDTW-03).

## What Was Built

### SoftDtwBarycenterConfig (fdars-core/src/metric/soft_dtw.rs)

- `pub struct SoftDtwBarycenterConfig` with fields `gamma: f64`, `max_iter: usize`, `tol: f64`, `n_restarts: usize`, `seed: u64`
- `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` (SEAL-02 convention)
- Manual `impl Default` returning `gamma=1.0`, `max_iter=100`, `tol=1e-6`, `n_restarts=10`, `seed=0`
- Each field has a doc comment naming its default value

### soft_dtw_barycenter_with_config (fdars-core/src/metric/soft_dtw.rs)

- `#[must_use] pub fn soft_dtw_barycenter_with_config(data: &FdMatrix, config: &SoftDtwBarycenterConfig) -> SoftDtwBarycenterResult`
- Delegates to `soft_dtw_barycenter_multirestart(data, config.gamma, config.max_iter, config.tol, config.n_restarts, config.seed)`
- Includes a running doctest that constructs a 3×10 FdMatrix, calls the function, and asserts `result.barycenter.len() == m`

### PartialEq on SoftDtwBarycenterResult

- `#[derive(Debug, Clone)]` extended to `#[derive(Debug, Clone, PartialEq)]`
- Resolves deferred Phase-102 IN-02

### Removed dead_code annotations

- `#[allow(dead_code)]` removed from `per_col_std`, `perturbed_init`, and `soft_dtw_barycenter_multirestart`
- `soft_dtw_barycenter_multirestart` promoted to `pub(crate)` so the config entry point can delegate

### Additive re-exports (Tasks 2+3)

- `metric/mod.rs`: Added `soft_dtw_barycenter_with_config` and `SoftDtwBarycenterConfig` to existing `pub use soft_dtw::{ ... }` block
- `lib.rs`: Added the same two items to the crate-root `pub use metric::{ ... }` block
- `prelude.rs`: Added the same two items to the `pub use crate::metric::{ ... }` block

### Test coverage (barycenter_config_tests module)

| Test | What it verifies |
|------|------------------|
| `config_default_values` | `SoftDtwBarycenterConfig::default()` returns exact documented defaults |
| `config_path_runs` | Config path produces finite barycenter of correct length |
| `legacy_equals_config_single_restart` | Bit-identical match between legacy 4-arg fn and config at n_restarts=1 |
| `crate_root_reexport_reachable` | Items reachable via `crate::SoftDtwBarycenterConfig` + `crate::soft_dtw_barycenter_with_config` |
| `prelude_reexport_reachable` | Items reachable via `crate::prelude::*` |

## Verification Results

| Gate | Result |
|------|--------|
| `cargo test -p fdars-core --features linalg,parallel soft_dtw` | 33 tests passed |
| `cargo test -p fdars-core --doc soft_dtw` | 2 doctests passed |
| `cargo test -p fdars-core --features linalg,parallel` (full suite) | 2939 tests passed |
| `cargo build -p fdars-core --examples --features linalg,parallel` | All 28 examples compile |
| `cargo check --target wasm32-unknown-unknown` | Passed |
| `cargo check -p fdars-core` (default features) | Passed |
| `CARGO_TARGET_DIR=/tmp/fdars-serde-target cargo build --features serde` | Passed |
| `cargo fmt -- --check` | Clean |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | Clean |
| `test_soft_dtw_barycenter_vs_tslearn` regression test | Passed |

## Commits

| Hash | Description |
|------|-------------|
| 477487e2 | feat(103-01): SoftDtwBarycenterConfig API + legacy wrapper (SDTW-03) |

## Deviations from Plan

### Execution note: Tasks 1-3 committed atomically

Tasks 1, 2, and 3 were implemented and committed in a single atomic commit rather than three separate commits. The reason: the running doctest on `soft_dtw_barycenter_with_config` references `fdars_core::SoftDtwBarycenterConfig` via the crate-root re-export path (`use fdars_core::{SoftDtwBarycenterConfig, ...}`). The doctest could not pass under `cargo test --doc` until the crate-root re-export (Task 2) was in place. Implementing Task 1 alone and committing would have left the doctest failing, violating the tracer gate requirement. Combining all three tasks into one commit preserves correctness.

No API changes, no behavior changes, no additional dependencies introduced.

## Known Stubs

None. All deliverables are fully wired to production code.

## Threat Flags

None. No new network endpoints, auth paths, file access patterns, or schema changes introduced. Changes are purely additive public API surface within an existing module.

## Self-Check

- [x] fdars-core/src/metric/soft_dtw.rs exists and contains SoftDtwBarycenterConfig, soft_dtw_barycenter_with_config
- [x] fdars-core/src/metric/mod.rs re-exports both new items
- [x] fdars-core/src/lib.rs re-exports both new items at crate root
- [x] fdars-core/src/prelude.rs re-exports both new items in prelude
- [x] Commit 477487e2 exists and contains all changes
