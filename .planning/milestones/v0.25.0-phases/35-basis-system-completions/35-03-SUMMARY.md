---
phase: 35-basis-system-completions
plan: "03"
subsystem: multi_fdata
tags: [container, multi-domain, REP-01, functional-data]
status: complete

dependency_graph:
  requires: [35-01, 35-02]
  provides: [MultiFunData, FdComponent]
  affects: [fdars-core/src/lib.rs]

tech_stack:
  added: []
  patterns:
    - Multi-domain functional data container (funData::multiFunData analogue)
    - FdComponent bundles FdMatrix + argvals for per-component domain
    - Constructor invariant validation with FdarError
    - Non-exhaustive public struct with conditional serde

key_files:
  created:
    - fdars-core/src/multi_fdata.rs
  modified:
    - fdars-core/src/lib.rs

decisions:
  - Accessors return Result<_, FdarError> on out-of-range index (not Option), matching FdarError conventions and T-35-04 threat mitigation.
  - argvals stored as Vec<f64> on FdComponent (public field); multi-domain: no cross-component argvals equality required.
  - MultiFunData is #[non_exhaustive]; FdComponent fields are public (no accessor needed for data/argvals fields directly).
  - pub mod multi_fdata placed alphabetically among other pub mod declarations in lib.rs.

metrics:
  duration_minutes: 12
  completed: "2026-08-21T08:16:05Z"
  tasks_completed: 2
  commits: 2

estimate:
  tokens: 42000
  raw_tokens: 21000

actuals:
  tokens: 9800
  tasks: 2
  commits: 2
---

# Phase 35 Plan 03: MultiFunData Multi-Domain Container Summary

Multi-domain functional data container (`MultiFunData` + `FdComponent`) with validated invariants, crate-root re-exports, and a full clippy-green suite — REP-01 SC2 satisfied.

## What Was Built

### Task 1 — MultiFunData + FdComponent in new multi_fdata.rs

Created `/home/simonm/projects/rust/fdars/fdars-core/src/multi_fdata.rs` implementing:

- **`FdComponent`** — public struct with `pub data: FdMatrix` and `pub argvals: Vec<f64>`. Derives `Debug, Clone, PartialEq` + conditional serde. Bundles one functional data block with its evaluation grid.

- **`MultiFunData`** — public struct with a private `components: Vec<FdComponent>` field. Derives `Debug, Clone, PartialEq`, marked `#[non_exhaustive]`, conditional serde.

- **`MultiFunData::new(components)`** — validates three invariants before construction:
  1. Non-empty components vec (`InvalidParameter` on violation).
  2. Equal observation count (`data.nrows()`) across all components (`InvalidDimension` on violation).
  3. Each component's `argvals.len() == data.ncols()` (`InvalidDimension` on violation).

- **Accessors** — all panic-free:
  - `n_obs() -> usize` — shared observation count.
  - `n_components() -> usize` — number of components.
  - `component(k) -> Result<&FdComponent, FdarError>` — `InvalidParameter` on out-of-range `k`.
  - `argvals(k) -> Result<&[f64], FdarError>` — `InvalidParameter` on out-of-range `k`.

- **Multi-domain feature documented** — components may live on different domains/grids; no cross-component argvals equality is required or checked.

- **15 inline tests** covering: two-component different-grid construction, single component, three components, empty-vec error, nrows mismatch, argvals-length mismatch (first and later component), component accessor (valid + OOB), argvals accessor (valid + OOB), per-component argvals preservation, no-panic guarantee on usize::MAX index, and derives.

Commit: `d587a29d`

### Task 2 — Register module + crate-root re-exports; clippy gate

Added to `fdars-core/src/lib.rs`:
- `pub mod multi_fdata;` (alongside existing pub mod declarations).
- `pub use multi_fdata::{FdComponent, MultiFunData};` at crate root.

`cargo clippy --all-targets --features linalg,parallel -- -D warnings` — **clean**.

Commit: `4d689ab8`

## Test Results

```
running 15 tests
test result: ok. 15 passed; 0 failed; 0 ignored; 0 measured
```

All 15 multi_fdata tests pass. Clippy gate clean.

## Deviations from Plan

None — plan executed exactly as written.

## Threat Model Verification

| Threat | Disposition | Status |
|--------|-------------|--------|
| T-35-04: Out-of-range index panic | mitigate — `InvalidParameter` returned | Implemented — `component(k)` and `argvals(k)` return `Err` on OOB; no panics. |
| T-35-SC: No new cargo package | accept | Confirmed — zero new dependencies; reuses existing `FdMatrix` and `FdarError`. |

## Known Stubs

None.

## Self-Check

- [x] `fdars-core/src/multi_fdata.rs` exists and non-empty.
- [x] `pub mod multi_fdata;` present in `lib.rs`.
- [x] `pub use multi_fdata::{FdComponent, MultiFunData};` present in `lib.rs`.
- [x] Commits `d587a29d` and `4d689ab8` exist.
- [x] 15 tests pass; clippy clean.

## Self-Check: PASSED
