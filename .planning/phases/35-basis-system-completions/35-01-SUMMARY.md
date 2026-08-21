---
phase: 35-basis-system-completions
plan: "01"
subsystem: basis
status: complete
tags: [basis, monomial, BasisSystem, penalty-matrix, analytic-gram, pub-crate]
depends_on: []
provides:
  - BasisSystem struct (column-major eval + penalty + metadata)
  - monomial_basis factory (analytic 2nd-derivative Gram penalty)
  - pub(crate) numeric-Gram helpers for downstream plans
affects:
  - fdars-core/src/basis/basis_system.rs
  - fdars-core/src/basis/monomial.rs
  - fdars-core/src/basis/mod.rs
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/lib.rs
tech_stack:
  added: []
  patterns:
    - "BasisSystem: non_exhaustive + serde-conditional derive, column-major eval+penalty bundle"
    - "Analytic Gram via falling-factorial formula (no quadrature for monomial)"
    - "pub(crate) fn visibility for reusable internal helpers"
key_files:
  created:
    - fdars-core/src/basis/basis_system.rs
    - fdars-core/src/basis/monomial.rs
  modified:
    - fdars-core/src/basis/mod.rs
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/lib.rs
decisions:
  - "Analytic Gram over numeric quadrature for monomial penalty — exact, no grid dependency"
  - "BasisSystem in dedicated basis_system.rs — shared across all four factories"
  - "pub(crate) promotion (Option B) over copy-paste for smooth_basis helpers"
  - "Clippy erasing_op/identity_op: rewrote test indices to use computed offsets rather than 0+j*n patterns"
metrics:
  duration_minutes: 20
  completed: "2026-08-21"
  tasks_completed: 3
  commits: 4
estimate:
  tokens: 55000
  tasks: 3
actuals:
  tokens: 9000
  tasks: 3
  commits: 4
---

# Phase 35 Plan 01: BasisSystem + monomial_basis (Tracer) Summary

**One-liner:** `BasisSystem` struct + `monomial_basis` analytic-Gram factory end-to-end, with `pub(crate)` numeric-Gram helpers promoted for downstream reuse — tracer slice for REP-01 basis-family architecture.

## What Was Built

### Task 1: BasisSystem struct + monomial_basis factory

**`fdars-core/src/basis/basis_system.rs`** — new public struct:
- `BasisSystem { eval_matrix, penalty_matrix, nbasis, n_eval, lfd_order }` — all fields `pub`
- `#[non_exhaustive]` (forward-compatible: `domain`, `basis_type` fields can be added later)
- Conditional `serde::Serialize/Deserialize` behind `feature = "serde"`, matching `PsplineFitResult`
- Column-major layout documented: element `(t_i, j)` at index `i + j * n_eval`

**`fdars-core/src/basis/monomial.rs`** — factory + analytic penalty:
- `monomial_basis(argvals, nbasis) -> Result<BasisSystem, FdarError>`
- Eval matrix: `eval_matrix[ti + j*n] = argvals[ti].powi(j as i32)` — exact `t^j`
- Analytic penalty (lfd_order=2): falling-factorial Gram `R[i,j] = c_i * c_j * (b^p - a^p) / p` with `ln(b/a)` guard for the degenerate `p≈0` case
- Error paths: `InvalidDimension` for `argvals.len() < 2`, `InvalidParameter` for `nbasis < 1`
- 9 inline unit tests: closed-form eval, P[2,2]=4.0 on [0,1], P[2,2]=8.0 on [0,2], symmetry, PSD diagonal, P[0,0]=P[1,1]=0, shape invariants, derive compatibility, error paths

**Reference values verified:**
- `t=[0,1,2], nbasis=3` → `eval_matrix = [1,1,1, 0,1,2, 0,1,4]` (column-major)
- `domain=[0,1], lfd_order=2, nbasis=3`: P[0,0]=0, P[1,1]=0, P[2,2]=4.0 (exact analytic)
- `domain=[0,2]`: P[2,2]=8.0 (verified by doctest)

### Task 2: Promote smooth_basis numeric-Gram helpers to pub(crate)

**`fdars-core/src/smooth_basis.rs`** — two-line visibility change only:
- `fn differentiate_basis_columns` → `pub(crate) fn differentiate_basis_columns`
- `fn integrate_symmetric_penalty` → `pub(crate) fn integrate_symmetric_penalty`
- Zero signature or body changes; `cargo build` clean; no public API impact
- Enables 35-02 (exponential/polygonal bases) to reuse numeric Gram pattern without copy-paste

### Task 3: Register in basis/mod.rs and crate-root lib.rs

**`fdars-core/src/basis/mod.rs`** — additive registration:
- `pub mod basis_system;` and `pub mod monomial;` added
- `pub use basis_system::BasisSystem;` and `pub use monomial::monomial_basis;` added
- All existing exports preserved in order

**`fdars-core/src/lib.rs`** — crate-root re-export:
- `monomial_basis` and `BasisSystem` added to `pub use basis::{...}` block
- Callers can use `fdars_core::monomial_basis` and `fdars_core::BasisSystem` directly

## Verification Results

| Check | Result |
|-------|--------|
| `cargo test basis::monomial` | 9/9 passed |
| `cargo test basis::` | 163/163 passed |
| `cargo build -p fdars-core --features linalg,parallel` | clean |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | clean |
| Existing bspline/fourier/constant untouched | confirmed (git diff additive only) |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Clippy erasing_op/identity_op in test indexing**
- **Found during:** Task 3 clippy gate (`--all-targets`)
- **Issue:** Test code used `bs.eval_matrix[0 + 0 * n]`, `bs.eval_matrix[1 + 0 * n]` etc. — clippy flags `0 * n` as erasing_op (always 0) and `1 * n` as identity_op
- **Fix:** Rewrote test indices to use computed offsets (`bs.n_eval`, `bs.nbasis`) and direct arithmetic (`n + 1`, `2 * n`, `1 + k`) that clippy does not flag
- **Files modified:** `fdars-core/src/basis/monomial.rs`
- **Commit:** `7464d994`

## Known Stubs

None. The monomial factory evaluates to exact closed-form values; the penalty is analytic. No placeholder data.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. All additions are pure numerical computation. T-35-01 (DoS via unchecked input) is mitigated: `argvals.len() < 2` and `nbasis < 1` both return `FdarError` at entry — no unbounded allocation.

## Self-Check: PASSED

All created files confirmed on disk; all commits confirmed in git log.

| Item | Status |
|------|--------|
| `basis_system.rs` | FOUND |
| `monomial.rs` | FOUND |
| `35-01-SUMMARY.md` | FOUND |
| `bdb2482d` (task 1) | FOUND |
| `4f450d27` (task 2) | FOUND |
| `a713f57a` (task 3) | FOUND |
| `7464d994` (clippy fix) | FOUND |
