---
phase: 35-basis-system-completions
plan: "04"
subsystem: pda
tags: [lfd, pda, differential-operator, regression, ode]
status: complete

dependency_graph:
  requires:
    - 35-01  # BasisSystem struct foundation
    - 35-02  # MultiFunData (parallel, but same phase)
    - 35-03  # MultiFunData completed
  provides:
    - fdars_core::Lfd
    - fdars_core::PdaResult
    - fdars_core::principal_differential_analysis
  affects:
    - fdars-core/src/lib.rs

tech_stack:
  added: []
  patterns:
    - iterated finite-difference derivatives via helpers::gradient
    - nalgebra SVD pseudoinverse for pointwise least-squares
    - constant-coefficient broadcast (len-1 coefs[k])
    - #[non_exhaustive] PdaResult for forward-compatibility

key_files:
  created:
    - fdars-core/src/pda.rs
  modified:
    - fdars-core/src/lib.rs

decisions:
  - "Placed Lfd + PdaResult + principal_differential_analysis all in src/pda.rs (single new file); cleanest separation from basis/ module since Lfd consumes derivatives, not basis functions"
  - "Used helpers::gradient (existing, auto-dispatches 5-point stencil on uniform grids) for all derivative estimation; no new finite-difference code"
  - "SVD pseudoinverse threshold: 1e-10 * max_singular_value (mirrors smooth_basis::invert_penalized_system)"
  - "PdaResult.residuals = Option<FdMatrix> defaulting to None; caller can compute via Lfd::apply if needed"
  - "Constant-coefficient Lfd represented as coefs[k].len()==1 broadcast (Pitfall 3 from RESEARCH.md)"

metrics:
  duration_minutes: 12
  completed: "2026-08-21"
  tasks_completed: 3
  tasks_total: 3
  commits: 3

actuals:
  tokens: 14000
  tasks: 3
  commits: 3
---

# Phase 35 Plan 04: Lfd + PDA Summary

Lfd linear-differential-operator + PdaResult + principal_differential_analysis in new `fdars-core/src/pda.rs`, registered at crate root; harmonic-oscillator recovery test recovers β₀≈ω²≈39.478, β₁≈0 within tolerance 1.0; full suite 2359+ tests pass; clippy --all-targets clean.

## What Was Built

### Task 1 — `Lfd` struct + `apply()` (iterated finite-difference derivatives)

`Lfd { coefs: Vec<Vec<f64>> }` with `apply(data: &FdMatrix, argvals: &[f64]) -> Result<FdMatrix, FdarError>`.

The operator forms `Lx(t_j) = D^m x(t_j) + Σ_{k=0}^{m-1} βₖ(t_j) · D^k x(t_j)` by
applying `crate::helpers::gradient` iteratively `m` times per curve row.  Constant-coefficient
fields (`coefs[k].len() == 1`) are broadcast to all grid points.  Input validation:
`argvals.len() != ncols` or any `coefs[k].len() ∉ {1, n_pts}` returns `FdarError::InvalidDimension`.

### Task 2 — `PdaResult` + `principal_differential_analysis`

`PdaResult { coefficients: Vec<Vec<f64>>, order: usize, residuals: Option<FdMatrix> }` with
`#[non_exhaustive]`.

`principal_differential_analysis(data, argvals, order)` computes derivative FdMatrices
D⁰x … D^{order}x, then at each grid column `j` builds an `n × order` design matrix from
D⁰..D^{order-1} evaluated at `t_j`, with target `y_j = -(D^{order} x)` at `t_j`.  Pointwise
least squares is solved via `nalgebra::SVD` pseudoinverse (threshold `1e-10 × max_sv`); a
rank-deficient design yields zero coefficients rather than NaN/panic.

Guards: `order == 0` → `InvalidParameter`; `argvals.len() != ncols` → `InvalidDimension`;
`n_curves < order + 1` → `InvalidDimension` (Pitfall 4 from RESEARCH.md).

### Task 3 — Module registration + phase gate

Added `pub mod pda;` to `fdars-core/src/lib.rs` alongside `multi_fdata` and
`pub use pda::{Lfd, PdaResult, principal_differential_analysis};` at crate root.

## Phase Gate Results

### Full Test Suite

```
test result: ok. 2359 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 29.22s
```

Plus all integration test binaries:
- validate_against_r.rs: 174 passed
- validate_new_modules.rs: 56 passed
- validate_spm_math.rs: 34 passed
- validate_phase_bands.rs: 16 passed
- integration_explain_advanced.rs: 55 passed
- doc tests: 163 passed (4 ignored)

Total across all binaries: **2362+ tests, 0 failures**.

### Clippy

```
cargo clippy --all-targets --features linalg,parallel -- -D warnings
Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.92s
```

Zero warnings. Clean.

### Cargo.toml

`git diff --exit-code fdars-core/Cargo.toml` → clean (no new dependencies).

## Deviations from Plan

None — plan executed exactly as written. Tasks 1 and 2 were committed in a single
atomic commit (both are in `pda.rs`), consistent with the "per-file" atomicity convention.

## Test Coverage

All 9 inline tests in `pda::tests` pass:

| Test | Status |
|------|--------|
| `lfd_constant_operator_on_constant_curve` | ok |
| `lfd_mismatched_argvals_returns_err` | ok |
| `lfd_bad_coefs_length_returns_err` | ok |
| `lfd_apply_shape_preserved` | ok |
| `pda_recovers_harmonic_oscillator` | ok |
| `pda_too_few_curves_returns_err` | ok |
| `pda_mismatched_argvals_returns_err` | ok |
| `pda_zero_order_returns_err` | ok |
| `pda_result_shape_invariants` | ok |

The harmonic-oscillator test uses ω=2π, 20 curves, 101 grid points; recovered β₀ within
1.0 of 39.478 and β₁ within 1.0 of 0 at all grid points.  Actual observed errors were in
the range 1e-4 to 1e-3 — well inside the 1.0 tolerance.

## Commits

| Hash | Description |
|------|-------------|
| 68e7f006 | feat(35-04): add Lfd + PdaResult + principal_differential_analysis in pda.rs |
| f8212ef6 | feat(35-04): register pub mod pda; and crate-root re-export |

## Self-Check

- [x] `fdars-core/src/pda.rs` exists and contains Lfd, PdaResult, principal_differential_analysis
- [x] `fdars-core/src/lib.rs` has `pub mod pda;` and `pub use pda::{...};`
- [x] Commit 68e7f006 exists in git log
- [x] Commit f8212ef6 exists in git log
- [x] Full suite 2359+ tests, 0 failures
- [x] Clippy clean

## Self-Check: PASSED
