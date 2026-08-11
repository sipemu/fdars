---
phase: 10-capability-gaps-spline-interpolation-functional-summary-stat
plan: "01"
subsystem: interpolation
tags: [spline-interpolation, b-spline, functional-data, capability-gap, feat-01]
status: complete

dependency_graph:
  requires: []
  provides:
    - fdars_core::spline_interpolate
    - fdars_core::helpers::spline_interpolate
  affects:
    - fdars-core/src/helpers.rs
    - fdars-core/src/lib.rs

tech_stack:
  added: []
  patterns:
    - "B-spline fit-then-evaluate: bspline_basis on argvals -> nalgebra SVD pseudoinverse -> bspline_basis_from_knots on query_points"
    - "Column-major basis layout: basis[ti + k*m] = B_k(argvals[ti]), mirrors pspline.rs:86-87"

key_files:
  created: []
  modified:
    - fdars-core/src/helpers.rs
    - fdars-core/src/lib.rs

decisions:
  - "Removed #[must_use] from spline_interpolate signature: Result<T, E> is already must_use (clippy::double_must_use); the plan called for must_use but clippy correctly identifies it as redundant on Result-returning functions"
  - "SVD pseudoinverse tolerance set to NUMERICAL_EPS * max(nrows, ncols) — consistent with nalgebra's own pseudo_inverse example and the existing pattern in basis/helpers.rs"
  - "nknots = m.saturating_sub(order).max(2) produces nbasis = nknots + order ≈ m: a near-interpolating system following the PLAN action and RESEARCH skeleton verbatim"

metrics:
  duration: "8 minutes"
  completed: "2026-08-10T21:02:16Z"
  tasks: 3
  commits: 3

estimate:
  tokens: 55000

actuals:
  tokens: 2908   # 11630 chars / 4 over the actual diff
  tasks: 3
  commits: 3
---

# Phase 10 Plan 01: Spline Interpolation Summary

**One-liner:** Adds `spline_interpolate` — order-k B-spline fit-then-evaluate interpolation using the existing `basis/bspline` system, resolving FEAT-01 (REPR-02) with full input validation and 5 inline tests covering exact reproduction and off-grid accuracy.

## What Was Built

A new public function `spline_interpolate` in `fdars-core/src/helpers.rs`, re-exported at the crate root in `lib.rs`:

```rust
pub fn spline_interpolate(
    data: &FdMatrix,
    argvals: &[f64],
    query_points: &[f64],
    order: usize,
) -> Result<FdMatrix, FdarError>
```

The implementation follows the fit-then-evaluate pattern from `basis/pspline.rs:163-187` without the P-spline smoothing penalty:

1. Validates all inputs (argvals length, non-empty query_points, order in [1,m), query_points within domain)
2. Builds knot vector once via `construct_bspline_knots`
3. Evaluates B-spline basis on argvals (`bspline_basis`)
4. Computes SVD pseudoinverse of the basis matrix once via `nalgebra::SVD::new`
5. For each curve: solves coefficients = pinv * y, then evaluates at query_points using `bspline_basis_from_knots`

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 (tracer) | End-to-end spline_interpolate — fit + evaluate, exact-reproduction test | eec9c070 | helpers.rs (+137 lines) |
| 2 | Off-grid accuracy + full input validation tests | b9c52076 | helpers.rs (+132 lines) |
| 3 | Re-export spline_interpolate at crate root | b4273b98 | lib.rs (+2 lines) |

## Tests Added (5 inline in `helpers::tests`)

| Test Name | What It Verifies |
|-----------|-----------------|
| `spline_interpolate_reproduces_argvals` | y=t^3 reproduced within 1e-10 when query_points == argvals (Success Criterion 1) |
| `spline_interpolate_cubic_offgrid` | cubic polynomial reproduced within 1e-10 at off-grid midpoints (Success Criterion 1) |
| `spline_interpolate_rejects_out_of_range` | query points outside [t_min, t_max] → FdarError::InvalidParameter{parameter:"query_points"} |
| `spline_interpolate_rejects_bad_order` | order==0 or order>=m → FdarError::InvalidParameter{parameter:"order"} |
| `spline_interpolate_rejects_dim_mismatch` | argvals mismatch and empty query_points → FdarError::InvalidDimension |

## Verification

- `cargo test -p fdars-core --features linalg spline_interpolate` — 5/5 tests pass
- `cargo clippy -p fdars-core --features linalg` — no warnings on new code
- `cargo test -p fdars-core --features linalg` — full 1939-test suite green
- Existing `fdata_interpolate`, `linear_interp`, `InterpolationMethod` re-exports preserved

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed redundant `#[must_use]` on `spline_interpolate`**
- **Found during:** Task 1 commit (pre-commit clippy hook)
- **Issue:** Plan called for `#[must_use]` per project convention on expensive computations, but `Result<T, E>` is already `#[must_use]` (Rust stdlib). Clippy lint `double_must_use` fires when a function explicitly adds `#[must_use]` to a `Result`-returning function.
- **Fix:** Removed the `#[must_use]` attribute from the function — the return type's own `#[must_use]` on `Result` provides the same guarantee. No behavior change.
- **Files modified:** `fdars-core/src/helpers.rs`
- **Commit:** eec9c070

**2. [Rule 1 - Bug] Applied `cargo fmt` formatting on two commit attempts**
- **Found during:** Task 1 and Task 2 commits (pre-commit fmt check)
- **Issue:** rustfmt reformatted long lines (map_err closure, basis call chains) differently from hand-written code
- **Fix:** Applied `cargo fmt` before each commit; no logic changes
- **Files modified:** `fdars-core/src/helpers.rs`

## Known Stubs

None — all code paths are fully implemented and tested.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The new function is a pure-Rust numeric computation function with no I/O surface. All STRIDE threat mitigations from the threat register are implemented (T-10-01-01: order validation, T-10-01-02: query point domain check, T-10-01-03: argvals/query_points dimension validation). T-10-01-04 (integer overflow) accepted — uses `saturating_sub`/`max`.

## Self-Check: PASSED

- FOUND: fdars-core/src/helpers.rs
- FOUND: fdars-core/src/lib.rs
- FOUND commit eec9c070 (Task 1 — spline_interpolate implementation)
- FOUND commit b9c52076 (Task 2 — validation tests)
- FOUND commit b4273b98 (Task 3 — crate root re-export)
- `pub fn spline_interpolate(` present in helpers.rs: 1 match
- `spline_interpolate` present in lib.rs re-export block: 1 match
- No `pspline_fit_1d` call inside `spline_interpolate` body
- No `unwrap()`, `panic!`, or `expect(` in function body
- All 5 spline_interpolate_* tests present and named correctly
