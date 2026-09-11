---
phase: 96-differentiable-basis-evaluation-inner-products
reviewed: 2026-09-11T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/basis/bspline.rs
  - fdars-core/src/basis/fourier.rs
  - fdars-core/src/basis/tests.rs
findings:
  critical: 0
  warning: 3
  info: 0
  total: 3
status: clean
---

# Phase 96: Code Review Report

**Reviewed:** 2026-09-11
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 96 generalized `bspline_basis_from_knots` / `evaluate_order_zero` / `bspline_recurrence_step` (B-spline path, in-place) and introduced `fourier_basis_eval<T: Scalar>` (additive Fourier core, with unchanged f64 wrappers). All code shipped after inline recovery following two consecutive agent connection-drops. The core numerical code is sound: operation order is preserved bit-for-bit, f64/T boundaries are correctly drawn, the repeated-knot guard stays an f64 comparison, and span-search uses value-only PartialOrd as intended. All 2916 lib tests plus 209 doctests pass.

Three warnings follow. None are regressions from the pre-phase-96 state, but all are introduced or exposed by this phase's work and should be addressed before Phase 97 downstream code depends on them.

## Warnings

### WR-01: Tautological f64 parity tests — neither actually verifies backward-compatibility with pre-phase-96 numerics

**File:** `fdars-core/src/basis/tests.rs:623-637` (B-spline) and `fdars-core/src/basis/tests.rs:771-785` (Fourier)

**Issue:** Both "f64 parity" tests compare the new generic function to itself, not to the original pre-generalization implementation.

For B-spline: `bspline_basis_from_knots` is now the generic implementation. `test_bspline_basis_from_knots_f64_parity` calls `bspline_basis_from_knots(&t, &knots, order)` (inferred `T=f64`) and `bspline_basis_from_knots::<f64>(&t, &knots, order)` (explicit turbofish). Both call identical code — the test cannot fail under any implementation of the generic body, even a broken one, because both arms go through the same path.

For Fourier: `fourier_basis_with_period` now internally calls `fourier_basis_eval`. `test_fourier_basis_eval_f64_parity` compares `fourier_basis_with_period` (which calls `fourier_basis_eval`) against `fourier_basis_eval` directly. Again, two calls to the same function with the same arguments.

The actual backward-compatibility evidence comes from the pre-existing partition-of-unity, DC-column, and sin/cos-range tests continuing to pass — not from these named "parity" tests. The comments in the tests ("same as the pre-change call", "bit-identical to fourier_basis_with_period") are actively misleading to future readers.

If a bug were introduced in the generic body (e.g. wrong operation order, wrong constant lift), the FD tests would catch it, but the parity tests would not. The risk is that readers trust "parity test passes" as evidence it isn't needed and skip the FD tests.

**Fix:** Replace the tautological `assert_eq!` with a golden-value assertion against a known correct output, or remove the parity tests and extend their comments to explicitly state that the pre-existing tests (`test_bspline_basis_partition_of_unity`, `test_fourier_basis_constant_first_column`, etc.) serve as the backward-compatibility guard:

```rust
// Before (tautological):
let reference: Vec<f64> = bspline_basis_from_knots(&t, &knots, order);  // same code as below
let generic: Vec<f64> = bspline_basis_from_knots::<f64>(&t, &knots, order);
assert_eq!(reference, generic, "...");

// After (golden value, or extend the partition-of-unity test instead):
// Inline a known-correct reference computed from the original code before generalization:
let expected: Vec<f64> = vec![
    // pre-computed from the original f64 bspline_basis_from_knots body
    // (or derive from bspline_basis which uses evaluate_order_zero internally via a separate path)
];
let generic: Vec<f64> = bspline_basis_from_knots::<f64>(&t, &knots, order);
assert_eq!(generic, expected, "bspline_basis_from_knots::<f64> must be bit-identical to pre-phase-96 output");
```

Alternatively: add a cross-check to `test_bspline_basis_partition_of_unity` that calls the explicit `bspline_basis_from_knots::<f64>` path and verifies partition of unity, so that function's behavior is directly tested against a non-trivial property.

---

### WR-02: B-spline FD tests use a near-zero absolute tolerance floor (1e-16) — inconsistent with Fourier tests and fragile at zero-gradient points

**File:** `fdars-core/src/basis/tests.rs:699` (`test_bspline_inner_product_objective_dual`) and `tests.rs:757` (`test_bspline_inner_product_objective_var`)

**Issue:** The B-spline FD tolerance is `1e-6 * fd.abs().max(1e-10)`, giving an absolute floor of `1e-6 * 1e-10 = 1e-16`. The Fourier FD tolerance is `1e-6 * (1.0 + fd.abs())`, giving an absolute floor of `1e-6`. The plan document (96-02-SUMMARY) explicitly documents the Fourier absolute-floor choice to handle grid points where the Fourier derivative is exactly zero, preventing `tol` from collapsing to machine-epsilon and causing false failures.

The same concern applies to B-splines: piecewise-polynomial basis functions have gradient exactly zero outside their support and can have numerically very small (but not exactly zero) FD values near knot boundaries due to cancellation in the central-difference formula. At such points, `tol = 1e-16` requires AD to match FD to 16 decimal places — a condition that depends on implementation details of `Dual` / `Var` arithmetic.

Currently all tests pass on this data set. The fragility manifests if the test grid is changed (e.g. Phase 97 reuses these helpers with a different `t`), or if a different knot structure places a test point closer to a knot boundary where FD undergoes larger cancellation.

**Fix:** Apply the same absolute-floor tolerance used for Fourier to B-spline FD tests:

```rust
// At tests.rs:699 (Dual) and tests.rs:757 (Var):
// Before:
let tol = 1e-6 * fd.abs().max(1e-10);
// After (matches Fourier convention, documented in 96-02-SUMMARY):
let tol = 1e-6 * (1.0 + fd.abs());
```

---

### WR-03: `fourier_basis_eval` is not re-exported at the basis module level, creating an asymmetric API surface with `bspline_basis_from_knots<T>`

**File:** `fdars-core/src/basis/mod.rs:32` and `fdars-core/src/basis/fourier.rs:58`

**Issue:** `bspline_basis_from_knots<T: Scalar>` is re-exported at `crate::basis::bspline_basis_from_knots` (via `basis/mod.rs:29`) and further at the crate root `fdars_core::bspline_basis_from_knots` (via `lib.rs:678`). A Phase 97 or external caller can write `use fdars_core::bspline_basis_from_knots` to access the generic B-spline entry point.

`fourier_basis_eval<T: Scalar>` is only accessible via the submodule path `fdars_core::basis::fourier::fourier_basis_eval`. It is not listed in `basis/mod.rs`'s `pub use` block and is absent from `lib.rs`. The `fourier` module is `pub` so the function IS reachable, but only through the unintuitive two-level path `crate::basis::fourier::fourier_basis_eval` rather than the flat path `crate::basis::fourier_basis_eval` or `crate::fourier_basis_eval`.

Phase 97 (differentiable regression) will need to call `fourier_basis_eval` from different modules. Forcing module traversal rather than flat-API access is inconsistent with this crate's convention (flat `pub use` barrel API in `basis/mod.rs`) and makes the function harder to discover.

**Fix:** Add `fourier_basis_eval` to the `basis/mod.rs` re-export block alongside `fourier_basis` and `fourier_basis_with_period`, and optionally also at `lib.rs`:

```rust
// In fdars-core/src/basis/mod.rs — add fourier_basis_eval to existing line 32:
pub use fourier::{fourier_basis, fourier_basis_eval, fourier_basis_with_period};
```

If crate-root visibility is desired for consistency with `bspline_basis_from_knots`, also add it to `lib.rs` near the existing Fourier re-exports (around line 680).

---

## Structural Findings (fallow)

No structural pre-pass was provided for this phase.

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
