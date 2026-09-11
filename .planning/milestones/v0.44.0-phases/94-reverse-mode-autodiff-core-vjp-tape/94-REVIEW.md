---
phase: 94-reverse-mode-autodiff-core-vjp-tape
reviewed: 2026-09-11T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/autodiff/reverse.rs
  - fdars-core/src/autodiff/mod.rs
  - fdars-core/src/autodiff/forward.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: clean
---

# Phase 94: Code Review Report

**Reviewed:** 2026-09-11
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 94 delivers a hand-written Wengert-list reverse-mode autodiff substrate (`Var`, `vjp`, thread-local `TAPE`) alongside the existing forward-mode `Dual`. The implementation is functionally correct: all chain-rule partials are mathematically sound, the double-clear tape lifecycle is panic-safe, the sentinel-constant design correctly prevents adjoint leakage from constants, index bounds in the backward sweep are safe, and the Copy-based self-reference pattern accumulates gradients correctly when the same input is used multiple times.

No critical issues were found. Three warnings address observable API surface problems (an over-exposed internal module structure, an unnecessarily restrictive closure bound, and an asymmetric crate-root re-export omission). Two informational items cover a missing direct test for one Div branch and a doc-level asymmetry with `Dual`.

Key correctness verifications performed (all pass):
- All 13 chain-rule partials (Add, Sub, Mul, Div quotient + two RHS-const branches, Neg, sqrt, exp, ln, sin, cos, powf, abs, signum) traced against mathematical definitions and cross-checked against the forward-mode `Dual` impls.
- Asymmetric sentinel cases (const op var, var op const) in Mul, Add, Sub, Div traced — all correct; the general-branch fallthrough correctly handles `SENTINEL` for unused parent slots.
- Backward-pass index bounds: `adjoints` has length `n = tape.len()`; all leaf node indices are `< n`; SENTINEL check prevents `adjoints[usize::MAX]` access.
- Double-clear lifecycle: pre-forward clear handles panics from prior calls; post-backward clear prevents leakage. The empty-input fast path also clears.
- Copy semantics with repeated input: `x[0] + x[0]` generates `push_binary(0, 1.0, 0, 1.0)`, producing adjoint `2.0` through two accumulations into the same leaf — correct.
- Constant-folding short-circuits in each binary op prevent tape bloat on constant-only subexpressions.

---

## Warnings

### WR-01: `pub mod forward` and `pub mod reverse` expose internal module paths

**File:** `fdars-core/src/autodiff/mod.rs:87-88`

**Issue:** Both submodules are declared `pub mod`, making the full internal paths `fdars_core::autodiff::forward::Dual` and `fdars_core::autodiff::reverse::Var` accessible from external code. This leaks the implementation detail that the module is split into two files, creating paths that could be used by downstream callers that would then break if the module structure is ever reorganized. Every other multi-file module in this codebase (e.g. `alignment/`, `classification/`, `depth/`) keeps its submodules private and re-exports items explicitly at the parent level.

**Fix:** Change both declarations to private and rely solely on the existing `pub use` re-exports for the public API:

```rust
// mod.rs — change:
pub mod forward;   // was: pub mod
pub mod reverse;   // was: pub mod

// to:
mod forward;
mod reverse;

// The pub use lines below are unchanged and provide the only public paths:
pub use forward::{diff, directional_derivative, grad, jacobian, Dual};
pub use reverse::{vjp, Var};
```

This is a non-breaking change for users following the documented API (`autodiff::Dual`, `autodiff::Var`, `autodiff::vjp`). It is breaking only for callers who accessed the undocumented path `autodiff::forward::Dual` or `autodiff::reverse::Var`.

---

### WR-02: `vjp` uses `Fn` bound where `FnOnce` would suffice

**File:** `fdars-core/src/autodiff/reverse.rs:496`

**Issue:** `vjp` calls the closure exactly once (one forward pass). The `Fn` bound requires the closure to be callable multiple times, which prevents users from passing closures that own non-Clone captured values (e.g., a `Vec` moved into the closure). `grad` legitimately requires `Fn` because it calls the closure `m` times — one per input. For `vjp`, `Fn` is an unnecessary restriction.

```rust
// Current (over-restrictive):
pub fn vjp<F: Fn(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)

// Correct (accepts FnOnce = FnMut = Fn):
pub fn vjp<F: FnOnce(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)
```

**Fix:** Change the bound from `Fn` to `FnOnce`. Since `FnOnce` is a supertrait of `Fn`, all existing call sites that pass `Fn` closures continue to work. The change only affects callers who want to pass a FnOnce closure and are currently rejected.

Note: If a future `vjp` API variant calls the closure more than once (e.g., for debugging or retry), the bound would need revisiting. At present, `FnOnce` is the correct minimal bound.

---

### WR-03: `vjp` and `Var` absent from crate-root re-exports in `lib.rs`

**File:** `fdars-core/src/lib.rs:580`

**Issue:** `lib.rs` re-exports the forward-mode autodiff items at the crate root:

```rust
pub use autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};
```

The reverse-mode counterparts `vjp` and `Var` are not included. This creates an asymmetry: `use fdars_core::Dual` and `use fdars_core::grad` work, but `use fdars_core::Var` and `use fdars_core::vjp` do not. Users who follow the forward-mode usage patterns by analogy will get a confusing compilation error.

The prelude (`fdars_core::prelude::*`) correctly includes `Var` and `vjp`, so wildcard users are unaffected. The issue is specifically the qualified single-item import path.

**Fix:**

```rust
// lib.rs:580 — extend to:
pub use autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};
```

---

## Info

### IN-01: Missing direct test for the `constant / variable` Div branch

**File:** `fdars-core/src/autodiff/reverse.rs:218-241`

**Issue:** The `Div` implementation handles three cases: both-sentinel (returns constant), RHS-sentinel (unary push), and the general quotient-rule branch (binary push). When `self.node == SENTINEL` (constant numerator) and `rhs.node != SENTINEL` (variable denominator), the code falls through to the general branch, which correctly produces `push_binary(SENTINEL, 1/rhs.value, rhs.node, -self.value/rhs.value^2)`. The partial for `rhs` — `d(c/v)/dv = -c/v^2` — is mathematically correct and the SENTINEL slot is safely skipped in the backward loop.

There is a test for `x / constant` (`var_div_rhs_const_known_answer`) but no direct test for `constant / x`. The composed FD test exercises mixed constant-variable arithmetic but not this specific Div branch in isolation.

**Fix (optional):** Add a focused test:

```rust
#[test]
fn var_div_lhs_const_known_answer() {
    // f(x) = 12.0 / x, df/dx = -12/x^2. At x = 4: f = 3, df/dx = -0.75.
    let (value, grad) = vjp(|x| Scalar::from_f64(12.0) / x[0], &[4.0]);
    assert!((value - 3.0).abs() < TOL, "primal {} != 3.0", value);
    assert!((grad[0] + 0.75).abs() < TOL, "grad[0] {} != -0.75", grad[0]);
}
```

---

### IN-02: `Var` lacks a public value accessor, creating asymmetry with `Dual`

**File:** `fdars-core/src/autodiff/reverse.rs:113-119`

**Issue:** `Dual` exposes `pub value: f64`, `pub tangent: f64`, and a `pub fn extract(self) -> (f64, f64)` method. `Var` has `pub(crate) value` and `pub(crate) node` with no public accessor. The asymmetry is intentional (the tape index is an internal implementation detail that has no meaning outside `vjp`), but it means users working with both `Dual` and `Var` find no equivalent to `Dual::extract` for reading the primal value mid-computation.

In practice this matters little, because the intended use of `Var` is always inside a `vjp` closure, and the primal value is returned by `vjp` itself. However, it would be worth a brief doc note on `Var` explaining why there is no `extract()` and pointing users to the `vjp` return value.

**Fix:** Add a doc note to the `Var` struct:

```rust
/// A reverse-mode scalar: a primal value paired with a tape node index.
///
/// ...existing docs...
///
/// Unlike [`super::Dual`], `Var` has no `extract()` method: the primal value
/// and all gradients are returned by [`vjp`] after the backward pass completes.
/// Use `Var` values only inside a `vjp` closure; do not store or inspect them
/// outside of one.
```

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
