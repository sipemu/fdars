---
phase: 94-reverse-mode-autodiff-core-vjp-tape
fixed_at: 2026-09-11T00:00:00Z
review_path: .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 94: Code Review Fix Report

**Fixed at:** 2026-09-11
**Source review:** .planning/phases/94-reverse-mode-autodiff-core-vjp-tape/94-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5
- Fixed: 5
- Skipped: 0

## Fixed Issues

### WR-01: `pub mod forward` and `pub mod reverse` expose internal module paths

**Files modified:** `fdars-core/src/autodiff/mod.rs`
**Commit:** 7f1b6d3c
**Applied fix:** Changed `pub mod forward;` and `pub mod reverse;` to `mod forward;` and `mod reverse;`. The existing `pub use forward::{...}` and `pub use reverse::{...}` lines remain unchanged and provide the only public API paths. Grep confirmed no external callers used `autodiff::forward::` or `autodiff::reverse::` paths directly.

### WR-03: `vjp` and `Var` absent from crate-root re-exports in `lib.rs`

**Files modified:** `fdars-core/src/lib.rs`
**Commit:** 7f1b6d3c
**Applied fix:** Extended the crate-root re-export from `pub use autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar}` to `pub use autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var}`. Updated the comment to mention both forward-mode and reverse-mode. WR-01 and WR-03 were committed atomically (same commit) as they affect the same public-API surface.

### WR-02: `vjp` uses `Fn` bound where `FnOnce` would suffice

**Files modified:** `fdars-core/src/autodiff/reverse.rs`
**Commit:** 6cdef665
**Applied fix:** Changed `pub fn vjp<F: Fn(&[Var]) -> Var>` to `pub fn vjp<F: FnOnce(&[Var]) -> Var>`. Since `FnOnce` is a supertrait of `Fn`, all existing call sites (including all test closures) continue to compile. Callers who pass move closures that own non-Clone values are now accepted.

### IN-01: Missing direct test for the `constant / variable` Div branch

**Files modified:** `fdars-core/src/autodiff/reverse.rs`
**Commit:** 6cdef665
**Applied fix:** Added `var_div_lhs_const_known_answer` test. Uses `<Var as Scalar>::from_f64(12.0) / x[0]` at `x=4.0` and asserts primal=3.0 and grad=-0.75 (d(12/v)/dv = -12/v² = -12/16). Used the fully-qualified UFCS form `<Var as Scalar>::from_f64` to resolve the type ambiguity that a bare `Scalar::from_f64` would produce (consistent with the existing `vjp_composed_objective_matches_finite_diff` test). Test result: ok.

### IN-02: `Var` lacks a doc note explaining the absence of `extract()`

**Files modified:** `fdars-core/src/autodiff/reverse.rs`
**Commit:** 6cdef665
**Applied fix:** Added a `# No extract() method` doc section to the `Var` struct explaining that the tape node index is an internal implementation detail, the primal and gradient are returned by `vjp`, and users should only use `Var` inside a `vjp` closure.

## Skipped Issues

None — all five findings were fixed.

---

## Gate Results

All gates ran in the main checkout after applying fixes.

- **cargo fmt:** clean (no reformatting needed)
- **cargo clippy --all-targets --features linalg,parallel -- -D warnings:** clean (0 warnings, 0 errors)
- **cargo test -p fdars-core --features linalg,parallel autodiff:** 73 passed, 0 failed (includes new `var_div_lhs_const_known_answer`)
- **cargo test -p fdars-core --features linalg,parallel --doc autodiff:** 7 passed, 0 failed
- **cargo test -p fdars-core --features linalg,parallel (full):** 2900 passed, 0 failed, 0 ignored

---

_Fixed: 2026-09-11_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
