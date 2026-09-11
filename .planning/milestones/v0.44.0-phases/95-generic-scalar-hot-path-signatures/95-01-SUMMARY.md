---
phase: 95-generic-scalar-hot-path-signatures
plan: "01"
subsystem: autodiff
tags: [autodiff, generics, scalar, l2_distance, dual, var, tdd]

requires:
  - phase: 94-reverse-mode-autodiff-core
    provides: Scalar trait, Dual (forward-mode), Var+vjp (reverse-mode) — all three needed for the GEN-01 Dual/Var flow tests

provides:
  - "l2_distance<T: Scalar> in helpers.rs — in-place generic over scalar type; existing f64 call sites unchanged (T=f64 inferred)"
  - "Inline parity test (bit-identical assert_eq! at T=f64)"
  - "Forward-mode Dual FD spot-check test (tangent within 1e-5 * max(|fd|, 1e-10))"
  - "Reverse-mode Var vjp spot-check test (all-coordinate grad within same tolerance)"

affects: [96-differentiable-basis, 97-differentiable-regression, 98-differentiable-depth, 99-end-to-end-autodiff]

actuals:
  tokens: 1485
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "In-place generic hot-path: rewrite shared kernel fn f<T: Scalar>(...) rather than adding _generic companion"
    - "f64/generic boundary: curve data → T; weights/grids/params stay f64; lift once via T::from_f64(f64_sub_expr)"
    - "T::zero() accumulator + explicit loop instead of .sum() (Scalar does not include Sum)"
    - "Scalar::sqrt resolves to f64::sqrt at T=f64 (bit-identical parity guaranteed)"

key-files:
  created: []
  modified:
    - fdars-core/src/helpers.rs

key-decisions:
  - "Removed invalid `= f64` default on free function — Rust rejects invalid_type_param_default on fns (future_incompatible error); T=f64 inference at all existing call sites is unaffected since T is inferred from &[T] arguments"
  - "use crate::autodiff::Scalar added at module top (not inside cfg(test)) so the generic bound is available to the public function"

patterns-established:
  - "GEN-01 in-place pattern proven: one function, one signature, zero call-site churn"
  - "f64-fold-before-lift: T::from_f64(weights[i]) lifts the scalar constant once, avoiding T*f64 type mismatch"

requirements-completed: [GEN-01]

coverage:
  - id: D1
    description: "l2_distance<T: Scalar> signature in helpers.rs; existing f64 callers (l2_distance_matrix) compile unchanged"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_l2_distance_parity"
        status: pass
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_l2_distance_identical"
        status: pass
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_l2_distance_different"
        status: pass
    human_judgment: false
  - id: D2
    description: "Forward-mode Dual gradient flows through l2_distance (tangent matches central FD within 1e-5 relative)"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_l2_distance_dual"
        status: pass
    human_judgment: false
  - id: D3
    description: "Reverse-mode Var vjp gradient flows through l2_distance (all-coordinate grad matches central FD within 1e-5 relative)"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_l2_distance_var"
        status: pass
    human_judgment: false
  - id: D4
    description: "Crate compiles clean with cargo clippy --all-targets --features linalg,parallel -D warnings; distance.rs untouched"
    requirement: GEN-01
    verification:
      - kind: integration
        ref: "cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-11
status: complete
---

# Phase 95 Plan 01: Tracer l2_distance Summary

**l2_distance generalized to `<T: Scalar>` in helpers.rs in-place with bit-identical f64 parity, forward-mode Dual tangent, and reverse-mode Var vjp — zero call-site churn across the crate**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-11T06:30:00Z
- **Completed:** 2026-09-11T06:42:00Z
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- Rewrote `l2_distance` in place to `pub fn l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T` — body uses `T::zero()` accumulator, `T::from_f64(weights[i])` single lift, `dist_sq.sqrt()` via `Scalar::sqrt`
- Added `use crate::autodiff::Scalar;` at module top of helpers.rs; no other file touched
- Three inline tests pass: bit-identical parity (`assert_eq!`), Dual tangent FD check (within 1e-5 relative), Var vjp all-coordinate FD check
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean; `l2_distance_matrix` (distance.rs:70) compiles unchanged with `T=f64` inferred; `git diff --name-only HEAD` shows only helpers.rs

## Task Commits

1. **Task 1+2+3: Generalize + test + fmt + commit** - `11eead60` (feat)

## Files Created/Modified

- `fdars-core/src/helpers.rs` — `l2_distance` generalized in place + `use crate::autodiff::Scalar` import + 3 inline tests (parity, Dual, Var)

## Decisions Made

- **Removed `= f64` default from function signature:** Rust rejects `invalid_type_param_default` on free functions (`future_incompatible` hard error on rustc 1.97). The plan noted the default as "cosmetic/documentary"; removing it does not affect any call site because `T` is inferred from the `&[T]` arguments. The doc comment was updated to reflect that inference (not a default) is the mechanism.
- **`use crate::autodiff::Scalar` at module top (not cfg(test)):** Required so the generic bound compiles for the public function. This is a 1-line addition matching the pattern from `regression.rs` and `metric/soft_dtw.rs`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed invalid `= f64` default type parameter on free function**

- **Found during:** Task 1 (first GREEN compile attempt)
- **Issue:** `pub fn l2_distance<T: Scalar = f64>(...)` triggers `error: defaults for generic parameters are not allowed here` — rustc 1.97 denies this with `#[deny(future_incompatible)]`. The plan stated the `= f64` default was "cosmetic but intentional" per ROADMAP wording; however Rust only allows default type params on structs/enums/traits, not free functions.
- **Fix:** Removed `= f64` from the signature. Type inference from `&[T]` arguments means all existing `f64` call sites continue to resolve `T = f64` without any turbofish; the parity test `let got: f64 = l2_distance(&c1, &c2, &w);` still compiles via return-type annotation propagation.
- **Files modified:** fdars-core/src/helpers.rs
- **Verification:** All 5 l2_distance tests pass; `cargo clippy --all-targets` clean
- **Committed in:** 11eead60

---

**Total deviations:** 1 auto-fixed (Rule 1 — rustc compile error from invalid syntax)
**Impact on plan:** Minimal — cosmetic only; T=f64 inference behavior is identical. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviation above.

## Known Stubs

None — all three tests use concrete numeric values; no placeholder data or wired-but-empty outputs.

## Threat Flags

None — in-place generalization of a pure numeric kernel; no I/O, network, deserialization, or trust boundary. T-95-01 (low) accepted per plan's threat model.

## Self-Check: PASSED

- [x] `fdars-core/src/helpers.rs` modified and committed (11eead60)
- [x] `git show --stat HEAD` shows only helpers.rs
- [x] `test -z "$(git diff --name-only HEAD -- fdars-core/src/distance.rs)"` → CHURN_OK
- [x] All 5 l2_distance tests pass (parity, identical, different, dual, var)
- [x] `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → Finished (clean)

## Next Phase Readiness

- GEN-01 tracer is proven: the Scalar bound, f64/generic boundary, `T::from_f64` lift, and T=f64 inference at existing call sites are all validated on `l2_distance`.
- Plan 02 can expand to `trapz`, `inner_product`, and `inner_product_l2` with confidence the pattern works.
- Plan 03 (compile-gate proof) will run the full 28-example + serde + WASM compile suite.

---
*Phase: 95-generic-scalar-hot-path-signatures*
*Completed: 2026-09-11*
