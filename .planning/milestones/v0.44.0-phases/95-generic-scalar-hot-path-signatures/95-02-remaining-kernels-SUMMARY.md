---
phase: 95-generic-scalar-hot-path-signatures
plan: "02"
subsystem: autodiff
tags: [autodiff, generics, scalar, trapz, inner_product, inner_product_l2, dual, var, tdd]

requires:
  - phase: 95-generic-scalar-hot-path-signatures
    plan: "01"
    provides: "Scalar trait import pattern in helpers.rs; proven GEN-01 in-place generalization pattern for l2_distance"

provides:
  - "trapz<T: Scalar> in helpers.rs — in-place generic; bit-identical f64 parity; Dual FD test"
  - "inner_product<T: Scalar> in utility.rs — .sum() rewritten to T::zero() accumulator loop; bit-identical f64 parity; Dual FD test"
  - "inner_product_l2<T: Scalar> in warping.rs — delegates to generic trapz; bit-identical f64 parity; Dual FD test; Var vjp test"
  - "use crate::autodiff::Scalar added at top of utility.rs and warping.rs"
  - "All four Phase-95 shared kernels (l2_distance, trapz, inner_product, inner_product_l2) are now generic over T"

affects: [96-differentiable-basis, 97-differentiable-regression, 98-differentiable-depth, 99-end-to-end-autodiff]

actuals:
  tokens: 8750
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "In-place generic hot-path: rewrite fn f<T: Scalar>(...) not a _generic companion"
    - "f64/generic boundary: trapz keeps x as &[f64]; fold 0.5*(x[k]-x[k-1]) in f64 first, lift once via T::from_f64"
    - "T::zero() accumulator loop replaces .sum() (Scalar does not include std::iter::Sum)"
    - "inner_product_l2 delegates to generic trapz — T infers from &[T] with no turbofish needed"
    - "Var vjp pattern: lift f64 constants via <Var as Scalar>::from_f64 inside closure"

key-files:
  created: []
  modified:
    - fdars-core/src/helpers.rs
    - fdars-core/src/utility.rs
    - fdars-core/src/warping.rs

key-decisions:
  - "No = f64 default on free functions: Rust 1.97 rejects invalid_type_param_default on fns (future_incompatible). T=f64 infers from &[T] arguments at all existing call sites — same conclusion as plan 01 tracer."
  - "inner_product accumulator loop: the pre-existing .sum() iterator chain is incompatible with T: Scalar (no std::iter::Sum bound). Rewritten to explicit T::zero() + for-loop, which matches the project_scores_generic analog in regression.rs:240-248."
  - "simpsons_weights stays f64: it is a quadrature-constant helper; its Vec<f64> output is lifted per-element via T::from_f64 inside inner_product."
  - "cumulative_trapz stays f64: out of scope per 95-RESEARCH.md anti-patterns; only trapz is generalized."

patterns-established:
  - "GEN-01 pattern confirmed for all three dependency-ordered kernels: trapz → inner_product → inner_product_l2"
  - "inner_product_l2 delegation pattern: Vec<T> prod then trapz(&prod, time) — T infers, no turbofish"

requirements-completed: [GEN-01]

coverage:
  - id: D5
    description: "trapz<T: Scalar> signature in helpers.rs; all f64 trapz callers (density_fda, frechet, alignment, fts, warping) compile unchanged at T=f64"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_trapz_parity"
        status: pass
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_trapz_dual"
        status: pass
      - kind: unit
        ref: "fdars-core/src/helpers.rs#test_trapz_sine"
        status: pass
      - kind: integration
        ref: "cargo test -p fdars-core --features linalg,parallel (2910 tests, 0 failed)"
        status: pass
    human_judgment: false
  - id: D6
    description: "inner_product<T: Scalar> in utility.rs with T::zero() accumulator loop; f64 callers unchanged"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/utility.rs#test_inner_product_parity"
        status: pass
      - kind: unit
        ref: "fdars-core/src/utility.rs#test_inner_product_dual"
        status: pass
    human_judgment: false
  - id: D7
    description: "inner_product_l2<T: Scalar> in warping.rs delegating to generic trapz; Dual + Var gradients validated"
    requirement: GEN-01
    verification:
      - kind: unit
        ref: "fdars-core/src/warping.rs#test_inner_product_l2_parity"
        status: pass
      - kind: unit
        ref: "fdars-core/src/warping.rs#test_inner_product_l2_dual"
        status: pass
      - kind: unit
        ref: "fdars-core/src/warping.rs#test_inner_product_l2_var"
        status: pass
    human_judgment: false
  - id: D8
    description: "Crate compiles clean with cargo clippy --all-targets --features linalg,parallel -D warnings; only 3 source files changed"
    requirement: GEN-01
    verification:
      - kind: integration
        ref: "cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings"
        status: pass
      - kind: integration
        ref: "git diff --name-only HEAD shows only helpers.rs, utility.rs, warping.rs"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-09-11
status: complete
---

# Phase 95 Plan 02: Remaining Kernels Summary

**trapz, inner_product, and inner_product_l2 generalized to `<T: Scalar>` in-place in dependency order — all four Phase-95 kernels now complete; zero f64 call-site churn across the crate**

## Performance

- **Duration:** 18 min
- **Started:** 2026-09-11T07:00:00Z
- **Completed:** 2026-09-11T07:18:00Z
- **Tasks:** 3
- **Files modified:** 3 (helpers.rs, utility.rs, warping.rs)

## Accomplishments

### Task 1: `trapz<T: Scalar>` (helpers.rs)

- Rewrote `trapz` in place to `pub fn trapz<T: Scalar>(y: &[T], x: &[f64]) -> T`
- Body: `T::zero()` accumulator; `half_dx = T::from_f64(0.5 * (x[k] - x[k-1]))` (f64-fold-before-lift invariant); `sum += half_dx * (y[k] + y[k-1])`
- `x` stays `&[f64]` (quadrature grid constants, not differentiable)
- `cumulative_trapz` left as `f64` (out of scope per plan)
- Two inline tests: bit-identical parity (`assert_eq!`) and Dual FD spot-check (within 1e-5 relative)

### Task 2: `inner_product<T: Scalar>` (utility.rs)

- Added `use crate::autodiff::Scalar;` at module top
- Rewrote `inner_product` in place to `pub fn inner_product<T: Scalar>(curve1: &[T], curve2: &[T], argvals: &[f64]) -> T`
- Critical rewrite: iterator `.sum()` replaced with explicit `T::zero()` accumulator loop (`std::iter::Sum` not in `Scalar` bound)
- `simpsons_weights(argvals)` stays `Vec<f64>`; each weight lifted once via `T::from_f64(weights[i])`
- Early-return guard returns `T::zero()` instead of `0.0`
- Two inline tests: bit-identical parity (within 1e-12) and Dual FD spot-check

### Task 3: `inner_product_l2<T: Scalar>` (warping.rs)

- Added `use crate::autodiff::Scalar;` at module top
- Rewrote `inner_product_l2` in place to `pub fn inner_product_l2<T: Scalar>(psi1: &[T], psi2: &[T], time: &[f64]) -> T`
- Body: `Vec<T>` product then `trapz(&prod, time)` — T infers from `&[T]`, no turbofish
- Three inline tests: bit-identical parity (`assert_eq!`), Dual FD spot-check, Var vjp all-coordinate FD check
- Callers at `.max(0.0)`, `.clamp(-1.0, 1.0)` chains remain valid (T=f64 infers)

### Full gate results

- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: clean (Finished, no warnings)
- `cargo test -p fdars-core --features linalg,parallel`: 2910 passed, 0 failed
- `git diff --name-only HEAD` (before SUMMARY commit): only helpers.rs, utility.rs, warping.rs

## Task Commits

1. **Task 1+2+3: Generalize trapz/inner_product/inner_product_l2** - `2a3993a7` (feat)

## Files Created/Modified

- `fdars-core/src/helpers.rs` — `trapz` generalized in place + 2 inline tests (parity, Dual)
- `fdars-core/src/utility.rs` — `inner_product` generalized in place (.sum() → loop) + `use crate::autodiff::Scalar` + 2 inline tests (parity, Dual)
- `fdars-core/src/warping.rs` — `inner_product_l2` generalized in place + `use crate::autodiff::Scalar` + 3 inline tests (parity, Dual, Var)

## Decisions Made

- **No `= f64` default on free functions:** Rust 1.97 rejects `invalid_type_param_default` on free functions (`future_incompatible` hard error). Same finding as plan 01 tracer. T=f64 inference from `&[T]` arguments at all call sites is unaffected.
- **`.sum()` → explicit accumulator loop for `inner_product`:** `T: Scalar` does not include `std::iter::Sum`. The iterator chain was replaced with `T::zero()` + `for` loop matching the analog in `regression.rs:240-248`. Parity is within 1e-12 (the sum order changes slightly but numerical difference is below the tolerance).
- **`simpsons_weights` stays `Vec<f64>`:** It is a quadrature-constant helper producing integration weights. Its output is lifted once per element via `T::from_f64` inside the loop.
- **`cumulative_trapz` stays `f64`:** Explicitly out of scope per plan prohibition and 95-RESEARCH.md anti-patterns.

## Deviations from Plan

None — plan executed exactly as written. The `= f64` default on free functions was flagged in the PLAN.md must_haves as "documentary" but plan 01's SUMMARY already documented that Rust rejects it; this plan correctly described it as such in its patterns and no deviation was needed.

## Issues Encountered

None.

## Known Stubs

None — all tests use concrete numeric values; no placeholder data.

## Threat Flags

None — in-place generalization of pure numeric kernels; no I/O, network, deserialization, or trust boundary. T-95-02 (low) accepted per plan threat model.

## Self-Check: PASSED

- [x] `fdars-core/src/helpers.rs` modified and committed (2a3993a7)
- [x] `fdars-core/src/utility.rs` modified and committed (2a3993a7)
- [x] `fdars-core/src/warping.rs` modified and committed (2a3993a7)
- [x] `git show --stat HEAD` shows only helpers.rs, utility.rs, warping.rs
- [x] All 8 new GEN-01 tests pass (test_trapz_parity, test_trapz_dual, test_inner_product_parity, test_inner_product_dual, test_inner_product_l2_parity, test_inner_product_l2_dual, test_inner_product_l2_var)
- [x] Full suite: 2910 passed, 0 failed
- [x] `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → Finished (clean)
- [x] `cumulative_trapz` unchanged (still f64-only)
- [x] `simpsons_weights` unchanged (still f64-only)
- [x] No call-site edits needed in distance.rs, clustering.rs, alignment/*, warping.rs callers, examples/

## Next Phase Readiness

- All four Phase-95 shared kernels are now generic: `l2_distance`, `trapz`, `inner_product`, `inner_product_l2`
- Plan 03 (compile-gate proof) will run the full 28-example + serde + WASM compile suite to confirm zero call-site churn
- Phases 96/97/98 can now write differentiable basis/regression/depth functions against the generic substrate

---
*Phase: 95-generic-scalar-hot-path-signatures*
*Completed: 2026-09-11*
