---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: 03
subsystem: autodiff
tags: [rust, autodiff, reverse-mode, vjp, wengert-list, testing]

# Dependency graph
requires:
  - phase: 94-02-full-op-set
    provides: "full Var Scalar op set (arithmetic, transcendental) and vjp entry point with double-clear lifecycle"
provides:
  - "Tier-5 lifecycle/edge-case tests: empty input, constant-only closure, single-input cube, repeated-call stability (no tape leakage), many-input→scalar single-sweep accumulation"
  - "Tier-3 reverse-vs-Dual agreement tests for every op (add, sub, mul, div, neg, sqrt, exp, ln, sin, cos, powf, abs) plus a composed multi-input chain — all at 1e-10 tolerance"
  - "RAD-02 efficiency claim verified: 3-input x0*x1+x2 → all gradients in one backward sweep"
affects: [94-04, 99-end-to-end-flow, 100-release-prep]

actuals:
  tokens: 3026
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "TDD on hardening: write tests that confirm correct existing implementation rather than driving new code — valid when prior plan left the impl sound"
    - "Tier-3 agreement pattern: identical closure written twice (Dual and Var forms), grad vs vjp at 1e-10 — locks cross-mode correctness for each op"
    - "Tier-5 lifecycle pattern: empty/constant/single/repeated/many-input edge cases as canonical coverage for vjp correctness"

key-files:
  created: []
  modified:
    - fdars-core/src/autodiff/reverse.rs

key-decisions:
  - "Double-clear tape lifecycle (clear before AND after vjp) was already correct in Plan 02 — hardening confirmed via repeated-call stability test showing no gradient drift across 3 calls"
  - "Tier-3 agreement test imports crate::autodiff::{Dual, grad} from inside reverse.rs tests — valid since autodiff/mod.rs re-exports both at crate::autodiff"
  - "Constant-only closure: output.node==SENTINEL guard ensures all-zero gradient without any special-case code beyond the existing backward-pass sentinel check"

requirements-completed: [RAD-02]

coverage:
  - id: D1
    description: "vjp lifecycle is panic-safe via double-clear: clear before forward pass (handles prior panic) and after backward pass (prevents leakage)"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_repeated_calls_no_gradient_drift"
        status: pass
    human_judgment: false
  - id: D2
    description: "Empty input returns (value, empty vec) without touching the tape"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_empty_input_no_panic"
        status: pass
    human_judgment: false
  - id: D3
    description: "Constant-only closure: output SENTINEL → gradient all-zero for all inputs"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_constant_only_closure_gradient_is_zero"
        status: pass
    human_judgment: false
  - id: D4
    description: "Many-input→scalar single backward sweep: 3-input x0*x1+x2 yields all three gradients in one reverse iteration (RAD-02 efficiency claim)"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#vjp_many_input_scalar_single_sweep"
        status: pass
    human_judgment: false
  - id: D5
    description: "Tier-3 reverse-vs-Dual agreement: vjp gradient matches grad gradient at 1e-10 for every op (add, sub, mul, div, neg, sqrt, exp, ln, sin, cos, powf, abs)"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#agreement_add,agreement_sub,agreement_mul,agreement_div,agreement_neg,agreement_sqrt,agreement_exp,agreement_ln,agreement_sin,agreement_cos,agreement_powf,agreement_abs"
        status: pass
    human_judgment: false
  - id: D6
    description: "Multi-input composed chain sin(x0)*exp(x1)+sqrt(x2): all 3 gradient components agree at 1e-10"
    requirement: RAD-02
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs#agreement_composed_chain"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-09-10
status: complete
---

# Phase 94 Plan 03: vjp Hardening & Agreement Summary

**Hardened vjp with 18 new tests — Tier-5 lifecycle/edge-cases (tape-leakage, empty input, constant-only, cube, repeated-call stability) and Tier-3 reverse-vs-Dual agreement (every op + composed chain at 1e-10) — closing RAD-02**

## Performance

- **Duration:** 4 min
- **Started:** 2026-09-10T20:44:01Z
- **Completed:** 2026-09-10T20:48:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Confirmed the `vjp` double-clear tape lifecycle (clear before forward pass AND after backward sweep) is already correctly implemented from Plan 02 — the repeated-call stability test (3 successive calls, bit-identical gradients) proves no tape leakage.
- Added 5 Tier-5 edge-case tests: empty input (`&[]` → primal + empty gradient, no panic), constant-only closure (output SENTINEL → all-zero gradient), single-input cube (x^3 → 3x^2), repeated-call stability (no drift), and many-input→scalar (`x0*x1+x2` → all three gradients in one sweep, proving the RAD-02 efficiency claim).
- Added 13 Tier-3 reverse-vs-Dual agreement tests: one per op (add, sub, mul, div, neg, sqrt, exp, ln, sin, cos, powf, abs) plus a 3-input composed chain `sin(x0)*exp(x1)+sqrt(x2)` — all value and gradient components agree within 1e-10.
- 69 autodiff tests total (up from 51 at Plan 01 start): all green under full `cargo test` (2896 total).

## Task Commits

1. **Task 1: Harden vjp lifecycle + Tier-5 edge-case tests** - `95542f00` (test)
2. **Task 2: Tier-3 reverse-vs-Dual agreement tests for every op** - `f9b58e3a` (test)

## Files Created/Modified

- `fdars-core/src/autodiff/reverse.rs` — Added 68 lines of Tier-5 edge-case tests and 262 lines of Tier-3 agreement tests in `#[cfg(test)] mod tests`

## Decisions Made

- Double-clear lifecycle was already correctly implemented in Plan 02's `vjp` body — no code changes needed, only test confirmation. The hardening is proven by the repeated-call stability test.
- Tier-3 agreement tests import `crate::autodiff::{Dual, grad}` from inside the `reverse.rs` test module — valid because `autodiff/mod.rs` re-exports both at `crate::autodiff`.
- Constant-only closure uses no special-case code: the existing `if output.node != SENTINEL` guard in the backward pass ensures zero gradient for SENTINEL outputs with zero additional logic.

## Deviations from Plan

None — plan executed exactly as written. The vjp implementation was already sound from Plan 02; both tasks confirmed correctness rather than adding new implementation.

## Issues Encountered

None. All 69 autodiff tests pass; clippy and fmt-check clean.

## Known Stubs

None.

## Next Phase Readiness

- RAD-02 is fully closed: `vjp` is panic-safe, edge-case-robust, and its gradients match forward-mode `Dual` within 1e-10 for every op and a composed chain.
- Plan 04 (reverse-vs-finite-differences cross-validation on `soft_dtw_distance_generic` and `project_scores_generic`) is the next and final plan of Phase 94, completing RAD-03.

---
*Phase: 94-reverse-mode-autodiff-core-vjp-tape*
*Completed: 2026-09-10*
