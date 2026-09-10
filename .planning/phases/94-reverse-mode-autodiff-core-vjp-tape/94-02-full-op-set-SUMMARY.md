---
phase: 94-reverse-mode-autodiff-core-vjp-tape
plan: "02"
subsystem: autodiff
tags: [reverse-mode, autodiff, wengert-list, tape, vjp, Var, Scalar, transcendentals, VJP, arithmetic]

# Dependency graph
requires:
  - phase: 94-01
    provides: "Var/Node/TAPE skeleton with Mul + sentinel constants + vjp + unimplemented!() op stubs"
provides:
  - "Add/Sub/Div/Neg for Var with correct VJP partials and constant-folding"
  - "AddAssign/SubAssign/MulAssign delegate to binary ops (gradient flows through assign-ops)"
  - "All 8 transcendentals (sqrt/exp/ln/sin/cos/powf/abs/signum) in impl Scalar for Var"
  - "Tier-1 known-answer tests at 1e-10 for every op"
  - "Tier-2 singular-point guard tests for sqrt(0)/ln(0)/powf(0,0.5) — non-finite adjoint, no panic"
  - "Zero unimplemented!() stubs remaining in Scalar for Var or op impls"
  - "push_unary #[allow(dead_code)] removed — now genuinely used by Neg, Div, and all transcendentals"
affects: [94-03, 94-04, 95, 96, 97, 98, 99]

# Actuals (#2632)
actuals:
  tokens: 4828
  tasks: 2
  commits: 4

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Arithmetic op constant-folding guard: if self.node == SENTINEL && rhs.node == SENTINEL → return SENTINEL constant (no tape push)"
    - "Div three-case folding: both-const → SENTINEL, rhs-const → push_unary(self.node, 1/v), general → push_binary quotient rule"
    - "Unary op sentinel guard: if self.node == SENTINEL → return SENTINEL constant (transcendentals)"
    - "Subdifferential abs: if value == 0.0 { 0.0 } else { value.signum() } — honest-zero selection matching Dual@444"
    - "signum zero-gradient via SENTINEL: piecewise-constant, always returns node=SENTINEL regardless of input on-tape"
    - "TDD RED-GREEN per task: failing tests committed first, then implementation"

key-files:
  created: []
  modified:
    - fdars-core/src/autodiff/reverse.rs

key-decisions:
  - "Div handles three constant cases explicitly: both-const → SENTINEL, rhs-const → push_unary (common x/gamma path), general → push_binary quotient rule — per 94-RESEARCH Open Question 2"
  - "signum returns SENTINEL constant (no tape node) rather than push_unary with weight 0.0 — semantically cleaner, matches Dual @459 zero-tangent convention"
  - "push_unary #[allow(dead_code)] removed once Neg, Div (RHS-const branch), and all transcendentals use it actively"

patterns-established:
  - "Arithmetic sentinel guard before push: every binary op checks both-sentinel first, returns constant — keeps tape free of pure-constant subexpressions"
  - "Unary op pattern: compute forward value, guard SENTINEL → return constant, push_unary with local partial"

requirements-completed: [RAD-01]

# Coverage metadata (#1602)
coverage:
  - id: D1
    description: "Add/Sub/Div/Neg ops for Var with correct VJP partials and three-case constant-folding"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_add_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_sub_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_div_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_div_rhs_const_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_neg_known_answer"
        status: pass
    human_judgment: false
  - id: D2
    description: "AddAssign/SubAssign/MulAssign propagate gradient via delegation to binary ops"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_add_assign_propagates_gradient"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_sub_assign_propagates_gradient"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_mul_assign_propagates_gradient"
        status: pass
    human_judgment: false
  - id: D3
    description: "All 8 transcendentals (sqrt/exp/ln/sin/cos/powf/abs/signum) in Scalar for Var with correct VJP partials; zero unimplemented!() remaining"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_sqrt_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_exp_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_ln_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_sin_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_cos_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_powf_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_abs_known_answer"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_abs_at_zero_gradient_is_zero"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_signum_gradient_is_zero"
        status: pass
    human_judgment: false
  - id: D4
    description: "Tier-2 singular-point guards: sqrt(0)/ln(0)/powf(0,0.5) produce non-finite adjoints, not panics"
    requirement: RAD-01
    verification:
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_sqrt_at_zero_adjoint_is_nonfinite"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_ln_at_zero_adjoint_is_nonfinite"
        status: pass
      - kind: unit
        ref: "fdars-core/src/autodiff/reverse.rs::tests::var_powf_at_zero_adjoint_is_nonfinite"
        status: pass
    human_judgment: false

# Metrics
duration: 4min
completed: 2026-09-10
status: complete
---

# Phase 94 Plan 02: Full Op Set Summary

**Complete `Scalar for Var` with VJP partials for all 12 ops (Add/Sub/Div/Neg/sqrt/exp/ln/sin/cos/powf/abs/signum), 21 Tier-1+Tier-2 tests green, zero `unimplemented!()` remaining**

## Performance

- **Duration:** 4 min
- **Started:** 2026-09-10T20:36:35Z
- **Completed:** 2026-09-10T20:40:30Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Replaced all Plan-01 `unimplemented!()` stubs: `Add`, `Sub`, `Div`, `Neg` with correct VJP partials and constant-folding (sentinel guards), `AddAssign`/`SubAssign`/`MulAssign` already delegated
- Implemented all 8 `Scalar` transcendentals for `Var` (sqrt/exp/ln/sin/cos/powf/abs/signum), each computing the forward value and calling `push_unary` with the correct local partial
- `Div` handles three constant cases: both-const → SENTINEL result, RHS-const → simpler unary push (the common `x/gamma` scaling path), general → full quotient-rule binary push
- `signum` returns a SENTINEL constant regardless of input: piecewise-constant function has zero gradient everywhere, no tape node needed — matches `Dual@459`
- `abs` uses subdifferential convention (0.0 at exactly 0.0) matching `Dual@444`; positive/negative branches use `f64::signum(v)`
- Singular points (sqrt/ln/powf at 0) push the mathematically correct non-finite local partial onto the tape, propagating Inf adjoint without panicking — no clamping, matching Dual behavior exactly
- Removed `#[allow(dead_code)]` from `push_unary` (now actively used by Neg, Div RHS-const branch, and all 8 transcendentals)
- All 51 autodiff tests pass (30 forward-mode Dual + 21 reverse-mode Var)

## Task Commits

TDD tasks each have two commits (RED failing test → GREEN implementation):

1. **Task 1 RED: Add failing tests for arithmetic ops** — `0818d4da` (test)
2. **Task 1 GREEN: Implement Add/Sub/Div/Neg** — `e7a52b44` (feat)
3. **Task 2 RED: Add failing tests for transcendentals + Tier-2 guards** — `f936bc63` (test)
4. **Task 2 GREEN: Complete Scalar for Var transcendentals** — `dc087a60` (feat)

## Files Created/Modified

- `fdars-core/src/autodiff/reverse.rs` — all 12 op impls completed; 21 tests added; 0 unimplemented!() remaining

## Decisions Made

- **Div three-case constant-folding** (per 94-RESEARCH Open Question 2): both-const → SENTINEL, rhs-const → `push_unary(self.node, 1/v)`, general → `push_binary` quotient rule. The rhs-const branch is the common case in `softmin3_generic` (`x / Scalar::from_f64(gamma)`).
- **signum returns SENTINEL constant** (not a zero-weight push_unary): cleaner semantics — no tape node for a piecewise-constant function, matches the `Dual` zero-tangent convention at `forward.rs:398`.
- **No clamping at singular points**: singular-point adjoints are Inf as the math dictates; callers own range checking, same contract as `Dual`.

## Deviations from Plan

None — plan executed exactly as written. The implementation matched the plan's explicit partial formulas and Dual-mirroring requirements on the first attempt.

## Issues Encountered

None. The constant-folding guard patterns and Dual mirroring specifications in 94-RESEARCH.md and 94-PATTERNS.md were exact; no debugging required.

## Known Stubs

None — all `unimplemented!()` stubs have been replaced.

## Threat Flags

None — pure numeric computation on caller-provided f64 slices; no I/O, network, deserialization, auth, or privilege boundary. Thread-local tape is process-local.

## Next Phase Readiness

- `Var: Scalar` is now complete with the full op set — the precondition for Plan 03 (validation: Var vs Dual agreement + central FD on `soft_dtw_distance_generic`/`project_scores_generic`) and Plan 04 (crate-root doctest + final prelude) is met.
- `cargo check -p fdars-core --features linalg` succeeds — `soft_dtw_distance_generic::<Var>` and `project_scores_generic::<Var>` will instantiate.
- All gates green: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, 51 autodiff tests, no new dependency.

---
*Phase: 94-reverse-mode-autodiff-core-vjp-tape*
*Completed: 2026-09-10*

## Self-Check: PASSED

- `fdars-core/src/autodiff/reverse.rs`: FOUND
- Zero `unimplemented!()` stubs: CONFIRMED (grep returned empty)
- Commits 0818d4da, e7a52b44, f936bc63, dc087a60: PRESENT in git log
- 21 reverse-mode tests: PASS
- 51 total autodiff tests: PASS
- cargo fmt --check: PASS
- cargo clippy --all-targets --features linalg,parallel -- -D warnings: PASS
- git diff Cargo.toml fdars-core/Cargo.toml: EMPTY (no new dep)
