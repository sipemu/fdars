---
phase: 94-reverse-mode-autodiff-core-vjp-tape
verified: 2026-09-11T07:45:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 94: Reverse-Mode Autodiff Core (VJP Tape) Verification Report

**Phase Goal:** An in-crate reverse-mode (vector-Jacobian-product) autodiff core exists alongside the forward-mode `Dual`, with backward-pass gradient accumulation validated against the existing differentiable subset.
**Verified:** 2026-09-11T07:45:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | A reverse-mode Wengert-list tape records ops on a `Var`/tape scalar type supporting the full forward-mode op set (±, ×, ÷, sqrt, exp, ln, sin/cos, powf, abs, comparisons) | ✓ VERIFIED | `fdars-core/src/autodiff/reverse.rs` contains `Var`, `Node`, thread-local `TAPE`, `push_binary`/`push_unary`; `impl Add/Sub/Mul/Div/Neg/AddAssign/SubAssign/MulAssign for Var` (lines 156–292); `impl Scalar for Var` with all 8 transcendentals (sqrt/exp/ln/sin/cos/powf/abs/signum, lines 298–470); value-only `PartialEq`/`PartialOrd` (lines 131–145); zero `unimplemented!()` stubs (grep confirmed empty). Tier-1 known-answer tests for each op plus Tier-2 singular-point guards: all 44 reverse-mode unit tests pass. |
| 2 | A backward pass seeds the output adjoint and accumulates input gradients, exposed through a `vjp` entry point efficient for many-input→scalar objectives (ONE backward sweep) | ✓ VERIFIED | `vjp` at `reverse.rs:505` is `#[must_use] pub fn vjp<F: FnOnce(&[Var]) -> Var>(f: F, x: &[f64]) -> (f64, Vec<f64>)`. Double-clear lifecycle (clear before + after). One forward pass + one `for i in (0..n).rev()` sweep accumulates all input gradients. `vjp_many_input_scalar_single_sweep` passes: `x0*x1+x2` at `(2,3,4)` yields gradient `[3,2,1]` in one sweep. Tier-5 edge-case tests (empty, constant-only, single-input cube, repeated-call stability, many-input) all pass. |
| 3 | Reverse-mode gradients match the forward-mode `Dual` path AND central finite differences within tolerance on elastic soft-DTW distance and FPCA scores (the existing differentiable subset) | ✓ VERIFIED | Tier-3 reverse-vs-Dual agreement tests for every op (add/sub/mul/div/neg/sqrt/exp/ln/sin/cos/powf/abs) + composed chain `sin(x0)*exp(x1)+sqrt(x2)` all pass at 1e-10. Tier-4 FD cross-checks: `vjp_soft_dtw_matches_finite_diff` (m=16, h=1e-8, tol=1e-6 per component — passes), `vjp_project_scores_matches_finite_diff` (m=24, n=40, ncomp=3, h=1e-8, tol=1e-6 — passes), `vjp_composed_objective_matches_finite_diff` (soft_dtw + λ·Σscores², h=1e-6, tol=1e-6 — passes). Both `soft_dtw_distance_generic` and `project_scores_generic` called with `Var` with ZERO call-site changes confirmed by `git diff` showing those files unmodified. |
| 4 | No new crate dependency (hand-written in-crate tape) | ✓ VERIFIED | `git diff eaeb1381^ HEAD -- Cargo.toml fdars-core/Cargo.toml` produced empty output — no dependency added. The tape uses only `std::cell::RefCell` and `std::ops::*` from the Rust standard library. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/autodiff/mod.rs` | `Scalar` trait + shared re-exports for `Dual`/`Var`/`vjp`/`grad`/`diff`/`jacobian`/`directional_derivative` | ✓ VERIFIED | 98 lines; `Scalar` trait defined here (lines 38–85); `mod forward; mod reverse;` (private submodules per WR-01 fix); `pub use forward::{...}; pub use reverse::{vjp, Var};` (line 96–97) |
| `fdars-core/src/autodiff/forward.rs` | `Dual` + forward entry points; moved from old `autodiff.rs` via `git mv` | ✓ VERIFIED | File exists; all 30 forward-mode `Dual` tests in `autodiff::forward::tests` pass (confirmed in full test run) |
| `fdars-core/src/autodiff/reverse.rs` | `Var`, `Node`, `TAPE`, `push_binary`/`push_unary`, full `Scalar for Var`, `vjp`, 72+ tests | ✓ VERIFIED | 1480 lines; all constituent elements present and fully implemented; 44 tests covering Tier-1 through Tier-5; zero stubs |
| `fdars-core/src/prelude.rs` | Re-exports `vjp` and `Var` alongside forward-mode exports | ✓ VERIFIED | Line 21: `pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};` |
| `fdars-core/src/lib.rs` (crate root) | Re-exports `vjp` and `Var` (WR-03 fix) | ✓ VERIFIED | Line 581: `pub use autodiff::{diff, directional_derivative, grad, jacobian, vjp, Dual, Scalar, Var};` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `autodiff/mod.rs` | `autodiff/forward.rs` | `mod forward;` + `pub use forward::{...}` | ✓ WIRED | Private submodule declared, all public items re-exported at `autodiff::` level |
| `autodiff/mod.rs` | `autodiff/reverse.rs` | `mod reverse;` + `pub use reverse::{vjp, Var}` | ✓ WIRED | Private submodule declared; `Var` and `vjp` publicly accessible as `autodiff::Var` and `autodiff::vjp` |
| `lib.rs` | `autodiff` | `pub mod autodiff;` (line 76, unchanged) | ✓ WIRED | Cargo resolves to `autodiff/mod.rs` directory automatically; `lib.rs:76` confirmed unchanged |
| `prelude.rs` | `autodiff::{vjp, Var}` | `pub use crate::autodiff::{..., vjp, ..., Var}` | ✓ WIRED | Both discoverable via `use fdars_core::prelude::*` |
| `reverse.rs` tests | `soft_dtw_distance_generic` | `use crate::metric::soft_dtw_distance_generic` | ✓ WIRED | Called with `Var` unchanged — Tier-4 test passes |
| `reverse.rs` tests | `project_scores_generic` | `use crate::regression::project_scores_generic` | ✓ WIRED | Called with `Var` unchanged — Tier-4 test passes |
| `Var: Scalar` supertrait bounds | `Add/Sub/Mul/Div/Neg/AddAssign/SubAssign/MulAssign for Var` | implemented on `Var` | ✓ WIRED | `impl Scalar for Var` compiles; every supertrait op is fully implemented (not `unimplemented!()`) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `vjp` | `gradient` | backward sweep over TAPE nodes, seeded from real leaf nodes | Yes — real tape accumulation, no static return | ✓ FLOWING |
| `soft_dtw_distance_generic::<Var>` | return value | primal computed by the DTW recurrence over `Var` ops | Yes — each `Var` records real ops; gradient flows through tape | ✓ FLOWING |
| `project_scores_generic::<Var>` | `Vec<Var>` scores | dot products of centered curve against FPCA rotation columns | Yes — weight/rotation are real f64 constants; gradient flows through `Var` arithmetic | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 2900 lib tests pass (including 44 reverse-mode tests) | `cargo test -p fdars-core --features linalg,parallel --lib` | 2900 passed; 0 failed; finished in 37.70s | ✓ PASS |
| `vjp_soft_dtw_matches_finite_diff` | part of above lib run | `autodiff::reverse::tests::vjp_soft_dtw_matches_finite_diff ... ok` | ✓ PASS |
| `vjp_project_scores_matches_finite_diff` | part of above lib run | `autodiff::reverse::tests::vjp_project_scores_matches_finite_diff ... ok` | ✓ PASS |
| `vjp_composed_objective_matches_finite_diff` | part of above lib run | `autodiff::reverse::tests::vjp_composed_objective_matches_finite_diff ... ok` | ✓ PASS |
| `vjp` doctest runs (not `no_run`/`ignore`) | `cargo test -p fdars-core --features linalg,parallel --doc` | `test fdars-core/src/autodiff/reverse.rs - autodiff::reverse::vjp (line 494) ... ok`; 209 passed; 0 failed | ✓ PASS |

### Probe Execution

No probes declared or applicable (no `scripts/*/tests/probe-*.sh` for this phase).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| RAD-01 | 94-01, 94-02 | In-crate Wengert-list tape, `Var` type supporting full forward-mode op set | ✓ SATISFIED | `reverse.rs`: `Var`/`Node`/`TAPE` skeleton (Plan 01); all 12 ops + 8 transcendentals implemented with VJP partials, zero `unimplemented!()` stubs (Plan 02) |
| RAD-02 | 94-01, 94-03 | Backward pass seeds output adjoint, accumulates gradients via `vjp`, efficient for many-input→scalar | ✓ SATISFIED | `vjp` with double-clear lifecycle; Tier-5 edge-case tests including `vjp_many_input_scalar_single_sweep`; Tier-3 agreement for every op |
| RAD-03 | 94-04 | Gradients match forward-mode `Dual` and central FD on soft-DTW and FPCA scores | ✓ SATISFIED | Tier-3 agreement tests (1e-10); Tier-4 FD cross-checks on `soft_dtw_distance_generic`, `project_scores_generic`, composed objective (1e-6); validation targets unmodified |

No orphaned requirements for Phase 94 (RAD-01/02/03 are the only Phase 94 requirements per `REQUIREMENTS.md:66–68`).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | — | — | — |

Debt markers: none (`TBD`, `FIXME`, `XXX` grep empty). `unimplemented!()` grep: empty. No stubs, no placeholder returns.

The post-plan fix commits (`7f1b6d3c`, `6cdef665`) addressed code-review findings: WR-01 (privatise submodules), WR-02 (`FnOnce` bound on `vjp`), WR-03 (add `vjp`/`Var` to crate root `lib.rs`), IN-01 (lhs-const div test), IN-02 (Var doc note). All fixes are clean improvements with no outstanding issues.

### Human Verification Required

None. All success criteria are verifiable programmatically and all tests pass.

---

## Gaps Summary

No gaps found. All four ROADMAP success criteria are fully verified in the actual codebase:

1. **Op set completeness (SC-1 / RAD-01):** `reverse.rs` has complete `impl Scalar for Var` with all 12 ops (arithmetic + transcendentals). Zero `unimplemented!()` stubs. 21 Tier-1+Tier-2 tests all pass.

2. **`vjp` backward sweep (SC-2 / RAD-02):** `vjp` at line 505 implements the full 7-step lifecycle (double-clear, real leaf seeding, forward pass, backward sweep in one `for i in (0..n).rev()` loop, gradient collection). 18 Tier-3/Tier-5 tests all pass.

3. **FD cross-validation (SC-3 / RAD-03):** Three Tier-4 tests validate against central finite differences on the real FDA functions: soft-DTW (m=16, tol 1e-6), FPCA scores (m=24, tol 1e-6), and composed objective (tol 1e-6). Both `soft_dtw_distance_generic` and `project_scores_generic` are called with `Var` with zero code changes to those functions.

4. **No new dependency (SC-4):** `git diff` on both `Cargo.toml` files is empty from phase-start to HEAD.

---

_Verified: 2026-09-11T07:45:00Z_
_Verifier: Claude (gsd-verifier)_
