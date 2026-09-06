---
phase: 77-gradient-api-composition-demo-integration
verified: 2026-09-06T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 77: Gradient API, Composition Demo & Integration Verification Report

**Phase Goal:** An ergonomic public gradient entry point over the `Scalar`-generic subset ships, together with a worked end-to-end composition example, full crate-root + prelude re-exports, and a running module doctest — proving AD flows through arbitrary compositions of the differentiable ops. (DIF-04)
**Verified:** 2026-09-06
**Status:** passed
**Re-verification:** No — initial verification

This is the FINAL phase of milestone v0.39.0. Completes GAP-08 and exhausts the v0.31.0 GAP-BACKLOG.

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `grad(f, x)` returns `(value, gradient)` with `gradient.len() == x.len()`; on `Sum(xi^2)` it matches `2*xi` to ≤1e-12 (SC #1) | ✓ VERIFIED | `pub fn grad<F: Fn(&[Dual]) -> Dual>(f, x) -> (f64, Vec<f64>)` at autodiff.rs:511, `#[must_use]`. Behavioral test `grad_sum_of_squares_closed_form` (autodiff.rs:963) ran green: value 14.0, grad [2,4,6] ≤1e-12, `gradient.len()==3`. Also `jacobian` (559) + `directional_derivative` (611) shipped, tests `jacobian_known_answer` / `directional_derivative_projects_gradient` pass. |
| 2 | A composed objective over ≥2 Phase-76 ops has its grad-gradient matching central FD ≤1e-6 on spanning full-rank curves (SC #2) | ✓ VERIFIED | `grad_composed_objective_matches_finite_diff` (autodiff.rs:1029) ran green. Composes `soft_dtw_distance_generic` (op 1) + `lambda*sum(project_scores_generic^2)` (op 2) into one scalar; n=40 ≫ m=24 seeded spanning curves, grid [0.1,0.9], every component vs central FD h=1e-6 asserted <1e-6, plus f64 composition parity <1e-12. Behavioral cross-check, not a presence check. |
| 3 | The full differentiable surface (Scalar, Dual, diff, grad, soft_dtw_distance_generic, amplitude_distance_at_warp_generic, project_scores_generic) reachable via BOTH crate root and prelude (SC #3) | ✓ VERIFIED | External-crate integration test `tests/autodiff_reexports.rs` — `reexports_reachable_via_crate_root` + `via_prelude::reexports_reachable_via_prelude` both ran green (live calls to every symbol on both paths). lib.rs re-exports at :272/:582/:585/:640; prelude at :18/:21/:50/:103. |
| 4 | The autodiff module doctest composes ops and calls grad; `cargo test --doc` exits 0 (SC #4) | ✓ VERIFIED | Composed module doctest at autodiff.rs:48 (soft-DTW + FPCA scores, calls `grad`, asserts length/finite). `cargo test --doc`: 209 passed, 0 failed, 4 ignored — incl. all 6 autodiff doctests (module composed demo + grad/jacobian/directional_derivative/diff per-fn + existing). |
| 5 | Additive/non-breaking: no existing signature changed, no new `[dependencies]`, MSRV 1.81 | ✓ VERIFIED | `git show f3b4064b -- fdars-core/Cargo.toml` produces no diff → `[dependencies]` untouched. Commit diff on autodiff.rs is pure insertion (356+); zero removed `pub fn/struct/trait` lines. lib.rs/prelude.rs changes are re-export list edits (rustfmt reflows, identical identifier sets). Clippy `--all-targets` clean. |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `fdars-core/src/autodiff.rs` | grad + jacobian + directional_derivative + module doctest + inline tests | ✓ VERIFIED | grad (511), jacobian (559), directional_derivative (611); composed module doctest (48); 6 grad/jacobian/dir-deriv/composed inline tests, all pass. |
| `fdars-core/src/lib.rs` | crate-root re-exports | ✓ VERIFIED | `pub use autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};` (585); `soft_dtw_distance_generic` (640); `amplitude_distance_at_warp_generic` (272); `project_scores_generic` present exactly once (582, pre-existing, not duplicated). |
| `fdars-core/src/prelude.rs` | prelude re-exports | ✓ VERIFIED | autodiff surface (21); `project_scores_generic` (18); `soft_dtw_distance_generic` (50); `amplitude_distance_at_warp_generic` (103). |
| `fdars-core/tests/autodiff_reexports.rs` | external-crate reachability test (both paths) | ✓ VERIFIED | Gated `#![cfg(all(feature="linalg", feature="parallel"))]`; both crate-root and prelude tests pass with live calls to every symbol. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| grad | seeding index k → gradient[k] | m forward passes, `Dual::seed(x[k])` / `Dual::constant(x[j])` | ✓ WIRED | autodiff.rs:518-533; pinned by closed-form test (2,4,6 ≤1e-12) and FD cross-check. |
| composed objective | Dual tangent flow through chain | ops instantiated at Dual | ✓ WIRED | Objective closure (autodiff.rs:1082) instantiates soft_dtw + project_scores at Dual; FD cross-check ≤1e-6 confirms tangents flow end-to-end. |
| project_scores_generic | lib.rs re-export | not duplicated | ✓ WIRED | Appears exactly once in lib.rs (grep count = 1). |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| grad + composition + jacobian/dir-deriv tests | `cargo test ... grad` | 14 passed, 0 failed | ✓ PASS |
| Doctests (incl. composed autodiff demo) | `cargo test ... --doc` | 209 passed, 0 failed, 4 ignored | ✓ PASS |
| Whole crate + reexports integration | `cargo test ... ` (all) | 2859 lib passed, 0 failed; autodiff_reexports 2 passed, 0 failed; all suites 0 failed | ✓ PASS |
| Clippy all-targets, deny warnings | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | exit 0, 0 warnings | ✓ PASS |
| No new dependency | `git show f3b4064b -- fdars-core/Cargo.toml` | empty diff | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| DIF-04 | 77-01-PLAN.md | Ergonomic (value, gradient)/Jacobian entry point + composition demo + crate-root/prelude re-exports + module doctest | ✓ SATISFIED | All 4 ROADMAP success criteria verified behaviorally (SC #1–#4) + additive/no-new-dep constraint. |

### Anti-Patterns Found

None. No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers in any of the four modified files. WR-01 from 77-REVIEW.md (`directional_derivative` release-build truncation) is FIXED — a hard runtime `assert_eq!(direction.len(), x.len(), ...)` at autodiff.rs:616 replaces the release-noop `debug_assert`, with a `# Panics` doc note.

### Human Verification Required

None. All success criteria are programmatically verifiable and were verified by running the gates.

### Gaps Summary

No gaps. Every ROADMAP success criterion (SC #1–#4) plus the additive/no-new-dependency constraint is backed by green gate results run in this verification (not SUMMARY claims). The gradient API, composition demo, both re-export paths, and the module doctest are all present, wired, and behaviorally exercised. This closes the final phase of milestone v0.39.0 (GAP-08).

---

_Verified: 2026-09-06_
_Verifier: Claude (gsd-verifier)_
