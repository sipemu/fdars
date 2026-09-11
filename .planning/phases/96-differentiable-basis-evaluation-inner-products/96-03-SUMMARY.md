---
phase: 96-differentiable-basis-evaluation-inner-products
plan: "03"
subsystem: basis
tags: [gate, non-breaking, DOP-01, examples, serde, wasm, doctests, verification]

requires:
  - phase: 96-differentiable-basis-evaluation-inner-products
    plan: "01"
    provides: generic bspline_basis_from_knots<T> + helpers
  - phase: 96-differentiable-basis-evaluation-inner-products
    plan: "02"
    provides: generic fourier_basis_eval<T> core + delegating f64 wrappers

provides:
  - "GEN/DOP-01 non-breaking compile-time proof: the basis-eval generalization changed nothing outside basis/{bspline,fourier,tests}.rs and broke no example, binding, or build"

affects: [97-differentiable-regression]

actuals:
  tokens: 0
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Non-breaking proof = whole-crate gate: clippy --all-targets + full test + doctests + 28 examples + serde + wasm + git-diff churn confinement + no-Cargo-change"

key-files:
  created: []
  modified: []

key-decisions:
  - "Executed inline by the orchestrator (verification-only plan; reliable to run directly, and consistent with the inline recovery of 96-01/96-02 this phase)."

commits:
  - "(gate is verification-only; SUMMARY commit records the evidence)"

gates:
  - "GATE 1 clippy --all-targets --features linalg,parallel -- -D warnings: clean"
  - "GATE 2 full test (cargo test -p fdars-core --features linalg,parallel): 2916 lib passed + all integration binaries green"
  - "GATE 6 doctests: 209 passed, 5 ignored"
  - "GATE 4 examples (cargo build --examples): all 28 build clean"
  - "GATE 3 serde (cargo build --features serde): clean (no regression)"
  - "GATE 5 wasm (cargo build --target wasm32-unknown-unknown --features js): clean"
  - "GATE 7 churn: git diff base..HEAD -- fdars-core/src confined to basis/{bspline,fourier,tests}.rs (CHURN_CONFINED)"
  - "GATE 8 no-new-dep: git diff base..HEAD -- Cargo.toml fdars-core/Cargo.toml empty"
---

# Phase 96 Plan 03 — Non-Breaking Compile-Time Gate (DOP-01)

The differentiable basis-evaluation generalization (Plans 01 B-spline in-place, 02 Fourier additive core) is proven **strictly non-breaking** by the whole-crate gate:

| Gate | Command | Result |
|------|---------|--------|
| clippy | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | clean |
| full test | `cargo test -p fdars-core --features linalg,parallel` | 2916 lib + integration, 0 failed |
| doctests | (same run, Doc-tests fdars-core) | 209 passed, 5 ignored |
| 28 examples | `cargo build -p fdars-core --examples --features linalg,parallel` | clean (28 `[[example]]` targets) |
| serde | `cargo build -p fdars-core --features serde` | clean (no regression) |
| wasm | `cargo build -p fdars-core --target wasm32-unknown-unknown --features js` | clean |
| churn confined | `git diff base..HEAD -- fdars-core/src` | only `basis/{bspline,fourier,tests}.rs` |
| no new dep | `git diff base..HEAD -- Cargo.toml fdars-core/Cargo.toml` | empty |

Public signatures of `fourier_basis` / `fourier_basis_with_period` are unchanged; `bspline_basis_from_knots<T>` callers compile at inferred `T = f64`. The `fdars-r` R package is external (not in this repo) — its binding surface is unaffected because only internal basis-eval bodies + a new additive core changed, not any consumed public signature.

DOP-01 (family 1 of DIF-F2) is complete: B-spline and Fourier basis evaluation are generic over `Scalar`, differentiable through `t`, bit-identical at `f64`, and FD-verified at `Dual` and `Var`.
