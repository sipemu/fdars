---
phase: 97-differentiable-regression-prediction-smoothing-penalties
plan: "03"
subsystem: scalar_on_function
tags: [gate, non-breaking, DOP-02, DOP-03, examples, serde, wasm, doctests]

requires:
  - phase: 97-differentiable-regression-prediction-smoothing-penalties
    plan: "01"
    provides: predict_curve_generic
  - phase: 97-differentiable-regression-prediction-smoothing-penalties
    plan: "02"
    provides: penalty_value_generic

provides:
  - "DOP-02/03 non-breaking proof: change confined to 6 files; no public signature broken; no new dependency"

affects: [98-differentiable-depth]

actuals: { tokens: 0, tasks: 3, commits: 0 }

tech-stack: { added: [], patterns: [] }
key-files: { created: [], modified: [] }
key-decisions:
  - "Verification-only gate; executed inline."

gates:
  - "clippy --all-targets --features linalg,parallel -- -D warnings: clean"
  - "full test (cargo test -p fdars-core --features linalg,parallel): 0 failed across lib + integration"
  - "doctests: 209 passed, 5 ignored"
  - "28 examples build: clean"
  - "serde build: clean"
  - "wasm32-unknown-unknown --features js: clean"
  - "churn: git diff cd150e3f..HEAD -- fdars-core/src confined to fregre_lm.rs, scalar_on_function/{mod,tests}.rs, smooth_basis.rs, lib.rs, prelude.rs"
  - "no-new-dep: git diff Cargo.toml empty"
---

# Phase 97 Plan 03 — Non-Breaking Gate (DOP-02 + DOP-03)

Whole-crate proof that adding `predict_curve_generic` (DOP-02) and `penalty_value_generic` (DOP-03) is strictly non-breaking:

| Gate | Result |
|------|--------|
| clippy --all-targets | clean |
| full test | 0 failed (lib + integration) |
| doctests | 209 passed |
| 28 examples | clean |
| serde | clean |
| wasm | clean |
| churn confined | 6 files (2 new fns + their tests + additive re-exports) |
| no new dep | confirmed |

`predict_fregre_lm` and the penalty-matrix constructors keep their public signatures. DOP-02 + DOP-03 complete.
