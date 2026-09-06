---
phase: "77"
plan: "01"
requirements: [DIF-04]
status: COMPLETE
commit: f3b4064be1738076f8ab351562837a89c9ad145c
---

# Phase 77 / Plan 01 — Gradient API, Composition Demo & Integration (DIF-04)

**Status: COMPLETE.** Final plan of milestone v0.39.0. Completes GAP-08 and
exhausts the v0.31.0 GAP-BACKLOG.

## Symbols added

### `fdars-core/src/autodiff.rs`
- `pub fn grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>)` —
  `#[must_use]`, multi-input gradient via m forward passes (seed index k → gradient[k]);
  m==0 evaluates f once and returns empty gradient. Per-fn doctest included.
- `pub fn jacobian<F: Fn(&[Dual]) -> Vec<Dual>>(f: F, x: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>)`
  — `#[must_use]`, n×m Jacobian (row=output, col=input). Per-fn doctest.
- `pub fn directional_derivative<F: Fn(&[Dual]) -> Dual>(f, x, direction) -> (f64, f64)`
  — `#[must_use]`, single-pass ∇f·direction, `debug_assert` on length. Per-fn doctest.
- Module `//!` composed-objective doctest (SC #4): builds an inline trained
  `FpcaResult` via `fdata_to_pc_1d`, composes `soft_dtw_distance_generic` +
  `project_scores_generic` into one scalar, calls `grad`, asserts length/finite.
- Inline tests: `grad_sum_of_squares_closed_form` (value 14, grad [2,4,6] ≤1e-12),
  `grad_single_input_agrees_with_diff`, `grad_empty_input_returns_constant`,
  `jacobian_known_answer`, `directional_derivative_projects_gradient`,
  `grad_composed_objective_matches_finite_diff` (SC #2: spanning full-rank n=40,
  m=24, grid [0.1,0.9]; every component vs central FD h=1e-6 ≤1e-6, plus f64
  composition parity ≤1e-12).

### `fdars-core/src/lib.rs` (crate-root re-exports)
- `pub use autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};`
- `amplitude_distance_at_warp_generic` added to the `alignment` re-export block.
- `soft_dtw_distance_generic` added to the `metric` re-export block.
- `project_scores_generic` left as-is (already present) — not duplicated.

### `fdars-core/src/prelude.rs` (prelude re-exports)
- `pub use crate::autodiff::{diff, directional_derivative, grad, jacobian, Dual, Scalar};`
- `project_scores_generic` folded into the regression re-export.
- `soft_dtw_distance_generic` folded into the metric re-export.
- `amplitude_distance_at_warp_generic` added to the alignment block.

### `fdars-core/tests/autodiff_reexports.rs` (new integration test, SC #3)
- Gated `#![cfg(all(feature = "linalg", feature = "parallel"))]`.
- `reexports_reachable_via_crate_root` — touches the full surface via
  `use fdars_core::{...}` with live calls to each generic op.
- `via_prelude::reexports_reachable_via_prelude` — same surface via
  `use fdars_core::prelude::*;`. Both paths proven by compilation + assertions.

## Gate results
- New grad tests: **14 passed, 0 failed** (`... grad` filter — includes all new grad/jacobian/dir-deriv/compos tests).
- Doctests: **209 passed, 0 failed, 4 ignored** (incl. 6 autodiff doctests: module composed demo + grad/jacobian/directional_derivative/diff per-fn + existing).
- Whole crate: **2859 lib passed, 0 failed**; integration `autodiff_reexports` **2 passed, 0 failed**; all other suites 0 failed.
- Clippy `--all-targets --features linalg,parallel -- -D warnings`: **clean** (fixed initial `type_complexity` by replacing fn-pointer bindings with live calls).
- `cargo fmt` applied; `cargo fmt --check`: **clean**.
- No-new-dep: `git diff fdars-core/Cargo.toml` empty — **no [dependencies] change**.

## Deviations
- **Module doctest import:** `fdata_to_pc_1d` is not in the prelude, so the
  composed module doctest adds one explicit `use fdars_core::regression::fdata_to_pc_1d;`
  alongside `use fdars_core::prelude::*;`. Did not widen the prelude (out of scope).
- **Reachability test:** initial fn-pointer type-annotated bindings tripped
  clippy `type_complexity` under `--all-targets`; replaced with live calls to each
  generic op (still touches every symbol, no dead code). Behavior unchanged.
- Included both optional helpers (`jacobian`, `directional_derivative`) — clean,
  additive, same seeding loop, no new dependency.

## Commit
`f3b4064be1738076f8ab351562837a89c9ad145c` —
`feat(autodiff): grad/jacobian API + composition demo + re-exports (DIF-04)`
