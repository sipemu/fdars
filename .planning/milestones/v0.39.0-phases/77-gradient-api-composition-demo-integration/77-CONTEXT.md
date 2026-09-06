# Phase 77: Gradient API, Composition Demo & Integration - Context

**Gathered:** 2026-09-06
**Status:** Ready for planning
**Mode:** Auto-generated (integration/API phase — wiring + ergonomics over the Phase 75/76 surface; no external domain research needed)

<domain>
## Phase Boundary

The wrap-up of milestone v0.39.0 (DIF-04): ship the ergonomic public gradient entry point over the `Scalar`-generic differentiable subset, a worked end-to-end composition example, full crate-root + prelude re-exports, and a running module doctest — proving forward-mode AD flows through arbitrary compositions of the Phase 76 differentiable ops.

Consumes the shipped surface:
- **Phase 75 (`src/autodiff.rs`):** `Scalar` trait, `Dual{value,tangent}`, `Dual::seed/constant/extract`, and `diff<F: Fn(Dual)->Dual>(f, x) -> (f64,f64)` (SINGLE-input gradient helper).
- **Phase 76:** `soft_dtw_distance_generic<S>`, `amplitude_distance_at_warp_generic<S>`, `project_scores_generic<S>`, `FpcaResult::project_generic<S>`.

This is the LAST phase of the milestone. No new differentiable ops — only the gradient API, the demo, re-exports, and the doctest.

</domain>

<decisions>
## Implementation Decisions

### Locked (REQUIREMENTS.md)
- Additive/non-breaking; no new crate dependency; forward-mode only.

### Claude's Discretion (resolve at plan time)
- **Multi-input gradient API (SC #1):** `diff` handles a single scalar input. Add a multi-input entry point in `src/autodiff.rs` (or a small `autodiff` submodule) that seeds each of the m inputs in turn (m forward passes) and collects the length-m gradient. Recommended shape:
  - `grad<F: Fn(&[Dual]) -> Dual>(f: F, x: &[f64]) -> (f64, Vec<f64>)` — value + gradient of a scalar objective over an m-vector input (the ergonomic core; SC #1).
  - Optionally `jacobian<F: Fn(&[Dual]) -> Vec<Dual>>(f: F, x: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>)` for vector-valued objectives (nice-to-have; include if cheap).
  - Optionally a `directional_derivative` helper (seed a supplied tangent direction) — nice-to-have.
- **Composition demo (SC #2):** a worked example composing ≥2 differentiable ops from Phase 76 into ONE scalar objective, then `grad(...)` over its input curve — e.g. an objective like `soft_dtw_distance_generic(curve, ref, γ) + λ·Σ project_scores_generic(curve, fpca)²`, showing the gradient flows through the whole composition. This is the make-or-break "AD composes" proof. It should be BOTH the module doctest (SC #4) AND (optionally) a runnable `examples/` file. Validate the composed gradient against central finite differences in a test.
- **Re-exports (SC #3):** add crate-root re-exports in `lib.rs` and prelude re-exports in `prelude.rs` for the full differentiable surface: `Scalar`, `Dual`, `diff`, `grad` (+ `jacobian`/`directional_derivative` if added), and the generic ops (`soft_dtw_distance_generic`, `amplitude_distance_at_warp_generic`, `project_scores_generic`). Keep `pub mod autodiff;` too. Follow the existing `pub use` style in `lib.rs`/`prelude.rs`.
- **Doctest (SC #4):** the module-level `//!` doc on `autodiff` (or the gradient API) carries a runnable `\`\`\`` example that constructs an input, calls `grad` over a composed objective, and asserts something about the returned value/gradient — must run green under `cargo test --doc`.

</decisions>

<code_context>
## Existing Code Insights

### Re-export patterns
- `src/lib.rs:76` currently has only `pub mod autodiff;` (no crate-root re-export of its symbols yet).
- `src/lib.rs` re-exports crate-root types via `pub use` (e.g. `FdMatrix`, `FdarError`); `src/prelude.rs` (124 lines) re-exports commonly-used types via `pub use crate::<module>::{...};`. Match these styles.

### Symbols to surface
- From `autodiff`: `Scalar`, `Dual`, `diff` (+ new `grad`/`jacobian`).
- Generic ops: `soft_dtw_distance_generic` (`metric/soft_dtw.rs:157`), `amplitude_distance_at_warp_generic` (`alignment/differentiable.rs:125`), `project_scores_generic` (`regression.rs:232`), `FpcaResult::project_generic` (`regression.rs:139`, method — reachable via `FpcaResult`).

### Conventions
- Doctests run under `cargo test --doc`; the crate already has ~200 doctests. Module doc `//!` with a fenced ` ```rust ` block that `use fdars_core::prelude::*;` (or explicit paths) and asserts.
- `#[must_use]` on expensive pure fns; `Result` returns where validation applies (the gradient helpers are pure numeric → may return values directly).

### Integration point
- 28 `examples/` exist with `[[example]]` entries in `Cargo.toml`. IF adding a runnable example, add the `[[example]]` entry too (additive). A doctest alone satisfies SC #4; an `examples/` file is optional polish.

</code_context>

<specifics>
## Specific Ideas

Gates (SC-driven): SC #1 `grad` returns (value, length-m gradient); SC #2 composed-objective gradient matches central finite differences (≤1e-6) in a test using SPANNING full-rank curves; SC #3 the full surface is reachable from BOTH `fdars_core::<symbol>` (crate root) and `fdars_core::prelude::*` (assert with a compile-level use in a test or the doctest); SC #4 `cargo test --doc` green.

This phase completes GAP-08 and exhausts the v0.31.0 `GAP-BACKLOG.md`.

</specifics>

<deferred>
## Deferred Ideas

- Reverse-mode/VJP — DIF-F1.
- Broadening the differentiable subset beyond elastic + FPCA — DIF-F2.
- Making existing f64 public signatures generic — DIF-F3.
- Fixing the pre-existing `soft_dtw_backward` zero-gradient bug — separate behavior-changing backlog item (found in Phase 76), NOT this additive milestone.

</deferred>
