# Phase 75: Scalar Trait & Forward-Mode Dual Substrate - Context

**Gathered:** 2026-09-06
**Status:** Ready for planning
**Mode:** Auto-generated (infrastructure phase — smart discuss skipped, no user-facing grey areas)

<domain>
## Phase Boundary

Deliver the numeric substrate for milestone v0.39.0's differentiable subset: an in-crate `Scalar` trait plus a forward-mode `Dual<T>` number carrying value + tangent, with every arithmetic/transcendental op the elastic-distance and FPCA-score subset needs, and gradient seed/extract helpers — verified against analytical derivatives.

This phase delivers DIF-01 only. It creates the substrate; Phase 76 makes elastic distance + FPCA scores generic over it, and Phase 77 exposes the public gradient API. No differentiable FDA ops are made generic in this phase — only the substrate and its self-contained tests.

</domain>

<decisions>
## Implementation Decisions

### Locked at milestone questioning (see REQUIREMENTS.md "Milestone Design Decisions")
- **In-crate forward-mode dual numbers** — NOT hand-written per-op gradients, NOT an external AD crate. Gradients must compose through arbitrary op chains.
- **Forward-mode only** (JVP / dual numbers), matching the ForwardDiff reference. Reverse-mode/VJP is deferred (DIF-F1).
- **No new crate dependency** — the `Dual` type and `Scalar` trait are entirely in-crate.
- **Additive/non-breaking** — existing f64 public signatures are untouched; the `Scalar` trait must be implemented for `f64` so f64-instantiated generic code compiles and runs identically to the current numerics.

### Claude's Discretion
All remaining implementation choices are at Claude's discretion — this is a numeric-substrate infrastructure phase with no user-facing behavior. Guidance for planning:
- **Op set (from the ROADMAP success criteria):** `Dual` must support ±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, and partial comparisons, plus seed (set one input's tangent to 1) and extract (read value + derivative) helpers.
- **`Scalar` trait shape:** define the minimal trait the differentiable subset (Phase 76) will be written against — the arithmetic/transcendental ops above, `From<f64>`/constants (zero/one), and comparison/ordering as needed. Implement it for both `f64` and `Dual<f64>` (and keep it generic-friendly, e.g. `Dual<T: Scalar>` if it composes cleanly, but nested duals / higher-order AD are NOT required this milestone).
- **Module placement:** a new self-contained module (e.g. `src/autodiff.rs` or `src/autodiff/`) is the natural home; follow existing module/naming conventions and wire minimal re-exports as needed (full crate-root/prelude surfacing is Phase 77's job).
- **Whether to lean on an existing numeric trait** (e.g. a `num-traits`-style bound) is fine ONLY if it is already a transitive/available dependency and adds no new crate to `Cargo.toml`; otherwise define the trait in-crate. Confirm at plan time that no `Cargo.toml` change is needed.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets / Reference
- `fdars-core/src/metric/soft_dtw.rs` — already carries a **hand-written gradient**; it is the reuse pilot and the validation oracle for Phase 76. Studying its gradient here informs the `Scalar` op set the substrate must provide.
- `fdars-core/src/regression.rs` — FPCA (`fdata_to_pc_1d`, `FpcaResult`); Phase 76 will make its scoring path generic over `Scalar`.
- `fdars-core/src/elastic_*` — amplitude/phase/elastic distances that Phase 76 targets.

### Established Patterns
- Column-major `FdMatrix` (`src/matrix.rs`); all public fns return `Result<T, FdarError>`; `#[derive(Debug, Clone, PartialEq)]` on public types; inline `#[cfg(test)] mod tests`.
- Feature-gated parallelism (`parallel`), `linalg` feature (Rust 1.84+). The substrate itself should be feature-agnostic.

### Integration Points
- New module registered in `src/lib.rs`. Minimal re-exports for now; the full crate-root + prelude surface is finalized in Phase 77 (DIF-04).

</code_context>

<specifics>
## Specific Ideas

Known-answer tests are the make-or-break gate: dual arithmetic must reproduce the analytical derivative of composed elementary functions (chains of the supported ops) to ≤1e-10. Include a test that seeds a tangent, runs a non-trivial composition (e.g. `sqrt(exp(x)·sin(x) + x²)`), and checks the extracted derivative against the closed-form value.

</specifics>

<deferred>
## Deferred Ideas

- Nested/higher-order duals (second derivatives) — not required this milestone.
- Reverse-mode/VJP — deferred (DIF-F1).
- Making existing f64 public signatures themselves generic — deferred (DIF-F3); this milestone keeps generic code additive-alongside.

</deferred>
