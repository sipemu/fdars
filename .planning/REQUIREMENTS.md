# Requirements: fdars — v0.39.0 DIFF (Differentiable FDA Core)

**Defined:** 2026-09-06
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against reference ecosystems — this milestone promotes GAP-08 (the last v0.31.0 `GAP-BACKLOG.md` item) by adding an in-crate forward-mode automatic-differentiation core so exact gradients flow through a scoped subset of FDA operations into optimization/ML pipelines.

## Milestone Design Decisions (locked at questioning)

- **AD mechanism:** in-crate forward-mode **dual numbers** + a `Scalar` trait; a scoped subset of ops is generic over the scalar so gradients **compose** through arbitrary op chains. NOT hand-written per-op gradients; NOT an external AD crate.
- **Differentiation mode:** **forward-mode** only (JVP / dual numbers), matching the ForwardDiff reference. Reverse-mode/VJP deferred (DIF-F1).
- **Scope:** **elastic distance + FPCA scores** (the two ops DIF-01 names). Broadening deferred (DIF-F2).
- **API shape:** strictly **additive** — existing f64 public signatures are untouched; generic versions live alongside (protects R + WASM bindings + 28 examples). Deprecate, never remove.
- **No new crate dependency.** Reuse-first pilot: `metric/soft_dtw.rs` (existing hand-written gradient to validate against), `elastic_*`, `regression.rs` FPCA.
- Reference baseline: Julia generic-programming idiom (ElasticFDA.jl + ForwardDiff).

## v1 Requirements

Requirements for milestone v0.39.0. Each maps to a roadmap phase.

### Differentiable Core (DIF)

- [x] **DIF-01**: In-crate `Scalar` trait + forward-mode `Dual<T>` number — dual carries value + tangent and implements the ops the differentiable subset needs (±, ×, ÷, `sqrt`, `exp`, `ln`, `sin`, `cos`, `powf`, `abs`, partial comparisons), plus gradient seed (set an input's tangent to 1) / extract helpers. Known-answer tests: dual arithmetic reproduces analytical derivatives of composed elementary functions to ≤1e-10. No new crate dependency.
- [x] **DIF-02**: Differentiable elastic distance — a generic-over-`Scalar` elastic (soft-DTW / amplitude+phase) distance path. At `Dual` it yields exact forward-mode gradients of the distance w.r.t. a curve's values, validated vs central finite differences AND vs the existing hand-written `soft_dtw` gradient; at `f64` it reproduces the current `elastic_distance` / `amplitude_distance` numerics within 1e-12.
- [x] **DIF-03**: Differentiable FPCA scores — a generic-over-`Scalar` FPCA score projection so gradients of FPC scores w.r.t. input-curve values flow through when instantiated at `Dual`; validated vs central finite differences; the `f64` instantiation reproduces the existing FPCA scores within tolerance.
- [ ] **DIF-04**: Gradient API + composition demo + integration — an ergonomic public `(value, gradient)` / directional-derivative / Jacobian entry point over the `Scalar`-generic subset; a worked end-to-end example composing differentiable ops into a scalar objective and taking its gradient (proving AD flows through composition); full crate-root + prelude re-exports; a running module doctest under `cargo test --doc`.

## Future Requirements

Deferred to a later milestone. Tracked but not in this roadmap.

### Differentiable Core (DIF)

- **DIF-F1**: Reverse-mode / VJP autodiff (backprop for a scalar loss w.r.t. many inputs — typical ML training). Requires a tape/graph engine; out of proportion for a scoped forward-mode pilot.
- **DIF-F2**: Broaden the differentiable subset beyond elastic + FPCA — basis evaluation, functional inner products, SRSF/warping ops, other regressions.
- **DIF-F3**: Make existing public f64 hot-path signatures themselves generic over the scalar type (this milestone keeps generic versions additive-alongside only, to avoid breaking R/WASM/examples).

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| External AD crate dependency (num-dual, autodiff, …) | Violates the standing no-new-crate-dependency convention; an in-crate `Dual` type is sufficient for forward-mode. |
| Reverse-mode tape/graph autodiff engine | Out of proportion for a scoped forward-mode pilot; deferred to DIF-F1. |
| Refactoring existing f64 public signatures to be generic | Breaking risk to R + WASM bindings + 28 examples; generic versions are added alongside instead (DIF-F3 covers the eventual refactor). |
| GPU / SIMD-specialized AD kernels | Not a functional-parity gap; well beyond a forward-mode pilot's scope. |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DIF-01 | Phase 75 | Complete |
| DIF-02 | Phase 76 | Complete |
| DIF-03 | Phase 76 | Complete |
| DIF-04 | Phase 77 | Pending |

**Coverage:**

- v1 requirements: 4 total
- Mapped to phases: 4 (Phase 75: DIF-01; Phase 76: DIF-02, DIF-03; Phase 77: DIF-04)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-06*
*Last updated: 2026-09-06 after roadmap creation (phases 75–77 mapped)*
