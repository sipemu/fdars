# Requirements: fdars — v0.44.0 Differentiable Core (Reverse-Mode & Broadened Subset)

**Defined:** 2026-09-10
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps — this milestone completes the **differentiable-core** section of `documentation/ROADMAP-TO-1.0.md` (DIF-F1/F2/F3).

This is an **implementation milestone**: real `fdars-core/src/` changes, **strictly additive/non-breaking** (defaulted type params — protects R + WASM bindings + 28 examples), **no new crate dependency** (in-crate reverse-mode tape). Builds directly on the v0.39.0 forward-mode AD core (`Scalar` trait, `Dual<T>`, `grad`/`jacobian`). Ships as 0.44.0 (still 0.x).

## v0.44.0 Requirements

### Reverse-Mode Autodiff (RAD) — DIF-F1

- [x] **RAD-01**: An in-crate reverse-mode tape (Wengert-list) core records operations on a `Var`/tape scalar type, supporting the same operation set as the forward-mode `Dual` (±, ×, ÷, sqrt, exp, ln, sin/cos, powf, abs, comparisons).
- [x] **RAD-02**: A backward pass seeds the output adjoint and accumulates input gradients, exposing a vector-Jacobian-product (VJP) entry point efficient for many-input→scalar objectives.
- [x] **RAD-03**: Reverse-mode gradients match the forward-mode `Dual` path and finite differences within tolerance on the existing differentiable subset (elastic soft-DTW distance, FPCA scores).

### Differentiable Operations (DOP) — DIF-F2 (four families)

- [x] **DOP-01**: Basis evaluation (B-spline / Fourier) and functional inner products are generic over `Scalar` and differentiable; the f64 path reproduces current numerics.
- [x] **DOP-02**: Scalar-on-function regression prediction (`fregre_lm` / FPCR path) is generic over `Scalar` and differentiable w.r.t. inputs; validated vs finite differences with f64 parity preserved.
- [x] **DOP-03**: Smoothing / roughness-penalty evaluation is generic over `Scalar` and differentiable (penalty w.r.t. curve values / smoothing inputs); f64 parity preserved.
- [x] **DOP-04**: Functional depth and curve distances (beyond the existing soft-DTW) are generic over `Scalar` and differentiable; validated vs finite differences with f64 parity preserved.

### Generic Scalar Hot-Paths (GEN) — DIF-F3

- [x] **GEN-01**: Targeted hot-path signatures are generalized over the scalar type via defaulted type params (`T = f64`) so every existing f64 call site, R + WASM binding, and all 28 examples compile unchanged.
- [x] **GEN-02**: Autodiff types (`Dual` and the reverse-mode `Var`) flow through the generalized hot-paths end-to-end; a composed objective yields correct gradients validated by finite differences.

### API & Release (REL)

- [x] **API-01**: Ergonomic gradient entry points (`grad` / `jacobian` / `vjp`) plus a worked end-to-end composition demo (finite-difference-checked), full crate-root + prelude re-exports for all new public surface, and a running module doctest.
- [ ] **REL-01**: Crate bumped 0.43.0 → 0.44.0 with CHANGELOG `[0.44.0]`; the differentiable-core items (DIF-F1/F2/F3) checked off on `documentation/ROADMAP-TO-1.0.md`; whole-crate gates green — `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, full `cargo test`, a `--features serde` build guard, all 28 examples + doctests, and `cargo package`. The `git tag v0.44.0` → crates.io publish is the deferred operator step.

## Future Requirements

Deferred to future milestones / tracked on `documentation/ROADMAP-TO-1.0.md`.

### Algorithm

- **SDTW-O1**: Replace the `soft_dtw_barycenter` MM-step descent with a proper global optimizer (L-BFGS / multi-restart). Independent of the differentiable core; own milestone.

### Ecosystem

- **fdars-j75**: Migrate the external `fdars-r` R wrapper to the `FdMatrix` API. Separate package, out of `fdars-core` scope.

### Release

- **1.0-CUT**: Bump to `1.0.0` and declare the public API stable once every `ROADMAP-TO-1.0.md` item clears. Terminal; must be last.

## Out of Scope

Explicitly excluded from v0.44.0. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New crate dependency for AD | In-crate tape only, consistent with every prior milestone; forward-mode `Dual` was hand-written the same way (v0.39.0) |
| Breaking API changes | Strictly additive — generalization is via defaulted type params (`T = f64`); f64 signatures, bindings, and examples must compile unchanged |
| `soft_dtw_barycenter` optimizer redesign (SDTW-O1) | Separate algorithm-quality item; differentiability of the distance is in scope, the barycenter optimizer is not |
| Differentiating warp-searched `elastic_distance` (the DP itself) | Non-differentiable dynamic program; deferred at v0.39.0 (DIF-02) and unchanged here — the amplitude-at-warp / soft-DTW surrogates remain the differentiable paths |
| GPU / SIMD acceleration of the tape | Correctness-and-coverage milestone; performance tuning of reverse-mode is a later concern |
| `git tag` → crates.io publish | Irreversible/outward-facing; the deliberate publish trigger stays an operator step (`release.yml` couples tag-push to publish) |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| RAD-01 | Phase 94 | Complete |
| RAD-02 | Phase 94 | Complete |
| RAD-03 | Phase 94 | Complete |
| DOP-01 | Phase 96 | Complete |
| DOP-02 | Phase 97 | Complete |
| DOP-03 | Phase 97 | Complete |
| DOP-04 | Phase 98 | Complete |
| GEN-01 | Phase 95 | Complete |
| GEN-02 | Phase 99 | Complete |
| API-01 | Phase 99 | Complete |
| REL-01 | Phase 100 | Pending |

**Coverage:**

- v0.44.0 requirements: 11 total
- Mapped to phases: 11 ✓
- Unmapped: 0 ✓

---
*Requirements defined: 2026-09-10*
*Last updated: 2026-09-10 after roadmap creation (7 phases 94–100; 11/11 mapped)*
