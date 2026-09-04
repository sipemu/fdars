# Phase 71: Prediction, Diagnostics & Integration - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — recommendations aligned to locked v0.37.0 STATE.md decisions + the Phase 69/70 wavelet surface

<domain>
## Phase Boundary

Final phase of the WAV milestone. Make the wavelet-domain regressors usable end-to-end and expose the full public surface:

- **Out-of-sample `predict`** for both `wcr` and `wnet` fitted results — re-transform new curves via the *same fitted DWT* used at fit time, then apply the fitted coefficient-space weights + intercept. Self-consistent: re-passing the training curves reproduces the training fitted values within tolerance; prediction on new curves returns finite values.
- **Accessors** — coefficient-function (β(t)) and fitted-value accessors on both result structs.
- **Integration** — promote the full wavelet surface to crate-root + `prelude` re-exports (currently reachable only as `crate::wavelet::...` / `crate::wavelet::regression::...`), with a running end-to-end module doctest under `cargo test --doc`.
- All additive/non-breaking: no existing public signature changes; R + WASM bindings + 28 examples unaffected; whole-crate `cargo test` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` green.

Out of scope: the crate version bump 0.36.0 → 0.37.0 + git tag + crates.io publish — that is the operator ship-time step run at milestone completion, NOT this phase.

</domain>

<decisions>
## Implementation Decisions

### Prediction, Accessors & Integration
- **`predict` API shape:** methods on the fitted result structs — `WcrResult::predict(&self, new: &FdMatrix) -> Result<Vec<f64>, FdarError>` and `WnetResult::predict(...)`. Each re-transforms new curves with the *stored fitted* DWT config (family/mode/level/CoeffLayout) → coefficient design → stored coefficient-space β + intercept. New curves must share the training grid length (validate → `FdarError`, never panic).
- **Result-struct fields:** if `WcrResult`/`WnetResult` do not already carry what `predict` needs (fitted family/mode/level or CoeffLayout, coefficient-space β weights, intercept), ADD those fields. This is additive/non-breaking — the structs are new this milestone (v0.37.0) and nothing external depends on them yet. Do NOT reconstruct predict inputs lossily from β(t).
- **Accessors:** add `beta_t()` / `coefficient_function()` and `fitted_values()` methods (returning `&[f64]`) to both result structs, matching existing crate accessor conventions (cf. `FregreLmResult`). Keep existing public fields intact.
- **Re-export scope:** expose the FULL public surface at BOTH the crate root (`lib.rs`) and `prelude` (`prelude.rs`): `WaveletFamily`, `BoundaryMode`, `WaveletCoeffs`, `decompose`, `reconstruct`, `decompose_matrix`, `max_level`, `WcrMethod`, `WcrConfig`, `WcrResult`, `WnetConfig`, `WnetResult`, `wcr`, `wnet` (+ any predict/accessor entry points that are free functions). Follow the existing prelude grouping/comment style.
- **Doctest:** one self-contained end-to-end module doctest (synthetic data → fit `wcr` or `wnet` → `predict` → read β(t)/fitted) that passes under `cargo test --doc`. Use `use fdars_core::prelude::*;` to also prove the prelude surface.

### Claude's Discretion
- Exact accessor method names (`beta_t` vs `coefficient_function` — provide the one that best matches crate conventions; may provide both if cheap).
- Whether `predict` lives as an inherent method only, or also a thin free function — inherent method is the baseline requirement.
- Minor field additions to result structs (names/visibility) — keep `#[non_exhaustive]` so future additions stay non-breaking.
- Which module the doctest lives in (`wavelet/mod.rs` `//!` module doc or `wavelet/regression.rs`) — pick the most discoverable.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crate::wavelet` (Phase 69, already `pub`): `WaveletFamily`, `BoundaryMode`, `WaveletCoeffs` (+ accessors), `decompose`, `reconstruct`, `decompose_matrix`, `max_level`.
- `crate::wavelet::regression` (Phase 70, `pub mod`): `WcrMethod`, `WcrConfig`, `WcrResult`, `wcr`; `WnetConfig`, `WnetResult`, `wnet`; `pub(crate)` shared helpers `curves_to_coeff_design`, `coeff_weights_to_beta_t`, `CoeffLayout` (predict reuses `curves_to_coeff_design` with the fitted config to build the new-curve design).
- `lib.rs:123` — `pub mod wavelet;` (no crate-root re-exports yet — this phase adds them).
- `prelude.rs` — standard `pub use crate::...` grouping with section comments (e.g. "// Regression results"); add a "// Wavelet-domain regression (v0.37.0)" group.
- `FdMatrix` (`matrix.rs`) — column-major; rows = curves.

### Established Patterns
- Accessor methods return `&[f64]` (cf. `WaveletCoeffs::approx`, `FregreLmResult` field access). `#[must_use]` on non-trivial getters/predict.
- All public fns return `Result<T, FdarError>`, validate at entry, never panic. Result structs `#[non_exhaustive]` + serde-gated.
- Doctests across the crate use `use fdars_core::prelude::*;` and run under `cargo test --doc`.

### Integration Points
- After this phase the milestone is code-complete; the crate version bump + tag + publish is the operator ship step (gsd-complete-milestone / manual), NOT this phase.
- No `Cargo.toml` dependency change; MSRV stays 1.81.
- Pre-existing (unrelated) `--features serde` build break in `shapelet/classifier.rs` — do NOT let new re-exports pull it into a required build path; new wavelet types are serde-clean.

</code_context>

<specifics>
## Specific Ideas

- **`predict` self-consistency (SC1):** a test must fit on training curves, `predict` on the SAME training curves, and assert the returned values match the stored training fitted values within a tight tolerance (≤1e-8 or tighter). Prediction on fresh curves must be finite/NaN-free. Do NOT loosen tolerance to paper over a re-transform mismatch — the fitted DWT config must be applied identically.
- **Full-surface reachability (SC3):** a doctest (or test) that imports via `fdars_core::prelude::*` and via `fdars_core::{WaveletFamily, wcr, ...}` crate-root paths, exercising decompose/reconstruct + a wcr/wnet fit + predict end-to-end.
- **Non-breaking (SC4):** whole-crate `cargo test` (lib + `--doc`) green, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean, `cargo fmt --check` clean; the 28 examples + R/WASM bindings must still build (no existing signature changed).

</specifics>

<deferred>
## Deferred Ideas

- Crate version bump 0.36.0 → 0.37.0 + `v0.37.0` git tag + crates.io publish → operator ship-time step (milestone completion), not this phase.
- Binomial/logistic wavelet regression + predict for GLM families → WAV-F1.
- Exposing the DWT/regressors to the R + WASM bindings → follow-up (issue `fdars-j75`), not this milestone.

</deferred>
