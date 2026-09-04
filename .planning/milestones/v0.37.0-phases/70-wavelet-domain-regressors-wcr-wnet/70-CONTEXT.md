# Phase 70: Wavelet-Domain Regressors (`wcr` + `wnet`) - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — recommendations aligned to locked v0.37.0 STATE.md decisions + Phase 69 DWT surface

<domain>
## Phase Boundary

Deliver two Gaussian-response wavelet-domain scalar-on-function regressors, each transforming curves → wavelet coefficients via the Phase 69 DWT (`crate::wavelet`), then reusing fdars' existing regression machinery:

- **`wcr`** — PCR *and* PLS in wavelet-coefficient space (reusing `regression::fdata_to_pc_1d` for PCR and `scalar_on_function/pls.rs` for PLS). Returns β(t), intercept, fitted values.
- **`wnet`** — elastic-net (L1 lasso + L2 ridge) on wavelet coefficients via coordinate descent + soft-thresholding, with cross-validated λ. Returns β(t), intercept, fitted values, and the selected (nonzero) coefficients.

`wcr` and `wnet` are independent of each other once the DWT exists. Gaussian response only (binomial/logistic deferred → WAV-F1). Numeric-output parity with refund `wcr`/`wnet` is the goal — NOT basis-internal parity with refund's `wavethresh`/`wmtsa`.

Out of scope: prediction / out-of-sample `predict`, coefficient-function accessors as public API surface, and crate-root/prelude re-exports — all deferred to **Phase 71**. Phase 70 keeps the regressors reachable internally (`crate::wavelet::...` or a new `wavelet_regression` submodule) and tested via inline `#[cfg(test)]`.

</domain>

<decisions>
## Implementation Decisions

### Wavelet-Domain Regressor Design
- **`wnet` elastic-net engine:** a NEW thin per-coefficient coordinate-descent adapter (L1 soft-threshold + L2 ridge shrinkage) living in the wavelet-regression module, reusing the *soft-threshold pattern* from `scalar_on_function/additive.rs` — NOT `variable_selection` directly (that is group-lasso in FPC-score space; `wnet` is per-coefficient elastic-net, so singleton-group force-fitting is rejected). Still additive/non-breaking; no change to `additive.rs`'s public signatures.
- **`wcr` PCR/PLS wiring:** reuse `regression::fdata_to_pc_1d` (PCR) and `scalar_on_function/pls.rs` (PLS) on the wavelet-coefficient design matrix (treat each curve's concatenated wavelet coefficients as the predictor vector). No new PCA/PLS implementation.
- **DWT defaults for the regressors:** `Daubechies(4)` (db4), `BoundaryMode::Periodic`, auto max-level, with **all** coefficients (final approximation + every detail level) concatenated into the per-curve design vector. All caller-configurable via the config struct (family, boundary, level, ncomp for wcr / λ-grid + α for wnet).
- **`wnet` CV-λ selection:** deterministic K-fold (fixed fold partition, fixed seed — reproducible across runs per SC3), geometric λ grid; select λ minimizing CV MSE, ties broken toward the LARGER λ (sparser). α (L1/L2 mix) is a config parameter with a sensible default.

### Claude's Discretion
- Module placement: a new `wavelet_regression.rs` OR a `wavelet/regression.rs` submodule (peer of `scalar_on_function/`) — planner's call; keep it discoverable and matching house layout. Types deriving serde must be serde-gated.
- Exact config/result struct names + fields (e.g. `WcrConfig`/`WcrResult`, `WnetConfig`/`WnetResult`), following existing `FregreLmResult` / config-struct conventions (`Debug, Clone, PartialEq`, `#[non_exhaustive]`, `#[must_use]`).
- Whether `wcr` returns PCR and PLS from one entry point (method enum) or two functions — pick the ergonomic option; both PCR and PLS must be reachable and tested.
- β(t) reconstruction: map the fitted wavelet-coefficient weights back to the time domain via the inverse DWT (apply the Phase 69 `reconstruct` to the coefficient-space β) — exact mechanism at planner discretion, but β(t) must be finite and recover the injected signal on synthetic data.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crate::wavelet` (Phase 69) — `WaveletFamily` (Haar, Daubechies(2..=10)), `BoundaryMode` (Periodic/Symmetric), `max_level`, `WaveletCoeffs`, `decompose`/`reconstruct`, and `decompose_matrix` (FdMatrix batch → per-curve coefficients). This is the curves→coefficients seam. NOTE: these are currently `pub(crate)` / not re-exported (Phase 71 exposes them) — the regressors consume them internally, which is fine.
- `regression::fdata_to_pc_1d` — FPCA/PCR scoring (SVD via nalgebra); the PCR path for `wcr`.
- `scalar_on_function/pls.rs` — PLS regression; the PLS path for `wcr`.
- `scalar_on_function/additive.rs` — GroupLasso coordinate-descent + soft-threshold pattern (`variable_selection`, private helpers) to model `wnet`'s per-coefficient CD after (pattern reuse, not direct call).
- `scalar_on_function/fregre_lm.rs` — `FregreLmResult` shape (β(t), intercept, fitted values) to mirror for the new result structs.
- `linalg::cholesky_solve`, `helpers::simpsons_weights` — ridge/normal-equation solves and integration weights.
- `FdMatrix` (`matrix.rs`) — column-major; rows = curves. Input to `decompose_matrix`.

### Established Patterns
- All public fns return `Result<T, FdarError>`, validate dims/params at entry, never panic. Config structs are builder-style with `#[non_exhaustive]`; result structs derive `Debug, Clone, PartialEq` + serde-gated.
- Deterministic seeding pattern: `StdRng::seed_from_u64(seed + k)` for reproducible CV folds.
- Inline `#[cfg(test)] mod tests`; shared `test_helpers::uniform_grid(n)`.

### Integration Points
- Consumed in Phase 71 (predict + coefficient/fitted-value accessors + crate-root/prelude re-exports + doctest) — keep the fitted result structs carrying everything predict will need (the fitted DWT config, coefficient-space weights, intercept, training design metadata).
- No `Cargo.toml` change; MSRV stays 1.81. Confirm at plan time whether any `linalg`/faer path is needed (PCR/PLS/ridge solves route through existing `linalg`).

</code_context>

<specifics>
## Specific Ideas

- **Recovery-test design (critical — from project memory):** the `wcr` β(t)-recovery and `wnet` sparse-support/β(t)-recovery synthetic tests MUST use an n≫m spanning, full-rank predictor design (pseudo-random curves), NEVER phase-shifted single-frequency sinusoids (those span only a 2-D subspace and make recovery tests silently pass/fail). Do not loosen tolerance to make a rank-deficient design pass — fix the design.
- **`wnet` determinism (SC3):** CV-λ must be identical across runs — fixed fold partition + fixed seed. Assert equality across two runs in a test.
- **`wnet` sparse recovery (SC2):** synthetic signal localized in a few wavelet coefficients; assert the selected/nonzero coefficients concentrate on the true support.
- Both regressors: dimension/parameter mismatch → descriptive `FdarError` (never panic); β(t) and fitted values finite / NaN-free (SC4).

</specifics>

<deferred>
## Deferred Ideas

- Out-of-sample `predict`, public coefficient-function/fitted-value accessors, crate-root/prelude re-exports, module doctest → **Phase 71**.
- Binomial/logistic / GLM-family wavelet regression (refund `family`) → WAV-F1.
- GroupMCP/GroupSCAD-style nonconvex penalties for `wnet` → out of scope (only convex L1+L2 this milestone).

</deferred>
