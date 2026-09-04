# Phase 69: Discrete Wavelet Transform Primitive - Context

**Gathered:** 2026-09-04
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — recommendations aligned to locked v0.37.0 STATE.md decisions

<domain>
## Phase Boundary

Deliver a reusable in-crate orthogonal discrete wavelet transform (DWT) primitive that the Phase 70 wavelet-domain regressors (`wcr`, `wnet`) build on. Scope:

- Forward (decompose) and inverse (reconstruct) orthogonal DWT with perfect reconstruction.
- Wavelet families: Haar (db1) + Daubechies db2–db10 (hardcoded filter-coefficient tables).
- Multi-level (Mallat pyramid) decomposition.
- Selectable boundary handling: periodic (default) and symmetric extension.
- Arbitrary (non-power-of-2) signal lengths.
- Descriptive `FdarError` on invalid inputs — never panic, never emit NaN.

Out of scope for this phase: the regressors themselves (Phase 70), prediction/exports (Phase 71), Symlets/Coiflets/biorthogonal families and wavelet packets (WAV-F2), 2D/surface DWT (WAV-F3).

</domain>

<decisions>
## Implementation Decisions

### DWT Primitive Design
- **Wavelet family / order range:** Haar (db1) + Daubechies **db2–db10**, filter coefficients from hardcoded tables (no runtime computation, no new dependency). db2–db20 and full family sets deferred to WAV-F2.
- **Default boundary mode:** **Periodic** (orthogonal, exact perfect reconstruction, standard DWT default). **Symmetric** extension also selectable; both modes must independently satisfy the round-trip gate on non-power-of-2 lengths.
- **Public API shape:** slice-based `decompose(&[f64], ...)` / `reconstruct(...)` pair returning a coefficient result struct holding per-level detail + final approximation coefficients, **plus** a batch path over `FdMatrix` rows so the Phase 70 regressors transform a whole curve set without re-looping externally.
- **Max decomposition level (when caller omits it):** **auto** = `floor(log2(n / (filter_len − 1)))` (standard max-useful depth); caller may also pass an explicit level, validated against this maximum.

### Claude's Discretion
- Module layout (single `wavelet.rs` vs a `wavelet/` directory with `filters.rs`) — planner's call based on file size; STATE.md notes both are acceptable. Peer of `kshape.rs` / `kernel_kmeans.rs`.
- Exact names of the config enum(s) (family, boundary mode) and the coefficient result struct — follow existing crate enum/result conventions (`CovType`, `ProjectionBasisType`, `#[non_exhaustive]` result structs).
- Internal storage layout of per-level coefficients (concatenated vs per-level `Vec`s) provided reconstruction is exact and the type is ergonomic for the regressors.
- Crate-root / prelude re-exports are intentionally deferred to Phase 71 (avoid exposing a partial public surface mid-milestone) — Phase 69 keeps the module reachable internally.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/error.rs` — `FdarError` (`InvalidDimension`, `InvalidParameter`, `ComputationFailed`, `InvalidEnumValue`); all public fns return `Result<T, FdarError>`, validate at entry, never panic on input.
- `src/matrix.rs` — `FdMatrix` column-major (`data[i + j*nrows]`), row helpers (`row_to_buf`, `column(j)` zero-copy slice); rows = curves/observations, columns = evaluation points.
- `src/helpers.rs` — numerical helpers (e.g. `simpsons_weights`); house style for small math utilities.
- The crate has `rustfft` and a Morlet CWT (`seasonal/strength.rs`) but **no** discrete orthogonal wavelet transform — this is genuinely new code.

### Established Patterns
- Config via builder-style structs / enums replacing bare ints (`ProjectionBasisType`, `CovType`); enums use `#[non_exhaustive]`, convert via `FdarError::InvalidEnumValue`.
- Result structs derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, optional serde behind the `serde` feature (keep new wavelet types serde-clean or serde-gated).
- `#[must_use]` on expensive computations; `#[inline]` on hot matrix row accessors.
- Inline unit tests via `#[cfg(test)] mod tests { ... }` in the same file; shared `test_helpers::uniform_grid(n)`.

### Integration Points
- Consumed in Phase 70 by `wcr`/`wnet` (curves → wavelet coefficients) — the `FdMatrix`-batch path is that seam.
- No `Cargo.toml` change; MSRV stays 1.81 (no `linalg`/faer path expected for the DWT itself — confirm at plan time).

</code_context>

<specifics>
## Specific Ideas

- Perfect-reconstruction round-trip (≤1e-10 relative) is the make-or-break numerical gate — front-loaded here before the regressors depend on it. Known-answer tests required: Haar single-level = hand-computed sum/difference over √2; round-trip for Haar + db2–db10 across multiple levels; both boundary modes on non-power-of-2 lengths.
- Numeric-output parity with a standard orthogonal DWT is the goal, not basis-internal parity with refund's `wavethresh`/`wmtsa` internals.

</specifics>

<deferred>
## Deferred Ideas

- Symlets / Coiflets / biorthogonal families + wavelet packets → WAV-F2.
- 2D / surface DWT → WAV-F3.
- Daubechies orders beyond db10 (db11–db20) → WAV-F2 if demand arises.

</deferred>
