# Phase 59 · Plan 59-01 — SUMMARY

**Requirement:** SHP-06 · **Milestone:** v0.33.0 · **Status:** complete
**Impl commit:** `6f53cfa0`

## Files
- **Created** `fdars-core/src/shapelet/transform.rs` (~440 lines incl. inline tests).
- **Modified** `fdars-core/src/shapelet/mod.rs` — added `pub mod transform;` + `pub use transform::{shapelet_transform, shapelet_transform_fit, ShapeletTransformFit};`; updated module doc (transform step now shipped). No crate-root re-exports (deferred to Phase 60).

No version bump, no new dependency. Additive/non-breaking. Phase 57/58 public behavior unchanged.

## Public API added
```rust
pub fn shapelet_transform(shapelets: &ShapeletSet, data: &FdMatrix) -> Result<FdMatrix, FdarError>; // #[must_use]

pub struct ShapeletTransformFit { pub shapelets: ShapeletSet, pub features: FdMatrix } // Debug,Clone,PartialEq; serde-gated; #[non_exhaustive]
impl ShapeletTransformFit {
    pub fn shapelets(&self) -> &ShapeletSet;                                  // #[must_use]
    pub fn features(&self) -> &FdMatrix;                                      // #[must_use]
    pub fn transform(&self, new_data: &FdMatrix) -> Result<FdMatrix, FdarError>; // #[must_use]
}

pub fn shapelet_transform_fit(data: &FdMatrix, labels: &[usize], config: &ShapeletDiscoveryConfig)
    -> Result<ShapeletTransformFit, FdarError>; // #[must_use]
```

## Implementation notes
- `shapelet_transform`: `K = shapelets.len()`; `K==0 → InvalidParameter`. Parallel over rows via `iter_maybe_parallel!(0..n)`: each row `row_to_buf`'d once into a contiguous buffer, then scanned against every shapelet with `shapelet_distance(&s.values, &buf, f64::INFINITY).0`. Per-row closures return `Result<Vec<f64>, FdarError>`; the first error is bubbled after collection (deterministic — rows are order-independent). Output assembled column-major (`i + j*n`) into an n×K `FdMatrix`.
- Shapelet `values` reused directly (already z-normalized from Phase 57) — no re-normalization. Short-series → `InvalidDimension` propagated unchanged from `shapelet_distance`.
- `ShapeletTransformFit::transform` = `shapelet_transform(self.shapelets(), new_data)` — identical code path guarantees consistency.
- `shapelet_transform_fit` = `discover_shapelets` → `shapelet_transform` on training data; stores both.

## Tests + results (all green)
Inline `#[cfg(test)] mod tests` — 6 tests + 2 doctests:
- `test_transform_fit_shape` — features n×K, K==discovered, all finite. PASS
- `test_transform_out_of_sample_shape` — `fit.transform` → n_new×K, n_new(=9)≠n_train(=16). PASS
- `test_transform_consistency` — `transform(train)` reproduces `features()` within 1e-12 + two calls bit-identical (`==`). PASS
- `test_transform_values_are_sdist` — tiny hand-checked case, `X[i,j] == shapelet_distance(shapelet_j, curve_i).0` exact equality. PASS
- `test_transform_short_series_error` — series shorter than longest shapelet → `Err(InvalidDimension)`. PASS
- `test_transform_empty_set_error` — K=0 → `Err(InvalidParameter)`. PASS
- Doctests on `shapelet_transform` and `shapelet_transform_fit`. PASS

## Gate tails
- `cargo fmt --check`: clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: 0 warnings/errors.
- `cargo test -p fdars-core --features linalg shapelet`: `20 passed; 0 failed` (lib) + transform doctests ok.
- `cargo test -p fdars-core shapelet` (default features): `20 passed; 0 failed`.

## Divergences
- One doctest assertion in `shapelet_transform` initially used a reversed column-major index (`k/n, k%n`); corrected to explicit `(i, j)` nested loop. No impact on the public API or the required `shapelet_transform_fit` doctest.

## Seams for Phase 60
- `ShapeletTransformFit` is the fit/transform artifact Phase 60's classifier will wrap (discover → transform → classify). Fields are `pub` and `#[non_exhaustive]`.
- Feature matrix is a plain column-major `FdMatrix` (rows=observations, cols=K shapelet-distance features) ready to feed an existing `classification/` classifier as `data`.
- Crate-root `pub use shapelet::{…}` flat re-exports still deferred to Phase 60 (per ROADMAP).
