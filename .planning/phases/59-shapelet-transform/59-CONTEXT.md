# Phase 59: Shapelet Transform - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` + Phase 57/58 API. No open user decisions.

<domain>
## Phase Boundary

Deliver the shapelet TRANSFORM: apply a fitted `ShapeletSet` to a curve set → an n×K distance-feature matrix, for both training and out-of-sample curves, with transform consistency. New `src/shapelet/transform.rs` (+ types in `mod.rs`). Builds on Phase 57 (`shapelet_distance`) + Phase 58 (`ShapeletSet`, `discover_shapelets`, `ShapeletDiscoveryConfig`). Additive/non-breaking, no new dependency. Crate-root re-exports still deferred to Phase 60.

In scope (SHP-06):
- A **transform** that maps each curve `i` and shapelet `j` to `X[i,j] = sdist(shapelet_j, curve_i)` → an n×K `FdMatrix`.
- A **fit** convenience that discovers (Phase 58) then transforms the training set, storing the `ShapeletSet` for reuse.
- **Out-of-sample transform** applying the SAME stored shapelets + normalization to new curves.
- **Transform consistency** — re-transforming the training data reproduces the fit-time distances.

Out of scope: the bundled classifier (Phase 60), crate-root re-exports (Phase 60).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research + Phase 57/58 API)

1. **Standalone transform:** `pub fn shapelet_transform(shapelets: &ShapeletSet, data: &FdMatrix) -> Result<FdMatrix, FdarError>` (`#[must_use]`) → an n×K `FdMatrix` (n = data rows, K = shapelets.len()), where `X[(i,j)] = shapelet_distance(&shapelets.shapelets()[j].values, row_i, f64::INFINITY).0`. The shapelet `values` are ALREADY z-normalized (Phase 57 provenance) → reuse them directly, no re-normalization. Parallelize the row (or row×shapelet) loop with `iter_maybe_parallel!` (deterministic; distances are order-independent). `row_to_buf` each series row contiguously before scanning.

2. **Fit result:** `pub struct ShapeletTransformFit { shapelets: ShapeletSet, features: FdMatrix }` — Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`; accessors `shapelets() -> &ShapeletSet`, `features() -> &FdMatrix` (the training n×K matrix). Method `transform(&self, new_data: &FdMatrix) -> Result<FdMatrix, FdarError>` = `shapelet_transform(self.shapelets(), new_data)` (out-of-sample, same stored shapelets/normalization).

3. **Fit entry point:** `pub fn shapelet_transform_fit(data: &FdMatrix, labels: &[usize], config: &ShapeletDiscoveryConfig) -> Result<ShapeletTransformFit, FdarError>` (`#[must_use]`) — calls `discover_shapelets` (Phase 58), then `shapelet_transform` on the training data, stores both. This is the fit/transform pipeline STC (Phase 60) will reuse.

4. **Transform consistency (the key gate):** `fit.transform(&train_data)` must reproduce `fit.features()` within 1e-12 — because both go through the identical `shapelet_distance` with the identical stored z-normalized shapelets and `best_so_far = INFINITY`. (Exact-equality is expected; assert within 1e-12 to allow only FP-reassociation noise, which there should be none of since it is the same code path.)

5. **Validation / robustness:** any series shorter than a shapelet → `FdarError::InvalidDimension` (already surfaced by Phase 57's `shapelet_distance`; propagate). Empty `ShapeletSet` (K=0) → `FdarError::InvalidParameter` (a 0-column feature matrix is not useful). All output entries finite (guaranteed by Phase 57's z-norm guard). Column count K == `shapelets.len()`; row count == data rows.

6. **Feature-matrix layout:** the n×K output is a standard `FdMatrix` (column-major); downstream Phase 60 feeds it to an fdars classifier as `data` (rows = observations, cols = the K shapelet-distance features). Document that these columns are shapelet distances, not functional evaluation points.
</decisions>

<code_context>
## Existing Code Insights
- Phase 57 `src/shapelet/distance.rs`: `shapelet_distance(shapelet_z, series, best_so_far) -> Result<(f64,usize),FdarError>` (use `.0` for the distance).
- Phase 58 `src/shapelet/discovery.rs`: `ShapeletSet` (accessor `shapelets() -> &[Shapelet]`, `len()`), `discover_shapelets(data, labels, config)`, `ShapeletDiscoveryConfig`.
- `src/matrix.rs`: `FdMatrix::new`/from-columns/rows constructors (mirror how Phase 54's `gak_gram_matrix` built its output), `row_to_buf`.
- `src/parallel.rs`: `iter_maybe_parallel!`; `src/error.rs`: `FdarError`.
- Conventions: `#[must_use]`, `Debug,Clone,PartialEq` + serde-gated, `Result<_,FdarError>`, doc examples.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md)
Tests the plan must include:
- `test_transform_fit_shape`: `shapelet_transform_fit` on labeled data → features is n×K with K == discovered shapelets, all finite.
- `test_transform_out_of_sample_shape`: `fit.transform(new_data)` → n_new×K (assert both dims; a new set with n_new ≠ n_train catches a transpose).
- `test_transform_consistency`: `fit.transform(&train_data)` reproduces `fit.features()` within 1e-12 (THE key gate).
- `test_transform_values_are_sdist`: for a tiny hand-checked case, X[i,j] equals `shapelet_distance(shapelet_j, curve_i)` exactly.
- `test_transform_short_series_error`: a new series shorter than the longest shapelet → `Err(InvalidDimension)`.
- `test_transform_empty_set_error`: K=0 → `Err(InvalidParameter)`.
- Doctest on `shapelet_transform_fit` (fit → transform new data).
</specifics>

<deferred>
## Deferred Ideas
- Bundled `ShapeletTransformClassifier` (fit→classify, predict) → Phase 60.
- Crate-root `pub use shapelet::{…}` flat re-exports → Phase 60.
- Criterion benchmark of the transform → Phase 60 (with the classifier).
</deferred>
