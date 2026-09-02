# Phase 57 — Summary 57-01: Shapelet Distance Core

**Status:** Complete. Requirements SHP-01, SHP-02 delivered.

## Files changed
- `fdars-core/src/shapelet/mod.rs` (new) — submodule barrel: `//!` header, `pub mod distance;`, re-exports `{shapelet_distance, z_normalize_into, z_normalize_window, Shapelet}` within the crate.
- `fdars-core/src/shapelet/distance.rs` (new, ~500 lines with tests) — z-norm + sdist + `Shapelet`.
- `fdars-core/src/lib.rs` — one additive line: `pub mod shapelet;` (alphabetical, after `seasonal`). No crate-root flat re-exports (deferred to Phase 60).

## Public API added
- `pub fn z_normalize_into(src: &[f64], dst: &mut [f64])` — in-place, population std (ddof=0), constant-window guard (std ≤ 1e-12 → zeros), two-pass (numerically stable).
- `#[must_use] pub fn z_normalize_window(slice: &[f64]) -> Vec<f64>` — allocating wrapper.
- `pub struct Shapelet { pub values: Vec<f64>, pub series_idx: usize, pub start: usize, pub length: usize, pub quality: f64 }` — `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]`.
  - `pub fn Shapelet::from_source(series: &[f64], series_idx: usize, start: usize, length: usize) -> Result<Shapelet, FdarError>` (z-normalizes window, quality=0.0).
  - `#[must_use] pub fn Shapelet::len(&self) -> usize`, `#[must_use] pub fn Shapelet::is_empty(&self) -> bool`.
- `#[must_use] pub fn shapelet_distance(shapelet_z: &[f64], series: &[f64], best_so_far: f64) -> Result<(f64, usize), FdarError>` — min z-normalized Euclidean over sliding windows; returns (min distance, first-min offset); early-abandon in squared space; `best_so_far = INFINITY` disables abandon; `InvalidDimension` if shapelet empty or longer than series.

## Tests + results
Inline `#[cfg(test)] mod tests` in `distance.rs` — 7 unit tests + 2 doctests, all pass:
- `test_znorm_constant_window` — constant + 1e-15-perturbed windows → finite zeros.
- `test_znorm_mean_std` — mean≈0, population std≈1.
- `test_sdist_scale_offset_invariant` — `sdist(S,T) == sdist(S,T+c) == sdist(S,T*a)` within 1e-10 (distance and offset). MAKE-OR-BREAK gate.
- `test_sdist_min_semantics` — planted exact motif → sdist≈0 at the correct offset (min, not mean).
- `test_sdist_early_abandon_identical` — tight bound (≥ true min) and exact-min bound both reproduce the INFINITY-bound answer and offset.
- `test_sdist_dimension_error` — shapelet longer than series, and empty shapelet → `Err(InvalidDimension)`.
- `test_shapelet_from_source` — provenance + z-norm values + range errors.
- Doctests on `shapelet_distance` and `z_normalize_window`.

## Divergences / notes
- **ddof=0 (population std)** chosen deliberately (pyts convention). sktime/aeon may use ddof=1; documented in the module `//!` header and on `z_normalize_into`/`z_normalize_window`. This is a resolved decision from 57-CONTEXT.md, not an open item.
- **Doctest motif adjustment:** the initial doctest used a linear-ramp motif `[1,2,3]`; because a ramp z-normalizes to the same shape at every offset, the first-min offset was ambiguous. Switched to a non-monotone motif `[1,4,2]` so a single window matches — this validated the min-semantics correctly rather than a spurious offset expectation.
- **`from_source` constructor** added beyond the literal spec (spec asked for "a constructor that z-normalizes a source slice + records provenance") — implemented as `Shapelet::from_source`, quality 0.0, clean seam for Phase 58 to set quality.
- **No new dependency, no version bump** (crate stays 0.32.0). No existing module edited except the single `mod shapelet;` line in `lib.rs`.

## Gate tails
- `cargo fmt --check -p fdars-core` → clean.
- `cargo clippy --all-targets --features linalg,parallel -p fdars-core -- -D warnings` → `Finished` (no warnings).
- `cargo test -p fdars-core --features linalg shapelet` → 7 passed, 0 failed.
- `cargo test -p fdars-core shapelet` (default features) → 7 passed, 0 failed.
- Doctests (both feature sets) → 2 passed, 0 failed.

## Seams left for Phases 58/59/60
- 58: set `Shapelet.quality`; call `shapelet_distance` in the candidate scan feeding the running best as `best_so_far` for cross-window pruning.
- 59: build the n×K feature matrix via `shapelet_distance`; short-series guard already returns `InvalidDimension` at the primitive level.
- 60: crate-root `pub use shapelet::{...}` flat re-exports.
