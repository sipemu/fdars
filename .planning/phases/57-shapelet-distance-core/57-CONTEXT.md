# Phase 57: Shapelet Distance Core - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — grey areas resolved from `.planning/research/` (SUMMARY/FEATURES/ARCHITECTURE/PITFALLS). No open user decisions.

<domain>
## Phase Boundary

Deliver the numerical shapelet-distance foundation that every downstream phase (58 discovery, 59 transform, 60 classifier) depends on. New `src/shapelet/` submodule directory; this phase creates it with the distance-core file. Additive/non-breaking, no new dependency. Crate-root `pub mod shapelet` re-exports are DEFERRED to Phase 60 (avoid partial public API exposure) — this phase's items are `pub` within the submodule and `pub(crate)` where only internal.

In scope (SHP-01/02):
- Per-window **z-normalization** (population std, ddof=0) with a constant-window guard.
- **`sdist`** = min over sliding windows of ‖z(window) − z(shapelet)‖₂, with explicit `best_so_far` early-abandon.
- The **`Shapelet`** type (stores the z-normalized values + provenance: source series index, start offset, length).

Out of scope: candidate discovery/ranking (Phase 58), the transform (Phase 59), the classifier (Phase 60), crate-root re-exports (Phase 60).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research — treat as fixed)

1. **Module layout:** new `src/shapelet/` directory. This phase adds `src/shapelet/mod.rs` (submodule barrel — `pub mod distance;` / re-exports within the crate) and `src/shapelet/distance.rs` (this phase's code). Add `mod shapelet;` to `src/lib.rs` as a **`pub(crate)`/private `mod`** for now OR add `pub mod shapelet;` but DO NOT yet re-export items at the crate root — the public re-export surface is finalized in Phase 60. (Simplest: `pub mod shapelet;` so downstream phases and tests can reach it, but keep the crate-root `pub use shapelet::{…}` flat re-exports for Phase 60.) Zero edits to any existing module file except adding the `mod shapelet;` line to `lib.rs`.

2. **z-normalization (SHP-01):** `z_normalize_window(slice: &[f64]) -> Vec<f64>` (and/or an in-place `_into(&[f64], &mut [f64])` to avoid per-window allocation in the hot loop). Subtract mean, divide by **population** std (ddof=0). **Constant-window guard:** if std ≤ a small epsilon (e.g. 1e-12), return the zero vector (all zeros) — never divide by ~0, never emit NaN/Inf. Document the ddof=0 choice (matches pyts; sktime divergence noted).

3. **sdist (SHP-02):** `shapelet_distance(shapelet_z: &[f64], series: &[f64], best_so_far: f64) -> (f64, usize)` returning (min normalized Euclidean distance, best-match start offset). The shapelet is ALREADY z-normalized (stored that way); each length-L window of `series` is z-normalized on the fly, then squared-Euclidean distance accumulated. **Early-abandon:** accumulate the running squared sum inside the inner element loop and `break` as soon as it exceeds `best_so_far` (compare in squared space; take sqrt only for the returned min). A distance ≤ `best_so_far` updates the running best. Passing `best_so_far = f64::INFINITY` disables abandon. Distance metric = plain Euclidean over z-normalized values (NOT the weighted/Simpson L2 in `helpers.rs`/`metric/lp.rs` — those are wrong semantics here).

4. **`Shapelet` type:** `pub struct Shapelet { pub values: Vec<f64> /* z-normalized */, pub series_idx: usize, pub start: usize, pub length: usize, pub quality: f64 }` (quality set in Phase 58; 0.0 here). Derive `Debug, Clone, PartialEq`; serde-gated. Provides `len()`; construction z-normalizes the source slice.

5. **Hot-path / layout:** `FdMatrix` rows are NON-contiguous (column-major) — callers must `row_to_buf` a series row into a contiguous buffer before the window scan; z-norm and distance operate on `&[f64]`. Early-abandon requires a SEQUENTIAL inner loop (no parallelism inside the window scan). No parallelism in this phase (the parallel candidate scan is Phase 58, at the series/candidate level).

6. **Validation / errors:** public fallible entry points return `Result<_, FdarError>`; a shapelet longer than the series → `FdarError::InvalidDimension`. Pure helpers (`z_normalize_window`) may return the value directly (guard handles degenerate input).
</decisions>

<code_context>
## Existing Code Insights
- `src/matrix.rs`: column-major `FdMatrix`, `row_to_buf(i, &mut buf)` for contiguous row extraction; `row_l2_sq`.
- `src/helpers.rs`: has weighted/Simpson L2 (DO NOT use for sdist); `seed_for_thread` (Phase 58, not here).
- `src/metric/`: `soft_dtw.rs`/`lp.rs` — integration-weighted; NOT the discrete z-normalized Euclidean shapelets need. Shapelets get a fresh distance.
- `src/error.rs`: `FdarError::{InvalidDimension, InvalidParameter}`.
- Conventions: `#[must_use]` on expensive fns, `Debug,Clone,PartialEq` + serde-gated derives, `///` docs with the math, module `//!` header.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md)
Tests the plan must include (inline `#[cfg(test)] mod tests`):
- `test_znorm_constant_window`: constant window → all-zeros, finite (no NaN) — the std≈0 guard.
- `test_znorm_mean_std`: z-normalized slice has mean≈0, population std≈1 (for non-constant input).
- `test_sdist_scale_offset_invariant`: `sdist(S, T)` == `sdist(S, a·T + b)` within 1e-10 for a>0 — THE make-or-break gate (per-window z-norm).
- `test_sdist_min_semantics` / known-motif: a series containing an exact copy of the (pre-normalization) shapelet motif yields sdist ≈ 0 at the correct offset; proves MIN (not mean/sum) and correct offset.
- `test_sdist_early_abandon_identical`: `shapelet_distance(.., best_so_far=INF)` == the abandoned-search min for a tight `best_so_far` (same answer; abandon only prunes).
- `test_sdist_dimension_error`: shapelet longer than series → `Err(InvalidDimension)`.
- Doctest on `shapelet_distance` or `z_normalize_window`.
</specifics>

<deferred>
## Deferred Ideas
- Candidate generation, quality scoring, self-similarity pruning → Phase 58.
- Parallel candidate scan (`iter_maybe_parallel!` at series/candidate level), seeded sampling → Phase 58.
- Crate-root `pub use shapelet::{…}` re-exports → Phase 60.
- SIMD / cache-blocking of the window scan — a later perf pass.
</deferred>
