# Phase 58: Discovery & Ranking - Context

**Gathered:** 2026-09-02
**Status:** Ready for planning
**Mode:** Smart-discuss (autonomous) — resolved from `.planning/research/` + Phase 57 API. No open user decisions. HIGHEST-risk phase of the milestone (most pitfalls).

<domain>
## Phase Boundary

Deliver shapelet DISCOVERY: from a labeled training curve set, generate candidate subsequences, score them by discriminative quality, and select a non-redundant top-K `ShapeletSet`. New `src/shapelet/discovery.rs` (+ types in `src/shapelet/mod.rs`). Builds on Phase 57's `shapelet_distance`/`Shapelet`/`z_normalize_window`. Additive/non-breaking, no new dependency. Crate-root re-exports still deferred to Phase 60.

In scope (SHP-03/04/05):
- **Candidate generation** — enumerate subsequences across a length range, exhaustively OR via deterministic contracted/random sampling bounded by `max_candidates` (seeded).
- **Quality scoring** — information gain on the optimal distance-split threshold (default), or F-statistic (`QualityMeasure` enum).
- **Selection** — top-K by quality with self-similarity pruning → `ShapeletSet`.

Out of scope: the transform (Phase 59), the bundled classifier (Phase 60), crate-root re-exports (Phase 60).
</domain>

<decisions>
## Implementation Decisions

### Resolved (from research — treat as fixed)

1. **Config:** `pub struct ShapeletDiscoveryConfig { min_length: usize, max_length: usize, max_candidates: Option<usize>, max_shapelets: usize, quality: QualityMeasure, seed: u64 }` — Debug/Clone/PartialEq, serde-gated, `Default`. Defaults follow sktime: `max_candidates = Some(10_000)` (None = exhaustive), `max_shapelets = min(10·n, 1000)` computed at fit if left at a sentinel, `min_length = 3`, `max_length = series length` (clamp), `seed = 0`. `pub enum QualityMeasure { InfoGain, FStatistic }` (`#[non_exhaustive]`, default `InfoGain`).

2. **Candidate generation (SHP-03):** enumerate `(series_idx, start, length)` triples for lengths in `[min_length, max_length]`. If `max_candidates` is `Some(m)` and the exhaustive count exceeds `m`, RANDOM-SAMPLE `m` triples deterministically using `seed_for_thread(seed, k)` (from `helpers.rs`) — reproducible. Exhaustive when `None` or count ≤ m. Each sampled triple → a z-normalized `Shapelet` via `Shapelet::from_source`.

3. **Distance orderline + quality (SHP-04):** for a candidate, compute `sdist` to every training series (Phase 57 `shapelet_distance`, `best_so_far = f64::INFINITY` for exact quality — early-abandon is a discovery-time speedup where a quality lower-bound allows it, but correctness first) → an n-vector of distances (the "orderline").
   - **InfoGain:** sort the (distance, label) pairs; scan split thresholds at midpoints between consecutive distinct distances; `IG(θ) = H(all) − (|L|/n·H(L) + |R|/n·H(R))` with Shannon entropy over class proportions; `quality = max_θ IG(θ)`. O(n log n) per candidate.
   - **FStatistic:** one-way ANOVA F-stat of the distance values grouped by class label (between-group / within-group mean squares). A small scalar helper on the 1-D distance vector — the `pub(crate) integrated_f_statistic` in `function_on_scalar.rs` is for `FdMatrix` inputs; write a scalar `f_statistic_1d(distances, labels)` (documented as the 1-D analogue). Handles class imbalance better.

4. **Selection + self-similarity pruning (SHP-05):** sort scored candidates by quality descending, tie-break deterministically by `(series_idx, start, length)` using `total_cmp` on the f64 quality. Greedily select top candidates; when selecting a shapelet from series `i` spanning `[start, start+length)`, DISCARD any not-yet-selected candidate from the SAME series `i` whose `[start', start'+length')` overlaps it (pyts rule: keep only candidates whose range is fully before or after). Stop at `max_shapelets`. Result: `pub struct ShapeletSet { shapelets: Vec<Shapelet>, quality: QualityMeasure }` (each `Shapelet.quality` set to its score) — Debug/Clone/PartialEq, serde-gated, `#[non_exhaustive]`, accessors (`shapelets()`, `len()`).

5. **Public API:** `pub fn discover_shapelets(data: &FdMatrix, labels: &[usize], config: &ShapeletDiscoveryConfig) -> Result<ShapeletSet, FdarError>` (`#[must_use]`). Validation: labels length == n_rows, ≥2 classes, `min_length ≤ max_length ≤ ncols`, `max_length ≥ 1`, `max_shapelets ≥ 1` → `FdarError::{InvalidDimension, InvalidParameter}`.

6. **Parallelism + determinism:** parallelize the candidate-scoring loop with `iter_maybe_parallel!` (per-candidate; each candidate's sdist orderline is independent) — gated by `parallel`. Determinism: the candidate SET is fixed by the seed before scoring; scoring is pure; final sort uses `total_cmp` + the `(series_idx,start,length)` tie-break, so the selected `ShapeletSet` is byte-identical across runs with the same seed AND identical sequential-vs-parallel. Use `row_to_buf` to extract each series row contiguously before scanning.
</decisions>

<code_context>
## Existing Code Insights
- Phase 57 API (`src/shapelet/distance.rs`): `shapelet_distance(shapelet_z, series, best_so_far) -> Result<(f64,usize), FdarError>`, `Shapelet::from_source(series, series_idx, start, length) -> Result<Shapelet,_>`, `z_normalize_window`.
- `src/helpers.rs`: `pub(crate) fn seed_for_thread(seed, k) -> StdRng` — for deterministic candidate sampling.
- `src/function_on_scalar.rs:766`: `pub(crate) fn integrated_f_statistic(data: &FdMatrix, groups, labels) -> f64` — FdMatrix analogue; write a scalar `f_statistic_1d` for the 1-D distance orderline (document the relationship).
- `src/parallel.rs`: `iter_maybe_parallel!` (candidate-level parallelism).
- `src/matrix.rs`: `row_to_buf`; `src/error.rs`: `FdarError`.
- Conventions: config structs + `Default`, `#[must_use]`, serde-gated derives, `Result<_,FdarError>`.
</code_context>

<specifics>
## Specific Ideas (verification hooks — from PITFALLS.md; Phase 58 carries the most)
Tests the plan must include:
- `test_discover_known_motif`: synthetic 2-class dataset with a PLANTED discriminative subsequence in class A only → discovery recovers a shapelet matching that motif's location/shape (the key end-to-end gate).
- `test_discover_tractable_contracted`: with `max_candidates=Some(m)`, discovery on n=100, m_series=200 completes quickly (<10s) and returns ≤ max_shapelets shapelets.
- `test_infogain_optimal_split`: on a hand-constructed distance orderline with a clean class separation, IG picks the correct threshold and quality is maximal (≈ entropy of the prior).
- `test_fstatistic_measure`: `QualityMeasure::FStatistic` runs and ranks a clearly-discriminative candidate above a noise candidate.
- `test_self_similarity_pruning`: selected shapelets from the same series do not overlap; the set has no redundant near-duplicate columns.
- `test_discover_deterministic`: two runs with the same seed produce byte-identical `ShapeletSet` (sampling + tie-break determinism); sequential and `parallel` agree.
- `test_discover_validation`: <2 classes, label/row mismatch, min>max length, max_length>ncols → errors.
- Doctest on `discover_shapelets`.
</specifics>

<deferred>
## Deferred Ideas
- The transform (apply ShapeletSet → n×K) → Phase 59.
- Bundled classifier → Phase 60; crate-root re-exports → Phase 60.
- Quality lower-bound pruning / entropy early-abandon during discovery (a speedup that preserves the exact top-K) — optional Phase 58 refinement or later perf pass; correctness (full orderline) first.
</deferred>
