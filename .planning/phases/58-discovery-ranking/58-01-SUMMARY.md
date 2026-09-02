---
phase: 58
plan: 58-01
title: Shapelet Discovery & Ranking (SHP-03/04/05)
status: complete
requirements: [SHP-03, SHP-04, SHP-05]
commit: b18b0a7a
---

# Phase 58 Summary — Discovery & Ranking

## Files
- **Created** `fdars-core/src/shapelet/discovery.rs` (~858 LOC incl. tests) — config/result types, quality measures, candidate generation, fit driver, inline tests + doctest.
- **Modified** `fdars-core/src/shapelet/mod.rs` — `pub mod discovery;` + re-exports; updated module doc.

No crate version bump. No new dependency. No crate-root re-exports (deferred to Phase 60).

## Public API added
```rust
// enum
pub enum QualityMeasure { InfoGain, FStatistic }   // #[non_exhaustive], Default = InfoGain, Copy

// config
pub struct ShapeletDiscoveryConfig {
    pub min_length: usize,          // default 3
    pub max_length: usize,          // default 0 = sentinel → clamp to ncols
    pub max_candidates: Option<usize>, // default Some(10_000)
    pub max_shapelets: usize,       // default 0 = sentinel → min(10*n, 1000)
    pub quality: QualityMeasure,    // default InfoGain
    pub seed: u64,                  // default 0
}
impl Default for ShapeletDiscoveryConfig

// result
pub struct ShapeletSet { pub shapelets: Vec<Shapelet>, pub quality: QualityMeasure } // #[non_exhaustive]
impl ShapeletSet { fn shapelets(&self)->&[Shapelet]; fn len(&self)->usize; fn is_empty(&self)->bool; fn quality(&self)->QualityMeasure }

// entry point
#[must_use]
pub fn discover_shapelets(
    data: &FdMatrix, labels: &[usize], config: &ShapeletDiscoveryConfig,
) -> Result<ShapeletSet, FdarError>
```
All config/result types derive `Debug, Clone, PartialEq` and are serde-gated. Re-exported from `fdars_core::shapelet::{discover_shapelets, QualityMeasure, ShapeletDiscoveryConfig, ShapeletSet}`.

Private helpers: `information_gain(&mut [(f64,usize)], n_classes) -> f64` (optimal-split IG), `f_statistic_1d(&[f64], &[usize], n_classes) -> f64` (documented 1-D analogue of `function_on_scalar::integrated_f_statistic`), `generate_candidates(...)`, `decode_candidate(...)`, `entropy_from_counts(...)`.

## Design notes (SHP-03/04/05)
- **SHP-03 candidate gen:** enumerate `(series_idx,start,length)` for `L∈[min,max]`; exhaustive when `max_candidates=None` or exhaustive_count ≤ m; else deterministic seeded sampling of distinct flat indices via `seed_for_thread(seed,0)` + rejection, decoded back to triples and sorted by `(series_idx,start,length)` so candidate order is seed-fixed before scoring.
- **SHP-04 scoring:** per candidate, an sdist orderline (one `shapelet_distance(_, _, f64::INFINITY)` per training series). InfoGain = incremental left/right Shannon-entropy sweep over midpoints between distinct sorted distances (O(n log n)); F-statistic = one-way ANOVA F with the `ms_within ≤ 1e-15 → 0.0` guard.
- **SHP-05 selection:** rank by quality desc, tie-break `(series_idx,start,length)` via `f64::total_cmp`; greedy select with per-series accepted-interval overlap pruning; stop at resolved `max_shapelets`; each selected `Shapelet.quality` set.
- **Parallelism/determinism:** scoring loop uses `iter_maybe_parallel!` over the fixed candidate index range (per-candidate independent); `#[cfg(feature="parallel")] use rayon::iter::ParallelIterator;` added. Rows pre-extracted via `row_to_buf`. Sequential (default features) result == parallel result — both green.

## Tests + results
All inline `#[cfg(test)] mod tests` (8) + doctest:
- `test_discover_known_motif` — planted triangular motif recovered; top shapelet overlaps motif region, IG > 0.9 (≈ max entropy 1.0). PASS
- `test_discover_tractable_contracted` — n=100, m=200, max_candidates=800 → <10s, ≤ max_shapelets. PASS
- `test_infogain_optimal_split` — clean split → IG == 1.0; degenerate → 0.0. PASS
- `test_fstatistic_measure` — discriminative F-stat > noise F-stat; end-to-end FStatistic path returns scored set. PASS
- `test_self_similarity_pruning` — no same-series overlap among selected. PASS
- `test_discover_deterministic` — over-budget (sampling) same-seed fits byte-identical. PASS
- `test_discover_validation` — <2 classes / label-row mismatch / min>max / max_length>ncols → correct errors. PASS
- Doctest on `discover_shapelets`. PASS

Gate tails:
- `cargo test -p fdars-core --features linalg shapelet` → **14 passed; 0 failed** (lib).
- `cargo test -p fdars-core shapelet` (default features) → **14 passed; 0 failed**.
- `cargo test -p fdars-core --features linalg --doc shapelet` → **3 passed** (incl. discovery doctest).
- `cargo fmt --check` → clean (exit 0).
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → clean (Finished, no warnings).

## Divergences
- The planted-motif test dataset carries a small deterministic per-point shape jitter (survives per-window z-normalization) so the F-statistic path has nonzero within-class variance — otherwise the reference-faithful `ms_within ≤ 1e-15 → 0.0` guard (kept identical to `integrated_f_statistic`) would return F=0 on a pathologically clean perfect separator. Purely a test-fixture choice; no algorithm change.
- Config field names use `min_length`/`max_length`/`max_shapelets` (per 58-CONTEXT decisions), not FEATURES.md's `min_len`/`k_shapelets`.

## Seams for Phases 59/60
- `ShapeletSet.shapelets` holds already-z-normalized `Shapelet` values → Phase 59 transform reuses them directly (no re-normalization); `shapelet_distance(&shp.values, row, INFINITY)` gives the exact fit-time distance.
- `QualityMeasure` is `Copy`/`#[non_exhaustive]` — safe to extend.
- Crate-root `pub mod shapelet` re-exports still deferred to Phase 60.
