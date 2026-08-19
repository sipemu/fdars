---
phase: 28-depth-measure-long-tail
plan: "01"
subsystem: depth
tags: [depth, roahd, hypograph-index, epigraph-index, half-region-depth, fdmatrix, rayon]

requires:
  - phase: depth
    provides: DepthMethod enum + functional_depth dispatcher, band_1d/modified_epigraph_index_1d (MEI) analogs, iter_maybe_parallel!
provides:
  - hypograph_index_1d(data_obj, data_ori) -> Result<Vec<f64>, FdarError> — HI (global indicator, roahd)
  - epigraph_index_1d(...) -> Result<Vec<f64>, FdarError> — EI (un-modified, global indicator, roahd)
  - modified_hypograph_index_1d(...) -> Result<Vec<f64>, FdarError> — MHI (pointwise average, roahd)
  - half_region_depth_1d(...) -> Result<Vec<f64>, FdarError> — HRD = min(EI, HI), fused single pass (roahd)
  - modified_half_region_depth_1d(...) -> Result<Vec<f64>, FdarError> — MHRD = min(MEI, MHI) (roahd)
  - DepthMethod variants: HypographIndex, ModifiedHypographIndex, EpigraphIndex, HalfRegion, ModifiedHalfRegion (parameter-free, non_exhaustive additive)
  - crate-root re-exports of all 5 functions
affects: [phase-28-02, phase-28-03, phase-29-outlier-detector-suite]

actuals:
  tokens: 30000
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - "Result-returning depth measure with dimension guards at entry (diverges from legacy bare-Vec _1d functions)"
    - "Global-indicator measures (HI/EI): iter_maybe_parallel! outer loop + 'outer label continue on first crossing"
    - "Half-region composite HRD: FUSED single pass tracking j_above/j_below flags to avoid nested parallelism (Pitfall 2)"
    - "MHRD composes shipped MEI (infallible Vec) + MHI (Result) via sequential zip so internally-parallel calls don't nest"
    - "MEI <=/>= 0.5 tie convention reused uniformly across index measures"

key-files:
  created:
    - fdars-core/src/depth/hypo_epi.rs (433 lines — HI/MHI/EI + 17 tests)
    - fdars-core/src/depth/half_region.rs (287 lines — HRD/MHRD + 7 tests)
  modified:
    - fdars-core/src/depth/dispatch.rs (5 new DepthMethod variants + match arms with n<2 guards; existing variants untouched)
    - fdars-core/src/depth/mod.rs (pub mod + pub use for both new files)
    - fdars-core/src/lib.rs (crate-root re-exports)

key-decisions:
  - "New measures return Result<Vec<f64>, FdarError> with entry validation (satisfies DEPTH-01 criteria #1/#4), diverging from the legacy bare-Vec _1d convention"
  - "HI/EI/HRD require n>=2 reference curves (global indicators undefined for a single reference); MHI/MHRD accept n>=1 at function level, but the dispatcher guards both HalfRegion/ModifiedHalfRegion at n<2"
  - "MHI is a one-sided monotone INDEX, not a depth: the top curve maximizes it and the central curve sits at ~0.5 — corrected an executor test that wrongly asserted central-maximal"
  - "A far magnitude outlier's MHRD floors at ~1/n (self-comparison satisfies MEI's <=), so it is shallow but not uniquely the global minimum — the lowest boundary inlier ties near 0"

patterns-established:
  - "Additive DepthMethod extension: new parameter-free variants on the #[non_exhaustive] enum, existing 4 variants + dispatcher signature unchanged"
  - "Tracer-first: one measure family (hypo_epi) wired fully end-to-end and verified green before the composite half_region followed the proven path"

requirements-completed: [DEPTH-01]
---

# Phase 28 Plan 01 — Summary

## Accomplishments

- **HI / MHI / EI** (`depth/hypo_epi.rs`): the roahd hypograph index, modified-hypograph index,
  and un-modified epigraph index — complementing the already-shipped MEI. HI/EI are global
  indicators (one crossing excludes a reference); MHI is a pointwise average. All parallelized
  over objects with `iter_maybe_parallel!`.
- **HRD / MHRD** (`depth/half_region.rs`): half-region depth = `min(EI, HI)` computed in a fused
  single pass (avoids nested parallelism), and modified half-region depth = `min(MEI, MHI)`
  composing the shipped modified indices via a sequential zip.
- **Dispatcher wiring**: 5 new parameter-free `DepthMethod` variants (`HypographIndex`,
  `ModifiedHypographIndex`, `EpigraphIndex`, `HalfRegion`, `ModifiedHalfRegion`) with `n<2` guards
  mirroring the Band arm; existing variants and the `functional_depth` signature untouched
  (`#[non_exhaustive]` → additive/non-breaking). Crate-root re-exports added.
- **Tests**: 24 new inline tests (17 hypo_epi + 7 half_region) asserting monotonicity/ordering
  (central curve deepest by HRD, MHI monotone in curve height), the `min`-composite identities,
  dispatch round-trips, HI/EI `k/n` quantization, and error paths (empty / single-curve).

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib depth::` → **112 passed, 0 failed**.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- `cargo fmt -- --check` → clean.
- No new crate dependency; existing depth-measure and `DepthMethod` public signatures unchanged.

## Recovery note

The wave-1 executor subagent stalled (stream watchdog, 600s) after writing the tracer code but
before committing. The orchestrator salvaged the verified-green tracer, committed it, then
completed Task 2 (half_region) inline. This also surfaced and fixed a stale-incremental-cache
clippy gap (`manual_range_contains` in the executor's hypo_epi test module) and two incorrect
executor/plan test expectations about MHI/MHRD behavior.

## Handoff to Plan 02 / 03 / Phase 29

- Plans 28-02 (extremal/ERL/L∞) and 28-03 (TVD+MSSI) extend the same additive pattern on the
  `#[non_exhaustive]` `DepthMethod` enum.
- The `hypo_epi` / `half_region` functions are the reusable analogs for the remaining measures.
