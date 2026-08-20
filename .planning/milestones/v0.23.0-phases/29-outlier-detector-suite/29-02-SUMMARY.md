---
phase: 29-outlier-detector-suite
plan: "02"
subsystem: outliers
tags: [outliers, fdaoutlier, roahd, sequential-transform, depthgram, functional-boxplot]

requires:
  - phase: 29-01
    provides: iqr_fence shared helper, TvdMss/Muod plumbing + re-export block
  - phase: depth
    provides: functional_boxplot, modified_band_1d, modified_epigraph_index_1d
provides:
  - sequential_transform_outliers(data, &[SeqTransform], SeqTransformConfig) -> Result<SeqTransformOutliers, FdarError>
  - SeqTransform enum { T0, T1, T2, D1, D2 }, SeqTransformConfig, SeqTransformOutliers { per_transform_outliers, union_outliers }
  - depthgram(data, DepthgramConfig) -> Result<DepthgramResult, FdarError>
  - DepthgramConfig, DepthgramResult (six index vectors + shape/magnitude outliers + mbd/mei)
  - private seq_transform_apply helper
  - crate-root re-exports of all new symbols
affects: [phase-30-interval-testing]

actuals:
  tokens: 34000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Cumulative transform pipeline: a single mutated `current` FdMatrix threaded through the sequence; functional_boxplot per step; union = sort_unstable + dedup of all per-step flags (fdars convenience)"
    - "depthgram p=1: MBD/MEI of the sample, then MBD-of-MEI and MEI-of-MBD via n×1 matrix wrapping (reuse modified_band_1d / modified_epigraph_index_1d)"
    - "depthgram shape outliers via the roahd parabola (a2=a0=-2/(n(n-1)), a1=2(n+1)/(n-1)) + shared iqr_fence UPPER fence; magnitude via functional_boxplot on the MBD column"

key-files:
  modified:
    - fdars-core/src/outliers.rs (SeqTransform enum, SeqTransformConfig/Outliers, seq_transform_apply, sequential_transform_outliers, DepthgramConfig/Result, depthgram + 6 inline tests)
    - fdars-core/src/lib.rs (pub use outliers::{...} extended)

key-decisions:
  - "Transforms apply CUMULATIVELY (each step feeds the next) — [T1, D1] differences the centered data, not the raw data"
  - "union_outliers is an fdars convenience (sorted+deduped over all per-transform sets); the R baseline returns only per-transform sets"
  - "SeqTransformConfig is NOT serde-serializable — it holds a DepthMethod, which does not derive serde (documented); all other new structs derive conditional serde"
  - "depthgram handles univariate (p=1) only; the three representations (_d/_t/_t2) are identical and cloned — documented divergence from roahd's p-variate depthgram"

patterns-established:
  - "Depthgram/outliergram SHAPE-outlier test fixtures must oscillate WITHIN the inlier vertical band (MEI≈0.5, MBD≈0 → large parabola deviation). A wiggle centered on 0 (far below the bundle) gets a high MEI near the parabola edge → small deviation → caught by the magnitude detector, not the shape detector"

requirements-completed: [OUT-01]
---

# Phase 29 Plan 02 — Summary

## Accomplishments

- **`sequential_transform_outliers`**: applies a `&[SeqTransform]` sequence cumulatively (T0 identity,
  T1 centering, T2 L2-normalization, D1/D2 lag-1 differencing), runs `functional_boxplot` after each
  step, and returns per-transform flags plus a unioned set. T2 zero-norm → `ComputationFailed`; D1
  with `m<2` → `InvalidDimension`; `n<2` guard.
- **`depthgram`** (roahd depthGram, univariate): MBD/MEI of the sample, then the `(MBD-of-MEI,
  MEI-of-MBD)` index pairs via n×1 matrix wrapping. Shape outliers from the outliergram-parabola
  upper IQR fence (shared `iqr_fence`); magnitude outliers from a functional boxplot on MBD. The
  three p=1 representations are identical. Returns `DepthgramResult` with six index vectors + flags.
- Crate-root re-exports; existing detectors + DEPTH-01 signatures untouched; no new dependency.

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib outliers::` → **45 passed, 0 failed**.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- `cargo build -p fdars-core --features serde` → **clean** (conditional-serde derives).
- `fmt --check` clean.

## Debugging note

The depthgram shape test initially failed: the shape outlier `0.5·sin(12πx)` oscillates around 0,
far below the inlier bundle, so its MEI was ≈0.91 (near the parabola edge where the deviation is
small) and it was caught by the magnitude detector instead. Fixed by centering the wiggle on the
inlier trend (`sin(πx) + 0.4·sin(12πx)`), giving MEI≈0.5 and MBD≈0 → a large parabola deviation
that clears the shape fence. Recorded as a reusable outliergram-fixture lesson.

## Phase 29 complete

OUT-01 delivered: all four detectors (`tvdmss`, `muod` from 29-01; `sequential_transform_outliers`,
`depthgram` here) are `Result`-returning, numeric-output, crate-root re-exported — additive/
non-breaking, no new dependency. `tvdmss` consumes Phase 28's `TvdMssResult`; the suite reuses the
existing `functional_boxplot` / MBD / MEI machinery.
