---
phase: 29-outlier-detector-suite
plan: "01"
subsystem: outliers
tags: [outliers, fdaoutlier, tvdmss, muod, iqr-fence, functional-boxplot, tvd-mssi]

requires:
  - phase: 28-depth-measure-long-tail
    provides: total_variation_depth_1d -> TvdMssResult { tvd, mss } (the tvdmss dependency)
  - phase: depth
    provides: functional_boxplot (stage-2 magnitude detection), modified_band_1d
  - phase: helpers
    provides: quantile_sorted, sort_nan_safe (iqr_fence)
provides:
  - tvdmss(data, TvdMssConfig) -> Result<TvdMssOutliers, FdarError> — two-stage TVD+MSSI detector
  - muod(data, MuodConfig) -> Result<MuodResult, FdarError> — Fast-MUOD regression-vs-mean detector
  - TvdMssOutliers { magnitude_outliers, shape_outliers, tvd, mss }, TvdMssConfig (1.5/1.5/0.5 defaults)
  - MuodResult { shape/magnitude/amplitude _outliers + _index }, MuodConfig (factor 1.5)
  - private iqr_fence(values, factor) -> (lower, upper) shared helper; private muod_indices
  - crate-root re-exports of all four public symbols
affects: [phase-29-02]

actuals:
  tokens: 33000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Shared private iqr_fence (Q1/Q3 via quantile_sorted + sort_nan_safe): lower fence for tvdmss shape stage, upper fence per index for muod"
    - "tvdmss stage 2 reuses functional_boxplot on a reduced (non-shape) FdMatrix rebuilt column-major, outliers re-mapped via a keep-vec"
    - "Fast-MUOD: per-curve OLS on the pointwise mean (iter_maybe_parallel!), degenerate-variance 1e-15 guards → no NaN"

key-files:
  modified:
    - fdars-core/src/outliers.rs (iqr_fence, TvdMssConfig/Outliers, tvdmss, MuodConfig/Result, muod_indices, muod + 6 inline tests)
    - fdars-core/src/lib.rs (pub use outliers::{...} extended)

key-decisions:
  - "tvdmss consumes Phase 28's pinned TvdMssResult directly — no TVD/MSS reimplementation (the hard cross-phase dependency)"
  - "Divergence documented: fdaoutlier scales the stage-2 central region by n_orig/n_reduced; fdars' functional_boxplot fixes it at the deepest 50%, so central_region_tvd is informational only"
  - "muod is the Fast-MUOD variant (regression vs pointwise mean), not R's pairwise C++ block; only the boxplot cutoff (tangent deferred to backlog) — documented in rustdoc"
  - "Config structs are constructable (Default + public fields, NOT #[non_exhaustive]); result structs ARE #[non_exhaustive] with conditional serde"

patterns-established:
  - "Outlier-detector test fixtures MUST give inliers genuine per-curve shape variation — a constant-offset sinusoid family makes all derivatives identical, so MSS ranking collapses to floating-point noise and a magnitude outlier's cancellation error spuriously flags it as a shape outlier (fixed by adding a curve-specific secondary harmonic)"

requirements-completed: [OUT-01]
---

# Phase 29 Plan 01 — Summary

## Accomplishments

- **`tvdmss`** (tracer): the fdaoutlier two-stage TVD+MSSI detector. Stage 1 flags shape outliers
  where MSS is below the lower IQR fence and below the mean; stage 2 removes those and runs
  `functional_boxplot` on the remaining curves for magnitude outliers, re-mapping to original
  indices. Consumes Phase 28's `total_variation_depth_1d` directly. Returns `TvdMssOutliers`.
- **`muod`**: Fast-MUOD via per-curve OLS regression on the pointwise mean — shape/magnitude/
  amplitude indices flagged by the upper IQR boxplot whisker, with degenerate-variance guards.
  Returns `MuodResult`.
- **Shared `iqr_fence`** private helper (Q1/Q3 via `quantile_sorted`), used with the lower fence by
  tvdmss and the upper fence by muod; reused by Plan 02.
- Crate-root re-exports; existing detectors + DEPTH-01 signatures untouched; no new dependency.

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib outliers::` → **40 passed, 0 failed**.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**; `fmt` clean.

## Debugging note

The tvdmss magnitude test initially failed: the fixture used constant-offset sinusoids, giving every
curve identical first differences. MSS ranking then reduced to floating-point rounding noise, and the
`+10` magnitude outlier's catastrophic-cancellation error pushed its derivative ranks to an extreme →
spuriously low MSS → misclassified as a *shape* outlier (so stage 2 never saw it). Fixed by giving
inliers a genuine per-curve secondary harmonic so MSS is well-defined; the magnitude outlier then has
a normal shape and is correctly caught by stage 2. Recorded as a reusable test-fixture lesson.

## Handoff to Plan 02

- Plan 29-02 adds `sequential_transform_outliers` (T0/T1/T2/D1 + functional_boxplot base) and
  `depthgram` (MBD/MEI parabola, upper fence), **reusing this plan's `iqr_fence`** helper.
