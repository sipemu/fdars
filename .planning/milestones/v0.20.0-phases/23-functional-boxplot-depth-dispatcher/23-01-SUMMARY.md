---
phase: 23-functional-boxplot-depth-dispatcher
plan: 01
subsystem: depth
tags: [depth, functional-boxplot, dispatcher, outlier-detection, r-parity]
status: complete
requires: []
provides:
  - functional_depth
  - DepthMethod
  - functional_boxplot
  - FunctionalBoxplotResult
affects:
  - fdars-core/src/depth/dispatch.rs
  - fdars-core/src/depth/mod.rs
  - fdars-core/src/lib.rs
tech-stack:
  added: []
  patterns:
    - "Enum-dispatch self-depth wrapper over existing depth fns (additive, non-breaking)"
    - "Depth-fence functional boxplot (Lopez-Pintado-Romo), numeric-only"
key-files:
  created:
    - fdars-core/src/depth/dispatch.rs
  modified:
    - fdars-core/src/depth/mod.rs
    - fdars-core/src/lib.rs
decisions:
  - "DepthMethod is #[non_exhaustive], derives Copy — variants carry per-method params"
  - "functional_boxplot rejects non-finite factor in addition to negative (extra validation)"
  - "Central region = ceil(n/2) deepest rows; ties broken by index for determinism"
metrics:
  duration: ~11m
  completed: 2026-08-16
actuals:
  tokens: 9000
  tasks: 2
  commits: 2
---

# Phase 23 Plan 01: Functional Boxplot & Depth Dispatcher Summary

Delivered T-02: a `functional_depth(data, DepthMethod)` self-depth dispatcher wrapping the existing depth functions, plus the canonical López-Pintado–Romo depth-fence `functional_boxplot` with numeric central-region / whisker / outlier outputs — both additive, `Result`-returning, and crate-root re-exported.

## What Was Built

- **`DepthMethod`** enum (`#[non_exhaustive]`, `Debug/Clone/Copy/PartialEq`): `FraimanMuniz { scale }`, `Band`, `ModifiedBand`, `RandomProjection { nproj, seed }`.
- **`functional_depth(data, method) -> Result<Vec<f64>, FdarError>`**: self-depth (data as both object and reference), dispatching to `fraiman_muniz_1d(data,data,scale)`, `band_1d(data,data)`, `modified_band_1d(data,data)`, `random_projection_1d_seeded(data,data,nproj,Some(seed))`. Validates empty matrix, `<2` curves for band methods, `nproj==0`.
- **`FunctionalBoxplotResult`** struct (`#[non_exhaustive]`, serde-gated, `Debug/Clone/PartialEq`): `median`, `central_lower`, `central_upper`, `whisker_lower`, `whisker_upper`, `outliers: Vec<usize>`, `depths`.
- **`functional_boxplot(data, method, factor) -> Result<FunctionalBoxplotResult, FdarError>`**: rank by `functional_depth` → median = deepest curve → 50% central region = pointwise envelope of deepest `ceil(n/2)` curves → fence = central inflated by `factor×width` → outliers = curves exceeding the fence at any t. Numeric only (no plotting). Validates empty / `<2` curves / negative-or-non-finite factor.
- Crate-root re-exports for all four public items in `lib.rs`.

## Tasks Completed

| Task | Name | Commit | Files |
| ---- | ---- | ------ | ----- |
| 1 (tracer) | DepthMethod + functional_depth dispatcher | 34f56540 | dispatch.rs, depth/mod.rs, lib.rs |
| 2 | functional_boxplot depth-fence boxplot | 01fd6c54 | dispatch.rs, lib.rs |

Tracer feedback gate: the FraimanMuniz thin path (`fraiman_muniz_dispatch_equals_underlying`) was verified end-to-end before expanding to the other variants and the boxplot.

## Tests Added

14 inline tests in `depth::dispatch::tests` (9 dispatcher + 5 boxplot):
- Per-method dispatcher equality vs the underlying self-depth call (FraimanMuniz scale true/false, Band, ModifiedBand, RandomProjection).
- RandomProjection fixed-seed bit-reproducibility (dispatcher and whole boxplot result).
- Error paths: empty matrix, `<2` curves for band, `nproj==0`, negative factor, single-curve boxplot.
- Boxplot: planted gross-outlier flagged and inliers spared; median == deepest curve; central region brackets the median at every t; fence contains the central region at every t.

**Full-suite pass status:** `cargo test -p fdars-core --features linalg,parallel` → **2780 passed, 0 failed** (all test binaries aggregated).

**Clippy status:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean** (also enforced by the pre-commit gate on both commits).

## Existing Signatures Unchanged

Confirmed via grep — the dispatcher only wraps these; none were modified:
- `fraiman_muniz_1d(data_obj, data_ori, scale) -> Vec<f64>`
- `band_1d(data_obj, data_ori) -> Vec<f64>`, `modified_band_1d(data_obj, data_ori) -> Vec<f64>`
- `random_projection_1d_seeded(data_obj, data_ori, nproj, Option<u64>) -> Vec<f64>`
- `outliergram(data, factor) -> Result<OutligramResult, FdarError>` untouched.

## Deviations from Plan

**None functionally.** Two minor beneficial additions (Rule 2 — correctness):
1. `functional_boxplot` rejects a **non-finite** `factor` (NaN/±∞) in addition to negative, preventing NaN whiskers.
2. File placement: both items live in the single new `depth/dispatch.rs` (executor's discretion per 23-CONTEXT); no separate `boxplot.rs`.

## Self-Check: PASSED

- `fdars-core/src/depth/dispatch.rs` exists.
- Commits `34f56540` and `01fd6c54` present in git log.
- Both pre-commit gates (fmt + clippy + full test suite) passed.
