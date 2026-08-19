---
phase: 28-depth-measure-long-tail
plan: "03"
subsystem: depth
tags: [depth, fdaoutlier, total-variation-depth, mssi, huang-sun, TvdMssResult, fdmatrix, rayon]

requires:
  - phase: 28-02
    provides: DepthMethod additive-extension pattern, rank_slice/column_ranks helper convention, inline-test conventions
provides:
  - total_variation_depth_1d(data_obj, data_ori) -> Result<TvdMssResult, FdarError> — TVD (magnitude) + MSS (shape), fdaoutlier/Huang-Sun 2019
  - TvdMssResult { tvd: Vec<f64>, mss: Vec<f64> } — PINNED struct (Debug/Clone/PartialEq, #[non_exhaustive], conditional serde), crate-root re-exported
  - DepthMethod variant TotalVariation (projects .tvd, n<3 guard)
  - crate-root re-exports of total_variation_depth_1d + TvdMssResult
  - dispatcher all-9-variant round-trip + 4-existing-variant regression + min-n error-path tests
affects: [phase-29-outlier-detector-suite]

actuals:
  tokens: 26000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "TvdMssResult two-Vec struct mirroring FunctionalBoxplotResult derive/serde/non_exhaustive pattern"
    - "TVD: p=rank/n (÷n, NOT ÷n+1 — fdaoutlier convention), tvd[i]=mean_t p(1-p)"
    - "MSS: derivative-rank q(1-q) weighted by |Δ|/Σ|Δ| with flat-curve total-variation==0 guard → MSS 0.0"
    - "Precompute p + shape rank matrices sequentially, then iter_maybe_parallel! per-curve accumulation with unzip into (tvd, mss)"

key-files:
  created:
    - fdars-core/src/depth/tvd.rs (TvdMssResult + total_variation_depth_1d + rank_slice + 6 tests)
  modified:
    - fdars-core/src/depth/dispatch.rs (TotalVariation variant + arm; all-9 round-trip + regression + error-path tests)
    - fdars-core/src/depth/mod.rs (pub mod tvd; pub use total_variation_depth_1d, TvdMssResult)
    - fdars-core/src/lib.rs (crate-root re-exports)

key-decisions:
  - "TvdMssResult is the deliberately-pinned public interface of Phase 28 (reversibility: costly) — Phase 29's tvdmss calls total_variation_depth_1d(data,data)? directly for both fields; field names/types (tvd, mss) are a forward contract"
  - "TVD uses ÷n normalization (not ÷n+1) per fdaoutlier source — makes the highest curve (p=1) score TVD 0 while the median (p=0.5) scores max 0.25 (asymmetric on the high side)"
  - "MSS shape_variation reconstructed from Huang & Sun (2019) description (rank of first differences + same q(1-q) transform); fdaoutlier C++ internals not read directly — documented [ASSUMED] in rustdoc"
  - "Flat curve (zero total variation) → MSS 0.0 via guarded division, never NaN"

patterns-established:
  - "Result-struct depth measure (vs Vec) when a downstream phase needs multiple components; dispatcher projects the primary field"

requirements-completed: [DEPTH-01]
---

# Phase 28 Plan 03 — Summary

## Accomplishments

- **TVD + MSSI** (`depth/tvd.rs`): total-variation depth (magnitude) `mean_t p(1−p)` with
  `p = rank/n`, and the modified shape similarity index (shape) from first-difference ranks
  weighted by each interval's total-variation share, with a flat-curve guard (`MSS = 0.0`, no NaN).
  Returns the pinned **`TvdMssResult { tvd, mss }`** struct — the forward contract Phase 29's
  `tvdmss` consumes.
- **Dispatcher**: `DepthMethod::TotalVariation` projects the `.tvd` field (n<3 guard); crate-root
  re-exports of `total_variation_depth_1d` and `TvdMssResult`.
- **Dispatcher coverage tests**: an all-9-variant round-trip (every new parameter-free measure
  returns `Ok` with length `nrows`), a regression test proving the 4 pre-existing variants
  (`FraimanMuniz`, `Band`, `ModifiedBand`, `RandomProjection`) still work, and a min-n error-path
  sweep (single-curve and empty inputs return `Err`, no panic).
- **Tests**: 6 new tvd tests (hand-computed 3-curve TVD within 1e-9, max-TVD ≈ 0.25 at the median,
  magnitude-outlier low TVD, shape-outlier low MSS, flat-curve MSS == 0.0, dispatch projection,
  error paths) + 3 dispatcher coverage tests.

## Verification

- `cargo test -p fdars-core --features linalg,parallel` (FULL suite) → **2154 lib + all
  integration binaries pass, 0 failed**; depth module **135/135**; examples compile; doctests pass.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- `cargo fmt -- --check` → clean.
- No new crate dependency; existing depth-measure & `DepthMethod` public signatures unchanged.

## Environment note

The full-suite run initially failed at the **link** step for 4 of the 28 example binaries because
the `/home` partition was 100% full (`target/` had grown to 160G across the session). This is a
disk-capacity artifact, not a code defect — `cargo clippy --all-targets` (which type-checks the
examples) was clean throughout. Freed ~108G by removing the regenerable `target/debug/{incremental,
examples}` caches; the full suite then passed. Left `target/` at ~54G with 32G free.

## Phase 28 complete

DEPTH-01 delivered: all 9 measures (HI/MHI/EI/HRD/MHRD from 28-01, extremal/ERL/L∞ from 28-02,
TVD+MSSI here) are `Result`-returning, dispatch through `DepthMethod`, and are crate-root
re-exported — additive/non-breaking, no new dependency. The `TvdMssResult` interface is pinned for
Phase 29 (OUT-01).
