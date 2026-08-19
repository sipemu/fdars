---
phase: 28-depth-measure-long-tail
plan: "02"
subsystem: depth
tags: [depth, fdaoutlier, extremal-depth, extreme-rank-length, linfinity-depth, rank, fdmatrix, rayon]

requires:
  - phase: 28-01
    provides: DepthMethod additive-extension pattern (non_exhaustive enum + match arm + mod.rs/lib.rs re-export), inline-test conventions
provides:
  - extremal_depth_1d(data_obj, data_ori) -> Result<Vec<f64>, FdarError> — extremal depth (fdaoutlier, self-depth, n>=3)
  - extreme_rank_length_depth_1d(...) -> Result<Vec<f64>, FdarError> — ERL depth (fdaoutlier, self-depth, n>=2)
  - linfinity_depth_1d(...) -> Result<Vec<f64>, FdarError> — L-infinity depth (fdaoutlier, object-vs-reference, n>=1)
  - DepthMethod variants: Extremal (n<3 guard), ExtremeRankLength (n<2 guard), LInfinity (no guard)
  - private column_ranks helper (per-file, avg-tie ranks) in extremal.rs and erl.rs
  - crate-root re-exports of all 3 functions
affects: [phase-28-03, phase-29-outlier-detector-suite]

actuals:
  tokens: 32000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Average-tie column_ranks helper matching R rank(ties.method=average), defined per-file to avoid cross-module coupling (PATTERNS.md decision)"
    - "Deterministic ordering via sort_by with partial_cmp(...).unwrap_or(Equal).then(...).then(a.cmp(&b)) index tie-break"
    - "Extremal: pointwise depth 1-|2·rank-n-1|/n → (d_level=min, mass=fraction) → ordering → depth=k/n (sequential)"
    - "ERL: two-sided rank fold min(r, n+1-r) → per-curve sorted vector → pairwise lexicographic more-extreme count (iter_maybe_parallel! outer)"
    - "L-infinity: 1/(1 + mean_j max_t |Xi-Xj|), average sup-norm distance to ALL reference curves (iter_maybe_parallel! over objects)"

key-files:
  created:
    - fdars-core/src/depth/extremal.rs (extremal_depth_1d + column_ranks + 5 tests)
    - fdars-core/src/depth/erl.rs (extreme_rank_length_depth_1d + column_ranks + 4 tests)
    - fdars-core/src/depth/linf.rs (linfinity_depth_1d + 5 tests)
  modified:
    - fdars-core/src/depth/dispatch.rs (3 new DepthMethod variants + match arms; existing variants untouched)
    - fdars-core/src/depth/mod.rs (pub mod + pub use for erl/extremal/linf)
    - fdars-core/src/lib.rs (crate-root re-exports)

key-decisions:
  - "Extremal & ERL are self-depth measures evaluated on data_ori (the sample); data_obj is accepted for signature uniformity and validated to share the grid — documented in rustdoc"
  - "Extremal & ERL are SYMMETRIC rank measures: a top magnitude outlier (rank n) and the naturally-lowest boundary curve (rank 1) fold to identical extremeness and tie for shallowest. Tests assert the median curve is uniquely deepest and the outlier is among the most extreme, rather than the (false) claim that the outlier is the unique global minimum"
  - "L-infinity keeps genuine object-vs-reference semantics (d(i,j)=max_t|Xi-Xj|) and is valid for n=1 (self-distance 0 → depth 1.0); no n<2 guard, unlike extremal/ERL"
  - "column_ranks duplicated in extremal.rs and erl.rs per PATTERNS.md (avoid coupling two independent measure files through a shared helper)"

patterns-established:
  - "Rank-based depth test convention: assert median-curve-deepest + outlier-among-most-extreme (symmetric measures cannot uniquely rank a one-sided outlier below every boundary curve)"

requirements-completed: [DEPTH-01]
---

# Phase 28 Plan 02 — Summary

## Accomplishments

- **Extremal depth** (`depth/extremal.rs`): the Narisetty–Nair / fdaoutlier measure — pointwise
  depth `1 − |2·rank − n − 1|/n`, summarized per curve by `(d_level = min_t, mass = fraction at
  min)`, ordered ascending `d_level` then descending `mass` with an index tie-break, mapped to
  `k/n`. Sequential (O(n·m + n log n)). Requires `n ≥ 3`.
- **Extreme-rank-length depth** (`depth/erl.rs`): two-sided rank fold `min(r, n+1−r)`, per-curve
  ascending sort, then a pairwise lexicographic "more extreme" count over the sample
  (`iter_maybe_parallel!` outer loop, O(n²·m)). Requires `n ≥ 2`.
- **L∞ depth** (`depth/linf.rs`): `1 / (1 + mean_j max_t |X_i − X_j|)` — average sup-norm distance
  to all reference curves, inverted. Parallel over objects. Valid for `n ≥ 1`.
- **Dispatcher wiring**: `Extremal` (n<3 guard), `ExtremeRankLength` (n<2 guard), `LInfinity`
  (no guard) added to the `#[non_exhaustive]` `DepthMethod` enum with match arms; crate-root
  re-exports added. Existing variants + `functional_depth` signature untouched.
- **Tests**: 14 new inline tests covering unit-interval ranges, `deepest == 1.0`, deterministic
  tie ordering (extremal), central-deepest / outlier-most-extreme, `n=1` L∞ self-depth `== 1.0`,
  dispatch round-trips, and error paths (empty / min-n).

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib depth::` → **126 passed, 0 failed**.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` → **clean**.
- `cargo fmt -- --check` → clean.
- No new crate dependency; existing depth signatures unchanged (additive).

## Handoff to Plan 03

- Plan 28-03 adds `tvd.rs` (`TvdMssResult { tvd, mss }` + TVD/MSSI, `TotalVariation` variant) and
  the all-9-variant dispatcher round-trip test. It follows the same additive wiring pattern; the
  `TvdMssResult` struct is the forward contract Phase 29's `tvdmss` consumes.
