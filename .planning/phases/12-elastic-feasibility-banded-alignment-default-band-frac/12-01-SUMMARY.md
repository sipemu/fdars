---
phase: 12-elastic-feasibility-banded-alignment-default-band-frac
plan: "01"
subsystem: alignment
status: complete
tags: [elastic, banded, sakoe-chiba, api-surfacing, performance, PERF-03]
requirements: [PERF-03]

dependency_graph:
  requires: []
  provides:
    - fdars_core::karcher_mean_with_band
    - fdars_core::elastic_self_distance_matrix_with_band
    - fdars_core::elastic_cross_distance_matrix_with_band
    - fdars_core::karcher_mean_banded (crate-root re-export, was alignment-submodule only)
    - fdars_core::elastic_self_distance_matrix_banded (crate-root re-export)
    - fdars_core::elastic_cross_distance_matrix_banded (crate-root re-export)
  affects:
    - fdars-core/src/alignment/karcher.rs
    - fdars-core/src/alignment/pairwise.rs
    - fdars-core/src/alignment/mod.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/alignment/tests.rs
    - fdars-core/benches/alignment_benchmarks.rs

tech_stack:
  added: []
  patterns:
    - "*_with_band(…, band_frac: Option<f64>) wrapper pattern — ergonomic opt-in over existing _banded(…, band_frac: f64) variants"
    - ".and_then(|f| band_radius(f, m)) for Option<f64>→Option<usize> conversion (avoids clippy::map_flatten)"

key_files:
  created: []
  modified:
    - fdars-core/src/alignment/karcher.rs
    - fdars-core/src/alignment/pairwise.rs
    - fdars-core/src/alignment/mod.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/alignment/tests.rs
    - fdars-core/benches/alignment_benchmarks.rs

decisions:
  - "Used *_with_band(…, band_frac: Option<f64>) wrapper pattern instead of modifying existing positional signatures (which is breaking in Rust). Matches the codebase's _with_config precedent."
  - "Used .and_then(|f| band_radius(f, m)) not .map(…).flatten() to avoid clippy::map_flatten under CI -D warnings."
  - "Did not add #[must_use] to existing elastic_self_distance_matrix / elastic_cross_distance_matrix — this is a separate lint-only change outside this plan's scope."
  - "Added _banded variants for self/cross distance matrices to crate-root re-exports alongside _with_band — completes discoverability (RESEARCH Open Question 2)."
  - "Feasibility bench at n=20/m=50 only; large-grid cells (N=500/M=200) already covered by existing audit_hotpaths.rs bench_p3_karcher_banded."

metrics:
  completed_date: "2026-08-11"
  duration_minutes: 22
  tasks_completed: 3
  tasks_total: 3
  commits: 3

actuals:
  tokens: 6200
  tasks: 3
  commits: 3
---

# Phase 12 Plan 01: Elastic Banded Alignment API Surfacing Summary

## One-liner

Three thin `*_with_band(…, band_frac: Option<f64>)` delegation wrappers surface the existing correct Sakoe–Chiba banded DP path at the fdars crate root, with six crate-root re-exports, six equivalence tests, and a feasibility bench.

## What Was Built

**API surfacing only — no new algorithm.** The banded Sakoe-Chiba DP implementations (`karcher_mean_banded`, `elastic_self_distance_matrix_banded`, `elastic_cross_distance_matrix_banded`) and the `band_radius` helper already existed and were correct. This plan:

1. Added three `*_with_band(…, band_frac: Option<f64>)` public wrapper functions:
   - `karcher_mean_with_band` in `karcher.rs` — delegates via `band_frac.unwrap_or(0.0)` to `karcher_mean_impl`
   - `elastic_self_distance_matrix_with_band` in `pairwise.rs` — delegates via `.and_then(|f| band_radius(f, m))`
   - `elastic_cross_distance_matrix_with_band` in `pairwise.rs` — same pattern

2. Extended crate-root re-exports in `lib.rs` and `alignment/mod.rs` to include:
   - The three new `_with_band` functions
   - The three existing `_banded` variants (previously only accessible via `fdars_core::alignment::*`)

3. Added 6 equivalence tests in `alignment/tests.rs`:
   - `test_karcher_mean_with_band_none_matches_exact` — None path identical within 1e-15
   - `test_karcher_mean_with_band_wide_matches_unbanded` — Some(0.99) matches within 1e-12 at m=30
   - `test_self_distance_matrix_with_band_none_matches_exact` — None path identical within 1e-15
   - `test_self_distance_matrix_with_band_wide_matches_unbanded` — Some(0.99) matches within 1e-12 at m=40
   - `test_cross_distance_matrix_with_band_none_matches_exact` — None path identical within 1e-15
   - `test_cross_distance_matrix_with_band_wide_matches_unbanded` — Some(0.99) matches within 1e-12 at m=30

4. Added `bench_karcher_mean_with_band` group in `alignment_benchmarks.rs` comparing None (exact) vs Some(0.1) (banded) at n=20/m=50, with doc comment pointing to `audit_hotpaths.rs` for large-grid cells.

## Commits

| Hash | Message |
|------|---------|
| `ff85497b` | feat(12): add karcher_mean_with_band opt-in banded wrapper + equivalence tests |
| `d50bf1b5` | feat(12): add elastic distance-matrix _with_band wrappers + crate-root re-exports |
| `aadb4119` | bench(12): add karcher_mean_with_band feasibility bench, point to audit_hotpaths for large grids |

## Deviations from Plan

None — plan executed exactly as written. The RESEARCH.md note about `.map(…).flatten()` (Pitfall: clippy::map_flatten) was heeded; `.and_then(|f| band_radius(f, m))` was used throughout.

## Known Stubs

None. All wrappers are fully wired to existing correct implementations.

## Threat Flags

None. Pure algorithmic API extension — no I/O, no network, no unsafe, no user-controlled deserialization.

## Verification

- `cargo test -p fdars-core --features linalg` — all 1,954 lib tests + 134 doc tests pass
- `cargo build -p fdars-core --all-targets` — clean (no errors in elastic_changepoint.rs, elastic_fpca.rs, tsrvf.rs, etc.; all 30+ existing call sites compile unchanged)
- CI-parity clippy (`--all-targets --all-features -D warnings` + standard -A allows) — clean
- Bench registered: `karcher_mean_with_band/n20_m50_none` and `karcher_mean_with_band/n20_m50_band0.1` visible in `cargo bench --list`

## Self-Check: PASSED

- `fdars-core/src/alignment/karcher.rs` — FOUND: karcher_mean_with_band
- `fdars-core/src/alignment/pairwise.rs` — FOUND: elastic_self_distance_matrix_with_band, elastic_cross_distance_matrix_with_band
- `fdars-core/src/alignment/mod.rs` — FOUND: exports updated
- `fdars-core/src/lib.rs` — FOUND: 6 new/existing items added to pub use alignment block
- `fdars-core/src/alignment/tests.rs` — FOUND: 6 equivalence tests added
- `fdars-core/benches/alignment_benchmarks.rs` — FOUND: bench_karcher_mean_with_band registered
- Commits ff85497b, d50bf1b5, aadb4119 — FOUND in git log
