---
phase: 12-elastic-feasibility-banded-alignment-default-band-frac
reviewed: 2026-08-11T20:20:34Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - fdars-core/src/alignment/karcher.rs
  - fdars-core/src/alignment/pairwise.rs
  - fdars-core/src/alignment/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/alignment/tests.rs
  - fdars-core/benches/alignment_benchmarks.rs
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: needs-attention
---

# Phase 12: Code Review Report

**Reviewed:** 2026-08-11T20:20:34Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** needs-attention

## Summary

Phase 12 adds three thin `*_with_band(…, band_frac: Option<f64>)` delegation wrappers over the existing correct Sakoe–Chiba banded DP path, plus crate-root re-exports, six equivalence tests, and a feasibility bench. The algorithmic delegation is correct: `None` and `Some(0.0)` both produce the unbanded path; `Some(f > 0 && f < 1)` produces the banded path. Existing signatures are byte-for-byte unchanged. The DP scratch buffer is properly cleared on every call; NaN inputs are handled gracefully (treated as unbanded). No panics, no unsafe code, no security surface.

One **Warning** was found: the rustdoc on all three `_with_band` functions states a factually incorrect coverage threshold for `band_frac = 0.99`. Two **Info** items cover a delegation-pattern inconsistency and a missing sentinel-value test.

## Warnings

### WR-01: Rustdoc claims `band_frac = 0.99` gives full-warp coverage "at `m ≥ 10`" — the correct bound is `m < 200`

**File:** `fdars-core/src/alignment/karcher.rs:336`, `fdars-core/src/alignment/pairwise.rs:227`, `fdars-core/src/alignment/pairwise.rs:328`

**Issue:** All three `_with_band` functions carry the following rustdoc bullet:

```
/// - `Some(0.99)`: near-full band; at `m ≥ 10` this covers the full warp corridor
///   and produces output within `1e-12` of the unbanded result.
```

The claim "at `m ≥ 10`" is incorrect. The full-warp guarantee requires `band_radius(0.99, m) >= m - 1`, where `band_radius(f, m) = ceil(f * m)`. Expanding:

```
ceil(0.99 * m) >= m - 1
⟺  0.99 * m  > m - 2     (since any x > m-2 has ceil(x) >= m-1)
⟺  0.01 * m  < 2
⟺  m         < 200
```

For `m = 200`: `band_radius(0.99, 200) = ceil(198.0) = 198`, and `m - 1 = 199`; the band is one short. `dp_grid_solve_banded` skips any cell where `tr.abs_diff(tc) > band_radius`, so the two diagonally-extreme corners of the DP grid become unreachable, and the banded result can diverge from the unbanded result by more than `1e-12`. The guarantee breaks for every `m >= 200` (verified analytically and by exhaustive check over `m in [200, 999]`).

The three equivalence tests themselves are not affected — they use `m = 30` and `m = 40`, both well below the threshold — but any user who reads the doc and relies on this claim when `m >= 200` will get a misleading guarantee. The `audit_hotpaths.rs` bench targets `M = 200` exactly, making this a realistic and relevant case for the library's stated performance audience.

**Fix:** Change "at `m ≥ 10`" to "at `m < 200`" (or equivalently "at `m ≤ 199`") in all three doc-comment blocks. The corrected bullets:

```rust
/// - `Some(0.99)`: near-full band; at `m < 200` this covers the full warp corridor
///   and produces output within `1e-12` of the unbanded result. For `m ≥ 200`,
///   use `band_frac = 1.0 / m as f64` margin above `1 - 2.0 / m as f64` to
///   guarantee full coverage, or simply use `None` for exact results.
```

The simplest safe alternative for the doc is: "when `band_frac` is large enough that `ceil(band_frac * m) >= m - 1`" — this is exact and `m`-agnostic.

## Info

### IN-01: Delegation-pattern inconsistency between `karcher_mean_with_band` and the pairwise wrappers

**File:** `fdars-core/src/alignment/karcher.rs:350-357` vs `fdars-core/src/alignment/pairwise.rs:238`, `pairwise.rs:340`

**Issue:** `karcher_mean_with_band` converts `Option<f64>` to `f64` via `band_frac.unwrap_or(0.0)` and passes that to `karcher_mean_impl(…, band_frac: f64)`, which then calls `band_radius` internally. The pairwise wrappers convert `Option<f64>` to `Option<usize>` via `.and_then(|f| band_radius(f, m))` and pass that `Option<usize>` directly to the impl. Both paths are numerically correct — `unwrap_or(0.0)` routes to `band_radius(0.0, m) = None` (unbanded) identically to the `.and_then` path returning `None`. However, the inconsistency is a maintenance hazard: a future refactor of `karcher_mean_impl` that changes the semantics of a zero `band_frac` argument would silently break the `None` path for `karcher_mean_with_band` without compiler error, while the pairwise functions would remain correct. The `karcher_mean_impl` signature takes `band_frac: f64`, so refactoring to match the pairwise pattern exactly would require changing `karcher_mean_impl`'s signature — a non-trivial but worthwhile cleanup.

**Fix:** If `karcher_mean_impl` is ever refactored to take `band: Option<usize>` directly (as the pairwise impls do), update `karcher_mean_with_band` to use `.and_then(|f| band_radius(f, m))` as well. No action required now, but worth noting for future refactors.

### IN-02: No test for the `Some(0.0)` sentinel case that is explicitly documented

**File:** `fdars-core/src/alignment/tests.rs` (new test block, lines 2731–2856)

**Issue:** All three `_with_band` doc comments explicitly state that `Some(0.0)` is "treated as exact/unbanded (equivalent to `None`), because `band_radius(0.0, m)` returns `None`." This is a documented invariant and a named sentinel value in the API contract. None of the six new tests exercise `Some(0.0)`. The `None`-path tests verify that `unwrap_or(0.0)` / `.and_then` produce unbanded results, but they don't verify that the separate `Some(0.0)` code path (which takes a different branch in `unwrap_or` for the karcher wrapper) also produces identical results. If `band_radius` were ever changed to treat `0.0` as a tiny positive band rather than exact/unbanded, the `None` tests would still pass while the `Some(0.0)` guarantee silently broke.

**Fix:** Add three one-line tests (or extend the existing `None` equivalence tests) asserting that `elastic_*_with_band(…, Some(0.0))` produces results element-wise identical to the unbanded baseline within `1e-15`, mirroring the existing `None` tests.

---

_Reviewed: 2026-08-11T20:20:34Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
