---
phase: 117-release-preparation-verification
reviewed: 2026-09-25T21:02:36Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/alignment/mod.rs
  - fdars-core/Cargo.toml
  - .gitignore
findings:
  critical: 0
  warning: 0
  info: 1
  total: 1
status: clean
---

# Phase 117: Code Review Report

**Reviewed:** 2026-09-25T21:02:36Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** clean

## Summary

Diff against `2388f7d3` touches three files:

1. **`fdars-core/src/alignment/mod.rs`** — `dp_fill_grid` was refactored to
   extract a new private helper `dp_band_cols(tr, ncols, band) -> Range<usize>`
   that computes the per-row column range, replacing the previous two
   duplicated `match` arms (one for `band = None`, one for `band = Some(r)`)
   with a single unified loop.
2. **`fdars-core/Cargo.toml`** — version bump `0.46.0` → `0.47.0`.
3. **`.gitignore`** — two new ignore entries for `fdars-core/.pmat/` and
   `fdars-core/.pmat-metrics/`.

### `dp_band_cols` / `dp_fill_grid` correctness

Traced the corridor arithmetic by hand:

- `None` arm: `0..ncols` — identical to the old `for tc in 0..ncols`.
- `Some(r)` arm: `tr.saturating_sub(r) .. (tr + r).min(ncols - 1) + 1`. Since
  `tc_hi := (tr + r).min(ncols - 1) <= ncols - 1`, the `+ 1` upper bound is
  `<= ncols` and cannot overflow given `ncols >= 1` (guarded by the caller's
  `debug_assert!`). The half-open range `tc_lo..tc_hi+1` is set-equal to the
  original inclusive range `tc_lo..=tc_hi` cell-for-cell, and when
  `tc_lo > tc_hi` both forms yield an empty iteration (Rust's `Range` with
  `start > end` is simply empty, no panic) — matching the stated doc claim.
- Row-major visitation order and the `(tr, tc) == (0, 0)` origin skip are
  preserved unchanged inside the unified loop.
- The only caller that ever passes `band = Some(r)` is
  `dp_alignment_core_banded`, which always calls
  `dp_grid_solve_banded(m, m, ...)` — i.e. `nrows == ncols` is an invariant of
  every live call site, so `tr <= ncols - 1` always holds inside the `Some(r)`
  arm and `tc_lo <= tc_hi` is guaranteed (never triggers the empty-range
  case in practice, but the extra safety is real, not accidental).
- The pre-existing `ncols - 1` release-mode underflow risk when `ncols == 0`
  is carried over unchanged from the pre-refactor code (previously present in
  the `Some(r)` match arm) — not a regression introduced by this diff, and it
  remains behind the same `debug_assert!(nrows >= 1 && ncols >= 1, ...)` guard
  as before.

Verified empirically: `cargo test -p fdars-core --lib alignment::` — 267
passed, 0 failed, including the banded-vs-unbanded equivalence suite
(`test_banded_align_wide_matches_unbanded`,
`test_self_distance_matrix_with_band_none_matches_exact`,
`test_karcher_mean_with_band_some_zero_matches_exact`,
`test_karcher_mean_with_band_wide_matches_unbanded`, etc.), and
`cargo clippy -p fdars-core --lib --features linalg -- -D warnings` is clean.
No behavioral difference between the old two-arm implementation and the new
`dp_band_cols`-based implementation was found.

### Version bump

`0.47.0` is consistent with `fdars-core/CHANGELOG.md`, which already carries
a matching `## [0.47.0] - 2026-09-25` entry documenting this as a
non-breaking, numeric-identical performance milestone. No other manifest in
the repo (workspace root, `benchmarks/fpca_dump/Cargo.toml`) pins
`fdars-core`'s version, so the bump is self-contained and has no ripple
effect.

### `.gitignore` additions

Both new patterns (`fdars-core/.pmat/`, `fdars-core/.pmat-metrics/`) match
real, currently-untracked local directories (`ls -la fdars-core/` confirms
both exist) and `git check-ignore -v` confirms they now resolve to these new
lines. Neither directory was previously tracked in git, so this is a
pure hygiene addition with no risk of un-tracking previously-committed files.

All reviewed files meet quality standards. No Critical or Warning issues
found.

## Info

### IN-01: `Some(r)` corridor arithmetic implicitly assumes `nrows == ncols`

**File:** `fdars-core/src/alignment/mod.rs:539-544`
**Issue:** `dp_band_cols`'s `Some(r)` arm computes `tc_hi` against `ncols - 1`
but `tr` is driven by the caller's `nrows` loop
(`for tr in 0..nrows { for tc in dp_band_cols(tr, ncols, band) ... }` in
`dp_fill_grid`). Correctness of the "always non-empty corridor" claim in the
doc comment depends on `nrows == ncols` (true for every current call site,
since `dp_grid_solve_banded` is only ever invoked with `Some(r)` from
`dp_alignment_core_banded(m, m, ...)`), but `dp_grid_solve_banded` is a more
general `pub(super)` function that accepts independent `nrows`/`ncols` and
only asserts `nrows >= 1 && ncols >= 1` — not `nrows == ncols` when banding is
requested. If a future caller passed `band = Some(r)` with `nrows != ncols`,
rows where `tr > ncols - 1 + r` would silently produce an empty corridor
(no panic, but those rows would be entirely skipped, which may or may not be
the caller's intent). This is not a live bug today, only a latent
maintainability trap that this refactor's doc comment (line 537) papers over
by asserting "same cells visited" without stating the square-grid
precondition.
**Fix:** Either tighten the `debug_assert!` in `dp_grid_solve_banded` to
require `nrows == ncols` when `band.is_some()`, or make the precondition
explicit in `dp_band_cols`'s doc comment, e.g.:
```rust
debug_assert!(
    band.is_none() || nrows == ncols,
    "banded DP grid requires a square grid (nrows={nrows}, ncols={ncols})"
);
```

---

_Reviewed: 2026-09-25T21:02:36Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
