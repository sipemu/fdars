---
phase: 115-svd-fpca-covariance-optimization
reviewed: 2026-09-23T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/irreg_fdata/smoothing.rs
  - fdars-core/src/irreg_fdata/face.rs
  - fdars-core/tests/equivalence_phase115.rs
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 115: Code Review Report

**Reviewed:** 2026-09-23T00:00:00Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the three files in scope: the new `sym_cov_irreg` implementation in
`smoothing.rs`, its call-site in `face.rs`, and the golden-equivalence fixture
in `equivalence_phase115.rs`.

The core correctness claim — that `sym_cov_irreg` produces bit-identical upper-triangle
values to `cov_irreg(ifd, grid, grid, bandwidth)` — holds. The accumulation order
(`i → j1 → j2`), the weight table layout (`w[obs_idx + gi * total_obs]`), and the
division-by-sum_weights are all word-for-word identical to the reference function.
The mirror assignment (`cov[ti + si*m] = val`) writes the correct column-major index
for all off-diagonal pairs; for the diagonal (`si == ti`) both writes target the same
cell, which is harmless. The `m=1` and `m=0` edge cases cannot be reached through the
only call site (`face_covariance` pre-validates `grid.len() >= 2`).

One structural bug was found: the `#[must_use]` attribute and the full doc block
intended for `mface_covariance` are syntactically attached to the private helper
`fill_diag_blocks` due to an intervening doc comment. `mface_covariance` is therefore
missing both its documentation and the attribute.

The golden fixture is sound: it pins 7 distinct cells (3 diagonal, 2 upper-triangle,
2 mirrors) plus a full lower-triangle symmetry sweep, all at `1e-10`. The fixture
cannot self-satisfy because the constants were captured from the pre-optimization
binary and the tolerance is far below the ~1e-17 pre-opt asymmetry between mirrored
pairs.

---

## Warnings

### WR-01: `#[must_use]` and doc block orphaned on `fill_diag_blocks` — `mface_covariance` undocumented and missing the attribute

**File:** `fdars-core/src/irreg_fdata/face.rs:263-266`

**Issue:** The `#[must_use]` attribute (line 263) and the entire public doc block
(lines 241-262) that were written for `mface_covariance` are syntactically attached
to `fill_diag_blocks` (line 266), not to `mface_covariance` (line 330). Rust attaches
attributes and doc comments to the immediately following item; because line 264 is an
additional doc comment (`/// Fill diagonal blocks…`) for `fill_diag_blocks`, the
compiler treats everything from line 241 onward as belonging to `fill_diag_blocks`.
As a result:

- `mface_covariance` has no `/// …` documentation (violates the project convention
  that all public items carry doc comments).
- `mface_covariance` is missing `#[must_use]` (violates the project convention that
  74+ expensive-computation functions carry this attribute; the original intent is
  clear from the stranded attribute text).
- The attribute and doc comment on `fill_diag_blocks` are nonsensical there (it is
  a private `fn` with a `Result<(), FdarError>` return, so `#[must_use]` is vacuous
  and the consumer-facing "returns the block covariance" message is wrong).

This does not affect runtime correctness or the numeric identity of results, but it
means the public API surface of `mface_covariance` is invisible to `rustdoc` users
and the `#[must_use]` lint is absent on every call site.

**Fix:** Move the doc block and `#[must_use]` to `mface_covariance` and replace the
erroneous comment on `fill_diag_blocks` with its own brief doc:

```rust
// Before (lines 241-266 in face.rs):

/// Multivariate FACE (`mfaces`) block covariance …
///
/// … (long doc) …
#[must_use = "mface_covariance returns the block covariance; ignoring it wastes the computation"]
/// Fill diagonal blocks of the block covariance matrix with per-variable FACE covariance.
#[allow(clippy::too_many_arguments)]
fn fill_diag_blocks(…) { … }

// … (fill_offdiag_blocks) …

pub fn mface_covariance(…) { … }   // ← no doc, no must_use


// After:

/// Fill diagonal blocks of the block covariance matrix with per-variable FACE covariance.
#[allow(clippy::too_many_arguments)]
fn fill_diag_blocks(…) { … }

// … (fill_offdiag_blocks) …

/// Multivariate FACE (`mfaces`) block covariance across `P >= 2` sparse variables.
///
/// … (long doc moved here) …
#[must_use = "mface_covariance returns the block covariance; ignoring it wastes the computation"]
pub fn mface_covariance(…) { … }
```

---

## Info

### IN-01: Golden fixture pins only 7 of 121 cells; interior off-diagonal cells not locked

**File:** `fdars-core/tests/equivalence_phase115.rs:77-123`

**Issue:** The fixture asserts 3 diagonal cells, 2 upper-triangle cells, and their
2 mirrors (7 cells total out of 121 for the 11×11 grid), plus a full symmetry sweep.
The two upper-triangle cells chosen (`(0,5)` and `(2,7)`) are both in the same
general region (corners and off-centre). A transposition bug that swapped `si` and
`ti` only in interior index ranges (e.g., `3 ≤ si ≤ 8`) would go undetected by the
chosen constants alone. The symmetry sweep catches any such bug indirectly (because a
transposition would produce an asymmetric result), so this is not a correctness gap
in the current code — the symmetry invariant is the load-bearing safety net. However,
the fixture gives weaker direct evidence for the specific cells that carry the highest
numerical sensitivity (near the covariance diagonal where the kernel weight is largest).

**Fix:** Add two additional constant assertions for cells closer to the diagonal,
e.g. `(1,2)` and `(8,9)` (one-step off-diagonal, high kernel weight). This
strengthens the regression signal without increasing fixture size meaningfully.
The symmetry sweep already provides full structural coverage, so this is optional.

---

_Reviewed: 2026-09-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
