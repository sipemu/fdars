---
phase: 114-elastic-alignment-optimization
reviewed: 2026-09-23T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/alignment/mod.rs
  - fdars-core/tests/equivalence_phase114.rs
findings:
  critical: 0
  warning: 1
  info: 1
  total: 2
status: issues_found
---

# Phase 114: Code Review Report

**Reviewed:** 2026-09-23
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found (1 Warning, 1 Info — no Blockers)

## Summary

Reviewed the `dp_fill_grid` band-iteration refactor in `fdars-core/src/alignment/mod.rs`
(lines 533–567) and the golden equivalence fixture in
`fdars-core/tests/equivalence_phase114.rs`.

**Correctness verdict: PASS on all five criteria from the review scope.**

1. **Cell-set equivalence.** The banded arm computes
   `tc_lo = tr.saturating_sub(r)` and `tc_hi = (tr + r).min(ncols - 1)`.
   This is algebraically identical to the old `|tr - tc| > r` skip predicate:
   lower bound is `max(0, tr - r)` and upper bound is `min(tr + r, ncols - 1)`.
   Both arms visit exactly the same cells in the same row-major order.
   No off-by-one.

2. **`ncols - 1` underflow — unreachable via current call graph.**
   All callers pass `ncols = m` where `m >= 2` (guarded by the early-exit
   `if m < 2` in `dp_alignment_core_banded`). The `constrained.rs` caller of
   `dp_grid_solve` (the unbanded wrapper) has its own `if nc <= 1` guard and
   passes `band = None`, so the banded arm never fires there.
   `dp_grid_solve_banded` itself is `pub(super)` and currently called only
   from `mod.rs`. Underflow is unreachable today — but see WR-01 below.

3. **Origin `(0,0)` skip preserved.** The `if tr == 0 && tc == 0 { continue; }`
   guard is present in both the `None` and `Some(r)` arms. In the banded arm,
   when `tr == 0`, `tc_lo = 0` so the origin is the first cell in the iteration
   and is correctly skipped.

4. **`dp_skip_cell` removal complete.** A project-wide search finds zero
   references to `dp_skip_cell` anywhere under `fdars-core/src/` or
   `fdars-core/tests/`. No dangling references.

5. **Golden fixture quality.** The test uses real literal constants (non-trivial
   floating-point values at 17 significant-figure precision), a non-degenerate
   20×50 dataset, covers both the unbanded and banded arms independently, checks
   5 diverse off-diagonal cells plus symmetry in each arm, and uses an
   independent `assert_rel_close` at a tight 1e-10 relative tolerance.
   The identical `DIST_0_19` constant in both tests (4.85253341940299066e-1) is
   correct: curves 0 and 19 are well within the band-5 corridor so the banded
   and unbanded paths produce the same distance for that pair.
   The fixture is not self-satisfying.

---

## Warnings

### WR-01: `dp_grid_solve_banded` lacks a documented (or asserted) `ncols >= 1` precondition

**File:** `fdars-core/src/alignment/mod.rs:497`
**Issue:**
`dp_fill_grid` (called unconditionally from `dp_grid_solve_banded`) contains
`(tr + r).min(ncols - 1)` at line 557. In Rust, `0usize - 1` wraps to
`usize::MAX` in release builds (debug: panic). If `ncols == 0` were ever passed,
the expression would silently produce a nonsensical upper bound and the
subsequent `dp_traceback` at line 421 (`(nrows - 1) * ncols + (ncols - 1)`)
would also wrap.

The underflow is currently unreachable because every call site passes
`ncols = m >= 2`. However, `dp_grid_solve_banded` is `pub(super)` — any future
submodule added under `alignment/` can call it without the `m >= 2` guard.
The pre-existing `dp_traceback` function also has the `ncols - 1` subtract at
line 421, compounding the latent risk.

The refactor introduced a new site with this same pattern (`ncols - 1` at line
557) that did not exist before Phase 114 — the old skip-based code avoided this
expression by iterating `0..ncols` unconditionally. This is therefore a
net-new defensive gap introduced by the refactor.

**Fix:** Add a `debug_assert!(ncols >= 1 && nrows >= 1, "dp: empty grid")` at the
top of `dp_grid_solve_banded` (and symmetrically in `dp_traceback`), matching the
project's existing use of `debug_assert!` for internal invariants:

```rust
pub(super) fn dp_grid_solve_banded<F>(
    nrows: usize,
    ncols: usize,
    band: Option<usize>,
    edge_cost: F,
) -> Vec<(usize, usize)>
where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    debug_assert!(nrows >= 1 && ncols >= 1, "dp_grid_solve_banded: empty grid (nrows={nrows}, ncols={ncols})");
    DP_SCRATCH.with(|cell| {
        // ... existing body unchanged
    })
}
```

This costs nothing in release builds and catches the bug immediately in debug
builds / test runs if a future caller violates the implicit precondition.

---

## Info

### IN-01: `assert_rel_close` asymmetry — tolerance is relative to `a`, not `max(|a|, |b|)`

**File:** `fdars-core/tests/equivalence_phase114.rs:31`
**Issue:**
```rust
fn assert_rel_close(a: f64, b: f64, tol: f64) {
    let scale = a.abs().max(1e-12);
    assert!((a - b).abs() <= tol * scale, ...);
}
```
The scale is anchored to `a` (the reference constant). This means the assertion
`assert_rel_close(reference, computed, 1e-10)` correctly measures relative error
of the computed value against the reference, as intended. However, the argument
order is `assert_rel_close(dist[(i,j)], CONST, tol)` — i.e., `a` is the computed
value and `b` is the reference constant. The scale is therefore based on the
computed value, not the reference. For values near zero this is fine (the
`max(1e-12)` floor guards it), and for values in the range 0.5–3.0 the asymmetry
is inconsequential. This is not a bug in the current fixture, but a helper
ambiguity that could mislead a future contributor writing new assertions.

**Fix:** Either document the argument order in the function docstring, or rename the
parameter for clarity:
```rust
/// Asserts |a - b| <= tol * max(|a|, 1e-12).
/// Convention: pass the computed value as `a` and the reference constant as `b`.
fn assert_rel_close(computed: f64, reference: f64, tol: f64) {
    let scale = computed.abs().max(1e-12);
    assert!(
        (computed - reference).abs() <= tol * scale,
        "rel-close failed: computed={computed:.17e} reference={reference:.17e} \
         rel_err={:.3e} tol={tol:.1e}",
        (computed - reference).abs() / scale
    );
}
```

---

_Reviewed: 2026-09-23_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
