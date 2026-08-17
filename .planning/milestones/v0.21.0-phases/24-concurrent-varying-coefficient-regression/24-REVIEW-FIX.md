---
phase: 24-concurrent-varying-coefficient-regression
fixed_at: 2026-08-17T00:00:00Z
review_path: .planning/phases/24-concurrent-varying-coefficient-regression/24-REVIEW.md
iteration: 1
findings_in_scope: 2
fixed: 2
skipped: 0
status: all_fixed
---

# Phase 24: Code Review Fix Report

**Fixed at:** 2026-08-17
**Source review:** `.planning/phases/24-concurrent-varying-coefficient-regression/24-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 2
- Fixed: 2
- Skipped: 0

## Fixed Issues

### CR-01: NaN bandwidth bypasses validation and silently produces all-zero coefficients

**Files modified:** `fdars-core/src/concurrent_regression.rs`
**Commit:** `fac47648`
**Applied fix:** Replaced `if bandwidth <= 0.0` with `if !bandwidth.is_finite() || bandwidth <= 0.0`.
The review suggested `!(bandwidth > 0.0) || !bandwidth.is_finite()` but clippy
(`neg_cmp_op_on_partial_ord`) rejects negated partial-order comparisons, so
`!bandwidth.is_finite()` is placed first (short-circuits on NaN/Inf before the
`<=` comparison, which is always safe for finite values). Both forms are
semantically identical. The `# Errors` doc comment was updated to document NaN
and Inf as rejected values. Added regression test
`test_nan_inf_bandwidth_returns_error` that asserts both `f64::NAN` and
`f64::INFINITY` return `InvalidParameter`, and that a valid positive bandwidth
still succeeds.

### WR-01: Missing guard for underdetermined pointwise system (n <= p)

**Files modified:** `fdars-core/src/concurrent_regression.rs`
**Commit:** `5cef1d22`
**Applied fix:** Added `if n <= p { return Err(FdarError::InvalidDimension { ... }) }` guard
immediately after `p = predictors.len()` is computed (before step 1 / pointwise
OLS). The error message names the required minimum row count and the actual `p`.
Updated `# Errors` doc comment to document the new condition. Demoted the prior
`# Notes` advisory ("sample size should satisfy n > p + 1") to a simpler note
about the ridge regulariser, since the unguarded case no longer occurs. Added
regression test `test_underdetermined_system_returns_error` covering three
sub-cases: `n == p` (square, still underdetermined for the p+1-column design),
`n < p` (severely underdetermined), and `n > p` (valid, must succeed).

**Verification:** All gates ran inside the isolated worktree
(`gsd-reviewfix/24-607815` at `.claude/worktrees/rf-24-607815-1786960560`).

- `cargo fmt --check` — passed for both commits.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — passed
  for both commits (0 errors, 0 warnings).
- `cargo test -p fdars-core --features linalg,parallel --lib concurrent_regression`
  — 9/9 tests pass after both fixes (up from 7 before the phase, 8 after CR-01,
  9 after WR-01).

The worktree was not linked to the project's `node_modules` (no NPM build step
in this Rust-only project), so gate results are fully reproducible from the main
checkout after the fast-forward.

---

_Fixed: 2026-08-17_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
