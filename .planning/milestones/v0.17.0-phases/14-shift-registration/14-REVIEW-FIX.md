---
phase: 14-shift-registration
fixed_at: 2026-08-12T13:15:00Z
review_path: .planning/phases/14-shift-registration/14-REVIEW.md
iteration: 1
findings_in_scope: 4
fixed: 4
skipped: 0
status: all_fixed
---

# Phase 14: Shift Registration - Code Review Fix Report

**Fixed at:** 2026-08-12T13:15:00Z
**Source review:** .planning/phases/14-shift-registration/14-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 4 (CR-01 + WR-01 + WR-02 + WR-03; IN-01/02/03 excluded per scope)
- Fixed: 4
- Skipped: 0

## Fixed Issues

### CR-01: Dead-code / unused-variable warnings in shift.rs tests will fail CI

**Files modified:** `fdars-core/src/alignment/shift.rs`
**Commit:** `36c40c1a`
**Applied fix:**
- Deleted the unused `make_shifted_bumps` helper function from the `shift.rs` test module (lines ~277-288). The quality.rs test module has its own independently-defined copy that IS used.
- Removed the unused `let argvals = uniform_grid(m);` binding in `test_shift_registration_argvals_mismatch` (line ~411). The test only uses `wrong_argvals` and `short_argvals`.
- Both were `dead_code` / `unused_variables` lints promoted to hard errors by `cargo clippy --all-targets -D warnings` (the CI gate).

### WR-01: pairwise_correlation_score documented as Pearson but implements cosine similarity

**Files modified:** `fdars-core/src/alignment/quality.rs`
**Commit:** `97ed3b2a`
**Applied fix:**
Replaced the uncentered cosine-similarity implementation with true functional Pearson
correlation. Changes:
- Precompute Simpson-weighted functional mean μᵢ = (Σⱼ fᵢ(tⱼ)·wⱼ) / (Σⱼ wⱼ) for each curve.
- Centre each curve: f̃ᵢ = fᵢ − μᵢ (stored in a `centred: Vec<Vec<f64>>`).
- Inner products and norms are computed on centred curves.
- Zero-variance (nearly-constant) curves contribute 0.0 to all pairs (NaN guard unchanged).
- Added `argvals.len() < 2` guard (covering WR-02 for this function).
- Updated rustdoc to explicitly state "Pearson correlation (centred)" and clarify the
  zero-variance guard semantics.
- All 6 existing quality tests pass; `test_pairwise_corr_rises_after_registration` continues
  to pass because Gaussian-bump fixtures have non-trivial centred structure that increases
  in alignment after shift registration.

Note: this is a **semantic change** to the score value for non-zero-mean curves (Gaussian
bumps are strictly positive, so cosine ≈ 1.0 regardless of shift; Pearson is more
discriminative). The test only asserts `score_after > score_before`, so it is robust.
Marked `fixed: requires human verification` for the numerical magnitude.

### WR-02: score functions accept m==1, unlike the shift function

**Files modified:** `fdars-core/src/alignment/quality.rs`
**Commit:** `c4ee0926`
**Applied fix:**
- Added `if argvals.len() < 2 { return Err(InvalidParameter{...}) }` to `least_squares_score`
  (after the `argvals.len() != m` dimension check).
- Added the same guard to `sobolev_least_squares_score` (after the dimension checks, before the
  `lambda < 0.0` check).
- `pairwise_correlation_score` received this guard in the WR-01 commit.
- Added `test_score_fns_reject_single_point_grid`: asserts all three score functions return
  `Err(InvalidParameter)` for a 1-point argvals, exercising all three guard paths.

### WR-03: sobolev_least_squares_score silently ignores non-uniform grids

**Files modified:** `fdars-core/src/alignment/quality.rs`
**Commit:** `fc21d3ae`
**Applied fix:**
- When `lambda > 0`, compute `h = (argvals[m-1] - argvals[0]) / (m-1)` and verify every
  consecutive spacing is within `1e-9 * h.abs().max(1e-12)` of `h`.
- Non-uniform argvals with `lambda > 0` now return `Err(InvalidParameter)` with a message
  directing callers to `gradient_nonuniform`.
- The existing `test_sobolev_score_lambda_positive` uses `make_shifted_bumps` (uniform grid)
  and continues to pass.
- Updated rustdoc: changed "Uniform-grid assumption" to "Uniform-grid requirement (when
  lambda > 0)" and updated the `# Errors` section to list the new variant.

## Skipped Issues

None — all four in-scope findings were applied.

---

## Final Verification

**Command:** `cargo clippy --all-targets -p fdars-core --features linalg -- -D warnings`
**Result:** Clean (no errors, no warnings)

**Command:** `cargo test -p fdars-core --features linalg -- alignment::shift alignment::quality`
**Result:** 12 passed, 0 failed (5 shift tests + 7 quality tests)

Verification ran in the **main checkout** (no worktree — `workflow.use_worktrees=false` mode).

---

_Fixed: 2026-08-12T13:15:00Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
