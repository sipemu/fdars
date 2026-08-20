---
phase: 32-flexible-mixed-effects-regression
fixed_at: 2026-08-20T00:00:00Z
review_path: .planning/phases/32-flexible-mixed-effects-regression/32-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 32: Code Review Fix Report

**Fixed at:** 2026-08-20
**Source review:** `.planning/phases/32-flexible-mixed-effects-regression/32-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (CR-01 + WR-01..WR-04)
- Fixed: 5
- Skipped: 0

Also applied IN-03 (cheap doc-only fix); skipped IN-01 and IN-02 per instructions.

## Fixed Issues

### CR-01: `fast_fmm` ignores `config.max_iter` and `config.tol`

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Replaced the `fit_scalar_mixed_model` call in the `fast_fmm` per-gridpoint
`iter_maybe_parallel!` loop with `fit_scalar_mixed_model_tracked`, passing `config.max_iter`
and `config.tol` as arguments. `fit_scalar_mixed_model_tracked` is a module-private function
already in the same file; no visibility change was needed. Added `max_iter == 0` guard in
`fast_fmm` (per WR-03 guidance, which says to add the guard in `fast_fmm` when CR-01 is fixed).
Added `test_fast_fmm_max_iter_takes_effect` confirming that a 1-iter and 100-iter run produce
different `sigma2_eps` values.

### WR-01: Running-mean smoother gives wrong window width for even `smooth_window`

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Before the smoothing loop, rounded `config.smooth_window` up to the nearest
odd value: `let w = if config.smooth_window % 2 == 0 { config.smooth_window + 1 } else { config.smooth_window }`. Added an explanatory comment. Added `test_fast_fmm_even_smooth_window`
which verifies that `smooth_window: 4` produces the same output as `smooth_window: 5` (the
rounded-up equivalent) and that results are finite.

### WR-02: `random_slopes = true` is silently ignored

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Added a guard in `dense_flmm` immediately after the `max_iter == 0` check:
returns `FdarError::InvalidParameter { parameter: "random_slopes", ... }` when
`config.random_slopes` is `true`. Updated the `DenseFlmmConfig::random_slopes` doc comment to
state it returns `InvalidParameter` until the feature ships (replacing the previous "silently
falls back" language). Added `test_dense_flmm_random_slopes_errors` asserting the error.

### WR-03: `dense_flmm`/`fast_fmm` don't validate `config.max_iter == 0`

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Added `if config.max_iter == 0 { return Err(FdarError::InvalidParameter { parameter: "max_iter", ... }) }` in both `dense_flmm` (after the `ncomp == 0` check) and
`fast_fmm` (after the `smooth_window == 0` check). `multi_famm` delegates through `dense_flmm`
so it is covered. Added `test_dense_flmm_max_iter_zero_errors` and
`test_fast_fmm_max_iter_zero_errors` asserting `InvalidParameter { parameter: "max_iter" }`.

### WR-04: Missing tests REG-05-G, REG-05-K, REG-05-L

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Added all three required tests using struct-literal config construction:
- `test_dense_flmm_converged` (REG-05-G): asserts `result.converged` is true with 100 iters;
  asserts `result.n_iter == 1` when `max_iter: 1, tol: 1e-30`.
- `test_fast_fmm_detects_effect` (REG-05-K): asserts `beta_matrix` row 0 has positive
  L2-squared norm and at least one grid point has `|t| > 0.5` for data with a known effect.
- `test_fast_fmm_empty_data_error` (REG-05-L): passes `FdMatrix::zeros(0, 0)` to `fast_fmm`
  and asserts `InvalidDimension { parameter: "data" }`.

### IN-03: `sigma2_eps` doc comment (cheap, applied per instructions)

**Files modified:** `fdars-core/src/famm.rs`
**Commit:** 2890328b
**Applied fix:** Expanded the `sigma2_eps` field doc in `DenseFlmmResult` to note that values
are on the L²-normalized FPC-score scale and are not directly comparable to R's `lmer()` output.

## Skipped Issues

None — all in-scope findings were fixed. IN-01 and IN-02 were explicitly excluded per instructions.

---

**Gate status:**
- `cargo build -p fdars-core --features linalg,parallel --lib`: PASSED (clean, no warnings)
- `cargo test -p fdars-core --features linalg,parallel --lib famm`: PASSED — 41/41 tests
- `cargo fmt -p fdars-core`: applied before commit
- Verification ran in the main checkout (no worktree — `workflow.use_worktrees` not set to false; main checkout used directly per agent prompt configuration)

---

_Fixed: 2026-08-20_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
