---
phase: 107-spm-complexity-cluster
reviewed: 2026-09-14T10:30:00Z
depth: standard
files_reviewed: 12
files_reviewed_list:
  - fdars-core/src/spm/amewma.rs
  - fdars-core/src/spm/iterative.rs
  - fdars-core/src/spm/mfpca.rs
  - fdars-core/src/spm/ncomp.rs
  - fdars-core/src/spm/contrib.rs
  - fdars-core/src/spm/ewma.rs
  - fdars-core/src/spm/mewma.rs
  - fdars-core/src/spm/stats.rs
  - fdars-core/src/spm/phase.rs
  - fdars-core/src/spm/partial.rs
  - fdars-core/src/spm/frcc.rs
  - fdars-core/src/spm/profile.rs
findings:
  critical: 0
  warning: 3
  info: 4
  total: 7
status: issues_found
---

# Phase 107: Code Review Report

**Reviewed:** 2026-09-14T10:30:00Z
**Depth:** standard
**Files Reviewed:** 12
**Status:** issues_found

## Summary

Phase 107 was a pure behavior-preserving complexity refactor of the `src/spm/` cluster: 13 functions were decomposed into private helper `fn`s via VERBATIM block extraction. The review focused on the four highest-risk hazards identified in the phase brief.

**All four critical-path hazards verified clean:**

1. **AMEWMA sequential recurrence** (`amewma.rs`): `q_prev`/`lambda_prev`/`sum_lambda_sq_prod` update order in the extracted loop is identical to pre-refactor. The `amewma_ewma_step` helper correctly reads `smoothed[(i-1, l)]` (not `smoothed[(i, l)]`) via the `s_prev` staging vector, preserving causal ordering. Confirmed by diff against commit `e889c45e`.

2. **`mfpca` `scales` in-place mutation** (`mfpca.rs`): `mfpca_build_stacked` takes `scales: &mut [f64]` and writes `scales[p] = 1.0` when `!weighted`. The mutation propagates correctly back to `MfpcaResult.scales`. The `weighted=true, scale<=threshold` case does NOT update `scales[p]` (keeping the original near-zero value), but `project()` and `reconstruct()` both apply the same `scale_threshold` guard to fall back to `1.0` — behavior is numerically identical to training. Confirmed IDENTICAL to pre-107 code (commit `5fb18fdb` → `c91b1fb2`).

3. **`conditional_expectation` Tikhonov retry** (`partial.rs`): `diag_max` is captured from the ORIGINAL (pre-regularization) diagonal at lines 391–397, BEFORE the Tikhonov block (lines 399–405) modifies `m_matrix`. `cond_exp_solve_with_retry` receives both the possibly-regularized `m_matrix` and the original `diag_max` — the retry sequence correctly starts at `diag_max * 1e-8` and multiplies by 10 each attempt, exactly matching pre-refactor behavior (commit `8beb5eae`). No behavioral divergence.

4. **Floating-point accumulation order**: All extracted helpers preserve the exact loop structure and iteration order of the original code. No `fold`/`sum` reorderings were introduced. The `z.row(i)` call in `amewma_ewma_step` introduces a minor `Vec<f64>` allocation but not a summation-order change.

5. **Visibility**: All extracted helpers are private `fn` (no `pub` or `pub(crate)` widening). Verified across all 12 files.

No refactor regressions were found. Three pre-existing quality issues and four informational items are documented below.

## Warnings

### WR-01: `_svd` parameter named with underscore but actively used

**File:** `fdars-core/src/spm/mfpca.rs:424,442`
**Classification:** WARNING — pre-existing, NOT a refactor regression
**Issue:** The `mfpca_extract_components` helper takes `_svd: &nalgebra::SVD<...>` with a leading underscore (the Rust convention for intentionally unused bindings), but accesses it at line 442 via `_svd.singular_values.iter()`. The `_` prefix suppresses the unused-variable lint, but the variable IS used — the name is misleading and could cause confusion about whether this parameter is relied upon.
**Fix:** Rename the parameter to `svd` to accurately reflect that it is read:
```rust
fn mfpca_extract_components(
    svd: &nalgebra::SVD<f64, nalgebra::Dyn, nalgebra::Dyn>,
    ...
) -> ... {
    let singular_values: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();
```

### WR-02: `t2_pc_significance` computes threshold before guarding against `ncomp == 0`

**File:** `fdars-core/src/spm/contrib.rs:453–455`
**Classification:** WARNING — pre-existing, NOT a refactor regression
**Issue:** When a zero-column `contributions` matrix is passed (possible if upstream logic produces `ncomp = 0`), line 453 extracts `ncomp = 0`, then line 455 computes `chi2_quantile(1.0 - alpha / 0.0_f64, 1)`. Division of a non-zero float by zero yields `f64::INFINITY`; `1.0 - INFINITY = -INFINITY`. The behavior of `chi2_quantile` on a negative probability is implementation-defined (may return NaN or a large negative number). The subsequent loop over `0..ncomp` would not execute (so no panic from indexing), but the `threshold` value is garbage.
**Fix:** Add an early return before the threshold computation:
```rust
let (n, ncomp) = contributions.shape();
if ncomp == 0 {
    return Ok(FdMatrix::zeros(n, 0));
}
let threshold = chi2_quantile(1.0 - alpha / ncomp as f64, 1);
```

### WR-03: `t2_contributions_mfpca` discards the validated `ncomp` from its own validation helper

**File:** `fdars-core/src/spm/contrib.rs:328–329`
**Classification:** WARNING — introduced by refactor (107-02)
**Issue:** `t2_mfpca_validate` was extracted to perform the dimension guard `ncomp <= eigenvalues.len()` and return `(ncomp, p)`. However, the caller immediately suppresses the first return value with `let _ = ncomp_validated`, re-deriving `ncomp` from `scores.shape()` (line 327) which yields the same integer. The guard is performed (so the invariant holds), but the API contract of the helper is immediately broken: it claims to return a useful `ncomp` but the caller signals it is meaningless. This creates maintenance confusion about whether the helper's return value is intentional or vestigial.
**Fix:** Either change `t2_mfpca_validate` to return only `p` (renaming it `t2_mfpca_validate_p`), or use `ncomp_validated` and drop the prior `let (n, ncomp) = scores.shape()` destructuring:
```rust
// Option A: return only p
fn t2_mfpca_validate(scores: &FdMatrix, mfpca: &MfpcaResult) -> Result<usize, FdarError> { ... }
// caller:
let (n, ncomp) = scores.shape();
let p = t2_mfpca_validate(scores, mfpca)?;

// Option B: use the validated ncomp (avoids re-deriving)
let (ncomp, p) = t2_mfpca_validate(scores, mfpca)?;
let n = scores.nrows();
```

## Info

### IN-01: AMEWMA loop introduces allocation via `z.row(i)` in the hot path

**File:** `fdars-core/src/spm/amewma.rs:173`
**Classification:** INFO — refactor regression (minor)
**Issue:** The extracted `amewma_ewma_step` helper requires a `&[f64]` slice for the row of `z`. The caller uses `z.row(i)` which allocates a `Vec<f64>` per iteration (see `matrix.rs:146`: "This is an O(ncols) operation because rows are not contiguous in column-major layout"). The pre-refactor code accessed `z[(i, l)]` directly in the loop body with no intermediate allocation. For the typical `ncomp <= 10` case this is trivial; for large `ncomp` with many observations it is O(n * ncomp) allocations.
**Fix:** Use `z.row_to_buf` with a pre-allocated buffer, or pass `&z` and `i` directly to `amewma_ewma_step` so it accesses `z[(i, l)]` inline:
```rust
fn amewma_ewma_step(z: &FdMatrix, row: usize, s_prev: &[f64], lambda_cur: f64, ncomp: usize) -> Vec<f64> {
    (0..ncomp).map(|l| lambda_cur * z[(row, l)] + (1.0 - lambda_cur) * s_prev[l]).collect()
}
```

### IN-02: `conditional_expectation` and `zero_pad` accept `_argvals` parameter that is not used

**File:** `fdars-core/src/spm/partial.rs:302, 499`
**Classification:** INFO — pre-existing
**Issue:** Both private helpers accept `_argvals: &[f64]` but do not use it (the `_` prefix reflects this). The integration over the domain is handled by `chart.fpca.weights`, which were baked in during Phase I. The parameter exists for API symmetry but is dead weight for these two paths. The `DomainCompletion::PartialProjection` path in `partial_projection` DOES use `argvals` (correctly). Callers pass `_argvals` to all three branches uniformly.
**Fix:** Document the no-op status in the function signatures to prevent future confusion:
```rust
/// `_argvals` is accepted for API uniformity but not used; integration weights
/// come from `chart.fpca.weights` which were computed during Phase I.
fn conditional_expectation(..., _argvals: &[f64], ...) -> ...
```

### IN-03: `profile_phase1` destructures `n` from `profile_validate` but immediately discards it

**File:** `fdars-core/src/spm/profile.rs:197, 239`
**Classification:** INFO — introduced by refactor (107-04)
**Issue:** `let (n, _m, _expected_n_windows) = profile_validate(...)` binds `n` but line 239 immediately suppresses it with `let _ = n; // n captured in profile_validate; unused after rolling window check`. The `n` binding is never read between lines 197 and 239. The rolling-window check happens inside `rolling_betas` (via `windows` construction). The simplest fix is to use `_` in the destructuring directly.
**Fix:**
```rust
let (_n, _m, _) = profile_validate(y_curves, predictors, argvals, config)?;
```
(or rename `profile_validate` to return a named struct if `n` is expected to be useful in the future).

### IN-04: `ewma_compute_t2_exact` matches `mewma_compute_exact_t2` exactly but is duplicated

**File:** `fdars-core/src/spm/ewma.rs:327–349`, `fdars-core/src/spm/mewma.rs:246–268`
**Classification:** INFO — pre-existing
**Issue:** The two helpers are byte-for-byte identical in algorithm (both implement Lowry et al., 1992, Eq. 4). They differ only in module location. This duplication was noted in `ewma.rs` comments ("This pattern is intentionally duplicated across ewma, amewma, and mewma modules for clarity") but the comment applies to the projection pipeline, not to the exact-covariance computation itself. A shared helper in `stats.rs` or a `mod.rs` would eliminate the drift risk.
**Fix:** Move the computation to `stats.rs` as `pub(super) fn ewma_exact_t2(smoothed, eigenvalues, lambda, actual_ncomp) -> Vec<f64>` and call it from both `ewma.rs` and `mewma.rs`. (Low priority — duplication is contained and both callers are stable.)

---

_Reviewed: 2026-09-14T10:30:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
