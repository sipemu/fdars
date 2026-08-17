---
phase: 24-concurrent-varying-coefficient-regression
reviewed: 2026-08-17T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/concurrent_regression.rs
  - fdars-core/src/lib.rs
findings:
  critical: 1
  warning: 1
  info: 0
  total: 2
status: issues_found
---

# Phase 24: Code Review Report

**Reviewed:** 2026-08-17
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

The new `concurrent_regression` module is well-structured and follows project
conventions correctly: `Result<T, FdarError>` return, `#[must_use]`, `#[non_exhaustive]`
+ standard derives on the result type, gated serde, and an additive (non-breaking)
re-export in `lib.rs`. The two-step estimation approach (pointwise OLS then
local-linear kernel smoothing via `smoothing::solve_gaussian_pub` and
`smoothing::local_linear`) is mathematically correct. Column-major `FdMatrix`
indexing is used consistently and all buffer accesses are bounds-safe by
construction. The parallel closure correctly allocates per-task-local `xtx` /
`xty` / `row` buffers and iterates over an `IndexedParallelIterator`, so
`collect()` preserves order and there is no shared mutable state. Determinism is
guaranteed (no per-thread RNG needed here). The ridge stabiliser formula
`eps = 1e-10 * (xtx[0] + 1.0)` correctly equals `1e-10 * (n + 1)` because the
intercept column contributes `1.0^2` per observation.

Two real defects were found: one correctness blocker (NaN bandwidth silently
produces wrong output) and one missing guard that the rest of the codebase
treats as mandatory (underdetermined system).

## Critical Issues

### CR-01: NaN bandwidth bypasses validation and silently produces all-zero coefficients

**File:** `fdars-core/src/concurrent_regression.rs:142`

**Issue:** The guard `if bandwidth <= 0.0` does not catch `f64::NAN`. Under
IEEE 754, `NaN <= 0.0` evaluates to `false`, so a caller passing `f64::NAN`
(e.g., computed from a ratio of zero variances) passes the check and receives
`Ok(ConcurrentRegrResult)`. The NaN then propagates into
`smoothing::local_linear`, where the kernel weights (`exp(-0.5 * NaN^2)`)
become `NaN`, all weight-sum comparisons (`s0 > 1e-10`, `det.abs() > 1e-10`)
evaluate to `false`, and the function falls through to return `0.0` for every
grid point. The result is a structurally valid `ConcurrentRegrResult` whose
`intercept`, `beta_curve`, `fitted`, and `residuals` are all zeros — silently
wrong. The same silent failure occurs for `f64::NEG_INFINITY` (negative
infinity) because `-Inf <= 0.0` is `true` and IS caught, but `+f64::INFINITY`
also passes (infinity > 0) and produces over-smoothed (constant-mean) output
without an error.

**Fix:** Replace the comparison with one that rejects any non-finite value:

```rust
// Before (line 142):
if bandwidth <= 0.0 {

// After:
if !(bandwidth > 0.0) {
    // Catches NaN (NaN > 0.0 is false) as well as zero and negative values.
```

Or equivalently and more explicitly:

```rust
if bandwidth <= 0.0 || !bandwidth.is_finite() {
```

The same pattern should be applied to `smoothing::local_linear` and
`smoothing::nadaraya_watson` (those are not in this diff's scope, but they
share the same defect as callees).

## Warnings

### WR-01: Missing guard for underdetermined pointwise system (n <= p)

**File:** `fdars-core/src/concurrent_regression.rs:110`

**Issue:** The function validates `n >= 2` but does not validate `n > p`
(number of predictors). When `n <= p`, each per-column design matrix
`X_j ∈ R^{n×(p+1)}` is underdetermined: the normal equations `X^T X` have rank
at most `n < p+1`, so the system is rank-deficient by construction. The ridge
stabiliser `eps = 1e-10 * (n + 1)` is a numerical artefact (~`3e-10` for
`n = 2`), not a statistical regulariser. The function returns `Ok()` with
coefficient estimates that are determined almost entirely by the ridge rather
than the data, with no indication to the caller that the result is meaningless.
Other regression functions in the project enforce a minimum: `fof_regression`
requires `n >= 3`, and `fregre_cv` / `fregre_lm_multi_cv` check `n >= n_folds`.
The doc comment mentions `n > p + 1` only as an advisory note in `# Notes`,
which is insufficient to protect callers.

**Fix:** After `p` is computed (line 166), add:

```rust
if n <= p {
    return Err(FdarError::InvalidDimension {
        parameter: "response",
        expected: format!(
            "at least {} rows (more observations than predictors p={})",
            p + 1,
            p
        ),
        actual: format!("{n}"),
    });
}
```

Update the `# Errors` doc comment to document this new error condition. Remove
or demote the `# Notes` advisory about `n > p + 1`.

---

_Reviewed: 2026-08-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
