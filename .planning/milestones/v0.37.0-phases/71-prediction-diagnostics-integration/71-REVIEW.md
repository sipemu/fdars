---
phase: 71-prediction-diagnostics-integration
reviewed: 2026-09-04T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/wavelet/regression.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 0
  warning: 2
  info: 2
  total: 4
status: resolved
resolution:
  WR-01: resolved
  WR-02: resolved
  IN-01: resolved
  IN-02: resolved
  commit: 155a0a85
---

> **Resolution (2026-09-04, commit `155a0a85`):** All four findings addressed.
> WR-01 — both `WcrResult::predict` and `WnetResult::predict` now guard
> `new.nrows() == 0` up front, returning `InvalidDimension { parameter: "new", .. }`
> before delegating to `curves_to_coeff_design`; `# Errors` docs expanded; zero-row
> predict tests added for both (assert the error names `"new"`, no panic).
> WR-02 — `WcrResult::intercept` doc now states the affine convention (centering
> offset folded in); `WnetResult::intercept` doc aligned.
> IN-01 — module doctest now asserts `predict` reproduces `fitted_values`
> (`|Δ| < 1e-7`, matching the review's suggested tolerance for the small
> rank-deficient default fit; the full-rank unit test still holds ≤1e-8).
> IN-02 — redundant `#[must_use = "..."]` removed from both `Result`-returning
> `predict` methods. All gates green (wavelet lib tests, doctest, clippy
> `--all-targets`, `fmt --check`).

# Phase 71: Code Review Report

**Reviewed:** 2026-09-04
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 71 adds `predict`, `beta_t()`, `coefficient_function()`, and `fitted_values()` accessors to both `WcrResult` and `WnetResult`, re-exports the full wavelet surface at the crate root and prelude, and adds the module-level doctest. The core correctness concern is the wcr intercept deviation (folding the centering offset into the affine intercept so predict reproduces `fitted_values` exactly); that math is verified correct and the deviation sign is right. The wnet path uses `elastic_net_cd`'s own affine intercept directly and needs no adjustment — also correct.

All 36 regression unit tests pass. Clippy is clean under `--all-targets --features linalg,parallel -D warnings`. The doctest compiles and passes. The re-exports are additive, non-shadowing, and serde-independent.

Two warnings are raised: a misleading error message on the zero-row path and an underspecified field doc comment. Two info items cover the doctest's weak correctness assertion and a redundant `#[must_use]` on a `Result`-returning fn.

No critical/blocker issues found.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `predict` surfaces a misleading `parameter: "data"` error on zero-row input

**File:** `fdars-core/src/wavelet/regression.rs:640-641` (WcrResult::predict; identical at line 1266-1267 for WnetResult)

**Issue:** When a caller passes `new` with 0 rows and matching `ncols`, the column-count guard passes. Control reaches `curves_to_coeff_design`, which calls `decompose_matrix`, which returns `FdarError::InvalidDimension { parameter: "data", ... }`. The error text refers to `"data"` — an internal parameter name — not `"new"`, the caller's parameter. The doc comment for predict does not mention the zero-row rejection either, so a user who gets this error has no obvious hint that `new.nrows() == 0` is the cause.

**Fix:** Add an explicit zero-row guard before calling `curves_to_coeff_design` in both `WcrResult::predict` and `WnetResult::predict`:

```rust
pub fn predict(&self, new: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    let train_m = self.beta_t.len();
    if new.nrows() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "new",
            expected: "at least 1 row (curve)".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if new.ncols() != train_m {
        // ... existing check
    }
    // ...
}
```

Also document the zero-row error in the `# Errors` section of both predict methods.

---

### WR-02: `WcrResult::intercept` field doc does not state the affine convention

**File:** `fdars-core/src/wavelet/regression.rs:121`

**Issue:** The field is documented as `/// Intercept α.` (line 121). Phase 71 silently rewrites this field from the raw OLS intercept to the *affine* coefficient-space intercept (`α_affine = α_ols - Σ_j mean_j * w_j`) so that `predict` satisfies `ŷ_i = intercept + Σ_j X[i,j]*coeff_weights[j]` directly. Any user who reads `WcrResult::intercept` and tries to reconstruct predictions by hand (without using `predict`) needs to know which convention the stored value follows. The current one-liner is ambiguous.

**Fix:** Update the field doc comment to describe the affine convention explicitly:

```rust
/// Affine intercept α such that `ŷ_i = intercept + Σ_j design[i,j] · coeff_weights[j]`
/// holds for each training observation (matching the [`predict`](WcrResult::predict) formula).
/// Note: this is NOT the raw OLS intercept from the score regression; the centering
/// offset `Σ_j col_mean_j · coeff_weights[j]` has been folded in.
pub intercept: f64,
```

## Info

### IN-01: Doctest asserts lengths only — does not verify predict self-consistency

**File:** `fdars-core/src/wavelet/regression.rs:46-52` (module doctest)

**Issue:** The doctest calls `fit.predict(&data)` and `fit.fitted_values()` but only checks `preds.len() == fitted.len()` and `beta.len() == m`. It does not assert that `preds[i] ≈ fitted[i]` (the self-consistency property the intercept deviation was specifically added to guarantee). A reader looking at the doctest as a correctness witness cannot tell whether predict actually reproduces training fits. The unit test `wcr_predict_reproduces_training_fitted` does cover this, but the doctest is the public-facing example.

**Fix (low priority):** Add a self-consistency assertion to the doctest:

```rust
for (p, f) in preds.iter().zip(fitted) {
    assert!((p - f).abs() <= 1e-7, "predict diverges from fitted: {p} vs {f}");
}
```

---

### IN-02: Redundant `#[must_use = "..."]` on `Result`-returning `predict` methods

**File:** `fdars-core/src/wavelet/regression.rs:628` (WcrResult::predict), line 1254 (WnetResult::predict)

**Issue:** Both predict methods are annotated `#[must_use = "prediction result should not be discarded"]`. Since the return type is `Result<Vec<f64>, FdarError>`, and `Result` is already marked `#[must_use]` in std, this attribute is redundant. It will not cause a compile error (Rust allows double must-use), but it adds noise and may surprise future maintainers. The project's convention (`#[must_use]` on expensive-but-non-Result fns like `wcr` and `wnet`) does not apply to accessor/method calls like `predict`. Clippy does not warn on this, but a lint audit would flag it.

**Fix:** Change the annotation to `#[must_use]` (no message) or remove it, consistent with the accessor methods `beta_t()` and `fitted_values()` on the same structs.

---

_Reviewed: 2026-09-04_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
