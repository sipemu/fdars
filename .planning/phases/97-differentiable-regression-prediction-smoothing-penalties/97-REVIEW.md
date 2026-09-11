---
phase: 97-differentiable-regression-prediction-smoothing-penalties
reviewed: 2026-09-11T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - fdars-core/src/scalar_on_function/fregre_lm.rs
  - fdars-core/src/scalar_on_function/tests.rs
  - fdars-core/src/scalar_on_function/mod.rs
  - fdars-core/src/smooth_basis.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: clean
---

# Phase 97: Code Review Report

**Reviewed:** 2026-09-11
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 97 adds two generic-over-`Scalar` functions — `predict_curve_generic<T>` (DOP-02)
and `penalty_value_generic<T>` (DOP-03) — plus their tests and re-exports. The
implementations are additive, all gates pass, and the coefficient-index `1+k` rule is
implemented correctly in `predict_curve_generic`. No security surface exists (pure
numerical code).

Three quality issues were found:

1. `penalty_value_generic` silently panics (rather than returning `FdarError`) when
   `coef.len()` does not match `penalty.nrows()` or `penalty.ncols()`. The RESEARCH.md
   recipe included a `debug_assert_eq!` that was dropped; in release mode the inner
   `Vec<f64>` slice index will panic with an unhelpful message instead.

2. The `predict_curve_generic` doc comment contains a self-contradictory tolerance
   claim: "within ~1e-9" on line 487 but "a ~1e-14 divergence" on line 490 for the same
   phenomenon. The correct figure is ~1e-14 (a 5-order-of-magnitude error in the stated
   bound).

3. Neither `predict_curve_generic` nor `penalty_value_generic` is listed in their
   respective module-level `#[doc]` capability indexes, leaving new public API invisible
   to readers browsing module documentation.

Two additional info items: missing `# Panics` doc sections, and missing `# Examples`
doctests on both new public functions.

---

## Warnings

### WR-01: `penalty_value_generic` — Dimension Mismatch Panics Instead of Returning `FdarError`

**File:** `fdars-core/src/smooth_basis.rs:206-216`

**Issue:** The function unconditionally uses `k = coef.len()` for both loop bounds and
indexes `penalty[(i, j)]` for all `i, j` in `0..k`. When `k > penalty.nrows()` or
`k > penalty.ncols()`, the `FdMatrix::Index` implementation falls through to a `Vec<f64>`
slice index (`&self.data[row + col * self.nrows]`) that panics in both debug and release
builds with a generic out-of-bounds message. There is no early `FdarError` guard. The
RESEARCH.md recipe (Q-1) included `debug_assert_eq!(penalty.nrows(), k)` to catch the
mismatch early; that assertion was dropped in the final implementation, and nothing
replaces it with a `Result`-returning path.

The project convention is explicit: "All public functions return `Result<T, FdarError>`
(never panic on input validation)" (CLAUDE.md §Error Handling). The generic fn currently
violates this for the dimension-mismatch case. Every other input-validating public
function in `smooth_basis.rs` (e.g. `smooth_basis`, `smooth_monotone`) returns
`Err(FdarError::InvalidDimension { … })`.

The parallel function `predict_curve_generic` inherits the same gap via
`project_scores_generic`, but that function's predecessor pattern (Phase 94) was
explicitly designed as a low-level helper that trusts its callers; `penalty_value_generic`
is a top-level public API and should hold the stronger contract.

**Fix:**

```rust
pub fn penalty_value_generic<T: Scalar>(
    coef: &[T],
    penalty: &FdMatrix,
    lambda: f64,
) -> Result<T, crate::FdarError> {
    let k = coef.len();
    if penalty.nrows() != k || penalty.ncols() != k {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "penalty",
            expected: format!("{k}×{k}"),
            actual: format!("{}×{}", penalty.nrows(), penalty.ncols()),
        });
    }
    let mut sum = T::zero();
    for i in 0..k {
        for j in 0..k {
            sum += coef[i] * T::from_f64(penalty[(i, j)]) * coef[j];
        }
    }
    Ok(T::from_f64(lambda) * sum)
}
```

Note: changing the return type to `Result<T, FdarError>` requires updating the three
test call-sites, the `lib.rs` and `prelude.rs` re-exports (type unchanged at the
function pointer level), and the existing parity/FD tests to unwrap the result. If
keeping the infallible signature is preferred for API purity (consistent with
`project_scores_generic`), add at minimum a `# Panics` doc section and a
`debug_assert_eq!(penalty.nrows(), k, …); debug_assert_eq!(penalty.ncols(), k, …);`
at the top of the function body.

---

### WR-02: `predict_curve_generic` Doc Claims "within ~1e-9" but Body Immediately States "~1e-14 Divergence"

**File:** `fdars-core/src/scalar_on_function/fregre_lm.rs:487-490`

**Issue:** The doc comment contains a direct self-contradiction:

```
// line 487:
"this matches `predict_fregre_lm` per curve within ~1e-9"
// line 490 (same comment):
"a ~1e-14 divergence, far below any meaningful tolerance"
```

"Within ~1e-9" and "~1e-14 divergence" differ by five orders of magnitude and describe
the same thing. The RESEARCH.md analysis (§Risk P-1a) establishes the correct figure:
worst-case error for m=50 evaluation points is on the order of
`m * 2 * eps ≈ 50 * 2 * 2.2e-16 ≈ 2e-14`. The bound stated in the doc ("~1e-9") is
technically not false (the actual gap is within 1e-9), but it is misleadingly loose
and contradicted within the same sentence. Any caller reading only the "~1e-9" figure
and designing a downstream tolerance around it will be unnecessarily conservative by
five orders of magnitude.

**Fix:** Replace the first figure with the correct one:

```rust
/// this matches `predict_fregre_lm` per curve within ~1e-12 (the two paths differ
/// only in floating-point accumulation order: `project_scores_generic` pre-folds
/// `rotation · weights` into one `f64`, while the batch kernel does three separate
/// multiplies — a ~1e-14 divergence per accumulation step, far below any meaningful
/// tolerance).
```

(Using ~1e-12 as the documented bound provides a comfortable safety margin over the
theoretical ~1e-14 worst case while being accurate to within two orders of magnitude.)

---

### WR-03: New Public Functions Absent from Module-Level Capability Index

**File:** `fdars-core/src/smooth_basis.rs:7-10` and
`fdars-core/src/scalar_on_function/mod.rs:11-17`

**Issue:** `penalty_value_generic` is not listed in the `smooth_basis` module's
"Key capabilities" doc block (lines 8–10), and `predict_curve_generic` is not listed
in the `scalar_on_function` module's "# Methods" doc block (lines 12–17). Both are
public exported functions. Module-level capability indexes are the first navigation
point for users browsing `cargo doc`; omitting new public API from the index makes the
functions effectively invisible to discovery.

The CLAUDE.md convention states: "Module-level documentation: always include module doc
comment with examples."

**Fix:**

In `smooth_basis.rs` module doc:
```rust
//! - [`penalty_value_generic`] — Differentiable roughness-penalty value `λ·cᵀRc` (DOP-03)
```

In `scalar_on_function/mod.rs` module doc:
```rust
//! - [`predict_curve_generic`]: Differentiable per-curve FPCR prediction (DOP-02)
```

---

## Info

### IN-01: Missing `# Panics` Doc Section on Both New Public Functions

**File:** `fdars-core/src/scalar_on_function/fregre_lm.rs:476-494` and
`fdars-core/src/smooth_basis.rs:191-204`

**Issue:** Both `predict_curve_generic` and `penalty_value_generic` can panic at
runtime if their array inputs have mismatched lengths. Specifically:
- `predict_curve_generic` panics if `curve.len()` differs from `fit.fpca.mean.len()` or
  if `fit.ncomp` exceeds `fit.fpca.rotation.ncols()` (both via `project_scores_generic`).
- `penalty_value_generic` panics if `coef.len() > penalty.nrows()` or
  `coef.len() > penalty.ncols()` (via the `FdMatrix` slice index — see WR-01).

Neither doc comment includes a `# Panics` section to document these conditions. The
CLAUDE.md convention says: "Public functions: `Result<T, FdarError>` (never panic on
input validation)" — so either these should return `Result` (see WR-01 for
`penalty_value_generic`) or the panics must be explicitly documented.

**Fix:** Add `# Panics` documentation sections:

```rust
/// # Panics
///
/// Panics if `curve.len()` does not match the training data dimension
/// (`fit.fpca.mean.len()`), or if `fit.ncomp` exceeds `fit.fpca.rotation.ncols()`.
```

---

### IN-02: Missing `# Examples` Doctest on Both New Public Functions

**File:** `fdars-core/src/scalar_on_function/fregre_lm.rs:476-510` and
`fdars-core/src/smooth_basis.rs:191-216`

**Issue:** Both `predict_curve_generic` and `penalty_value_generic` are public
crate-root exports with no `# Examples` doctest. The adjacent `predict_fregre_lm`
(lines 423–444) and `bspline_penalty_matrix` have examples. The CLAUDE.md convention
states: "Public item documentation: required for all public types, functions, and
fields." With 209 passing doctests already in the suite, adding examples would both
document usage and provide coverage.

**Fix:** Add a minimal `# Examples` block to each doc comment, similar to the pattern
used by `predict_fregre_lm`:

```rust
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::{fregre_lm, predict_curve_generic};
///
/// let (n, m) = (20, 30);
/// // ... (construct data, fit, call predict_curve_generic::<f64>) ...
/// ```
```

---

_Reviewed: 2026-09-11_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
