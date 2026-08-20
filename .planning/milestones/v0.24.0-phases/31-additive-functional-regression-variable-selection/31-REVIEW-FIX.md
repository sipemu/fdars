---
phase: 31-additive-functional-regression-variable-selection
fixed_at: 2026-08-20T00:00:00Z
review_path: .planning/phases/31-additive-functional-regression-variable-selection/31-REVIEW.md
iteration: 1
findings_in_scope: 7
fixed: 7
skipped: 0
status: all_fixed
---

# Phase 31: Code Review Fix Report

**Fixed at:** 2026-08-20
**Source review:** .planning/phases/31-additive-functional-regression-variable-selection/31-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 7 (CR-01, CR-02, WR-01, WR-02, WR-03, WR-04, WR-05)
- Fixed: 7
- Skipped: 0

## Fixed Issues

### CR-01: `select_group_lasso_lambda` uses training MSE — lambda selection is broken

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Replaced the in-sample MSE loop with a proper 5-fold cross-validation
implementation. The new code splits observations into `min(5, n)` folds deterministically
(fold assignment: `i % n_folds`), trains group-lasso on 4/5 of data, and evaluates
held-out prediction error on the remaining 1/5. The lambda with the lowest mean
CV prediction error is selected. This avoids the monotone-training-MSE trap where
smaller lambda always won. The `varselect_active_subset_recovery` test (which was
already verifying active-subset recovery) now passes under the CV-selected lambda.

---

### CR-02: `group_lasso_cd` fallback on singular `X_g'X_g` uses raw gradient as coefficient

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Replaced `.unwrap_or_else(|_| xty_g.clone())` with a ridge-regularized
re-solve: `(X_g'X_g + δI)β = X_g'r` where `δ = max(1e-8, 1e-6 × mean_diagonal)`.
If the ridge solve also fails, the group is zeroed out (conservative and safe). This
ensures the group-lasso threshold is applied to a properly-scaled coefficient rather
than a raw gradient vector with mismatched units.

---

### WR-01: `fregre_gkam` does not validate `y.len() == 0`, returns NaN result

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Added `if n == 0 { return Err(FdarError::InvalidDimension { parameter: "y", ... }) }`
immediately after `let n = y.len();`, consistent with the guard in `fam`, `fregre_gsam`,
and `history_index`. Added test `gkam_empty_y_returns_err` verifying the error is returned.

---

### WR-02: `FamResult`/`GsamResult` field docs lie about lengths when `scalar_covariates` is provided

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Updated doc comments for `FamResult::component_fits`, `FamResult::bandwidths`,
`GsamResult::component_fits`, and `GsamResult::bandwidths` to state the true invariant:
length = `ncomp + scalar_covariates.ncols()`, with clear description that indices 0..ncomp
are FPC components and subsequent entries are scalar covariates. Added two new tests:
`fam_scalar_covariates_component_fits_len` and `gsam_scalar_covariates_component_fits_len`
that assert `component_fits.len() == ncomp + p_scalar`.

---

### WR-03: `resolve_ncomp_additive` auto-selection picks best individual component index, not optimal component count

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Replaced the per-component individual GCV loop (which picked the index of
the single most predictive FPC score) with a forward-selection loop. For each candidate
count j = 1..=cap, the code computes the partial residual after fitting components
1..(j-1), evaluates the GCV of a 1-D NW smooth on the j-th component against that
partial residual, and accumulates the fitted component. The count j giving the best
incremental GCV is selected — correctly interpreted as "use the first j components"
rather than "the best single component has index j". Updated the function doc comment
to explain the forward-selection semantics.

---

### WR-04: `permutation_test_fam` p-value denominator uses requested `n_perm`, not successful refits

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Changed denominator from `(n_perm + 1)` to `(n_perm_success + 1)` so that
failed permutation refits (which are skipped) do not bias the p-value conservative.
Also updated `PermTestResult::p_value` doc comment to reflect the corrected formula.
Added `n_perm == 0` guard that returns `FdarError::InvalidParameter` before the main
fit, and test `perm_zero_nperm_returns_err` verifying the guard.

---

### WR-05: `select_group_lasso_lambda` doc says "prediction error" but performs no cross-validation

**Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
**Commit:** 3e5a56d5
**Applied fix:** Updated the function's doc comment to accurately state "5-fold
cross-validation" and describe the held-out evaluation approach (train on 4/5,
evaluate on 1/5). Also updated the `variable_selection` algorithm doc in step 3
to say "5-fold CV-select λ ... Each fold trains on 4/5 of the data and evaluates
held-out prediction error." This is consistent with the CR-01 fix.

---

## Skipped Issues

None — all 7 in-scope findings were addressed.

---

## Gate Results

Verification ran in the **main checkout** (workflow.use_worktrees=false).

- `cargo test -p fdars-core --features linalg,parallel additive`: **28 passed, 0 failed**
  - 4 new tests added: `gkam_empty_y_returns_err`, `fam_scalar_covariates_component_fits_len`,
    `gsam_scalar_covariates_component_fits_len`, `perm_zero_nperm_returns_err`
  - Existing `varselect_active_subset_recovery` passes under CV-selected lambda
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings`: **clean (0 warnings)**

---

_Fixed: 2026-08-20_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
