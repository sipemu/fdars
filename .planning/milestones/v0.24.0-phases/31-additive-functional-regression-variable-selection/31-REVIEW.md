---
phase: 31-additive-functional-regression-variable-selection
reviewed: 2026-08-20T00:00:00Z
depth: deep
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/scalar_on_function/additive.rs
  - fdars-core/src/scalar_on_function/mod.rs
  - fdars-core/src/lib.rs
findings:
  critical: 2
  warning: 5
  info: 2
  total: 9
status: issues_found
---

# Phase 31: Code Review Report

**Reviewed:** 2026-08-20
**Depth:** deep
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Phase 31 adds 2,818 lines across one new file (`additive.rs`) and two modified barrel files. The FAM, GKAM, and GSAM estimators are structurally sound: panic-safety at public boundaries is largely correct, the one-pass FPC approach is mathematically justified, GKAM backfitting is correctly iterative, and the NW denominator guard (`1e-15`) is applied consistently in hand-coded kernel loops. Import wiring, `#[must_use]`, `#[non_exhaustive]`, and serde gating are all applied correctly.

Two correctness defects require fixing before this code ships. The first renders the auto-lambda selection in `variable_selection` completely inoperative (always selects the most overfit model). The second introduces numerical corruption in `group_lasso_cd` when the per-group design matrix is singular. Additionally, `fregre_gkam` fails to validate `y.len() == 0` and would silently return a result struct with `NaN` fields.

---

## Critical Issues

### CR-01: `select_group_lasso_lambda` uses training MSE — lambda selection is broken

**File:** `fdars-core/src/scalar_on_function/additive.rs:1444-1493`

**Issue:** The function is documented as "LOO-proxy CV" and "prediction error", but it evaluates training MSE (in-sample squared error averaged over the same data used to fit the model). For group lasso, training MSE is monotonically non-increasing as lambda decreases. Therefore `select_group_lasso_lambda` always returns the smallest lambda in the grid (most overfit model), making the CV selection completely inoperative. When `config.lambda == 0.0`, `variable_selection` is supposed to select a regularized, sparse solution; instead it always produces the densest possible fit — identical to or worse than OLS in terms of sparsity.

**Fix:** Replace training MSE with leave-one-out cross-validation error. A minimal LOO proxy reuses the existing `select_bandwidth_loo` pattern — fit without observation `i`, predict at `i`, accumulate squared error:

```rust
fn select_group_lasso_lambda(
    y: &[f64],
    mu_y: f64,
    n: usize,
    score_groups: &[Vec<Vec<f64>>],
    k_sizes: &[usize],
    lambda_max: f64,
    n_grid: usize,
    max_iter: usize,
    epsilon: f64,
    scalar_covariates: Option<&FdMatrix>,
) -> f64 {
    let grid_size = n_grid.max(2);
    let mut best_lambda = lambda_max * 0.1;
    let mut best_cv_err = f64::INFINITY;

    for gi in 0..grid_size {
        let frac = (gi as f64 + 1.0) / grid_size as f64;
        let lam = lambda_max * 0.01_f64.powf(1.0 - frac);
        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - mu_y).collect();

        // Full-data fit to get coefficients
        if let Ok((coeffs, _, _)) = group_lasso_cd(
            y, &y_centered, mu_y, n, score_groups, k_sizes,
            lam, max_iter, epsilon, scalar_covariates,
        ) {
            let big_p = score_groups.len();
            // Approximate LOO error via hat matrix diagonal (or k-fold if n is large)
            // Minimal version: evaluate on held-out observation i by refitting on n-1 obs
            // Practical proxy: use (residual_i / (1 - h_ii)) approximation, or
            // use k-fold (k=5) split for n >= 20, LOO for n < 20.
            let fitted = compute_varselect_fitted(
                n, mu_y, score_groups, &coeffs, scalar_covariates, big_p,
            );
            // k-fold CV (k=5): split into 5 folds, refit on 4, evaluate on 1
            let cv_err = kfold_cv_error(
                y, mu_y, score_groups, k_sizes, lam, max_iter, epsilon,
                scalar_covariates, n, 5,
            );
            if cv_err < best_cv_err {
                best_cv_err = cv_err;
                best_lambda = lam;
            }
        }
    }
    best_lambda
}
```

The simplest correct fix that avoids a full rewrite is to replace the training MSE block with a 5-fold CV loop (refitting on 4/5 of data and predicting the held-out 1/5). If k-fold is too expensive in CI, the lambda should instead be computed analytically as a fraction of `lambda_max` using the standard glmnet convention (`lambda = lambda_max * alpha_grid[i]` with a fixed grid like `[1, 0.5, 0.1, 0.05, 0.01] * lambda_max`), and documented that no CV is performed.

---

### CR-02: `group_lasso_cd` fallback on singular `X_g'X_g` uses raw gradient as coefficient

**File:** `fdars-core/src/scalar_on_function/additive.rs:1574-1575`

**Issue:**

```rust
let beta_ols = crate::linalg::cholesky_solve(&xtx_g, &xty_g, k_g)
    .unwrap_or_else(|_| xty_g.clone()); // fall back to gradient step on singular
```

When `cholesky_solve` fails (singular or near-singular `X_g'X_g`), the fallback sets `beta_ols = xty_g` where `xty_g = X_g' * partial_residual`. This vector has units of (predictor units × response units), not (response units). Applying the group-lasso threshold to it treats a raw gradient as a coefficient, which can produce arbitrarily large, incorrect coefficient updates — potentially corrupting all subsequent coordinate descent iterations.

Singularity of `X_g'X_g` arises when the k_g FPC score columns are numerically dependent (e.g., near-zero singular values from SVD, or k_g >= n). This is not a pathological case; it can occur with moderate n and the default `ncomp=3`.

**Fix:** On Cholesky failure, set `beta_ols` to zero (conservative shrink-to-zero) or apply a small ridge regularization:

```rust
let beta_ols = crate::linalg::cholesky_solve(&xtx_g, &xty_g, k_g)
    .unwrap_or_else(|_| {
        // Cholesky failed: X_g is (near-)singular.
        // Add ridge regularization: solve (X_g'X_g + delta*I) beta = X_g'partial
        let delta = 1e-6 * xtx_g.iter().enumerate()
            .filter(|(idx, _)| idx % (k_g + 1) == 0)
            .map(|(_, &v)| v.abs())
            .sum::<f64>()
            .max(1e-8);
        let mut xtx_ridge = xtx_g.clone();
        for d in 0..k_g {
            xtx_ridge[d * k_g + d] += delta;
        }
        crate::linalg::cholesky_solve(&xtx_ridge, &xty_g, k_g)
            .unwrap_or_else(|_| vec![0.0; k_g]) // final fallback: zeros
    });
```

---

## Warnings

### WR-01: `fregre_gkam` does not validate `y.len() == 0`, returns NaN result

**File:** `fdars-core/src/scalar_on_function/additive.rs:543`

**Issue:** `fregre_gkam` derives `n` from `y.len()`. If called with an empty `y` slice and a predictor with zero rows, `n == 0` passes all dimension checks (since `pred.nrows() != n` evaluates `0 != 0` = false). The backfitting loop produces empty `fitted_values`, and `compute_r_squared` divides by zero when computing `y_mean` on an empty slice, producing `NaN` in `r_squared`. The function returns `Ok` with a partially corrupt result struct.

All other public functions (`fam`, `fregre_gsam`, `history_index`) explicitly check `n == 0`. This one was missed.

**Fix:** Add the missing check immediately after `let n = y.len();`:

```rust
let n = y.len();
if n == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "y",
        expected: "at least 1 observation".to_string(),
        actual: "0".to_string(),
    });
}
```

---

### WR-02: `FamResult`/`GsamResult` field docs lie about lengths when `scalar_covariates` is provided

**File:** `fdars-core/src/scalar_on_function/additive.rs:154, 158, 200, 205`

**Issue:** `FamResult::component_fits` is documented as `"ncomp × n"` and `FamResult::bandwidths` as `"length ncomp"`. When `scalar_covariates` is provided, both have length `ncomp + p_scalar`, not `ncomp`. The `ncomp` field in the result still reflects only the FPC component count, so a caller checking `result.component_fits.len() == result.ncomp` finds them unequal. Similarly for `GsamResult`. No test covers this case.

**Fix:** Update the field doc comments to reflect the true invariant:

```rust
/// Component fits f_k(ξ_k) for each observation. Length = `ncomp + scalar_covariates.ncols()`.
/// Indices 0..ncomp correspond to FPC components; subsequent entries to scalar covariates.
pub component_fits: Vec<Vec<f64>>,

/// Per-component optimal bandwidth. Length = `ncomp + scalar_covariates.ncols()`.
pub bandwidths: Vec<f64>,
```

Add a test that exercises `fam` and `fregre_gsam` with `scalar_covariates = Some(...)` and asserts `component_fits.len() == ncomp + p_scalar`.

---

### WR-03: `resolve_ncomp_additive` auto-selection picks best individual component index, not optimal component count

**File:** `fdars-core/src/scalar_on_function/additive.rs:231-249`

**Issue:** The auto-select loop iterates `k = 1..=cap` and tracks `best_ncomp = k` where the GCV of a 1D NW smooth of `y` on `xi_k` alone is minimized. This selects the INDEX of the single most-predictive FPC score as the number of components to use. These are different concepts: if `xi_2` is the most individually predictive score, the algorithm returns `ncomp = 2` meaning "use both xi_1 and xi_2", but xi_1 may be uninformative. The correct procedure is to evaluate the FAM fit quality as a function of the number of components retained (e.g., nested GCV), not to use individual component GCV as a proxy for component count.

As a secondary issue, when `ncomp = 0` is triggered, `resolve_ncomp_additive` runs `fdata_to_pc_1d` and then `fam` / `fregre_gsam` runs it again (double FPCA computation, O(nm²) wasted work).

**Fix (minimal):** Change the loop to accumulate GCV over the joint FAM fit (sequential pass through all k=1..=j components) rather than per-component individual GCV:

```rust
// For each candidate ncomp j, evaluate the j-component FAM partial residual GCV
let mut best_ncomp = 1usize;
let mut best_gcv = f64::INFINITY;
let mut component_fits: Vec<Vec<f64>> = vec![vec![0.0; n]; cap];
for j in 1..=cap {
    let xi_j: Vec<f64> = (0..n).map(|i| fpca_full.scores[(i, j - 1)]).collect();
    // Partial residual for component j given prior fits
    let partial: Vec<f64> = (0..n).map(|i| {
        y[i] - y.iter().sum::<f64>() / n as f64
            - (0..j-1).map(|k| component_fits[k][i]).sum::<f64>()
    }).collect();
    let bw = optim_bandwidth(&xi_j, &partial, None, CvCriterion::Gcv, kernel, n_grid);
    let gcv_j = bw.value;
    component_fits[j - 1] = nadaraya_watson(&xi_j, &partial, &xi_j, bw.h_opt, kernel)
        .unwrap_or_else(|_| vec![0.0; n]);
    if gcv_j < best_gcv {
        best_gcv = gcv_j;
        best_ncomp = j;
    }
}
```

---

### WR-04: `permutation_test_fam` p-value denominator uses requested `n_perm`, not successful refits

**File:** `fdars-core/src/scalar_on_function/additive.rs:1740`

**Issue:**

```rust
let p_value = (n_ge + 1) as f64 / (n_perm + 1) as f64;
```

`n_perm` is the number of requested permutations; `n_perm_success` is the number that returned `Ok`. When permutation refits fail (e.g., bandwidth selection encounters degenerate shuffled data), `n_ge` only counts among successes but the denominator counts all `n_perm` attempts. This produces a conservatively biased p-value. The result struct exposes `n_perm_success` so callers can detect the discrepancy, but the p-value itself is computed incorrectly with respect to the actual null distribution.

A secondary issue: `n_perm = 0` is accepted without validation, returning `p_value = 1.0` with an empty `null_statistics` — a misleading result.

**Fix:**

```rust
// Use actual successful refits in both numerator and denominator
let p_value = (n_ge + 1) as f64 / (n_perm_success + 1) as f64;
```

And add a guard for `n_perm == 0`:

```rust
if perm_config.n_perm == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "perm_config.n_perm",
        message: "n_perm must be >= 1 for a meaningful permutation test".to_string(),
    });
}
```

---

### WR-05: `select_group_lasso_lambda` is documented as "LOO-proxy CV" but performs no cross-validation

**File:** `fdars-core/src/scalar_on_function/additive.rs:1439-1442`

**Issue:** The function's doc comment says "Evaluates a grid of lambda values… and returns the lambda with the lowest mean squared **prediction** error." The word "prediction" implies out-of-sample evaluation. The implementation computes in-sample training MSE. This misleads any reader trying to understand or modify the CV logic. Even if CR-01 is fixed, the documentation should be updated to accurately describe whatever method is chosen.

**Fix:** Update the doc comment to accurately state what is computed:

```rust
/// Select lambda via 5-fold cross-validation for group lasso.
///
/// Evaluates a geometric grid of lambda values from `0.01·lambda_max` to `lambda_max`
/// and returns the lambda with the lowest 5-fold cross-validated mean squared error.
```

---

## Info

### IN-01: `resolve_ncomp_additive` calls `fdata_to_pc_1d` twice when `config.ncomp == 0`

**File:** `fdars-core/src/scalar_on_function/additive.rs:238, 463`

**Issue:** When `config.ncomp == 0`, `resolve_ncomp_additive` runs FPCA internally (line 238) to compute GCV scores. Then `fam` / `fregre_gsam` runs FPCA a second time (line 463) with the resolved `ncomp`. Each `fdata_to_pc_1d` call costs O(nm²) for the SVD. For large n or m, this doubles the FPCA cost of the auto-select path.

**Fix:** Refactor `resolve_ncomp_additive` to return `(ncomp, Option<FpcaResult>)` so the FPCA result from auto-selection can be reused by the caller.

---

### IN-02: `PermTestResult` missing serde feature gate inconsistent with RESEARCH.md note

**File:** `fdars-core/src/scalar_on_function/additive.rs:1022-1034`

**Issue:** `PermTestResult` does not derive `serde::Serialize/Deserialize` even conditionally, unlike every other result struct in the file. The PATTERNS.md notes this is an acceptable deviation ("apply the gate only if the planner decides to include it") but the struct is re-exported at the crate root, so users enabling the `serde` feature cannot serialize permutation test results while they can serialize all other result types. This is a minor API inconsistency.

**Fix:** Add the feature gate for consistency, or add a crate-level doc note explaining why `PermTestResult` is excluded from serde:

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermTestResult {
    // ...
}
```

---

_Reviewed: 2026-08-20_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_
