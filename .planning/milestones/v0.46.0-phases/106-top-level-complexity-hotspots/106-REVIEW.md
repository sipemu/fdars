---
phase: 106-top-level-complexity-hotspots
reviewed: 2026-09-14T10:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - fdars-core/src/wavelet/regression.rs
  - fdars-core/src/clustering_advanced.rs
  - fdars-core/src/boosting_regression/boost_fofr.rs
  - fdars-core/src/boosting_regression/bayesian.rs
  - fdars-core/src/pace_fpca.rs
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 106: Code Review Report

**Reviewed:** 2026-09-14T10:00:00Z
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Phase 106 was a behavior-preserving complexity refactor: 8 large functions were decomposed into private helpers via verbatim block extraction. The primary risk of such a refactor is silently altering RNG draw order, floating-point accumulation sequence, or captured variable semantics.

**No critical bugs were found.** All extracted helpers are correctly `fn` (private), not `pub fn`. RNG draw order is preserved in both `kcfc_kmeans_pp_init` and `funfem_kmeans_pp_init` (both document the byte-identical requirement and implement it correctly). The Gibbs sampler loop structure — outer `iter`, inner `kk` — is preserved verbatim. The `pace_fpca` BLUP closure-to-named-function extraction correctly converts all captures to explicit parameters, maintaining `Send`-compatibility with the `iter_maybe_parallel!` macro. Floating-point accumulation loops are unchanged in structure (loop bounds, operand order).

Four warnings were found — all pre-existing design issues surfaced by the review rather than introduced by the refactor. Three info-level observations complete the picture.

## Warnings

### WR-01: `max_iter == 0` is not validated for `kcfc_cluster`, `funfem_cluster`, or `align_cluster_fd`

**File:** `fdars-core/src/clustering_advanced.rs:406, 863, 1574`
**Issue:** `KcfcConfig::max_iter`, `FunFemConfig::max_iter`, and `AlignClusterConfig::max_iter` have no validation guard. When `max_iter == 0`, the `for _iter in 0..config.max_iter` loop runs zero times. For `kcfc_cluster` this returns `KcfcResult` with all `fpca_models` as `None` and all reconstruction errors as `f64::INFINITY`, and `cluster` retains the k-means++ initial assignment — a silently incorrect result returned as `Ok`. For `funfem_cluster` the GMM never runs and `resp` retains uniform responsibility values, again silently wrong.

This is consistent with `BayesianConfig::n_iter == 0` being validated (line 135 in `bayesian.rs`) but the equivalent parameter in the clustering functions going unchecked.

**Fix:**
```rust
// In kcfc_validate (clustering_advanced.rs, after the ncomp check):
if config.max_iter == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "max_iter",
        message: "max_iter must be >= 1".to_string(),
    });
}

// Same pattern in funfem_validate and at the top of align_cluster_fd.
```

---

### WR-02: `n_iter * thin` integer overflow in `bayesian_fosr` is unvalidated

**File:** `fdars-core/src/boosting_regression/bayesian.rs:400`
**Issue:** `total = config.burn_in + config.n_iter * config.thin` is a `usize` multiplication with no overflow guard. On a 64-bit target, overflow requires unrealistically large values, but the parameters `n_iter` and `thin` have no upper bound and `thin == 0` is the only checked edge case. If a caller passes `n_iter = usize::MAX / 2` and `thin = 3`, the loop count silently wraps, potentially running for far fewer or far more iterations than intended.

**Fix:** Validate a reasonable upper bound or use checked arithmetic:
```rust
let n_iter_total = config.n_iter
    .checked_mul(config.thin)
    .and_then(|x| x.checked_add(config.burn_in))
    .ok_or_else(|| FdarError::InvalidParameter {
        parameter: "n_iter/thin/burn_in",
        message: "n_iter * thin + burn_in overflows usize".to_string(),
    })?;
let total = n_iter_total;
```

---

### WR-03: `FunFemConfig::tol` is unvalidated — NaN or negative values silently alter convergence behaviour

**File:** `fdars-core/src/clustering_advanced.rs:933`
**Issue:** The convergence check `if _iter > 0 && delta < config.tol` is never reached when `config.tol < 0.0` (since `delta >= 0.0` always), causing `funfem_cluster` to silently run all `max_iter` iterations regardless of actual convergence. When `config.tol` is `f64::NAN`, `delta < NaN` is always `false` — same silent behavior. Neither value is caught by `funfem_validate`.

By contrast, `BayesianConfig` validates all critical numeric parameters (tau2, ig_a0, ig_b0), and `elastic_net_cd` validates its `tol` parameter. `funfem_cluster` is the outlier.

**Fix:**
```rust
// In funfem_validate, after the ncomp check:
if !config.tol.is_finite() || config.tol < 0.0 {
    return Err(FdarError::InvalidParameter {
        parameter: "tol",
        message: format!("tol must be finite and >= 0, got {}", config.tol),
    });
}
```

---

### WR-04: `kcfc_cluster` computes `reconstruction_errors` twice on every convergence iteration

**File:** `fdars-core/src/clustering_advanced.rs:419-432`
**Issue:** On the iteration where `changed == false` (convergence), `kcfc_reconstruction_errors` is called twice in sequence: once to obtain the matrix used for `kcfc_reassign` (line 419-420), and immediately again to obtain `reconstruction_errors_final` (line 427-428) which is returned in the result. Both calls use the same `fpca_models` and `cluster`, so the result is always identical. The redundant call is O(n × k × m²) — nontrivial for large inputs.

This is a logic error introduced during extraction: the original code could avoid the second call by reusing the first matrix. As written, the intermediate `reconstruction_errors` variable is shadowed by `reconstruction_errors_final` in the early-return path, making the first call's value immediately orphaned.

**Fix:** Reuse the first computation:
```rust
if !changed {
    converged = true;
    return Ok(KcfcResult {
        cluster,
        fpca_models,
        reconstruction_errors,  // already correct; no need to recompute
        iterations,
        converged,
    });
}
```

## Info

### IN-01: `ncomp == 0` validation is placed in `pace_fpca_validate_dims` rather than `pace_fpca_validate_config`

**File:** `fdars-core/src/pace_fpca.rs:381-386`
**Issue:** `ncomp` is a `PaceFpcaConfig` field, so its validation logically belongs in `pace_fpca_validate_config`. It is currently checked inside `pace_fpca_validate_dims` (the dimension validator). This misclassification does not affect correctness because both validators are always called together via `pace_fpca_validate`, but it makes future maintenance harder: a reader extending `pace_fpca_validate_config` would not expect to find a config-parameter check in the dimension validator.

**Fix:** Move the `ncomp == 0` check to `pace_fpca_validate_config` alongside the other scalar config guards (`bandwidth`, `sigma2`, `alpha`, `work_grid`).

---

### IN-02: `_n_k_soft` parameter is accepted but entirely unused in `funfem_subspace_update`

**File:** `fdars-core/src/clustering_advanced.rs:1189`
**Issue:** `funfem_subspace_update` accepts `_n_k_soft: &[f64]` (the responsibility-weighted effective cluster sizes from the projected subspace). The parameter is correctly suppressed with a leading underscore, but it was presumably included as a hook for a more principled between/within scatter weighting. The function instead uses `n_k_new` (the score-space effective sizes from the post-E-step responsibilities). The discrepancy is harmless but signals incomplete implementation intent.

**Fix:** If `_n_k_soft` has no future role, remove the parameter and its call-site argument. If it is intended for a future correction to the between-scatter weight, add a comment explaining the deferral.

---

### IN-03: `sigma2_mean` in `BayesianFosrResult` is documented as "posterior mean σ²(t)" but computed as mean squared residual

**File:** `fdars-core/src/boosting_regression/bayesian.rs:327-339`, `fdars-core/src/boosting_regression/mod.rs:302-303`
**Issue:** The `BayesianFosrResult::sigma2_mean` field is documented as "Posterior mean σ²(t) across the response grid" but is computed in `bayesian_fosr_summaries` as the average squared residual `(1/n) Σ_i (Y_i(t) - fitted_i(t))²`. This is the posterior-mean MSE, not the posterior-mean noise variance (which would aggregate the per-component σ²_k draws weighted by the eigenfunction expansion). The distinction matters: the Gibbs posterior for σ²_k conditions on FPC scores, not on functional residuals at the response grid.

The computation is internally consistent and the test passes (`sigma2_mean > 0`), but users relying on this field for Bayesian uncertainty quantification will get a frequentist MSE rather than the Bayesian posterior quantity the documentation promises.

**Fix:** Either rename the field to `mse_t` (mean squared error at each time point) and update the doc comment, or replace the computation with an actual aggregation of posterior σ² draws:
```rust
// Document accurately:
/// Pointwise mean-squared error `(1/n) Σᵢ (Yᵢ(t) − ŷᵢ(t))²`
/// using the posterior-mean fitted values. This is a frequentist
/// diagnostic, not the Bayesian posterior of σ².
pub sigma2_mean: Vec<f64>,
```

---

_Reviewed: 2026-09-14T10:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
