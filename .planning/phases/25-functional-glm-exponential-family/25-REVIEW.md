---
phase: 25-functional-glm-exponential-family
reviewed: 2026-08-17T12:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/src/scalar_on_function/glm.rs
  - fdars-core/src/scalar_on_function/mod.rs
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

> **Iteration 2 resolution:** all 8 original findings verified fixed. The one new
> WARNING raised during iteration 2 (module-doc blanket claim that dispersion φ is
> not estimated, made inaccurate by the WR-01 fix) was resolved by narrowing the doc
> to state φ IS applied to standard errors but is NOT folded into the AIC/BIC kernel.
> Status: clean.

# Phase 25: Code Review Report (Iteration 2)

**Reviewed:** 2026-08-17T12:00:00Z
**Depth:** standard
**Files Reviewed:** 2
**Status:** issues_found

## Summary

Re-review of `glm.rs` and `mod.rs` after the auto-fix iteration. All eight prior findings
are verified resolved. No new correctness bugs or security issues were introduced. One
warning-level documentation inaccuracy was introduced by the WR-01 dispersion fix.

**Verification of all prior findings:**

| ID | Description | Status |
|----|-------------|--------|
| CR-01 | Gamma IRLS weight inverted (was `1/μ²`, should be `μ²`) | **FIXED** — line 92: `mu.max(1e-10).powi(2)` |
| CR-02 | Poisson log-factorial O(y) loop + `INFINITY` saturation | **FIXED** — replaced with Lanczos `ln_gamma(yi+1.0)` |
| CR-03 | `predict_functional_glm` panics on column count mismatch | **FIXED** — lines 595-601 return `Err(InvalidDimension)` |
| CR-04 | `functional_glm` missing scalar_covariates row-count guard | **FIXED** — lines 539-548 return `Err(InvalidDimension)` |
| WR-01 | Gaussian/Gamma std_errors used `σ²=1` (dispersion not estimated) | **FIXED** — lines 392-413 estimate `φ` via Pearson χ²/dof |
| WR-02 | Binomial parity test fragile (different stopping rules) | **MITIGATED** — test uses 100 iter + tol=1e-12; residual fragility noted below |
| WR-03 | `predict_functional_glm` missing scalar covariate column guard | **FIXED** — lines 611-619 check `sc_cols != p_scalar` |
| IN-01 | `NaN` Gamma response slipped past `yi <= 0.0` guard | **FIXED** — upfront `!v.is_finite()` guard at lines 170-175 |

**New code quality checks performed:**

- **Lanczos `ln_gamma` correctness:** Verified numerically against `math.lgamma` to within
  `~8.9e-16` for `x ∈ {1, 1.5, 2, 3, 5, 10, 20, 100, 1000}`. The Lanczos series sum `a` is
  strictly positive for all Poisson-valid inputs (`yi ≥ 0`, so `x = yi+1 ≥ 1`); `a.ln()` is
  always safe. The reflection branch (`x < 0.5`) is unreachable for Poisson usage (`min x = 1`).
  Loop indexing `enumerate().skip(1)` over 9 coefficients correctly yields `k=1..8`, matching
  the Lanczos standard form.

- **Gamma IRLS weight (`μ²`) and working response (`−1/μ²`) are independent:** `irls_weight`
  returns `μ²` (line 92); `link_deriv` returns `−1/μ²` (line 73). The IRLS step uses
  `link_deriv` for the working response `z` and `irls_weight` for the weight `w`. These are
  not confused anywhere in the loop.

- **Dispersion φ math:** For Gaussian, `φ = RSS/(n−p)` with `V(μ)=1` equals Pearson χ²/dof.
  For Gamma, `φ = Σ((yᵢ−μᵢ)/μᵢ)²/(n−p)` with `V(μ)=μ²`. Both are passed as `sigma2` to
  `compute_ols_std_errors`, which computes `SE[j] = sqrt(φ · [(X'WX)⁻¹]ⱼⱼ)`. This is the
  correct `Var(β̂) = φ·(X'WX)⁻¹` formula. The `dof = n.saturating_sub(p).max(1)` guard
  prevents division by zero even when `n ≤ p`. The `mi.max(1e-10)` guard in the Gamma
  Pearson χ² term prevents division by near-zero fitted values.

- **`predict_functional_glm` bounds safety:** After the `m != m_train` guard, all accesses to
  `fit.fpca.mean[j]`, `fit.fpca.rotation[(j,k)]`, `fit.fpca.weights[j]` are in-bounds for
  `j ∈ 0..m`. `fit.coefficients[1+k]` for `k ∈ 0..ncomp` is in-bounds since `coefficients`
  has length `1 + ncomp + p_scalar`. The scalar path `sc[(i,j)]` for `j ∈ 0..p_scalar` is
  in-bounds after the `sc_cols != p_scalar` guard.

- **Non-finite guard completeness:** `!v.is_finite()` catches `f64::NAN`, `f64::INFINITY`, and
  `f64::NEG_INFINITY`. The check fires before all per-family guards, so no non-finite value
  reaches IRLS.

- **AIC/BIC consistency:** AIC and BIC still use the kernel-only `log_likelihood` without
  dispersion adjustment. This is intentional and noted in the module doc. The dispersion
  fix is correctly scoped to std_errors only. No regression in AIC/BIC computation.

---

## Warnings

### WR-01: Module doc still says "φ is not separately estimated" after dispersion fix

**File:** `fdars-core/src/scalar_on_function/glm.rs:29-32`

**Issue:** The module-level `# Convention divergences from R glm()` bullet at line 29 reads:

```
AIC/BIC: computed as … using the log-likelihood kernel per family (dispersion φ is not
separately estimated for Gamma/Gaussian).
```

The WR-01 fix (now implemented at lines 392-413) **does** estimate φ via Pearson χ²/dof and
applies it to all coefficient standard errors. The parenthetical `(dispersion φ is not
separately estimated for Gamma/Gaussian)` is now factually inaccurate as a blanket statement —
it is only true for AIC/BIC, not for `std_errors` or `beta_se`. A user reading the module doc
will be misled into thinking SEs are computed with `φ = 1` for Gaussian/Gamma fits.

**Fix:** Narrow the parenthetical to clarify it applies only to AIC/BIC:

```rust
//! - **AIC/BIC:** computed as `−2·log_likelihood + 2p` and `−2·log_likelihood + p·ln(n)`
//!   using the log-likelihood kernel per family (φ not folded into AIC/BIC; for
//!   Gamma/Gaussian, standard errors do account for an estimated φ via Pearson χ²/dof).
//!   Gamma and Gaussian AIC magnitudes are therefore **not** directly comparable to
//!   R's `glm()` / `lm()` output.
```

---

## Notes on WR-02 residual fragility

The parity test (`test_binomial_parity_with_logistic`) now uses 100 iterations and `tol=1e-12`
for both paths, which substantially reduces the risk. A theoretical fragility remains: near the
optimum, deviance-change converges quadratically (O(Δβ²)) while coefficient-change converges
linearly (O(Δβ)), so the `functional_glm` deviance criterion can fire one or two steps before
`functional_logistic`'s coefficient criterion. The final Δβ at that point is O(√1e-12) ≈ 1e-6,
which is right at the `< 1e-6` assertion threshold.

This fragility does not manifest on the current fixture (CI is green) and the mitigation
(high max_iter + tight tol) is the correct pragmatic approach for a parity test. It is
documented here for completeness, not as a blocking issue.

---

_Reviewed: 2026-08-17T12:00:00Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
