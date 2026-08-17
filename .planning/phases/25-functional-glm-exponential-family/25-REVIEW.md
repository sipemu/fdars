---
phase: 25-functional-glm-exponential-family
reviewed: 2026-08-17T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - fdars-core/src/scalar_on_function/glm.rs
  - fdars-core/src/scalar_on_function/mod.rs
  - fdars-core/src/lib.rs
findings:
  critical: 4
  warning: 3
  info: 1
  total: 8
status: issues_found
---

# Phase 25: Code Review Report

**Reviewed:** 2026-08-17
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the new `functional_glm` module (`glm.rs`) and supporting definitions in `mod.rs` and `lib.rs`. The module implements a four-family exponential-family GLM via IRLS on FPC scores. Structural quality is good: the `#[non_exhaustive]` and `#[must_use]` annotations are correct, the module doc is accurate, the `GlmFamily` enum and `FunctionalGlmResult` struct are well-formed, the `validate_response` guards fire before FPCA, and the Gamma intercept initialisation (`β₀ = 1/mean(y)`) correctly avoids divide-by-zero.

Four blockers were found: (1) the Gamma IRLS weight is inverted (`1/μ²` vs the correct `μ²`), causing silently incorrect coefficient estimates for all Gamma fits; (2) the Poisson `log(y!)` sum iterates `O(y)` times and has no guard against `f64::INFINITY`, which passes the integer-check and saturates the `as u64` cast to `u64::MAX`, creating an effective infinite loop; (3) `predict_functional_glm` has no check that `new_data.ncols()` equals the training grid width, causing an out-of-bounds panic when the new data has more evaluation points than the training data; (4) `functional_glm` does not validate that `scalar_covariates.nrows() == n`, causing an OOB panic.

---

## Critical Issues

### CR-01: Gamma IRLS weight is `1/μ²` — should be `μ²`

**File:** `fdars-core/src/scalar_on_function/glm.rs:86`

**Issue:** The IRLS weight for the Gamma family with canonical inverse link is derived as:

```
w_i = (dμ/dη)² / V(μ)
    = (−μ²)²  / μ²        [inverse link: dμ/dη = −μ²]
    = μ⁴ / μ²
    = μ²
```

The code computes `1.0 / mu.max(1e-10).powi(2)` = `1/μ²`, which is the **reciprocal** of the correct weight. The docstring claims `Gamma = 1/μ²` as if this follows from the formula `1/(V(μ)·g′(μ)²)`, but that formula also gives `μ²`:

```
1 / (V(μ) · [g′(μ)]²)
= 1 / (μ² · (−1/μ²)²)
= 1 / (μ² · 1/μ⁴)
= μ²
```

Using `1/μ²` instead of `μ²` inflates weights for small-μ observations and deflates them for large-μ observations — the opposite of the correct Fisher information weighting. The `test_gamma_recovery` test only asserts `v.is_finite() && v > 0.0` and does not check numerical accuracy, so the wrong weight is not caught by the current test suite.

**Fix:**
```rust
// irls_weight for Gamma — correct canonical-link formula w = μ²
GlmFamily::Gamma => mu.max(1e-10).powi(2),
```

---

### CR-02: Poisson `log(y!)` loop is `O(y)` and has no guard against `f64::INFINITY`

**File:** `fdars-core/src/scalar_on_function/glm.rs:136`

**Issue:** The log-factorial is computed as:

```rust
let ln_y_fact: f64 = (1..=(yi as u64)).map(|k| (k as f64).ln()).sum();
```

Two sub-issues:

**Sub-issue A — `f64::INFINITY` passes `validate_response`.**  
The Poisson check (`yi < 0.0 || yi != yi.floor()`) passes for `f64::INFINITY` because:  
- `INFINITY < 0.0` → false  
- `INFINITY != INFINITY.floor()` → `INFINITY == INFINITY` → false  

Then `INFINITY as u64` saturates to `u64::MAX = 18_446_744_073_709_551_615`. The range `1..=u64::MAX` will iterate for the lifetime of the process — an effective infinite loop / DoS on user-controlled input.

**Sub-issue B — `O(y)` cost for large finite counts.**  
For `y = 1_000_000` (one million — a valid Poisson integer) the range iterates one million times *per observation*. With `n = 100` observations, `log_likelihood` alone performs `10⁸` floating-point operations at the end of each fit. For `y` in the billions this is a practical hang.

The safe fix is `lgamma(y+1)`, which `statrs` already exposes as `Gamma::ln_gamma(yi + 1.0)`. Since `statrs` is already a dependency, this requires no new imports. Alternatively, add a `f64::is_finite()` guard and a ceiling cap (e.g. `yi <= 1e9`) before the loop.

**Fix — add isfinite guard to validation and replace the loop:**
```rust
// validate_response, Poisson branch:
GlmFamily::Poisson => {
    if y.iter().any(|&yi| !yi.is_finite() || yi < 0.0 || yi != yi.floor()) {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: "all values must be finite non-negative integers for Poisson family"
                .to_string(),
        });
    }
}

// log_likelihood, Poisson branch — replace O(y) sum with lgamma:
GlmFamily::Poisson => {
    let mi = mi.max(1e-300);
    // ln(y!) = lgamma(y+1);  statrs::function::gamma::ln_gamma is available
    let ln_y_fact = statrs::function::gamma::ln_gamma(yi + 1.0);
    yi * mi.ln() - mi - ln_y_fact
}
```
If pulling in `statrs` is undesirable, a threshold-based fallback (exact sum for `y ≤ 170`, then Stirling for larger values) also works.

---

### CR-03: `predict_functional_glm` panics on `new_data` with more columns than training data

**File:** `fdars-core/src/scalar_on_function/glm.rs:497-514`

**Issue:** `predict_functional_glm` derives `m` from `new_data.shape()` and then indexes into `fit.fpca.mean[j]` and `fit.fpca.rotation[(j, k)]` for `j in 0..m`. Both `fit.fpca.mean` and `fit.fpca.rotation` have length/height equal to the *training* grid size `m_train`. If `new_data.ncols() > m_train`, the access `fit.fpca.mean[j]` with `j >= m_train` is an out-of-bounds slice index and panics at runtime. If `new_data.ncols() < m_train`, the inner-product integral is truncated, producing a silently incorrect projection without any error.

The same gap exists in `predict_functional_logistic` in `logistic.rs`, but that is pre-existing code; the new `predict_functional_glm` should not repeat the pattern.

**Fix:**
```rust
pub fn predict_functional_glm(
    fit: &FunctionalGlmResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m_new) = new_data.shape();
    let m_train = fit.fpca.mean.len();
    // Fail loudly rather than panic or silently truncate.
    assert_eq!(
        m_new, m_train,
        "predict_functional_glm: new_data has {m_new} columns but model was trained on {m_train}"
    );
    // OR: return a Result and propagate FdarError::InvalidDimension
    ...
}
```
Prefer returning `Result<Vec<f64>, FdarError>` for consistency with the rest of the public API.

---

### CR-04: `functional_glm` does not validate `scalar_covariates` row count

**File:** `fdars-core/src/scalar_on_function/glm.rs:438-475`

**Issue:** `functional_glm` performs its own dimension checks for `n`, `m`, and `y.len()`, but never checks `scalar_covariates.nrows() == n`. `validate_fregre_inputs` (which does perform this check) is not called. `build_design_matrix` then accesses `sc[(i, j)]` for all `i in 0..n` without bounds checking: if `sc.nrows() < n`, the `FdMatrix` index-operator panics at runtime for the first `i >= sc.nrows()`.

**Fix:**
```rust
// In functional_glm, after the y.len() check:
if let Some(sc) = scalar_covariates {
    if sc.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "scalar_covariates",
            expected: format!("{n} rows"),
            actual: format!("{} rows", sc.nrows()),
        });
    }
}
```

---

## Warnings

### WR-01: `std_errors` / `beta_se` are wrong for Gaussian and Gamma (dispersion not estimated)

**File:** `fdars-core/src/scalar_on_function/glm.rs:336-340`

**Issue:** The SE computation always passes `sigma2 = 1.0` to `compute_ols_std_errors`:

```rust
let std_errors = cholesky_factor(&xtwx, p).map_or_else(
    |_| vec![f64::NAN; p],
    |l| compute_ols_std_errors(&l, p, 1.0),   // sigma2 hardcoded to 1
);
```

For Binomial and Poisson the canonical dispersion `φ = 1`, so `sigma2 = 1.0` is correct. For **Gaussian** and **Gamma**, `φ ≠ 1` in general:

- Gaussian: `φ = σ² = RSS / (n − p)`, so all reported SEs are too small by a factor of `sqrt(RSS/(n−p))`.
- Gamma: `φ` must be estimated from the deviance or moments; `φ = 1` only holds by coincidence.

A user who reads `fit.std_errors` or `fit.beta_se` for a Gaussian or Gamma fit and constructs Wald confidence intervals will get incorrectly narrow intervals. The module doc does not mention this limitation.

**Fix:**
```rust
let sigma2 = match family {
    GlmFamily::Gaussian => {
        // Pearson chi-squared dispersion estimate
        let rss: f64 = (0..n)
            .map(|i| (y[i] - fitted_values[i]).powi(2))
            .sum();
        rss / (n as f64 - p as f64).max(1.0)
    }
    GlmFamily::Gamma => {
        // Method-of-moments dispersion estimate
        let pearson_x2: f64 = (0..n)
            .map(|i| {
                let mi = fitted_values[i].max(1e-10);
                let r = y[i] - mi;
                r * r / (mi * mi)
            })
            .sum();
        pearson_x2 / (n as f64 - p as f64).max(1.0)
    }
    _ => 1.0, // Binomial and Poisson: canonical dispersion
};
let std_errors = cholesky_factor(&xtwx, p).map_or_else(
    |_| vec![f64::NAN; p],
    |l| compute_ols_std_errors(&l, p, sigma2),
);
```

---

### WR-02: `Binomial` convergence criterion differs from `functional_logistic`, making the parity test fragile

**File:** `fdars-core/src/scalar_on_function/glm.rs:275` / `logistic.rs:79`

**Issue:** `functional_logistic` converges on `max |β_new − β| < tol` (coefficient max-norm change), while `functional_glm` with `GlmFamily::Binomial` converges on `|deviance_new − deviance_old| < tol`. Per-step IRLS updates are numerically identical for Binomial (the working response and weight formulas expand to the same expression), but the stopping rule fires at different iterations. If `deviance-change < tol` fires one iteration later than `coeff-change < tol` (or vice versa), the final β vectors will differ by up to one more Newton step, potentially exceeding the `1e-6` tolerance in `test_binomial_parity_with_logistic`.

The test passes on the current `make_data(30, 50)` fixture because IRLS is well-separated from saddle points, but the test is not robust to different data scales.

**Fix:** Either (a) use the same convergence criterion in both, or (b) document the known difference and relax the parity test tolerance to `1e-4`.

---

### WR-03: `predict_functional_glm` does not validate `new_scalar` column count

**File:** `fdars-core/src/scalar_on_function/glm.rs:509-512`

**Issue:** The inner loop accesses `sc[(i, j)]` for `j in 0..p_scalar` where `p_scalar = fit.gamma.len()`. If the caller passes a `new_scalar` matrix with fewer columns than `p_scalar`, this panics via the `FdMatrix` index operator. There is no check that `sc.ncols() == fit.gamma.len()`.

**Fix:**
```rust
if let Some(sc) = new_scalar {
    assert_eq!(sc.ncols(), p_scalar,
        "predict_functional_glm: new_scalar has {} columns but model has {} scalar coefficients",
        sc.ncols(), p_scalar);
    for j in 0..p_scalar {
        eta += fit.gamma[j] * sc[(i, j)];
    }
}
```
Again, prefer returning `Result` for a library API.

---

## Info

### IN-01: `validate_response` does not catch `f64::NAN` for `GlmFamily::Gamma`

**File:** `fdars-core/src/scalar_on_function/glm.rs:175-181`

**Issue:** The Gamma guard `yi <= 0.0` is false for `NaN` (IEEE 754: all comparisons with NaN return false), so `NaN` values in `y` pass validation and propagate into the IRLS loop. The working response `z = η + (y − μ) · g′(μ)` becomes `NaN` at the first step; subsequent Cholesky solve receives a `NaN` right-hand side; `β` becomes `NaN`; `deviance` becomes `NaN`; `NaN < tol` is always false, so the loop runs all `max_iter` iterations before returning a result struct where every field is `NaN`. The error is silent.

The Poisson guard (`yi != yi.floor()`) already catches `NaN` correctly because `NaN != NaN` is true. The Binomial guard (`yi != 0.0 && yi != 1.0`) also catches `NaN`. Gaussian is documented as unrestricted.

Note: CR-02 fix (adding `!yi.is_finite()` to the Poisson check) implicitly catches `NaN` for Poisson as well; a parallel fix is needed for Gamma.

**Fix:**
```rust
GlmFamily::Gamma => {
    if y.iter().any(|&yi| !yi.is_finite() || yi <= 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: "all values must be finite and strictly positive for Gamma family"
                .to_string(),
        });
    }
}
```

---

_Reviewed: 2026-08-17_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
