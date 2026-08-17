# Phase 25: Functional GLM (Exponential Family) - Research

**Researched:** 2026-08-17
**Domain:** Generalized Linear Models — IRLS over FPC scores, exponential family, Rust implementation
**Confidence:** HIGH (codebase reads) / MEDIUM (GLM theory via web)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Signature mirrors `functional_logistic` with `family` inserted after `y`:
  `functional_glm(data, y, family, scalar_covariates, ncomp, max_iter, tol) -> Result<FunctionalGlmResult, FdarError>`.
- `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` — exactly these four, each carrying its
  canonical link + variance function: Binomial = logit / μ(1−μ), Poisson = log / μ,
  Gamma = inverse / μ², Gaussian = identity / 1.
- `functional_logistic` is retained verbatim (additive/non-breaking); `functional_glm(…, Binomial)`
  reproduces its fit.
- `GlmFamily` is `#[non_exhaustive]` (forward-compat; matches project public-enum convention).
- `FunctionalGlmResult` fields: `{ intercept, beta_t, gamma, fitted_values (μ), linear_predictors (η), ncomp, coefficients, std_errors, log_likelihood, deviance, iterations, fpca, aic, bic, family }`.
  Classification-only fields (`probabilities`, `predicted_classes`, `accuracy`) dropped.
- Binomial parity (SC2) via `intercept`/`beta_t`/`coefficients`+`fitted_values` agreeing with
  `functional_logistic` within tolerance — no separate classification fields.
- Canonical links (Gamma = inverse), with μ/variance clamped away from boundary.
- Per-family response-domain guards → `FdarError::InvalidParameter` (never panic).
- Reuse the `functional_logistic` IRLS loop over FPC scores; converge on deviance/coefficient
  change < tol, capped at max_iter.
- Gaussian runs through the same IRLS uniformly (weights ≡ 1, identity link → OLS in one step).

### Claude's Discretion

(None documented — all critical decisions locked.)

### Deferred Ideas (OUT OF SCOPE)

- Extra families (inverse-Gaussian, negative-binomial) and configurable/non-canonical links.
- Log-link Gamma alternative (canonical inverse link is the chosen default).
- Step-halving/line-search in the IRLS loop (reuse the existing convergence policy).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-02 | User can fit a functional GLM for a scalar response over functional predictors via `functional_glm(data, y, family)` with `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` enum, each carrying its canonical link and variance function. Existing `functional_logistic` retained unchanged. Result is `Result`-returning and re-exported at crate root. | IRLS generalization fully mapped (§ IRLS Generalization); per-family formulas derived (§ Per-Family Reference); reuse targets read (§ Standard Stack); test designs specified (§ Code Examples). |
</phase_requirements>

---

## Summary

Phase 25 adds `functional_glm` — a scalar-on-function GLM over the four mainstream exponential-family distributions — by generalizing the existing `functional_logistic` IRLS loop. The core change is parameterizing the IRLS step by a `GlmFamily` enum that supplies the link function g(μ), inverse link g⁻¹(η), link derivative g'(μ), and variance function V(μ) for each family. Everything else (FPC-score design matrix via `fdata_to_pc_1d`, Cholesky-based weighted least squares, SE computation, AIC/BIC) reuses the existing machinery verbatim.

The implementation lives in a new file `fdars-core/src/scalar_on_function/glm.rs` with `GlmFamily` and `FunctionalGlmResult` defined in `mod.rs` alongside the existing result types. The barrel exports in `mod.rs` and the crate-root `lib.rs` receive additive `pub use` lines for `functional_glm`, `predict_functional_glm`, `GlmFamily`, and `FunctionalGlmResult`.

**Primary recommendation:** Implement a single generic `irls_step_glm` that accepts closures for link/inverse-link/variance, dispatch to it from `functional_glm` by matching on `GlmFamily`, and validate the response domain at function entry before the FPCA step. All four families share exactly the same weighted-normal-equations solver already present in the logistic module.

---

## Project Constraints (from CLAUDE.md)

| Directive | Applies Here |
|-----------|-------------|
| All public fns return `Result<T, FdarError>` (no panics) | Yes — `functional_glm` and `predict_functional_glm` |
| Public types derive `Debug, Clone, PartialEq`, `#[non_exhaustive]`, conditional serde | Yes — `FunctionalGlmResult`, `GlmFamily` |
| `#[must_use]` on expensive computations | Yes — `functional_glm` |
| `GlmFamily` is `#[non_exhaustive]` | Locked by CONTEXT.md |
| Inline `#[cfg(test)] mod tests` | Yes — add GLM tests to existing tests.rs |
| No new crate dependency | Confirmed — all required math is pure `f64` ops |
| `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must stay green | Yes — implement all match arms exhaustively |
| MSRV 1.81 | No issue — only basic `f64` arithmetic used |
| Column-major `FdMatrix` via `fdata_to_pc_1d` | Reused unchanged |
| Parameter ordering: `(data, y, [family,] [scalar_covariates,] …)` | Matches locked signature |

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GLM family enum + link/variance dispatch | `scalar_on_function/glm.rs` | `scalar_on_function/mod.rs` (pub re-export) | Encapsulates family-specific math; matches module pattern of all other SoF estimators |
| IRLS weighted least squares | `scalar_on_function/glm.rs` (reuse pattern from `logistic.rs`) | `linalg.rs` (`cholesky_solve`) | Weighted normal equations solved via existing Cholesky path |
| FPC-score design matrix | `regression.rs` (`fdata_to_pc_1d`) | — | Shared unchanged; produces `FpcaResult` + design matrix |
| Response-domain validation | Entry point of `functional_glm` | — | Must occur before FPCA (early exit) |
| Result struct + SE/deviance/AIC/BIC | `scalar_on_function/glm.rs` | — | Family-specific log-likelihood feeds common AIC/BIC formula |
| Crate-root re-export | `src/lib.rs` | — | Additive `pub use scalar_on_function::{…}` lines |

---

## Standard Stack

### Core (no new dependencies)

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `f64` arithmetic | Rust stdlib | Link/variance/deviance math | All required ops (ln, exp, recip) in `std` |
| `crate::linalg::cholesky_solve` | internal | Weighted normal equations | Already used by logistic IRLS; no change needed |
| `crate::regression::fdata_to_pc_1d` | internal | FPC-score design | Shared across entire SoF module |
| `crate::scalar_on_function::build_design_matrix` | internal | Design matrix assembly | Reused verbatim from logistic |
| `crate::scalar_on_function::recover_beta_t` | internal | β(t) reconstruction | Reused verbatim |
| `crate::scalar_on_function::compute_beta_se` | internal | SE propagation | Reused verbatim |
| `crate::scalar_on_function::compute_ols_std_errors` | internal | SE from Fisher info | Reused verbatim |
| `crate::scalar_on_function::compute_fitted` | internal | η = Xβ | Reused verbatim |
| `f64::ln_gamma` / `statrs` | N/A — use `f64::ln_1p` + lgamma via stdlib | Poisson log-likelihood | Rust `std` has no `lgamma`; use approximation or `(y+1).ln_gamma()` from statrs |

**Note on lgamma:** The Poisson log-likelihood includes `−ln(y!)`. Rust `std` has no `lgamma`. The existing codebase does NOT use `statrs` for logistic (no factorial term needed). For Poisson, use `(0..=y_int).fold(0.0, |acc, k| acc + (k as f64 + 1.0).ln())` for integer y only — or declare log-likelihood "up to additive constant" (standard for IRLS convergence) and only emit the deviance-based AIC: `AIC = deviance + 2p`. See § Dispersion and AIC for Gamma/Gaussian. [ASSUMED — no lgamma in std; verify statrs availability]

**Actually: statrs IS listed** in CLAUDE.md as a key dependency. So `statrs::function::gamma::ln_gamma` is available. [VERIFIED: .claude/CLAUDE.md — "statrs - Statistical distributions and functions"]

### No New Dependencies

The phase explicitly forbids new crate dependencies. All math is achievable with `f64` ops and the existing `statrs` crate (already depended upon).

**Installation:** None required.

---

## Package Legitimacy Audit

No external packages are introduced in this phase. The phase constraint explicitly states "no new crate dependency."

| Package | Status |
|---------|--------|
| (none new) | N/A |

---

## Architecture Patterns

### System Architecture Diagram

```
functional_glm(data, y, GlmFamily::Poisson, scalar_covariates, ncomp, max_iter, tol)
        │
        ▼
  [Entry validation]
  ─ data shape (n≥3, m≥1, y.len()==n)
  ─ family-specific response guard (Poisson: y≥0 integer, Gamma: y>0, Binomial: y∈[0,1])
        │
        ▼
  fdata_to_pc_1d(data, ncomp, argvals) → FpcaResult
        │
        ▼
  build_design_matrix(scores, ncomp, scalar_covariates, n) → FdMatrix  [1 | ξ₁…ξK | z₁…zP]
        │
        ▼
  irls_loop_glm(design, y, family, max_iter, tol)
  ┌─────────────────────────────────────────────────────────────────────┐
  │  for iter in 0..max_iter:                                           │
  │    η_i = design[i,:] · β                                           │
  │    μ_i = family.inv_link(η_i)   (clamped)                          │
  │    w_i = family.irls_weight(μ_i)                                   │
  │    z_i = η_i + (y_i - μ_i) * family.link_deriv(μ_i)               │
  │    β_new = cholesky_solve(X'WX, X'Wz)  [or None if singular]       │
  │    deviance_new = family.deviance(y, μ)                             │
  │    if |deviance_new - deviance_old| < tol → break                   │
  └─────────────────────────────────────────────────────────────────────┘
        │
        ▼
  build_glm_result(design, beta, y, fpca, ncomp, m, iterations, family)
  ─ β(t) = Σ_k γ_k φ_k(t)   via recover_beta_t
  ─ fitted_values μ_i = family.inv_link(η_i)
  ─ linear_predictors η_i
  ─ SE from Fisher info (X'WX)^{-1}  via cholesky_factor + compute_ols_std_errors
  ─ log_likelihood = family.log_likelihood(y, μ)
  ─ deviance = family.deviance(y, μ)
  ─ AIC = -2*ll + 2*p   [Binomial/Poisson: φ=1; Gamma/Gaussian: φ estimated]
  ─ BIC = -2*ll + p*ln(n)
        │
        ▼
  FunctionalGlmResult { intercept, beta_t, gamma, fitted_values, linear_predictors,
                         ncomp, coefficients, std_errors, log_likelihood, deviance,
                         iterations, fpca, aic, bic, family }
```

### Recommended Project Structure

```
fdars-core/src/scalar_on_function/
├── mod.rs              # Add GlmFamily, FunctionalGlmResult definitions + pub use glm::*
├── glm.rs              # New file: functional_glm, predict_functional_glm, irls helpers
├── logistic.rs         # UNCHANGED — functional_logistic retained verbatim
├── tests.rs            # Add GLM tests (Binomial parity, Poisson recovery, Gamma recovery, error paths)
└── ... (existing files unchanged)
fdars-core/src/lib.rs   # Add pub use scalar_on_function::{functional_glm, predict_functional_glm,
                        #   GlmFamily, FunctionalGlmResult} to existing block
```

---

## IRLS Generalization — Exact Formulas

This is the core technical content the planner and executor need. All formulas cross-checked against the Timothy Barry GLM reference and the Duke stats lecture notes. [CITED: https://timothy-barry.github.io/posts/2020-07-07-generalized-linear-models/]

### General IRLS Structure

For each iteration:
1. Compute linear predictor: `η_i = Σ_j design[(i,j)] * β[j]`
2. Compute mean (inverse link): `μ_i = g⁻¹(η_i)`  (clamped)
3. Compute IRLS weight: `w_i = (dμ/dη)² / V(μ_i)` = `1 / (V(μ_i) * g'(μ_i)²)`
4. Compute working response: `z_i = η_i + (y_i - μ_i) * g'(μ_i)`
5. Solve weighted normal equations: `(X'WX)β_new = X'Wz`
6. Check convergence on deviance change or max coefficient change

**Note on convergence criterion:** The existing `logistic.rs` converges on max coefficient change (`change < tol`). The CONTEXT.md specifies "deviance/coefficient change < tol". Using either is acceptable; recommend keeping the coefficient-change criterion for exact Binomial parity. Deviance-change convergence is also valid and more numerically standard. [ASSUMED — planner should choose one; coefficient-change is safer for Binomial parity]

### Per-Family Reference Table

| Family | g(μ) | g⁻¹(η) | g'(μ) = dη/dμ | V(μ) | IRLS weight w_i | Working response z_i |
|--------|------|---------|--------------|------|-----------------|---------------------|
| Binomial | log(μ/(1-μ)) | 1/(1+e^{-η}) | 1/(μ(1-μ)) | μ(1-μ) | μ(1-μ) | η + (y-μ)/(μ(1-μ)) |
| Poisson | log(μ) | exp(η) | 1/μ | μ | μ | η + (y-μ)/μ |
| Gamma | 1/μ | 1/η | -1/μ² | μ² | 1/μ² | η + (y-μ)*(-1/μ²) |
| Gaussian | μ | η | 1 | 1 | 1 | z = y (exact OLS) |

**Simplification check — Binomial parity:** The Binomial row matches the existing `irls_step` in `logistic.rs:23`:
- `w: Vec<f64> = mu.iter().map(|&p| (p * (1.0 - p)).max(1e-10)).collect();` ✓ matches `w_i = μ(1-μ)`
- `z_work: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / w[i]).collect();` ✓ matches `z_i = η + (y-μ)/(μ(1-μ))` [VERIFIED: fdars-core/src/scalar_on_function/logistic.rs:23-24]

### Gamma Inverse Link — Working Response Sign

For Gamma: g'(μ) = -1/μ². So:
- `z_i = η_i + (y_i - μ_i) * (-1/μ_i²)`
- `z_i = (1/μ_i) + (y_i - μ_i) * (-1/μ_i²)` [since η = 1/μ]
- `z_i = 1/μ_i - y_i/μ_i² + 1/μ_i`
- Simplifies to: `z_i = 2/μ_i - y_i/μ_i²`

The weight `w_i = 1/μ_i²` (positive, well-defined when μ > 0).

### μ/η Clamping Thresholds

| Family | What to Clamp | Where | Recommended Threshold |
|--------|--------------|-------|----------------------|
| Binomial | μ = sigmoid(η) | already bounded (0,1) via sigmoid | clamp μ ∈ [1e-10, 1-1e-10] for w_i and log |
| Poisson | η before exp | clamp η ≤ 500.0 to avoid overflow | μ = exp(η.min(500.0)), clamp μ ≥ 1e-10 |
| Gamma | η must stay > 0 | μ = 1/η so η must be positive | clamp η ≥ 1e-10 (→ μ ≤ 1e10), clamp μ ≥ 1e-10 |
| Gaussian | none needed | identity link | no clamping required |

**Critical Gamma pitfall:** If the IRLS step produces η ≤ 0, then μ = 1/η is undefined or negative. Clamp η from below at `1e-10` before computing `μ = 1.0/η`. Similarly, clamp μ ≥ `1e-10` before computing weights.

---

## Per-Family Deviance Formulas

[CITED: https://grodri.github.io/glms/notes/a2s5 (Poisson); Timothy Barry GLM reference; glum docs]

Deviance = 2 × (log-likelihood of saturated model − log-likelihood of fitted model):

| Family | Unit deviance d(y_i, μ_i) | Total deviance D = Σ d(y_i, μ_i) |
|--------|--------------------------|----------------------------------|
| Binomial | 2[y·log(y/μ) + (1-y)·log((1-y)/(1-μ))] | Σ above (with 0·log(0) = 0 convention) |
| Poisson | 2[y·log(y/μ) - (y-μ)] | Σ above (with 0·log(0) = 0 convention) |
| Gamma | 2[(y-μ)/μ - log(y/μ)] | Σ above |
| Gaussian | (y - μ)² | Σ(y_i - μ_i)² = RSS |

**0·log(0) convention:** When y_i = 0 (Binomial or Poisson), the term `y·log(y/μ)` = 0. In Rust: `if yi == 0.0 { 0.0 } else { yi * (yi / mu_i).ln() }`.

---

## Per-Family Log-Likelihood Formulas

| Family | log L (per observation, up to additive constant) | Notes |
|--------|--------------------------------------------------|-------|
| Binomial | y·log(μ) + (1-y)·log(1-μ) | Same as logistic; clamp μ ∈ [1e-15, 1-1e-15] |
| Poisson | y·log(μ) - μ - log(y!) | `log(y!) = lgamma(y+1)`; use `statrs::function::gamma::ln_gamma(yi + 1.0)` |
| Gamma | −y/μ − log(μ) | Up to dispersion-dependent constant; this is the kernel sufficient for AIC when φ estimated separately |
| Gaussian | −(y−μ)²/(2σ²) | Up to constant; σ² estimated as RSS/n. For AIC use: ll = −n/2·ln(RSS/n) − n/2 |

**Gamma and Gaussian AIC:**
- For Binomial and Poisson (φ = 1 fixed): `AIC = deviance + 2p` and `BIC = deviance + p·ln(n)`.
- For Gamma and Gaussian (φ estimated): The full log-likelihood includes the dispersion parameter. For practical AIC, estimate φ̂ via Pearson statistic: `phi_hat = pearson_chi_sq / (n - p)` where `pearson_chi_sq = Σ (y_i - μ_i)² / V(μ_i)`. Then `ll = Σ log f(y_i; μ_i, φ̂)`. [ASSUMED — standard practice; R's `glm()` follows this for Gamma AIC]

**Simpler AIC for Gamma/Gaussian:** Since the phase requires AIC/BIC but not hypothesis tests, using `AIC = deviance + 2p` (treating φ=1) produces a valid information criterion for model comparison within the same family, just not directly comparable to Gaussian linear models. The CONTEXT.md does not specify whether φ should be estimated; recommend: emit deviance (family-specific), use `AIC = -2*ll + 2*p` where ll is the kernel log-likelihood, and note in rustdoc that dispersion is not estimated. [ASSUMED — clarify with user if hypothesis test parity with R is required]

---

## Binomial Parity Test Design (SC2)

The goal: `functional_glm(data, y, GlmFamily::Binomial, …)` coefficients and `fitted_values` agree with `functional_logistic(data, y, …)` within tolerance.

**Why parity holds:** When `GlmFamily::Binomial` is dispatched, the generic IRLS step is mathematically identical to `irls_step` in `logistic.rs` (same w_i and z_i). The `fitted_values` in `FunctionalGlmResult` for Binomial = μ = sigmoid(η) = probabilities in `FunctionalLogisticResult`. So `glm_result.fitted_values[i]` ≈ `logistic_result.probabilities[i]`.

**Test design:**
```rust
// Generate binary response data
let data = ...; // 30 x 50 FdMatrix
let y_bin = ...; // 30 binary values 0.0/1.0

let fit_logistic = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
let fit_glm = functional_glm(&data, &y_bin, GlmFamily::Binomial, None, 3, 25, 1e-6).unwrap();

// Coefficient parity
for (a, b) in fit_logistic.coefficients.iter().zip(&fit_glm.coefficients) {
    assert!((a - b).abs() < 1e-8, "coefficient mismatch: {} vs {}", a, b);
}
// Fitted value (probability) parity
for (a, b) in fit_logistic.probabilities.iter().zip(&fit_glm.fitted_values) {
    assert!((a - b).abs() < 1e-8, "fitted value mismatch");
}
```

**Tolerance:** 1e-8 is achievable if both paths use identical IRLS step code. Use 1e-6 if there is any floating-point ordering difference.

---

## Per-Family Recovery Test Designs (SC3)

### Poisson Recovery Test

Generate count data from a known log-linear model: `log(μ_i) = α + β·score_i` where `score_i` is the projection of a synthetic functional predictor onto a known direction.

```rust
// Design: n=100, m=50, k=3 FPC components
// True model: log(mu_i) = 0.5 + 1.5 * score_i[0]  (dominant first PC)
// Generate mu_i, then draw Poisson counts (approximated deterministically):
//   y_i = mu_i.round() as f64  [or use a fixed-seed pseudo-Poisson]

// Expected: fit.coefficients[0] ≈ 0.5 (intercept), fit.coefficients[1] ≈ 1.5
// Tolerance: within 0.3 in coefficients (n=100 deterministic approximation)
// fitted_values[i] = exp(eta_i) should be > 0 for all i

assert!(fit.fitted_values.iter().all(|&mu| mu > 0.0));
assert!((fit.intercept - 0.5).abs() < 0.5); // loose — deterministic data
```

**Realistic tolerance:** For deterministic (non-stochastic) test data, recovery tolerance should be loose (±0.3 to ±0.5 on coefficients) since we are not drawing true random counts. The test verifies that (a) the estimator converges, (b) fitted values are positive, and (c) predictions order-match the true μ (Pearson correlation > 0.9).

### Gamma Recovery Test

Generate positive responses from `μ_i = 1 / (α + β·score_i)` (inverse link model):

```rust
// True model: 1/mu_i = 0.5 + 1.0 * score_i[0]
// y_i = mu_i  (no noise — deterministic; or y_i = mu_i * 1.05 for mild perturbation)

// Expected: fit.coefficients[0] ≈ 0.5, fit.coefficients[1] ≈ 1.0
// Key check: all fitted_values > 0 (μ = 1/η must be positive)
// Tolerance: ±0.3 on coefficients for deterministic data

assert!(fit.fitted_values.iter().all(|&mu| mu > 0.0 && mu.is_finite()));
```

**Pitfall:** If true μ_i is very small (< 0.01), the clamping at 1e-10 must not cause loss of precision in the test. Use μ_i ∈ [0.1, 10] range.

---

## Common Pitfalls

### Pitfall 1: Gamma η Sign Flip
**What goes wrong:** The Gamma canonical link g(μ) = 1/μ has a negative derivative g'(μ) = -1/μ². If implemented naively as `z_i = η_i + (y_i - μ_i) / w_i` (same as logistic) where w_i = weight = 1/μ_i² and the denominator is the weight, the sign of g'(μ) is lost.
**Root cause:** The working response formula uses g'(μ) = dη/dμ, not 1/w. For canonical links, w_i = 1/(V(μ)·g'(μ)²) = μ_i² (positive), but z_i = η_i + (y_i - μ_i) · g'(μ_i) = η_i + (y_i - μ_i) · (−1/μ_i²) — the sign matters.
**How to avoid:** Store `link_deriv(mu)` separately from `irls_weight(mu)`. Never derive the working response from the weight: use `z_i = eta_i + (y_i - mu_i) * family.link_deriv(mu_i)`.
**Warning signs:** IRLS diverges (deviance increases) for Gamma family on positive data.

### Pitfall 2: Poisson with η Overflow
**What goes wrong:** `exp(η)` overflows to `inf` when η > ~709 (f64 overflow), causing μ = inf, w = inf, and NaN in the design solve.
**Root cause:** Without clamping, a poorly initialized β can produce runaway η in the first few IRLS steps.
**How to avoid:** Clamp η: `let mu = eta.min(500.0_f64).exp();` Alternatively clamp μ after exp.
**Warning signs:** `fitted_values` contains `inf` or `NaN`.

### Pitfall 3: Gamma Inverse Link η ≤ 0
**What goes wrong:** μ = 1/η is undefined or negative when η ≤ 0. Can happen in early IRLS iterations if β is initialized to zero and the design includes a negative-valued predictor score.
**Root cause:** Zero-initialized β → η = Xβ = 0 → μ = 1/0 = inf in the first iteration.
**How to avoid:** For Gamma, initialize β so that η > 0. One approach: initialize intercept to `1/y_mean` (so η_i ≈ 1/mean(y) > 0 for all i) and other coefficients to 0. Alternatively, clamp η ≥ 1e-10 always.
**Warning signs:** First IRLS step returns None (singular) or produces inf weights.

**Practical Gamma initialization:** Set `beta[0] = 1.0 / y.iter().sum::<f64>() * n as f64` (inverse of mean) so that the initial linear predictor η₀ = β₀ is positive and μ₀ = 1/η₀ = mean(y). All-zero initialization works for Binomial (η=0 → μ=0.5 ✓) and Poisson (η=0 → μ=1 ✓) and Gaussian (η=0 → μ=0, OLS step converges in 1 step anyway). For Gamma, zero-initialized β → η=0 is the problem. [ASSUMED — standard practice; verify by running the test]

### Pitfall 4: 0·log(0) in Deviance/Log-Likelihood
**What goes wrong:** When y_i = 0 (valid for Poisson/Binomial), `y * ln(y/μ)` = `0 * ln(0)` which is NaN in IEEE 754.
**Root cause:** `0.0_f64.ln()` = `-inf`; `0.0 * (-inf)` = NaN by IEEE rules.
**How to avoid:** 
```rust
fn xlogy(x: f64, y: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x * y.ln() }
}
```
Apply `xlogy(yi, yi/mu_i)` in deviance computation.
**Warning signs:** `deviance` field is NaN with Poisson/Binomial data containing zeros.

### Pitfall 5: Poisson Integer Validation
**What goes wrong:** The spec requires `y ≥ 0 integer-valued` for Poisson. Validating this requires checking not just `y ≥ 0` but also `y == y.floor()` (i.e., no fractional counts).
**Root cause:** `y = 1.5` passes `y ≥ 0` but is not a valid Poisson count.
**How to avoid:** Check `y.iter().any(|&yi| yi < 0.0 || yi != yi.floor())` → `InvalidParameter`.
**Warning signs:** Odd log-likelihood values for non-integer Poisson responses.

### Pitfall 6: Non-Exhaustive Match Arms (clippy)
**What goes wrong:** `GlmFamily` is `#[non_exhaustive]`. A `match family { ... }` inside the crate compiles without a `_` arm, but future variants would break it.
**Root cause:** `#[non_exhaustive]` only affects downstream crates, not the defining crate. Match arms within `fdars-core` are still exhaustive at compile time.
**How to avoid:** Add all four variants explicitly; no wildcard needed inside the crate. Clippy `--all-targets` will catch it if a variant is missed.

---

## Code Examples

### Pattern 1: Generic IRLS Step (Dispatched by Family)

```rust
// Source: Generalizes logistic.rs:irls_step by adding family dispatch.
// The logistic.rs irls_step is lines 14-49 of fdars-core/src/scalar_on_function/logistic.rs.

fn irls_step_glm(
    design: &FdMatrix,
    y: &[f64],
    beta: &[f64],
    family: GlmFamily,
) -> Option<Vec<f64>> {
    let (n, p) = design.shape();
    let eta: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| design[(i, j)] * beta[j]).sum())
        .collect();
    let mu: Vec<f64> = eta.iter().map(|&e| family.inv_link(e)).collect();
    // IRLS weight: w_i = (dmu/deta)^2 / V(mu) = 1 / (V(mu) * g'(mu)^2)
    let w: Vec<f64> = mu.iter().map(|&m| family.irls_weight(m)).collect();
    // Working response: z_i = eta_i + (y_i - mu_i) * g'(mu_i)
    let z: Vec<f64> = (0..n)
        .map(|i| eta[i] + (y[i] - mu[i]) * family.link_deriv(mu[i]))
        .collect();
    // Weighted normal equations: (X'WX)beta = X'Wz
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let s: f64 = (0..n).map(|i| design[(i, k)] * w[i] * design[(i, j)]).sum();
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    let xtwz: Vec<f64> = (0..p)
        .map(|k| (0..n).map(|i| design[(i, k)] * w[i] * z[i]).sum())
        .collect();
    cholesky_solve(&xtwx, &xtwz, p).ok()
}
```

### Pattern 2: GlmFamily Methods

```rust
// Source: Derived from canonical GLM formulas.
// [CITED: https://timothy-barry.github.io/posts/2020-07-07-generalized-linear-models/]

impl GlmFamily {
    /// Inverse link: η → μ  (clamped to valid range).
    pub(crate) fn inv_link(&self, eta: f64) -> f64 {
        match self {
            GlmFamily::Binomial  => sigmoid(eta),                         // clamp via sigmoid impl
            GlmFamily::Poisson   => eta.min(500.0_f64).exp().max(1e-10),  // exp, clamp low
            GlmFamily::Gamma     => (1.0 / eta.max(1e-10)).max(1e-10),    // 1/eta, eta must be >0
            GlmFamily::Gaussian  => eta,
        }
    }

    /// Link derivative: dη/dμ = g'(μ).
    pub(crate) fn link_deriv(&self, mu: f64) -> f64 {
        match self {
            GlmFamily::Binomial  => 1.0 / (mu * (1.0 - mu)).max(1e-10),
            GlmFamily::Poisson   => 1.0 / mu.max(1e-10),
            GlmFamily::Gamma     => -1.0 / mu.max(1e-10).powi(2),         // negative!
            GlmFamily::Gaussian  => 1.0,
        }
    }

    /// IRLS weight: w_i = 1 / (V(μ) * g'(μ)²).
    /// For canonical links, simplifies to: Binomial=μ(1-μ), Poisson=μ, Gamma=1/μ², Gaussian=1.
    pub(crate) fn irls_weight(&self, mu: f64) -> f64 {
        match self {
            GlmFamily::Binomial  => (mu * (1.0 - mu)).max(1e-10),
            GlmFamily::Poisson   => mu.max(1e-10),
            GlmFamily::Gamma     => (1.0 / mu.max(1e-10).powi(2)).max(1e-10),
            GlmFamily::Gaussian  => 1.0,
        }
    }

    /// Total deviance D = 2 Σ d(y_i, μ_i).
    pub(crate) fn deviance(&self, y: &[f64], mu: &[f64]) -> f64 {
        fn xlogy(x: f64, y: f64) -> f64 { if x == 0.0 { 0.0 } else { x * y.ln() } }
        y.iter().zip(mu).map(|(&yi, &mi)| {
            match self {
                GlmFamily::Binomial  => 2.0 * (xlogy(yi, yi / mi.max(1e-15))
                                              + xlogy(1.0 - yi, (1.0 - yi) / (1.0 - mi).max(1e-15))),
                GlmFamily::Poisson   => 2.0 * (xlogy(yi, yi / mi.max(1e-15)) - (yi - mi)),
                GlmFamily::Gamma     => 2.0 * ((yi - mi) / mi.max(1e-15) - (yi / mi.max(1e-15)).ln()),
                GlmFamily::Gaussian  => (yi - mi).powi(2),
            }
        }).sum()
    }

    /// Log-likelihood kernel (sufficient for AIC/BIC; excludes normalizing constants).
    pub(crate) fn log_likelihood(&self, y: &[f64], mu: &[f64]) -> f64 {
        use statrs::function::gamma::ln_gamma;
        y.iter().zip(mu).map(|(&yi, &mi)| {
            let mi = mi.max(1e-300);
            match self {
                GlmFamily::Binomial  => {
                    let mi = mi.clamp(1e-15, 1.0 - 1e-15);
                    yi * mi.ln() + (1.0 - yi) * (1.0 - mi).ln()
                }
                GlmFamily::Poisson   => yi * mi.ln() - mi - ln_gamma(yi + 1.0),
                GlmFamily::Gamma     => -yi / mi - mi.ln(),
                GlmFamily::Gaussian  => -(yi - mi).powi(2), // relative; scale by -1/(2*sigma^2) for abs LL
            }
        }).sum()
    }
}
```

**Note on Gaussian log-likelihood for AIC:** The exact Gaussian log-likelihood is `-(n/2)*ln(2π*σ²) - RSS/(2σ²)`. For AIC consistency, estimate σ² = RSS/n, giving `ll = -(n/2)*ln(2π*RSS/n) - n/2`. If the phase only needs relative AIC (comparing Gaussian models against each other), the kernel `-Σ(y-μ)²` suffices. For cross-family comparison, use the full formula. [ASSUMED — planner should decide; recommend full formula for the Gaussian branch]

### Pattern 3: Response Validation (Entry-Point Guards)

```rust
// [VERIFIED: fdars-core/src/scalar_on_function/logistic.rs:227-232] — shows existing Binomial guard pattern
fn validate_response(y: &[f64], family: GlmFamily) -> Result<(), FdarError> {
    match family {
        GlmFamily::Binomial => {
            if y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be 0.0 or 1.0 for Binomial family".to_string(),
                });
            }
        }
        GlmFamily::Poisson => {
            if y.iter().any(|&yi| yi < 0.0 || yi != yi.floor()) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be non-negative integers for Poisson family".to_string(),
                });
            }
        }
        GlmFamily::Gamma => {
            if y.iter().any(|&yi| yi <= 0.0) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be strictly positive for Gamma family".to_string(),
                });
            }
        }
        GlmFamily::Gaussian => {} // unrestricted
    }
    Ok(())
}
```

### Pattern 4: GlmFamily-Specific β Initialization for IRLS

```rust
// Standard zero-init works for Binomial/Poisson/Gaussian.
// For Gamma, initialize intercept = 1/mean(y) to ensure η₀ > 0.
fn init_beta(p: usize, y: &[f64], family: GlmFamily) -> Vec<f64> {
    let mut beta = vec![0.0; p];
    if let GlmFamily::Gamma = family {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        beta[0] = 1.0 / mean_y.max(1e-10); // so η₀ = β₀ = 1/mean(y) > 0
    }
    beta
}
```

### Pattern 5: Crate-Root Re-Export Addition

```rust
// fdars-core/src/lib.rs — additive lines only; existing block untouched.
// Current block [VERIFIED: fdars-core/src/lib.rs:246-254]:
//   pub use scalar_on_function::{ ..., functional_logistic, ..., FunctionalLogisticResult, ... };
// Add alongside:
pub use scalar_on_function::{
    functional_glm, predict_functional_glm, GlmFamily, FunctionalGlmResult,
};
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Weighted normal equations | Custom GEMM + inversion | `cholesky_solve` in `linalg.rs` | Already handles the `(X'WX)β = X'Wz` structure; tested and numerically stable |
| FPC-score computation | Custom SVD | `fdata_to_pc_1d` in `regression.rs` | Handles centering, integration weights (Simpson's), FPCA result assembly |
| Design matrix assembly | Custom intercept/score/scalar stacking | `build_design_matrix` in `scalar_on_function/mod.rs` | Already tested with edge cases |
| β(t) reconstruction | Custom rotation multiply | `recover_beta_t` in `mod.rs` | Already correct for all functional regression paths |
| SE propagation | Custom variance propagation | `compute_beta_se` + `compute_ols_std_errors` | Tested against analytical results in logistic tests |
| logit / sigmoid | Custom implementation | `sigmoid` in `scalar_on_function/mod.rs` | Numerically stable two-branch implementation already present |
| log-gamma function | Custom Stirling series | `statrs::function::gamma::ln_gamma` | Already in crate dependency; accurate |

**Key insight:** The entire IRLS scaffold already exists in `logistic.rs`. The Binomial IRLS loop is a special case of the general GLM IRLS. The only new code is: (a) the `GlmFamily` enum with four method implementations, (b) the generic IRLS step that dispatches to them, and (c) the `FunctionalGlmResult` struct with the two additional fields (`deviance`, `linear_predictors`, `family`) relative to `FunctionalLogisticResult`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Separate estimators per family | Unified GLM with family dispatch | 1980s (McCullagh-Nelder) | One IRLS loop covers all exponential-family members |
| Logistic-only functional regression | Functional GLM via FPC scores | This phase | Poisson + Gamma coverage for FDA |
| Step-halving for non-convergence | Capped iteration (existing policy) | N/A | Simpler; matches existing logistic behavior |

**Deprecated/outdated:**
- Separate Gaussian OLS branch: Not needed — Gaussian IRLS with w=1 converges in 1 step (equivalent to OLS). Implementing as a special case would complicate the code without benefit.

---

## Reuse Targets — Exact Line References

| Asset | File | Lines | Verbatim Key Content |
|-------|------|-------|---------------------|
| `irls_step` (Binomial IRLS step to generalize) | `fdars-core/src/scalar_on_function/logistic.rs` | 14–49 | `w: Vec<f64> = mu.iter().map(\|&p\| (p * (1.0 - p)).max(1e-10)).collect();\nz_work: Vec<f64> = (0..n).map(\|i\| eta[i] + (y[i] - mu[i]) / w[i]).collect();` [VERIFIED: logistic.rs:23-24] |
| `irls_loop` (IRLS iteration with convergence) | `fdars-core/src/scalar_on_function/logistic.rs` | 64–84 | Converges on `max coefficient change < tol`; returns `(beta, iterations)` [VERIFIED: logistic.rs:64-84] |
| `build_logistic_result` (result assembly pattern) | `fdars-core/src/scalar_on_function/logistic.rs` | 87–157 | SE from Fisher info, AIC = deviance + 2p, BIC = deviance + n.ln()*p [VERIFIED: logistic.rs:127-138] |
| `FunctionalLogisticResult` struct | `fdars-core/src/scalar_on_function/mod.rs` | 136–169 | Fields: intercept, beta_t, beta_se, gamma, probabilities, predicted_classes, ncomp, accuracy, std_errors, coefficients, log_likelihood, iterations, fpca, aic, bic [VERIFIED: mod.rs:136-169] |
| `FdarError` variants | `fdars-core/src/error.rs` | 6–25 | `InvalidDimension { parameter: &'static str, expected: String, actual: String }`, `InvalidParameter { parameter: &'static str, message: String }`, `ComputationFailed { operation: &'static str, detail: String }` [VERIFIED: error.rs:6-25] |
| `build_design_matrix` | `fdars-core/src/scalar_on_function/mod.rs` | 452–473 | `pub(crate) fn build_design_matrix(fpca_scores: &FdMatrix, ncomp: usize, scalar_covariates: Option<&FdMatrix>, n: usize) -> FdMatrix` [VERIFIED: mod.rs:452-473] |
| `recover_beta_t` | `fdars-core/src/scalar_on_function/mod.rs` | 476–485 | `fn recover_beta_t(fpc_coeffs: &[f64], rotation: &FdMatrix, m: usize) -> Vec<f64>` [VERIFIED: mod.rs:476-485] |
| `compute_beta_se` | `fdars-core/src/scalar_on_function/mod.rs` | 490–501 | `fn compute_beta_se(gamma_se: &[f64], rotation: &FdMatrix, m: usize) -> Vec<f64>` [VERIFIED: mod.rs:490-501] |
| `compute_fitted` | `fdars-core/src/scalar_on_function/mod.rs` | 503–515 | `fn compute_fitted(design: &FdMatrix, coeffs: &[f64]) -> Vec<f64>` [VERIFIED: mod.rs:503-515] |
| `sigmoid` | `fdars-core/src/scalar_on_function/mod.rs` | 561–568 | `pub(crate) fn sigmoid(x: f64) -> f64` — two-branch stable form [VERIFIED: mod.rs:561-568] |
| Crate-root re-export block | `fdars-core/src/lib.rs` | 246–254 | `pub use scalar_on_function::{ ..., functional_logistic, ..., FunctionalLogisticResult, ... }` [VERIFIED: lib.rs:246-254] |
| `cholesky_factor` re-export | `fdars-core/src/scalar_on_function/mod.rs` | 332 | `pub(crate) use crate::linalg::cholesky_factor;` [VERIFIED: mod.rs:332] |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (cargo test) |
| Config file | None — inline `#[cfg(test)] mod tests` |
| Quick run command | `cargo test -p fdars-core scalar_on_function -- --test-threads=1` |
| Full suite command | `cargo test -p fdars-core --features linalg` |
| Clippy command | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | File | Status |
|--------|----------|-----------|------|--------|
| REG-02 SC1 | `functional_glm(data, y, GlmFamily::Binomial)` returns valid result | unit | `tests.rs` | Wave 0 |
| REG-02 SC2 | Binomial parity with `functional_logistic` (coefficients + fitted values within 1e-6) | unit | `tests.rs` | Wave 0 |
| REG-02 SC3a | Poisson recovery: fitted_values all positive; predictions correlate with known μ | unit | `tests.rs` | Wave 0 |
| REG-02 SC3b | Gamma recovery: fitted_values all positive; predictions correlate with known μ | unit | `tests.rs` | Wave 0 |
| REG-02 SC4 | Negative Poisson count → `InvalidParameter` | unit | `tests.rs` | Wave 0 |
| REG-02 SC4 | Non-positive Gamma response → `InvalidParameter` | unit | `tests.rs` | Wave 0 |
| REG-02 SC4 | Out-of-range Binomial y → `InvalidParameter` | unit | `tests.rs` | Wave 0 |
| REG-02 SC4 | Dimension mismatch → `InvalidDimension` | unit | `tests.rs` | Wave 0 |
| REG-02 SC5 | Full clippy + test suite green | integration | CI / `cargo test --all` | Phase gate |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core scalar_on_function --features linalg -- --test-threads=1`
- **Per wave merge:** `cargo test -p fdars-core --features linalg`
- **Phase gate:** Full suite + clippy green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `fdars-core/src/scalar_on_function/glm.rs` — new file (entire implementation)
- [ ] `GlmFamily` enum + `FunctionalGlmResult` in `mod.rs` — new definitions
- [ ] `pub use glm::{...}` in `mod.rs` barrel
- [ ] `pub use scalar_on_function::{functional_glm, predict_functional_glm, GlmFamily, FunctionalGlmResult}` in `lib.rs`
- [ ] GLM tests in `tests.rs` covering all SC requirements

---

## Security Domain

This phase involves no authentication, input sanitization of untrusted network data, or cryptography. All inputs are numeric arrays processed within the library. ASVS categories V2/V3/V4/V6 do not apply.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | Yes (limited) | Response-domain guards at function entry → `FdarError::InvalidParameter`; no panics |
| V2/V3/V4/V6 | No | N/A |

**Known threat patterns:** Not applicable — this is a pure numeric library with no network, auth, or user session surface.

---

## Environment Availability

This phase is code-only (no external tools beyond the Rust toolchain). All dependencies are already in `Cargo.toml`.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Implementation | ✓ | 1.97.0 | — |
| statrs (ln_gamma) | Poisson log-likelihood | ✓ | (existing dep) | Approximate with `lgamma` via summation |
| cargo clippy | CI gate | ✓ | bundled with toolchain | — |

**Step 2.6: SKIPPED (no external service dependencies — pure Rust library code).**

---

## Open Questions

1. **Gaussian log-likelihood for AIC: kernel vs. full formula?**
   - What we know: Kernel `−Σ(y−μ)²` is proportional to LL but excludes the `−(n/2)·ln(2π·σ²)` term.
   - What's unclear: Whether AIC/BIC for Gaussian GLM should be comparable to Gaussian linear models in other software (e.g., R's `lm`). Using the kernel produces relative rankings among Gaussian GLMs but not cross-family comparisons.
   - Recommendation: Use full formula `ll = −(n/2)·ln(2π·RSS/n) − n/2` for the Gaussian branch. Estimate σ² = RSS/n (ML estimator). This makes `AIC` comparable to `lm()` in R.

2. **Gamma/Gaussian dispersion in AIC: estimate φ or fix φ=1?**
   - What we know: Standard R behavior estimates φ for Gamma and Gaussian (Pearson χ²/(n−p)), then computes profile log-likelihood including φ.
   - What's unclear: CONTEXT.md does not specify. The `FunctionalLogisticResult` uses deviance = −2·LL (no dispersion) which works for Binomial (φ=1).
   - Recommendation: For Gamma, emit `log_likelihood` as the kernel (−Σ y/μ − Σ log μ) and `AIC = −2·ll + 2·(p+1)` where the +1 counts the estimated dispersion parameter. Flag in rustdoc. [ASSUMED]

3. **Convergence criterion: deviance-change vs. coefficient-change?**
   - What we know: Existing `irls_loop` in `logistic.rs` uses max coefficient change. The CONTEXT.md says "deviance/coefficient change < tol".
   - Recommendation: Use deviance-change as the primary criterion (more numerically standard; avoids scale-sensitivity of coefficient magnitudes); fall back to max-iter. This also makes the Binomial parity test tolerant of minor iteration-count differences.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `statrs::function::gamma::ln_gamma` is the correct path for Poisson log-likelihood | Standard Stack | Poisson LL will be wrong or absent; workaround: use sum-of-logs for integer y |
| A2 | Zero-initialized β fails for Gamma (η=0 → μ=1/0); needs intercept initialized to 1/mean(y) | Common Pitfalls (Pitfall 3) | Gamma IRLS diverges in first step; test will catch this |
| A3 | Convergence criterion: deviance-change is preferred over coefficient-change for the generic IRLS | Open Questions | Binomial parity test may fail by a small number of iterations; fix by using same criterion as logistic |
| A4 | AIC for Gamma uses kernel LL + count dispersion as +1 parameter | Open Questions | AIC not comparable to R `glm()` output; cosmetic issue unless user runs cross-validation on AIC |
| A5 | Gaussian AIC uses full log-likelihood with estimated σ² = RSS/n | Open Questions | AIC magnitude differs from R's `lm()`; cosmetic unless cross-family AIC comparison needed |

---

## Sources

### Primary (HIGH confidence — codebase reads)

- `fdars-core/src/scalar_on_function/logistic.rs:14-84` — IRLS step and loop implementation (the exact code to generalize)
- `fdars-core/src/scalar_on_function/mod.rs:136-169` — `FunctionalLogisticResult` struct (template for `FunctionalGlmResult`)
- `fdars-core/src/scalar_on_function/mod.rs:452-568` — shared helpers (build_design_matrix, recover_beta_t, compute_beta_se, sigmoid)
- `fdars-core/src/error.rs:6-25` — `FdarError` enum variants (exact field names)
- `fdars-core/src/lib.rs:246-254` — crate-root re-export block (additive pattern)

### Secondary (MEDIUM confidence — web references)

- [Timothy Barry: GLMs](https://timothy-barry.github.io/posts/2020-07-07-generalized-linear-models/) — IRLS weights, working response, variance functions
- [Germán Rodríguez GLM notes — Poisson](https://grodri.github.io/glms/notes/a2s5) — Poisson deviance, working response, weights
- [glum documentation](https://glum.readthedocs.io/en/latest/glm.html) — deviance definition, AIC/BIC for exponential dispersion families

### Tertiary (LOW confidence — training knowledge)

- Gamma canonical link behavior and η clamping
- lgamma availability in statrs

---

## Metadata

**Confidence breakdown:**
- Standard stack (reuse targets): HIGH — all files read this session with line citations
- IRLS generalization formulas: MEDIUM — confirmed via two independent web sources
- Per-family deviance/LL: MEDIUM — Binomial/Poisson from cited sources; Gamma from training + glum docs
- Pitfalls: MEDIUM — Gamma sign flip and η overflow are mathematically derivable; clamping thresholds are conventional
- Test designs: HIGH — deterministic data patterns are independent of stochastic behavior

**Research date:** 2026-08-17
**Valid until:** 2026-09-17 (stable GLM theory; codebase read this session)
