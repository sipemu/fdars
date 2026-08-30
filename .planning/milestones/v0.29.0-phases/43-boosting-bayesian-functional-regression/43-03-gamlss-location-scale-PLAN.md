---
phase: 43-boosting-bayesian-functional-regression
plan: 03
type: execute
wave: 2
depends_on: [43-01]
files_modified:
  - fdars-core/src/boosting_regression/gamlss.rs
autonomous: true
requirements: [REG-06-03]
estimate:
  tokens: 58000
  raw_tokens: 29000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "gamlss_fosr fits a GAMLSS-style distributional functional regression modelling BOTH location μ(t) and scale σ(t) of a Gaussian response (REG-06-03) via cyclic gamboostLSS-style boosting over the two distributional parameters"
    - "The fitted scale σ̂(t) is strictly positive everywhere (log link + NUMERICAL_EPS clip guard), and the Gaussian log-likelihood is non-decreasing (or improves) along the cyclic boosting path"
    - "gamlss_fosr returns per-parameter functional coefficients (mu_beta, sigma_beta), fitted μ̂(t) and σ̂(t), and the final log-likelihood; error paths return FdarError"
  artifacts:
    - fdars-core/src/boosting_regression/gamlss.rs
  key_links:
    - "gamlss_fosr drives the μ-step and σ-step by reusing the boosting primitive from Plan 01 (boost_fosr_one_step or boost_fosr on the working-residual pseudo-response)"
    - "Gaussian negative gradients: μ-step u = (Y−μ)/σ²; σ-step (log link) u = −1 + (Y−μ)²/σ² (RESEARCH §Algorithm 3)"
---

<objective>
Implement GAMLSS-style distributional functional regression for a Gaussian response modelling location μ(t) and scale σ(t) (REG-06-03), via cyclic gamboostLSS-style boosting: alternate a μ-boosting step (identity link) and a σ-boosting step (log link) each cyclic iteration, using analytically-derived Gaussian negative gradients. This fills the `gamlss.rs` skeleton from Plan 01.

Purpose: Distributional regression (modelling >1 parameter of the response distribution) is absent from fdars; it is the "location + scale" milestone target. Reuses the boosting core from Plan 01 — the only new logic is the two negative-gradient formulas and the cyclic driver.
Output: A working `gamlss_fosr()` returning `GamlssResult`, with inline recovery + positivity + monotonicity + error-path tests.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md
@.planning/phases/43-boosting-bayesian-functional-regression/43-01-boosting-core-fosr-SUMMARY.md

Depends on Plan 01: `BoostingConfig`, `GamlssResult`, and the `boost_fosr_one_step` primitive (or `boost_fosr`). `gamlss.rs` already exists as a compiling skeleton — this plan replaces its body. `NUMERICAL_EPS` (1e-10) is in `src/helpers.rs`. Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.
</context>

<artifacts_produced>
New public symbol implemented (signature declared in Plan 01; drift excludes it):
- `pub fn gamlss_fosr(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BoostingConfig) -> Result<GamlssResult, FdarError>`

`GamlssResult` (Plan 01 mod.rs) fields populated: `mu_fitted`, `sigma_fitted`, `mu_intercept`, `sigma_intercept`, `mu_beta`, `sigma_beta`, `log_likelihood`, `ll_path`, `mstop`, `nu`.

Private helpers (this file, new algorithm logic — no analog):
- `fn mu_neg_gradient(y, mu, sigma) -> FdMatrix` → (Y−μ)/σ² with σ² clipped at NUMERICAL_EPS
- `fn sigma_neg_gradient(y, mu, sigma) -> FdMatrix` → −1 + (Y−μ)²/σ² with σ clipped at NUMERICAL_EPS
- `fn gaussian_loglik(y, mu, sigma) -> f64` → Σ_i Σ_t [−log σ − (Y−μ)²/(2σ²)]
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end gamlss_fosr — cyclic μ/σ boosting with Gaussian negative gradients</name>
  <files>fdars-core/src/boosting_regression/gamlss.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Algorithm 3: model, both negative-gradient derivations, cyclic algorithm steps, σ numerical guard) + Common Pitfall 2 (σ collapse)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md (gamlss.rs section: result struct shape, σ clipping guard, negative-gradient helper signatures, public signature)
    - fdars-core/src/boosting_regression/boost_fosr.rs (Plan 01 — boost_fosr / boost_fosr_one_step contract to drive each parameter's step)
    - fdars-core/src/helpers.rs line 4 (NUMERICAL_EPS)
  </read_first>
  <action>
Replace the `gamlss.rs` skeleton body with the cyclic gamboostLSS implementation per RESEARCH §Algorithm 3.

Validate inputs (same dimension/parameter checks as boost_fosr: n>=3, m>0, predictors.nrows()==n, mstop>=1, 0<nu<=1, nbasis>=4, lambda>0) → `FdarError`.

Implement the two negative-gradient helpers and the log-likelihood helper (see `<artifacts_produced>`), each iterating time-points in the outer loop (contiguous column access) and clipping σ (and σ²) at `NUMERICAL_EPS` before any division per Pitfall 2.

Initialization: μ̂(t) = pointwise column mean of Y; η_σ = 0 → σ̂(t) = exp(0) = 1 everywhere; accumulate mu_beta / sigma_beta as zero (p × m_t).

Cyclic loop for m = 1..=mstop:
1. μ-step: compute U_μ = mu_neg_gradient(Y, μ̂, σ̂); run ONE boosting step (via boost_fosr_one_step, or `boost_fosr` with mstop=1 on U_μ as pseudo-response) selecting the best μ base-learner; update μ̂ += nu·ĥ_μ and accumulate mu_beta.
2. σ-step: compute U_σ = sigma_neg_gradient(Y, μ̂ (updated), σ̂); run ONE boosting step on U_σ; update η_σ += nu·ĥ_σ, then σ̂(i,t) = exp(η_σ(i,t)).max(NUMERICAL_EPS) pointwise; accumulate sigma_beta (coefficients live on the log-σ scale).
3. Push gaussian_loglik(Y, μ̂, σ̂) to ll_path.

Assemble `GamlssResult` (mu_fitted, sigma_fitted, mu_intercept = initial μ column mean, sigma_intercept = exp(initial η_σ)=1, mu_beta, sigma_beta, log_likelihood = last ll_path value, ll_path, mstop, nu). Mark `#[must_use]`. Rustdoc must document: Gaussian family only, identity link for μ + log link for σ, cyclic (not noncyclic) gamboostLSS, and the divergence note (cyclic v1 per CONTEXT decisions) citing Hofner et al. 2016.

Do NOT inline fenced code — follow the RESEARCH §Algorithm 3 equations and the boost_fosr contract named in read_first.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core --features linalg,parallel` exits 0
    - `gamlss.rs` no longer contains the string `not yet implemented`
    - `grep -q "fn mu_neg_gradient" fdars-core/src/boosting_regression/gamlss.rs` AND `grep -q "fn sigma_neg_gradient" fdars-core/src/boosting_regression/gamlss.rs` both succeed
    - `grep -q "NUMERICAL_EPS" fdars-core/src/boosting_regression/gamlss.rs` succeeds (σ guard present)
  </acceptance_criteria>
  <done>gamlss_fosr is implemented end-to-end: cyclic μ/σ boosting with Gaussian negative gradients and the log-link positivity guard; the crate compiles.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: gamlss_fosr positivity, recovery, monotonicity, and error-path tests + gate</name>
  <files>fdars-core/src/boosting_regression/gamlss.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Validation Architecture → REG-06-03 rows: σ̂(t) > 0 everywhere; log-likelihood non-decreasing)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-VALIDATION.md (GAMLSS oracles: recover known μ(t) and σ(t) on heteroscedastic data)
    - fdars-core/src/test_helpers.rs (`uniform_grid`)
  </read_first>
  <behavior>
    - sigma_fitted[(i,t)] > 0 for every (i,t) (log link + clip guarantees positivity)
    - ll_path is non-decreasing (Gaussian log-likelihood improves or holds across cyclic iterations) on a signal-bearing heteroscedastic synthetic dataset
    - On synthetic heteroscedastic data with a known μ(t) (driven by a scalar predictor) and non-constant σ(t), fitted μ̂ tracks the true mean and σ̂ captures the larger-variance region (larger where true σ is larger)
    - mu_fitted and sigma_fitted both have shape (n, m_t); mu_beta and sigma_beta both (p, m_t)
    - Error path: predictors.nrows() != data.nrows() returns FdarError::InvalidDimension
    - Error path: mstop == 0 (or nu<=0, lambda<=0) returns FdarError::InvalidParameter
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests` block. Build a synthetic heteroscedastic Gaussian dataset (n>=30, m>=15) where the mean is driven by a scalar predictor and the noise standard deviation varies across the grid (e.g. larger in one region). Write tests for every `<behavior>` bullet: `gamlss_sigma_positive_everywhere`, `gamlss_loglik_non_decreasing`, `gamlss_recovers_mean_and_scale`, `gamlss_result_shapes`, `gamlss_errors_on_dimension_mismatch`, `gamlss_errors_on_invalid_params`. For the recovery test keep tolerances loose (structural: σ̂ larger where true σ is larger, μ̂ correlates with true mean). Then run the full clippy gate + full test suite; fix any findings in this file.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::gamlss 2>&1 | tail -12 && TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg,parallel boosting_regression::gamlss` exits 0
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits 0
    - `grep -c "#\[test\]" fdars-core/src/boosting_regression/gamlss.rs` returns >= 5
  </acceptance_criteria>
  <done>REG-06-03 has inline positivity + recovery + monotonicity + error-path tests, all green; clippy gate clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure numeric in-process Rust library function. No I/O, network, deserialization, auth, or external attack surface — inputs are in-memory `FdMatrix` + config from calling Rust code. |

## STRIDE Threat Register

Attack surface: NONE — pure numeric computation on in-memory `FdMatrix`. Only failure modes are numerical, handled as `FdarError` returns / numerical guards.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-43-03a | Tampering (numerical) | σ→0 collapse in GAMLSS gradients | low | mitigate | Clip σ and σ² at `NUMERICAL_EPS` before any division in both negative-gradient helpers and the log-likelihood (RESEARCH Pitfall 2); NaN/Inf cannot propagate |
| T-43-03b | DoS (numerical) | `mstop` / config params | low | mitigate | Validate mstop>=1, 0<nu<=1, nbasis>=4, lambda>0 at entry → `FdarError::InvalidParameter` |

No package installs. No supply-chain checkpoint required.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::gamlss` green (REG-06-03).
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- Only `gamlss.rs` modified (no mod.rs edit — collision-free with sibling wave-2 plans).
</verification>

<success_criteria>
- REG-06-03 satisfied: `gamlss_fosr` fits a Gaussian location+scale distributional model via cyclic boosting, with guaranteed-positive σ̂(t), non-decreasing log-likelihood path, per-parameter coefficients, and error-path handling.
- Inline tests green; full clippy gate clean.
</success_criteria>

<output>
Create `.planning/phases/43-boosting-bayesian-functional-regression/43-03-gamlss-location-scale-SUMMARY.md` when done.
</output>
