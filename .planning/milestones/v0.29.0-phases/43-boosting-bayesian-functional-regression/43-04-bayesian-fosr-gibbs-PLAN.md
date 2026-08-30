---
phase: 43-boosting-bayesian-functional-regression
plan: 04
type: execute
wave: 2
depends_on: [43-01]
files_modified:
  - fdars-core/src/boosting_regression/bayesian.rs
autonomous: true
requirements: [REG-06-04]
estimate:
  tokens: 60000
  raw_tokens: 30000
  tasks: 2
  confidence: low
must_haves:
  truths:
    - "bayesian_fosr fits a Bayesian function-on-scalar regression via a conjugate Normal / Inverse-Gamma Gibbs sampler on FPC-score coefficients (REG-06-04), producing a posterior-mean β(t) plus pointwise credible bands"
    - "The sampler is deterministic under a fixed seed: two runs with the same BayesianConfig.seed yield identical results (StdRng::seed_from_u64)"
    - "The posterior mean β(t) is close to a penalized/OLS point estimate, and the pointwise credible bands cover a known synthetic truth at close to the nominal rate; error paths (burn_in>=n_iter, dimension mismatch) return FdarError"
  artifacts:
    - fdars-core/src/boosting_regression/bayesian.rs
  key_links:
    - "bayesian_fosr calls fdata_to_pc_1d to compress the response into FPC scores, then runs conjugate Gibbs draws (Normal for γ via Cholesky, Inverse-Gamma for σ² via rand_distr::Gamma) seeded by StdRng::seed_from_u64(config.seed)"
    - "IG(α,β) draw uses rand_distr::Gamma with scale = 1/β (0.4 param: Gamma::new(shape, scale)); IG sample = 1.0 / Gamma(α, 1/β)"
---

<objective>
Implement Bayesian function-on-scalar regression via a conjugate Normal / Inverse-Gamma Gibbs sampler (REG-06-04): compress the response via FPCA, run a seeded deterministic Gibbs chain (Normal full-conditional for the FPC-space coefficients γ(t), Inverse-Gamma for the pointwise variances σ²(t)), and summarise the retained thinned draws into a posterior-mean β(t) and pointwise credible bands. This fills the `bayesian.rs` skeleton from Plan 01.

Purpose: fdars has no Bayesian regression machinery. The conjugate Gibbs form reuses existing Cholesky + FPCA primitives and the transitively-available `rand_distr` distributions — no new dependency, fully deterministic under seed.
Output: A working `bayesian_fosr()` returning `BayesianFosrResult` (posterior mean + pointwise credible bands + posterior σ²), with inline tests for point-estimate agreement, band coverage, determinism, and error paths.
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

Depends on Plan 01: `BayesianConfig`, `BayesianFosrResult`. `bayesian.rs` already exists as a compiling skeleton — this plan replaces its body. RESOLVED research assumptions (do NOT re-flag): `rand_distr = "0.4"` is a direct dependency (Cargo.toml:37); `rand_distr::Gamma::new(shape, scale)` uses the SCALE parameterization, so IG(α,β) draw = `1.0 / rng.sample(Gamma::new(α, 1.0/β))`. `StandardNormal`/`Normal` already used elsewhere (`src/outliers.rs`, `src/simulation.rs`). Cholesky helpers are `pub(crate)`. Prefix cargo with `TMPDIR=/home/simonm/.cache/fdars-bench-tmp`.
</context>

<artifacts_produced>
New public symbol implemented (signature declared in Plan 01; drift excludes it):
- `pub fn bayesian_fosr(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BayesianConfig) -> Result<BayesianFosrResult, FdarError>`

`BayesianFosrResult` (Plan 01 mod.rs) fields populated: `beta_mean`, `beta_lower`, `beta_upper`, `fitted`, `residuals`, `sigma2_mean`, `n_iter`, `burn_in`, `thin`, `ncomp`.

Private helpers (this file, new MCMC logic — no analog):
- `fn gibbs_draw_gamma(...)` — multivariate Normal full-conditional draw for γ(t) via `cholesky_factor` of the posterior covariance + `StandardNormal` z
- `fn gibbs_draw_sigma2(...)` — Inverse-Gamma draw via `rand_distr::Gamma` (scale = 1/rate) as `1.0 / Gamma(a0 + n/2, 1/(b0 + RSS/2))`
</artifacts_produced>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end bayesian_fosr — FPCA compression + conjugate Gibbs + credible bands</name>
  <files>fdars-core/src/boosting_regression/bayesian.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Algorithm 4: model in FPC score space, conjugate priors, both full-conditional distributions, Gibbs loop, credible-band construction, and the divergence-from-refund note) + Common Pitfall 3 (Gibbs mixing) + Open Question 2 (Gamma parameterization — RESOLVED: scale param)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md (bayesian.rs section: imports, single-chain seeding, result struct shape, FPCA preprocessing, quantile helper via quantile_sorted)
    - fdars-core/src/scalar_on_function/bootstrap.rs lines 80-100 (StdRng::seed_from_u64 seeding pattern)
    - fdars-core/src/regression.rs lines 25-38, 287-292 (FpcaResult: rotation m × K, scores n × K; fdata_to_pc_1d)
    - fdars-core/src/helpers.rs line 283 (quantile_sorted for credible-band quantiles)
    - fdars-core/src/simulation.rs lines 26, 316 (Normal::new / StandardNormal usage) for the Gamma/Normal draw idiom
  </read_first>
  <action>
Replace the `bayesian.rs` skeleton body with the conjugate Gibbs implementation per RESEARCH §Algorithm 4.

Validate inputs: n>=3, m>0, predictors.nrows()==n → `FdarError::InvalidDimension`; config: ncomp>=1, tau2>0, ig_a0>0, ig_b0>0, n_iter>0, thin>=1, burn_in < (burn_in + n_iter*thin) i.e. at least one retained draw, ncomp<=n → `FdarError::InvalidParameter`.

Note on model: the response is the functional Y (n × m_t); use FPC scores of the response as the score design S = fpca.scores (n × K) per RESEARCH §Algorithm 4 (score-space regression). Compress via `let fpca = fdata_to_pc_1d(data, config.ncomp, argvals)?;`. The regression is per response time point t: y(t) = S·γ(t) + ε(t). (Predictors provide the modelled coefficient reconstruction target; follow RESEARCH §Algorithm 4's score-space form and document the pointwise-Gibbs limitation from Open Question 3 in rustdoc.)

Seed a single chain: `let mut rng = StdRng::seed_from_u64(config.seed);`. Initialize γ(t)=0, σ²(t)=1 for all t. Precompute S'S (K×K) once. Gibbs loop over `burn_in + n_iter*thin` iterations; per iteration, for each time point t:
- γ(t) draw (gibbs_draw_gamma): posterior precision = S'S/σ²(t) + I_K/τ²; posterior covariance Σ_post = inverse (via cholesky_factor + solves); posterior mean μ_post = Σ_post · S'y(t)/σ²(t); draw γ = μ_post + L·z where L = cholesky_factor(Σ_post) and z ~ StandardNormal per component.
- σ²(t) draw (gibbs_draw_sigma2): RSS(t) = ‖y(t) − S·γ(t)‖²; draw σ²(t) ~ IG(a0 + n/2, b0 + RSS/2) as `1.0 / rng.sample(Gamma::new(a0 + n/2.0, 1.0/(b0 + rss/2.0))?)`.
After burn-in and every `thin`-th iteration, store the reconstructed β_draw(t) = fpca.rotation · γ(t) (m_t-vector) into the retained draws.

Summaries: beta_mean = mean over retained draws; beta_lower / beta_upper = 2.5% / 97.5% pointwise quantiles via `quantile_sorted` over sorted per-time-point draws; sigma2_mean = mean σ²(t); fitted = posterior-mean reconstruction, residuals = Y − fitted. Assemble `BayesianFosrResult` with n_iter, burn_in, thin, ncomp. Mark `#[must_use]`. Rustdoc must document the divergence from `refund` (FPCA score-compression + pointwise conjugate Gibbs instead of spline-space random effects; pointwise — not simultaneous — credible bands) citing Jiang et al. 2025.

Do NOT inline fenced code — follow the RESEARCH §Algorithm 4 full-conditional equations and the seeding/quantile patterns named in read_first.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo build -p fdars-core --features linalg,parallel 2>&1 | tail -5</automated>
  </verify>
  <acceptance_criteria>
    - `cargo build -p fdars-core --features linalg,parallel` exits 0
    - `bayesian.rs` no longer contains the string `not yet implemented`
    - `grep -q "seed_from_u64" fdars-core/src/boosting_regression/bayesian.rs` succeeds (seeded chain)
    - `grep -q "Gamma::new" fdars-core/src/boosting_regression/bayesian.rs` succeeds (IG draw wired)
    - `grep -q "beta_lower" fdars-core/src/boosting_regression/bayesian.rs` succeeds (credible bands present)
  </acceptance_criteria>
  <done>bayesian_fosr is implemented end-to-end: FPCA compression, seeded conjugate Gibbs, posterior mean + pointwise credible bands; the crate compiles.</done>
</task>

<task type="auto" tdd="true">
  <name>Task 2: bayesian_fosr point-estimate, coverage, determinism, and error-path tests + gate</name>
  <files>fdars-core/src/boosting_regression/bayesian.rs</files>
  <read_first>
    - .planning/phases/43-boosting-bayesian-functional-regression/43-RESEARCH.md (Validation Architecture → REG-06-04 rows: posterior mean ≈ penalized OLS; bands cover truth at ≥90% of grid points; identical results for same seed)
    - .planning/phases/43-boosting-bayesian-functional-regression/43-VALIDATION.md (Bayesian oracles)
    - fdars-core/src/test_helpers.rs (`uniform_grid`)
  </read_first>
  <behavior>
    - Determinism: two `bayesian_fosr` calls with the same config.seed produce bit-identical results (beta_mean, beta_lower, beta_upper, sigma2_mean all equal)
    - Coverage: on synthetic data with a known β(t) truth, the pointwise [beta_lower, beta_upper] band contains the truth at a high fraction of grid points (>= ~0.8, allowing conservative pointwise bands)
    - Point estimate: beta_mean is finite everywhere and ordered beta_lower <= beta_mean <= beta_upper at every grid point
    - sigma2_mean entries are strictly positive and finite
    - Error path: burn_in configured so that zero draws are retained returns FdarError::InvalidParameter
    - Error path: predictors.nrows() != data.nrows() returns FdarError::InvalidDimension
  </behavior>
  <action>
Add an inline `#[cfg(test)] mod tests` block. Build a synthetic FOSR dataset (n>=30, m>=15) with a known smooth β(t) truth and moderate noise, using a small config (e.g. ncomp=4, n_iter=300, burn_in=100, thin=2) to keep the test fast but stable. Write tests for every `<behavior>` bullet: `bayesian_deterministic_under_seed`, `bayesian_band_covers_truth`, `bayesian_band_ordering`, `bayesian_sigma2_positive`, `bayesian_errors_on_zero_retained_draws`, `bayesian_errors_on_dimension_mismatch`. Keep coverage/agreement tolerances loose (pointwise bands are conservative). Then run the full clippy gate + full test suite; fix any findings in this file.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::bayesian 2>&1 | tail -12 && TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings 2>&1 | tail -6</automated>
  </verify>
  <acceptance_criteria>
    - `cargo test -p fdars-core --features linalg,parallel boosting_regression::bayesian` exits 0
    - `cargo clippy --all-targets --features linalg,parallel -- -D warnings` exits 0
    - `grep -c "#\[test\]" fdars-core/src/boosting_regression/bayesian.rs` returns >= 5
    - A determinism test asserting two same-seed runs are equal is present
  </acceptance_criteria>
  <done>REG-06-04 has inline determinism + coverage + ordering + error-path tests, all green; clippy gate clean.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| (none) | Pure numeric in-process Rust library function. No I/O, network, deserialization, auth, or external attack surface — inputs are in-memory `FdMatrix` + config from calling Rust code. RNG is for statistical reproducibility, not security. |

## STRIDE Threat Register

Attack surface: NONE — pure numeric computation on in-memory `FdMatrix`. Only failure modes are numerical, handled as `FdarError` returns.

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-43-04a | DoS (numerical) | `burn_in >= n_iter` giving zero retained draws | low | mitigate | Validate at least one retained draw (burn_in < burn_in + n_iter*thin, n_iter>0, thin>=1) at entry → `FdarError::InvalidParameter` |
| T-43-04b | Tampering (numerical) | posterior-covariance Cholesky on large/ill-conditioned K | low | mitigate | `cholesky_factor` returns `FdarError::ComputationFailed` on non-PD; validate `ncomp<=n`; weakly-informative IG(0.001,0.001) + τ²=100 defaults keep Σ_post well-conditioned (Pitfall 3) |

No package installs (`rand_distr` already a direct dependency). No supply-chain checkpoint required.
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel boosting_regression::bayesian` green (REG-06-04), including the same-seed determinism test.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.
- Only `bayesian.rs` modified (no mod.rs edit — collision-free with sibling wave-2 plans).
</verification>

<success_criteria>
- REG-06-04 satisfied: `bayesian_fosr` fits a conjugate Gibbs Bayesian FOSR producing posterior-mean β(t) + pointwise credible bands + posterior σ², deterministic under seed, with error-path handling.
- Inline tests green; full clippy gate clean.
</success_criteria>

<output>
Create `.planning/phases/43-boosting-bayesian-functional-regression/43-04-bayesian-fosr-gibbs-SUMMARY.md` when done.
</output>
