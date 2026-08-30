# Phase 43: Boosting / Bayesian Functional Regression - Research

**Researched:** 2026-08-23
**Domain:** Statistical gradient boosting + Bayesian inference for functional data (Rust, no new deps)
**Confidence:** MEDIUM (algorithm derivations from published literature; codebase signatures verified in-session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Functional base-learner family: penalized B-spline base-learners, reusing `smooth_basis`/`FdPar` penalty-matrix machinery (`bspline_penalty_matrix`) — matches FDboost `bbs()`.
- Per-iteration base-learner selection: component-wise — select the single base-learner minimizing residual sum of squares per boosting iteration (FDboost standard).
- Step size ν (learning rate): fixed ν = 0.1 (FDboost default), configurable via `BoostingConfig`.
- Stopping rule: fixed `mstop` from config, with GCV/AIC tracked along the boosting path (early-stopping optional/deferred).
- Same boosting framework serves both boosted FOSR (function-on-scalar response) and boosted FoFR (function-on-function) base-learners.
- Distribution family for GAMLSS: Gaussian, modelling location μ(t) + scale σ(t).
- Fitting scheme for GAMLSS: component-wise boosting cycling over distributional parameters (gamboostLSS style).
- Link functions: identity for μ, log for σ (guarantees positivity of scale).
- Sampler for Bayesian FOSR: Gibbs sampler on FPC-score coefficients (conjugate, deterministic + seeded via `StdRng::seed_from_u64`).
- Prior structure: Normal prior on coefficients + Inverse-Gamma on variances (conjugate, weakly-informative).
- Credible bands: pointwise credible bands from posterior quantiles.
- Resampling scheme for stability selection: subsampling ⌊n/2⌋ without replacement, B resamples (Meinshausen–Bühlmann / FDboost default), seeded per replicate (`seed.wrapping_add(b)`).
- Module layout: folder `src/boosting_regression/` with submodules (boost_fosr, boost_fofr, gamlss, bayesian, stability) + `mod.rs` barrel.
- Config/result API: `BoostingConfig` builder struct + per-method Result structs following the existing `FosrResult` field convention.
- No new crate dependency.
- Additive/non-breaking only; zero changes to existing public signatures.

### Claude's Discretion

- Exact submodule split, internal helper structure, per-struct field naming, and plan decomposition.
- Precise credible-band / ICL-style diagnostic numerics beyond posterior mean + pointwise bands.

### Deferred Ideas (OUT OF SCOPE)

- Variational Bayes (VB) alternative to Gibbs; simultaneous (rather than pointwise) credible bands; horseshoe/g-prior alternatives.
- Line-search optimal boosting step and CV-selected `mstop`.
- Full mgcv/BayesX-grade sampler diagnostics (multiple chains, R̂, convergence tests).
- Additional GAMLSS distribution families / shape parameters beyond Gaussian location+scale.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-06-01 | User can fit component-wise gradient-boosting functional regression with functional base-learners for a function-on-scalar response (boosted FOSR), selecting one base-learner per iteration | Algorithm 1 (boosting core), base-learner fit via existing `bspline_penalty_matrix` + `cholesky_solve` |
| REG-06-02 | User can fit component-wise gradient-boosting functional regression for a function-on-function predictor/response (boosted FoFR base-learners) | Algorithm 2 (bfpc / signal-compression variant), reuses `fdata_to_pc_1d` + boosting core |
| REG-06-03 | User can fit a GAMLSS-style distributional functional regression — modelling location + scale | Algorithm 3 (cyclic gamboostLSS), negative-gradient derivations for Gaussian μ and log-link σ |
| REG-06-04 | User can fit a Bayesian function-on-scalar regression via a Gibbs sampler, obtaining coefficient posterior summaries (mean + credible bands) | Algorithm 4 (conjugate Gibbs), full-conditional derivations, credible-band reconstruction |
| REG-06-05 | User can run FDboost-style stability selection over the boosting base-learners to obtain selection frequencies / a stable predictor set | Algorithm 5 (subsampling + frequency aggregation + PFER bound) |
</phase_requirements>

---

## Summary

Phase 43 adds a new `src/boosting_regression/` module implementing five related but distinct algorithms. All five share the project's core building blocks: `FdMatrix` (column-major), `fdata_to_pc_1d`/`FpcaResult` for dimension reduction, `bspline_penalty_matrix` + `cholesky_solve` for penalized normal-equation solves, and `StdRng::seed_from_u64` for deterministic randomness. No new crate dependencies are needed.

The algorithms are theoretically well-understood and map directly to the existing codebase machinery: (1) the boosting core (REG-06-01) is a loop over `mstop` iterations, fitting the negative gradient with the same penalized B-spline solve already used in `smooth_basis`; (2) the FoFR extension (REG-06-02) replaces scalar-predictor design columns with FPC scores of the functional predictor; (3) GAMLSS (REG-06-03) runs two interleaved boosting loops, one per distributional parameter, with analytically derived negative gradients; (4) the Bayesian sampler (REG-06-04) is a textbook conjugate Gibbs on FPC-score regression — two closed-form full-conditional draws per iteration; (5) stability selection (REG-06-05) is a subsampling wrapper around the boosting fit, aggregating selection frequencies.

The main implementation risk is numerical: the penalized normal equations for base-learner fitting can become ill-conditioned when `lambda` is too small or bases are nearly collinear. Gibbs mixing can stall if the prior variance `tau^2` is poorly specified. Both are mitigated by the ridge jitter already used in `smooth_basis.rs` and by weakly-informative IG(0.001, 0.001) priors.

**Primary recommendation:** Decompose into five PLAN.md tasks (one per submodule) sharing a Wave 0 for `BoostingConfig`, `mod.rs` barrel, and integration tests. The boosting core (REG-06-01) should be Wave 1 and must be locked before REG-06-02/REG-06-03/REG-06-05, which all call it. REG-06-04 (Bayesian Gibbs) is fully independent and can be a separate wave.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Boosted FOSR base-learner fit | `src/boosting_regression/boost_fosr.rs` | `src/linalg.rs`, `src/smooth_basis.rs` | Penalized normal-equation solve lives in linalg; basis matrices from smooth_basis |
| Boosted FoFR base-learner fit | `src/boosting_regression/boost_fofr.rs` | `src/regression.rs` (fdata_to_pc_1d) | FPC score compression of functional predictor lives in regression module |
| GAMLSS cyclic distributional boosting | `src/boosting_regression/gamlss.rs` | `src/boosting_regression/boost_fosr.rs` | Calls boosting core per parameter; Gaussian negative-gradient logic is self-contained |
| Bayesian Gibbs sampler | `src/boosting_regression/bayesian.rs` | `src/regression.rs`, `src/linalg.rs` | FPCA dimension reduction then conjugate Gibbs draws using Cholesky |
| Stability selection subsampling | `src/boosting_regression/stability.rs` | `src/boosting_regression/boost_fosr.rs` | Resampling wrapper around the boosting fit; parallelisable via iter_maybe_parallel! |
| Config + result types | `src/boosting_regression/mod.rs` | — | Shared `BoostingConfig`, per-method Result structs; barrel re-exports |

---

## Standard Stack

### Core (All in-crate — no new dependencies)

| Component | Location (verified) | Purpose | Used by |
|-----------|---------------------|---------|---------|
| `bspline_penalty_matrix` | `src/smooth_basis.rs:82` | K×K roughness penalty matrix R for penalized normal equations | All base-learner fits |
| `smooth_basis` / `FdPar` / `BasisType` | `src/smooth_basis.rs:174` | Basis evaluation + penalty setup | Base-learner design matrix construction |
| `cholesky_factor` / `cholesky_solve` / `cholesky_forward_back` | `src/linalg.rs:85,131` | Penalized least-squares solve `(X'X + λR)c = X'u` | Every base-learner fit per boosting iteration |
| `compute_xtx` | `src/linalg.rs:137` | Normal equations `X'X` | Boosting normal equations |
| `fdata_to_pc_1d` | `src/regression.rs:287` | FPCA — FPC scores + rotation | Boosted FoFR predictor compression; Bayesian FOSR |
| `FpcaResult` (fields: `singular_values, rotation, scores, mean, centered, weights`) | `src/regression.rs:25-38` | Stores FPCA output; `.project()` and `.reconstruct()` methods | Bayesian FOSR; FoFR |
| `simpsons_weights` | `src/helpers.rs:57` | Integration weights for functional inner products | Base-learner normalisation; coefficient reconstruction |
| `iter_maybe_parallel!` / `slice_maybe_parallel!` | `src/parallel.rs:42,62` | Feature-gated rayon parallelism | Stability-selection resample loop |
| `StdRng::seed_from_u64` (pattern from `src/scalar_on_function/bootstrap.rs:89`) | (pattern, not a fn) | Deterministic RNG per replicate: `seed.wrapping_add(b as u64)` | Gibbs sampler; stability selection |
| `FosrResult` fields (template) | `src/function_on_scalar.rs:29-48` | Field naming convention: intercept, beta, fitted, residuals, r_squared_t, r_squared, lambda, gcv | New Result structs follow this layout |

[VERIFIED: src/smooth_basis.rs:82-116] — `pub fn bspline_penalty_matrix(argvals: &[f64], nbasis: usize, order: usize, lfd_order: usize) -> Vec<f64>`

[VERIFIED: src/linalg.rs:85-107] — `pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError>`

[VERIFIED: src/linalg.rs:131-134] — `pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>, FdarError>`

[VERIFIED: src/regression.rs:287-292] — `pub fn fdata_to_pc_1d(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FpcaResult, FdarError>`

[VERIFIED: src/regression.rs:25-38] — `FpcaResult { singular_values: Vec<f64>, rotation: FdMatrix, scores: FdMatrix, mean: Vec<f64>, centered: FdMatrix, weights: Vec<f64> }`

[VERIFIED: src/helpers.rs:57] — `pub fn simpsons_weights(argvals: &[f64]) -> Vec<f64>`

[VERIFIED: src/function_on_scalar.rs:29-48] — `FosrResult { intercept: Vec<f64>, beta: FdMatrix, fitted: FdMatrix, residuals: FdMatrix, r_squared_t: Vec<f64>, r_squared: f64, beta_se: FdMatrix, lambda: f64, gcv: f64 }`

### No External Packages — Legitimacy Audit N/A

This phase installs zero new crate dependencies. The Package Legitimacy Audit section is omitted by design (milestone constraint: no new crates).

---

## Architecture Patterns

### Recommended Project Structure

```
src/boosting_regression/
├── mod.rs              # BoostingConfig, BayesianConfig, StabilityConfig; barrel pub use
├── boost_fosr.rs       # REG-06-01: boosted FOSR (function-on-scalar response)
├── boost_fofr.rs       # REG-06-02: boosted FoFR (function-on-function)
├── gamlss.rs           # REG-06-03: GAMLSS location+scale distributional boosting
├── bayesian.rs         # REG-06-04: conjugate Gibbs FOSR
└── stability.rs        # REG-06-05: stability selection wrapper
```

Registration in `src/lib.rs` at the module list (~line 104 after `pub mod regression;`) and crate-root re-exports (~line 287 region). Key result types also added to `src/prelude.rs`.

### System Architecture Diagram

```
Scalar predictors (n x p)              Functional response Y(t) (n x m)
        │                                          │
        ▼                                          ▼
  [boost_fosr] ──── penalized B-spline ──►  Residual U(t) (n x m)
        │              base-learner fit          │
        │           (Φ'Φ + λR)c = Φ'u           ▼
        │                                  best h_j*(x,t)
        └──────────────────────────────► F += ν * h_j*  ──► BoostFosrResult

Functional predictor X(s) (n x m_x)   Functional response Y(t) (n x m_y)
        │                                          │
        ▼                                          ▼
  fdata_to_pc_1d → FPC scores (n x K)     Residual U(t) (n x m_y)
        │                                          ▼
  [boost_fofr] ──── OLS on scores ──────► BoostFofrResult (coef surface)

Y(t) (n x m)  →  [gamlss]  →  μ̂(t) + σ̂(t)  →  GamlssResult
                   ↕ cyclic
             boost_fosr core (2x)

Y(t), X_scalar (n x p)  →  fdata_to_pc_1d  →  scores (n x K)
        │
        ▼
  [bayesian] ─── conjugate Gibbs (N draw γ, IG draw σ²) ──► BayesianFosrResult
  StdRng(seed)

[stability] ─── B subsamples of size ⌊n/2⌋ ──► B × boost_fosr runs
        │
        └── aggregate selection freq per base-learner ──► StabilityResult
            PFER bound: E[V] ≤ q² / ((2π_thr − 1)·p)
```

---

## Algorithm 1: Component-wise Gradient Boosting for Functional Response (REG-06-01)

[ASSUMED] — Algorithm derivation from FDboost/mboost literature; not from a single authoritative source read verbatim.

### Model

```
Y_i(t) = F(x_i, t) + ε_i(t),   F(x,t) = F_0(t) + Σ_j h_j(x, t)
```

where each base-learner `h_j` is a penalized B-spline in scalar predictor `x_j`.

### Algorithm Steps

**Initialization:** `F̂_0(t) = Ȳ(t)` (pointwise mean of Y, an `m`-vector).

**For** `m = 1, …, mstop`:

1. **Negative gradient (working residual):** For L2 loss,
   ```
   u_i(t) = Y_i(t) − F̂_{m−1}(x_i, t)   ∀ i ∈ {1..n}, t ∈ {1..m}
   ```
   Result: `U` is an `(n × m_t)` `FdMatrix`.

2. **Fit each base-learner `j` to `U`:** For each predictor `j = 1..p`:
   - Build design matrix `Φ_j` (`n × K`): evaluate B-spline basis at scalar predictor values `x_{1j}..x_{nj}` (one column per basis function).
   - Compute penalty matrix `R_j = bspline_penalty_matrix(x_argvals, nbasis, order=4, lfd_order=2)` — `K × K`.
   - Solve the penalized normal equations for each response time point `t`:
     ```
     (Φ_j'Φ_j + λ·R_j) · ĉ_j(t) = Φ_j' · U[:,t]
     ```
     Using `cholesky_solve`: factor `(Φ_j'Φ_j + λ·R_j)` once (it doesn't depend on `t`), back-solve for each `t`. Gives `ĉ_j ∈ R^K` per time point → `h_j(x_i, t) = Φ_j[i,:] · ĉ_j(t)`.

3. **Select best base-learner:**
   ```
   j* = argmin_j  Σ_i Σ_t  (u_i(t) − h_j(x_i, t))²
   ```
   Equivalently: `j* = argmax_j RSS_reduction_j` where `RSS_reduction_j = ||U||_F² − ||U − Ĥ_j||_F²`.

4. **Update:**
   ```
   F̂_m(x_i, t) = F̂_{m−1}(x_i, t) + ν · h_{j*}(x_i, t)
   Coef_{j*}(t) += ν · ĉ_{j*}(t)     (accumulate per-predictor coefficients)
   ```
   All other coefficient vectors remain unchanged.

5. **Track GCV/AIC** (optional, for path diagnostics): record `||U||_F²` per iteration.

**Output fields (follows `FosrResult` convention):**
```rust
pub struct BoostFosrResult {
    pub intercept: Vec<f64>,          // F_0(t), length m_t
    pub beta: FdMatrix,               // p × m_t accumulated coefficient matrix
    pub fitted: FdMatrix,             // n × m_t fitted functional values
    pub residuals: FdMatrix,          // n × m_t residuals
    pub r_squared_t: Vec<f64>,        // pointwise R² (length m_t)
    pub r_squared: f64,               // global integrated R²
    pub mstop: usize,                 // iterations used
    pub nu: f64,                      // learning rate used
    pub selected_learners: Vec<usize>, // which base-learner j* was selected at each iteration
    pub gcv_path: Vec<f64>,           // GCV per iteration (for path diagnostics)
}
```

### Critical Implementation Note

The Cholesky factor `L_j = cholesky_factor(Φ_j'Φ_j + λ·R_j, K)` is formed once per base-learner per iteration (or cached since `Φ_j` doesn't change across iterations). The only per-time-point work is the back-solve `cholesky_forward_back(L_j, Φ_j'·U[:,t], K)`. This amortization is the key to efficient functional-response boosting.

---

## Algorithm 2: Boosted FoFR — Function-on-Function Base-Learners (REG-06-02)

[ASSUMED] — bfpc/bsignal pattern from FDboost documentation; mapped to fdars conventions.

### Model

```
Y_i(t) = F_0(t) + Σ_j ∫ X_j,i(s) β_j(s,t) ds + ε_i(t)
```

For the **bfpc** (FPC-compression) variant used here:

```
∫ X_j,i(s) β_j(s,t) ds ≈ Σ_k ξ_{ijk} γ_{jk}(t)
```

where `ξ_{ijk} = FPC score k of curve X_j,i` and `γ_{jk}(t)` is a smooth coefficient function.

### Algorithm Steps

**Preprocessing:** For each functional predictor `j`:
- Compute `FpcaResult_j = fdata_to_pc_1d(X_j, ncomp_x, x_argvals)` → scores `S_j ∈ R^{n × K_j}`.
- Store scores as the "design matrix" for base-learner `j`.

**Boosting loop** (same structure as Algorithm 1):

1. Negative gradient `U` is still `(n × m_t)`.

2. **Fit base-learner `j`:** treat each FPC score column `k` as a scalar predictor column, giving a "stacked" design matrix `Φ_{fofr,j} = S_j ∈ R^{n × K_j}` (no additional B-spline expansion of the predictor — the FPC basis already provides smoothness).
   - Penalty `R_j = bspline_penalty_matrix(t_argvals, nbasis_t, order=4, lfd_order=2)` applied to the response direction.
   - Alternatively (simpler for v1): solve the unpenalized OLS in score space per response time point:
     ```
     (S_j'S_j) · ĉ_j(t) = S_j' · U[:,t]
     ```
   - Fitted: `Ĥ_j(t) = S_j · ĉ_j(t)` — a matrix product giving `n × m_t`.

3. Select `j*`, update with ν.

4. **Reconstruct coefficient surface:** `β̂_j*(s,t) = Φ_{FPC,j*}(s)' · ĉ_{j*}(t)` — outer product of FPC eigenfunctions `(m_x × K)` and score coefficients `(K × m_t)`, giving `m_x × m_t`.

**Output (`BoostFofrResult`):** includes fitted `(n × m_t)`, residuals, accumulated score-coefficient matrices per predictor, and reconstructed coefficient surfaces on demand.

**Divergence from R baseline:** FDboost's `bsignal` uses trapezoidal-rule integration over a B-spline basis for `β(s,t)` jointly; the bfpc variant used here compresses via truncated KL expansion, which is simpler to implement without new deps and is an accepted equivalent. Document in rustdoc.

---

## Algorithm 3: GAMLSS Distributional Boosting — Location + Scale (REG-06-03)

[ASSUMED] — Cyclic gamboostLSS algorithm from Hofner et al. (2016), Journal of Statistical Software 74(1).

### Model

Gaussian: `Y_i(t) ~ N(μ_i(t), σ_i(t)²)`

Predictors:
- `η_μ(t) = F_μ(x_i, t)` (identity link: `μ_i(t) = η_μ_i(t)`)
- `η_σ(t) = log(σ_i(t))` (log link: `σ_i(t) = exp(η_σ_i(t))`)

### Negative Gradients (Gaussian)

Log-likelihood (per observation, per time point):
```
ℓ_i(t) = −log(σ_i(t)) − (Y_i(t) − μ_i(t))² / (2σ_i(t)²)
```

**Negative gradient w.r.t. μ (identity link):**
```
u_μ,i(t) = ∂ℓ_i(t)/∂μ_i(t) = (Y_i(t) − μ_i(t)) / σ_i(t)²
```
(For a starting scale `σ_i(t)² = 1`, this reduces to the ordinary residual.)

**Negative gradient w.r.t. η_σ = log(σ) (log link, chain rule):**
```
∂ℓ_i(t)/∂η_σ,i(t) = ∂ℓ/∂σ · ∂σ/∂η_σ
                   = [−1/σ_i(t) + (Y_i(t)−μ_i(t))²/σ_i(t)³] · σ_i(t)
                   = −1 + (Y_i(t)−μ_i(t))²/σ_i(t)²
```

### Cyclic Algorithm

**Initialization:**
```
F̂_μ,0(t) = Ȳ(t)                       // pointwise mean
η̂_σ,0(t) = 0  →  σ̂_0(t) = exp(0) = 1  // unit scale
```

**For** `m = 1, …, mstop`:

1. **μ-step:** Compute `U_μ` using current `μ̂`, `σ̂`. Run one boosting step on `U_μ` using Algorithm 1 (but only one iteration, selecting the best base-learner for μ). Update `F̂_μ += ν · h_μ,j*`.

2. **σ-step:** Compute `U_σ` using updated `μ̂`, current `σ̂`. Run one boosting step on `U_σ`. Update `η̂_σ += ν · h_σ,j*`. Then `σ̂_i(t) = exp(η̂_σ,i(t))` pointwise.

3. Track log-likelihood across iterations for GCV/AIC path.

**Numerical guard for σ:** Clip `σ̂_i(t) ≥ NUMERICAL_EPS` (use existing `NUMERICAL_EPS = 1e-10` from `helpers.rs`) before computing any ratio to prevent division-by-zero. Log `FdarError::ComputationFailed` if clipping count exceeds a threshold.

**Output (`GamlssResult`):**
```rust
pub struct GamlssResult {
    pub mu_fitted: FdMatrix,          // n × m_t fitted μ(t)
    pub sigma_fitted: FdMatrix,       // n × m_t fitted σ(t)
    pub mu_intercept: Vec<f64>,       // F_μ,0(t)
    pub sigma_intercept: Vec<f64>,    // exp(η_σ,0(t))
    pub mu_beta: FdMatrix,            // p × m_t accumulated μ coefficients
    pub sigma_beta: FdMatrix,         // p × m_t accumulated log-σ coefficients
    pub log_likelihood: f64,          // final Gaussian log-likelihood
    pub ll_path: Vec<f64>,            // log-likelihood per iteration
    pub mstop: usize,
    pub nu: f64,
}
```

---

## Algorithm 4: Bayesian FOSR via Conjugate Gibbs (REG-06-04)

[ASSUMED] — Standard conjugate Normal / Inverse-Gamma Gibbs for linear regression; adapted for functional coefficient reconstruction from FPCA. Cross-referenced with Goldsmith et al. (2015) and Jiang et al. (2025) (arXiv:2505.05633).

### Model in FPC Score Space

```
Y_i(t) ≈ Σ_k score_{ik} · γ_k(t) + ε_i(t)
       = scores_i' · γ(t) + ε_i(t)
```

where `scores_i ∈ R^K` are FPC scores from `fdata_to_pc_1d`, and `γ(t) = rotation' · β(t)` — the FPC-space regression coefficient. In practice, work with the scalar response model per time point OR jointly over the compressed FPC representation.

**Preferred approach for fdars (no new deps, MSRV 1.81):** Operate in FPC space, fitting a multivariate scalar regression model: `score matrix S ∈ R^{n × K}`, response `Y ∈ R^{n × m_t}` (functional response), or optionally work one time-point at a time.

**Simpler conjugate form per response time point `t`:**

Model: `y(t) = S · γ(t) + ε(t)`,  `y(t) ∈ R^n`,  `S ∈ R^{n × K}`

### Conjugate Priors

```
γ(t) | σ²(t) ~ N(0, τ² · I_K)          // weakly informative; τ² = 100
σ²(t)         ~ IG(a₀, b₀)             // a₀ = b₀ = 0.001
```

### Full Conditional Distributions (one time point `t`)

**For γ(t):**
```
Σ_post(t) = (S'S/σ²(t) + I_K/τ²)⁻¹                   // K × K matrix
μ_post(t) = Σ_post(t) · S'y(t) / σ²(t)               // K-vector
γ(t) | rest ~ N(μ_post(t), Σ_post(t))                  // multivariate Normal draw
```

Draw via Cholesky: `γ(t) = μ_post(t) + L · z` where `L = cholesky_factor(Σ_post(t))` and `z ~ N(0, I_K)`.

**For σ²(t):**
```
RSS(t) = ||y(t) − S · γ(t)||²
σ²(t) | rest ~ IG(a₀ + n/2,  b₀ + RSS(t)/2)
```

Draw IG(α, β) as `1 / Gamma(α, 1/β)`. Rust's `rand_distr::Gamma` is available (already in dependency tree via existing crate usage — no new dep).

### Gibbs Loop

```
Initialize: γ̂(t) = 0 for all t;  σ̂²(t) = 1.0
For iteration i = 0 .. (burn_in + n_iter * thin):
  For each time point t = 0..m_t:
    Draw γ(t) from N(μ_post(t), Σ_post(t))
    Draw σ²(t) from IG(a₀ + n/2, b₀ + RSS(t)/2)
  If i >= burn_in and (i - burn_in) % thin == 0:
    Store reconstructed β_draw(t) = rotation · γ(t)   (m × K) · (K × m_t) → m_t-vector per predictor
```

**RNG seeding:** `let mut rng = StdRng::seed_from_u64(seed);` — single RNG for the chain; deterministic across runs with the same seed.

### Credible Band Construction

From `Q` thinned posterior draws of `β(t)` (stored as `Q × m_t` matrices per predictor):

```
β̂_post_mean(t) = mean over draws                    // posterior mean
β̂_lower(t) = quantile(draws[:,t], 0.025)           // 2.5th percentile
β̂_upper(t) = quantile(draws[:,t], 0.975)           // 97.5th percentile
```

All quantile operations are simple sorts over `Q` values per time point.

**Output (`BayesianFosrResult`):**
```rust
pub struct BayesianFosrResult {
    pub beta_mean: FdMatrix,           // posterior mean β(t), p × m_t
    pub beta_lower: FdMatrix,          // pointwise 2.5% credible band, p × m_t
    pub beta_upper: FdMatrix,          // pointwise 97.5% credible band, p × m_t
    pub fitted: FdMatrix,              // posterior-mean fitted values, n × m_t
    pub residuals: FdMatrix,           // posterior-mean residuals, n × m_t
    pub sigma2_mean: Vec<f64>,         // posterior mean σ²(t), length m_t
    pub n_iter: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub ncomp: usize,                  // FPC components used
}
```

**Divergence from R baseline:** `refund`'s Bayesian FOSR operates in spline basis space with random effects for smooth functional coefficients. The fdars implementation uses FPCA score compression (reusing `fdata_to_pc_1d`) for simplicity and zero new dependencies. Document in rustdoc.

---

## Algorithm 5: Stability Selection (REG-06-05)

[ASSUMED] — Meinshausen–Bühlmann (2010) + Hofner et al. (boosting adaptation), confirmed via stabs R package documentation.

### Algorithm

```
Input: data Y (n × m_t), predictors X (n × p), BoostingConfig, B resamples, pi_thr threshold
Initialize: select_count[j] = 0  for j = 1..p

For b = 0..B:
  1. Subsample floor(n/2) indices without replacement (seeded: StdRng::seed_from_u64(seed.wrapping_add(b as u64)))
  2. Run boost_fosr on subsample until mstop
  3. Mark base-learner j as selected if j appeared in selected_learners[0..mstop]
  4. select_count[j] += 1 if selected

Compute selection frequencies: pi_hat[j] = select_count[j] as f64 / B as f64
Stable set: stable = {j : pi_hat[j] >= pi_thr}

PFER bound (informational): E[V] <= q² / ((2*pi_thr − 1) * p)
  where q = mean per-subsample selection count = sum(select_count) / B
```

**Parallelism:** The B resample loop is a prime candidate for `iter_maybe_parallel!(0..B)` — each subsample is independent.

**Output (`StabilityResult`):**
```rust
pub struct StabilityResult {
    pub selection_freq: Vec<f64>,      // pi_hat[j] for j = 0..p
    pub stable_set: Vec<usize>,        // indices j with pi_hat[j] >= pi_thr
    pub pi_thr: f64,                   // threshold used (default 0.9)
    pub pfer_bound: f64,               // E[V] upper bound
    pub n_resamples: usize,            // B
}
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Penalized B-spline normal equations | Custom spline solver | `bspline_penalty_matrix` + `cholesky_solve` | Already handles roughness penalty, Cholesky stability |
| FPCA / score compression | Custom SVD+centering | `fdata_to_pc_1d` | Correct weighted SVD with Simpson's weights; tested |
| Integration weights | Trapezoidal or ad-hoc quadrature | `simpsons_weights` | Already handles uniform/non-uniform grids correctly |
| Random number generation | `rand::random()` | `StdRng::seed_from_u64` | Guarantees reproducibility; established pattern |
| Parallel resample loops | Mutex + thread::spawn | `iter_maybe_parallel!(0..B)` | Feature-gates rayon/WASM compatibility automatically |
| Normal variate draws for Gibbs | Box–Muller from scratch | `rand_distr::Normal` (already in dep tree) | Correct, tested; already transitively available |
| Gamma draws for IG | Inverse CDF | `rand_distr::Gamma` (already in dep tree) | Correct; IG(α,β) = 1/Gamma(α, 1/β) |

**Key insight:** Every numerical primitive this phase needs already exists in the codebase or in transitively-available dependencies. The only new code is the algorithm logic (loops, selection rules, Gibbs update equations) — not any new numerical machinery.

---

## Common Pitfalls

### Pitfall 1: Ill-Conditioned Penalized Normal Equations
**What goes wrong:** `(Φ'Φ + λR)` becomes singular when `lambda` is too small, the basis has near-collinear columns, or `nbasis >> n`.
**Why it happens:** B-spline basis matrices are nearly rank-deficient for large `nbasis`; small `lambda` provides insufficient ridge-like regularization.
**How to avoid:** Use the ridge jitter already applied in `smooth_basis.rs` (`1e-10 * I`). Apply it to the base-learner normal equations: `A = Φ'Φ + λ·R + 1e-10·I`. Validate `lambda > 0` in `BoostingConfig`. Return `FdarError::ComputationFailed` from `cholesky_factor` on non-positive diagonal.
**Warning signs:** `cholesky_factor` returns `Err`; fitted values are NaN.

### Pitfall 2: σ Collapse to Zero in GAMLSS
**What goes wrong:** The σ-boosting step overshoots in log-scale, driving `exp(η_σ) → 0` or `exp(η_σ) → ∞`, producing NaN in the μ-gradient `(Y−μ)/σ²`.
**Why it happens:** Early iterations of σ-boosting with large residuals can push `η_σ` far from zero.
**How to avoid:** Clip `σ̂_i(t) = σ̂_i(t).max(NUMERICAL_EPS)` after every σ update. Apply the same clip to `σ̂²` in the μ-gradient denominator. Log a warning (but do not fail) when clipping activates.
**Warning signs:** `u_μ` contains Inf or NaN; log-likelihood becomes −∞.

### Pitfall 3: Gibbs Non-Convergence / Slow Mixing
**What goes wrong:** Posterior draws of `γ(t)` wander slowly if `τ²` is very large (vague prior) or very small (dogmatic prior); credible bands become too wide or too narrow.
**Why it happens:** Conjugate Gibbs mixing rate is determined by the signal-to-noise ratio `||S||² / (n·σ²)`; flat priors (`τ² → ∞`) give poor conditioning of `Σ_post`.
**How to avoid:** Default to `τ² = 100` (large but not infinite) and `IG(0.001, 0.001)` for `σ²`. Let the user override via `BayesianConfig`. Default `burn_in = 1000`, `thin = 5`, `n_iter = 2000` gives 2000 retained draws.
**Warning signs:** Posterior mean of `σ²(t)` is implausibly large (> 1000 × sample variance) — check `b₀` and `a₀`.

### Pitfall 4: Base-Learner Selection Bias from Unequal df
**What goes wrong:** If different base-learners have different degrees of freedom (different `nbasis` or different `lambda`), the selection rule (minimum residual SS) favours more flexible learners regardless of signal.
**Why it happens:** A learner with more df can fit arbitrary noise better, producing lower RSS even when no signal exists.
**How to avoid:** Standardize all base-learners to the same effective degrees of freedom (`edf = tr(H_j)` where `H_j = Φ_j(Φ_j'Φ_j + λR)⁻¹Φ_j'`), or at minimum use the same `nbasis` and `lambda` for all. Document this in `BoostingConfig`'s rustdoc.
**Warning signs:** One base-learner is selected at nearly every iteration regardless of data; `selected_learners` has little variety.

### Pitfall 5: Column-Major Access Patterns in Residual Updates
**What goes wrong:** Iterating over time points in the inner loop with row-access to `FdMatrix` triggers non-contiguous memory reads, causing cache-miss overhead in the boosting loop (inner loop is `m_t` long).
**Why it happens:** `FdMatrix` is column-major; `data[(i,t)]` for varying `i` at fixed `t` is contiguous (column access), but `data[(i,t)]` for varying `t` at fixed `i` (row access) is strided.
**How to avoid:** Structure the residual update to iterate over columns (time points) in the outer loop and observations in the inner loop, or use `data.column(t)` for contiguous slice access. This matches `FdMatrix`'s zero-copy column access pattern.
**Warning signs:** Profiling shows cache-miss hot-spot in the residual computation loop.

### Pitfall 6: RNG State Sharing Across Parallel Resamples
**What goes wrong:** Using a shared `Rng` across parallel resample iterations produces a data race or incorrect seeding.
**Why it happens:** `iter_maybe_parallel!` with rayon shares no state — each closure must own its own RNG.
**How to avoid:** Inside each resample `b`, create `let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));` locally. This is the established pattern (`src/scalar_on_function/bootstrap.rs:89`).
**Warning signs:** Results are non-deterministic across runs with the same seed when `parallel` feature is enabled.

---

## Code Examples

### Base-Learner Fit (Core of Algorithm 1)

```rust
// Source: derived from existing smooth_basis.rs + linalg.rs patterns [ASSUMED]
// Build B-spline design matrix for scalar predictor values x_vals (length n)
// and compute penalized normal equation system once per base-learner.

let phi = build_bspline_design(x_vals, nbasis, order);  // n × K
let r = bspline_penalty_matrix(x_argvals, nbasis, order, lfd_order);  // K × K (col-major)

// Build X'X + λR (K × K, row-major for cholesky_factor)
let xtx = compute_xtx_from_slice(&phi, n, k);           // K × K
let mut a = xtx.clone();
for i in 0..k {
    for jj in 0..k {
        a[i * k + jj] += lambda * r[jj * k + i];        // r is col-major; a is row-major
    }
    a[i * k + i] += 1e-10;                              // ridge jitter
}
let l = cholesky_factor(&a, k)?;

// Solve for each time point t (the amortised loop)
for t in 0..m_t {
    let rhs: Vec<f64> = (0..k).map(|kk| {
        (0..n).map(|i| phi[i * k + kk] * u[(i, t)]).sum()  // Φ'u[:,t]
    }).collect();
    let c_t = cholesky_forward_back(&l, &rhs, k);
    // Store c_t → accumulated coefficient for predictor j at time t
}
```

### Gaussian Negative Gradient for GAMLSS

```rust
// Source: gamboostLSS families documentation [ASSUMED]
// Compute working residuals for mu-step (identity link)
fn mu_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> FdMatrix {
    let (n, m) = y.shape();
    let mut u = FdMatrix::zeros(n, m);
    for t in 0..m {
        for i in 0..n {
            let s2 = sigma[(i, t)].powi(2).max(NUMERICAL_EPS);
            u[(i, t)] = (y[(i, t)] - mu[(i, t)]) / s2;
        }
    }
    u
}

// Compute working residuals for sigma-step (log link)
fn sigma_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> FdMatrix {
    let (n, m) = y.shape();
    let mut u = FdMatrix::zeros(n, m);
    for t in 0..m {
        for i in 0..n {
            let s = sigma[(i, t)].max(NUMERICAL_EPS);
            let r2 = (y[(i, t)] - mu[(i, t)]).powi(2);
            u[(i, t)] = -1.0 + r2 / (s * s);
        }
    }
    u
}
```

### Gibbs Draw for Bayesian FOSR (single time point)

```rust
// Source: standard Normal-IG conjugate Gibbs [ASSUMED]
// S: n × K score matrix (flat row-major), y_t: response at time t (length n)
fn gibbs_draw_gamma(
    s: &[f64], y_t: &[f64], sigma2_t: f64, tau2: f64, n: usize, k: usize,
    rng: &mut StdRng,
) -> Result<Vec<f64>, FdarError> {
    // Sigma_post^{-1} = S'S / sigma2 + I / tau2
    let mut prec = compute_sts(s, n, k);          // S'S, K×K row-major
    for i in 0..k {
        prec[i * k + i] += sigma2_t.recip() * (1.0 / tau2); // wrong — fix below
    }
    // Correct: prec[j,j] += 1/tau2 (not /sigma2 — that's in S'S term)
    // prec = S'S / sigma2 + I/tau2
    for j in 0..k * k { prec[j] /= sigma2_t; }
    for i in 0..k { prec[i * k + i] += 1.0 / tau2; }

    let sigma_post = invert_kxk(&prec, k)?;    // K×K via Cholesky
    let sts_y: Vec<f64> = (0..k).map(|kk| {
        (0..n).map(|i| s[i * k + kk] * y_t[i]).sum::<f64>() / sigma2_t
    }).collect();
    let mu_post = mat_vec_mul(&sigma_post, &sts_y, k);  // K-vector

    // Draw z ~ N(0,I_K), then gamma = mu_post + L*z
    let l = cholesky_factor(&sigma_post, k)?;
    let z: Vec<f64> = (0..k).map(|_| rng.sample(StandardNormal)).collect();
    let mut gamma: Vec<f64> = mu_post.clone();
    for i in 0..k {
        for jj in 0..=i { gamma[i] += l[i * k + jj] * z[jj]; }
    }
    Ok(gamma)
}
```

---

## Integration Points

[VERIFIED: src/lib.rs:64-135] — Module list structure; new module goes after `pub mod regression;` (~line 103).

[VERIFIED: src/lib.rs:137-138] — Re-export pattern: `pub use boosting_regression::{...};`

[VERIFIED: src/prelude.rs:14-19] — Where to add key result types.

Exact lines to modify (three files):
1. `src/lib.rs` — add `pub mod boosting_regression;` at module list.
2. `src/lib.rs` — add `pub use boosting_regression::{BoostFosrResult, BoostFofrResult, GamlssResult, BayesianFosrResult, StabilityResult, BoostingConfig, BayesianConfig, StabilityConfig, boost_fosr, boost_fofr, gamlss_fosr, bayesian_fosr, stability_selection};` in re-export block.
3. `src/prelude.rs` — add `pub use crate::boosting_regression::{BoostFosrResult, BayesianFosrResult};`.

---

## State of the Art

| Old Approach | Current Approach | Fdars v1 Scope |
|--------------|------------------|----------------|
| Plain OLS boosting | Penalized B-spline base-learners (FDboost `bbs`) | Yes — via existing `bspline_penalty_matrix` |
| Noncyclic GAMLSS boosting (better var. selection) | Cyclic coordinate-wise gamboostLSS | Deferred — cyclic is v1 per CONTEXT.md |
| VB for Bayesian FOSR | Conjugate Gibbs | Gibbs is v1; VB deferred |
| Simultaneous credible bands | Pointwise credible bands | Pointwise only in v1 |
| Adaptive ν (line search) | Fixed ν = 0.1 | Fixed per CONTEXT.md |
| CV-based mstop selection | Fixed mstop + GCV/AIC path tracking | Path tracking only; CV-mstop deferred |

**Deprecated/outdated:**
- Using plain OLS base-learners without a roughness penalty: biases base-learner selection toward high-df learners; always use the penalized form.

---

## Validation Architecture

`workflow.nyquist_validation = true` in `.planning/config.json` — Validation Architecture section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[cfg(test)]` / `cargo test`) |
| Config file | None — inline `#[cfg(test)] mod tests { ... }` in each submodule |
| Quick run command | `cargo test -p fdars-core --features linalg boosting_regression` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REG-06-01 | `boost_fosr` reduces training RSS monotonically | unit | `cargo test -p fdars-core --features linalg boosting_regression::boost_fosr::tests` | ❌ Wave 0 |
| REG-06-01 | `boost_fosr` recovers known β(t) on synthetic data | unit | same | ❌ Wave 0 |
| REG-06-01 | `boost_fosr` returns `FdarError` on dimension mismatch | unit | same | ❌ Wave 0 |
| REG-06-02 | `boost_fofr` fitted values shape = (n, m_y) | unit | `cargo test -p fdars-core --features linalg boosting_regression::boost_fofr::tests` | ❌ Wave 0 |
| REG-06-02 | `boost_fofr` residuals decrease over iterations | unit | same | ❌ Wave 0 |
| REG-06-03 | `gamlss_fosr` σ̂(t) > 0 everywhere | unit | `cargo test -p fdars-core --features linalg boosting_regression::gamlss::tests` | ❌ Wave 0 |
| REG-06-03 | `gamlss_fosr` log-likelihood increases (or is non-decreasing) over iterations | unit | same | ❌ Wave 0 |
| REG-06-04 | Gibbs posterior mean ≈ penalized OLS estimate (within 2σ) | unit | `cargo test -p fdars-core --features linalg boosting_regression::bayesian::tests` | ❌ Wave 0 |
| REG-06-04 | Credible bands contain the true β(t) at ≥ 90% of grid points (synthetic truth known) | unit | same | ❌ Wave 0 |
| REG-06-04 | Results are identical for same seed across two runs | unit | same | ❌ Wave 0 |
| REG-06-05 | `stability_selection` frequencies ∈ [0.0, 1.0] | unit | `cargo test -p fdars-core --features linalg boosting_regression::stability::tests` | ❌ Wave 0 |
| REG-06-05 | Stable set is a subset of all base-learners | unit | same | ❌ Wave 0 |
| REG-06-05 | PFER bound ≥ 0 | unit | same | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg boosting_regression`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + full test suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `src/boosting_regression/mod.rs` — module barrel + `BoostingConfig`, `BayesianConfig`, `StabilityConfig` structs (no logic, just types + tests for config validation)
- [ ] `src/boosting_regression/boost_fosr.rs` — skeleton with `BoostFosrResult` and stub `boost_fosr()` returning `Ok(BoostFosrResult { ... })` with correct types
- [ ] `src/boosting_regression/boost_fofr.rs` — skeleton
- [ ] `src/boosting_regression/gamlss.rs` — skeleton
- [ ] `src/boosting_regression/bayesian.rs` — skeleton
- [ ] `src/boosting_regression/stability.rs` — skeleton
- [ ] Register `pub mod boosting_regression;` in `src/lib.rs` + crate-root re-exports
- [ ] Add key result types to `src/prelude.rs`

---

## Security Domain

`security_enforcement = true` (config.json line 47). ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | Pure numeric library — no user sessions |
| V3 Session Management | No | Stateless function calls |
| V4 Access Control | No | No privilege model |
| V5 Input Validation | Yes | Dimension checks via `FdarError::InvalidDimension`; parameter range checks (`mstop > 0`, `nu ∈ (0,1]`, `nbasis >= 2`, `n_iter > 0`, `burn_in < n_iter`, `B >= 1`, `pi_thr ∈ (0.5, 1.0]`) |
| V6 Cryptography | No | RNG is for statistical reproducibility, not security |

### Known Threat Patterns for this Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in `seed.wrapping_add(b as u64)` | Tampering | `wrapping_add` is the correct Rust idiom — no panic on overflow |
| NaN propagation from σ → 0 in GAMLSS | Tampering | Clip `σ >= NUMERICAL_EPS` before division; gate in `FdarError::ComputationFailed` |
| Stack overflow from deep Cholesky on large K | DoS | Validate `nbasis <= n` in `BoostingConfig` at construction time |
| `burn_in >= n_iter` giving zero retained draws | DoS | Validate `burn_in < n_iter` in `BayesianConfig` constructor |

---

## Open Questions

1. **Base-learner design matrix construction for scalar predictors**
   - What we know: `bspline_penalty_matrix` is called with `argvals` (the grid points), not with the actual predictor values. For a base-learner on scalar predictor `x_j`, we need to evaluate the B-spline basis at the `n` predictor values `{x_{1j}, ..., x_{nj}}`, which are not necessarily a uniform grid.
   - What's unclear: Is `src/basis.rs::bspline_basis(x_vals, nknots, order)` the right entry point for evaluating basis at arbitrary values, or is there a wrapper? The plan should verify the signature of `bspline_basis` before building the design matrix.
   - Recommendation: The planner should `Read src/basis.rs` and document the exact call needed; implement a private `build_bspline_design_at(x_vals, nbasis, order) -> Vec<f64>` helper as part of Wave 1.

2. **IG(α, β) sampling with `rand_distr::Gamma`**
   - What we know: `IG(α, β) = 1/Gamma(α, 1/β)` using rate parameterization. `rand_distr::Gamma` is transitively available.
   - What's unclear: The exact parameterization (shape + scale vs. shape + rate) used by `rand_distr::Gamma` — must confirm to avoid off-by-one on the β parameter.
   - Recommendation: The plan Wave 4 (Bayesian task) must verify `rand_distr::Gamma::new(alpha, scale)` vs. `Gamma::new(alpha, rate)` before writing the IG draw.

3. **Pointwise vs. joint Gibbs for functional response**
   - What we know: Running Gibbs pointwise (one time point at a time) is simple but ignores functional smoothness of `γ(t)`.
   - What's unclear: Whether ignoring cross-time correlation in the Gibbs sampler leads to noticeably poor credible bands for smooth `β(t)`.
   - Recommendation: v1 uses pointwise Gibbs (CONTEXT.md: no simultaneous bands). Accept the conservative credible bands; document the limitation in rustdoc.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Negative gradient for μ in Gaussian GAMLSS is `(Y−μ)/σ²`; for log-link σ it is `−1 + (Y−μ)²/σ²` | Algorithm 3 | σ-model fails to converge; log-likelihood path is wrong |
| A2 | Conjugate Gibbs full-conditional for γ is multivariate Normal with precision `S'S/σ² + I/τ²` | Algorithm 4 | Posterior mean is biased; credible bands are wrong |
| A3 | `rand_distr::Gamma` is available transitively in the fdars dependency tree | Algorithm 4 | Compile error; need to find alternative IG draw (can use Box–Muller trick) |
| A4 | `bspline_basis(x_vals, nknots, order)` in `src/basis.rs` evaluates the B-spline at arbitrary predictor values (not just a uniform grid) | Algorithm 1 | Base-learner design matrix build fails or is incorrect |
| A5 | `iter_maybe_parallel!(0..B)` in stability selection compiles correctly with `move` closure capturing `rng` seed independently per iteration | Algorithm 5 | Compile error — may need to convert to `(0..B).map(|b| { ... }).collect()` |
| A6 | PFER bound formula is `E[V] ≤ q² / ((2π_thr − 1) · p)` | Algorithm 5 | Incorrect PFER bound reported (informational only — no effect on the selection itself) |

**If this table is empty:** All claims were verified — but this table has 6 entries. The planner must verify A3 and A4 before writing Wave 1/Wave 4 plans.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All | ✓ | 1.97.0 | — |
| `cargo test --features linalg` | All algorithms (Cholesky) | ✓ | via Cargo | — |
| `cargo test --features linalg,parallel` | Stability selection parallel path | ✓ | via Cargo | Sequential fallback |
| `rand` / `rand_distr` (transitive) | Bayesian Gibbs | Likely ✓ (transitive dep) | ~0.8 | Box–Muller for Normal; rejection sampling for IG |
| `FdMatrix::zeros(n, m)` constructor | All algorithms | Must verify | — | Use `FdMatrix::from_column_major(vec![0.0; n*m], n, m)` |

---

## Project Constraints (from CLAUDE.md)

- No new crate dependency — all machinery must come from in-crate helpers.
- MSRV 1.81; `linalg` feature requires 1.84 (Cholesky is behind `linalg`).
- Column-major `FdMatrix` throughout — never row-major intermediates in public API.
- All public functions return `Result<T, FdarError>` — no panics on user input.
- Inline `#[cfg(test)] mod tests { ... }` per submodule file.
- Crate-root `pub use boosting_regression::{...}` re-exports for all public types.
- `src/prelude.rs` gets key result types.
- `#[must_use]` on all expensive computation functions.
- `#[non_exhaustive]` on public result structs.
- `#[derive(Debug, Clone, PartialEq)]` on all public types.
- Full CI gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.
- No changes to existing public signatures — additive only.
- Clippy allows at crate level (already in `lib.rs`): `needless_range_loop`, `too_many_arguments`, `type_complexity` — safe to use similar patterns.
- Document divergences from R baseline (FDboost 1.1-4 + refund) in rustdoc per-function.

---

## Sources

### Primary (HIGH confidence — in-session file reads)
- `src/smooth_basis.rs:82-116` — `bspline_penalty_matrix` signature and penalty integration
- `src/linalg.rs:85-134` — `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve`
- `src/regression.rs:25-38, 287-292` — `FpcaResult` fields, `fdata_to_pc_1d` signature
- `src/helpers.rs:57` — `simpsons_weights` signature
- `src/function_on_scalar.rs:29-48` — `FosrResult` field template
- `src/parallel.rs:42-155` — macro definitions for `iter_maybe_parallel!` etc.
- `src/scalar_on_function/bootstrap.rs:89` — RNG seeding pattern
- `src/lib.rs:64-135` — module list and re-export structure
- `src/prelude.rs` — prelude re-export pattern

### Secondary (MEDIUM confidence — published literature / official R package docs)
- FDboost CRAN documentation (rdrr.io) — component-wise boosting algorithm structure, `bbs()` base-learner description, ν parameter role
- gamboostLSS family documentation (rdrr.io) — Gaussian negative-gradient formulas, cyclic vs. noncyclic algorithm
- Meinshausen & Bühlmann stability selection (metricgate.com summary) — PFER bound formula, π_thr threshold
- stabs R package (CRAN) — B = 100 default, floor(n/2) subsampling, selection frequency definition
- Jiang et al. (2025), arXiv:2505.05633 — Bayesian FoSR model specification, IG(0.001, 0.001) priors, credible band construction from posterior samples

### Tertiary (LOW confidence — training knowledge, not verified this session)
- Algorithm 1 (boosting core) — derives from mboost/FDboost documentation + standard gradient boosting texts; the exact negative-gradient expression for L2 loss (ordinary residual) is standard and low-risk.
- Algorithm 4 (Gibbs conjugate posteriors) — standard Normal-IG conjugate analysis; cross-checked with multiple sources.

---

## Metadata

**Confidence breakdown:**
- Standard stack (in-crate tools): HIGH — verified by reading source files this session
- Algorithm recipes: MEDIUM — derived from published R package documentation and secondary sources; mathematical derivations are standard textbook
- Numerical pitfalls: MEDIUM — drawn from known FDA/boosting literature and existing codebase patterns
- Assumptions A1–A6: LOW — flag for planner verification before writing Wave 1/4 plans

**Research date:** 2026-08-23
**Valid until:** 2026-11-23 (stable algorithms; 90-day validity)
