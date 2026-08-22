# Phase 40: Fréchet / Object-Data Regression — Research

**Researched:** 2026-08-22
**Domain:** Metric-space (object-data) statistics — Fréchet mean/variance, global/local Fréchet regression, 1D 2-Wasserstein distance, Fréchet ANOVA; Rust implementation in `fdars-core/src/frechet/`
**Confidence:** MEDIUM (core APIs verified from codebase reads; algorithm formulas cited from R package source inspection and published papers)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- `MetricSpace` trait: `distance(&a, &b)` + `weighted_frechet_mean(objects, weights)`; ONE concrete `WassersteinDensitySpace` backend (densities on a shared grid as FdMatrix rows).
- Density-space Fréchet mean delegates to `density_fda::wasserstein_barycenter` (already weighted).
- Fréchet variance = mean squared distance to the Fréchet mean.
- 1D 2-Wasserstein distance = L2 of quantile functions: W₂ = (∫₀¹ (Q_F(t)−Q_G(t))² dt)^{1/2}, reusing density_fda's density→quantile machinery.
- Global Fréchet regression = Petersen–Müller global linear weights wᵢ(x) = 1 + (x−x̄)ᵀΣ⁻¹(xᵢ−x̄) → weighted Fréchet mean; local Fréchet regression = local-linear Gaussian-kernel weights with bandwidth param; Euclidean predictors as FdMatrix (n×p); density-response variant predicts the conditional density (weighted barycenter) in 2-Wasserstein space.
- Fréchet ANOVA = Dubey–Müller statistic; seeded permutation p-value (999 default) + asymptotic statistic.
- Layout: frechet/ dir with submodules + mod.rs re-exports.

### Claude's Discretion

- Exact submodule filenames, trait method signatures, result-struct field names, kernel/bandwidth defaults, documented tolerance constants, and whether distance is computed via stored quantile functions or recomputed per call are at Claude's discretion, guided by the `frechet`/`fdadensity` references and codebase style.

### Deferred Ideas (OUT OF SCOPE)

- FRE-02: additional object-data Fréchet spaces (covariance/correlation matrices, spherical, network/graph-Laplacian, point-process).
- Non-density metric-space backends beyond 1D-Wasserstein.
- Plotting/rendering of Fréchet fits or regression surfaces (numeric outputs only).
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FRE-01-01 | `MetricSpace` trait + `WassersteinDensitySpace` backend | §Trait Design; §Wasserstein Backend |
| FRE-01-02 | Fréchet mean of a sample via weighted-barycenter solver | §Fréchet Mean; `wasserstein_barycenter` signature verified |
| FRE-01-03 | Fréchet variance = mean squared distance to the Fréchet mean | §Fréchet Variance formula |
| FRE-01-04 | Global Fréchet regression (Petersen–Müller weights) | §Global Regression — the KEY RISK section |
| FRE-01-05 | Local (local-linear / kernel-weighted) Fréchet regression | §Local Regression kernel formula |
| FRE-01-06 | 1D 2-Wasserstein distance (quantile L2) | §W₂ Distance; density_fda machinery verified |
| FRE-01-07 | Density-response Fréchet regression (conditional density) | §Density-Response Prediction |
| FRE-01-08 | Fréchet ANOVA group-difference test (Dubey–Müller) | §Fréchet ANOVA formulas |
</phase_requirements>

---

## Summary

Phase 40 delivers a new `fdars-core/src/frechet/` module implementing metric-space (object-data) statistics and regression. The primary abstraction is a `MetricSpace` trait with two methods; the only concrete backend this phase is `WassersteinDensitySpace` for 1D density responses in 2-Wasserstein geometry, reusing existing `density_fda.rs` machinery.

The most important implementation decision — confirmed by reading the R package source — is that **global Fréchet regression does NOT call `wasserstein_barycenter` for the density prediction step**. Because the Petersen–Müller global weights sᵢ(x) = 1 + (Xᵢ − X̄)ᵀΣ̂⁻¹(x − X̄) can be negative (for observations whose predictor value lies "opposite" to x relative to X̄), the R reference implementation (`GloWassReg`, `LocWassReg`) computes a **signed weighted average of quantile functions directly**: `Q̄(t) = (1/n) Σᵢ sᵢ(x) Qᵢ(t)` (i.e. `colMeans(qin * s)`), then converts the resulting function back to a density by enforcing monotonicity. In the Rust implementation, since we have no osqp QP solver (no new crate dependency allowed), we apply a post-hoc monotone-enforcement step: sort/isotonic clamp the signed quantile average, then invert via the existing `inverse_lqd` machinery (or the quantile→density inversion already in `wasserstein_barycenter`'s back-half). This is the one algorithm-level deviation from the CONTEXT.md description that says "weighted Fréchet mean"; in the density space with signed weights, the computation is a signed quantile average, not a `wasserstein_barycenter` call. The MetricSpace trait's `weighted_frechet_mean` method can still be defined to accept signed weights, but the `WassersteinDensitySpace` implementation must handle them via the quantile-average path, not by delegating to `wasserstein_barycenter` directly.

**Primary recommendation:** Implement the signed-weight quantile average for density regression as a private helper `quantile_average_signed(density_matrix, argvals, weights) -> Result<Vec<f64>, FdarError>` in `frechet/space.rs`; delegate non-negative-weight calls (plain Fréchet mean) to `wasserstein_barycenter`; and document the deviation from the Euclidean weighted-mean analogy in rustdoc.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| MetricSpace trait definition | `frechet/space.rs` | — | Single trait, one backend; keeps trait and impl together |
| WassersteinDensitySpace backend | `frechet/space.rs` | `density_fda.rs` (reused) | Backend delegates mean to density_fda, dist uses quantile machinery |
| Fréchet mean + variance | `frechet/mean.rs` | `frechet/space.rs` | Generic over MetricSpace; calls trait methods |
| 1D W₂ distance | `frechet/space.rs` | `density_fda.rs` | Private helper; no new public API needed beyond trait method |
| Global Fréchet regression | `frechet/regression.rs` | `linalg.rs` (Σ⁻¹), `density_fda.rs` | Weight formula + signed quantile average |
| Local Fréchet regression | `frechet/regression.rs` | `helpers.rs` (gaussian_kernel) | Kernel weights + local-linear correction + signed quantile average |
| Fréchet ANOVA | `frechet/anova.rs` | `frechet/mean.rs`, `rand` | Dubey-Müller Tn statistic + permutation loop |
| Crate-root re-exports | `src/lib.rs` | `frechet/mod.rs` | Standard pattern (mirrors `fts/mod.rs`) |

---

## Project Constraints (from CLAUDE.md)

- Column-major `FdMatrix`; all public fns `Result<T, FdarError>`
- Result structs: `derive(Debug, Clone, PartialEq)` + `#[cfg_attr(feature = "serde", ...)]` + `#[non_exhaustive]`
- `#[must_use]` on expensive computations (regression fits, ANOVA)
- Permutation: `StdRng::seed_from_u64(seed + k as u64)`, 999 default, per-thread seeded
- Feature-gated rayon: use `iter_maybe_parallel!` / `slice_maybe_parallel!` macros from `parallel.rs`
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for cargo build/test (pre-commit doctest linking)
- No new crate dependency (no osqp, no monotone-QP solver)
- Additive/non-breaking: zero changes to existing public signatures

---

## Standard Stack

### Core (all existing — no new crate dependency)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nalgebra | 0.33 | Σ̂⁻¹ for global weights (p×p matrix inverse) | Already in Cargo.toml; `elastic_changepoint.rs` shows Cholesky inverse pattern |
| rand / rand_distr | 0.8 / 0.4 | Per-thread seeded RNG for ANOVA permutation | Established pattern; `StdRng::seed_from_u64` |
| `density_fda.rs` | in-crate | Fréchet mean (wasserstein_barycenter), quantile machinery | DENS-01 deliverable; reuse-first mandate |
| `linalg.rs` | in-crate | Cholesky factor + solve for Σ̂⁻¹(x − X̄) | `cholesky_factor` / `cholesky_forward_back` already public(crate) |
| `helpers.rs` | in-crate | `trapz`, `cumulative_trapz`, `linear_interp`, `gaussian_kernel`, `NUMERICAL_EPS` | All confirmed present |
| `parallel.rs` | in-crate | `iter_maybe_parallel!` / `maybe_par_chunks_mut!` macros | Feature-gated rayon pattern |

**Installation:** No `cargo add` needed — all dependencies are in-crate or already in Cargo.toml.

### Package Legitimacy Audit

No new external crate dependencies. Section not applicable — milestone constraint forbids new deps.

---

## Algorithm Details (pinned formulas)

### 1. MetricSpace Trait

```rust
// Source: CONTEXT.md locked decisions + codebase style inference
pub trait MetricSpace: Send + Sync {
    /// Type alias for a single object (a density row as Vec<f64> or FdMatrix row reference).
    type Object;

    /// Distance between two objects.
    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError>;

    /// Weighted Fréchet mean.
    /// `weights` must be non-negative and sum to 1 (enforced at call site).
    /// For signed-weight callers (global regression), use the backend-specific
    /// signed-quantile-average helper directly.
    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError>;
}
```

The trait is `Send + Sync` to support `rayon`-parallel regression loops (consistent with `FpcPredictor` trait pattern). The associated type `Object` should be `Vec<f64>` for the density backend (one density row evaluated on the shared grid).

### 2. WassersteinDensitySpace Backend

```rust
// Object = Vec<f64> (density values on the shared grid)
pub struct WassersteinDensitySpace {
    pub argvals: Vec<f64>,  // shared strictly-increasing grid, length m
}
```

- `distance(a, b)` → compute W₂ (see §W₂ Distance below)
- `weighted_frechet_mean(objects, weights)` → assemble into FdMatrix, call `wasserstein_barycenter(&mat, &self.argvals, Some(weights))` [VERIFIED: density_fda.rs:407]

### 3. 1D 2-Wasserstein Distance (FRE-01-06)

**Formula:** W₂(F, G) = (∫₀¹ (Q_F(t) − Q_G(t))² dt)^{1/2}

**Reusable pattern** (extracted from `wasserstein_barycenter` inner loop, lines 476-496 of density_fda.rs):

```rust
// Private helper: density row → quantile function on t_grid
fn density_to_quantile(row: &[f64], argvals: &[f64], t_grid: &[f64]) -> Vec<f64> {
    let integral = trapz(row, argvals);
    let norm: Vec<f64> = row.iter().map(|&v| v / integral).collect();
    let cdf = cumulative_trapz(&norm, argvals);
    // Q_F(t) = argvals value at CDF level t
    t_grid.iter().map(|&t| linear_interp(&cdf, argvals, t)).collect()
}

// W₂ distance
fn w2_distance(a: &[f64], b: &[f64], argvals: &[f64], n_q: usize) -> Result<f64, FdarError> {
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();
    let qa = density_to_quantile(a, argvals, &t_grid);
    let qb = density_to_quantile(b, argvals, &t_grid);
    let sq_diff: Vec<f64> = qa.iter().zip(qb.iter()).map(|(&u, &v)| (u - v).powi(2)).collect();
    Ok(trapz(&sq_diff, &t_grid).sqrt())
}
```

This is a small private helper, **not** going through `lqd_transform` (which requires strictly positive densities and works in log-space). Instead it mirrors the CDF→quantile interpolation pattern already in `wasserstein_barycenter`. [VERIFIED: density_fda.rs:476-496]

**Decision:** Use the same `n_q` as `wasserstein_barycenter` defaults: `m.max(101)` where m = argvals.len(). [VERIFIED: density_fda.rs:469]

### 4. Fréchet Mean and Variance (FRE-01-02, FRE-01-03)

```rust
// Fréchet mean of a sample — delegates to trait
pub fn frechet_mean<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    weights: Option<&[f64]>,
) -> Result<S::Object, FdarError>

// Fréchet variance — mean squared distance to the Fréchet mean
// V = (1/n) Σᵢ d²(Yᵢ, μ̂)   (or weighted: Σᵢ wᵢ d²(Yᵢ, μ̂))
pub fn frechet_variance<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    mean: &S::Object,
    weights: Option<&[f64]>,
) -> Result<f64, FdarError>
```

[CITED: Dubey & Müller 2019, Biometrika 106(4):803-821; formula V̂ = n⁻¹Σᵢ d²(Yᵢ, μ̂)]

### 5. Global Fréchet Regression — KEY RISK (FRE-01-04, FRE-01-07)

**Weight formula** (Petersen & Müller 2019):
```
sᵢ(x) = 1 + (Xᵢ − X̄)ᵀ Σ̂⁻¹ (x − X̄)
```
where X̄ = sample mean of predictors (p-vector), Σ̂ = sample covariance of predictors (p×p).

[CITED: https://arxiv.org/html/2605.19519 — "sⱼ(x) = 1 + (Xⱼ−X̄)Σ̂⁻¹(x−X̄)"]

**Critical fact:** Weights sᵢ(x) can be **negative** for observations where (Xᵢ − X̄) points opposite to (x − X̄) in the Σ⁻¹ metric. The R reference (`GloWassReg.R`) computes:

```r
gx <- colMeans(qin * s)  # direct signed weighted average of quantile functions
```

This is a **signed quantile average**, NOT a call to `getWFmean` / `wasserstein_barycenter`. [CITED: https://github.com/functionaldata/tFrechet/blob/master/R/GloWassReg.R — source code read via WebFetch]

**Rust implementation for density response (FRE-01-07):**

```rust
// Step 1: compute predictor moments
let x_bar: Vec<f64> = ...;         // p-vector mean of rows of predictors
let sigma_hat: Vec<f64> = ...;     // p×p flat row-major sample covariance (divide by n-1)
// Ridge regularize: sigma_hat[j*p+j] += 1e-8 if needed (mirrors elastic_changepoint)
let chol = cholesky_factor(&sigma_hat, p)?;

// Step 2: for each training observation i, compute sᵢ(x_pred)
//   diff_i = Xᵢ − X̄   (length p)
//   diff_x = x_pred − X̄  (length p)
//   sᵢ = 1 + diff_i · (Σ̂⁻¹ diff_x)
//       = 1 + diff_i · cholesky_solve(sigma_hat, diff_x)
// Note: (Xᵢ−X̄)ᵀ Σ̂⁻¹ (x−X̄) = diff_i · v   where v = Σ̂⁻¹ diff_x

// Step 3: compute the signed quantile average Q̄(t) = (1/n) Σᵢ sᵢ(x) Qᵢ(t)
//   For each observation i: Qᵢ(t) via density_to_quantile(row_i, argvals, t_grid)
//   q_bar[j] = (1/n) Σᵢ s_i * Q_i(t_grid[j])

// Step 4: q_bar may be non-monotone due to signed weights — apply isotonic/sort fix
//   Simple fix: if q_bar is mostly monotone, sort it; OR apply the rescaling from
//   wasserstein_barycenter (rescale to target support) then call quantile_density_from_q.
//   This is the NO-osqp alternative to the R QP approach.

// Step 5: invert sorted q_bar to density via the same back-map as wasserstein_barycenter
//   (interpolate q_bar onto argvals using linear_interp, renormalize)
```

**Divergence to document in rustdoc:** The R `frechet` package uses an osqp quadratic program to enforce monotonicity of the signed quantile average. fdars instead applies a sort-based monotone projection (equivalent for smooth data; may introduce O(m log m) overhead per prediction but avoids a new crate dependency). Document this in the public function's rustdoc.

**Covariance inverse implementation:** Use the existing `linalg::cholesky_factor` + `cholesky_forward_back` (already `pub(crate)` at `src/linalg.rs:85,113`). [VERIFIED: src/linalg.rs:85-134] For the vector-matrix product: compute `v = Σ̂⁻¹(x − X̄)` via `cholesky_solve`, then `sᵢ = 1 + dot(Xᵢ − X̄, v)`.

### 6. Local Fréchet Regression (FRE-01-05)

**Local-linear kernel weight formula** (Fan–Gijbels style; from `LocWassReg.R`):

```
Kᵢ = ∏ⱼ gaussian_kernel(Xᵢⱼ − x₀ⱼ, h)   (product kernel over predictors)

mu0 = mean(K)                           // scalar
mu1 = (1/n) Σᵢ Kᵢ (Xᵢ − x₀)           // p-vector
mu2 = (1/n) Σᵢ Kᵢ (Xᵢ − x₀)(Xᵢ − x₀)ᵀ  // p×p matrix

// Local-linear weight for observation i at prediction point x₀:
sᵢ = Kᵢ · (1 − mu1ᵀ μ₂⁻¹ (Xᵢ − x₀))
```

[CITED: https://github.com/functionaldata/tFrechet/blob/master/R/LocWassReg.R — via WebFetch]

The local-linear bias correction means sᵢ can also be negative. Same signed-quantile-average + sort-monotone approach applies.

**Bandwidth parameter:** User-supplied scalar `h` (same `h` per predictor dimension); no default cross-validation is required for Phase 40. Validate h > 0 at entry; return `FdarError::InvalidParameter` otherwise.

**`gaussian_kernel(d, h)` already exists in `helpers.rs`:** [VERIFIED: src/helpers.rs:247]

```rust
// helpers.rs:247
pub fn gaussian_kernel(d: f64, h: f64) -> f64 {
    if h < 1e-15 { return 0.0; }
    (-d * d / (2.0 * h * h)).exp()
}
```

For the product kernel over p predictors: `K_i = ∏ⱼ gaussian_kernel(X_i[j] - x0[j], h)`.

### 7. Fréchet ANOVA — Dubey–Müller Statistic (FRE-01-08)

**Reference:** Dubey & Müller (2019), *Biometrika* 106(4):803-821.

**Sample quantities** (k groups, nₗ observations in group l, n = Σ nₗ, λₗ = nₗ/n):

```
μ̂ₗ  = Fréchet mean of group l  (minimize Σᵢ∈l d²(Yᵢ, ·))
μ̂   = Fréchet mean of all n observations (pooled)
V̂ₗ  = (1/nₗ) Σᵢ∈l d²(Yᵢ, μ̂ₗ)      (group Fréchet variance)
V̂ₚ  = (1/n)  Σᵢ   d²(Yᵢ, μ̂)       (pooled Fréchet variance)
σ̂ₗ² = (1/nₗ) Σᵢ∈l [d²(Yᵢ, μ̂ₗ) − V̂ₗ]²  (within-group variance of squared dist)
```

**Test statistics:**
```
Fₙ = V̂ₚ − Σₗ λₗ V̂ₗ                           (between-group dispersion contrast)
Uₙ = Σⱼ<ₗ (λⱼ λₗ / σ̂ⱼ² σ̂ₗ²)(V̂ⱼ − V̂ₗ)²      (variance heterogeneity)
Tₙ = n·Uₙ / Σₗ(λₗ/σ̂ₗ²)  +  n·Fₙ² / Σₗ(λₗ²σ̂ₗ²)
```

**Asymptotic p-value:** Tₙ →ᵈ χ²(k−1) under H₀; p = P(χ²(k−1) > Tₙ).

The R package uses `1 - pchisq(t0, df = k - 1)`. [CITED: rdrr.io/cran/frechet/src/R/DenANOVA.R via WebFetch]

**Permutation p-value (fdars convention):** 999 permutations (matching `inference/permutation.rs` `DEFAULT_N_PERM = 999` [VERIFIED: src/inference/permutation.rs:18]); shuffle group labels, recompute Tₙ per permutation; p = (count(Tₚₑᵣₘ ≥ Tₒbₛ) + 1) / (B + 1).

```rust
// Permutation loop pattern (from elastic_explain.rs:313-314):
for k in 0..n_perm {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
    // Fisher-Yates shuffle of group labels
    // recompute Tn
}
// p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0)
```

[VERIFIED: src/elastic_explain.rs:313-314 — seed pattern `StdRng::seed_from_u64(seed.wrapping_add(p as u64))`]

**Result struct:**

```rust
pub struct FrechetAnovaResult {
    pub statistic: f64,        // Tn (Dubey-Müller combined statistic)
    pub p_value_asymptotic: f64,
    pub p_value_permutation: f64,
    pub n_perm: usize,
    pub group_frechet_variances: Vec<f64>,  // V̂ₗ per group
    pub pooled_frechet_variance: f64,       // V̂ₚ
    pub fn_statistic: f64,     // Fₙ (between-group component)
    pub un_statistic: f64,     // Uₙ (variance-heterogeneity component)
    pub group_labels: Vec<usize>,
}
```

---

## Architecture Patterns

### System Architecture Diagram

```
Caller (external crate or tests)
         │
         ▼
  frechet/mod.rs  ─── re-exports all public items ──► lib.rs (crate root)
         │
    ┌────┼─────────────┬──────────────────┐
    ▼    ▼             ▼                  ▼
 space.rs           mean.rs         regression.rs     anova.rs
 MetricSpace trait  frechet_mean()  frechet_global_reg()  frechet_anova()
 WassersteinDens    frechet_var()   frechet_local_reg()
 w2_distance()                      ▲                  ▲
 signed_q_avg()                     │                  │
    │                               │ signed_q_avg()   │ frechet_mean/var
    ▼                               │                  │
 density_fda.rs                  space.rs          mean.rs
 wasserstein_barycenter()        linalg.rs         space.rs
 (non-negative weights only)     helpers.rs
```

### Recommended Project Structure

```
fdars-core/src/frechet/
├── mod.rs          # Result types + pub use re-exports
├── space.rs        # MetricSpace trait + WassersteinDensitySpace + w2_distance + signed quantile avg
├── mean.rs         # frechet_mean(), frechet_variance() generic functions
├── regression.rs   # frechet_global_reg(), frechet_local_reg() (density variant = same fns with density space)
└── anova.rs        # frechet_anova() — Dubey-Müller Tn + permutation
```

### Pattern 1: Signed Quantile Average (core divergence from barycenter)

```rust
// Source: R package GloWassReg.R / LocWassReg.R (via WebFetch inspection)
// Private helper in frechet/space.rs
fn signed_quantile_average(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    weights: &[f64],   // signed, sums to 1 (or not — we normalize to 1/n for global)
    n_q: usize,
) -> Result<Vec<f64>, FdarError> {
    let (n, _) = density_matrix.shape();
    let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();
    let mut q_bar = vec![0.0_f64; n_q];
    for i in 0..n {
        let row: Vec<f64> = (0..density_matrix.ncols()).map(|j| density_matrix[(i, j)]).collect();
        let q_i = density_to_quantile(&row, argvals, &t_grid);  // private helper (mirrors wasserstein_barycenter inner loop)
        for j in 0..n_q {
            q_bar[j] += weights[i] * q_i[j];
        }
    }
    // Monotone projection: sort q_bar (isotonic — simplest no-QP alternative)
    q_bar.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Invert to density using wasserstein_barycenter back-half pattern
    // (rescale to target support, finite-difference density, interpolate, renormalize)
    let dens = quantile_to_density(&q_bar, &t_grid, argvals)?;
    Ok(dens)
}
```

### Pattern 2: Covariance Inverse for Global Regression

```rust
// Use existing linalg::cholesky_factor and cholesky_forward_back
use crate::linalg::{cholesky_factor, cholesky_forward_back};

// Compute sigma_hat (p×p, flat row-major), regularize, then:
let chol = cholesky_factor(&sigma_hat_reg, p)?;  // fallback: add 1e-8 diagonal
// For each prediction point x_pred:
let diff_x: Vec<f64> = x_pred.iter().zip(x_bar.iter()).map(|(&a,&b)| a-b).collect();
let v = cholesky_forward_back(&chol, &diff_x, p);  // v = Sigma^{-1}(x-xbar)
// For observation i: s_i = 1 + dot(X_i - xbar, v)
let s_i = 1.0 + diff_i.iter().zip(v.iter()).map(|(&a,&b)| a*b).sum::<f64>();
```

[VERIFIED: src/linalg.rs:85-134]

### Anti-Patterns to Avoid

- **Calling `wasserstein_barycenter` with signed/negative weights:** The function validates `w >= 0` and returns `FdarError::InvalidParameter` for negative weights. [VERIFIED: density_fda.rs:450] Never pass global/local regression weights to it.
- **Re-implementing density→quantile from scratch:** Use the private `density_to_quantile` pattern (CDF = `cumulative_trapz` + `linear_interp`) — mirrors the code already in `wasserstein_barycenter` inner loop. [VERIFIED: density_fda.rs:492-496]
- **Using nalgebra DMatrix for the Σ̂ inverse:** Prefer the in-crate `linalg::cholesky_factor` + `cholesky_forward_back` which avoids the nalgebra matrix allocation. Use nalgebra's Cholesky only if p > ~50 (not a concern for typical Euclidean predictor dimensions in FDA).
- **Forgetting per-thread RNG in permutation loop:** Always `StdRng::seed_from_u64(seed.wrapping_add(k as u64))` inside the loop, not a single RNG mutated across iterations. [VERIFIED: src/elastic_explain.rs:313-314]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Monotone QP for quantile projection | Custom active-set QP | Sort-based isotonic projection (Vec::sort) | No new crate needed; sort is O(m log m) vs O(m²) QP; R uses osqp but we have no dependency budget |
| Density→quantile conversion | Custom CDF inversion | Pattern from `wasserstein_barycenter` inner loop (lines 476-496 density_fda.rs) | Already tested, handles dedup/numerical edge cases |
| Gaussian kernel | Custom exp(-d²/2h²) | `helpers::gaussian_kernel` [VERIFIED: helpers.rs:247] | Already present, tested |
| Cholesky solve for Σ⁻¹ | LU or explicit inverse | `linalg::cholesky_factor` + `cholesky_forward_back` [VERIFIED: linalg.rs:85-134] | Already pub(crate), handles non-PD with regularization pattern |
| Fréchet mean of non-negative weights | Custom quantile avg | `density_fda::wasserstein_barycenter` [VERIFIED: density_fda.rs:407] | Tested, handles edge cases |
| Trapezoidal/Simpson integration | Custom integrator | `helpers::trapz`, `helpers::cumulative_trapz` [VERIFIED: helpers.rs:197,234] | Established |

---

## Critical Risk: Negative Weights in Global/Local Regression

### What the CONTEXT.md says vs what the R reference does

CONTEXT.md (locked): "Global Fréchet regression = ... → weighted Fréchet mean"

R reference reality: `gx = colMeans(qin * s)` — signed weighted quantile average, **not** `getWFmean`. The R package then calls an osqp QP to enforce monotonicity of `gx`.

### Why `wasserstein_barycenter` cannot be used directly

`wasserstein_barycenter` validates `weights[i] >= 0` at line 450 of density_fda.rs:
```rust
// density_fda.rs:450-453 [VERIFIED: density_fda.rs:450]
if w.iter().any(|&wi| wi < 0.0 || !wi.is_finite()) {
    return Err(FdarError::InvalidParameter { ... });
}
```

Global regression weights sᵢ(x) = 1 + (Xᵢ−X̄)ᵀΣ̂⁻¹(x−X̄) can be negative, so a direct call would error.

### Recommended resolution (no new crate, no osqp)

Implement a private `signed_quantile_average` helper in `frechet/space.rs` that:
1. Computes Q̄(t) = (1/n) Σᵢ sᵢ(x) Qᵢ(t) directly (signed dot product)
2. Applies sort-based monotone projection on Q̄
3. Inverts Q̄ to a density using the `wasserstein_barycenter` back-half pattern

This is the ONE point where the density backend's `weighted_frechet_mean` implementation must diverge from `wasserstein_barycenter`. The trait's `weighted_frechet_mean` method can remain with non-negative weight semantics; the regression functions bypass it and call the signed helper directly. Document this in rustdoc.

### Rustdoc divergence note to include

```
/// **Divergence from R `frechet`:** The R package uses an `osqp` quadratic program
/// to enforce quantile-function monotonicity after computing the signed weighted average
/// Q̄(t) = (1/n) Σᵢ sᵢ(x) Qᵢ(t). fdars applies a sort-based isotonic projection
/// (Vec::sort on Q̄) as a zero-dependency alternative. On smooth data these are
/// equivalent; on highly non-smooth data the sort projection may produce a slightly
/// more conservative (flatter) density estimate than the QP projection.
```

---

## Concrete Test Ideas

### Fréchet Mean and Variance (FRE-01-02, FRE-01-03)

```rust
// 1. Fréchet mean of identical densities → variance ≈ 0
let dens = truncated_gaussian(&argvals, 0.0);
let mat = FdMatrix::from_rows(&[dens.clone(), dens.clone(), dens.clone()]);
let mean = wasserstein_barycenter(&mat, &argvals, None)?;
// mean ≈ dens; variance = 0
let var = frechet_variance(&space, &[dens.clone(),...], &mean, None)?;
assert!(var < 1e-8);

// 2. Fréchet mean agrees with wasserstein_barycenter for non-negative weights
// frechet_mean(...) should return the same result as wasserstein_barycenter(...)
// within tolerance 1e-6

// 3. Variance grows with dispersion: 3 densities spread ±2 → variance > 3 densities spread ±0.5
```

### 1D Wasserstein Distance (FRE-01-06)

```rust
// W₂ = 0 for identical densities
let d = w2_distance(&dens, &dens, &argvals, n_q)?;
assert!(d < 1e-8);

// Hand-computed shift: two unit Gaussians shifted by δ on sufficiently wide grid
// W₂(N(0,1), N(δ,1)) = δ  (exact in L2-Wasserstein for location families)
// Use δ = 0.5 on [-5, 5] grid, expect W₂ ≈ 0.5 within 0.05
let d_shifted = w2_distance(&gauss0, &gauss_delta, &argvals, n_q)?;
assert!((d_shifted - delta).abs() < 0.05);
// Note: tolerance is loose due to piecewise-linear quantile approximation
```

### Global Regression (FRE-01-04, FRE-01-07)

```rust
// Synthetic: n=20 scalar predictors x_i uniform on [-1,1],
// response density = N(x_i, 0.3) on [-3, 3] grid
// Regression at x_pred = 0.5 should produce a density near N(0.5, 0.3)
// Check: W₂(predicted, N(0.5, 0.3)) < 0.2
```

### Local Regression (FRE-01-05)

```rust
// Same synthetic setup as global regression but with local regression
// Local should produce tighter fit near training points
// At x_pred = 0.0 (near x_i = 0): W₂(pred, N(0, 0.3)) < 0.15 (tighter than global)
```

### Fréchet ANOVA (FRE-01-08)

```rust
// 1. Flagging real differences: group A = N(-1,0.3), group B = N(1,0.3)
//    p_value_permutation < 0.05 (clear shift)

// 2. No false flag on homogeneous sample: group A = N(0,0.3), group B = N(0,0.3)
//    p_value_permutation > 0.05 (no difference)

// 3. Seeded reproducibility: same seed → same p_value across two calls

// 4. Error path: fewer than 2 groups → InvalidParameter
```

---

## Common Pitfalls

### Pitfall 1: Calling wasserstein_barycenter with signed weights

**What goes wrong:** Returns `FdarError::InvalidParameter` for negative weights.
**Why it happens:** Global regression weights can be negative (observations opposite to x in predictor space).
**How to avoid:** Use `signed_quantile_average` helper for regression; only call `wasserstein_barycenter` for plain Fréchet mean (non-negative weights).
**Warning signs:** Test for `InvalidParameter` errors on regression when predictor range > 1 std dev.

### Pitfall 2: Non-monotone signed quantile average

**What goes wrong:** The sorted quantile average Q̄ might be nearly flat or slightly non-monotone due to cancellation from large negative weights; the density inversion then produces garbage.
**Why it happens:** For extrapolation points x far from X̄, some sᵢ become very large positive while others become very large negative; their quantile contributions nearly cancel.
**How to avoid:** Validate that Q̄ after sort has a reasonable range (> some ε like 1e-6); return `FdarError::ComputationFailed` if the range is degenerate (matches `wasserstein_barycenter`'s `q_range < 1e-15` guard [VERIFIED: density_fda.rs:505]).

### Pitfall 3: Σ̂ singular or near-singular

**What goes wrong:** `cholesky_factor` returns `ComputationFailed` for degenerate predictor configurations (all predictors identical, or collinear p > 1 case).
**Why it happens:** Singular predictor covariance means Σ̂⁻¹ doesn't exist.
**How to avoid:** Add ridge regularization `sigma_hat[j*p+j] += 1e-8` before Cholesky (matches `elastic_changepoint.rs:299` pattern with 1e-6 ridge [VERIFIED: elastic_changepoint.rs:299]). Document in rustdoc.

### Pitfall 4: Fréchet ANOVA σ̂ₗ² near zero

**What goes wrong:** If all objects in a group are identical, σ̂ₗ² = 0, and Uₙ / σ̂ₗ² diverges.
**Why it happens:** σ̂ₗ² = within-group variance of squared distances; zero means all objects identical.
**How to avoid:** Guard: if any σ̂ₗ² < NUMERICAL_EPS, return an appropriate result (Tₙ = 0 or flag). Alternative: clamp denominator to NUMERICAL_EPS. Document behavior.

### Pitfall 5: Kernel bandwidth = 0 in local regression

**What goes wrong:** `gaussian_kernel` returns 0.0 for any h < 1e-15, so all weights become 0, and the signed quantile average becomes 0/degenerate.
**Why it happens:** User passes h = 0 or negative.
**How to avoid:** Validate `h > 0` at entry; return `InvalidParameter`. Already handled by `gaussian_kernel` returning 0 for h < 1e-15 — but that produces silent wrong output. Explicit validation at function entry is required.

---

## Code Examples

### Fréchet mean call pattern

```rust
// Source: density_fda.rs:407 (wasserstein_barycenter signature)
// frechet_mean for WassersteinDensitySpace internally calls:
pub fn wasserstein_barycenter(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    weights: Option<&[f64]>,   // None → uniform 1/n
) -> Result<Vec<f64>, FdarError>
```

### Permutation loop pattern (from elastic_explain.rs)

```rust
// Source: elastic_explain.rs:313-314 [VERIFIED]
for k in 0..n_perm {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
    let mut perm_idx: Vec<usize> = (0..n).collect();
    perm_idx.shuffle(&mut rng);
    // recompute Tn with permuted group labels
}
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

### Result struct pattern (from fts/mod.rs)

```rust
// Source: fts/mod.rs:28-45 pattern [VERIFIED]
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrechetGlobalRegResult {
    /// Predicted response objects (one per xout row), shape n_out × m (FdMatrix rows are densities).
    pub predicted: FdMatrix,
    /// Predictor values at which predictions were made, shape n_out × p.
    pub xout: FdMatrix,
    /// Sample mean of training predictors, length p.
    pub x_bar: Vec<f64>,
}
```

### `mod.rs` barrel pattern

```rust
// Source: fts/mod.rs structure [VERIFIED]
mod anova;
mod mean;
mod regression;
mod space;

pub use anova::{frechet_anova, FrechetAnovaResult};
pub use mean::{frechet_mean, frechet_variance};
pub use regression::{frechet_global_reg, frechet_local_reg, FrechetGlobalRegResult, FrechetLocalRegResult};
pub use space::{MetricSpace, WassersteinDensitySpace};
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Euclidean mean | Fréchet mean in metric space (Petersen & Müller 2019) | Handles non-Euclidean responses (densities, matrices, etc.) |
| Kernel density regression | Global/local Fréchet regression in Wasserstein space | Preserves density shape constraints |
| Classical ANOVA | Dubey–Müller Fréchet ANOVA (2019) | Tests for differences in metric-space object distributions |
| Direct `wasserstein_barycenter` with signed weights | Signed quantile average + sort-monotone projection | No external QP solver; O(m log m) instead of QP overhead |

**Deprecated/outdated:**
- Direct Wasserstein barycenter with unconstrained weights: replaced by signed quantile average for regression (the barycenter path only works for non-negative weights).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Sort-based isotonic projection on Q̄ is an acceptable no-osqp alternative | §Signed Quantile Average; §Pitfall 2 | Q̄ may be non-monotone in extreme extrapolation; density inversion returns garbage. Mitigation: add guard on Q̄ range. |
| A2 | σ̂ₗ² for Dubey-Müller Tn matches the formula `(1/nₗ) Σᵢ [d²(Yᵢ,μ̂ₗ) - V̂ₗ]²` | §Fréchet ANOVA | Wrong σ̂ₗ² formula → wrong Tn → wrong p-value. Verify against Dubey-Müller (2019) paper. |
| A3 | Asymptotic p-value uses χ²(k-1) degrees of freedom (same as R DenANOVA: `pchisq(t0, df=k-1)`) | §Fréchet ANOVA | Wrong df → miscalibrated asymptotic p. Permutation p-value is robust regardless. |
| A4 | Product kernel (∏ Gaussian per predictor dimension) is correct for multi-predictor local regression | §Local Regression | If R uses a different multi-predictor kernel (e.g., Mahalanobis-weighted), predictions will differ. Document as single-bandwidth product kernel. |
| A5 | `cholesky_forward_back` is accessible as `pub(crate)` from `frechet/` submodules | §Covariance Inverse | If visibility is more restricted, the planner must add `pub(crate)` to linalg.rs. Verify at implementation time. |

---

## Open Questions

1. **σ̂ₗ² formula for Dubey–Müller Tn**
   - What we know: V̂ₗ is the group Fréchet variance; σ̂ₗ² enters the Tn denominator as a variance-of-squared-distances estimate.
   - What's unclear: Exact definition — is it `(1/nₗ) Σᵢ [d²(Yᵢ,μ̂ₗ) − V̂ₗ]²` or a different moment?
   - Recommendation: Treat as `[ASSUMED]` and verify against the full Dubey-Müller 2019 Biometrika paper at implementation time. The asymptotic chi-sq(k-1) result is confirmed from R source; the exact σ̂ₗ² formula determines Tn magnitude only.

2. **Monotone projection adequacy for global regression extrapolation**
   - What we know: Sort-based monotone projection works when Q̄ is mostly monotone (weights are small perturbations from uniform 1/n).
   - What's unclear: How badly the projection degrades for strongly negative weights (extrapolation far from X̄).
   - Recommendation: Add a guard returning `ComputationFailed` when Q̄ range after sort is < 1e-6 (analogous to `wasserstein_barycenter`'s `q_range < 1e-15` guard). Document the extrapolation limitation in rustdoc.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 40 is a pure Rust code addition. No external tools, services, or CLIs beyond the existing cargo toolchain are required. All dependencies are in-crate.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`#[cfg(test)] mod tests`) |
| Config file | none — inline test modules per file |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel frechet 2>/dev/null` |
| Full suite command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FRE-01-01 | MetricSpace trait + WassersteinDensitySpace construct without panic | unit | `cargo test frechet::space` | ❌ Wave 0 |
| FRE-01-02 | Fréchet mean of identical densities ≈ input density | unit | `cargo test frechet::mean::tests::mean_identical` | ❌ Wave 0 |
| FRE-01-03 | Fréchet variance = 0 for identical sample, grows with dispersion | unit | `cargo test frechet::mean::tests::variance` | ❌ Wave 0 |
| FRE-01-04 | Global regression tracks linear density-mean-in-x relationship | unit | `cargo test frechet::regression::tests::global_reg` | ❌ Wave 0 |
| FRE-01-05 | Local regression tracks same relationship, tighter near training pts | unit | `cargo test frechet::regression::tests::local_reg` | ❌ Wave 0 |
| FRE-01-06 | W₂ = 0 for identical; ≈ δ for two shifted Gaussians | unit | `cargo test frechet::space::tests::w2_distance` | ❌ Wave 0 |
| FRE-01-07 | Density-response regression = global regression with density space | unit | `cargo test frechet::regression::tests::density_response` | ❌ Wave 0 |
| FRE-01-08 | ANOVA flags shifted groups; p > 0.05 for homogeneous; seeded reproducibility | unit | `cargo test frechet::anova::tests` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel frechet 2>/dev/null`
- **Per wave merge:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work`; plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps

- [ ] `fdars-core/src/frechet/mod.rs` — barrel file + result types
- [ ] `fdars-core/src/frechet/space.rs` — MetricSpace trait + WassersteinDensitySpace + w2_distance + signed_quantile_average
- [ ] `fdars-core/src/frechet/mean.rs` — frechet_mean + frechet_variance
- [ ] `fdars-core/src/frechet/regression.rs` — frechet_global_reg + frechet_local_reg
- [ ] `fdars-core/src/frechet/anova.rs` — frechet_anova
- [ ] Wire `pub mod frechet;` into `fdars-core/src/lib.rs`

---

## Security Domain

`security_enforcement` not configured — this is a pure algorithm/library phase with no network, authentication, cryptography, or user-input surfaces. ASVS categories: not applicable.

---

## Sources

### Primary (VERIFIED — codebase reads this session)

- `fdars-core/src/density_fda.rs` (entire file, lines 1-682) — `wasserstein_barycenter` signature (line 407), weight validation (line 450), density_to_quantile pattern (lines 476-496), `q_range < 1e-15` guard (line 505)
- `fdars-core/src/linalg.rs` (lines 1-152) — `cholesky_factor` (line 85), `cholesky_forward_back` (line 113)
- `fdars-core/src/helpers.rs` — `gaussian_kernel` (line 247), `NUMERICAL_EPS` (line 4), `cumulative_trapz` (line 197), `trapz` (line 234), `linear_interp` (line 172)
- `fdars-core/src/elastic_explain.rs` — permutation RNG pattern (lines 313-314)
- `fdars-core/src/inference/permutation.rs` — `DEFAULT_N_PERM = 999` (line 18)
- `fdars-core/src/elastic_changepoint.rs` — Cholesky regularization pattern (line 299)
- `fdars-core/src/fts/mod.rs` — result struct pattern, barrel file structure

### Secondary (CITED — official/authoritative sources)

- [Petersen & Müller 2019 — "Fréchet regression for random objects with Euclidean predictors", Annals of Statistics 47(2)](https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Fr%C3%A9chet-regression-for-random-objects-with-Euclidean-predictors/10.1214/17-AOS1624.full) — global weight formula s(z,x) = 1 + (z−μ)ᵀΣ⁻¹(x−μ)
- [Dubey & Müller 2019 — "Fréchet analysis of variance for random objects", Biometrika 106(4):803-821](https://academic.oup.com/biomet/article-abstract/106/4/803/5609104) — Tn statistic, asymptotic χ²(k-1), Fn and Un components
- [GloWassReg.R source (functionaldata/tFrechet)](https://github.com/functionaldata/tFrechet/blob/master/R/GloWassReg.R) — `gx = colMeans(qin * s)` signed quantile average, no negative weight clipping
- [LocWassReg.R source (functionaldata/tFrechet)](https://github.com/functionaldata/tFrechet/blob/master/R/LocWassReg.R) — local-linear weights via mu0/mu1/mu2, Gaussian kernel, `gx = colMeans(qin * s) * n`
- [DenANOVA.R source (rdrr.io/cran/frechet)](https://rdrr.io/cran/frechet/src/R/DenANOVA.R) — bootstrap p-value (`pvalBoot`), asymptotic p-value via `1 - pchisq(t0, df = k - 1)`
- [DenANOVAStatistic.R source (rdrr.io/cran/frechet)](https://rdrr.io/cran/frechet/src/R/DenANOVAStatistic.R) — Fn = Vp − Σλₗ Vₗ, Tn formula

### Tertiary (LOW confidence — training / non-authoritative)

- arxiv.org/html/2605.19519 — confirmation of weight formula; the specific expression `sⱼ(x) = 1 + (Xⱼ−X̄)Σ̂⁻¹(x−X̄)` is consistent with the Petersen-Müller paper

---

## Metadata

**Confidence breakdown:**
- MetricSpace trait design: HIGH — derived directly from codebase conventions + CONTEXT.md locked decisions
- Signed quantile average (KEY RISK): MEDIUM — confirmed from R source code (GloWassReg.R); formula is clear
- W₂ distance formula: HIGH — directly derivable from density_fda.rs machinery (lines 476-496)
- Fréchet ANOVA Tn: MEDIUM — confirmed from DenANOVA.R R source; σ̂ₗ² exact formula is [ASSUMED]
- Local regression kernel weights: MEDIUM — confirmed from LocWassReg.R R source
- Cholesky inverse for Σ̂⁻¹: HIGH — verified against linalg.rs existing API

**Research date:** 2026-08-22
**Valid until:** 2026-11-22 (stable algorithm domain; 90 days)
