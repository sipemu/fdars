# Phase 32: Flexible Mixed-Effects Regression - Research

**Researched:** 2026-08-20
**Domain:** Functional mixed-effects models (denseFLMM / multiFAMM / fastFMM / FoF-RE) in Rust, extending `fdars-core/src/famm.rs` and `fdars-core/src/fof_regression.rs`
**Confidence:** MEDIUM

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Parametrization:** Mixed-model equations formulated over **FPC scores** (reuse `regression::fdata_to_pc_1d` + the existing `famm.rs` fixed-effect machinery), not spline/basis coefficients. Document this choice vs R baselines' basis-coefficient formulations in rustdoc.
- **Variance Components:** REML-style / method-of-moments variance-component estimation over the per-component score models — no new crate dependency, reuse existing linear-algebra (`linalg`, nalgebra). Return variance components (random-intercept/slope variances + residual) alongside fixed effects.
- **Variant Depth:** multiFAMM and fastFMM implemented faithful-by-capability, not to exact R signatures. fastFMM realized as a **massively-univariate** fit (per-gridpoint / per-component mixed model with a fast inference path), documenting the divergence from the R `fastFMM` internals in rustdoc. multiFAMM covers the multivariate (stacked-response) additive mixed case reusing the denseFLMM core.
- **Correctness Tests:** Synthetic-recovery: generate data from a known mixed model (fixed effect + grouped random intercepts/slopes with known variance components); assert recovery of fixed effects and the variance-component structure within a documented tolerance, and fitted curves track truth. Invalid-input `FdarError` paths: empty data / mismatched grouping-factor length / singular design / mismatched dimensions — never panic.
- **Strictly additive/non-breaking:** NO changes to existing public signatures; extend files + `pub use` + crate-root re-export only.
- **No new crate dependency.** Reuse existing linalg/FPCA infra.
- `Result<T, FdarError>`; MSRV 1.81; column-major `FdMatrix`; `linalg` feature for Cholesky.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.

### Claude's Discretion

- Config/result struct field names, default REML iteration counts / tolerances, internal helper factoring, and exact test counts are at Claude's discretion within the above.

### Deferred Ideas (OUT OF SCOPE)

- Plotting/rendering of mixed-model diagnostics (out of scope — numeric outputs only).
- Changing the base function-on-function capability (already at parity — REG-05 extends only the RE variant).
- Bayesian functional mixed models (out of scope).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-05 | Extend functional mixed models beyond fixed-effect testing to full random-effects estimation — dense FLMM, multiFAMM, fastFMM in `famm.rs`, and flexible-RE FoF in `fof_regression.rs` | All four estimators mapped to exact reuse paths; API surface, math, and test strategy documented below |
</phase_requirements>

---

## Summary

Phase 32 extends `fdars-core` with four functional mixed-effects estimators, all additive/non-breaking on top of the already-mature `famm.rs` fixed-effects machinery and `fof_regression.rs` base FoF. The core algorithmic pattern is uniform across all four: decompose functional responses into FPC score series (via `fdata_to_pc_1d`), fit a scalar linear mixed model per FPC component (reusing `fit_scalar_mixed_model` from `famm.rs`), and reconstruct functional estimates by back-projecting through the rotation matrix.

The **denseFLMM** estimator extends the existing `fmm` by adding random slopes alongside random intercepts and returning the convergence metadata that distinguishes it from `fmm`. The **multiFAMM** stacks `D` response `FdMatrix` slices, runs `dense_flmm` independently per dimension, and aggregates fitted curves and variance components. The **fastFMM** fits a reduced scalar mixed model at each of the `m` grid points (massively-univariate), then applies a simple running-mean smoother and returns per-gridpoint t-statistics and p-values — diverging from R's mgcv-based smoothing by design. The **flexible-RE FoF** estimator in `fof_regression.rs` calls the existing double-FPCA path then fits a scalar mixed model on each Y-score component, producing random-effect-adjusted fitted curves.

**Primary recommendation:** Implement in two sequential plans — (1) `famm.rs` additions (`dense_flmm`, `multi_famm`, `fast_fmm`, config+result structs, re-exports) and (2) `fof_regression.rs` addition (`fof_re_regression`, `FofReResult`, `FofReConfig`, re-exports + predict variant). Both plans reuse the private helpers already in `famm.rs` verbatim; no new numeric primitive or crate dependency is needed.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FPC decomposition | `regression.rs` | — | `fdata_to_pc_1d` is the crate's canonical FPCA entry; all four estimators call it |
| Per-component scalar mixed model (REML EM) | `famm.rs` private helpers | `linalg.rs` | `fit_scalar_mixed_model`, `reml_variance_update`, `compute_blup`, `gls_update_gamma` already exist |
| denseFLMM fitting | `famm.rs` (new `dense_flmm`) | `famm.rs` private helpers | Thin wrapper extending `fmm` with random slopes |
| Multivariate stacking | `famm.rs` (new `multi_famm`) | `dense_flmm` | Per-dimension loop calling `dense_flmm` core |
| Massively-univariate per-gridpoint LMM | `famm.rs` (new `fast_fmm`) | `famm.rs` private helpers | Per-column loop calling `fit_scalar_mixed_model` directly |
| Functional random-effects FoF regression | `fof_regression.rs` (new `fof_re_regression`) | `famm.rs` private helpers | Extends base double-FPCA with per-Y-score-component mixed model |
| Cholesky / GLS solves | `linalg.rs` | — | Already behind `linalg` feature; no new dep |
| Rayon parallelism | `parallel.rs` macros | — | `iter_maybe_parallel!` gates all per-component loops |

---

## Standard Stack

### Core (all existing — no new dependencies)

| Library / Module | Version / Location | Purpose | Why Standard |
|------------------|--------------------|---------|--------------|
| `crate::regression::fdata_to_pc_1d` | `src/regression.rs` | FPC decomposition | Canonical fdars FPCA entry; returns `FpcaResult` with `scores`, `rotation`, `mean`, `weights` |
| `crate::linalg::{cholesky_factor, cholesky_forward_back}` | `src/linalg.rs` | Cholesky solve for GLS and variance components | Already used in `famm.rs`; behind `linalg` feature |
| `crate::famm::{fit_scalar_mixed_model, SubjectStructure, ...}` | `src/famm.rs` (private) | Per-component scalar LMM | Complete REML EM already implemented; reuse verbatim |
| `crate::iter_maybe_parallel!` | `src/parallel.rs` | Gate per-component loops on rayon | Project-wide pattern for parallel iteration |
| `nalgebra` | 0.33 | Used internally by `fdata_to_pc_1d` SVD | Already in dep tree; no action needed |

### No New Dependencies

This phase installs **zero** new crates. All numeric primitives required (Cholesky, GLS, BLUP, FPCA) are already in-tree.

**Package Legitimacy Audit:** Not applicable — this phase adds zero external packages.

---

## Architecture Patterns

### System Architecture Diagram

```
Caller
  │
  ├─► dense_flmm(data, subject_ids, covariates, config)
  │       │
  │       ├─► fdata_to_pc_1d(data, ncomp, argvals)          [regression.rs]
  │       │       └─► FpcaResult {scores, rotation, mean, weights}
  │       │
  │       ├─► iter_maybe_parallel!(0..k)
  │       │   └─► fit_scalar_mixed_model(scores_k, ...)     [famm.rs private]
  │       │           ├─► estimate_fixed_effects (OLS init)
  │       │           ├─► estimate_variance_components (moments init)
  │       │           ├─► loop (REML EM, ≤ max_iter):
  │       │           │       shrinkage_weights → gls_update_gamma → reml_variance_update
  │       │           └─► compute_blup → ScalarMixedResult
  │       │
  │       ├─► recover_beta_functions(gamma, rotation)        [famm.rs private]
  │       ├─► recover_random_effects(u_hat, rotation)        [famm.rs private]
  │       └─► DenseFlmmResult {mean_function, beta_functions, random_effects,
  │                             fitted, residuals, sigma2_u, sigma2_eps,
  │                             random_variance, n_iter, converged, ...}
  │
  ├─► multi_famm(data_vec: &[FdMatrix], subject_ids, covariates, config)
  │       │
  │       ├─► for d in 0..D:
  │       │   └─► dense_flmm(data_vec[d], ...)               [above]
  │       │           └─► DenseFlmmResult per dimension
  │       │
  │       └─► MultiFammResult {components: Vec<DenseFlmmResult>,
  │                             stacked_fitted, stacked_residuals}
  │
  ├─► fast_fmm(data, subject_ids, covariates, config)
  │       │
  │       ├─► iter_maybe_parallel!(0..m)                    [per gridpoint]
  │       │   └─► fit_scalar_mixed_model(data.column(t), ...)
  │       │           └─► (gamma_t, u_hat_t, sigma2_u_t, sigma2_eps_t)
  │       │
  │       ├─► smooth_columns(raw_beta_matrix, window)        [running mean]
  │       ├─► compute_wald_stats(smoothed_beta, sigma2_eps)
  │       └─► FastFmmResult {beta_matrix, t_stats, p_values, sigma2_eps, n_grid}
  │
  └─► fof_re_regression(x_data, y_data, subject_ids, x_argvals, y_argvals, config)
          │
          ├─► fdata_to_pc_1d(x_data, ncomp_x, ...)           [regression.rs]
          ├─► fdata_to_pc_1d(y_data, ncomp_y, ...)
          ├─► fpca_x.project(x_data) → x_scores (n × ncomp_x)
          ├─► fpca_y.project(y_data) → y_scores (n × ncomp_y)
          │
          ├─► for l in 0..ncomp_y:
          │   └─► fit_scalar_mixed_model(y_scores.col(l), subject_ids, x_scores_as_cov)
          │           └─► (gamma_l, u_hat_l, sigma2_u_l, sigma2_eps_l)
          │
          ├─► coef_matrix from gamma_l (ncomp_x × ncomp_y)
          ├─► reconstruct beta_surface (m_y × m_x) from coef_matrix
          ├─► recover_random_effects(u_hat_l, fpca_y.rotation)  → random_effects (n_subj × m_y)
          └─► FofReResult {intercept, beta_surface, fitted, residuals, r_squared,
                            random_effects, sigma2_u, sigma2_eps, n_subjects,
                            fpca_x, fpca_y, coef_matrix, ncomp_x, ncomp_y}
```

### Recommended Project Structure

No new files or directories. All additions go into the two existing files:

```
fdars-core/src/
├── famm.rs          # Add: DenseFlmmConfig, DenseFlmmResult, dense_flmm
│                    #      MultiFammConfig, MultiFammResult, multi_famm
│                    #      FastFmmConfig, FastFmmResult, fast_fmm
│                    #      (private helpers reused verbatim)
├── fof_regression.rs # Add: FofReConfig, FofReResult, fof_re_regression, predict_fof_re
└── lib.rs           # Extend two pub use lines (lines 233, 242)
```

---

## Reference Formulations

### denseFLMM — FPC-Score Parametrization

**R baseline:** `denseFLMM` (Cederbaum, Scheipl, Greven). R's implementation estimates eigenfunctions from raw covariance smoothing (gamm/bam REML). [CITED: https://www.rdocumentation.org/packages/denseFLMM/versions/0.1.3/topics/denseFLMM]

**fdars parametrization (locked decision, document in rustdoc):**

Model at observation level (curves already centered by `fdata_to_pc_1d`):

```
Y_i(t) = μ(t) + Σ_j x_ij β_j(t) + b_i(t) + ε_i(t)

Where b_i(t) = Σ_k u_ik φ_k(t)   [random intercepts over FPC basis]
      β_j(t) = Σ_k γ_jk φ_k(t)   [fixed effects over FPC basis]
```

Per-component scalar mixed model (for FPC component k):

```
ξ_ij^(k) = x_i' γ^(k) + u_i^(k) + e_ij^(k)

  u_i^(k) ~ N(0, σ²_u^(k))
  e_ij^(k) ~ N(0, σ²_ε^(k))
```

REML EM updates (already in `famm.rs::reml_variance_update` — reuse verbatim): [VERIFIED: fdars-core/src/famm.rs:395-431]

```rust
// Verbatim from famm.rs:395-431
fn reml_variance_update(
    residuals: &[f64],
    ss: &SubjectStructure,
    weights: &[f64],
    sigma2_u: f64,
    p: usize,
) -> (f64, f64)
// Returns (sigma2_u_new, sigma2_e_new)
// sigma2_u_new = mean_s(u_hat_s^2 + sigma2_u*(1 - w_s))
// sigma2_e_new = sum_si((r_ij - u_hat_s)^2 + n_s*sigma2_u*(1-w_s)) / (n - p)
```

**denseFLMM vs fmm:** `dense_flmm` adds (1) optional random slopes — a second random effect z_ij * v_i(t) where v_i(t) = Σ_k s_ik φ_k(t) — requiring a two-random-effect scalar LMM per component; (2) convergence metadata (n_iter, converged) in the result. For the random-intercept-only case (`random_slopes = false`), `dense_flmm` is equivalent to `fmm` but exposes convergence info. [ASSUMED: random-slope extension requires augmenting `fit_scalar_mixed_model` or a new helper; the exact REML update for two RE terms needs to extend the current single-u update]

**BLUP reconstruction:** [VERIFIED: fdars-core/src/famm.rs:617-646]

```rust
// Verbatim from compute_blup, famm.rs:617-646
fn compute_blup(residuals, subject_map, n_subjects, sigma2_u, sigma2_eps) -> Vec<f64>
// û_i = (σ²_u / (σ²_u + σ²_ε/n_i)) * mean_i(r_ij)
```

---

### multiFAMM — Stacked Multivariate Extension

**R baseline:** `multifamm` (Volkmann, Stöcker, Scheipl, Greven 2021). R stacks the D-dimensional response into a DNT-length vector with dimension-specific fixed effects via interaction with dimension indicators, and uses multivariate FPCA for shared cross-dimension random effects. [CITED: https://journals.sagepub.com/doi/10.1177/1471082X211056158]

**fdars parametrization (faithful-by-capability):**

fdars implementation treats each response dimension independently (per-dimension `dense_flmm`), which captures within-dimension auto-correlation and within-group random effects but not cross-dimension covariance. This is a documented capability divergence: R's multiFAMM captures inter-dimension cross-covariances via joint multivariate FPCA; fdars uses D independent univariate FPCAs. Document in rustdoc.

```
Input:  data_vec: &[FdMatrix]   // D slices, each n_total × m_d
Output: MultiFammResult {
    components: Vec<DenseFlmmResult>,   // one per dimension
    stacked_fitted: FdMatrix,           // (n_total * D) × max_m, zero-padded if grids differ
    stacked_residuals: FdMatrix,
}
```

The per-dimension approach is sufficient for: (1) fixed-effect estimation per dimension, (2) random-intercept variance per dimension, (3) within-dimension fitted curves. It does NOT produce cross-dimension covariance kernels K_g(d,e)(t,t'). [ASSUMED]

---

### fastFMM — Massively-Univariate Path

**R baseline:** `fastFMM` / FUI (Cui, Leroux, Smirnova, Crainiceanu 2022, JCGS 31(1):219–230). R implementation: (1) per-gridpoint GLMM via `lme4`; (2) smooth raw coefficients via mgcv splines (GCV/REML); (3) analytic Wald CIs for Gaussian or bootstrap otherwise. [CITED: https://cran.r-project.org/web/packages/fastFMM/index.html]

**fdars parametrization (divergences documented):**

Three-step path mirroring FUI, but without mgcv:

```
Step 1: For each grid point t in 0..m:
    y_i(t) = x_i' β(t) + u_i(t) + ε_i(t)
    → fit_scalar_mixed_model(data.column(t), subject_map, n_subjects, covariates, p)
    → raw (γ̂(t), û_i(t), σ̂²_u(t), σ̂²_ε(t))

Step 2: Smooth raw β̂(t) along t:
    → running-mean smoother (window = 3 or configurable)
    fdars DIVERGENCE: R uses mgcv thin-plate splines; fdars uses running mean.
    Document in rustdoc.

Step 3: Wald-style inference (Gaussian path only):
    t_stat_jt = β̂_j(t) / se_j(t)   where se² estimated from σ̂²_ε(t) and X'X
    p_value_jt = 2 * (1 - Φ(|t_stat_jt|)) via standard-normal approximation
    fdars DIVERGENCE: R uses bootstrap for non-Gaussian; fdars provides Wald only.
```

Returns: [VERIFIED: fdars-core/src/famm.rs — pattern from PATTERNS.md]

```rust
pub struct FastFmmResult {
    pub beta_matrix: FdMatrix,   // p × m smoothed fixed-effect functions
    pub t_stats: FdMatrix,       // p × m Wald t-statistics
    pub p_values: FdMatrix,      // p × m pointwise p-values
    pub sigma2_eps: Vec<f64>,    // length m, per-gridpoint residual variance
    pub n_grid: usize,
}
```

---

### Flexible-RE Function-on-Function

**R baseline:** `refund::pffr` with `pcre()` term — FPCA-based functional random intercepts for function-on-function regression. Uses mgcv backend with penalized splines for random effects. [CITED: https://rdrr.io/cran/refund/man/pcre.html]

**fdars parametrization:**

Extends the existing double-FPCA path in `fof_regression.rs` with subject-level random effects on Y-score components:

```
Step 1 (same as fof_regression):
    X-FPCA: fdata_to_pc_1d(x_data, ncomp_x, x_argvals) → fpca_x
    Y-FPCA: fdata_to_pc_1d(y_data, ncomp_y, y_argvals) → fpca_y
    x_scores = fpca_x.project(x_data)   (n × ncomp_x)
    y_scores = fpca_y.project(y_data)   (n × ncomp_y)

Step 2 (new — replaces OLS with scalar LMM):
    For each Y-score component l in 0..ncomp_y:
        y_scores.col(l) = x_scores * gamma_l + u_il + e_il
        fit_scalar_mixed_model(y_scores.col(l), subject_ids, n_subjects,
                               Some(x_scores_as_FdMatrix), ncomp_x)
        → gamma_l (ncomp_x-vector), u_hat_l (n_subjects-vector),
           sigma2_u_l, sigma2_eps_l

Step 3 (reconstruction — same structure as fof_regression):
    coef_matrix[k, l] = gamma_l[k]
    beta_surface(s, t) = Σ_k Σ_l B[k,l] * phi_x^k(t) * phi_y^l(s)
    fitted_ij(s) = mean_y(s) + Σ_l (x_scores_ij * gamma_l + u_hat_{subject(i),l}) * phi_y^l(s)
    random_effects_i(s) = Σ_l u_hat_{i,l} * phi_y^l(s)   (n_subjects × m_y)
```

fdars DIVERGENCE from refund::pffr: pffr uses penalized splines for both fixed and random effects (GAMM framework); fdars uses FPC-score parametrization without basis penalties. Document in rustdoc.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cholesky solve for GLS / variance updates | New Cholesky impl | `linalg::cholesky_factor` + `linalg::cholesky_forward_back` | Already exists, tested, handles near-singularity |
| FPCA of functional data | Any SVD/covariance routine | `fdata_to_pc_1d` | Canonical fdars FPCA; includes L²-weighted inner product, correct `argvals` normalization |
| Per-component scalar LMM fit | New LMM solver | `famm.rs::fit_scalar_mixed_model` (private) | Complete REML EM with GLS fixed effects already implemented and tested |
| BLUP computation | New shrinkage estimator | `famm.rs::compute_blup` (private) | Already correct: ûᵢ = (σ²ᵤ / (σ²ᵤ + σ²ε/nᵢ)) * mean_rᵢ |
| Subject grouping | New hashmap or sort | `famm.rs::build_subject_map` (private) | Handles non-contiguous IDs, returns (map, n_subjects) |
| Parallel component loops | rayon direct | `iter_maybe_parallel!` macro | Project-wide pattern; gates rayon behind feature flag |
| X'X computation | Manual loop | `linalg::compute_xtx` | Already used in `fof_regression.rs` |

**Key insight:** Every numeric primitive this phase needs already exists in `famm.rs` as a private helper. The implementation task is to compose these helpers in slightly different orders for each estimator variant — it is not a novel algorithm implementation.

---

## Common Pitfalls

### Pitfall 1: Exposing Private `famm.rs` Helpers Across Modules

**What goes wrong:** `fof_re_regression` in `fof_regression.rs` needs `fit_scalar_mixed_model` which is private to `famm.rs`. Calling across module boundaries without making it `pub(crate)` causes a compile error.

**Why it happens:** Helpers in `famm.rs` are currently `fn` (private). Adding `fof_re_regression` in a sibling module requires `pub(crate)` on the helpers it calls.

**How to avoid:** Change precisely these helpers to `pub(crate)` in `famm.rs`:
- `fit_scalar_mixed_model`
- `build_subject_map`
- `SubjectStructure` (struct + impl)
- `ScalarMixedResult` (struct)
- `shrinkage_weights`

Do NOT expose `gls_update_gamma`, `reml_variance_update`, `compute_blup` — `fof_re_regression` calls `fit_scalar_mixed_model` which internally calls these.

**Warning signs:** `error[E0603]: function 'fit_scalar_mixed_model' is private` at compile time.

---

### Pitfall 2: Scoring Scales for Mixed Model in `fof_re_regression`

**What goes wrong:** `fof_regression.rs` uses `fpca_x.project(x_data)` which applies L²-weighted inner products (`weights[j]`). When these scores are passed as covariates to `fit_scalar_mixed_model`, the scale of the GLS system differs from what `famm.rs` expects (scores already normalized by `score_scale = h.sqrt()`).

**Why it happens:** `fit_all_components` in `famm.rs` applies a `score_scale` normalization before calling `fit_scalar_mixed_model`. The `fof_re_regression` path calls `fit_scalar_mixed_model` directly with projected scores that already carry the L² weighting.

**How to avoid:** In `fof_re_regression`, pass x_scores directly as covariates WITHOUT re-applying the h.sqrt() normalization — the weighting is already embedded in the projection. Add a comment marking this intentional divergence.

**Warning signs:** Variance components or fixed effect estimates wildly different in scale from `fmm` on equivalent data.

---

### Pitfall 3: `fast_fmm` Per-Gridpoint Parallelism vs mutable output

**What goes wrong:** Fitting `m` per-gridpoint models in parallel and writing results into a shared `Vec<Vec<f64>>` causes data races without proper synchronization.

**Why it happens:** `iter_maybe_parallel!` wraps `rayon::par_iter` which requires `Send + Sync` on closures. Writing into a pre-allocated `Vec` indexed by grid point is not naturally safe.

**How to avoid:** Collect into `Vec<PointwiseResult>` (immutable per-item result struct) then unpack into separate `Vec<f64>` columns after the parallel collect — exactly as `fit_all_components` does for component results:

```rust
// Correct pattern (mirrors famm.rs::fit_all_components):
let per_point: Vec<PointwiseResult> = iter_maybe_parallel!(0..m)
    .map(|t| { ... })
    .collect();
// Then unpack:
for (t, r) in per_point.iter().enumerate() { ... }
```

**Warning signs:** Compiler error about `Sync` bounds or race conditions in parallel context.

---

### Pitfall 4: Column Access for Per-Gridpoint Fit in `fast_fmm`

**What goes wrong:** Extracting column `t` from a column-major `FdMatrix` for the per-gridpoint fit requires copying into a `Vec<f64>`. Using `data.column(t)` returns a `&[f64]` (contiguous slice for column-major layout) — this is correct and zero-copy.

**Why it happens:** Rows are NOT contiguous in column-major layout (`row_to_buf` copies). Columns ARE contiguous.

**How to avoid:** Use `data.column(t)` (returns `&[f64]`) to pass the t-th column to `fit_scalar_mixed_model` — no copy needed. [VERIFIED: fdars-core/src/matrix.rs:1-44 — column-major layout documented]

**Warning signs:** Using `data.row_to_buf(i)` patterns in a per-gridpoint context (those iterate over observations, not time points).

---

### Pitfall 5: `MultiFammResult` Stacked Fitted Curves Dimension Mismatch

**What goes wrong:** Response dimensions may have different grid sizes (`m_1 ≠ m_2 ≠ ... ≠ m_D`). A naïve row-stacking attempt creates an `FdMatrix` with inconsistent column count.

**Why it happens:** `FdMatrix::from_column_major` requires `nrows * ncols == data.len()`.

**How to avoid:** For `multi_famm`, require all response dimensions to share the same grid size `m` (validate at entry: `InvalidDimension` if any slice has different `ncols`). If different grids are needed in the future, return `Vec<DenseFlmmResult>` and leave stacking to the caller. Document this constraint.

**Warning signs:** Panic or `FdarError::InvalidDimension` on `FdMatrix::from_column_major` when stacking.

---

### Pitfall 6: `dense_flmm` Random-Slope Scalar LMM

**What goes wrong:** Random slopes require a two-random-effect scalar mixed model per component: `ξ_ijk = x_i' γ + u_i + v_i * z_ij + e_ijk`. The current `fit_scalar_mixed_model` only handles one random effect (`u_i`).

**Why it happens:** `SubjectStructure` and `reml_variance_update` assume a single random intercept per subject.

**How to avoid when `random_slopes = true`:** Add a separate private helper `fit_scalar_mixed_model_with_slopes` in `famm.rs` that handles two variance components (`sigma2_intercept`, `sigma2_slope`). When `random_slopes = false`, call the existing `fit_scalar_mixed_model` unchanged. [ASSUMED]

---

## API Surface

All function signatures and struct layouts come from the PATTERNS.md pre-analysis (read verbatim from `32-PATTERNS.md`). They are reproduced here for planner consumption.

### Config Structs (in `famm.rs`)

```rust
// Config structs: no #[non_exhaustive]; must implement Default
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseFlmmConfig {
    /// Number of FPC components (default: 3)
    pub ncomp: usize,
    /// Maximum REML EM iterations (default: 50)
    pub max_iter: usize,
    /// Relative convergence tolerance for variance components (default: 1e-10)
    pub tol: f64,
    /// Include random slopes in addition to random intercepts (default: false)
    pub random_slopes: bool,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiFammConfig {
    /// FPC components per response dimension (default: 3)
    pub ncomp: usize,
    /// Max REML iterations per component model (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-10)
    pub tol: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FastFmmConfig {
    /// Smoothing window for running-mean post-smoothing (default: 3; 1 = no smoothing)
    pub smooth_window: usize,
    /// Max iterations for each per-point scalar mixed model (default: 30)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// Compute Wald t-statistics and p-values (default: true)
    pub compute_inference: bool,
}
```

### Result Structs (in `famm.rs`)

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct DenseFlmmResult {
    pub mean_function: Vec<f64>,         // length m
    pub beta_functions: FdMatrix,        // p × m
    pub random_effects: FdMatrix,        // n_subjects × m
    pub fitted: FdMatrix,                // n_total × m
    pub residuals: FdMatrix,             // n_total × m
    pub random_variance: Vec<f64>,       // length m (Var_i(b_i(t)))
    pub sigma2_eps: f64,
    pub sigma2_u: Vec<f64>,              // per-component random-intercept variance (length k)
    pub sigma2_slope: Vec<f64>,          // per-component random-slope variance (0s if !random_slopes)
    pub ncomp: usize,
    pub n_subjects: usize,
    pub eigenvalues: Vec<f64>,           // length k
    pub n_iter: usize,                   // REML EM iterations to convergence
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MultiFammResult {
    pub components: Vec<DenseFlmmResult>,  // one per response dimension
    pub stacked_fitted: FdMatrix,          // (n_total × D) × m if all dims share grid
    pub stacked_residuals: FdMatrix,       // (n_total × D) × m
    pub n_dims: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FastFmmResult {
    pub beta_matrix: FdMatrix,    // p × m smoothed fixed-effect functions
    pub t_stats: FdMatrix,        // p × m Wald t-statistics (zeros if !compute_inference)
    pub p_values: FdMatrix,       // p × m pointwise p-values (ones if !compute_inference)
    pub sigma2_eps: Vec<f64>,     // length m, per-gridpoint residual variance
    pub sigma2_u: Vec<f64>,       // length m, per-gridpoint random-intercept variance
    pub n_grid: usize,
}
```

### Public Functions (in `famm.rs`)

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dense_flmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &DenseFlmmConfig,
) -> Result<DenseFlmmResult, FdarError>

#[must_use = "expensive computation whose result should not be discarded"]
pub fn multi_famm(
    data: &[FdMatrix],            // one FdMatrix per response dimension; all must share ncols
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &MultiFammConfig,
) -> Result<MultiFammResult, FdarError>

#[must_use = "expensive computation whose result should not be discarded"]
pub fn fast_fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &FastFmmConfig,
) -> Result<FastFmmResult, FdarError>
```

### `fof_regression.rs` additions

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FofReConfig {
    pub ncomp_x: usize,    // predictor FPC components (default: 3)
    pub ncomp_y: usize,    // response FPC components (default: 3)
    pub max_iter: usize,   // max REML EM iterations (default: 50)
    pub tol: f64,          // convergence tolerance (default: 1e-10)
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FofReResult {
    pub intercept: Vec<f64>,         // length m_y
    pub beta_surface: FdMatrix,      // m_y × m_x
    pub fitted: FdMatrix,            // n_total × m_y
    pub residuals: FdMatrix,         // n_total × m_y
    pub r_squared_t: Vec<f64>,       // length m_y
    pub r_squared: f64,
    pub ncomp_x: usize,
    pub ncomp_y: usize,
    pub fpca_x: FpcaResult,
    pub fpca_y: FpcaResult,
    pub coef_matrix: FdMatrix,       // ncomp_x × ncomp_y
    pub random_effects: FdMatrix,    // n_subjects × m_y
    pub sigma2_u: Vec<f64>,          // per Y-score component (length ncomp_y)
    pub sigma2_eps: f64,
    pub n_subjects: usize,
}

#[must_use = "expensive computation whose result should not be discarded"]
pub fn fof_re_regression(
    x_data: &FdMatrix,
    y_data: &FdMatrix,
    subject_ids: &[usize],
    x_argvals: &[f64],
    y_argvals: &[f64],
    config: &FofReConfig,
) -> Result<FofReResult, FdarError>

#[must_use = "prediction result should not be discarded"]
pub fn predict_fof_re(fit: &FofReResult, new_x: &FdMatrix) -> Result<FdMatrix, FdarError>
// Note: predict_fof_re produces fixed-effect-only prediction (no random effects for
// new subjects — same convention as fmm_predict).
```

### `lib.rs` re-export extensions (lines 233, 242)

```rust
// Line 233 — extend:
pub use famm::{
    fmm, fmm_predict, fmm_test_fixed, FmmResult, FmmTestResult,
    dense_flmm, DenseFlmmResult, DenseFlmmConfig,
    multi_famm, MultiFammResult, MultiFammConfig,
    fast_fmm, FastFmmResult, FastFmmConfig,
};

// Line 242 — extend:
pub use fof_regression::{
    fof_cv, fof_regression, predict_fof, FofCvResult, FofResult,
    fof_re_regression, predict_fof_re, FofReResult, FofReConfig,
};
```

---

## Reuse Map

For each estimator, the exact private functions it calls (all from `famm.rs` unless noted):

| New Function | Calls (private helpers to make `pub(crate)`) | Calls (already public) |
|--------------|----------------------------------------------|------------------------|
| `dense_flmm` | `build_subject_map`, `SubjectStructure::new`, `fit_scalar_mixed_model` (or new slopes variant), `recover_beta_functions`, `recover_random_effects`, `compute_random_variance`, `compute_fitted_residuals` | `fdata_to_pc_1d` (regression.rs) |
| `multi_famm` | none directly (calls `dense_flmm` per dim) | `dense_flmm` (same file, public) |
| `fast_fmm` | `build_subject_map`, `SubjectStructure::new`, `fit_scalar_mixed_model` | `data.column(t)` (matrix.rs) |
| `fof_re_regression` | `build_subject_map` (famm.rs), `fit_scalar_mixed_model` (famm.rs), `recover_random_effects` (famm.rs) | `fdata_to_pc_1d`, `fpca_x.project`, `compute_xtx`, `cholesky_factor`, `cholesky_forward_back` (linalg.rs) |

**pub(crate) changes required in `famm.rs`:**
- `fit_scalar_mixed_model` → `pub(crate)`
- `build_subject_map` → `pub(crate)`
- `SubjectStructure` (struct) → `pub(crate)`
- `SubjectStructure::new` → `pub(crate)`
- `ScalarMixedResult` (struct) → `pub(crate)`
- `recover_random_effects` → `pub(crate)` (needed by `fof_re_regression`)

All others (reml_variance_update, compute_blup, gls_update_gamma, etc.) remain private.

---

## Validation Architecture

> nyquist_validation is enabled (config.json: `"nyquist_validation": true`).

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none — inline tests via `#[cfg(test)] mod tests { ... }` in each source file |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel famm 2>/dev/null` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>/dev/null` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REG-05-A | `dense_flmm` returns correct shape (mean_function len=m, beta_functions p×m, random_effects n_subj×m, fitted n_total×m) | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_basic` | ❌ Wave 0 |
| REG-05-B | `dense_flmm` invariant: fitted + residuals == data within 1e-8 | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_invariant` | ❌ Wave 0 |
| REG-05-C | `dense_flmm` synthetic recovery: fixed-effect function correlates > 0.9 with truth | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_fixed_recovery` | ❌ Wave 0 |
| REG-05-D | `dense_flmm` variance components non-negative and sigma2_u > 0 for grouped data | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_variance_positive` | ❌ Wave 0 |
| REG-05-E | `dense_flmm` invalid input: empty data → `FdarError::InvalidDimension` | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_empty_error` | ❌ Wave 0 |
| REG-05-F | `dense_flmm` invalid input: subject_ids length mismatch → `FdarError::InvalidDimension` | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_ids_mismatch` | ❌ Wave 0 |
| REG-05-G | `dense_flmm` converged field reflects actual convergence | unit | `cargo test -p fdars-core --features linalg,parallel test_dense_flmm_converged` | ❌ Wave 0 |
| REG-05-H | `multi_famm` returns components vec of length D, stacked fitted shape (n_total*D) × m | unit | `cargo test -p fdars-core --features linalg,parallel test_multi_famm_basic` | ❌ Wave 0 |
| REG-05-I | `multi_famm` invalid input: mismatched grid size across dimensions → `InvalidDimension` | unit | `cargo test -p fdars-core --features linalg,parallel test_multi_famm_grid_mismatch` | ❌ Wave 0 |
| REG-05-J | `fast_fmm` returns beta_matrix p×m, t_stats p×m, p_values in [0,1] | unit | `cargo test -p fdars-core --features linalg,parallel test_fast_fmm_basic` | ❌ Wave 0 |
| REG-05-K | `fast_fmm` detects covariate with known effect: beta_matrix norm > 0 | unit | `cargo test -p fdars-core --features linalg,parallel test_fast_fmm_detects_effect` | ❌ Wave 0 |
| REG-05-L | `fast_fmm` invalid input: empty data → error | unit | `cargo test -p fdars-core --features linalg,parallel test_fast_fmm_empty_error` | ❌ Wave 0 |
| REG-05-M | `fof_re_regression` returns correct shapes: beta_surface m_y×m_x, random_effects n_subj×m_y | unit | `cargo test -p fdars-core --features linalg,parallel test_fof_re_regression_dims` | ❌ Wave 0 |
| REG-05-N | `fof_re_regression` invariant: fitted + residuals == y_data within 1e-6 | unit | `cargo test -p fdars-core --features linalg,parallel test_fof_re_regression_invariant` | ❌ Wave 0 |
| REG-05-O | `fof_re_regression` random_effects non-zero when grouped structure present | unit | `cargo test -p fdars-core --features linalg,parallel test_fof_re_regression_re_nonzero` | ❌ Wave 0 |
| REG-05-P | `fof_re_regression` invalid input: subject_ids length mismatch → `InvalidDimension` | unit | `cargo test -p fdars-core --features linalg,parallel test_fof_re_regression_ids_mismatch` | ❌ Wave 0 |
| REG-05-Q | `predict_fof_re` output shape matches n_new × m_y, all finite | unit | `cargo test -p fdars-core --features linalg,parallel test_predict_fof_re_shape` | ❌ Wave 0 |
| REG-05-R | lib.rs re-exports compile: `use fdars_core::{dense_flmm, DenseFlmmResult, ...}` | integration | `cargo test -p fdars-core --features linalg,parallel 2>/dev/null` | ❌ Wave 0 |

### Synthetic Recovery Test Pattern

All four estimators must include at least one synthetic-recovery test following this structure:

```rust
// Generate known-signal data
fn generate_known_mixed_data(
    n_subjects: usize, n_visits: usize, m: usize,
    true_beta_scale: f64, true_sigma2_u: f64,
) -> (FdMatrix, Vec<usize>, FdMatrix, Vec<f64>);

// Test: recovered fixed-effect function correlates with truth
let corr = pearson_correlation(&result.beta_functions.row_to_buf(0), &true_beta);
assert!(corr > 0.8, "Fixed effect recovery correlation {corr} < 0.8");

// Test: variance component order of magnitude correct
assert!(result.sigma2_u[0] > 0.0);
assert!((result.sigma2_u[0] - true_sigma2_u).abs() / true_sigma2_u < 2.0,
        "Variance component off by more than 2x");
```

Tolerance guidance: FPC-score parametrization recovers exact signal up to truncation; for 10 subjects × 3 visits with SNR ≈ 10, expect correlation > 0.85 for the leading component and variance-component ratio within 3× of truth. Use generous tolerances — the goal is catching numerical bugs, not proving statistical consistency. [ASSUMED]

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel famm 2>/dev/null` (only famm tests) or `fof_regression` tests
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel 2>/dev/null`
- **Phase gate (pre-verify):** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps

All test functions listed in the table above are new — none exist today. Wave 0 must create them in `famm.rs` and `fof_regression.rs` inline test modules before implementation begins.

- [ ] `famm.rs #[cfg(test)] mod tests` — add tests for `dense_flmm`, `multi_famm`, `fast_fmm` (REG-05-A through L, G)
- [ ] `fof_regression.rs #[cfg(test)] mod tests` — add tests for `fof_re_regression`, `predict_fof_re` (REG-05-M through Q)
- [ ] `lib.rs` — integration smoke test verifying re-exports compile (REG-05-R)

---

## Security Domain

> `security_enforcement: true`, ASVS level 1 per `config.json`.

This phase is a numeric computation library with no authentication, session management, network I/O, or user-facing input pathways. All inputs are typed Rust structs passed by the calling code. The applicable ASVS categories are limited:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | N/A — library has no auth |
| V3 Session Management | No | N/A |
| V4 Access Control | No | N/A |
| V5 Input Validation | Yes | All public fns validate dimensions + parameter ranges at entry, returning `FdarError` never panicking |
| V6 Cryptography | No | N/A — no secrets, no crypto |
| V7 Error Handling | Yes | All errors return `FdarError`, no unwrap() in public API, no panic on user input |

### Relevant Threat Patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Panic on malformed input (0×0 matrix, length mismatch) | Denial of Service | Validate at function entry → `FdarError::InvalidDimension` |
| Integer overflow in index arithmetic (very large n × m) | Tampering | `usize` arithmetic is checked in debug mode; column-major index `row + col * nrows` is bounded by `FdMatrix::from_column_major` validation |
| Non-positive-definite Cholesky (collinear FPC scores) | Availability | `cholesky_factor` returns `ComputationFailed`; `dense_flmm` propagates with `?` |
| Divergent REML EM (sigma2 → 0) | Availability | `reml_variance_update` clamps: `.max(1e-15)` [VERIFIED: fdars-core/src/famm.rs:428-430] |

No new threat surface compared to existing `fmm`/`fof_regression`.

---

## Environment Availability

Step 2.6: SKIPPED — this phase makes only source-code changes to an existing Rust crate. The only external dependency is the existing Rust toolchain and Cargo, already confirmed present by the project being buildable (recent Phase 31 completed successfully). No new external tools, services, CLIs, databases, or package managers are required.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Basis-coefficient parametrization (R's denseFLMM/mgcv) | FPC-score parametrization (fdars choice since fmm was introduced) | Phase 32 design decision (2026-08-20) | Simpler implementation; reuses existing fdata_to_pc_1d; slight loss of model flexibility vs basis penalties |
| Per-gridpoint fit with mgcv smoothing (fastFMM R) | Per-gridpoint fit with running-mean smoother (fdars) | Phase 32 design decision | Avoids mgcv dependency; less smooth coefficient functions; document in rustdoc |
| pffr mgcv-GAMM backend (refund) | Double-FPCA with per-score-component scalar LMM (fdars) | Phase 32 design decision | Consistent with existing FoF implementation; no penalty tuning needed |
| `fmm` (intercept-only random effects) | `dense_flmm` (intercept + optional slope random effects) | This phase | Adds expressiveness; `fmm` stays unchanged |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Random-slope REML update requires a new helper `fit_scalar_mixed_model_with_slopes`; the existing single-u updater cannot be extended in place | Reference Formulations (denseFLMM) | Low risk: even if it can be extended in place, the change is small; worst case is a slightly different helper factoring |
| A2 | Per-dimension independent FPCA for `multi_famm` (not joint multivariate FPCA) captures sufficient variation for the fdars use case | Reference Formulations (multiFAMM) | Medium risk: cross-dimension covariances are not modeled; users needing those should be warned in rustdoc |
| A3 | Running-mean smoothing (window=3) is sufficient for `fast_fmm` post-smoothing; users can set `smooth_window=1` to disable | Common Pitfalls (Pitfall 3) | Low risk: produces noisier coefficient estimates than mgcv; acceptable for the capability-faithful target |
| A4 | Synthetic-recovery tolerance for variance components: within 3× of truth at n=10 subjects × 3 visits | Validation Architecture | Low risk: generous tolerance; tests may pass even with imperfect estimation |
| A5 | `stacked_fitted` in `MultiFammResult` stacks all D dimensions row-wise; requires all dimensions to share the same `m` (grid size) | API Surface | Medium risk: if a user needs different grids, the stacked matrix approach fails; documented constraint |
| A6 | `pub(crate)` promotion of `fit_scalar_mixed_model`, `build_subject_map`, `SubjectStructure`, `ScalarMixedResult`, `recover_random_effects` is sufficient for cross-module reuse; no other private helper is needed by `fof_re_regression` | Reuse Map | Low risk: confirmed by tracing the call graph through `famm.rs` |

---

## Open Questions

1. **Random-slope variance separate from intercept variance in `DenseFlmmResult`?**
   - What we know: `FmmResult` has `sigma2_u: Vec<f64>` (per-component intercept variance). A random-slope model adds `sigma2_slope: Vec<f64>`.
   - What's unclear: Whether the planner should add `sigma2_slope` as a always-present field (zero-filled when `random_slopes=false`) or as an `Option<Vec<f64>>`.
   - Recommendation: Always-present, zero-filled when `random_slopes=false` — simpler API, consistent with fdars convention of no `Option` in result fields.

2. **fastFMM smoothing: running-mean vs Savitzky-Golay?**
   - What we know: Both are implementable without new deps. Savitzky-Golay better preserves peaks.
   - What's unclear: Whether the polynomial SG filter justifies the implementation effort over a simple running mean.
   - Recommendation: Implement running-mean (O(m) trivial), expose `smooth_window` config. Note SG as a future improvement in rustdoc.

---

## Code Examples

### Pattern 1: Per-Component Parallel Loop (canonical from `famm.rs`)

```rust
// Source: fdars-core/src/famm.rs:228-235 [VERIFIED]
let per_comp: Vec<ScalarMixedResult> = iter_maybe_parallel!(0..k)
    .map(|comp| {
        let comp_scores: Vec<f64> = (0..n_total)
            .map(|i| scores[(i, comp)] * score_scale)
            .collect();
        fit_scalar_mixed_model(&comp_scores, subject_map, n_subjects, covariates, p)
    })
    .collect();
```

### Pattern 2: Column Access for Per-Gridpoint Fit

```rust
// fast_fmm: per-gridpoint scalar mixed model
// data.column(t) returns &[f64] — contiguous in column-major layout
let per_point: Vec<PointwiseResult> = iter_maybe_parallel!(0..m)
    .map(|t| {
        let y_t: Vec<f64> = data.column(t).to_vec();
        fit_scalar_mixed_model(&y_t, subject_map, n_subjects, covariates, p)
    })
    .collect();
```

### Pattern 3: Reconstruct Functional Curves from Score Coefficients

```rust
// Source: fdars-core/src/famm.rs:652-691 [VERIFIED]
// recover_beta_functions: beta[j, t] = Σ_k gamma[j][k] * rotation[(t, k)]
// recover_random_effects: re[s, t] = Σ_k u_hat[s][k] * rotation[(t, k)]
let beta_functions = recover_beta_functions(&gamma, &fpca.rotation, p, m, k);
let random_effects = recover_random_effects(&u_hat, &fpca.rotation, n_subjects, m, k);
```

### Pattern 4: Entry Validation (exact form from `famm.rs`)

```rust
// Source: fdars-core/src/famm.rs:94-115 [VERIFIED]
let n_total = data.nrows();
let m = data.ncols();
if n_total == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "non-empty matrix".to_string(),
        actual: format!("{n_total} x {m}"),
    });
}
if subject_ids.len() != n_total {
    return Err(FdarError::InvalidDimension {
        parameter: "subject_ids",
        expected: format!("length {n_total}"),
        actual: format!("length {}", subject_ids.len()),
    });
}
```

### Pattern 5: Config Struct Default (exact form from `pace_fpca.rs`)

```rust
// Source: fdars-core/src/pace_fpca.rs — PaceFpcaConfig pattern [VERIFIED: 32-PATTERNS.md]
impl Default for DenseFlmmConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            max_iter: 50,
            tol: 1e-10,
            random_slopes: false,
        }
    }
}
```

### Pattern 6: `pub(crate)` Visibility Change

```rust
// In famm.rs — change exactly these fn/struct declarations:
pub(crate) fn fit_scalar_mixed_model(...) -> ScalarMixedResult { ... }
pub(crate) fn build_subject_map(...) -> (Vec<usize>, usize) { ... }
pub(crate) struct SubjectStructure { ... }
pub(crate) struct ScalarMixedResult { ... }
pub(crate) fn recover_random_effects(...) -> FdMatrix { ... }
// All others remain private (fn, not pub(crate))
```

---

## Sources

### Primary (codebase — VERIFIED by Read this session)
- `fdars-core/src/famm.rs` (lines 1–1395) — complete existing fmm/fmm_test_fixed implementation; all private helpers
- `fdars-core/src/fof_regression.rs` (lines 1–687) — complete base FoF implementation
- `fdars-core/src/linalg.rs` (lines 1–152) — Cholesky helpers
- `fdars-core/src/regression.rs` (lines 1–120) — `FpcaResult`, `fdata_to_pc_1d`
- `fdars-core/src/matrix.rs` (lines 1–80) — column-major layout
- `.planning/phases/32-flexible-mixed-effects-regression/32-PATTERNS.md` — pre-analyzed API surface and analog patterns
- `.planning/phases/32-flexible-mixed-effects-regression/32-CONTEXT.md` — locked decisions
- `fdars-core/src/lib.rs` (grep lines 86, 88, 233, 242) — existing re-exports

### Secondary (web — MEDIUM confidence for architectural context)
- [denseFLMM RDocumentation](https://www.rdocumentation.org/packages/denseFLMM/versions/0.1.3/topics/denseFLMM) — model formulation Y_i(t_d) = μ + z_i^T U(t_d) + ε
- [multiFAMM paper (SAGEPUB)](https://journals.sagepub.com/doi/10.1177/1471082X211056158) — stacking approach, shared vs per-response RE
- [fastFMM CRAN](https://cran.r-project.org/web/packages/fastFMM/index.html) — FUI three-step method (Cui et al. 2022)
- [refund pcre](https://rdrr.io/cran/refund/man/pcre.html) — FPCA-based functional random intercepts in pffr
- [FAMM PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC4560367/) — FAMM general formulation with tensor-product representation

### Tertiary (LOW confidence)
- WebSearch: Henderson MME, REML EM update equations — general LMM background

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all reused from verified in-repo code
- Architecture: HIGH — derived from direct Read of famm.rs + fof_regression.rs
- Reference formulations: MEDIUM — denseFLMM/multiFAMM/fastFMM math from web sources; divergences from R documented
- Pitfalls: HIGH — derived from reading actual code; Pitfall 1, 2, 4 verified from source
- Test strategy: HIGH — mirrors existing test patterns in famm.rs

**Research date:** 2026-08-20
**Valid until:** 2026-09-20 (stable algorithms; API divergences from R documented)
