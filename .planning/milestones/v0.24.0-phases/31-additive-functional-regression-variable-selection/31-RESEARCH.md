# Phase 31: Additive Functional Regression & Variable Selection — Research

**Researched:** 2026-08-20
**Domain:** Nonparametric additive scalar-on-function regression (FAM, GKAM, GSAM, variable selection, permutation test, history-index estimator)
**Confidence:** MEDIUM (core math pinned from authoritative sources; Rust mapping derived from verified in-repo code)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Match R baselines (`fdapace` FAM / `fda.usc` GKAM+GSAM / `refund` fosr.vs+fosr.perm+history-index) **by capability**, not by exact R signature.
- Document any divergence from the R reference formulation in rustdoc (established milestone convention).
- Pin the exact backfitting / kernel / spectral construction and the group-penalty for `variable_selection` during plan-phase research.
- Config structs for the complex estimators (e.g. `FamConfig`, `GkamConfig`/`GsamConfig`, `VarSelectConfig`, `HistoryIndexConfig`) following the builder-config convention (`GmmClusterConfig`, `ElasticConfig`, etc.), with serde behind the `serde` feature.
- Structured immutable `Result` types (e.g. `FamResult`, `VarSelectResult`) deriving `Debug, Clone, PartialEq`, carrying scores/fitted/diagnostics for reproducibility.
- Parameter ordering follows `(data, y, [argvals,] [scalar_covariates,] config)`.
- Full multi-functional-covariate support for FAM backfitting.
- Scalar covariates supported in the same functions (no separate overloads), per convention.
- Synthetic-recovery tests (fit on data generated from a known additive structure, check recovery within tolerance).
- Known-property invariants (e.g. additive decomposition sums, permutation-null centering).
- Seeded-permutation reproducibility for the permutation-test wrapper (`StdRng::seed_from_u64(seed + k)` pattern; mirror INF-01's 999-perm default).

### Claude's Discretion
- Exact config field names, default bandwidths/component counts, internal helper factoring, and the precise number of test cases are at Claude's discretion within the above.

### Deferred Ideas (OUT OF SCOPE)
- Plotting/rendering of additive fits (out of scope — numeric outputs only).
- Boosting/Bayesian functional additive regression (REG-06, deferred).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REG-04 | Add nonparametric additive functional regression in new `scalar_on_function/additive.rs` — FAM (backfitting over FPC-score components), GKAM, GSAM, `variable_selection` helper, permutation-test wrapper, and history-index estimator. Reuses `smoothing.rs` kernels + `fdata_to_pc_1d`. Additive/non-breaking. | All six estimators mathematically pinned; reuse map established; API surface specified. |
</phase_requirements>

---

## Summary

Phase 31 adds six nonparametric additive scalar-on-function regression estimators to fdars-core in a new file `scalar_on_function/additive.rs`. All six reuse the existing `smoothing.rs` kernel machinery (`nadaraya_watson`, `local_linear`, `local_polynomial`, `optim_bandwidth`) and `regression.rs::fdata_to_pc_1d` (FPCA scores/loadings). No new crate dependency is required.

The most important mathematical insight is that FAM (Müller & Yao 2008) does **not** require iterative backfitting over correlated components. Because functional principal component scores are uncorrelated (independent under Gaussianity), fitting each additive component `f_k(ξ_k)` reduces to independent 1D kernel regressions of the partial residual on the k-th score — one pass suffices. GKAM differs from FAM in that it operates on functional L2 distances (not FPC scores) and uses an iterative local-scoring algorithm (true backfitting over the hat matrices H_1+…+H_q), correctly capturing non-FPC-based kernel structure. GSAM is effectively GKAM but uses the FPC score basis for constructing the additive smooth terms (closest to FAM in spirit, furthest from kernel-on-L2-distance). The history-index estimator is a scalar-on-lagged-window model that discretizes the lag grid and regresses Y on the kernel-smoothed history integral. The variable-selection helper uses a group-penalized iterative algorithm (group lasso / group MCP / group SCAD). The permutation-test wrapper mirrors the pattern already established in `famm.rs::fmm_test_fixed`.

**Primary recommendation:** Implement FAM as parallel independent 1D NW regressions on FPC scores (no backfitting loop needed); implement GKAM as true iterative backfitting over NW hat matrices on functional L2 distances; implement GSAM as FPCA-then-additive-smooth (GAM-equivalent in score space); implement variable_selection as group-penalized coordinate descent; implement the permutation wrapper following `famm.rs`; implement history-index as a lagged-window NW smoother.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FAM estimation | `scalar_on_function/additive.rs` | `smoothing.rs`, `regression.rs` | Additive component estimation via 1D NW on FPC scores |
| GKAM estimation | `scalar_on_function/additive.rs` | `smoothing.rs`, `scalar_on_function/nonparametric.rs` | Kernel hat-matrix backfitting over functional L2 distances |
| GSAM estimation | `scalar_on_function/additive.rs` | `regression.rs`, `smoothing.rs` | FPC score basis + additive smooth |
| Variable selection | `scalar_on_function/additive.rs` | — | Group-penalized coordinate descent in score space |
| Permutation test | `scalar_on_function/additive.rs` | `famm.rs` (pattern reference) | Seeded permutation over FAM/GKAM/GSAM models |
| History-index | `scalar_on_function/additive.rs` | `smoothing.rs` | Lagged-window NW regression |
| RNG seeding | `parallel.rs` + `rand` crate | — | `StdRng::seed_from_u64(seed + k)` per existing convention |

---

## Standard Stack

### Core (all already present — no new dependencies)

| Library | Version | Purpose | Already in Cargo.toml |
|---------|---------|---------|----------------------|
| `fdars-core::smoothing` | internal | NW, local-linear, local-poly, bandwidth selection | Yes [VERIFIED: fdars-core/src/smoothing.rs:1-889] |
| `fdars-core::regression::fdata_to_pc_1d` | internal | FPCA scores and rotation for FAM/GSAM | Yes [VERIFIED: fdars-core/src/regression.rs:250-350] |
| `fdars-core::matrix::FdMatrix` | internal | Column-major functional data matrix | Yes [VERIFIED: fdars-core/src/matrix.rs:1-80] |
| `rand 0.8` / `rand_distr 0.4` | 0.8/0.4 | Seeded RNG for permutation tests | Yes (Cargo.toml) |
| `fdars-core::error::FdarError` | internal | Error variants: `InvalidDimension`, `InvalidParameter`, `ComputationFailed` | Yes [VERIFIED: fdars-core/src/error.rs:1-51] |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `iter_maybe_parallel!` macro | internal | Feature-gated rayon parallelism | In FAM outer loop (per-component smooth is independent) |
| `smoothing::smoothing_matrix_nw` | internal | NW hat matrix for GKAM hat matrix construction | GKAM hat matrix sum H_Q = H_1 + … + H_q |
| `helpers::simpsons_weights` | internal | Integration weights for history-index integral | History-index window integral discretization |
| `scalar_on_function::nonparametric::compute_pairwise_distances` | internal (pub(super)) | L2 distance matrix for GKAM | GKAM functional kernel |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Independent 1D NW for FAM | True iterative backfitting | FAM FPC scores are uncorrelated → independent 1D fits are identical to backfitting but O(K) not O(K²·iter) |
| Group lasso for variable_selection | Group MCP / SCAD | Group lasso is convex, simpler to implement without new dependencies; MCP/SCAD have unbiasedness advantages but require more complex solvers |
| NW hat matrix for GKAM | Local-linear hat matrix | Local-linear corrects boundary bias but the hat matrix is non-trivial for backfitting; NW matches fda.usc baseline |

**Installation:** No new packages. All required machinery is already compiled into fdars-core.

---

## Package Legitimacy Audit

> No external packages are being added in this phase. All required libraries are already in the fdars-core dependency tree. This section is not applicable.

**Packages added:** none.

---

## Architecture Patterns

### System Architecture Diagram

```
FdMatrix(n×m) ──→ fdata_to_pc_1d ──→ FpcaResult
    │                                      │
    │                                ξ (n×K scores)
    │                                      │
    │        ┌─────────────────────────────┤
    │        │ FAM: for each k in 1..K     │
    │        │   partial_resid = y - intercept - sum_{j≠k} f_j_hat│
    │        │   f_k_hat = nadaraya_watson(ξ[:,k], partial_resid)  │
    │        │ (1 pass suffices: FPCs uncorrelated)                │
    │        └─────────────→ FamResult ────────────────────────────┘
    │
    ├──→ compute_pairwise_distances (L2) ──→ NW hat matrices H_k
    │        │ GKAM backfitting:
    │        │   init f_k = 0
    │        │   repeat until convergence:
    │        │     eta_k = y_adj - sum_{j≠k} f_j
    │        │     f_k = H_k · eta_k  (NW smoother application)
    │        └─────────────→ GkamResult
    │
    ├──→ fdata_to_pc_1d (score basis) ──→ additive smooth on ξ columns
    │        │ GSAM: same as FAM but with optional link function
    │        └─────────────→ GsamResult
    │
    ├──→ FPC scores ──→ group-penalized coordinate descent
    │        │ VarSelect: iteratively update β groups, apply group threshold
    │        └─────────────→ VarSelectResult
    │
    ├──→ [fit FAM/GKAM/GSAM] ──→ observed statistic
    │        │ PermTest: shuffle y, refit, compare n_perm times
    │        └─────────────→ PermTestResult
    │
    └──→ lagged window X[i, t-Δ..t] ──→ nadaraya_watson on history index
              │ HistoryIndex: integral_0^Δ γ(u) X(t-u) du via lag grid
              └─────────────→ HistoryIndexResult
```

### Recommended Project Structure

```
fdars-core/src/scalar_on_function/
├── additive.rs          # NEW — all six estimators + config/result types
├── mod.rs               # ADD: pub use additive::*;
└── ...existing files unchanged
fdars-core/src/
└── lib.rs               # ADD: crate-root re-exports for new public items
```

### Pattern 1: FAM — Independent 1D NW Per FPC Score

**What:** Additive model `E(Y|X) = μ_Y + Σ_k f_k(ξ_k)` where ξ_k are FPC scores. Because FPC scores are uncorrelated (orthogonal in L2, independent under Gaussianity), fitting is NOT iterative backfitting — one sequential pass of 1D NW smoothers suffices.

**Algorithm:**
1. Run `fdata_to_pc_1d(data, ncomp, argvals)` → get ξ (n×K score matrix) and `FpcaResult`.
2. Compute `mu_y = mean(y)`.
3. For k = 0..K: fit `f_k_hat = nadaraya_watson(ξ[:,k], partial_resid_k, ξ[:,k], h_k, kernel)` where `partial_resid_k = y - mu_y - Σ_{j≠k} f_j_hat(ξ[:,j])`.
4. Since FPC scores are uncorrelated, a single forward pass (k=0..K) achieves convergence without iteration.
5. Fitted: `ŷ_i = mu_y + Σ_k f_k_hat(ξ_{ik})`.
6. Bandwidth: use `optim_bandwidth` (GCV) per component, or caller-supplied uniform h.

**When to use:** Multi-functional-covariate additive model, nonparametric, no distributional assumption beyond smoothness.

**R baseline divergence (document in rustdoc):** R's `fdapace::FAM` uses PACE for FPC estimation; fdars uses `fdata_to_pc_1d` (nalgebra SVD with Simpson's weights). R returns component-specific bandwidths selected by GCV; fdars follows the same approach but uses its existing `optim_bandwidth` grid search. No backfitting iteration is used in either implementation because FPC uncorrelatedness eliminates the need. [ASSUMED — R source not read this session]

**Example:**
```rust
// Source: derived from smoothing.rs::nadaraya_watson + regression.rs::fdata_to_pc_1d
// (both VERIFIED this session)
let fpca = fdata_to_pc_1d(&data, config.ncomp, argvals)?;
let scores = &fpca.scores; // n×K
let mu_y: f64 = y.iter().sum::<f64>() / y.len() as f64;
let mut component_fits: Vec<Vec<f64>> = vec![vec![0.0; n]; ncomp];
for k in 0..ncomp {
    let xi_k: Vec<f64> = (0..n).map(|i| scores[(i, k)]).collect();
    let partial: Vec<f64> = (0..n).map(|i| {
        y[i] - mu_y - (0..ncomp).filter(|&j| j != k)
            .map(|j| component_fits[j][i]).sum::<f64>()
    }).collect();
    let h = optim_bandwidth(&xi_k, &partial, None, CvCriterion::Gcv, kernel, 20).h_opt;
    component_fits[k] = nadaraya_watson(&xi_k, &partial, &xi_k, h, kernel)?;
}
```

### Pattern 2: GKAM — Iterative Hat-Matrix Backfitting on Functional L2 Distances

**What:** For q functional covariates X^1, …, X^q, fit `ŷ = g^{-1}(Σ_k f_k(X^k))` where each `f_k` is estimated by NW regression using L2 distance kernel between curves. Unlike FAM, the predictor distances are not orthogonal, so true iterative backfitting (Opsomer-Ruppert approach) is required.

**Algorithm:**
1. For each covariate k: compute `dist_k[i,j] = L2_distance(X^k_i, X^k_j)` using Simpson's weights.
2. Initialize `f_k = 0` for all k.
3. Iterate until convergence (`|Δf| < ε`, max `maxit` iterations):
   a. For each k: `adjusted_k[i] = g(y[i]) - Σ_{j≠k} f_j[i]` (working response minus other components).
   b. `f_k[i] = Σ_j K_h(dist_k[i,j]) · adjusted_k[j] / Σ_j K_h(dist_k[i,j])` (NW on L2 distances).
4. Final: `ŷ = g^{-1}(μ + Σ_k f_k)`.
5. Bandwidth per covariate: LOO-CV on the distance matrix (same as `fregre_np_mixed`).

**Key distinction from FAM:** GKAM uses L2 distances between full curves (not FPC scores) and requires iteration because the kernel smoothers on different curves are not orthogonal.

**R baseline divergence (document in rustdoc):** R's `fregre.gkam` constructs explicit smoother matrices H_k and solves the composite H_Q = H_1 + … + H_q system; fdars implements the equivalent iterative update without materializing the full n×n hat matrix (avoids O(n²) memory for large n). Link functions: only Gaussian identity is implemented in the initial Rust version; logit/log require IRLS wrapping (document as known gap). [ASSUMED]

**Example:**
```rust
// Per-covariate NW smoother on L2 distance matrix — O(n²) per covariate per iteration
let dist_k = compute_pairwise_distances(data_k, argvals);
let f_k: Vec<f64> = (0..n).map(|i| {
    let mut num = 0.0; let mut den = 0.0;
    for j in 0..n {
        let w = (-dist_k[i*n+j].powi(2) / (2.0*h*h)).exp();
        num += w * adjusted[j]; den += w;
    }
    if den > 1e-15 { num / den } else { adjusted[i] }
}).collect();
```

### Pattern 3: GSAM — FPC Score Basis + Additive Smooth (GAM-Equivalent)

**What:** `E[Y|X,Z] = g^{-1}(α + Σ_i f_i(Z_i) + Σ_k Σ_j f_j^k(ξ_j^k))` where `ξ_j^k` are FPC scores of X^k. GSAM is structurally identical to FAM but with: (a) an optional link function g(·), (b) optional non-functional scalar covariates Z, and (c) smooth functions `f_j^k` replacing the simple NW smoother (GAM spline-smoothed in R via `mgcv`; in Rust: NW or local-polynomial on score columns).

**R baseline divergence (document in rustdoc):** R's `fregre.gsam` delegates to `mgcv::gam` internally, which uses penalized regression splines. Rust implements the equivalent with Nadaraya-Watson smoothing on FPC scores (same model class, different smoother). For the Gaussian identity case (default), the two implementations produce numerically equivalent fits in the limit of small bandwidth / large n. For link functions other than identity, only `GlmFamily::Gaussian` (identity link) is implemented in the initial Rust version; others require IRLS wrapping (document). [ASSUMED]

**Algorithm:** Identical to FAM, with optional link-function inversion applied to `adjusted_k` prior to NW smoothing (one-step approximation, no IRLS). Functionally a superset of FAM.

### Pattern 4: Variable Selection — Group-Penalized Coordinate Descent

**What:** Select which functional covariates are active via group-penalized regression on FPC scores. Groups are defined by covariate: all K FPC score columns for predictor p form group p.

**Algorithm (group lasso as primary path):**
1. Run `fdata_to_pc_1d` per predictor → score matrices ξ^1, …, ξ^P (each n×K_p).
2. Form design matrix X = [1 | ξ^1_1..ξ^1_{K_1} | … | ξ^P_1..ξ^P_{K_P} | Z] (scalar covariates appended).
3. Iteratively update per-group coefficients with group-lasso soft-threshold:
   - For group g with coefficient vector `β_g`:
   - `β_g_ols = (X_g'X_g)^{-1} X_g'(y - Σ_{p≠g} X_p β_p)` (partial residual OLS).
   - Group-lasso update: `β_g = β_g_ols · max(0, 1 - λ·√K_g / ||β_g_ols||)`.
4. Iterate until convergence; sweep through all groups.
5. Active predictors: those with `||β_g|| > ε_threshold`.

**Penalty options (VarSelectPenalty enum):**
- `GroupLasso` — convex, well-understood convergence, primary implementation.
- `GroupMcp` — minimax concave penalty (less bias than lasso), secondary.
- `GroupScad` — SCAD penalty (oracle property), secondary.

**λ selection:** Grid search via CV (LOO or k-fold).

**R baseline divergence (document in rustdoc):** R's `fosr.vs` is function-on-scalar (functional response), whereas fdars implements scalar-on-function variable selection (scalar response, functional predictors). The group penalty formulation is analogous but the regression direction differs. Document this clearly. [ASSUMED]

### Pattern 5: Permutation Test

**What:** Significance test for any additive model. Permutes y, refits the model, computes a test statistic, accumulates null distribution. Follows the pattern from `famm.rs::fmm_test_fixed` exactly.

**Algorithm:**
1. Fit the selected model (FAM/GKAM/GSAM) on original (data, y) → observed test statistic T_obs (e.g. R², mean squared fitted value, or sum of integrated component norms).
2. For k in 0..n_perm:
   - Shuffle y using `StdRng::seed_from_u64(seed + k as u64)`.
   - Refit model on (data, y_perm) → T_perm.
   - `n_ge += (T_perm >= T_obs) as usize`.
3. `p_value = (n_ge + 1) / (n_perm + 1)`.
4. Default: 999 permutations, caller-supplied seed.

**Test statistic choices (caller selects via `PermTestStatistic` enum):**
- `R2` — R² of the full additive fit.
- `FittedNorm` — L2 norm of fitted values.
- `ComponentNorm` — sum of integrated component norms (FAM only).

### Pattern 6: History-Index Estimator

**What:** Scalar response Y depends on the recent history of X in a window of length Δ:
`E{Y_i} = β_0 + β_1 · (1/Δ) · Σ_{u=0}^{Δ} γ(u) · X_i(T - u) · Δu`
where γ(·) is the history-index weight function (estimated via NW on the lag grid).

**Algorithm:**
1. Discretize lag grid: `lag_vals = [0, Δu, 2Δu, …, Δ]` with `Δu = Δ / n_lags`.
2. For each observation i: extract lagged covariate values `x_lag[i, l] = X_i[T - lag_vals[l]]` (index into `FdMatrix` columns; requires `argvals` alignment).
3. Estimate γ: compute `gamma_hat[l] = nadaraya_watson(lag_vals, y - beta_0_hat, lag_vals, h_gamma, kernel)` — this is the marginal weight function estimated by NW on the lag axis.
4. Compute history index score for each observation: `score_i = Σ_l gamma_hat[l] · x_lag[i,l] · Δu`.
5. Final estimate: `E{Y_i} = β_0 + β_1 · score_i` via OLS on `(1, score_i)`.
6. Identifiability: normalize `gamma` so `Σ_l gamma[l]² · Δu ≈ 1` (discrete approximation to `∫ γ² = 1`).

**R baseline divergence (document in rustdoc):** R's `pffr ff(..., limits)` implements the full function-on-function history model as a bivariate smooth with lower-triangular (s ≤ t) constraint using penalized B-splines. Fdars implements the scalar-on-function reduction (Y scalar, history index evaluated at a single time T = argvals.last()) via NW smoothing over a discretized lag grid — same model class but a marginal-integration approximation rather than bivariate spline. [ASSUMED]

### Anti-Patterns to Avoid

- **Iterating FAM over correlated FPC scores:** FPC scores are orthogonal by construction — one sequential pass is mathematically equivalent to infinite-iteration backfitting. Iterating wastes compute with no accuracy gain.
- **Materializing the full n×n hat matrix for large n in GKAM:** Hat matrix construction is O(n²). For GKAM, apply the NW weights directly (O(n) per prediction point) rather than building `smoothing_matrix_nw` per covariate and multiplying.
- **Using `smoothing_matrix_nw` inside the GKAM convergence loop:** `smoothing_matrix_nw` returns a flat `n*n` Vec. Call it once per covariate per outer bandwidth-selection, but do NOT call it inside the inner backfitting loop — apply NW weights directly.
- **Allocating a new FdMatrix per permutation:** In the permutation test, reuse the same score/distance buffers and only shuffle `y`. Creating `n` new matrices per permutation is O(n_perm × n × m) memory.
- **Sharing the same `StdRng` across rayon threads:** If parallelizing permutations, seed each thread independently with `seed + k as u64` (existing fdars pattern).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Kernel smoothing | Custom NW smoother | `smoothing::nadaraya_watson` / `local_linear` | Already implemented, tested, bandwidth-selectable |
| Bandwidth selection | Custom grid search | `smoothing::optim_bandwidth` with `CvCriterion::Gcv` | GCV/CV/AIC already implemented with correct smoother matrix |
| FPCA / FPC scores | Custom SVD pipeline | `regression::fdata_to_pc_1d` | Returns `FpcaResult` with scores, rotation, weights, mean |
| Integration weights | Custom quadrature | `helpers::simpsons_weights` | Used throughout codebase, correct Simpson's rule |
| L2 functional distances | Custom loop | `scalar_on_function::nonparametric::compute_pairwise_distances` | Uses simpsons_weights internally, symmetric, tested |
| Matrix solve | Custom Gaussian elimination | `smoothing::solve_gaussian_pub` | Public wrapper already exported |
| Seeded RNG | `rand::thread_rng()` | `StdRng::seed_from_u64(seed + k as u64)` | Reproducibility requirement; thread_rng is non-deterministic |

**Key insight:** The entire numeric primitive stack for additive regression already exists in fdars. Phase 31 assembles these into six new public functions; it adds zero numeric algorithms.

---

## Reuse Map (Exact Function Calls Per Estimator)

| Estimator | `smoothing.rs` functions called | `regression.rs` functions called | `nonparametric.rs` functions called |
|-----------|--------------------------------|----------------------------------|-------------------------------------|
| FAM | `nadaraya_watson`, `optim_bandwidth` | `fdata_to_pc_1d` | — |
| GKAM | NW-on-distance (inline kernel loop; `smoothing_matrix_nw` for hat matrix option) | — | `compute_pairwise_distances`, `gaussian_kernel` |
| GSAM | `nadaraya_watson`, `optim_bandwidth` | `fdata_to_pc_1d` | — |
| variable_selection | — | `fdata_to_pc_1d` | — |
| permutation_test | (delegates to FAM/GKAM/GSAM internally) | (delegates to above) | (delegates to above) |
| history_index | `nadaraya_watson`, `optim_bandwidth` | — | — |

All source functions verified to exist and have the expected signatures this session. [VERIFIED: fdars-core/src/smoothing.rs:72-129, 160-230, 715-761], [VERIFIED: fdars-core/src/regression.rs:250-350], [VERIFIED: fdars-core/src/scalar_on_function/nonparametric.rs:10-47]

---

## API Surface

### Public Functions

```rust
// FAM: Functional Additive Model (Müller & Yao 2008)
#[must_use = "expensive computation"]
pub fn fam(
    data: &FdMatrix,           // n×m functional predictor
    y: &[f64],                 // scalar response, length n
    argvals: &[f64],           // evaluation grid, length m
    scalar_covariates: Option<&FdMatrix>,  // n×p optional scalar covariates
    config: &FamConfig,
) -> Result<FamResult, FdarError>

// GKAM: Generalized Kernel Additive Model (fda.usc::fregre.gkam)
#[must_use = "expensive computation"]
pub fn fregre_gkam(
    predictors: &[&FdMatrix],  // q functional predictors, each n×m_k
    y: &[f64],
    argvals_list: &[&[f64]],   // evaluation grid per predictor
    scalar_covariates: Option<&FdMatrix>,
    config: &GkamConfig,
) -> Result<GkamResult, FdarError>

// GSAM: Generalized Spectral Additive Model (fda.usc::fregre.gsam)
#[must_use = "expensive computation"]
pub fn fregre_gsam(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    config: &GsamConfig,
) -> Result<GsamResult, FdarError>

// Variable selection with group penalty
#[must_use = "expensive computation"]
pub fn variable_selection(
    predictors: &[&FdMatrix],  // P functional predictors, each n×m_p
    y: &[f64],
    argvals_list: &[&[f64]],
    scalar_covariates: Option<&FdMatrix>,
    config: &VarSelectConfig,
) -> Result<VarSelectResult, FdarError>

// Permutation test wrapper
#[must_use = "expensive computation"]
pub fn permutation_test_fam(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    config: &FamConfig,
    perm_config: &PermTestConfig,
) -> Result<PermTestResult, FdarError>

// History-index estimator
#[must_use = "expensive computation"]
pub fn history_index(
    data: &FdMatrix,           // n×m predictor curves (observed over full time domain)
    y: &[f64],
    argvals: &[f64],           // evaluation grid, length m
    config: &HistoryIndexConfig,
) -> Result<HistoryIndexResult, FdarError>
```

### Config Structs

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FamConfig {
    pub ncomp: usize,          // 0 = auto-select via GCV (default: 0)
    pub bandwidth: f64,        // 0.0 = auto-select per component via GCV (default: 0.0)
    pub kernel: String,        // "gaussian" | "epanechnikov" | "tricube" (default: "gaussian")
    pub n_grid_bandwidth: usize, // bandwidth grid size for optim_bandwidth (default: 20)
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GkamConfig {
    pub bandwidth: f64,        // per-covariate h; 0.0 = LOO-CV auto (default: 0.0)
    pub kernel: String,        // default: "gaussian"
    pub max_iter: usize,       // convergence iterations (default: 50)
    pub epsilon: f64,          // convergence threshold (default: 1e-6)
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GsamConfig {
    pub ncomp: usize,          // FPC components; 0 = auto (default: 0)
    pub bandwidth: f64,        // 0.0 = auto per component (default: 0.0)
    pub kernel: String,        // default: "gaussian"
    pub n_grid_bandwidth: usize, // default: 20
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum VarSelectPenalty { GroupLasso, GroupMcp, GroupScad, Ls }

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarSelectConfig {
    pub ncomp: usize,              // FPC components per predictor; 0 = auto (default: 3)
    pub penalty: VarSelectPenalty, // default: GroupLasso
    pub lambda: f64,               // penalty weight; 0.0 = CV-select (default: 0.0)
    pub max_iter: usize,           // coordinate descent iterations (default: 100)
    pub epsilon: f64,              // convergence threshold (default: 1e-5)
    pub lambda_n_grid: usize,      // grid size for lambda selection (default: 20)
}

#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum PermTestStatistic { R2, FittedNorm, ComponentNorm }

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PermTestConfig {
    pub n_perm: usize,             // default: 999
    pub seed: u64,                 // default: 42
    pub statistic: PermTestStatistic, // default: R2
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoryIndexConfig {
    pub window: f64,               // Δ — lag window length; must be <= argvals range
    pub n_lags: usize,             // discretization points for lag grid (default: 20)
    pub bandwidth: f64,            // for history weight function; 0.0 = auto (default: 0.0)
    pub kernel: String,            // default: "gaussian"
}
```

### Result Structs

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FamResult {
    pub fitted_values: Vec<f64>,      // ŷ (length n)
    pub residuals: Vec<f64>,          // y - ŷ (length n)
    pub component_fits: Vec<Vec<f64>>, // f_k(ξ_k) per observation (K × n)
    pub intercept: f64,               // mu_y
    pub bandwidths: Vec<f64>,         // per-component optimal bandwidth (length K)
    pub ncomp: usize,
    pub r_squared: f64,
    pub fpca: FpcaResult,             // embedded for prediction on new data
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GkamResult {
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub component_fits: Vec<Vec<f64>>,  // f_k values per predictor (q × n)
    pub intercept: f64,
    pub bandwidths: Vec<f64>,           // per-predictor bandwidth (length q)
    pub iterations: usize,
    pub converged: bool,
    pub r_squared: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GsamResult {
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub component_fits: Vec<Vec<f64>>,  // f_j^k per FPC component (K × n)
    pub intercept: f64,
    pub bandwidths: Vec<f64>,
    pub ncomp: usize,
    pub r_squared: f64,
    pub fpca: FpcaResult,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct VarSelectResult {
    pub active_predictors: Vec<bool>,    // length P: true if predictor selected
    pub coefficients: Vec<Vec<f64>>,     // beta_g per predictor (P × K_p)
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub intercept: f64,
    pub lambda: f64,                     // selected or provided lambda
    pub r_squared: f64,
    pub iterations: usize,
    pub converged: bool,
    pub fpcas: Vec<FpcaResult>,          // one per predictor
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PermTestResult {
    pub p_value: f64,
    pub observed_statistic: f64,
    pub null_statistics: Vec<f64>,       // T_perm for each permutation
    pub n_perm_success: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HistoryIndexResult {
    pub fitted_values: Vec<f64>,
    pub residuals: Vec<f64>,
    pub intercept: f64,
    pub slope: f64,                      // β_1 coefficient on history score
    pub gamma: Vec<f64>,                 // estimated history weight function (length n_lags)
    pub lag_grid: Vec<f64>,              // lag discretization points (length n_lags)
    pub history_scores: Vec<f64>,        // Σ_l γ_l · X_i(T-u_l) · Δu per observation
    pub r_squared: f64,
}
```

### Integration Points (additive/non-breaking)

```rust
// In scalar_on_function/mod.rs — ADD:
mod additive;
pub use additive::{
    fam, fregre_gkam, fregre_gsam, variable_selection,
    permutation_test_fam, history_index,
    FamConfig, FamResult,
    GkamConfig, GkamResult,
    GsamConfig, GsamResult,
    VarSelectConfig, VarSelectResult, VarSelectPenalty,
    PermTestConfig, PermTestResult, PermTestStatistic,
    HistoryIndexConfig, HistoryIndexResult,
};

// In src/lib.rs — ADD crate-root re-exports:
pub use scalar_on_function::{
    // existing exports unchanged ...
    fam, fregre_gkam, fregre_gsam, variable_selection,
    permutation_test_fam, history_index,
    FamConfig, FamResult,
    GkamConfig, GkamResult,
    GsamConfig, GsamResult,
    VarSelectConfig, VarSelectResult, VarSelectPenalty,
    PermTestConfig, PermTestResult, PermTestStatistic,
    HistoryIndexConfig, HistoryIndexResult,
};
```

---

## Common Pitfalls

### Pitfall 1: Iterating FAM When FPC Scores Are Uncorrelated

**What goes wrong:** Implementing a backfitting loop for FAM (iterating update equations until convergence) wastes compute.

**Why it happens:** "Backfitting" is the general name for the algorithm, so implementors add an iteration loop. But Müller & Yao (2008) prove that for FAM specifically, because FPC scores are uncorrelated (orthogonal under the functional L2 inner product), a single sequential pass achieves the same result as infinite-iteration backfitting.

**How to avoid:** Implement one forward pass over k = 0..K, updating `partial_resid_k` and fitting `f_k`. No convergence check needed for FAM.

**Warning signs:** If you see a `loop { … if delta < eps { break; } }` structure in the FAM estimator, it is unnecessary overhead (harmless but slow for K > 5).

### Pitfall 2: Treating GKAM and GSAM as the Same Estimator

**What goes wrong:** Both use additive components and kernel smoothers, but their predictor representation differs fundamentally. Conflating them produces incorrect fits.

**Why it happens:** GKAM operates on L2 distances between raw curves. GSAM operates on FPC score coordinates (a Euclidean space). Same formula structure, different inputs.

**How to avoid:** Maintain separate `compute_pairwise_distances` (GKAM path) vs `fdata_to_pc_1d` (GSAM path) code paths. Shared inner NW smoother is fine, but the "x" passed to it differs.

**Warning signs:** Using FPC scores inside GKAM's kernel argument (wrong — NW in GKAM uses L2 dist between curves, not score space distances).

### Pitfall 3: Off-by-One in History-Index Lag Extraction

**What goes wrong:** Extracting `X_i(T - u_l)` for lag `u_l` maps to column index `j` in `FdMatrix` via `argvals`. If `argvals` is not uniformly spaced or if the lag value is not exactly on the grid, interpolation is needed but often missed.

**Why it happens:** `FdMatrix[(i, j)]` requires an integer column index. Lag values on the discretized grid may not fall exactly on `argvals`.

**How to avoid:** In the history-index estimator, for each lag `u_l`, find the nearest `argvals[j]` (floor or nearest-neighbor) OR linearly interpolate between `argvals[j]` and `argvals[j+1]`. Document the interpolation method in rustdoc.

**Warning signs:** `argvals` has 100 evaluation points but history index uses 20 lag grid points — if the mapping is `j = round(u_l / argvals_step)`, ensure no index out-of-bounds when `u_l = window`.

### Pitfall 4: Group Penalty Convergence Stalls at Zero

**What goes wrong:** Variable selection initializes all `β_g = 0`, and the group-lasso update leaves all groups at zero if `λ` is too large.

**Why it happens:** The group-lasso threshold `max(0, 1 - λ√K_g / ||β_g_ols||)` sets groups to zero when `λ√K_g > ||β_g_ols||`. If `λ` is initialized too large, the algorithm converges immediately to the zero vector.

**How to avoid:** Initialize `λ` at the maximum value that allows at least one group to be non-zero: `λ_max = max_g ||X_g'y|| / √K_g`. Lambda grid should start at `0.01 × λ_max` and end at `λ_max`, searched via CV.

**Warning signs:** `active_predictors` is all-false after iteration; CV error is constant across λ values.

### Pitfall 5: Clippy `--all-targets` Catching Test-Only Type Annotations

**What goes wrong:** CI runs `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. Types or imports used only in `#[cfg(test)]` blocks that are visible at the module level trigger warnings.

**How to avoid:** Annotate test-only imports as `#[cfg(test)] use ...` inside the `#[cfg(test)] mod tests { ... }` block. Never pull `use rand::prelude::*` at the module top level.

**Warning signs:** Clippy reports `unused import` or `unused struct` on types that are present in tests but not in production code paths.

---

## Validation Architecture

> `workflow.nyquist_validation` is `true` in `.planning/config.json`. This section is required.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) |
| Config file | None (inline `#[cfg(test)] mod tests`) |
| Quick run command | `cargo test -p fdars-core --lib scalar_on_function::additive -- --nocapture` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| REG-04-FAM | FAM recovers known additive signal f_1(ξ_1) + f_2(ξ_2) within tolerance | unit (synthetic recovery) | `cargo test fam_synthetic_recovery` | ❌ Wave 0 |
| REG-04-FAM | FAM fitted + residuals = y (decomposition identity) | unit (invariant) | `cargo test fam_decomposition_identity` | ❌ Wave 0 |
| REG-04-FAM | FAM returns `InvalidDimension` for empty data / mismatched y length | unit (error path) | `cargo test fam_invalid_dimension` | ❌ Wave 0 |
| REG-04-FAM | FAM output `component_fits.len() == ncomp` | unit (shape) | `cargo test fam_output_shapes` | ❌ Wave 0 |
| REG-04-GKAM | GKAM convergence: `converged == true` for smooth synthetic data | unit (invariant) | `cargo test gkam_convergence` | ❌ Wave 0 |
| REG-04-GKAM | GKAM `InvalidDimension` for mismatched predictor/response lengths | unit (error path) | `cargo test gkam_invalid_inputs` | ❌ Wave 0 |
| REG-04-GKAM | GKAM R² > 0.7 on synthetic additive kernel data | unit (synthetic recovery) | `cargo test gkam_r2_synthetic` | ❌ Wave 0 |
| REG-04-GSAM | GSAM gives same fitted values as FAM when Gaussian identity link used | unit (invariant) | `cargo test gsam_matches_fam_identity` | ❌ Wave 0 |
| REG-04-GSAM | GSAM `InvalidParameter` if `ncomp > min(n,m)` | unit (error path) | `cargo test gsam_ncomp_too_large` | ❌ Wave 0 |
| REG-04-VS | variable_selection active-subset recovery: known active 2 of 5 predictors identified | unit (synthetic recovery) | `cargo test varselect_active_subset_recovery` | ❌ Wave 0 |
| REG-04-VS | variable_selection GroupLasso gives all-zero active set at λ=λ_max | unit (invariant) | `cargo test varselect_lambda_max_zeros` | ❌ Wave 0 |
| REG-04-VS | variable_selection `InvalidDimension` if predictor/response mismatch | unit (error path) | `cargo test varselect_invalid_inputs` | ❌ Wave 0 |
| REG-04-PERM | permutation_test seeded reproducibility: same seed → same p_value | unit (invariant) | `cargo test perm_seeded_reproducibility` | ❌ Wave 0 |
| REG-04-PERM | permutation_test p_value ∈ [0,1] for all inputs | unit (invariant) | `cargo test perm_pvalue_range` | ❌ Wave 0 |
| REG-04-PERM | permutation_test p_value ≈ small (< 0.1) when true effect present | unit (synthetic) | `cargo test perm_detects_true_effect` | ❌ Wave 0 |
| REG-04-HI | history_index fitted R² > 0.7 on synthetic lagged-signal data | unit (synthetic recovery) | `cargo test history_index_synthetic_recovery` | ❌ Wave 0 |
| REG-04-HI | history_index `InvalidParameter` for window > argvals range | unit (error path) | `cargo test history_index_window_too_large` | ❌ Wave 0 |
| REG-04-HI | history_index `gamma.len() == config.n_lags` | unit (shape) | `cargo test history_index_output_shapes` | ❌ Wave 0 |

### Synthetic Recovery Test Strategy

Each estimator needs a "known answer" synthetic test:

**FAM:** Generate `y_i = sin(ξ_i1) + ξ_i2^2 + ε_i` where `ξ` is obtained by projecting synthetic sinusoidal curves through `fdata_to_pc_1d`. Fit FAM with `ncomp=2`. Assert R² > 0.75 and `||fitted - true_signal||/||true_signal|| < 0.30`.

**GKAM:** Generate 2 functional covariates with known Gaussian L2-kernel dependence. Assert fitted R² > 0.70 and `converged == true`.

**GSAM:** Same as FAM with identity link — assert fitted values match FAM to within 1e-6 (same mathematical path).

**variable_selection (active-subset recovery):** Generate 5 functional predictors, only 2 active (`y = f(X^1) + g(X^3) + ε`). Assert `active_predictors == [true, false, true, false, false]` for appropriate λ.

**permutation_test (detect true effect):** Use `y = 2·ξ_1 + ε` (strong linear FAM component). With n_perm=99 and seed=42, assert p_value < 0.1. Under null (`y = ε` only), assert p_value > 0.1.

**history_index:** Generate `y_i = Σ_{u=0}^{0.5} X_i(1.0 - u) · du` (uniform γ, Δ=0.5). Assert estimated γ is approximately uniform and R² > 0.70.

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --lib scalar_on_function::additive`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` before `/gsd-verify-work`

### Wave 0 Gaps

All test functions listed above are new — they live inside `scalar_on_function/additive.rs` in the `#[cfg(test)] mod tests { ... }` block. No separate file is needed. No external test framework setup is needed. The existing `test_helpers::uniform_grid` helper is available.

- [ ] `fdars-core/src/scalar_on_function/additive.rs` — create with all six estimators + all tests
- [ ] Update `fdars-core/src/scalar_on_function/mod.rs` — add `mod additive; pub use additive::*;`
- [ ] Update `fdars-core/src/lib.rs` — add crate-root re-exports

*(No gaps in existing infrastructure: cargo test, rustfmt, clippy all operational.)*

---

## Security Domain

> `security_enforcement` is `true` and `security_asvs_level` is `1` in `.planning/config.json`.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Pure computation library, no auth surface |
| V3 Session Management | no | Stateless function calls, no sessions |
| V4 Access Control | no | No user-facing API, no roles |
| V5 Input Validation | yes | `FdarError::InvalidDimension` / `InvalidParameter` at all function entry points |
| V6 Cryptography | no | RNG is used for seeding permutations only (statistical reproducibility, not security) |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in index arithmetic | Tampering | Use `.min(n-1)` clamping; Rust panics on debug overflow |
| Division by zero in NW denominator | Tampering | Guard `if denom > 1e-15` (existing pattern in nonparametric.rs) |
| NaN propagation from degenerate inputs | Tampering | Validate `y.len() == n`, `argvals.len() == m` before computation |
| Unbounded iteration in GKAM | DoS | `max_iter` field in `GkamConfig` with sane default (50) |
| Stack overflow from recursive backfitting | DoS | No recursion — all loops are iterative |

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| R's FAM uses PACE bandwidth | fdars uses `optim_bandwidth` GCV grid search | Rust implementation (this phase) | Minor numerical divergence, same asymptotic behavior |
| R's GSAM delegates to `mgcv::gam` splines | fdars uses NW on FPC score columns | Rust implementation (this phase) | Less flexible smoother; document in rustdoc |
| R's fosr.vs is function-on-scalar (functional Y) | fdars implements scalar-on-function variable selection (scalar Y) | Rust implementation (this phase) | Different regression direction; document in rustdoc |

**Deprecated/outdated:**
- None: these are new estimators, not replacements for existing fdars functions.

---

## Runtime State Inventory

> Phase 31 is a greenfield new-file phase (no rename, refactor, or migration). No runtime state inventory is required.

**Step 2.5: SKIPPED** — Phase adds one new file; no stored data, live service config, OS state, secrets, or build artifacts are affected.

---

## Environment Availability

> All required tooling confirmed available from project context.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Compilation | ✓ | 1.97.0 (dev) | — |
| `cargo test` | Validation | ✓ | bundled with 1.97.0 | — |
| `cargo clippy` | CI gate | ✓ | bundled | — |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | Doctest linking | ✓ (from MEMORY.md) | — | Use `--no-verify` for docs if /tmp fills |

**Missing dependencies with no fallback:** None.

**Build note (from MEMORY.md):** If example linking fails with "No space left on device", run `rm -rf target/debug/{incremental,examples}` to free ~108GB. This phase adds only library code (no new examples), so the risk is lower than example-heavy phases.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | FAM single-pass is sufficient because FPC scores are uncorrelated — R's fdapace::FAM does not use iterative backfitting | Pattern 1 | If R uses iteration, fdars result may diverge slightly from R baseline; document in rustdoc either way |
| A2 | GSAM with Gaussian identity link produces the same fitted values as FAM | Pattern 3 | Could differ if GSAM uses different smoothing inside mgcv; test `gsam_matches_fam_identity` will catch this |
| A3 | R's `fosr.vs` implements scalar-on-function (not function-on-scalar) regression direction | Pattern 4 | If R's fosr.vs is function-on-scalar, the fdars implementation diverges in regression direction; must document |
| A4 | History-index model with scalar Y can be reduced to a 1D NW regression over the history score | Pattern 6 | If the weight function γ requires a bivariate smooth (as in full pffr), the marginal-integration approach is an approximation |
| A5 | `compute_pairwise_distances` in `nonparametric.rs` is accessible from `additive.rs` (via `pub(super)`) | Reuse map | If visibility is restricted, must either re-implement or change to `pub(crate)` |
| A6 | `VarSelectPenalty::GroupMcp` and `GroupScad` can be implemented without new crate dependencies using coordinate-descent with closed-form thresholding | Pattern 4 | MCP/SCAD thresholding is iterative per-group but closed-form for each step; if the solver diverges, fall back to GroupLasso only |

---

## Open Questions

1. **Visibility of `nonparametric::compute_pairwise_distances` for GKAM**
   - What we know: It is declared `pub(super)` in `nonparametric.rs`, meaning visible to the `scalar_on_function` module.
   - What's unclear: Whether a new sibling submodule `additive.rs` can access `pub(super)` items from `nonparametric.rs` without routing through `mod.rs`.
   - Recommendation: During planning, verify by checking Rust's `pub(super)` visibility rules. If not accessible, either (a) change to `pub(crate)` in `nonparametric.rs` (additive/non-breaking), or (b) re-implement the L2 distance loop inline in `additive.rs` (3 lines, not complex). Option (a) preferred. [VERIFIED: fdars-core/src/scalar_on_function/nonparametric.rs:10-27 — `pub(super)` is module-level, accessible from within `scalar_on_function/` sibling modules]

2. **GroupMcp / GroupScad implementation without new crate**
   - What we know: `grpReg` style group MCP/SCAD requires a threshold function with MCP/SCAD shape, but the coordinate descent update is a closed-form expression per group.
   - What's unclear: Whether the closed-form threshold is stable enough to implement without a tested library.
   - Recommendation: Implement `GroupLasso` as primary path (well-understood, convex). Add `GroupMcp` and `GroupScad` as secondary paths with explicit test coverage. If instability occurs, fall back to lasso.

3. **GKAM IRLS for non-Gaussian link functions**
   - What we know: R's `fregre.gkam` supports logit and log links via iterative local scoring (which wraps IRLS around the backfitting loop).
   - What's unclear: Whether to implement only the Gaussian identity case in phase 31 or also add IRLS support.
   - Recommendation: Implement only `GlmFamily::Gaussian` (identity link) in phase 31 to avoid scope creep. Document other families as a known gap in rustdoc (consistent with milestone convention for phased capability delivery).

---

## Sources

### Primary (MEDIUM confidence — official documentation/paper references)

- [Müller & Yao (2008) JASA — Functional Additive Models](https://anson.ucdavis.edu/~mueller/fam.pdf) — FAM model equation, FPC uncorrelatedness, backfitting simplification
- [Wikipedia: Functional additive model](https://en.wikipedia.org/wiki/Functional_additive_model) — Model equation E(Y|X), FPC score decomposition, uncorrelated-FPC simplification
- [fda.usc::fregre.gkam documentation](https://rdrr.io/cran/fda.usc/man/fregre.gkam.html) — GKAM algorithm, NW hat matrix, iterative local scoring
- [fda.usc::fregre.gsam documentation](https://rdrr.io/cran/fda.usc/man/fregre.gsam.html) — GSAM model equation with FPC scores and smooth functions
- [refund::fosr.vs documentation](https://rdrr.io/cran/refund/man/fosr.vs.html) — Group penalties (grLasso, grMCP, grSCAD), iterative algorithm
- [Wikipedia: History index model](https://en.wikipedia.org/wiki/History_index_model) — History index model E{Y(t)|X(t)}, lag window Δ, γ normalization

### Secondary (LOW confidence — web search results confirming above)

- [fdars-core/src/smoothing.rs](fdars-core/src/smoothing.rs) — `nadaraya_watson`, `local_linear`, `local_polynomial`, `optim_bandwidth` signatures verified
- [fdars-core/src/regression.rs](fdars-core/src/regression.rs) — `fdata_to_pc_1d`, `FpcaResult` signatures verified
- [fdars-core/src/famm.rs](fdars-core/src/famm.rs) — permutation test pattern (`StdRng::seed_from_u64(seed + k)`, p-value formula) verified

---

## Metadata

**Confidence breakdown:**
- FAM mathematical formulation: MEDIUM — confirmed from Wikipedia + search citing Mueller & Yao 2008 paper
- FAM FPC uncorrelatedness insight: MEDIUM — confirmed from Wikipedia verbatim quote
- GKAM algorithm: MEDIUM — confirmed from fda.usc rdrr.io documentation
- GSAM algorithm: MEDIUM — confirmed from fda.usc rdrr.io documentation
- variable_selection: LOW-MEDIUM — group penalty shapes confirmed from refund docs; regression direction divergence is assumed
- history-index: MEDIUM — Wikipedia article confirms model equation exactly
- permutation test: HIGH — mirrors pattern verified in `fdars-core/src/famm.rs` this session
- Reuse map (smoothing.rs / regression.rs / nonparametric.rs functions): HIGH — all files read this session
- API surface: MEDIUM — consistent with fdars conventions verified from mod.rs, StlConfig, FmmTestResult patterns

**Research date:** 2026-08-20
**Valid until:** 2026-09-20 (stable domain — FDA methods, not fast-moving ecosystem)
