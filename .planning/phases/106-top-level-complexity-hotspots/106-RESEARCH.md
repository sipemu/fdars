# Phase 106: Top-Level Complexity Hotspots — Research

**Researched:** 2026-09-13
**Domain:** Rust internal refactor — cognitive/cyclomatic complexity reduction by private helper extraction
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

All implementation choices are at Claude's discretion — pure refactor phase.
Constraints (from PROJECT.md / STATE.md, non-negotiable):

- **Strictly non-breaking + numeric-preserving.** Every change is pure internal
  helper extraction. No public API / signature / visibility changes.
- **Reduce complexity by helper extraction only** — never by changing signatures.
- **`fdars-core` only** — protects R + WASM bindings.
- **No new crate dependency** — pmat is a dev/CI tool invoked externally.
- Proof of behavior-preservation = the full existing test suite (2942 lib tests +
  215 doctests + 28 examples) staying green, plus `pmat analyze complexity`
  confirming each target function drops below threshold.
- Extracted helpers are private (`fn`, not `pub fn`), placed adjacent to their
  parent in the same module, following the crate's existing helper conventions.

### Claude's Discretion

All naming, exact seam placement, and internal helper structure choices are at Claude's discretion.

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| QUAL-01 | The worst top-level complexity hotspots (`funfem_cluster`, `elastic_net_cd`, `wnet_cv_lambda`, `wnet`, `boost_fofr`, `pace_fpca`, `bayesian_fosr`, `kcfc_cluster`) are refactored into smaller helpers, each under pmat's cognitive/cyclomatic thresholds, with behavior and numeric output identical (verified by the existing test suite). | All 8 functions located, baseline scores confirmed, decomposition seams identified per function, borrow-checker hazards documented. |
</phase_requirements>

---

## Summary

Phase 106 targets 8 functions in `fdars-core` that exceed pmat 3.18.0's thresholds
(cyclomatic > 30 or cognitive > 25). All 8 have been located, their precise line
ranges confirmed via live pmat output, and natural decomposition seams identified
from reading the source.

The dominant structure across the 8 targets is a common 3-part pattern:
(1) input validation block, (2) setup/preprocessing, (3) one or more inner
iteration loops, (4) result assembly. Each stage is readily extractable into a
private helper. The functions do NOT share state across helpers — all communication
can be through value/reference parameters, meaning no borrowck hazards beyond
standard slice/Vec borrows.

**Primary recommendation:** Extract private helpers stage-by-stage (validate →
setup → iter-body → assemble), keeping floating-point accumulation order
byte-identical with the original. Numeric preservation is trivially ensured by
not reordering loop iterations or summation operands.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Helper extraction | Internal module (private impl) | — | Pure internal refactor; no tier boundary crosses |
| Validation blocks | Same module, private fn | — | Error paths unchanged, no public surface |
| Inner loop bodies | Same module, private fn | — | State passed as refs/values; no cross-module impact |
| Result assembly | Same module, private fn | — | Output types unchanged |

---

## pmat 3.18.0 Baseline — Confirmed Scores

> [VERIFIED: pmat 3.18.0 live output, 2026-09-13]

All scores below were obtained by running:
```bash
pmat analyze complexity --path fdars-core/src/<file> 2>/dev/null
pmat quality-gate 2>/dev/null   # for threshold messages
```

**pmat thresholds (from quality-gate output):**
- Cyclomatic: error at > 30, warning at > 28
- Cognitive: error at > 25, warning at > 23

| Function | File (line) | Cyclomatic | Cognitive | Gate Violations |
|----------|-------------|-----------|-----------|-----------------|
| `funfem_cluster` | `clustering_advanced.rs:701` | **59** | **167** | cyc + cog errors |
| `kcfc_cluster` | `clustering_advanced.rs:371` | 28 | **93** | cog error |
| `elastic_net_cd` | `wavelet/regression.rs:843` | 19 | **54** | cog error |
| `wnet_cv_lambda` | `wavelet/regression.rs:1015` | 15 | **45** | cog error |
| `wnet` | `wavelet/regression.rs:1145` | 12 | **31** | cog error |
| `boost_fofr` | `boosting_regression/boost_fofr.rs:197` | **39** | **98** | cyc + cog errors |
| `pace_fpca` | `pace_fpca.rs:266` | **37** | **78** | cyc + cog errors |
| `bayesian_fosr` | `boosting_regression/bayesian.rs:101` | **36** | **84** | cyc + cog errors |

**Note on roadmap vs live scores:** The roadmap listed cognitive scores only for
`elastic_net_cd` (54), `wnet_cv_lambda` (45), `wnet` (31); all confirmed. Roadmap
listed cyclomatic for the remaining five; all confirmed. Live pmat also shows
cognitive scores for these functions that are much higher than the roadmap column
implies — the roadmap's single-number was just the worst-axis score that drove
the original classification.

**Exact pmat invocations for per-function verification:**
```bash
# Single-file (fastest — confirms function scores):
pmat analyze complexity --path fdars-core/src/clustering_advanced.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/wavelet/regression.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/boosting_regression/boost_fofr.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/pace_fpca.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/boosting_regression/bayesian.rs 2>/dev/null

# Full project (used for quality-gate):
pmat quality-gate 2>/dev/null
```

---

## Per-Function Analysis

### 1. `funfem_cluster` — `clustering_advanced.rs:701–1096`

> [VERIFIED: fdars-core/src/clustering_advanced.rs:701-1096]

**Scores:** cyclomatic 59, cognitive 167 (both error-level)

**Function shape (395 lines):**

```
Lines 701–741   Validation block (6 guard returns)
Lines 742–756   Step 1: FPCA score extraction + p_disc_eff computation
Lines 757–833   Step 2: k-means++ initialization loop (center selection + D² sampling)
Lines 834–851   Step 3: Initial hard labels + GMM param initialization
Lines 852–854   Convergence state init
Lines 855–1070  Main EM iteration loop (the giant for _iter block):
  Lines 858–870     Project scores onto discriminative subspace
  Lines 872–906     GMM in projected space: means + variances
  Lines 908–938     E-step: log-responsibilities + log-sum-exp
  Lines 940–975     M-step: update pi + hard assignments + score-space params
  Lines 977–1061    Fisher-EM subspace update (scatter matrices + Cholesky + SVD)
  Lines 1063–1070   Convergence check
Lines 1072–1093 Result assembly (membership FdMatrix + disc_subspace FdMatrix)
Lines 1086–1094 Ok(FunFemResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch | Notes |
|-----------------|-----------|-----------------|-------|
| `funfem_validate` | 701–741 | `fn funfem_validate(data: &FdMatrix, argvals: &[f64], config: &FunFemConfig) -> Result<(usize, usize), FdarError>` returns `(n, m)` | 6 guards, clean extraction |
| `funfem_kmeans_pp_init` | 757–833 | `fn funfem_kmeans_pp_init(row_major_scores: &[f64], n: usize, ncomp_eff: usize, k: usize, rng: &mut StdRng) -> Vec<usize>` returns `center_indices` | D² sampling loop + initial label assignment |
| `funfem_project_scores` | 858–870 | `fn funfem_project_scores(scores: &FdMatrix, disc_dirs: &[f64], n: usize, ncomp_eff: usize, p_disc_eff: usize) -> Vec<f64>` returns `proj_scores` | Pure matrix multiply, no side effects |
| `funfem_gmm_disc_stats` | 872–906 | `fn funfem_gmm_disc_stats(proj_scores: &[f64], resp: &[f64], n: usize, k: usize, p_disc_eff: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>)` returns `(mu_disc, var_disc, n_k_soft)` | Soft mean + variance in projected space |
| `funfem_e_step` | 908–938 | `fn funfem_e_step(proj_scores: &[f64], mu_disc: &[Vec<f64>], var_disc: &[Vec<f64>], pi: &[f64], n: usize, k: usize, p_disc_eff: usize) -> (Vec<f64>, f64)` returns `(resp, ll)` | Log-sum-exp E-step, no side effects |
| `funfem_subspace_update` | 977–1061 | `fn funfem_subspace_update(row_major_scores: &[f64], resp: &[f64], mu_k: &[Vec<f64>], n_k_new: &[f64], n: usize, ncomp_eff: usize, k: usize, p_disc_eff: usize) -> Option<Vec<f64>>` returns updated `disc_dirs` or None on Cholesky failure | Between/within scatter + Cholesky + SVD block |

**Borrow-checker hazards:**
- `disc_dirs` is updated inside the main loop (rebind, not mutation). The proposed `funfem_subspace_update` returns a new `Vec<f64>` and the parent assigns it — no aliasing.
- `resp` is read by `funfem_e_step` and written (via `fill` + loop) by the parent. Pass `&mut resp` to E-step or have E-step return a new `Vec<f64>` and overwrite — the latter is simpler and matches the current flow.
- The `scores` FdMatrix borrow (from `fpca.scores`) is read-only throughout — no hazard.

---

### 2. `kcfc_cluster` — `clustering_advanced.rs:371–700`

> [VERIFIED: fdars-core/src/clustering_advanced.rs:371-700]

**Scores:** cyclomatic 28, cognitive 93 (cognitive error-level; cyclomatic within warning range)

**Function shape (330 lines):**

```
Lines 371–411   Validation (5 guards)
Lines 413–413   Simpson weights
Lines 416–476   k-means++ init (center selection + D² sampling loop) — identical pattern to funfem_cluster
Lines 478–570   Main reassignment loop:
  Lines 488–515     Per-cluster FPCA fit (gather members, build sub-FdMatrix, fdata_to_pc)
  Lines 517–548     Reconstruction error for all (curve, cluster) pairs
  Lines 550–569     Reassignment + convergence check
Lines 572–578   Ok(KcfcResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `kcfc_validate` | 371–411 | `fn kcfc_validate(data: &FdMatrix, argvals: &[f64], config: &KcfcConfig) -> Result<(usize, usize), FdarError>` |
| `kcfc_kmeans_pp_init` | 416–476 | `fn kcfc_kmeans_pp_init(row_major: &[f64], n: usize, m: usize, k: usize, weights: &[f64], rng: &mut StdRng) -> Vec<usize>` — note this is the raw-curve D² init (uses `weights`), distinct from `funfem_kmeans_pp_init` which is score-space Euclidean |
| `kcfc_fit_cluster_fpca` | 488–515 | `fn kcfc_fit_cluster_fpca(data: &FdMatrix, cluster: &[usize], ki: usize, ncomp: usize, argvals: &[f64]) -> Option<FpcaResult>` returns `Some(fpca)` or `None` on degenerate cluster |
| `kcfc_reconstruction_errors` | 517–548 | `fn kcfc_reconstruction_errors(data: &FdMatrix, fpca_models: &[Option<FpcaResult>], weights: &[f64], n: usize, m: usize, k: usize) -> Result<FdMatrix, FdarError>` returns n×k error matrix |
| `kcfc_reassign` | 550–569 | `fn kcfc_reassign(reconstruction_errors: &FdMatrix, cluster: &mut [usize], n: usize, k: usize) -> bool` returns `changed` |

**Borrow-checker hazards:**
- `fpca_models: Vec<Option<FpcaResult>>` — the FPCA fit loop writes `fpca_models[ki]`. If `kcfc_fit_cluster_fpca` takes `&mut Vec<Option<FpcaResult>>` and writes at `ki`, callers must ensure they don't hold references simultaneously. Simpler: return `Option<FpcaResult>` from the helper and let the caller assign at `ki`.
- `reconstruction_errors` FdMatrix is written in the error loop. Passing a mutable reference or returning a new matrix — returning is cleaner.

---

### 3. `elastic_net_cd` — `wavelet/regression.rs:843–956`

> [VERIFIED: fdars-core/src/wavelet/regression.rs:843-956]

**Scores:** cyclomatic 19, cognitive 54 (cognitive error-level)

**Function shape (114 lines):**

```
Lines 843–882   Validation (5 guards: y dim, alpha bounds, lambda bounds, tol bounds, max_iter)
Lines 884–903   Preprocessing: center y and X columns, compute col_norm_sq_over_n
Lines 904–911   Coefficients init (beta, fit, l1, l2)
Lines 913–951   Coordinate-descent sweep loop (for _sweep in 0..max_iter):
  Lines 916–928     Dead-column branch (denom ≤ 0)
  Lines 931–946     Normal coordinate update (z_j, soft_threshold, fit update)
  Lines 948–950     Convergence check
Lines 953–955   Intercept recovery + Ok(...)
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `elastic_net_validate` | 843–882 | `fn elastic_net_validate(design: &FdMatrix, y: &[f64], lambda: f64, alpha: f64, max_iter: usize, tol: f64) -> Result<(), FdarError>` |
| `elastic_net_center_design` | 884–903 | `fn elastic_net_center_design(design: &FdMatrix, y: &[f64], n: usize, p: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)` returns `(col_means, xc, col_norm_sq_over_n, y_centered)` |
| `elastic_net_cd_sweep` | 913–951 | `fn elastic_net_cd_sweep(xc: &[f64], y_centered: &[f64], col_norm_sq_over_n: &[f64], l1: f64, l2: f64, n: usize, p: usize, max_iter: usize, tol: f64, beta: &mut [f64], fit: &mut [f64])` — the hot-path inner loop; `#[inline]` candidate |

**Borrow-checker hazards:**
- `beta` and `fit` are both mutated inside the sweep. Passing both as `&mut [f64]` to the helper is clean because they are disjoint slices.
- `xc` is an immutable borrow; no conflict with mutating `beta`/`fit`.
- Numeric preservation: the loop order (`for j in 0..p`, then `for i in 0..n`) must not change. The helper takes ownership of the loop body exactly.

---

### 4. `wnet_cv_lambda` — `wavelet/regression.rs:1015–1111`

> [VERIFIED: fdars-core/src/wavelet/regression.rs:1015-1111]

**Scores:** cyclomatic 15, cognitive 45 (cognitive error-level)

**Function shape (97 lines):**

```
Lines 1015–1056  Validation (5 guards: y dim, n_folds ≥ 2, n_folds ≤ n, alpha, lambda_grid non-empty)
Lines 1058–1064  Grid build + fold creation + fold_sets precompute
Lines 1066–1108  Main λ-grid scan loop:
  Lines 1070–1096   Per-fold CV (skip empty folds, train elastic_net_cd, score on test)
  Lines 1097–1100   MSE computation (skip if scored == 0)
  Lines 1102–1107   Tie-breaking update of best_lambda
Lines 1110          Ok(best_lambda)
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `wnet_cv_validate` | 1015–1056 | `fn wnet_cv_validate(design: &FdMatrix, y: &[f64], config: &WnetConfig) -> Result<(), FdarError>` |
| `wnet_cv_score_lambda` | 1070–1100 | `fn wnet_cv_score_lambda(design: &FdMatrix, y: &[f64], lam: f64, config: &WnetConfig, fold_sets: &[(Vec<usize>, Vec<usize>)]) -> Result<Option<f64>, FdarError>` returns `Some(mse)` or `None` if no fold scored — inner fold loop |

**Borrow-checker hazards:**
- `fold_sets` is constructed before the lambda loop and borrowed inside — no issue.
- `elastic_net_cd` is called inside the helper, using standard `?` propagation.
- `design` is borrowed immutably by both the outer function and the helper — clean.

---

### 5. `wnet` — `wavelet/regression.rs:1145–1257`

> [VERIFIED: fdars-core/src/wavelet/regression.rs:1145-1257]

**Scores:** cyclomatic 12, cognitive 31 (cognitive error-level only)

**Function shape (113 lines):**

```
Lines 1145–1208  Validation (8 guards: n < 3, m == 0, y.len(), alpha, n_folds ≥ 2, n_folds ≤ n, max_iter, tol, lambda_grid)
Lines 1210–1212  Design matrix construction
Lines 1215–1223  CV lambda selection + full refit
Lines 1225–1242  Post-fit derivations (selected, fitted_values, residuals, beta_t)
Lines 1244–1256  Ok(WnetResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `wnet_validate` | 1145–1208 | `fn wnet_validate(data: &FdMatrix, y: &[f64], config: &WnetConfig) -> Result<(), FdarError>` — consolidates 8 guards |

The validation block alone brings `wnet` from cognitive 31 to approximately 12–15
(matching `wnet_cv_validate` + `wnet_validate` pattern). No other seam is needed;
the actual computation is already delegated to `curves_to_coeff_design`, `wnet_cv_lambda`,
`elastic_net_cd`, and `coeff_weights_to_beta_t`.

**Borrow-checker hazards:** None. After extraction, the function body is entirely
sequential calls to existing helpers. No mutable aliasing.

---

### 6. `boost_fofr` — `boosting_regression/boost_fofr.rs:197–523`

> [VERIFIED: fdars-core/src/boosting_regression/boost_fofr.rs:197-523]

**Scores:** cyclomatic 39, cognitive 98 (both error-level)

**Function shape (327 lines):**

```
Lines 197–261   Validation (multi-predictor dimension checks + config guards)
Lines 282–297   Preprocessing: per-predictor FPCA (fpca_x vec)
Lines 299–338   Pre-factor score-space learners (ScoreLearner struct with Cholesky)
Lines 341–364   Initialization: intercept (Ȳ), F̂ matrix, score_coefs
Lines 362–463   Boosting iteration loop (for _iter in 0..config.mstop):
  Lines 370–384    Residual + RSS_before
  Lines 386–440    Best learner selection (inner j loop: solve, compute fitted, compute RSS)
  Lines 442–463    Update F̂ + accumulate score_coefs
Lines 466–493   Coefficient surface reconstruction (beta_surfaces vec)
Lines 495–508   Final residuals + R²
Lines 509–522   Ok(BoostFofrResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `boost_fofr_validate` | 197–261 | `fn boost_fofr_validate(x_data: &[&FdMatrix], x_argvals: &[&[f64]], y_data: &FdMatrix, y_argvals: &[f64], config: &BoostingConfig) -> Result<(usize, usize, usize), FdarError>` returns `(n, m_y, p)` |
| `boost_fofr_build_learners` | 299–338 | `fn boost_fofr_build_learners(fpca_x: &[FpcaResult], n: usize) -> Result<Vec<ScoreLearner>, FdarError>` — builds ScoreLearner vec (S'S Cholesky) |
| `boost_fofr_best_learner` | 386–440 | `fn boost_fofr_best_learner(learners: &[ScoreLearner], residuals: &FdMatrix, n: usize, m_y: usize) -> (usize, FdMatrix, FdMatrix)` returns `(best_j, fitted_star, coefs_star)` — inner learner-selection loop |
| `boost_fofr_build_beta_surfaces` | 466–493 | `fn boost_fofr_build_beta_surfaces(fpca_x: &[FpcaResult], score_coefs: &[FdMatrix], m_y: usize) -> Vec<FdMatrix>` |

**Borrow-checker hazards:**
- `best_fitted: Option<FdMatrix>` and `best_coefs: Option<FdMatrix>` are moved out via `.expect(...)`. The proposed helper returns owned values — clean.
- `score_coefs` is `&mut Vec<FdMatrix>` accessed by predictor index. Pass it as `&mut Vec<FdMatrix>` to an accumulation helper, or perform the update inline after the helper returns `(best_j, coefs_star)`. Inline update avoids double-borrow hazard.
- `f_current` (n × m_y FdMatrix) is mutated each iteration. The update loop (lines 449–455) is simple and can stay inline or become a tiny private helper.

---

### 7. `pace_fpca` — `pace_fpca.rs:266–616`

> [VERIFIED: fdars-core/src/pace_fpca.rs:266-616]

**Scores:** cyclomatic 37, cognitive 78 (both error-level)

**Function shape (351 lines):**

```
Lines 266–350   Validation (7 guards: n==0, per-curve n_pts, m < 2, ncomp==0, bandwidth, sigma2, alpha, work_grid sorted)
Lines 352–402   Step 1: Kernel-smoothed mean + NaN guard
Lines 383–401   Step 2: Smoothed covariance
Lines 403–413   Step 3: Eigendecomposition
Lines 406–415   actual_ncomp guard + ef_cols precompute
Lines 417–584   Per-curve BLUP closure (the iter_maybe_parallel! map):
  Lines 436–450    Mean interpolation + residual
  Lines 452–459    Build Φ_i
  Lines 461–474    Build Σ_yi
  Lines 476–502    Ridge fallback + Cholesky solve for v
  Lines 504–521    BLUP scores + fitted trajectory
  Lines 523–580    Confidence band computation (sigma_inv_phi_lam + A_i + Ω_i)
Lines 586–601   Assemble results into output matrices
Lines 603–615   Ok(PaceFpcaResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `pace_fpca_validate` | 266–350 | `fn pace_fpca_validate(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<usize, FdarError>` returns `n` |
| `pace_fpca_blup_curve` | 435–582 | `fn pace_fpca_blup_curve(i: usize, data: &IrregFdata, config: &PaceFpcaConfig, mean: &[f64], ef_cols: &[Vec<f64>], eigenvalues: &[f64], eigenfunctions: &FdMatrix, actual_ncomp: usize, m: usize, z: f64, sigma2: f64) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), FdarError>` | This is already almost a closure; extracting it as a named fn makes the parallel map cleaner and reduces the closure's nesting depth significantly |

**Key note on `iter_maybe_parallel!`:** The closure passed to `iter_maybe_parallel!(0..n).map(|i| { ... })` spans lines 435–583. Extracting this closure body into a named private function `pace_fpca_blup_curve(i, ...)` keeps the parallel map call readable and eliminates the deep nesting that drives most of the cognitive score. The extracted function is not `pub` and has no effect on the WASM or parallel feature gating.

**Borrow-checker hazards:**
- `ef_cols: Vec<Vec<f64>>` is shared across all parallel iterations (read-only). A reference `&Vec<Vec<f64>>` passed to the helper is fine — rayon requires `Sync` but `Vec<Vec<f64>>` is `Sync`.
- `sigma_yi_resolved` is computed inside the closure and used for multiple solves. Moving it into the extracted function means it stays local — no cross-iteration aliasing.

---

### 8. `bayesian_fosr` — `boosting_regression/bayesian.rs:101–332`

> [VERIFIED: fdars-core/src/boosting_regression/bayesian.rs:101-332]

**Scores:** cyclomatic 36, cognitive 84 (both error-level)

**Function shape (232 lines):**

```
Lines 101–164   Validation (7 guards)
Lines 166–202   Setup: FPCA of response, center predictors, precompute X'X + X'ξ_k
Lines 204–215   Gibbs state init (b_state, sigma2, rng, totals)
Lines 217–219   Allocate beta_draws, beta_sum
Lines 221–283   Gibbs sampler loop (for iter in 0..total):
  Lines 222–266    Per-component k block (Cholesky precision, sample b, compute RSS, sample σ²_k)
  Lines 270–282    Post-burn-in thinned draw recording (beta reconstruction)
Lines 285         q_retained
Lines 287–298   Posterior summaries (beta_mean/lower/upper via sort + quantile)
Lines 301–318   Fitted values, residuals, sigma2_mean
Lines 320–331   Ok(BayesianFosrResult {...})
```

**Natural decomposition seams:**

| Proposed helper | Line range | Signature sketch |
|-----------------|-----------|-----------------|
| `bayesian_fosr_validate` | 101–164 | `fn bayesian_fosr_validate(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BayesianConfig) -> Result<(usize, usize, usize), FdarError>` returns `(n, m_t, p)` |
| `bayesian_fosr_precompute` | 166–202 | `fn bayesian_fosr_precompute(data: &FdMatrix, predictors: &FdMatrix, argvals: &[f64], config: &BayesianConfig, n: usize, p: usize) -> Result<(FpcaResult, FdMatrix, Vec<f64>, Vec<Vec<f64>>), FdarError>` returns `(fpca, xc, xbar, xt_xi)` |
| `bayesian_fosr_gibbs_step` | 222–266 | `fn bayesian_fosr_gibbs_step(kk: usize, b_state: &mut [Vec<f64>], sigma2: &mut [f64], xtx: &[f64], xt_xi: &[Vec<f64>], fpca: &FpcaResult, xc: &FdMatrix, config: &BayesianConfig, rng: &mut StdRng, n: usize, p: usize, a_post_shape: f64, inv_tau2: f64) -> Result<(), FdarError>` — one Gibbs step for component `kk` |
| `bayesian_fosr_record_draw` | 270–282 | `fn bayesian_fosr_record_draw(b_state: &[Vec<f64>], fpca: &FpcaResult, beta_draws: &mut [Vec<f64>], beta_sum: &mut [f64], k: usize, m_t: usize, p: usize)` |
| `bayesian_fosr_summaries` | 287–318 | `fn bayesian_fosr_summaries(beta_draws: Vec<Vec<f64>>, beta_sum: &[f64], n: usize, p: usize, m_t: usize, fpca: &FpcaResult, xc: &FdMatrix, data: &FdMatrix, q: usize) -> Result<(FdMatrix, FdMatrix, FdMatrix, FdMatrix, FdMatrix, Vec<f64>), FdarError>` returns `(beta_mean, beta_lower, beta_upper, fitted, residuals, sigma2_mean)` |

**Borrow-checker hazards:**
- `b_state: Vec<Vec<f64>>` and `sigma2: Vec<f64>` are mutated per Gibbs step per component. Pass as `&mut` slices to `bayesian_fosr_gibbs_step` — the outer loop iterates over `kk` sequentially, so no aliasing.
- `rng: &mut StdRng` is passed to `bayesian_fosr_gibbs_step`. Since the outer loop calls the helper sequentially, this is standard single-owner mutable reference — no issue.
- `beta_draws: Vec<Vec<f64>>` is moved into `bayesian_fosr_summaries` for sorting — move semantics, not a borrow hazard.
- **Critical numeric constraint:** The Gibbs loop iterates `for iter in 0..total { for kk in 0..k { ... } }`. The `rng` sampling order must be preserved identically: the outer iter loop must remain the outer loop, the kk loop stays inner. If `bayesian_fosr_gibbs_step` is called `for kk in 0..k` inside the iter loop, the RNG sequence is identical.

---

## Standard Stack

No new dependencies. All tools already in the crate:

| Utility | Source | Used by |
|---------|--------|---------|
| `FdMatrix`, `FdMatrix::zeros`, `from_column_major` | `src/matrix.rs` | All 8 functions |
| `fdata_to_pc` | `src/regression.rs` | `kcfc_cluster`, `funfem_cluster`, `boost_fofr`, `bayesian_fosr` |
| `cholesky_factor`, `cholesky_forward_back` | `src/linalg.rs` | `funfem_cluster`, `boost_fofr`, `bayesian_fosr` |
| `iter_maybe_parallel!` | `src/parallel.rs` | `pace_fpca` — retain macro in parent, extract body fn |
| `simpsons_weights` | `src/helpers.rs` | `kcfc_cluster` |
| `StdRng`, `seed_from_u64` | `rand` (existing dep) | `funfem_cluster`, `kcfc_cluster`, `bayesian_fosr` |

---

## Architecture Patterns

### Recommended helper placement

All extracted helpers are placed **immediately below the public function they serve**,
before the next public function or module-level comment. This matches the existing
pattern for `update_gmm_params_from_hard` / `update_gmm_params_from_soft` in
`clustering_advanced.rs` (lines 1097–1198) — they were extracted from the GMM
clustering logic and placed directly below the consuming function.

### Pattern: Extract-and-call

```rust
// Before:
pub fn big_fn(args...) -> Result<Out, FdarError> {
    // 50-line validation block
    // ...
    // 100-line loop body
}

// After:
pub fn big_fn(args...) -> Result<Out, FdarError> {
    let (n, m) = big_fn_validate(args...)?;
    // ...
    for _iter in 0..max_iter {
        let partial = big_fn_iter_body(...)?;
        // ...
    }
    Ok(result)
}

fn big_fn_validate(args...) -> Result<(usize, usize), FdarError> {
    // validation body verbatim
}
```

### Anti-patterns to avoid

- **Re-order floating-point loops** — the accumulation order must be preserved
  exactly (FP addition is not associative). All proposed helpers preserve loop
  order by taking the loop body verbatim.
- **Change visibility** — all helpers are `fn` (private to module), never `pub fn`
  or `pub(crate) fn`. The existing `pub(crate)` helpers (`elastic_net_cd`,
  `wnet_cv_lambda`) keep their current visibility.
- **Add `unwrap()` in helpers** — all error paths use `?` or `return Err(...)`;
  the crate norm is `Result<T, FdarError>` throughout.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Linear solves | Custom solver | `crate::linalg::cholesky_factor` + `cholesky_forward_back` (existing) | Already vectorized, tested |
| Parallel iteration | Manual thread spawning | `iter_maybe_parallel!` macro (existing) | Feature-gates rayon correctly |
| Integration weights | Custom quadrature | `crate::helpers::simpsons_weights` (existing) | Simpson rule already implemented |

---

## Common Pitfalls

### Pitfall 1: Breaking the RNG sequence in Gibbs/k-means loops

**What goes wrong:** If a helper function calls `rng.sample()` or `rng.gen()` a
different number of times (or in a different order) than the original, all
downstream samples differ. Tests that compare numeric outputs will fail.

**How to avoid:** The extracted helper must consume the RNG object by `&mut`
reference and call exactly the same draw calls in the same order. Restructuring
to avoid an early-return that was previously after a draw changes the number of
draws.

**Warning signs:** Test failures in `bayesian_fosr_recovers_beta`,
`kcfc_cluster_*`, `funfem_cluster_*` with marginal numeric differences rather
than structural errors.

### Pitfall 2: FdMatrix column-major indexing in helpers

**What goes wrong:** A helper receiving `&[f64]` treats data as row-major when
the caller intended column-major (or vice versa). Silent wrong-answer bug.

**How to avoid:** Document the layout in each helper's comment. Pass `FdMatrix`
references (not raw slices) wherever possible so the type enforces layout.
For raw slice helpers (like `funfem_project_scores`), include an explicit
`// column-major: element (i,j) at i + j*nrows` comment.

### Pitfall 3: Cognitive score surprises after extraction

**What goes wrong:** The extracted helper itself is flagged by pmat because it
inherits deeply-nested loops. The parent score drops but a new violation appears.

**How to avoid:** Count nesting depth before extraction. Any helper with 4+ levels
of nesting (for / for / for / if) will likely remain flagged. Break those into
two levels of helpers or reduce nesting by early-continue / extract inner body.
The `funfem_project_scores` triple loop (n × p_disc_eff × ncomp_eff) stays but
has no branching — its cyclomatic score is 1; only the cognitive score from the
nesting depth matters, and 3-level nesting scores ~6–9 in cognitive.

### Pitfall 4: `iter_maybe_parallel!` closures vs extracted fns

**What goes wrong:** A closure passed to `iter_maybe_parallel!` captures local
variables by reference. Extracting it as a free function requires all captures
to become explicit parameters — missing one causes a compile error.

**How to avoid:** List all captured variables before extraction. For `pace_fpca`,
the BLUP closure captures: `data`, `config`, `mean`, `ef_cols`, `eigenvalues`,
`eigenfunctions`, `actual_ncomp`, `m`, `z`, `sigma2`. All become explicit
parameters in `pace_fpca_blup_curve`. The `i` index is the parallel iterator
value and becomes the first parameter.

---

## Validation Architecture

### Behavior-preservation proof

The existing test suite is the behavior-preservation oracle:

```bash
# Full test suite (2943 tests) — the gate
cargo test -p fdars-core --features linalg,parallel 2>&1 | tail -5

# Clippy (catches visibility/lifetime issues introduced by extraction)
cargo clippy --all-targets --features linalg,parallel -- -D warnings

# Format check
cargo fmt --check
```

### Complexity-drop proof

After each function is refactored, run pmat on that file to confirm the score
dropped below threshold:

```bash
# Per-function confirmation:
pmat analyze complexity --path fdars-core/src/clustering_advanced.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/wavelet/regression.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/boosting_regression/boost_fofr.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/pace_fpca.rs 2>/dev/null
pmat analyze complexity --path fdars-core/src/boosting_regression/bayesian.rs 2>/dev/null

# Full project quality-gate (run after all 8 are done):
pmat quality-gate 2>/dev/null
```

Thresholds to clear (from pmat quality-gate output, confirmed live):
- Cyclomatic: must be ≤ 30 (error threshold)
- Cognitive: must be ≤ 25 (error threshold)

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| QUAL-01: funfem_cluster | Cluster assignments + log-likelihood unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- clustering_advanced::tests 2>&1` |
| QUAL-01: kcfc_cluster | Cluster assignments + reconstruction errors unchanged | Regression | Same as above |
| QUAL-01: elastic_net_cd | Intercept + beta coefficients unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- wavelet::regression::tests 2>&1` |
| QUAL-01: wnet_cv_lambda | Selected lambda unchanged (deterministic) | Regression | Same |
| QUAL-01: wnet | WnetResult fields unchanged | Regression | Same |
| QUAL-01: boost_fofr | Fitted values + beta surfaces unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- boosting_regression::boost_fofr::tests 2>&1` |
| QUAL-01: pace_fpca | Scores + fitted + bands unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- pace_fpca::tests 2>&1` |
| QUAL-01: bayesian_fosr | beta_mean correlation with true β unchanged | Regression | `cargo test -p fdars-core --features linalg,parallel -- boosting_regression::bayesian::tests 2>&1` |

### Sampling Rate

- **Per-helper extraction:** `cargo test -p fdars-core --features linalg,parallel -- <module>::tests` (fast, ~5–30 sec)
- **Per-function completion:** `cargo clippy --all-targets --features linalg,parallel -- -D warnings`
- **Phase gate:** Full `cargo test -p fdars-core --features linalg,parallel` + `pmat quality-gate`

---

## Hard Constraints Summary

> [VERIFIED: fdars-core/src/clustering_advanced.rs, wavelet/regression.rs, boosting_regression/bayesian.rs, boosting_regression/boost_fofr.rs, pace_fpca.rs — all read this session]

1. **No public API change.** Extracted helpers are `fn` (module-private). `pub fn` and `pub(crate) fn` signatures are untouched.
2. **Numeric output identical.** Loop order preserved; FP accumulation order preserved; no algorithm changes.
3. **No new Cargo.toml dependency.** pmat is invoked externally.
4. **`fdars-core` only.** No changes to bindings crates.
5. **Test suite stays green.** 2943 lib tests are the behavior oracle.
6. **`cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays clean.** Applies to test and bench code too.
7. **`cargo fmt --check` stays clean.** Run `cargo fmt` before committing each helper extraction.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `pmat` | Complexity verification | ✓ | 3.18.0 | — |
| `cargo` | Build + test | ✓ | (Rust 1.97.0) | — |
| `cargo clippy` | Lint gate | ✓ | same toolchain | — |
| `cargo fmt` | Format gate | ✓ | same toolchain | — |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Extracting the BLUP closure in `pace_fpca` as a named fn works with `iter_maybe_parallel!` (rayon `par_iter` requires `Send`; all captured types are `Send`) | pace_fpca decomposition | Compile error on the parallel feature; fallback is to keep the closure but factor inner sub-blocks only |
| A2 | `funfem_subspace_update` returning `Option<Vec<f64>>` (rather than mutating a `&mut Vec<f64>`) avoids borrowck issues with `disc_dirs` being both read (projected scores) and updated in the same loop | funfem_cluster decomposition | Borrow conflict; fallback is `&mut disc_dirs` parameter |
| A3 | pmat cognitive scores for the extracted helpers will be below 25 given their nesting depth | Per-function | New pmat violations on helpers; would need further decomposition |

---

## Open Questions

1. **Are there additional pmat violations introduced by the helper fns themselves?**
   - What we know: The proposed helpers have nesting depths of 1–3 levels, which score 0–9 cognitive penalty points per level.
   - What's unclear: pmat's exact cognitive formula for Rust closures-vs-fns.
   - Recommendation: After each extraction, run `pmat analyze complexity --path <file>` and verify the new helpers are clean before moving to the next function.

2. **Should `kcfc_kmeans_pp_init` and `funfem_kmeans_pp_init` share a single generic helper?**
   - What we know: Both implement D² k-means++ sampling with the same control flow but different distance functions (`l2_dist_rowmajor` with `weights` vs uniform).
   - What's unclear: Whether a generic or parametric shared helper saves enough code to be worthwhile for a pure refactor phase.
   - Recommendation: Keep them separate per function for now (Phase 106 scope). A shared generic can be deferred to Phase 109 (complexity tail) if pmat flags both helpers.

---

## Sources

### Primary (HIGH confidence)
- [VERIFIED: fdars-core/src/clustering_advanced.rs:371-1096] — kcfc_cluster + funfem_cluster source read directly
- [VERIFIED: fdars-core/src/wavelet/regression.rs:843-1257] — elastic_net_cd + wnet_cv_lambda + wnet source read directly
- [VERIFIED: fdars-core/src/boosting_regression/boost_fofr.rs:197-523] — boost_fofr source read directly
- [VERIFIED: fdars-core/src/pace_fpca.rs:266-616] — pace_fpca source read directly
- [VERIFIED: fdars-core/src/boosting_regression/bayesian.rs:101-332] — bayesian_fosr source read directly
- [VERIFIED: pmat 3.18.0 live output] — complexity scores and gate thresholds confirmed by running `pmat analyze complexity` and `pmat quality-gate`

### Secondary (MEDIUM confidence)
- [VERIFIED: .planning/phases/106-top-level-complexity-hotspots/106-CONTEXT.md] — phase constraints and decisions
- [VERIFIED: .planning/STATE.md] — milestone gates and decisions
- [VERIFIED: .planning/REQUIREMENTS.md] — QUAL-01 requirement text

---

## Metadata

**Confidence breakdown:**
- Exact file locations: HIGH — all confirmed via grep + pmat per-file output
- Complexity baselines: HIGH — live pmat 3.18.0 output
- Decomposition seams: HIGH — derived from reading source directly
- Post-extraction pmat scores: MEDIUM — estimated from nesting analysis; requires verification after extraction
- Borrow-checker hazards: HIGH for the main patterns; LOW for edge cases in parallel closures (A1, A2)

**Research date:** 2026-09-13
**Valid until:** Stable until source files are modified (next refactor iteration). pmat thresholds are pmat 3.18.0-specific.
