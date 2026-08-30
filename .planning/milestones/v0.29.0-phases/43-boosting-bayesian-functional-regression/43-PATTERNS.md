# Phase 43: Boosting / Bayesian Functional Regression - Pattern Map

**Mapped:** 2026-08-23
**Files analyzed:** 9 (6 new module files + 2 registration files + 1 prelude)
**Analogs found:** 9 / 9

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `src/boosting_regression/mod.rs` | module-barrel + config types | — | `src/gmm/mod.rs` | exact |
| `src/boosting_regression/boost_fosr.rs` | service (iterative estimation) | batch/transform | `src/function_on_scalar.rs` | exact |
| `src/boosting_regression/boost_fofr.rs` | service (iterative estimation) | batch/transform | `src/fof_regression.rs` | exact |
| `src/boosting_regression/gamlss.rs` | service (cyclic estimation) | batch/transform | `src/function_on_scalar.rs` + `boost_fosr.rs` (sibling) | role-match |
| `src/boosting_regression/bayesian.rs` | service (MCMC sampler) | batch/transform | `src/scalar_on_function/bootstrap.rs` (seeding) + `src/regression.rs` (FPCA) | role-match |
| `src/boosting_regression/stability.rs` | service (resampling wrapper) | batch/event-driven | `src/scalar_on_function/bootstrap.rs` | exact |
| `src/lib.rs` | registration (mod + re-exports) | — | `src/lib.rs` lines 104, 278 (existing patterns) | exact |
| `src/prelude.rs` | registration (re-export) | — | `src/prelude.rs` lines 15-18 (existing patterns) | exact |

---

## Pattern Assignments

### `src/boosting_regression/mod.rs` (module barrel, config + result types)

**Analog:** `src/gmm/mod.rs`

**Module header and submodule declarations** (`src/gmm/mod.rs` lines 1-20):
```rust
//! Model-based functional clustering via Gaussian mixture models.
//!
//! Key functions:
//! - [`gmm_cluster`] — Main clustering entry point …

use crate::matrix::FdMatrix;

pub mod cluster;
pub mod covariance;
pub mod em;
// …
#[cfg(test)]
mod tests;
```
Apply as:
```rust
//! Component-wise gradient boosting and Bayesian regression for functional responses.
//!
//! Implements REG-06: FDboost-style penalized functional base-learner boosting,
//! GAMLSS distributional boosting, conjugate Gibbs Bayesian FOSR, and stability selection.
//!
//! # References
//!
//! Hothorn et al. (2010). Model-Based Boosting. *Journal of Statistical Software*.
//! Hofner et al. (2016). gamboostLSS. *Journal of Statistical Software*, 74(1).
//! Jiang et al. (2025). arXiv:2505.05633 (Bayesian FoSR).
//!
//! Divergences from R baselines (FDboost 1.1-4, refund) documented per function.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

pub mod bayesian;
pub mod boost_fofr;
pub mod boost_fosr;
pub mod gamlss;
pub mod stability;
```

**Config struct shape** (`src/gmm/mod.rs` GmmClusterConfig pattern):
```rust
/// Configuration for component-wise gradient boosting.
#[derive(Debug, Clone, PartialEq)]
pub struct BoostingConfig {
    /// Number of boosting iterations.
    pub mstop: usize,
    /// Learning rate ν ∈ (0, 1] (FDboost default: 0.1).
    pub nu: f64,
    /// Number of B-spline basis functions per base-learner (must be ≥ 4).
    pub nbasis: usize,
    /// B-spline order (typically 4 for cubic).
    pub order: usize,
    /// Penalty derivative order (typically 2 for roughness).
    pub lfd_order: usize,
    /// Smoothing parameter λ for penalized base-learners.
    pub lambda: f64,
    /// Number of predictor FPC components for FoFR base-learners.
    pub ncomp_x: usize,
    /// RNG seed for stability selection (unused in pure boosting).
    pub seed: u64,
}

/// Configuration for Bayesian FOSR Gibbs sampler.
#[derive(Debug, Clone, PartialEq)]
pub struct BayesianConfig {
    /// Number of FPC components for score compression.
    pub ncomp: usize,
    /// Prior variance τ² on FPC-space coefficients (default: 100.0).
    pub tau2: f64,
    /// IG prior shape a₀ (default: 0.001).
    pub ig_a0: f64,
    /// IG prior rate b₀ (default: 0.001).
    pub ig_b0: f64,
    /// Number of Gibbs iterations after burn-in.
    pub n_iter: usize,
    /// Burn-in iterations discarded.
    pub burn_in: usize,
    /// Thinning interval (keep every `thin`-th draw).
    pub thin: usize,
    /// RNG seed (deterministic: `StdRng::seed_from_u64(seed)`).
    pub seed: u64,
}

/// Configuration for stability selection.
#[derive(Debug, Clone, PartialEq)]
pub struct StabilityConfig {
    /// Number of resamples B.
    pub n_resamples: usize,
    /// Selection threshold π ∈ (0.5, 1.0] (default: 0.9).
    pub pi_thr: f64,
    /// Base RNG seed; replicate b uses `seed.wrapping_add(b as u64)`.
    pub seed: u64,
}
```

**Result struct derives** — copy exactly from `src/function_on_scalar.rs` lines 27-28:
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoostFosrResult { … }
```

**Barrel re-exports** at bottom of `mod.rs` (follow `src/gmm/mod.rs` pattern):
```rust
pub use self::bayesian::bayesian_fosr;
pub use self::boost_fofr::boost_fofr;
pub use self::boost_fosr::boost_fosr;
pub use self::gamlss::gamlss_fosr;
pub use self::stability::stability_selection;
pub use self::mod::{
    BayesianConfig, BayesianFosrResult, BoostFofrResult, BoostFosrResult,
    BoostingConfig, GamlssResult, StabilityConfig, StabilityResult,
};
```

---

### `src/boosting_regression/boost_fosr.rs` (boosted FOSR, core algorithm)

**Analog:** `src/function_on_scalar.rs`

**Imports pattern** (`src/function_on_scalar.rs` lines 14-20):
```rust
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
```
For `boost_fosr.rs`, replace `fdata_to_pc_1d` import with `smooth_basis`:
```rust
use crate::error::FdarError;
use crate::helpers::{simpsons_weights, NUMERICAL_EPS};
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::smooth_basis::bspline_penalty_matrix;
use super::{BoostFosrResult, BoostingConfig};
```

**Penalized-solve inner helper** — copy from `src/function_on_scalar.rs` lines 121-149:
```rust
fn penalized_solve(
    xtx: &[f64],
    xty: &FdMatrix,
    penalty: &[f64],
    lambda: f64,
) -> Result<FdMatrix, FdarError> {
    let p = xty.nrows();
    let m = xty.ncols();
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }
    let l = cholesky_factor(&a, p)?;
    let mut beta = FdMatrix::zeros(p, m);
    for t in 0..m {
        let b: Vec<f64> = (0..p).map(|j| xty[(j, t)]).collect();
        let x = cholesky_forward_back(&l, &b, p);
        for j in 0..p {
            beta[(j, t)] = x[j];
        }
    }
    Ok(beta)
}
```
For the boosting base-learner, the same structure applies but: (1) the "xty" is `Φ_j' · U` (residual at iteration m), (2) the Cholesky factor `L_j` is cached outside the time-point loop (amortize across `m_t`).

**Pointwise R² helper** — copy verbatim from `src/function_on_scalar.rs` lines 152-168:
```rust
pub(crate) fn pointwise_r_squared(data: &FdMatrix, fitted: &FdMatrix) -> Vec<f64> {
    let (n, m) = data.shape();
    (0..m)
        .map(|t| {
            let mean_t: f64 = (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64;
            let ss_tot: f64 = (0..n).map(|i| (data[(i, t)] - mean_t).powi(2)).sum();
            let ss_res: f64 =
                (0..n).map(|i| (data[(i, t)] - fitted[(i, t)]).powi(2)).sum();
            if ss_tot > 1e-15 { 1.0 - ss_res / ss_tot } else { 0.0 }
        })
        .collect()
}
```

**Public function signature shape** (mirror `src/function_on_scalar.rs` `fosr()` at line 184+):
```rust
/// Component-wise gradient boosting for function-on-scalar regression.
///
/// # Arguments
/// * `data` — Functional response Y (n × m_t)
/// * `predictors` — Scalar predictor matrix (n × p); one base-learner per column
/// * `argvals` — Response grid (length m_t)
/// * `config` — [`BoostingConfig`] (mstop, nu, nbasis, lambda, …)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] on shape mismatch.
/// Returns [`FdarError::InvalidParameter`] if `mstop == 0`, `nu <= 0` or `nu > 1`,
/// `nbasis < 4`, or `lambda <= 0`.
/// Returns [`FdarError::ComputationFailed`] if any base-learner Cholesky fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BoostingConfig,
) -> Result<BoostFosrResult, FdarError> { … }
```

**Error handling pattern** (copy from `src/scalar_on_function/bootstrap.rs` lines 62-80):
```rust
let (n, m) = data.shape();
if n < 3 || m == 0 || predictors.nrows() != n {
    return Err(FdarError::InvalidDimension {
        parameter: "data/predictors",
        expected: format!("n >= 3, m > 0, predictors.nrows() == n (n={n})"),
        actual: format!("n={}, m={}, predictors.nrows()={}", n, m, predictors.nrows()),
    });
}
if config.mstop == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "mstop",
        message: "must be >= 1".to_string(),
    });
}
```

**Column-major access pattern** (from `src/function_on_scalar.rs` pointwise loop structure + Pitfall 5 note):
```rust
// Iterate columns (time points) in outer loop — contiguous memory access in FdMatrix
for t in 0..m_t {
    let col_u = residuals.column(t);   // zero-copy contiguous slice, length n
    // … inner computation on col_u …
}
```

**Inline test structure** (from `src/scalar_on_function/bootstrap.rs`, `src/function_on_scalar.rs`):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use crate::matrix::FdMatrix;

    #[test]
    fn boost_fosr_reduces_rss_monotonically() { … }

    #[test]
    fn boost_fosr_recovers_known_beta() { … }

    #[test]
    fn boost_fosr_errors_on_dimension_mismatch() {
        // shape mismatch → FdarError::InvalidDimension
    }
}
```

---

### `src/boosting_regression/boost_fofr.rs` (boosted FoFR)

**Analog:** `src/fof_regression.rs`

**Imports pattern** (`src/fof_regression.rs` lines 18-22):
```rust
use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
```
For `boost_fofr.rs`, add boosting-config import:
```rust
use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use super::{BoostFofrResult, BoostingConfig};
```

**Result struct shape** (mirror `src/fof_regression.rs` lines 28-53):
```rust
/// Result of boosted function-on-function regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoostFofrResult {
    /// Intercept function F_0(t) (length m_y)
    pub intercept: Vec<f64>,
    /// Fitted response curves (n × m_y)
    pub fitted: FdMatrix,
    /// Residual curves (n × m_y)
    pub residuals: FdMatrix,
    /// R-squared per response grid point (length m_y)
    pub r_squared_t: Vec<f64>,
    /// Overall R-squared
    pub r_squared: f64,
    /// FPCA of each functional predictor (one per predictor)
    pub fpca_x: Vec<FpcaResult>,   // mirrors fof_regression.rs FofResult.fpca_x
    /// Accumulated FPC-space score coefficients per predictor: Vec[j] is (K_j × m_y)
    pub score_coefs: Vec<FdMatrix>,
    /// Reconstructed coefficient surfaces β_j(s,t) per predictor: Vec[j] is (m_x × m_y)
    pub beta_surfaces: Vec<FdMatrix>,
    /// Which base-learner was selected at each boosting iteration
    pub selected_learners: Vec<usize>,
    /// GCV path (length mstop)
    pub gcv_path: Vec<f64>,
    /// Boosting iterations used
    pub mstop: usize,
    /// Learning rate used
    pub nu: f64,
}
```

**Double-FPCA precomputation** (mirror `src/fof_regression.rs` pattern at line 59+):
```rust
// Preprocessing: compute FPC scores for each functional predictor
let fpca_list: Vec<FpcaResult> = predictors_x
    .iter()
    .map(|(x, argvals_x)| fdata_to_pc_1d(x, config.ncomp_x, argvals_x))
    .collect::<Result<Vec<_>, _>>()?;
```

**Function signature** (mirror `fof_regression`):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fofr(
    x_data: &[&FdMatrix],          // slice of functional predictors
    x_argvals: &[&[f64]],          // evaluation grids for each predictor
    y_data: &FdMatrix,             // functional response (n × m_y)
    y_argvals: &[f64],             // response grid
    config: &BoostingConfig,
) -> Result<BoostFofrResult, FdarError> { … }
```

---

### `src/boosting_regression/gamlss.rs` (GAMLSS distributional boosting)

**Analog:** `src/function_on_scalar.rs` (structure) + `boost_fosr.rs` (sibling, calling boosting core)

**Imports**:
```rust
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::matrix::FdMatrix;
use super::boost_fosr::boost_fosr_one_step;   // internal: one boosting iteration
use super::{BoostingConfig, GamlssResult};
```

**Result struct shape**:
```rust
/// Result of GAMLSS-style distributional functional regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GamlssResult {
    /// Fitted location μ̂(t) per observation (n × m_t)
    pub mu_fitted: FdMatrix,
    /// Fitted scale σ̂(t) per observation (n × m_t); always positive
    pub sigma_fitted: FdMatrix,
    /// Intercept for μ model (length m_t)
    pub mu_intercept: Vec<f64>,
    /// Intercept for log-σ model (length m_t)
    pub sigma_intercept: Vec<f64>,
    /// Accumulated μ coefficient functions (p × m_t)
    pub mu_beta: FdMatrix,
    /// Accumulated log-σ coefficient functions (p × m_t)
    pub sigma_beta: FdMatrix,
    /// Final Gaussian log-likelihood
    pub log_likelihood: f64,
    /// Log-likelihood per cyclic iteration (length mstop)
    pub ll_path: Vec<f64>,
    /// Boosting iterations used
    pub mstop: usize,
    /// Learning rate used
    pub nu: f64,
}
```

**σ clipping guard** (from Pitfall 2, mirrors NUMERICAL_EPS usage in `src/helpers.rs`):
```rust
use crate::helpers::NUMERICAL_EPS;
// After every σ update:
for t in 0..m_t {
    for i in 0..n {
        sigma_fitted[(i, t)] = sigma_fitted[(i, t)].max(NUMERICAL_EPS);
    }
}
```

**Gaussian negative-gradient helpers** (GAMLSS inner functions, no analog — new logic):
```rust
/// Negative gradient w.r.t. μ (identity link): (Y − μ) / σ²
fn mu_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> FdMatrix { … }

/// Negative gradient w.r.t. log σ (log link): −1 + (Y − μ)² / σ²
fn sigma_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> FdMatrix { … }
```

**Public function signature**:
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gamlss_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BoostingConfig,
) -> Result<GamlssResult, FdarError> { … }
```

---

### `src/boosting_regression/bayesian.rs` (conjugate Gibbs FOSR)

**Analog:** `src/scalar_on_function/bootstrap.rs` (seeding + resample loop shape) + `src/regression.rs` (FPCA)

**Imports pattern** (`src/scalar_on_function/bootstrap.rs` lines 1-8):
```rust
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use rand::prelude::*;
// Add for Bayesian draws:
use rand_distr::{Gamma, Normal, StandardNormal};
use crate::linalg::{cholesky_factor, cholesky_forward_back};
use crate::regression::fdata_to_pc_1d;
use super::{BayesianConfig, BayesianFosrResult};
```

**RNG seeding pattern** (`src/scalar_on_function/bootstrap.rs` line 89):
```rust
// Single-chain Gibbs — one RNG, seeded at function entry
let mut rng = StdRng::seed_from_u64(config.seed);
```
(Not per-iteration: the Gibbs chain is sequential; contrast with bootstrap/stability which seed per replicate.)

**Result struct shape**:
```rust
/// Result of Bayesian function-on-scalar regression via conjugate Gibbs.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BayesianFosrResult {
    /// Posterior mean coefficient functions β̄(t), p × m_t
    pub beta_mean: FdMatrix,
    /// Pointwise 2.5% credible band, p × m_t
    pub beta_lower: FdMatrix,
    /// Pointwise 97.5% credible band, p × m_t
    pub beta_upper: FdMatrix,
    /// Posterior-mean fitted values (n × m_t)
    pub fitted: FdMatrix,
    /// Posterior-mean residuals (n × m_t)
    pub residuals: FdMatrix,
    /// Posterior mean σ²(t) (length m_t)
    pub sigma2_mean: Vec<f64>,
    /// Gibbs iterations run after burn-in
    pub n_iter: usize,
    /// Burn-in discarded
    pub burn_in: usize,
    /// Thinning interval
    pub thin: usize,
    /// FPC components used
    pub ncomp: usize,
}
```

**FPCA preprocessing** (from `src/regression.rs` `fdata_to_pc_1d`, lines 287-292):
```rust
pub fn fdata_to_pc_1d(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FpcaResult, FdarError>
```
Call as: `let fpca = fdata_to_pc_1d(data, config.ncomp, argvals)?;`
Score matrix `S = fpca.scores` is `n × K`.

**Quantile helper** (from `src/scalar_on_function/bootstrap.rs` line 26):
```rust
use crate::helpers::quantile_sorted as quantile;
// Apply per time point over retained draws:
let mut vals: Vec<f64> = draws.iter().map(|draw| draw[t]).collect();
vals.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
lower[(pred, t)] = quantile(&vals, 0.025);
upper[(pred, t)] = quantile(&vals, 0.975);
```

**Public function signature**:
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bayesian_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BayesianConfig,
) -> Result<BayesianFosrResult, FdarError> { … }
```

---

### `src/boosting_regression/stability.rs` (stability selection wrapper)

**Analog:** `src/scalar_on_function/bootstrap.rs` (exact pattern: seeded parallel resampling)

**Imports** (`src/scalar_on_function/bootstrap.rs` lines 1-9):
```rust
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use super::boost_fosr::boost_fosr;
use super::{BoostingConfig, StabilityConfig, StabilityResult};
```

**Per-replicate RNG seeding** (`src/scalar_on_function/bootstrap.rs` line 88-89):
```rust
let counts: Vec<Vec<bool>> = iter_maybe_parallel!(0..config.n_resamples)
    .map(|b| {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(b as u64));
        // subsample floor(n/2) indices WITHOUT replacement:
        let subsample_size = n / 2;
        let indices = sample_without_replacement(&mut rng, n, subsample_size);
        // … run boost_fosr on subsample … collect selected_learners …
    })
    .collect();
```

**Subsampling without replacement** (no existing exact analog — implement as private helper):
```rust
fn sample_without_replacement(rng: &mut StdRng, n: usize, k: usize) -> Vec<usize> {
    // Fisher-Yates partial shuffle
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = rng.gen_range(i..n);
        indices.swap(i, j);
    }
    indices[..k].to_vec()
}
```

**Result struct**:
```rust
/// Result of FDboost-style stability selection.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct StabilityResult {
    /// Selection frequency π̂[j] ∈ [0,1] for each base-learner j = 0..p
    pub selection_freq: Vec<f64>,
    /// Indices j where π̂[j] >= pi_thr
    pub stable_set: Vec<usize>,
    /// Threshold used
    pub pi_thr: f64,
    /// PFER upper bound: q² / ((2·pi_thr − 1)·p)
    pub pfer_bound: f64,
    /// Number of resamples B used
    pub n_resamples: usize,
}
```

**Public function signature**:
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn stability_selection(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    boost_config: &BoostingConfig,
    stab_config: &StabilityConfig,
) -> Result<StabilityResult, FdarError> { … }
```

---

### `src/lib.rs` (module registration + crate-root re-exports)

**Analog:** `src/lib.rs` — existing module list (lines 64-135) and re-export block (lines 138+)

**Module declaration** — insert after `pub mod regression;` (currently line 104):
```rust
pub mod boosting_regression;
```

**Crate-root re-exports** — insert in the re-export block (after line 278, following the `concurrent_regression` block pattern at line 278-279):
```rust
// Re-export boosting and Bayesian functional regression types (Phase 43 REG-06)
pub use boosting_regression::{
    bayesian_fosr, boost_fofr, boost_fosr, gamlss_fosr, stability_selection,
    BayesianConfig, BayesianFosrResult, BoostFofrResult, BoostFosrResult,
    BoostingConfig, GamlssResult, StabilityConfig, StabilityResult,
};
```

---

### `src/prelude.rs` (prelude re-exports)

**Analog:** `src/prelude.rs` lines 15-18 (regression results block):
```rust
// Regression results
pub use crate::function_on_scalar::FosrResult;
#[cfg(feature = "linalg")]
pub use crate::regression::RidgeResult;
pub use crate::regression::{FpcaResult, PlsResult};
pub use crate::scalar_on_function::{FregreLmResult, FunctionalLogisticResult};
```
Add after this block:
```rust
// Boosting and Bayesian functional regression results (Phase 43 REG-06)
pub use crate::boosting_regression::{BayesianFosrResult, BoostFosrResult};
```

---

## Shared Patterns

### Column-major FdMatrix access
**Source:** `src/function_on_scalar.rs` lines 152-168 (pointwise loop), `src/matrix.rs` (column method)
**Apply to:** `boost_fosr.rs`, `boost_fofr.rs`, `gamlss.rs`, `bayesian.rs`
```rust
// Column (time-point t) access — contiguous, preferred:
let col = data.column(t);   // &[f64], length n

// Row access via buffer — use when row is needed:
let mut buf = vec![0.0; m];
data.row_to_buf(i, &mut buf);
```
Never iterate `data[(i, t)]` for varying `t` at fixed `i` in a hot inner loop — that is strided.

### Error handling (dimension + parameter validation)
**Source:** `src/scalar_on_function/bootstrap.rs` lines 62-80
**Apply to:** All public `fn` in all 5 submodules
```rust
let (n, m) = data.shape();
if n < 3 || m == 0 || predictors.nrows() != n {
    return Err(FdarError::InvalidDimension {
        parameter: "data/predictors",
        expected: format!("n >= 3, m > 0, predictors.nrows() == n (n={n})"),
        actual: format!("n={n}, m={m}, predictors.nrows()={}", predictors.nrows()),
    });
}
if config.lambda <= 0.0 {
    return Err(FdarError::InvalidParameter {
        parameter: "lambda",
        message: format!("must be > 0, got {}", config.lambda),
    });
}
```
Pattern: dimension checks first, parameter checks second, using `FdarError::InvalidDimension` and `FdarError::InvalidParameter` variants with descriptive context.

### Penalized Cholesky solve (base-learner fit per time point)
**Source:** `src/function_on_scalar.rs` lines 121-149 (`penalized_solve`) + `src/linalg.rs` lines 85-134
**Apply to:** `boost_fosr.rs`, `boost_fofr.rs`, `gamlss.rs`
```rust
// Factor once outside the time-point loop:
let l = cholesky_factor(&a, k)?;   // a = Φ'Φ + λR + 1e-10·I (ridge jitter)

// Solve for each time point t inside the loop:
for t in 0..m_t {
    let rhs = phi_transpose_times_col(&phi, residuals.column(t), n, k);
    let c_t = cholesky_forward_back(&l, &rhs, k);
    // accumulate c_t into coefficient matrix
}
```

### Seeded parallel resampling
**Source:** `src/scalar_on_function/bootstrap.rs` lines 87-100
**Apply to:** `stability.rs` (resample loop), `bayesian.rs` (single-chain, sequential)
```rust
// Parallel resamples (stability.rs):
let results: Vec<_> = iter_maybe_parallel!(0..config.n_resamples)
    .map(|b| {
        let mut rng = StdRng::seed_from_u64(config.seed.wrapping_add(b as u64));
        // each closure owns its own RNG — no sharing across threads
        …
    })
    .collect();

// Sequential chain (bayesian.rs):
let mut rng = StdRng::seed_from_u64(config.seed);
for iter in 0..(config.burn_in + config.n_iter * config.thin) { … }
```

### `#[must_use]` + derive boilerplate
**Source:** `src/scalar_on_function/bootstrap.rs` line 51; `src/function_on_scalar.rs` line 27-28
**Apply to:** All 5 public functions + all 5 result structs
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fosr(…) -> Result<BoostFosrResult, FdarError> { … }

#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoostFosrResult { … }
```

### Pointwise R² and GCV
**Source:** `src/function_on_scalar.rs` lines 152-178
**Apply to:** `boost_fosr.rs`, `boost_fofr.rs` (copy `pointwise_r_squared`; adapt `compute_fosr_gcv` for per-iteration path tracking)

---

## No Analog Found

All files have analogs. The following functions within new files have no existing analog (new algorithm logic):

| Function | File | Reason |
|---|---|---|
| `mu_neg_gradient` / `sigma_neg_gradient` | `gamlss.rs` | GAMLSS Gaussian negative gradients — no distributional boosting exists in codebase |
| `gibbs_draw_gamma` / `gibbs_draw_sigma2` | `bayesian.rs` | Conjugate Gibbs full-conditional draws — no MCMC sampler exists in codebase |
| `sample_without_replacement` | `stability.rs` | Fisher-Yates partial shuffle — subsampling without replacement not needed elsewhere |

For these, use the algorithm derivations in RESEARCH.md (Algorithms 3, 4, 5) directly. The numerical primitives they depend on (`cholesky_factor`, `rand_distr::Gamma`, `rand_distr::Normal`) are all in the analog patterns above.

---

## Open Questions for Planner

Two assumptions from RESEARCH.md require verification before Wave 1 / Wave 4 plans are written:

1. **A4 — `bspline_basis` signature** (`src/basis.rs`): verify that it evaluates at arbitrary predictor values (not just a uniform grid). The planner should read `src/basis.rs` and document the exact call for `build_bspline_design_at(x_vals, nbasis, order)`.

2. **A3 — `rand_distr::Gamma` availability**: verify `rand_distr` is a direct or transitive dependency in `fdars-core/Cargo.toml`. If not available, the IG draw in `bayesian.rs` must use an alternative (e.g., rejection sampling or Box–Muller trick).

---

## Metadata

**Analog search scope:** `fdars-core/src/` — all submodules
**Files scanned:** 9 source files read; 4 additional via Bash/Grep
**Pattern extraction date:** 2026-08-23

---

## PATTERN MAPPING COMPLETE

**Phase:** 43 - Boosting / Bayesian Functional Regression
**Files classified:** 9
**Analogs found:** 9 / 9 (all files have structural analogs; 3 internal functions are new logic)

### Coverage
- Files with exact analog: 4 (`mod.rs`→`gmm/mod.rs`, `boost_fosr.rs`→`function_on_scalar.rs`, `stability.rs`→`bootstrap.rs`, `lib.rs`/`prelude.rs` registrations)
- Files with role-match analog: 5 (`boost_fofr.rs`→`fof_regression.rs`, `gamlss.rs`→`function_on_scalar.rs`+sibling, `bayesian.rs`→`bootstrap.rs`+`regression.rs`)
- Files with no analog: 0

### Key Patterns Identified
- All 5 submodule algorithms share the `penalized_solve` / `cholesky_factor` + `cholesky_forward_back` pattern from `src/function_on_scalar.rs` and `src/linalg.rs`
- Seeded resampling follows `StdRng::seed_from_u64(seed.wrapping_add(b as u64))` per replicate from `src/scalar_on_function/bootstrap.rs` line 89
- All result structs follow `FosrResult` field convention (`intercept`, `beta`, `fitted`, `residuals`, `r_squared_t`, `r_squared`) from `src/function_on_scalar.rs` lines 29-48
- Config structs follow `GmmClusterConfig` builder pattern from `src/gmm/mod.rs`
- Column-major access: use `data.column(t)` (contiguous) in outer time-point loop, never strided row iteration in hot path
- Parallel resample loops use `iter_maybe_parallel!` macro; per-thread RNG created locally inside closure

### File Created
`/home/simonm/projects/rust/fdars/.planning/phases/43-boosting-bayesian-functional-regression/43-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
