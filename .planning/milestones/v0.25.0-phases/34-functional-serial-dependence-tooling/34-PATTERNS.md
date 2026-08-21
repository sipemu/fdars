# Phase 34: Functional Serial-Dependence Tooling - Pattern Map

**Mapped:** 2026-08-21
**Files analyzed:** 3 (2 new, 1 modified)
**Analogs found:** 3 / 3

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/fts/mod.rs` | module barrel | — | `fdars-core/src/inference/mod.rs` | exact |
| `fdars-core/src/fts/acf.rs` | computation module | batch, request-response | `fdars-core/src/inference/permutation.rs` | exact |
| `fdars-core/src/lib.rs` (modified) | public API surface | — | `fdars-core/src/lib.rs` lines 94, 226–231 | exact |

---

## Pattern Assignments

### `fdars-core/src/fts/mod.rs` (module barrel)

**Analog:** `fdars-core/src/inference/mod.rs`

**Module doc + submodule wiring pattern** (lines 1–42):
```rust
//! Functional time series serial-dependence diagnostics.
//!
//! # R baselines
//! * [`functional_acf`] / [`functional_pacf`] — `fdaACF::facf`
//!   (Mestre et al. 2021, *Computational Statistics & Data Analysis*)
//! * [`stationarity_test`] — `ftsa::T_stationary` (Horváth, Kokoszka, Rice 2014)
//! * [`long_run_covariance`] — `ftsa::long_run_covariance_estimation` (Bartlett HAC)
//! * [`functional_difference`] — `ftsa::diff.fts`
//!
//! # Conventions
//!
//! Entry points take an explicit deterministic `seed` (`StdRng::seed_from_u64(seed)`)
//! and default Monte-Carlo replications of 999. All public functions return
//! `Result<_, FdarError>` and validate inputs at entry. Result structs derive
//! `Debug, Clone, PartialEq` and are serde-gated.

mod acf;

pub use acf::{
    functional_acf, functional_pacf, functional_difference,
    stationarity_test, long_run_covariance,
    FacfResult, StationarityResult, LongRunCovResult,
};
```

**Result struct pattern** (inference/mod.rs lines 44–59 — copy and extend):
```rust
/// Result of functional ACF/PACF estimation.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag.
    pub acf: Vec<f64>,
    /// Functional partial autocorrelation (Durbin-Levinson scalar approximation).
    pub pacf: Vec<f64>,
    /// Upper confidence band under the strong-white-noise null (MC quantile).
    pub upper_band: Vec<f64>,
}

/// Result of the functional stationarity test.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct StationarityResult {
    /// Test statistic T (KPSS-style partial-sum L2 norm).
    pub statistic: f64,
    /// Monte-Carlo p-value.
    pub p_value: f64,
    /// Number of permutations used.
    pub n_perm: usize,
}

/// Result of the Bartlett kernel-sandwich long-run covariance estimator.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LongRunCovResult {
    /// Estimated m×m long-run covariance matrix (column-major flat Vec).
    pub cov_matrix: Vec<f64>,
    /// Grid dimension m (cov_matrix is m×m).
    pub m: usize,
    /// Bandwidth used.
    pub bandwidth: usize,
    /// Number of curves N.
    pub n_curves: usize,
}
```

---

### `fdars-core/src/fts/acf.rs` (computation module, batch/request-response)

**Analog:** `fdars-core/src/inference/permutation.rs`

**Imports pattern** (permutation.rs lines 9–16 — adapt for fts/acf.rs):
```rust
use super::{FacfResult, LongRunCovResult, StationarityResult};
use crate::error::FdarError;
use crate::helpers::{simpsons_weights, trapz, cumulative_trapz, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
```

**Input validation pattern** (permutation.rs lines 24–60):
```rust
fn validate_fts_input(
    data: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    Ok((n, m))
}
```

**Seeded RNG + Fisher-Yates shuffle pattern** (permutation.rs lines 119–127, 173):
```rust
// RNG construction — always single shared seed (NOT per-lag seed+k):
let mut rng = StdRng::seed_from_u64(seed);

// Fisher-Yates shuffle of row index vector:
fn shuffle_indices(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}
```

**MC p-value formula** (permutation.rs line 183):
```rust
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

**Public function signature pattern** (permutation.rs lines 152–188 — `t_perm_test`):
```rust
/// Functional autocorrelation and partial autocorrelation.
///
/// Computes the L2-norm fACF at lags 1..=max_lag following the `fdaACF` convention,
/// scalar Durbin-Levinson fPACF, and Monte-Carlo strong-white-noise confidence bands.
///
/// # Arguments
/// * `data` - Time-ordered functional observations (`N × m`, column-major).
/// * `argvals` - Evaluation points (length `m`).
/// * `max_lag` - Maximum lag to compute (default: `min(20, N/4)`). `None` uses the default.
/// * `n_sim` - MC replications for the white-noise band (default 999).
/// * `ci` - Confidence level for the band (default 0.95).
/// * `seed` - Deterministic RNG seed.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `data` is empty, `argvals` length
/// mismatches `data` columns, or `N < max_lag + 1`. Returns
/// [`FdarError::ComputationFailed`] if the lag-0 covariance diagonal is degenerate.
pub fn functional_acf(
    data: &FdMatrix,
    argvals: &[f64],
    max_lag: Option<usize>,
    n_sim: usize,
    ci: f64,
    seed: u64,
) -> Result<FacfResult, FdarError> {
    // validate → compute → return Ok(FacfResult { ... })
}
```

**Column-major FdMatrix access pattern** (matrix.rs lines 84–143 — critical for the hot inner loop):
```rust
// shape:
let (n, m) = data.shape();

// Element access (column-major index: row + col*nrows):
let val = data[(i, j)];

// Zero-copy column slice:
let col_j: &[f64] = data.column(j);

// Allocate result matrix (column-major):
let mut c_h = FdMatrix::zeros(m, m);
// OR as flat Vec with manual indexing:
let mut c_h = vec![0.0f64; m * m];
// access: c_h[j1 + j2 * m]  (j1 = row, j2 = col)
```

**helpers quadrature pattern** (helpers.rs lines 37–86):
```rust
// Integration weights:
let weights = simpsons_weights(argvals);

// Scalar integration of a 1-D slice (e.g., diagonal of C_0):
let norm_denom = trapz(&diag_c0, argvals);

// Degenerate-denominator guard (use NUMERICAL_EPS from helpers):
if norm_denom.abs() < NUMERICAL_EPS {
    return Err(FdarError::ComputationFailed {
        operation: "functional_acf",
        detail: "lag-0 covariance diagonal integrates to near zero (degenerate data)".to_string(),
    });
}
```

**MC chi-squared mixture band pattern** (from RESEARCH.md §Code Examples — new computation, no existing analog):
```rust
use rand_distr::{ChiSquared, Distribution};

fn mc_band_threshold(
    eigenvalues: &[f64],
    n: usize,
    n_sim: usize,
    ci: f64,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let chi2 = ChiSquared::new(1.0).expect("df=1 is valid");
    let mut realizations = Vec::with_capacity(n_sim);
    for _ in 0..n_sim {
        let mut q = 0.0f64;
        for &lj in eigenvalues {
            for &lk in eigenvalues {
                q += lj * lk * chi2.sample(&mut rng);
            }
        }
        realizations.push(q / n as f64);
    }
    realizations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((ci * n_sim as f64) as usize).min(n_sim - 1);
    realizations[idx]
}
```

**Test structure pattern** (permutation.rs lines 254–418):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::{generate_gaussian_process, CovKernel};
    use crate::test_helpers::uniform_grid;

    // White-noise test data via existing covariance module:
    fn make_whitenoise_curves(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let kernel = CovKernel::WhiteNoise { variance: 1.0 };
        let gp = generate_gaussian_process(&argvals, &kernel, n, seed).unwrap();
        (gp.samples, argvals)
    }

    // Seeded determinism test (mirrors permutation.rs t_perm_deterministic):
    #[test]
    fn deterministic_seed() {
        let (data, argvals) = make_whitenoise_curves(50, 20, 1);
        let r1 = functional_acf(&data, &argvals, None, 200, 0.95, 42).unwrap();
        let r2 = functional_acf(&data, &argvals, None, 200, 0.95, 42).unwrap();
        assert_eq!(r1, r2, "same seed must give bit-identical result");
    }

    // Error-handling test (mirrors permutation.rs t_perm_invalid_input):
    #[test]
    fn error_handling() {
        let argvals = uniform_grid(20);
        let empty = FdMatrix::zeros(0, 20);
        assert!(matches!(
            functional_acf(&empty, &argvals, None, 99, 0.95, 1),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
```

---

### `fdars-core/src/lib.rs` (modified — add fts module + re-exports)

**Analog:** `fdars-core/src/lib.rs` lines 94 (`pub mod inference;`) and 226–231 (inference re-export block)

**`pub mod` registration pattern** (line 94 area):
```rust
// Add in alphabetical position (after `fof_regression`, before `function_on_scalar`):
pub mod fts;
```

**`pub use` re-export block pattern** (lines 226–231):
```rust
// Re-export functional time series serial-dependence types
pub use fts::{
    functional_acf, functional_pacf, functional_difference,
    stationarity_test, long_run_covariance,
    FacfResult, StationarityResult, LongRunCovResult,
};
```

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/inference/permutation.rs` lines 31–60, `fdars-core/src/error.rs`
**Apply to:** All 5 entry points in `fts/acf.rs`
```rust
// Dimension mismatch:
return Err(FdarError::InvalidDimension {
    parameter: "argvals",
    expected: format!("{m} elements (matching data columns)"),
    actual: format!("{} elements", argvals.len()),
});

// Out-of-range parameter (e.g. negative bandwidth expressed as 0 check, n_perm=0):
return Err(FdarError::InvalidParameter {
    parameter: "n_perm",
    message: "must be >= 1".to_string(),
});

// Numerical failure (degenerate covariance):
return Err(FdarError::ComputationFailed {
    operation: "functional_acf",
    detail: "lag-0 covariance diagonal integrates to near zero".to_string(),
});
```

### Seeded Monte-Carlo
**Source:** `fdars-core/src/inference/permutation.rs` lines 173, 183
**Apply to:** `functional_acf` (MC band), `stationarity_test` (permutation p-value)
```rust
let mut rng = StdRng::seed_from_u64(seed);
// ... loop ...
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

### Result Struct Derives
**Source:** `fdars-core/src/inference/mod.rs` lines 49–51
**Apply to:** `FacfResult`, `StationarityResult`, `LongRunCovResult`
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ...
```

### Quadrature (Integration Weights)
**Source:** `fdars-core/src/helpers.rs` lines 37–86
**Apply to:** Every L2-norm computation in `fts/acf.rs`
```rust
use crate::helpers::{simpsons_weights, trapz, cumulative_trapz, NUMERICAL_EPS};
let weights = simpsons_weights(argvals);
// Hilbert-Schmidt norm: Σ_{j1,j2} c_h[j1+j2*m]^2 * weights[j1] * weights[j2]
// Normalization denom: trapz(&diag_c0, argvals)
// Differencing round-trip: cumulative_trapz (used in tests, not main path)
```

### Column-Major Matrix Access
**Source:** `fdars-core/src/matrix.rs` lines 84–143
**Apply to:** Autocovariance inner loop in `fts/acf.rs`
```rust
// Prefer direct index in hot loop (no allocation):
let xi1 = data[(i, j1)] - xbar[j1];
// c_h stored column-major: c_h[j1 + j2 * m]
// Allocate: FdMatrix::zeros(m, m) or vec![0.0f64; m * m]
```

---

## No Analog Found

| File / Component | Role | Data Flow | Reason |
|------------------|------|-----------|--------|
| `mc_band_threshold` (internal fn in acf.rs) | chi²-mixture MC | batch | No existing chi-squared mixture sampling in codebase; use RESEARCH.md §Code Examples pattern + `rand_distr::ChiSquared` |
| Durbin-Levinson PACF recursion (internal fn in acf.rs) | scalar recursion | transform | Classical algorithm, not present in codebase; use RESEARCH.md §Algorithm Formulations §4 |
| Bartlett HAC accumulation (internal fn in acf.rs) | batch | transform | Long-run covariance is new; use RESEARCH.md §Algorithm Formulations §6 |

---

## Metadata

**Analog search scope:** `fdars-core/src/inference/`, `fdars-core/src/helpers.rs`, `fdars-core/src/matrix.rs`, `fdars-core/src/lib.rs`
**Files scanned:** 5 (inference/mod.rs, inference/permutation.rs, helpers.rs, matrix.rs, lib.rs)
**Pattern extraction date:** 2026-08-21
