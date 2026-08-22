# Phase 39: Functional Time-Series Forecasting — Pattern Map

**Mapped:** 2026-08-22
**Files analyzed:** 3 (1 new, 2 modified)
**Analogs found:** 3 / 3

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `fdars-core/src/fts/forecast.rs` | service (algorithm module) | request-response (fit + predict) | `fdars-core/src/fts/acf.rs` | exact — sibling FTS module, same role+data-flow |
| `fdars-core/src/fts/mod.rs` | config (module wiring) | n/a | `fdars-core/src/fts/mod.rs` (current state) | exact — additive edit only |
| `fdars-core/src/lib.rs` | config (crate-root re-export) | n/a | `fdars-core/src/lib.rs` lines 251–255 | exact — additive edit only |

---

## Pattern Assignments

### `fdars-core/src/fts/forecast.rs` (new file — algorithm module, request-response)

**Primary analog:** `fdars-core/src/fts/acf.rs`
**Secondary analogs:** `fdars-core/src/regression.rs` (FpcaResult struct + methods), `fdars-core/src/scalar_on_function/pls.rs` (fregre_pls validation pattern)

---

#### Module-doc convention (copy from `acf.rs` lines 1–18)

```rust
//! Functional time-series forecasting via FPCA decomposition and AR score models.
//!
//! Implements the Hyndman-Shang functional time-series model (`ftsm`) and five
//! entry points matching the R `ftsa` package: `ftsm` (fit), `ftsm_forecast`
//! (h-step forecast), `fplsr` (functional PLS forecasting variant), `ftsm_update`
//! (dynamic update without FPCA refit), and `ftsm_forecast_multistep` (iterative
//! h > 1 forecasts). All entry points return `Result<_, FdarError>`.
//!
//! # R baseline
//!
//! Hyndman & Shang (2009), `ftsa::ftsm`, `ftsa::fplsr`, `ftsa::dynupdate`,
//! `ftsa::ftsmiterativeforecasts`.
//!
//! # Divergences from R `ftsa`
//!
//! * No pre-smoothing before FPCA — operates on the raw input grid.
//! * `ncomp` is user-provided (no default); provide `6` to match ftsa default.
//! * AR score model only (Yule-Walker + AIC); no ETS/ARIMA/rwdrift.
//! * `fplsr` uses per-evaluation-point scalar PLS (not NIPALS functional operator).
//! * Prediction intervals are out of scope (numeric point forecasts only).
```

---

#### Imports pattern (copy from `acf.rs` lines 13–18, adapt)

```rust
use super::{FtsmForecastResult, FtsmResult, FplsrResult};
use crate::error::FdarError;
use crate::helpers::{simpsons_weights, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, fdata_to_pls_1d};
```

`rand` / `rand_distr` imports are only needed if a stochastic path is introduced (none planned for this phase — Yule-Walker is fully deterministic).

---

#### Input validation pattern (copy from `acf.rs` lines 25–42)

```rust
// from acf.rs:25-42 — exact pattern to replicate in forecast.rs
fn validate_fts_input(data: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError> {
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

Re-implement this private helper verbatim in `forecast.rs` (do not promote to `pub(crate)` in `acf.rs` — avoids touching shipped code).

---

#### Private numeric helper pattern — `mean_curve` (copy from `acf.rs` lines 47–58)

```rust
// from acf.rs:47-58 — re-implement verbatim (private to acf.rs)
fn mean_curve(data: &FdMatrix, n: usize, m: usize) -> Vec<f64> {
    let mut xbar = vec![0.0f64; m];
    let inv_n = 1.0 / n as f64;
    for j in 0..m {
        let mut s = 0.0;
        for i in 0..n {
            s += data[(i, j)];
        }
        xbar[j] = s * inv_n;
    }
    xbar
}
```

---

#### Scalar autocovariance helper (new — informed by `acf.rs` lines 63–95 normalization convention)

```rust
// New private helper in forecast.rs — NOT in acf.rs
// Uses 1/n normalization (not 1/(n-h)) matching acf.rs convention [VERIFIED acf.rs:80-95]
fn scalar_acov(series: &[f64], mean: f64, max_lag: usize) -> Vec<f64> {
    let n = series.len();
    let inv_n = 1.0 / n as f64;
    let mut gamma = vec![0.0f64; max_lag + 1];
    for h in 0..=max_lag {
        let mut s = 0.0;
        for t in 0..(n - h) {
            s += (series[t] - mean) * (series[t + h] - mean);
        }
        gamma[h] = s * inv_n;
    }
    gamma
}
```

---

#### Levinson-Durbin for Yule-Walker (new — informed by `acf.rs:durbin_levinson_pacf` lines 173–201)

The existing `durbin_levinson_pacf` operates on normalized ACF values and returns PACF values. The new `levinson_durbin_yw` operates on raw autocovariances and returns AR coefficients + residual variance. These are distinct functions.

```rust
// Structural pattern to follow from acf.rs:173-201 (phi allocation, early-exit guard)
// New in forecast.rs — takes gamma[0..=p], returns (phi_hat, sigma2)
fn levinson_durbin_yw(gamma: &[f64]) -> Result<(Vec<f64>, f64), FdarError> {
    let p = gamma.len() - 1; // order p: gamma has length p+1
    if p == 0 {
        return Ok((vec![], gamma[0])); // AR(0) = white noise
    }
    if gamma[0].abs() < NUMERICAL_EPS {
        return Err(FdarError::ComputationFailed {
            operation: "levinson_durbin_yw",
            detail: "gamma(0) near zero — degenerate score series".to_string(),
        });
    }
    // phi[k][j] 1-based; heap-allocated following acf.rs:179 pattern
    let mut phi = vec![vec![0.0f64; p + 1]; p + 1];
    let mut nu = vec![0.0f64; p + 1];

    phi[1][1] = gamma[1] / gamma[0];
    nu[1] = gamma[0] * (1.0 - phi[1][1] * phi[1][1]);

    for k in 2..=p {
        // numerator: gamma[k] - sum_{j=1}^{k-1} phi[k-1][j] * gamma[k-j]
        let num = gamma[k] - (1..k).map(|j| phi[k - 1][j] * gamma[k - j]).sum::<f64>();
        // Guard: mirror acf.rs:190-193
        if nu[k - 1].abs() < 1e-12 {
            // Collapse — use order k-1 model
            let phi_hat: Vec<f64> = (1..k).map(|j| phi[k - 1][j]).collect();
            let sigma2 = nu[k - 1].max(0.0);
            return Ok((phi_hat, sigma2));
        }
        phi[k][k] = num / nu[k - 1];
        for j in 1..k {
            phi[k][j] = phi[k - 1][j] - phi[k][k] * phi[k - 1][k - j];
        }
        nu[k] = nu[k - 1] * (1.0 - phi[k][k] * phi[k][k]);
    }
    let phi_hat: Vec<f64> = (1..=p).map(|j| phi[p][j]).collect();
    let sigma2 = nu[p].max(0.0);
    Ok((phi_hat, sigma2))
}
```

Key guard: `nu[k-1].abs() < 1e-12` — mirrors `acf.rs:190` guard `den.abs() < 1e-12`.

---

#### Private `ArModel` struct (new — follows `#[derive(Debug, Clone, PartialEq)]` convention from `regression.rs:22-24`)

```rust
// Private struct — follows derive convention from regression.rs:22-24
// No #[non_exhaustive] — private type
#[derive(Debug, Clone, PartialEq)]
struct ArModel {
    phi: Vec<f64>,    // AR coefficients phi_1..phi_p, 0-indexed (phi[0] = phi_1)
    sigma2: f64,      // residual variance
    mean: f64,        // series mean (center before YW, add back in forecast)
    order: usize,     // selected p (0 = white noise)
    history: Vec<f64>, // last min(p,n) obs (raw, NOT mean-centered), oldest first
}
```

AIC order selection (inside `ArModel::fit`):
```rust
// AIC(p) = n * ln(sigma_p^2) + 2*p  — constant term omitted (cancels), matching R's ar()
// p_max = min(n-1, floor(10*log10(n))).min(n/4)  — dual cap for short series
let p_max = ((10.0 * (n as f64).log10()).floor() as usize)
    .min(n - 1)
    .min(n / 4)
    .max(1);
```

---

#### `#[must_use]` annotation convention (copy from `acf.rs` lines 253, 403, 472, 551, 675)

```rust
// Every expensive public entry point gets #[must_use]:
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ftsm(...) -> Result<FtsmResult, FdarError> { ... }

// Cheaper derived functions (forecast, update) also get #[must_use]:
#[must_use = "returns forecast result; result should be examined"]
pub fn ftsm_forecast(...) -> Result<FtsmForecastResult, FdarError> { ... }
```

---

#### Parameter validation sequence (copy from `regression.rs:fdata_to_pc_1d` lines 292–321 and `pls.rs:fregre_pls` lines 52–90)

Entry-point validation order:
1. `validate_fts_input(data, argvals)?` — shape + argvals match
2. `ncomp >= 1` check → `InvalidParameter`
3. `n > ncomp` check → `InvalidParameter` (need more obs than components)
4. `n >= 2` check → `InvalidParameter` (need at least 2 obs for AR)
5. `h >= 1` check (for forecast functions) → `InvalidParameter`

```rust
// Pattern from fregre_pls:74-78 for ncomp check:
if ncomp == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "ncomp",
        message: "ncomp must be >= 1".to_string(),
    });
}
// Additional FTS-specific check:
if n <= ncomp {
    return Err(FdarError::InvalidParameter {
        parameter: "ncomp",
        message: format!("ncomp ({ncomp}) must be < n ({n})"),
    });
}
```

---

#### `fdata_to_pc_1d` call pattern (from `regression.rs:287-321`, verified)

```rust
// Direct delegate — no wrapping needed
use crate::regression::fdata_to_pc_1d;

let fpca = fdata_to_pc_1d(data, ncomp, argvals)?;
// fpca.mean     → Vec<f64>, length m
// fpca.rotation → FdMatrix, m × ncomp
// fpca.scores   → FdMatrix, n × ncomp
// fpca.weights  → Vec<f64>, length m (Simpson weights)

// Reconstruct fitted curves:
let fitted = fpca.reconstruct(&fpca.scores, ncomp)?;

// Project new obs onto frozen loadings (used in ftsm_update):
let new_scores = fpca.project(new_curve)?;
```

`FpcaResult::project` signature (verified `regression.rs:81-103`): `pub fn project(&self, data: &FdMatrix) -> Result<FdMatrix, FdarError>` — centers by mean, multiplies by rotation with weights, returns (n_new × ncomp).

`FpcaResult::reconstruct` signature (verified `regression.rs:142-170`): `pub fn reconstruct(&self, scores: &FdMatrix, ncomp: usize) -> Result<FdMatrix, FdarError>` — computes `mean[j] + Σ_k scores[i,k] * rotation[j,k]`.

---

#### `fdata_to_pls_1d` call pattern for `fplsr` (from `regression.rs:614`, referenced by `pls.rs:5,91`)

```rust
use crate::regression::fdata_to_pls_1d;

// fplsr lag-1 design: X_cur = rows 0..n-2, X_next = rows 1..n-1
// For each evaluation point j=0..m-1, extract scalar response y_j = X_next column j
// then call fdata_to_pls_1d(X_cur, &y_j, ncomp, argvals)
// PlsResult.project(last_curve) → PLS scores → apply regression coefficients → scalar pred at j

// ncomp clamping (mirrors pls.rs:90):
let ncomp = ncomp.min(n - 1).min(m); // n-1 because lag-1 design has n-1 rows
```

---

#### Error handling patterns (copy from `acf.rs` lines 117–128 and `regression.rs` lines 293–319)

```rust
// InvalidDimension — shape/argvals mismatch (acf.rs:27-40 pattern):
return Err(FdarError::InvalidDimension {
    parameter: "argvals",
    expected: format!("{m} elements (matching data columns)"),
    actual: format!("{} elements", argvals.len()),
});

// InvalidParameter — out-of-range (fregre_pls:74-78 pattern):
return Err(FdarError::InvalidParameter {
    parameter: "ncomp",
    message: "ncomp must be >= 1".to_string(),
});

// ComputationFailed — numerical failure in Levinson-Durbin (acf.rs:121-127 pattern):
return Err(FdarError::ComputationFailed {
    operation: "levinson_durbin_yw",
    detail: "gamma(0) near zero — degenerate score series".to_string(),
});
```

---

#### Test module structure (copy from `acf.rs` lines 734–755)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::covariance::{generate_gaussian_process, CovKernel};
    use crate::test_helpers::uniform_grid;

    // ── Test data helpers ──────────────────────────────────────────────────

    /// Generate n curves where each curve = mean_curve + score * loading,
    /// and score series follows AR(1): beta[t] = phi * beta[t-1] + eps.
    fn make_ar1_score_curves(n: usize, m: usize, phi: f64, seed: u64) -> (FdMatrix, Vec<f64>) {
        // ... use uniform_grid(m), generate AR(1) scores, construct curves
    }
}
```

Pattern from `acf.rs:753-775` for AR(1) data generation — adapt to produce curves
from known score series rather than functional AR(1) in curve space.

---

### `fdars-core/src/fts/mod.rs` (additive edit)

**Analog:** `fdars-core/src/fts/mod.rs` current state (lines 1–75)

Add after line 20 (after `mod acf;`):

```rust
// Pattern: mirror existing mod acf; / pub use acf::{...} block (mod.rs:20-24)
mod forecast;
pub use forecast::{ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update, fplsr};
```

Add new result structs after line 75, mirroring `FacfResult` (lines 31–43), `StationarityResult` (lines 48–58), `LongRunCovResult` (lines 63–75):

```rust
// Pattern from mod.rs:31-43 — derive, serde-gate, #[non_exhaustive], field docs
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FtsmResult {
    /// Sample mean curve μ(u), length m.
    pub mean: Vec<f64>,
    /// FPC loading matrix φ_k, shape m × ncomp (column-major).
    pub rotation: FdMatrix,
    /// FPC score time series β_{t,k}, shape n × ncomp.
    pub scores: FdMatrix,
    /// Reconstructed fitted curves, shape n × m.
    pub fitted: FdMatrix,
    /// Simpson integration weights, length m.
    pub weights: Vec<f64>,
    /// Number of retained FPC components.
    pub ncomp: usize,
    /// Fitted AR model per FPC component (order, coefficients, residual variance).
    pub ar_models: Vec<ArModelResult>,
}

/// Diagnostics for a fitted scalar AR(p) model on one FPC score series.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ArModelResult {
    /// Selected AR order p (0 = white noise).
    pub order: usize,
    /// AR coefficients φ_1..φ_p, 0-indexed.
    pub phi: Vec<f64>,
    /// Residual variance estimate σ².
    pub sigma2: f64,
}

/// Result of `ftsm_forecast` or `ftsm_forecast_multistep`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FtsmForecastResult {
    /// Forecast curves, shape h × m (one row per forecast horizon).
    pub forecast: FdMatrix,
    /// Forecast horizon h.
    pub h: usize,
}

/// Result of `fplsr` — functional PLS lag-1 forecasting.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FplsrResult {
    /// Predicted next curve (1-step ahead), shape 1 × m.
    pub forecast: FdMatrix,
    /// In-sample fitted curves (lag-1 design), shape (n-1) × m.
    pub fitted: FdMatrix,
    /// Number of PLS components used.
    pub ncomp: usize,
}
```

Note: `ArModelResult` (public diagnostic struct in `mod.rs`) is distinct from `ArModel` (private fitting struct in `forecast.rs`). The private `ArModel` in `forecast.rs` holds `history` for forecasting; `ArModelResult` exposes only diagnostics.

Also add `pub use forecast::{FtsmResult, FtsmForecastResult, FplsrResult, ArModelResult};` to the `pub use forecast::{}` block.

---

### `fdars-core/src/lib.rs` (additive edit)

**Analog:** `fdars-core/src/lib.rs` lines 251–255

Extend the existing `pub use fts::{...}` block:

```rust
// Before (lib.rs:251-255):
pub use fts::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
    FacfResult, LongRunCovResult, StationarityResult,
};

// After — add FTS-01 exports to the same block:
pub use fts::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
    ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update, fplsr,
    FacfResult, LongRunCovResult, StationarityResult,
    ArModelResult, FplsrResult, FtsmForecastResult, FtsmResult,
};
```

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/fts/acf.rs` lines 25–42, 117–128; `fdars-core/src/regression.rs` lines 293–319
**Apply to:** All five public entry points in `forecast.rs`

Three error variants used:
- `FdarError::InvalidDimension` — empty data, argvals length mismatch, shape inconsistency
- `FdarError::InvalidParameter` — `ncomp == 0`, `n <= ncomp`, `n < 2`, `h == 0`
- `FdarError::ComputationFailed` — Levinson-Durbin collapse (`gamma[0]` near zero), non-finite sigma2

### Result Struct Declaration
**Source:** `fdars-core/src/fts/mod.rs` lines 31–75 (`FacfResult`, `StationarityResult`, `LongRunCovResult`)
**Apply to:** `FtsmResult`, `FtsmForecastResult`, `FplsrResult`, `ArModelResult` in `mod.rs`

Exact derive chain: `#[derive(Debug, Clone, PartialEq)]` + `#[cfg_attr(feature = "serde", ...)]` + `#[non_exhaustive]` — all four must appear in this order.

### `#[must_use]` Annotation
**Source:** `fdars-core/src/fts/acf.rs` lines 253, 403, 472, 551, 675
**Apply to:** `ftsm` (expensive fit), `ftsm_forecast`, `ftsm_forecast_multistep`, `ftsm_update`, `fplsr`

String format:
- Fit: `"expensive computation whose result should not be discarded"`
- Forecast/update: `"returns forecast result; result should be examined"`

### `validate_fts_input` Helper
**Source:** `fdars-core/src/fts/acf.rs` lines 25–42
**Apply to:** Every entry point in `forecast.rs` as the first call

Re-implement verbatim as a private function in `forecast.rs` — do not modify `acf.rs`.

### Numerical Guard for Near-Zero Denominator
**Source:** `fdars-core/src/fts/acf.rs` lines 189–193 (`den.abs() < 1e-12` early exit)
**Apply to:** `levinson_durbin_yw` recursion — break when `nu[k-1].abs() < 1e-12`

### Column-Major Score Access
**Source:** `fdars-core/src/matrix.rs` (FdMatrix column-major layout, verified throughout acf.rs)
**Apply to:** Extracting per-component score series in `ftsm` and `ArModel::fit`

Use `fit.scores.column(k)` — returns `&[f64]` of length n (contiguous in column-major layout). Do NOT transpose.

### Simpson Weights Reuse
**Source:** `fdars-core/src/helpers.rs` line 57; used in `acf.rs:301`, `regression.rs:325`
**Apply to:** `ftsm` (pass through from `fdata_to_pc_1d`'s `fpca.weights` — do not recompute)

`fdata_to_pc_1d` already computes Simpson weights internally and returns them in `FpcaResult.weights`. Store `fpca.weights` directly in `FtsmResult.weights` — avoid a second `simpsons_weights` call.

### Test Data Helpers
**Source:** `fdars-core/src/fts/acf.rs` lines 744–775 (`make_whitenoise_curves`, `make_ar1_curves`)
**Apply to:** `#[cfg(test)] mod tests` in `forecast.rs`

Use `crate::covariance::{generate_gaussian_process, CovKernel}` and `crate::test_helpers::uniform_grid` — same imports as `acf.rs`.

---

## No Analog Found

All new symbols have direct analogs. No files lack a close match.

| Symbol | Closest Analog | Note |
|---|---|---|
| `levinson_durbin_yw` | `acf.rs:durbin_levinson_pacf` (private) | Different inputs (raw autocovariances vs normalized ACF); structural algorithm pattern is shared |
| `scalar_acov` | `acf.rs:autocovariance_matrix` (pub(crate)) | Scalar specialization (m=1 effective); avoid the full m×m operator |
| `ArModel` (private) | No existing AR struct in codebase | Derive pattern from `regression.rs:FpcaResult` struct |
| `ftsm_forecast_multistep` | `acf.rs:functional_pacf` (thin wrapper) | Implement as the primary function; `ftsm_forecast` delegates to it for h=1 consistency |

---

## Metadata

**Analog search scope:** `fdars-core/src/fts/`, `fdars-core/src/regression.rs`, `fdars-core/src/scalar_on_function/pls.rs`, `fdars-core/src/lib.rs`
**Files read:** 5 source files (acf.rs, mod.rs, regression.rs, pls.rs, lib.rs)
**Pattern extraction date:** 2026-08-22
