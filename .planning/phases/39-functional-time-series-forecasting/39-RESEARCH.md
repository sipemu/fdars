# Phase 39: Functional Time-Series Forecasting — Research

**Researched:** 2026-08-22
**Domain:** Functional time-series forecasting — FPCA decomposition, Yule-Walker AR, PLS lag-1, dynamic update
**Confidence:** HIGH (codebase APIs verified by direct Read; algorithms MEDIUM from official R docs cross-checked against textbooks)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Fit an AR(p) model to each FPC-score sequence via **Yule-Walker** (no new dependency, deterministic; matches `ftsa` default behaviour).
- Select AR order via **AIC** over a candidate order range, informed by the FTS-02 PACF machinery in `fts/acf.rs`.
- Model each FPC-score series with an **independent univariate AR** (ftsa convention — FPC scores are approximately uncorrelated); no joint VAR.
- Reconstruct forecast curves as `mean + Σ_k (forecast_score_k · loading_k)`, truncated to the retained `ncomp` components.
- `ncomp` is a user-provided parameter, validated; no auto variance-threshold pick.
- `Ftsm` result struct carries: mean curve, FPC loadings (rotation), the retained score-time-series, reconstructed fitted curves, integration weights, and `ncomp`.
- Five snake_case entry points, crate-root re-exported: `ftsm` (fit), `ftsm_forecast` (h-step FPC-score forecast), `fplsr` (PLS forecasting variant), `ftsm_update` (dynamic update), `ftsm_forecast_multistep` (iterative h>1).
- `fplsr` regresses the next curve on the current curve via a lag-1 PLS design, reusing existing PLS machinery (`scalar_on_function/pls.rs` patterns) rather than a new PLS subsystem.
- Dynamic update projects new observation(s) onto existing FPC loadings, appends new scores, re-forecasts per-score AR models **without refitting FPCA**; agrees with full refit within a documented tolerance.
- Multi-step (h>1) uses iterative plug-in: forecast scores are fed back into the AR recursion horizon by horizon.
- Multi-step h=1 curve must equal single-step `ftsm_forecast` output (test-enforced consistency).
- Invalid inputs return `FdarError` (not panic): empty/too-short series, fewer obs than components, `ncomp` out of range, `h < 1`, non-monotone/mismatched `argvals`, degenerate columns.
- Deterministic where possible (Yule-Walker AR fit is deterministic). `seed` added only on any stochastic path using `StdRng::seed_from_u64(seed + k)`.
- Forecast quality validated by reusing `scoring.rs` metrics (functional MSE/MAE) and asserting lower error than a naive last-curve baseline.
- Prediction intervals / forecast bands are **out of scope** (numeric point forecasts only).
- Zero changes to existing public signatures; no new crate dependency; additive/non-breaking.

### Claude's Discretion

- Exact `Ftsm` field names, internal AR/Yule-Walker helper structure, candidate AR order range, and specific documented tolerance constants.

### Deferred Ideas (OUT OF SCOPE)

- Prediction intervals / bootstrap forecast bands.
- Spectral / frequency-domain functional time series (FTS-03).
- Joint VAR across FPC scores.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FTS-01-01 | Fit FPCA-based `ftsm` over a time-ordered curve series — decompose via `fdata_to_pc_1d`, retain mean + FPC loadings + score-time-series, reconstruct fitted curves | §Algorithm: ftsm Decomposition; §Reusable APIs |
| FTS-01-02 | Forecast future curves by fitting scalar AR models to each FPC-score sequence and reconstructing the h-step-ahead forecast curve(s) from forecast scores | §Algorithm: Yule-Walker AR; §Algorithm: AR Forecasting Recursion |
| FTS-01-03 | Functional PLS forecasting variant (`fplsr`) — PLS-score-based forecasting as an alternative to FPC-score regression | §Algorithm: fplsr — Functional PLS Forecasting |
| FTS-01-04 | Dynamically update an existing forecast as new curve observation(s) arrive, without refitting from scratch | §Algorithm: Dynamic Update |
| FTS-01-05 | Iterative multi-step (h > 1) forecasts with per-horizon forecast curves | §Algorithm: Multi-Step Forecasting |
</phase_requirements>

---

## Summary

Phase 39 implements functional time-series forecasting in a new file `fdars-core/src/fts/forecast.rs`, wired into the existing `fts/` module. The core approach (matching R's `ftsa` package) decomposes a time-ordered curve series via FPCA (reusing `fdata_to_pc_1d`), then fits an independent scalar AR(p) model to each FPC-score series via Yule-Walker estimation, and reconstructs forecast curves from the forecast scores. All five entry points are new; none of the existing public APIs change.

The most algorithmically intricate sub-problems are: (1) implementing Yule-Walker AR(p) fitting with Levinson-Durbin recursion and AIC-based order selection using only the existing autocovariance infrastructure from `fts/acf.rs`; (2) adapting the `fplsr` functional-PLS forecasting variant to work with a functional response (next curve) rather than the scalar response that `fregre_pls` currently handles; and (3) correctly wiring the dynamic-update path so it projects onto frozen loadings without triggering FPCA refit.

No new crate dependency is required. The entire implementation reuses `fdata_to_pc_1d` (FPCA), `FpcaResult.project` + `FpcaResult.reconstruct` (scoring new obs / curve reconstruction), `autocovariance_matrix` from `fts/acf.rs` (scalar autocovariances for Yule-Walker), and `scoring.rs` metrics.

**Primary recommendation:** Implement in three layers — (1) a private `ArModel` struct with Yule-Walker fit + iterative h-step forecast; (2) the `Ftsm` fit struct and the `ftsm` entry point that wraps `fdata_to_pc_1d` + fits one `ArModel` per score column; (3) the four derived entry points (`ftsm_forecast`, `ftsm_update`, `ftsm_forecast_multistep`, `fplsr`) that build on the fit struct.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| FPCA decomposition | `fts/forecast.rs` → delegates to `regression::fdata_to_pc_1d` | `regression.rs` | FPCA is already implemented; `ftsm` is a thin wrapper that adds score-series storage |
| Yule-Walker AR fit | `fts/forecast.rs` (private `ArModel`) | `fts/acf.rs` (reuse `autocovariance_matrix`) | No existing AR module; must be added; autocovariance already computed in acf.rs |
| h-step AR forecast | `fts/forecast.rs` (private `ArModel::forecast`) | — | Pure iterative recursion; no external dependency |
| Curve reconstruction | `regression::FpcaResult::reconstruct` | `fts/forecast.rs` (calls it) | Already implemented; `forecast.rs` just calls it with forecast scores |
| PLS lag-1 fit | `fts/forecast.rs` (`fplsr`) → delegates to `regression::fdata_to_pls_1d` | `scalar_on_function/pls.rs` | Adapts scalar PLS to per-component curve prediction |
| Dynamic update | `fts/forecast.rs` (`ftsm_update`) → delegates to `FpcaResult::project` | — | Projection onto frozen loadings is already in `FpcaResult::project` |
| Forecast-error metrics | `scoring.rs` (unchanged) | `fts/forecast.rs` (calls it in tests) | Already implemented |
| Module wiring | `fts/mod.rs` (add `mod forecast; pub use forecast::{...}`) | `src/lib.rs` (crate-root re-export) | Follows existing pattern |

---

## Reusable APIs — Verified Signatures

> All signatures below are verified by direct Read of the source files this session.

### `fdata_to_pc_1d` — Dense FPCA
`[VERIFIED: fdars-core/src/regression.rs:287-321]`

```rust
pub fn fdata_to_pc_1d(
    data: &FdMatrix,   // n × m, column-major; rows = time-ordered curves
    ncomp: usize,
    argvals: &[f64],   // length m
) -> Result<FpcaResult, FdarError>
```

`FpcaResult` fields `[VERIFIED: fdars-core/src/regression.rs:25-38]`:
```rust
pub struct FpcaResult {
    pub singular_values: Vec<f64>,
    pub rotation: FdMatrix,    // m × ncomp — loadings phi_k
    pub scores: FdMatrix,      // n × ncomp — FPC score time series beta_{t,k}
    pub mean: Vec<f64>,        // length m — mu(u)
    pub centered: FdMatrix,    // n × m
    pub weights: Vec<f64>,     // length m — Simpson integration weights
}
```

`FpcaResult` methods `[VERIFIED: fdars-core/src/regression.rs:81-170]`:
- `project(&self, data: &FdMatrix) -> Result<FdMatrix, FdarError>` — centers by `mean`, multiplies by `rotation` with `weights`; returns scores (n_new × ncomp)
- `reconstruct(&self, scores: &FdMatrix, ncomp: usize) -> Result<FdMatrix, FdarError>` — computes `mean[j] + Σ_k scores[i,k] * rotation[j,k]`; returns (n × m) matrix

Both methods are already implemented exactly as needed for `ftsm` fit, update, and forecast reconstruction.

### `autocovariance_matrix` — Shared with fts/acf.rs
`[VERIFIED: fdars-core/src/fts/acf.rs:73-95]`

```rust
pub(crate) fn autocovariance_matrix(
    data: &FdMatrix,
    xbar: &[f64],
    h: usize,
    n: usize,
    m: usize,
) -> Vec<f64>  // flat m×m in column-major; c_h[j1 + j2*m]
```

This is `pub(crate)` — accessible from `fts/forecast.rs` as `use crate::fts::acf::autocovariance_matrix`. However, `mean_curve` (computes xbar) and `validate_fts_input` are private to `fts/acf.rs` `[VERIFIED: fdars-core/src/fts/acf.rs:25-58]`. The planner must decide: either (a) re-implement the two small private helpers in `forecast.rs` (trivial, 10 lines each), or (b) promote them to `pub(crate)` in `acf.rs` (clean but requires a small change to `acf.rs`). Option (a) is preferred since it avoids any change to shipped code.

For Yule-Walker scalar AR, only the **diagonal** of `autocovariance_matrix` at h=0,1,2,...,p is needed (since scores are scalar series, not functional). Specifically: `gamma(h)` for a scalar score series = `(1/n) * Σ_t score[t] * score[t+h]`. This can be computed directly without `autocovariance_matrix` (which is an m×m operator for functional data). The planner should implement a private `scalar_autocovariance(series: &[f64], max_lag: usize) -> Vec<f64>` that computes `gamma(0), gamma(1), ..., gamma(max_lag)` via a simple loop.

### `durbin_levinson_pacf` — Private in acf.rs
`[VERIFIED: fdars-core/src/fts/acf.rs:173-201]`

```rust
fn durbin_levinson_pacf(rho: &[f64]) -> Vec<f64>
```

This is **private** (no `pub` or `pub(crate)`). It works on ACF values (normalized by variance), not raw autocovariances. For Yule-Walker, we need the **coefficients** phi[1..p] from the Levinson-Durbin recursion, not just the PACF values. The Levinson-Durbin algorithm for solving the Yule-Walker system produces phi[1..p] as a byproduct of the recursion. The planner must implement a separate private `levinson_durbin_yw(gamma: &[f64]) -> (Vec<f64>, f64)` in `forecast.rs` that takes raw autocovariance values `gamma[0..=p]` and returns `(phi, sigma2)`. This is distinct from the ACF-input `durbin_levinson_pacf` in `acf.rs`.

### `scoring.rs` — Forecast Error Metrics
`[VERIFIED: fdars-core/src/scoring.rs:59-99]`

```rust
pub fn functional_mae(y_true: &FdMatrix, y_pred: &FdMatrix, argvals: &[f64]) -> Result<f64, FdarError>
pub fn functional_mse(y_true: &FdMatrix, y_pred: &FdMatrix, argvals: &[f64]) -> Result<f64, FdarError>
pub fn functional_mape(y_true: &FdMatrix, y_pred: &FdMatrix, argvals: &[f64]) -> Result<f64, FdarError>
```

Shape contract: `y_true.shape() == y_pred.shape()`, `argvals.len() == y_true.ncols()`, `n >= 1`, `m >= 2`. Used in tests to assert forecast MSE beats naive last-curve baseline.

### `fts/mod.rs` — Module Wiring Pattern
`[VERIFIED: fdars-core/src/fts/mod.rs:1-75]`

```rust
mod acf;
pub use acf::{functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test};
// Result structs declared inline in mod.rs
pub struct FacfResult { ... }
pub struct StationarityResult { ... }
pub struct LongRunCovResult { ... }
```

New `forecast.rs` follows the same pattern: add `mod forecast;` + `pub use forecast::{ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update, fplsr, FtsmResult, FtsmForecastResult, FplsrResult};` in `mod.rs`. New result structs (`FtsmResult`, `FtsmForecastResult`, `FplsrResult`) should be declared in `mod.rs` (consistent with `FacfResult`) or in `forecast.rs` itself — either location is acceptable, but `mod.rs` is the existing convention.

### `fts/lib.rs` Crate-Root Re-Export Pattern
`[VERIFIED: fdars-core/src/lib.rs:251-255]`

```rust
pub use fts::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
    FacfResult, LongRunCovResult, StationarityResult,
};
```

New exports: add `ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update, fplsr, FtsmResult, FtsmForecastResult, FplsrResult` to this block.

### `fregre_pls` / `PlsResult` — PLS Machinery
`[VERIFIED: fdars-core/src/scalar_on_function/pls.rs:45-183]`

`fregre_pls` takes a **scalar** response `y: &[f64]`. The `fplsr` functional PLS forecasting variant has a **functional response** (the next curve), which is incompatible with a direct call to `fregre_pls`. The adaptation strategy is to predict each evaluation point of the next curve independently as a separate scalar response — i.e., call `fdata_to_pls_1d(X_cur, score_j, ncomp, argvals)` for each column j of the response matrix, or equivalently use the `PlsResult.project` method to get PLS scores and then regress. See §Algorithm: fplsr for the recommended approach.

`PlsResult` fields `[VERIFIED: fdars-core/src/regression.rs:406-417]`:
```rust
pub struct PlsResult {
    pub weights: FdMatrix,            // m × ncomp — PLS weight vectors
    pub scores: FdMatrix,             // n × ncomp — score vectors
    pub loadings: FdMatrix,           // m × ncomp — loading vectors
    pub x_means: Vec<f64>,            // length m
    pub integration_weights: Vec<f64>, // length m
}
// .project(data: &FdMatrix) -> Result<FdMatrix, FdarError>
```

---

## Algorithm Specifications

### 1. ftsm Decomposition (FTS-01-01)

**R baseline:** `ftsa::ftsm(y, order=6, mean=TRUE, method="classical", ...)` `[CITED: rdrr.io/cran/ftsa/man/ftsm.html]`

**Model:** `X_t(u) = μ(u) + Σ_{k=1}^{K} β_{t,k} · φ_k(u) + ε_t(u)` `[CITED: robjhyndman.com/talks/JKSS.pdf]`

where:
- `μ(u)` = sample mean curve (computed by `fdata_to_pc_1d` internally; returned in `FpcaResult.mean`)
- `φ_k(u)` = k-th FPC loading = `FpcaResult.rotation` column k (m-vector)
- `β_{t,k}` = FPC score at time t for component k = `FpcaResult.scores[t, k]`
- `K = ncomp` (user-specified; ftsa default is 6, no default in Rust — must be provided)
- `fdata_to_pc_1d` clamps `ncomp = ncomp.min(n).min(m)` silently

**Fitted curve reconstruction:** `[VERIFIED: fdars-core/src/regression.rs:142-170]`

```rust
// Already implemented in FpcaResult::reconstruct
let fitted = fpca.reconstruct(&fpca.scores, ncomp)?;
// fitted[(i, j)] = mean[j] + Σ_k scores[(i,k)] * rotation[(j,k)]
```

**`FtsmResult` proposed struct:**

```rust
pub struct FtsmResult {
    pub mean: Vec<f64>,          // μ(u), length m
    pub rotation: FdMatrix,      // FPC loadings φ_k, shape m × ncomp
    pub scores: FdMatrix,        // β_{t,k}, shape n × ncomp (score time series)
    pub fitted: FdMatrix,        // reconstructed fitted curves, shape n × m
    pub weights: Vec<f64>,       // Simpson integration weights, length m
    pub ncomp: usize,
    // internal (not pub): one ArModel per component, stored for forecast/update
    // pub(crate) ar_models: Vec<ArModel>,  -- or store in a separate FtsmForecast struct
}
```

The `ar_models` cannot be part of `FtsmResult` if `FtsmResult` must derive `PartialEq` (since floats in `ArModel` are PartialEq but custom recursion doesn't add complexity). Keep them in the struct — `ArModel` can derive `Debug, Clone, PartialEq`.

**Divergence from ftsa:** ftsa `ftsm` applies optional smoothing (kernel, P-spline) before FPCA; fdars uses dense `fdata_to_pc_1d` directly (no smoothing step). Document in rustdoc: "Unlike `ftsa::ftsm`, this implementation operates on the raw input grid without prior smoothing. Users requiring pre-smoothed curves should smooth before calling `ftsm`."

**Validation:** `n > ncomp`, `n >= 2` (need at least 2 obs for AR), `m >= 2`, `argvals.len() == m`.

---

### 2. Yule-Walker AR(p) Estimation (FTS-01-02)

**R baseline:** `stats::ar(x, method="yule-walker", aic=TRUE)` `[CITED: stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html]`

**Scalar autocovariance:** For score series `beta_k[0..n-1]` (a scalar `Vec<f64>`):
```
gamma(h) = (1/n) * Σ_{t=0}^{n-h-1} (beta_k[t] - beta_bar) * (beta_k[t+h] - beta_bar)
```
This is `pub(crate) autocovariance_matrix` from `fts/acf.rs` evaluated at the diagonal, but since the score series is scalar (m=1), it simplifies to a direct loop. Implement `fn scalar_acov(series: &[f64], max_lag: usize) -> Vec<f64>` in `forecast.rs`. `[ASSUMED]` — the 1/n normalization convention (bias-corrected denominator would be 1/(n-h)) matches `fts/acf.rs`'s convention `[VERIFIED: fdars-core/src/fts/acf.rs:80-95]`.

**Levinson-Durbin recursion for AR(p) coefficients:** `[CITED: nmimoto.github.io/477/w5c.html]`

```
# Initialize
phi[1][1] = gamma(1) / gamma(0)
nu[1] = gamma(0) * (1 - phi[1][1]^2)

# Recursion for k = 2..p:
phi[k][k] = (gamma(k) - Σ_{j=1}^{k-1} phi[k-1][j] * gamma(k-j)) / nu[k-1]
phi[k][j] = phi[k-1][j] - phi[k][k] * phi[k-1][k-j]   for j=1..k-1
nu[k] = nu[k-1] * (1 - phi[k][k]^2)

# Result: phi_hat = phi[p][1..p],  sigma2_hat = nu[p] / gamma(0)  ← residual variance
```

The innovation variance `nu[p]` is the Yule-Walker residual variance. Note `sigma2 = gamma(0) - phi^T * [gamma(1)..gamma(p)]` is equivalent.

**AIC for order selection:** `[CITED: stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html]`

```
AIC(p) = n * ln(sigma_p^2) + 2*p      (p = 0, 1, ..., p_max)
```

For p=0: `sigma_0^2 = gamma(0)` (no AR coefficients — white noise model). Select `p_hat = argmin_p AIC(p)`. `[ASSUMED]` — the constant term `n*(1 + ln(2*pi))` cancels in comparisons and is omitted, matching R's `ar()` convention.

**Candidate order range:** `[CITED: stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html]`

```
p_max = min(n - 1, floor(10 * log10(n)))     # R's ar() default
```

For short score series (n=20, p_max=13; n=50, p_max=16; n=100, p_max=20). A hard cap of `p_max.min(n/4)` may be more conservative — at Claude's discretion per CONTEXT.md.

**`ArModel` private struct:**

```rust
struct ArModel {
    phi: Vec<f64>,       // AR coefficients phi[1..p] (0-indexed: phi[0]=phi_1)
    sigma2: Vec<f64>,    // residual variance
    mean: f64,           // series mean (series is mean-centered before fitting)
    order: usize,        // selected p
    history: Vec<f64>,   // last p observations (for forecasting)
}
```

**Stationarity note:** Levinson-Durbin can numerically fail if `nu[k]` approaches zero (near-unit-root series). Mirror `acf.rs`'s approach: if `nu[k-1].abs() < 1e-12`, stop recursion early and use the p-1 model.

---

### 3. AR(p) Forecasting Recursion (FTS-01-02, FTS-01-05)

**One-step forecast** (h=1): `[CITED: nmimoto.github.io/477/w9a.html]`

```
x_hat(n+1) = mean + Σ_{j=1}^{p} phi[j] * (x[n-j+1-1] - mean)
           = mean + Σ_{j=0}^{p-1} phi[j] * (history[p-1-j] - mean)
```

where `history = [x[n-p], x[n-p+1], ..., x[n-1]]` (last p observations, oldest first).

**h-step iterative plug-in** (h>1): `[CITED: princeton.edu/~mwatson/papers/hstep_3.pdf]`

```
# Build buffer of last p values; for i in 1..=h:
#   if i <= p: val = history[p-i] (actual observation, mean-shifted)
#   if i > p:  val = previously computed forecast[i-p-1]
# forecast[i-1] = mean + Σ_{j=0}^{p-1} phi[j] * buf[p-1-j]
# Update buf: slide window
```

Concretely — maintain a sliding window of length p; for each step k=1..h, compute the AR prediction using the window (mixing history with already-computed forecasts), then append the prediction to the window.

**h=1 consistency invariant (test):** `ftsm_forecast_multistep(fit, 1, argvals)` must produce bit-identical curves to `ftsm_forecast(fit, 1, argvals)`.

---

### 4. `ftsm_forecast` — FPC-Score Forecast → Curve Reconstruction (FTS-01-02)

**Algorithm:**

```
for each component k in 0..ncomp:
    scores_k = column k of fit.scores  (length n)
    ar_k = fit.ar_models[k]
    beta_hat_k = ar_k.forecast(h=1)    // scalar h-step ahead score forecast

forecast_scores = [beta_hat_0, ..., beta_hat_{ncomp-1}]  // (1 × ncomp) FdMatrix

// Reconstruction: use FpcaResult::reconstruct logic directly
forecast_curve[j] = fit.mean[j] + Σ_k forecast_scores[k] * fit.rotation[(j,k)]
```

Return: `FtsmForecastResult { forecast: FdMatrix (1 × m), h: usize }` (or h curves for multi-step).

---

### 5. `fplsr` — Functional PLS Forecasting (FTS-01-03)

**R baseline:** `ftsa::fplsr(data, order=6, type="simpls", ...)` `[CITED: rdrr.io/cran/ftsa/man/fplsr.html]`

**Lag-1 design:** Predictor = `X_cur` (rows 0..n-2, shape (n-1)×m), Response = `X_next` (rows 1..n-1, shape (n-1)×m). `[CITED: rdrr.io/cran/ftsa/man/fplsr.html]`

**Key challenge — functional response:** `fdata_to_pls_1d` and `fregre_pls` accept a **scalar** response `y: &[f64]`. The `fplsr` response is a functional curve (m evaluation points). The recommended adaptation — at Claude's discretion — is:

**Option A (recommended):** Fit one PLS model per response evaluation point. For each j in 0..m, extract the scalar response `y_j = X_next.column(j)` and call `fdata_to_pls_1d(X_cur, y_j, ncomp, argvals)`. This produces m separate `PlsResult` objects. Forecast: call `pls_j.project(&X_last_curve) -> scores_j`, then apply regression coefficients to get scalar prediction at point j. Assemble into forecast curve.

**Option B (more principled):** Use FPCA to reduce the response dimension first — regress PLS scores of predictors onto FPC scores of response. But this adds complexity and is not what ftsa does.

Option A is close to what ftsa's `fplsr` does (NIPALS/SIMPLS per-component deflation in functional space). It is O(m) PLS fits, each O(n·ncomp·m) — acceptable for typical m ≤ 200.

**`FplsrResult` proposed struct:**

```rust
pub struct FplsrResult {
    pub forecast: FdMatrix,  // 1 × m predicted next curve
    pub fitted: FdMatrix,    // (n-1) × m fitted curves (in-sample)
    pub ncomp: usize,
}
```

**Divergence from ftsa:** ftsa's `fplsr` uses NIPALS/SIMPLS in a unified functional operator; fdars uses per-point OLS on PLS scores (functionally equivalent for prediction, less elegant). Document in rustdoc.

---

### 6. Dynamic Update (FTS-01-04)

**R baseline:** `ftsa::dynupdate(data, newdata, holdoutdata, method="ols", ...)` `[CITED: rdrr.io/cran/ftsa/man/dynupdate.html]`

**Algorithm (projection update — "OLS" method in ftsa):**

```
Given: existing FtsmResult fit (frozen loadings, frozen mean)
New observation: new_curve (1 × m FdMatrix)

1. Project onto frozen loadings:
   new_scores = fit.fpca_rotation.project(new_curve)
   // i.e.: for k in 0..ncomp: new_score[k] = Σ_j (new_curve[j] - fit.mean[j]) * fit.rotation[(j,k)] * fit.weights[j]
   // This is exactly FpcaResult::project(new_curve) if FpcaResult is stored in FtsmResult

2. Append scores: extended_scores_k = [fit.scores.column(k); new_score[k]] for each k

3. Re-fit AR(p) on extended_scores_k via Yule-Walker (same AIC selection)

4. Produce 1-step-ahead forecast on the extended series

5. Return updated FtsmResult (same loadings, extended scores, new AR fits)
```

**Key invariant (test):** Update result agrees with full refit of `ftsm` on extended data within a documented tolerance (e.g., relative-L2 < 1e-2 on the forecast curve). The tolerance should be larger than machine epsilon because the update uses the same mean (frozen) while a full refit would re-estimate the mean — this is the documented divergence.

**What changes:** `scores` matrix gains a new row; `ar_models` are re-fit; `fitted` grows by one row; `mean` and `rotation` stay fixed.

**`FtsmResult` must store the `FpcaResult` or equivalent projection state** (mean, rotation, weights) to enable `project()`. Recommend storing `mean: Vec<f64>`, `rotation: FdMatrix`, `weights: Vec<f64>` directly on `FtsmResult` (which mirrors `FpcaResult` fields).

---

### 7. Multi-Step Forecasting (FTS-01-05)

**R baseline:** `ftsa::ftsmiterativeforecasts(object, components, iteration=20)` `[CITED: rdrr.io/cran/ftsa/man/ftsmiterativeforecasts.html]`

**Algorithm:**

```
ftsm_forecast_multistep(fit, h, argvals) -> FdMatrix (h × m):
  for step in 1..=h:
      forecast_score_k = ar_k.forecast_h(step, history=fit.scores.column(k))  for k in 0..ncomp
      forecast_curve[step-1, j] = fit.mean[j] + Σ_k forecast_score_k[k] * fit.rotation[(j,k)]
  return h × m matrix
```

The AR forecast at horizon `step` for each component is computed using the iterative plug-in recursion described in §3. The AR model uses the full `fit.scores.column(k)` as history (plus already-forecast scores for steps > p).

**h=1 consistency:** The step=1 forecast from `ftsm_forecast_multistep` must be bit-identical to `ftsm_forecast`. Enforce this with an internal debug_assert or a dedicated test.

---

## Architecture Patterns

### System Architecture Diagram

```
User Input: FdMatrix (n × m), ncomp, argvals
        │
        ▼
  ftsm() ──────────────────────────────────────────────────────────────┐
        │                                                               │
        ├─► fdata_to_pc_1d(data, ncomp, argvals)                       │
        │     → FpcaResult { mean, rotation, scores, weights }         │
        │                                                               │
        ├─► Per k=0..ncomp-1:                                          │
        │     scalar_acov(scores.column(k), p_max)                     │
        │     levinson_durbin_yw(gamma) → ArModel { phi, sigma2, ... } │
        │                                                               │
        └─► FtsmResult { mean, rotation, scores, fitted, weights,      │
                          ncomp, ar_models }                            │
                                                                        │
  ftsm_forecast(fit, h=1) ────────────────────────────────────────────►│
        │                                                               │
        ├─► per k: ar_models[k].forecast(1) → scalar score_hat        │
        ├─► assemble forecast_scores (1 × ncomp)                       │
        └─► fpca.reconstruct(forecast_scores, ncomp) → 1 × m curve    │
                                                                        │
  ftsm_forecast_multistep(fit, h) ────────────────────────────────────►│
        │                                                               │
        └─► repeat ftsm_forecast logic for steps 1..=h → h × m        │
                                                                        │
  ftsm_update(fit, new_curve) ─────────────────────────────────────────┘
        │
        ├─► project new_curve onto fit.rotation → new_score (ncomp)
        ├─► extend fit.scores by one row
        ├─► re-fit ArModel per component (Yule-Walker + AIC)
        └─► return updated FtsmResult (loadings frozen)
                                                                        
  fplsr(data, ncomp, argvals)
        │
        ├─► build X_cur (rows 0..n-2) and X_next (rows 1..n-1)
        ├─► per j=0..m-1: fdata_to_pls_1d(X_cur, y_j, ncomp, argvals)
        │     → PlsResult_j
        ├─► forecast point j: pls_j.project(last_curve) → scores; apply beta
        └─► assemble → FplsrResult { forecast (1×m), fitted ((n-1)×m), ncomp }
```

### Recommended Project Structure

```
fdars-core/src/fts/
├── mod.rs         # add: mod forecast; pub use forecast::{...}; new result structs
├── acf.rs         # UNCHANGED (shipped FTS-02)
└── forecast.rs    # NEW — all FTS-01 implementation
```

### Pattern 1: Private `ArModel` Helper Struct

```rust
// src/fts/forecast.rs — private, not pub
#[derive(Debug, Clone, PartialEq)]
struct ArModel {
    phi: Vec<f64>,    // AR coefficients phi[1..=p], 0-indexed (phi[0] = phi_1)
    sigma2: f64,      // residual variance estimate
    mean: f64,        // series mean
    order: usize,     // selected p (may be 0 = white noise)
    history: Vec<f64>, // last min(p,n) observations (pre-mean-centered), oldest first
}

impl ArModel {
    fn fit(series: &[f64], p_max: usize) -> Result<Self, FdarError> {
        // 1. Compute mean
        // 2. scalar_acov(centered, p_max) -> gamma[0..=p_max]
        // 3. for p=0..=p_max: levinson_durbin_yw(&gamma[0..=p]) -> (phi_p, sigma2_p)
        //                     aic_p = n as f64 * sigma2_p.ln() + 2.0 * p as f64
        // 4. select p_hat = argmin aic_p
        // 5. history = last p_hat obs (mean-centered)
        todo!()
    }

    fn forecast(&self, h: usize) -> Vec<f64> {
        // iterative plug-in, returns h scalar forecasts (mean-shifted back)
        todo!()
    }
}
```

### Pattern 2: Entry Point Signature Style (match existing codebase conventions)

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ftsm(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FtsmResult, FdarError> { ... }

pub fn ftsm_forecast(
    fit: &FtsmResult,
    h: usize,
    argvals: &[f64],
) -> Result<FtsmForecastResult, FdarError> { ... }

pub fn ftsm_update(
    fit: &FtsmResult,
    new_curve: &FdMatrix,   // 1 × m
    argvals: &[f64],
) -> Result<FtsmResult, FdarError> { ... }

pub fn ftsm_forecast_multistep(
    fit: &FtsmResult,
    h: usize,
    argvals: &[f64],
) -> Result<FtsmForecastResult, FdarError> { ... }

pub fn fplsr(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FplsrResult, FdarError> { ... }
```

`FtsmForecastResult` carries `forecast: FdMatrix` (h × m) and `h: usize`.

### Anti-Patterns to Avoid

- **Using `autocovariance_matrix` from acf.rs for scalar AR:** The m×m operator is overkill for a scalar series (m=1 effective). Implement `scalar_acov` directly — avoids allocating m²=1 slice wrappers and is clearer.
- **Not mean-centering before Yule-Walker:** AR(p) Yule-Walker assumes zero-mean process. Always center the score series before computing autocovariances; add mean back in the forecast.
- **Storing raw `FpcaResult` in `FtsmResult`:** Creates a nested struct where users can't derive `PartialEq` cleanly. Instead, flatten the needed fields (`mean`, `rotation`, `weights`, `scores`) directly into `FtsmResult`.
- **Refitting FPCA in `ftsm_update`:** Explicitly forbidden by CONTEXT.md. The update path reuses frozen `rotation` and `mean`; only `ar_models` are re-fit.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FPCA decomposition | Custom SVD | `fdata_to_pc_1d` (`regression.rs:287`) | Already implemented with sign-fixing, weight scaling, and sign normalization |
| Curve reconstruction from scores | Manual mean+loadings multiply | `FpcaResult::reconstruct` (`regression.rs:142`) | Already implemented; reconstructs `mean[j] + Σ_k scores[i,k] * rotation[j,k]` |
| Projecting new obs onto FPC space | Manual dot product | `FpcaResult::project` (`regression.rs:81`) | Already weight-scaled and mean-centered |
| PLS fit from functional predictor | Custom NIPALS | `fdata_to_pls_1d` (`regression.rs:614`) + `PlsResult::project` | Already NIPALS-implemented; project method handles new obs |
| Forecast error metrics | Custom L2 integration | `functional_mse`, `functional_mae` (`scoring.rs`) | Already Simpson-integrated |
| Integration weights | Manual trapezoidal | `simpsons_weights` (`helpers.rs:57`) | Already implemented; used throughout codebase |
| PACF-informed order hint | Re-derive D-L | `durbin_levinson_pacf` in `acf.rs` — but note: private. Use PACF output from `functional_pacf` call at user level if desired. AR order is selected by AIC in the implementation. | |

---

## Common Pitfalls

### Pitfall 1: Yule-Walker Divergence for Short Score Series

**What goes wrong:** For short score series (n < 30), `p_max = floor(10*log10(n))` can be close to n-1. With p near n, the p×p Toeplitz matrix becomes ill-conditioned and Levinson-Durbin's `nu[k]` approaches zero.

**Why it happens:** The Yule-Walker system is rank-deficient when p ≥ n/2 approximately.

**How to avoid:** Add a hard cap: `p_max = p_max.min(n / 4)` where n is the score series length. When `nu[k-1].abs() < 1e-12` in the Levinson-Durbin loop, break and use the order-k-1 model (mirrors `durbin_levinson_pacf`'s existing guard `[VERIFIED: fdars-core/src/fts/acf.rs:190-193]`).

**Warning signs:** AIC selects p = p_max repeatedly; sigma2 < 0 (should be impossible but can occur with fp overflow).

### Pitfall 2: fplsr Per-Point PLS Overfitting

**What goes wrong:** Calling `fdata_to_pls_1d` m times with ncomp close to n-1 produces perfect in-sample fit but terrible forecasts due to overfitting.

**Why it happens:** The lag-1 design has only n-1 rows; with large ncomp the PLS model exhausts degrees of freedom.

**How to avoid:** Validate `ncomp < n - 1`; recommended ncomp is 2–6 (matching ftsa default of 6). Note that `fdata_to_pls_1d` already clamps `ncomp = ncomp.min(n).min(m)` `[VERIFIED: fdars-core/src/regression.rs:619-638]`, but the clamping happens per-call with the lag-1 dataset (n-1 rows), so the max is `min(n-1, m)`.

**Warning signs:** In-sample MSE near zero but naive-baseline test fails.

### Pitfall 3: Dynamic Update Mean Drift

**What goes wrong:** After several updates, the projected scores diverge from what a full refit would produce, because the frozen mean does not track the evolving series mean.

**Why it happens:** `ftsm_update` freezes `mean` at initial fit time; each new obs shifts the true mean slightly.

**How to avoid:** Document this divergence in rustdoc ("The dynamic update freezes the mean curve from the initial `ftsm` fit; for long update sequences, periodic full refit via `ftsm` is recommended"). The test should assert agreement within tolerance for a small number of updates (1–3 new obs), not for long sequences.

**Warning signs:** update-vs-refit test fails with tolerance > 1e-1 after more than 10 updates.

### Pitfall 4: h=1 Inconsistency Between `ftsm_forecast` and `ftsm_forecast_multistep`

**What goes wrong:** The two functions produce slightly different h=1 results due to floating-point ordering differences in the iterative loop.

**Why it happens:** `ftsm_forecast` and the step=1 of `ftsm_forecast_multistep` must use identical arithmetic paths.

**How to avoid:** Implement `ftsm_forecast` as `ftsm_forecast_multistep(fit, 1, argvals)` internally — zero duplication, guaranteed consistency. Test with `assert_eq!(h1_from_single, h1_from_multi)`.

### Pitfall 5: Column-Major Index Confusion When Extracting Score Columns

**What goes wrong:** `FdMatrix.column(k)` returns a slice to the k-th column (score series for component k). This is the correct access for scores (n×ncomp matrix, column k = score series for component k).

**How to avoid:** Use `fit.scores.column(k)` (which returns `&[f64]` of length n) directly for the AR fit input. Do not transpose. `[VERIFIED: fdars-core/src/matrix.rs]` — column-major layout means `column(k)` returns contiguous data for rows 0..n.

---

## Concrete Test Specifications

### T-01: Synthetic AR(1) Score Recovery (FTS-01-01, FTS-01-02)

```
Setup: Generate 2D FdMatrix where curves = mean_curve + score * loading.
       score series follows known AR(1): beta[t] = 0.8 * beta[t-1] + eps
       (eps ~ N(0, 0.01), n = 80 curves, m = 20 grid points)

Test: ftsm(data, ncomp=1, argvals).unwrap()
      ftsm_forecast(fit, h=1, argvals)
      // Expected: |forecast_score - (0.8 * fit.scores[(n-1, 0)])| < 0.05
      // (allows for estimation noise with n=80)
```

### T-02: Fitted Curve Recovery (FTS-01-01)

```
// FPCA with ncomp components should reconstruct training data well
let fitted = fit.fitted;
let mse = functional_mse(&data, &fitted, &argvals).unwrap();
assert!(mse < 0.01 * data_variance,
        "fitted MSE must be < 1% of total variance for ncomp = rank(data)");
```

Relative-L2 tolerance constant: `0.01` (1% of variance) — at Claude's discretion per CONTEXT.md.

### T-03: Forecast Error Beats Naive Baseline (FTS-01-02)

```
// naive baseline: last curve repeated h=1 times
let naive_forecast = data.row_to_buf(n-1);
let naive_mse = functional_mse(&true_next_curve, &naive_as_matrix, &argvals)?;
let model_mse = functional_mse(&true_next_curve, &ftsm_forecast_curve, &argvals)?;
assert!(model_mse < naive_mse,
        "ftsm forecast MSE must beat naive last-curve baseline on AR-structured data");
```

Use a curve series with moderate AR(1) signal (phi=0.7) so the model has a genuine advantage.

### T-04: h=1 Multi-Step Consistency (FTS-01-02, FTS-01-05)

```
let h1_single = ftsm_forecast(&fit, 1, &argvals)?;
let h1_multi = ftsm_forecast_multistep(&fit, 1, &argvals)?;
// h1_multi returns 1×m; h1_single returns 1×m
for j in 0..m {
    assert!((h1_single.forecast[(0,j)] - h1_multi.forecast[(0,j)]).abs() < 1e-12,
            "h=1 multi-step must be bit-identical to single-step");
}
```

### T-05: Dynamic Update Agreement (FTS-01-04)

```
let fit = ftsm(&data[0..n-1], ncomp, &argvals)?;
let updated = ftsm_update(&fit, &data.row_slice(n-1), &argvals)?;
let full_refit = ftsm(&data[0..n], ncomp, &argvals)?;

// Update forecast vs full-refit forecast: relative L2 < 1% of signal
let updated_fc = ftsm_forecast(&updated, 1, &argvals)?.forecast;
let refit_fc = ftsm_forecast(&full_refit, 1, &argvals)?.forecast;
let rel_err = functional_mse(&updated_fc, &refit_fc, &argvals)? /
              functional_mse(&FdMatrix::zeros(1,m), &refit_fc, &argvals)?;
assert!(rel_err < 0.01, "update vs full-refit relative L2 < 1%");
```

Tolerance `0.01` (1%) — at Claude's discretion. The divergence arises from the frozen mean.

### T-06: Error Path Tests

```
// Empty data
assert!(matches!(ftsm(&FdMatrix::zeros(0,20), 2, &argvals), Err(FdarError::InvalidDimension{..})));
// n < ncomp  
assert!(matches!(ftsm(&data_3_curves, 5, &argvals), Err(FdarError::InvalidParameter{..})));
// h < 1 (h=0)
assert!(matches!(ftsm_forecast(&fit, 0, &argvals), Err(FdarError::InvalidParameter{..})));
// argvals mismatch
assert!(matches!(ftsm(&data, 2, &wrong_argvals), Err(FdarError::InvalidDimension{..})));
```

### T-07: Determinism (Yule-Walker is fully deterministic — no seed needed)

```
let r1 = ftsm(&data, ncomp, &argvals)?;
let r2 = ftsm(&data, ncomp, &argvals)?;
assert_eq!(r1, r2, "ftsm must be bit-identical across calls");
```

---

## Divergences from R `ftsa` to Document in Rustdoc

| Feature | ftsa behaviour | fdars behaviour | Reason |
|---------|---------------|-----------------|--------|
| Pre-smoothing | Optional kernel/P-spline smoothing before FPCA | Raw grid input, no smoothing | Reuse constraint (`fdata_to_pc_1d` has no smoothing) |
| ncomp default | Default order=6 | User-provided; no default | Consistency with `fdata_to_pc_1d` API which requires explicit ncomp |
| Score model | Supports ETS, ARIMA, AR, rwdrift, rw | AR(p) with Yule-Walker + AIC | No-new-dependency constraint |
| Prediction intervals | Parametric + nonparametric bootstrap | Not implemented (out of scope) | Deferred to future milestone |
| fplsr | NIPALS/SIMPLS functional operator | Per-point PLS with scalar response | Reuse-first: adapts `fdata_to_pls_1d` |
| Dynamic update methods | BM, OLS, RR, PLS | Projection/OLS only | Scope limit; RR/PLS update deferred |
| ngrid interpolation | Interpolates to 500+ grid points | Operates on input grid as-is | No smoothing/interpolation infrastructure |

---

## Implementation File Layout

```
fdars-core/src/fts/
├── mod.rs          — add: mod forecast; pub use forecast::{ftsm, ftsm_forecast,
│                         ftsm_forecast_multistep, ftsm_update, fplsr,
│                         FtsmResult, FtsmForecastResult, FplsrResult};
│                   — add result struct definitions (or in forecast.rs)
└── forecast.rs     — NEW, all implementation:
                        // private helpers:
                        fn scalar_acov(series: &[f64], max_lag: usize) -> Vec<f64>
                        fn levinson_durbin_yw(gamma: &[f64]) -> Result<(Vec<f64>, f64), FdarError>
                        struct ArModel { ... }
                        impl ArModel { fn fit(...), fn forecast(...) }
                        // public entry points:
                        pub fn ftsm(...) -> Result<FtsmResult, FdarError>
                        pub fn ftsm_forecast(...) -> Result<FtsmForecastResult, FdarError>
                        pub fn ftsm_update(...) -> Result<FtsmResult, FdarError>
                        pub fn ftsm_forecast_multistep(...) -> Result<FtsmForecastResult, FdarError>
                        pub fn fplsr(...) -> Result<FplsrResult, FdarError>
                        #[cfg(test)] mod tests { ... }
```

Changes to existing files (minimal, additive only):
- `fts/mod.rs`: add `mod forecast;`, add `pub use forecast::{...}`, add new result struct definitions
- `src/lib.rs`: add new exports to the existing `pub use fts::{...}` block

---

## Environment Availability

Step 2.6: Codebase-only phase — no external tools, services, or runtime dependencies beyond the existing Rust toolchain. All dependencies are already in `Cargo.lock`.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | All | ✓ | 1.97.0 | — |
| nalgebra 0.33 | `fdata_to_pc_1d` SVD | ✓ | 0.33 (Cargo.lock) | — |
| rayon 1.10 | `parallel` feature (optional for this phase) | ✓ | 1.10 | Sequential fallback via `parallel.rs` macros |
| cargo clippy --all-targets --features linalg,parallel | CI gate | ✓ | bundled | — |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(test)]` inline |
| Config file | None — uses `cargo test` |
| Quick run command | `cargo test -p fdars-core --features linalg fts::forecast` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FTS-01-01 | `ftsm` fits and reconstructs curves | unit | `cargo test -p fdars-core --features linalg fts::forecast::tests::ftsm_` | ❌ Wave 0 |
| FTS-01-02 | `ftsm_forecast` scores and reconstructs forecast curve | unit | `cargo test -p fdars-core --features linalg fts::forecast::tests::forecast_` | ❌ Wave 0 |
| FTS-01-03 | `fplsr` produces valid 1-step forecast | unit | `cargo test -p fdars-core --features linalg fts::forecast::tests::fplsr_` | ❌ Wave 0 |
| FTS-01-04 | `ftsm_update` agrees with full refit within tolerance | unit | `cargo test -p fdars-core --features linalg fts::forecast::tests::update_` | ❌ Wave 0 |
| FTS-01-05 | `ftsm_forecast_multistep` h=1 equals single-step | unit | `cargo test -p fdars-core --features linalg fts::forecast::tests::multistep_` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg fts`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps

- [ ] `fdars-core/src/fts/forecast.rs` — new file (all tests inline)
- [ ] `fdars-core/src/fts/mod.rs` — add `mod forecast;` and pub use block

---

## Security Domain

`security_enforcement` not explicitly set in `.planning/config.json` — treating as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No | n/a (pure computation library, no auth) |
| V3 Session Management | No | n/a |
| V4 Access Control | No | n/a |
| V5 Input Validation | Yes | `FdarError::InvalidDimension` / `InvalidParameter` at every entry point |
| V6 Cryptography | No | Yule-Walker is deterministic; no crypto |

### Known Threat Patterns for Rust numeric library

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Division by zero in Yule-Walker (gamma(0) = 0) | Tampering | Check `gamma[0].abs() < NUMERICAL_EPS` before Levinson-Durbin; return `ComputationFailed` |
| Integer overflow in AR candidate range | Tampering | Use `usize::min`; `n / 4` is safe for any n |
| NaN propagation from degenerate score series | Tampering | Check all `sigma2` values are finite and positive after Levinson-Durbin |

---

## Package Legitimacy Audit

No new external packages are installed in this phase. All algorithms use the existing `Cargo.lock` dependencies.

| Package | Registry | Verdict | Disposition |
|---------|----------|---------|-------------|
| (none new) | — | — | No new packages |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | AIC formula is `n * ln(sigma_p^2) + 2*p` (omitting constant), matching R's `ar()` yule-walker | §Yule-Walker AR | Would cause different order selection than R baseline; easy to correct |
| A2 | The `1/n` autocovariance normalization (not `1/(n-h)`) is correct for Yule-Walker input | §Yule-Walker AR | Biased-vs-unbiased convention; difference is small for n≥50 |
| A3 | `fplsr` per-evaluation-point approach produces results equivalent to ftsa NIPALS operator | §fplsr Algorithm | Could cause forecast to be less accurate than true NIPALS; divergence documented in rustdoc |
| A4 | Dynamic update tolerance `1%` relative-L2 is achievable for 1 new observation | §Dynamic Update | Test may be too tight or too loose; adjustable at Claude's discretion |

---

## Open Questions

1. **`FtsmResult` should store `ar_models` or separate into two structs (`FtsmFit` + `FtsmForecastModel`)?**
   - What we know: The CONTEXT.md shows `ftsm_forecast` takes a `FtsmResult` reference, implying AR models must be accessible from it.
   - What's unclear: Whether users want to inspect individual `ArModel` fields (orders, coefficients) for diagnostics.
   - Recommendation: Store `ar_models` as `pub` inside `FtsmResult` with a simple public `ArModelResult` struct (order, phi, sigma2) so users can inspect fitted AR orders. Keep it simple.

2. **Should `fplsr` accept an `argvals` parameter or infer grid spacing internally?**
   - What we know: All other `fts/` entry points take explicit `argvals`.
   - Recommendation: Take `argvals: &[f64]` for consistency; same signature style as `functional_acf`.

---

## Sources

### Primary (HIGH confidence — verified by direct Read this session)

- `fdars-core/src/regression.rs:25-321` — `FpcaResult` struct fields, `project`, `reconstruct`, `fdata_to_pc_1d` signature
- `fdars-core/src/fts/acf.rs:25-201` — `validate_fts_input`, `mean_curve`, `autocovariance_matrix` (pub(crate)), `durbin_levinson_pacf` (private), all signatures
- `fdars-core/src/fts/mod.rs:1-75` — module wiring pattern, `FacfResult` struct
- `fdars-core/src/scoring.rs:59-99` — `functional_mae`, `functional_mse` signatures
- `fdars-core/src/scalar_on_function/pls.rs:45-183` — `fregre_pls`, `predict_fregre_pls`, `PlsRegressionResult`
- `fdars-core/src/regression.rs:406-417` — `PlsResult` struct fields
- `fdars-core/src/helpers.rs:4,57` — `NUMERICAL_EPS`, `simpsons_weights`
- `fdars-core/src/lib.rs:251-255` — crate-root re-export pattern for `fts` module

### Secondary (MEDIUM confidence — official R docs)

- [rdrr.io/cran/ftsa/man/ftsm.html](https://rdrr.io/cran/ftsa/man/ftsm.html) — `ftsm` decomposition, `order` argument, return structure
- [search.r-project.org/CRAN/refmans/ftsa/html/forecast.ftsm.html](https://search.r-project.org/CRAN/refmans/ftsa/html/forecast.ftsm.html) — forecast reconstruction path
- [rdrr.io/cran/ftsa/man/fplsr.html](https://rdrr.io/cran/ftsa/man/fplsr.html) — `fplsr` signature, lag-1 design, NIPALS/SIMPLS
- [rdrr.io/cran/ftsa/man/dynupdate.html](https://rdrr.io/cran/ftsa/man/dynupdate.html) — `dynupdate` projection methods
- [rdrr.io/cran/ftsa/man/ftsmiterativeforecasts.html](https://rdrr.io/cran/ftsa/man/ftsmiterativeforecasts.html) — iterative multi-step forecasting
- [stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/ar.html) — `ar()` AIC formula, default order.max
- [nmimoto.github.io/477/w5c.html](https://nmimoto.github.io/477/w5c.html) — Yule-Walker matrix form, Levinson-Durbin

### Tertiary (LOW confidence)

- [princeton.edu/~mwatson/papers/hstep_3.pdf](https://www.princeton.edu/~mwatson/papers/hstep_3.pdf) — iterated vs direct multi-step AR forecasting

---

## Metadata

**Confidence breakdown:**
- Reusable APIs: HIGH — verified by direct Read of source files
- Algorithm specifications: MEDIUM — official R docs cross-checked with textbooks
- Pitfalls: MEDIUM — derived from code structure + R doc patterns
- Test tolerances: LOW (A3, A4) — reasonable but tunable

**Research date:** 2026-08-22
**Valid until:** 2026-09-22 (stable Rust crate, no moving parts)
