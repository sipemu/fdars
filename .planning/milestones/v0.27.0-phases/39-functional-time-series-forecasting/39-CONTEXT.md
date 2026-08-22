# Phase 39: Functional Time-Series Forecasting - Context

**Gathered:** 2026-08-22
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey areas proposed in tables, all accepted as recommended

<domain>
## Phase Boundary

Deliver functional time-series forecasting (FTS-01) as new code in `fdars-core/src/fts/forecast.rs`: an FPCA-based functional time-series model (`ftsm`), FPC-score-regression forecasting, a functional PLS forecasting variant (`fplsr`), a dynamic forecast-update path, and iterative multi-step (h>1) forecasting. All entry points are `Result`-returning, crate-root re-exported, consume a time-ordered curve series as column-major `FdMatrix`, and return structured numeric output. Reuse `fdata_to_pc_1d` (dense FPCA), `scoring.rs` (forecast-error metrics), and the shipped FTS-02 `fts/acf.rs` (serial-dependence / order selection). Additive/non-breaking — zero changes to existing public signatures, no new crate dependency. Numeric point forecasts only.

</domain>

<decisions>
## Implementation Decisions

### Score-Series Model & Forecasting
- Fit an AR(p) model to each FPC-score sequence via Yule-Walker (no new dependency, deterministic; matches `ftsa` default behaviour).
- Select AR order via AIC over a candidate order range, informed by the FTS-02 PACF machinery in `fts/acf.rs`.
- Model each FPC-score series with an **independent univariate AR** (ftsa convention — FPC scores are approximately uncorrelated); no joint VAR.
- Reconstruct forecast curves as `mean + Σ_k (forecast_score_k · loading_k)`, truncated to the retained `ncomp` components.

### `ftsm` Fit & API Shape
- `ncomp` is a user-provided parameter, validated (consistent with `fdata_to_pc_1d`); no auto variance-threshold pick.
- The `Ftsm` result struct carries: mean curve, FPC loadings (rotation), the retained score-time-series, reconstructed fitted curves, integration weights, and `ncomp`.
- Five snake_case entry points, crate-root re-exported: `ftsm` (fit), `ftsm_forecast` (h-step FPC-score forecast), `fplsr` (PLS forecasting variant), `ftsm_update` (dynamic update), `ftsm_forecast_multistep` (iterative h>1).
- Fitted-curve recovery is asserted against a documented **relative-L2 tolerance** on synthetic data (inline `#[cfg(test)]` tests).

### `fplsr`, Dynamic Update, Multi-step
- `fplsr` regresses the next curve on the current curve via a lag-1 PLS design, reusing existing PLS machinery (`scalar_on_function/pls.rs` patterns) rather than a new PLS subsystem.
- Dynamic update projects new observation(s) onto the existing FPC loadings, appends the new scores, and re-forecasts the per-score AR models **without refitting FPCA**; agrees with a full refit within a documented tolerance.
- Multi-step (h>1) uses iterative plug-in: forecast scores are fed back into the AR recursion horizon by horizon.
- The multi-step h=1 curve must equal the single-step `ftsm_forecast` output (test-enforced consistency).

### Validation, Determinism, Scope
- Invalid inputs return `FdarError` (not panic): empty/too-short series, fewer observations than requested components, `ncomp` out of range, `h < 1`, non-monotone/mismatched `argvals`, degenerate columns.
- Deterministic where possible (Yule-Walker AR fit is deterministic). A `seed` parameter is added **only** on any stochastic path (e.g. if a bootstrap/permutation path is introduced), using per-thread `StdRng::seed_from_u64(seed + k)`.
- Forecast quality is validated by reusing `scoring.rs` metrics (functional MSE/MAE) and asserting lower error than a naive last-curve baseline.
- Prediction intervals / forecast bands are **out of scope** (numeric point forecasts only) — deferred to a future milestone.

### Claude's Discretion
- Exact `Ftsm` field names, internal AR/Yule-Walker helper structure, candidate AR order range, and specific documented tolerance constants are at Claude's discretion, guided by ftsa conventions and codebase style.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdata_to_pc_1d(data, ncomp, argvals) -> Result<FpcaResult, FdarError>` (`src/regression.rs:287`) — dense 1D FPCA. `FpcaResult` carries `singular_values`, `rotation` (loadings, m×ncomp), `scores` (n×ncomp), `mean`, `centered`, `weights`, and `.project(&new_data)` for scoring new observations onto the FPC space.
- `scoring.rs` — `functional_mae`, `functional_mse`, `functional_mape`, `functional_msle`, `functional_explained_variance` (all `Result`-returning).
- `fts/acf.rs` — `functional_acf`/`functional_pacf` (public) plus private `durbin_levinson_pacf(rho)` for scalar PACF; `validate_fts_input(data, argvals)`, `mean_curve`, Simpson's-weight helpers. `FacfResult` in `fts/mod.rs`.
- `scalar_on_function/pls.rs` — `fregre_pls(...)` / `predict_fregre_pls(...)` for PLS regression patterns to mirror in `fplsr`.
- `helpers.rs` — `simpsons_weights`, `trapz`, `NUMERICAL_EPS`.

### Established Patterns
- Column-major `FdMatrix` (`src/matrix.rs`) with row helpers `row_to_buf`, `row_dot`, `row_l2_sq`.
- All public fns return `Result<T, FdarError>`; validate dimensions at entry; result structs derive `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]`.
- `fts/mod.rs` uses `mod acf; pub use acf::{...}` and declares result structs at module level — add `mod forecast; pub use forecast::{...}` and new result structs the same way.
- Feature-gated rayon parallelism via `parallel.rs` macros; per-thread RNG seeding `StdRng::seed_from_u64(seed + k)`.

### Integration Points
- New file `fdars-core/src/fts/forecast.rs`; wire into `fts/mod.rs` (`mod forecast; pub use forecast::{...}`) and crate-root re-export in `src/lib.rs`.
- Reuse (do not modify): `fdata_to_pc_1d`, `scoring.rs`, `fts/acf.rs`, `scalar_on_function/pls.rs`.

</code_context>

<specifics>
## Specific Ideas

- R baseline is `ftsa` (Hyndman & Shang). Match by capability, not exact R signatures; document any divergence from `ftsa` in rustdoc (as prior milestones did).
- `ftsm` = FPCA decomposition of the time-ordered curve series into mean + FPC loadings + score-time-series + reconstructed fitted curves.
- On a synthetic series whose FPC scores follow a known AR process, the forecast scores must recover the AR one-step prediction within a documented tolerance.

</specifics>

<deferred>
## Deferred Ideas

- Prediction intervals / bootstrap forecast bands (numeric point forecasts only this phase).
- Spectral / frequency-domain functional time series (FTS-03 — separate v2 backlog item, depends on this phase).
- Joint VAR across FPC scores (independent univariate AR chosen for this phase).

</deferred>
