//! Functional time-series forecasting (FTS-01).
//!
//! Forecasts future functional curves from a time-ordered curve series using an
//! FPCA-based functional time-series model. A curve series is decomposed via the
//! existing dense FPCA ([`crate::regression::fdata_to_pc_1d`]) into a mean curve,
//! FPC loadings, and a score-time-series; each retained FPC-score sequence is then
//! modelled as an independent univariate AR(p) process (Yule-Walker estimation,
//! AIC order selection) and forecast horizon by horizon, and the forecast scores
//! are recombined with the loadings and mean into forecast curves.
//!
//! # Entry points
//!
//! * [`ftsm`] — fit the FPCA-based functional time-series model.
//! * [`ftsm_forecast`] — h-step-ahead FPC-score-AR forecast reconstructed into
//!   forecast curves.
//! * `ftsm_forecast_multistep` — iterative multi-step forecast (added later).
//! * `ftsm_update` — dynamic forecast update on new observations (added later).
//! * `fplsr` — functional PLS forecasting variant (added later).
//!
//! # R baseline
//!
//! `ftsa` (Hyndman & Shang). Matched by capability, not by R's exact signatures.
//!
//! # Divergences from `ftsa`
//!
//! * **No pre-smoothing.** `ftsa::ftsm` optionally smooths (kernel / P-spline)
//!   before FPCA; this implementation operates on the raw input grid via dense
//!   `fdata_to_pc_1d`. Users requiring pre-smoothed curves should smooth before
//!   calling [`ftsm`].
//! * **User-provided `ncomp`.** `ftsa::ftsm` defaults to `order = 6`; here the
//!   number of retained components is a required, validated parameter.
//! * **AR-only score model.** Each FPC-score sequence is modelled as an
//!   independent univariate AR(p) (Yule-Walker + AIC), not a general ARIMA.
//! * **Point forecasts only.** Prediction intervals / forecast bands are out of
//!   scope for this milestone.
//!
//! # Conventions
//!
//! All public functions return `Result<_, FdarError>` and validate inputs at
//! entry (never panic). The AR estimation path is fully deterministic
//! (Yule-Walker), so no RNG seeding is required. Result structs derive
//! `Debug, Clone, PartialEq`, are serde-gated, and are declared in
//! [`super`](crate::fts).

use super::{ArModelResult, FplsrResult, FtsmForecastResult, FtsmResult};
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use crate::scalar_on_function::{fregre_pls, predict_fregre_pls};

// ─── Input validation ────────────────────────────────────────────────────────

/// Validate that `data` is non-empty and `argvals` length matches data columns.
///
/// Returns `(n, m)` on success. Re-implemented verbatim from `fts/acf.rs`
/// (private there) to keep this module self-contained without modifying `acf.rs`.
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

// ─── Scalar AR (Yule-Walker) machinery ───────────────────────────────────────

/// Scalar autocovariance of a series at lags `0..=max_lag`.
///
/// Uses the `1/n` normalization convention (matching `fts/acf.rs`), i.e.
/// `gamma(h) = (1/n) Σ_{t=0}^{n-h-1} (x[t] - mean) (x[t+h] - mean)`.
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

/// Levinson-Durbin recursion solving the Yule-Walker equations.
///
/// Takes autocovariances `gamma[0..=p]` and returns `(phi, sigma2)` where `phi`
/// holds the `p` AR coefficients (`phi[0]` is the lag-1 coefficient) and `sigma2`
/// is the innovation (residual) variance. Order 0 returns `(vec![], gamma[0])`.
///
/// Mirrors the structural pattern of `acf.rs::durbin_levinson_pacf`, including the
/// `nu[k-1].abs() < 1e-12` early-exit guard (near-unit-root collapse to the
/// order-`k-1` model). Returns [`FdarError::ComputationFailed`] when
/// `gamma[0]` is near zero (degenerate constant series).
fn levinson_durbin_yw(gamma: &[f64]) -> Result<(Vec<f64>, f64), FdarError> {
    let p = gamma.len() - 1;
    if p == 0 {
        return Ok((vec![], gamma[0]));
    }
    if gamma[0].abs() < NUMERICAL_EPS {
        return Err(FdarError::ComputationFailed {
            operation: "levinson_durbin_yw",
            detail: "gamma(0) near zero — degenerate score series".to_string(),
        });
    }
    let mut phi = vec![vec![0.0f64; p + 1]; p + 1];
    let mut nu = vec![0.0f64; p + 1];

    phi[1][1] = gamma[1] / gamma[0];
    nu[1] = gamma[0] * (1.0 - phi[1][1] * phi[1][1]);

    for k in 2..=p {
        if nu[k - 1].abs() < 1e-12 {
            let phi_hat: Vec<f64> = (1..k).map(|j| phi[k - 1][j]).collect();
            let sigma2 = nu[k - 1].max(0.0);
            return Ok((phi_hat, sigma2));
        }
        let num = gamma[k] - (1..k).map(|j| phi[k - 1][j] * gamma[k - j]).sum::<f64>();
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

/// A fitted univariate AR(p) model for a single FPC-score sequence.
///
/// Private: exposed to callers only through the [`ArModelResult`] diagnostics
/// carried on [`FtsmResult`].
#[derive(Debug, Clone, PartialEq)]
struct ArModel {
    /// AR coefficients `phi_1..phi_p` (0-indexed: `phi[0]` is the lag-1 coeff).
    phi: Vec<f64>,
    /// Innovation (residual) variance.
    sigma2: f64,
    /// Series mean (the series is mean-centered before Yule-Walker fitting).
    mean: f64,
    /// Selected AR order `p` (0 = white noise).
    order: usize,
    /// Last `order` raw observations (oldest first) used to seed forecasting.
    history: Vec<f64>,
}

impl ArModel {
    /// Fit an AR(p) model by Yule-Walker with AIC order selection.
    ///
    /// `AIC(p) = n·ln(sigma_p²) + 2p` over `p = 0..=p_max`, with
    /// `p_max = min(⌊10·log10(n)⌋, n-1, n/4)` (min 1) — R's `ar()` default with a
    /// conservative short-series cap.
    fn fit(series: &[f64], n: usize) -> Result<ArModel, FdarError> {
        let mean = series.iter().sum::<f64>() / n as f64;
        let p_max = ((10.0 * (n as f64).log10()).floor() as usize)
            .min(n - 1)
            .min(n / 4)
            .max(1);
        let gamma = scalar_acov(series, mean, p_max);

        // Degenerate constant series → white-noise (order 0) model.
        if gamma[0].abs() < NUMERICAL_EPS {
            return Ok(ArModel {
                phi: vec![],
                sigma2: gamma[0].max(0.0),
                mean,
                order: 0,
                history: vec![],
            });
        }

        // Order 0 baseline.
        let mut best_order = 0usize;
        let mut best_phi: Vec<f64> = vec![];
        let mut best_sigma2 = gamma[0];
        let mut best_aic = n as f64 * gamma[0].max(NUMERICAL_EPS).ln();

        for p in 1..=p_max {
            let (phi_p, sigma2_p) = match levinson_durbin_yw(&gamma[0..=p]) {
                Ok(v) => v,
                Err(_) => break,
            };
            if sigma2_p <= 0.0 {
                continue;
            }
            let aic = n as f64 * sigma2_p.ln() + 2.0 * p as f64;
            if aic < best_aic {
                best_aic = aic;
                best_order = p;
                best_phi = phi_p;
                best_sigma2 = sigma2_p;
            }
        }

        let history = if best_order == 0 {
            vec![]
        } else {
            series[n - best_order..n].to_vec()
        };

        Ok(ArModel {
            phi: best_phi,
            sigma2: best_sigma2,
            mean,
            order: best_order,
            history,
        })
    }

    /// Iterative plug-in `h`-step forecast; returns `h` forecast values.
    ///
    /// For each horizon the AR prediction is computed from a sliding window of the
    /// last `order` (mean-centered) values, mixing observed history with
    /// already-computed forecasts for horizons beyond the model order.
    fn forecast(&self, h: usize) -> Vec<f64> {
        if self.order == 0 {
            return vec![self.mean; h];
        }
        // Centered sliding window, oldest first; buf[order-1] is most recent.
        let mut buf: Vec<f64> = self.history.iter().map(|x| x - self.mean).collect();
        let mut out = Vec::with_capacity(h);
        for _ in 0..h {
            let mut pred = 0.0;
            for j in 0..self.order {
                pred += self.phi[j] * buf[self.order - 1 - j];
            }
            out.push(pred + self.mean);
            buf.remove(0);
            buf.push(pred);
        }
        out
    }
}

// ─── ftsm fit ────────────────────────────────────────────────────────────────

/// Fit an FPCA-based functional time-series model over a time-ordered curve series.
///
/// Decomposes `data` (rows = time-ordered curves, columns = evaluation points)
/// via [`fdata_to_pc_1d`], retains the mean curve, FPC loadings, and the
/// score-time-series, reconstructs fitted curves, and fits an independent
/// univariate AR(p) model (Yule-Walker + AIC) to each FPC-score sequence.
///
/// The number of retained components is `ncomp`, silently clamped by
/// `fdata_to_pc_1d` to `min(ncomp, n, m)`; the effective value is stored in
/// [`FtsmResult::ncomp`].
///
/// # Errors
///
/// Returns [`FdarError`] (never panics) for empty data / mismatched `argvals`
/// ([`FdarError::InvalidDimension`]), `ncomp == 0`, `n <= ncomp`, or `n < 2`
/// ([`FdarError::InvalidParameter`]), and propagates FPCA / AR failures.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn ftsm(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FtsmResult, FdarError> {
    let (n, _m) = validate_fts_input(data, argvals)?;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
    }
    if n <= ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: format!("ncomp ({ncomp}) must be < n ({n})"),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: format!("need at least 2 observations to fit AR score models, got n = {n}"),
        });
    }

    let fpca = fdata_to_pc_1d(data, ncomp, argvals)?;
    let effective_ncomp = fpca.rotation.ncols();
    let fitted = fpca.reconstruct(&fpca.scores, effective_ncomp)?;

    let mut ar_models = Vec::with_capacity(effective_ncomp);
    for k in 0..effective_ncomp {
        let col = fpca.scores.column(k);
        let ar = ArModel::fit(col, n)?;
        ar_models.push(ArModelResult {
            order: ar.order,
            phi: ar.phi,
            sigma2: ar.sigma2,
        });
    }

    Ok(FtsmResult {
        mean: fpca.mean,
        rotation: fpca.rotation,
        scores: fpca.scores,
        fitted,
        weights: fpca.weights,
        ncomp: effective_ncomp,
        ar_models,
    })
}

/// Reconstruct the private [`ArModel`] for component `k` from the stored
/// diagnostics plus the score-column history (the score columns are mean ≈ 0
/// because FPCA centers the data).
fn ar_model_from_fit(fit: &FtsmResult, k: usize) -> ArModel {
    let col = fit.scores.column(k);
    let n = col.len();
    let mean = col.iter().sum::<f64>() / n as f64;
    let order = fit.ar_models[k].order;
    let history = if order == 0 {
        vec![]
    } else {
        col[n - order..n].to_vec()
    };
    ArModel {
        phi: fit.ar_models[k].phi.clone(),
        sigma2: fit.ar_models[k].sigma2,
        mean,
        order,
        history,
    }
}

// ─── ftsm forecast ───────────────────────────────────────────────────────────

/// Forecast `h`-step-ahead curve(s) from a fitted [`FtsmResult`].
///
/// Each FPC-score AR model is forecast `h` horizons ahead via the iterative
/// plug-in recursion, and the forecast scores at each horizon are recombined into
/// a forecast curve as `mean[j] + Σ_k score_k · rotation[(j,k)]` (the same
/// arithmetic as [`crate::regression::FpcaResult::reconstruct`]). The returned
/// [`FtsmForecastResult::forecast`] is an `h × m` matrix (row `i` = horizon
/// `i + 1`).
///
/// For `h = 1` this is the single-step convenience forecast;
/// [`ftsm_forecast_multistep`] is the underlying multi-step path this delegates to
/// (so the `h = 1` output is bit-identical to a `h = 1` multi-step call).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] for `h < 1` and
/// [`FdarError::InvalidDimension`] when `argvals` length differs from the fitted
/// grid.
#[must_use = "returns forecast result; result should be examined"]
pub fn ftsm_forecast(
    fit: &FtsmResult,
    h: usize,
    argvals: &[f64],
) -> Result<FtsmForecastResult, FdarError> {
    ftsm_forecast_multistep(fit, h, argvals)
}

/// Iterative multi-step forecast: per-horizon forecast curves for `h >= 1`.
///
/// Each FPC-score AR model is forecast `h` horizons ahead via the iterative
/// plug-in recursion (forecast scores are fed back into the AR window for horizons
/// beyond the model order), and the forecast scores at each horizon are recombined
/// into a forecast curve as `mean[j] + Σ_k score_k · rotation[(j,k)]` (the same
/// arithmetic as [`crate::regression::FpcaResult::reconstruct`]). The returned
/// [`FtsmForecastResult::forecast`] is an `h × m` matrix (row `i` = horizon
/// `i + 1`). The `h = 1` row is bit-identical to [`ftsm_forecast`], which delegates
/// here.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] for `h < 1` and
/// [`FdarError::InvalidDimension`] when `argvals` length differs from the fitted
/// grid.
#[must_use = "returns forecast result; result should be examined"]
pub fn ftsm_forecast_multistep(
    fit: &FtsmResult,
    h: usize,
    argvals: &[f64],
) -> Result<FtsmForecastResult, FdarError> {
    if h == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "h",
            message: "h must be >= 1".to_string(),
        });
    }
    let m = fit.mean.len();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching fitted grid)"),
            actual: format!("{} elements", argvals.len()),
        });
    }

    // Forecast each component's score series h horizons ahead.
    let mut score_paths: Vec<Vec<f64>> = Vec::with_capacity(fit.ncomp);
    for k in 0..fit.ncomp {
        let ar = ar_model_from_fit(fit, k);
        score_paths.push(ar.forecast(h));
    }

    // Reconstruct forecast curves horizon by horizon.
    let mut forecast = FdMatrix::zeros(h, m);
    for step in 0..h {
        for j in 0..m {
            let mut val = fit.mean[j];
            for k in 0..fit.ncomp {
                val += score_paths[k][step] * fit.rotation[(j, k)];
            }
            forecast[(step, j)] = val;
        }
    }

    Ok(FtsmForecastResult { forecast, h })
}

// ─── ftsm dynamic update ─────────────────────────────────────────────────────

/// Dynamically update a fitted [`FtsmResult`] as new observation(s) arrive,
/// WITHOUT refitting the FPCA.
///
/// New curve row(s) are projected onto the FROZEN FPC loadings (mean, rotation,
/// weights from `fit`) using the same arithmetic as
/// [`crate::regression::FpcaResult::project`], the resulting scores are appended
/// to the score-time-series, and each per-component AR model is re-fit (same
/// Yule-Walker + AIC selection). The mean curve and loadings are never
/// recomputed.
///
/// # Divergence
///
/// The dynamic update freezes the mean curve from the initial [`ftsm`] fit; for
/// long update sequences the frozen mean can drift from the true sample mean, so
/// periodic full refit via [`ftsm`] is recommended. This is why the update agrees
/// with a full refit only up to a documented tolerance, not to machine epsilon.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] when `new_curve` has a column count
/// other than the fitted grid width, or when `argvals` length differs from the
/// fitted grid.
#[must_use = "returns forecast result; result should be examined"]
pub fn ftsm_update(
    fit: &FtsmResult,
    new_curve: &FdMatrix,
    argvals: &[f64],
) -> Result<FtsmResult, FdarError> {
    let m = fit.mean.len();
    let (k_new, m_new) = new_curve.shape();
    if k_new == 0 || m_new != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_curve",
            expected: format!("k x {m} (k >= 1 new rows matching the fitted grid)"),
            actual: format!("{k_new} rows, {m_new} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching fitted grid)"),
            actual: format!("{} elements", argvals.len()),
        });
    }

    let ncomp = fit.ncomp;
    let n_old = fit.scores.nrows();
    let n_ext = n_old + k_new;

    // Build the extended score matrix: old scores followed by projected new scores.
    let mut ext_scores = FdMatrix::zeros(n_ext, ncomp);
    for i in 0..n_old {
        for k in 0..ncomp {
            ext_scores[(i, k)] = fit.scores[(i, k)];
        }
    }
    // Project each new row onto the FROZEN loadings (FpcaResult::project arithmetic).
    for r in 0..k_new {
        for k in 0..ncomp {
            let mut sum = 0.0;
            for j in 0..m {
                sum += (new_curve[(r, j)] - fit.mean[j]) * fit.rotation[(j, k)] * fit.weights[j];
            }
            ext_scores[(n_old + r, k)] = sum;
        }
    }

    // Re-fit AR models per component on the extended score columns (no FPCA refit).
    let mut ar_models = Vec::with_capacity(ncomp);
    for k in 0..ncomp {
        let col = ext_scores.column(k);
        let ar = ArModel::fit(col, n_ext)?;
        ar_models.push(ArModelResult {
            order: ar.order,
            phi: ar.phi,
            sigma2: ar.sigma2,
        });
    }

    // Rebuild fitted curves for the extended series (mean + Σ_k score·rotation).
    let mut fitted = FdMatrix::zeros(n_ext, m);
    for i in 0..n_ext {
        for j in 0..m {
            let mut val = fit.mean[j];
            for k in 0..ncomp {
                val += ext_scores[(i, k)] * fit.rotation[(j, k)];
            }
            fitted[(i, j)] = val;
        }
    }

    Ok(FtsmResult {
        mean: fit.mean.clone(),
        rotation: fit.rotation.clone(),
        scores: ext_scores,
        fitted,
        weights: fit.weights.clone(),
        ncomp,
        ar_models,
    })
}

// ─── fplsr: functional PLS forecasting ───────────────────────────────────────

/// Functional PLS forecasting variant (a PLS-score alternative to FPC-score AR).
///
/// Fits a lag-1 PLS design — predictor = current curve (rows `0..n-2`), response
/// = next curve (rows `1..n-1`) — and forecasts the next curve one step ahead.
/// Because the shipped PLS machinery ([`crate::scalar_on_function::fregre_pls`])
/// takes a scalar response, the functional (curve) response is handled by fitting
/// one scalar PLS regression per evaluation point (Option A of the research
/// note): for each grid point `j`, the next-curve column `j` is regressed on the
/// current curve via PLS, the in-sample fits populate column `j` of `fitted`, and
/// the last observed curve is projected + predicted to give the forecast at `j`.
///
/// # Divergence from `ftsa`
///
/// `ftsa::fplsr` uses a unified NIPALS/SIMPLS functional operator; this
/// implementation uses per-evaluation-point scalar PLS regression (functionally
/// equivalent for point prediction, less elegant). Deterministic — no RNG.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] for empty data / mismatched `argvals`,
/// and [`FdarError::InvalidParameter`] for `ncomp == 0` or `n < 3` (need at least
/// two lag-1 rows plus a forecast origin). PLS/rank failures propagate as
/// [`FdarError::ComputationFailed`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fplsr(data: &FdMatrix, ncomp: usize, argvals: &[f64]) -> Result<FplsrResult, FdarError> {
    let (n, m) = validate_fts_input(data, argvals)?;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
    }
    if n < 3 {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: format!("need at least 3 observations for a lag-1 PLS forecast, got n = {n}"),
        });
    }
    // Lag-1 design has n-1 rows; clamp ncomp to avoid overfitting / rank deficiency.
    let nrows = n - 1;
    let ncomp = ncomp.min(nrows).min(m);

    let mut x_cur = FdMatrix::zeros(nrows, m);
    let mut x_next = FdMatrix::zeros(nrows, m);
    for i in 0..nrows {
        for j in 0..m {
            x_cur[(i, j)] = data[(i, j)];
            x_next[(i, j)] = data[(i + 1, j)];
        }
    }
    // Forecast origin: the last observed curve (row n-1), as a 1 x m matrix.
    let mut last = FdMatrix::zeros(1, m);
    for j in 0..m {
        last[(0, j)] = data[(n - 1, j)];
    }

    let mut forecast = FdMatrix::zeros(1, m);
    let mut fitted = FdMatrix::zeros(nrows, m);
    for j in 0..m {
        let y_j: Vec<f64> = (0..nrows).map(|i| x_next[(i, j)]).collect();
        let fit_j = fregre_pls(&x_cur, &y_j, argvals, ncomp, None)?;
        for i in 0..nrows {
            fitted[(i, j)] = fit_j.fitted_values[i];
        }
        let pred = predict_fregre_pls(&fit_j, &last, None)?;
        forecast[(0, j)] = pred[0];
    }

    Ok(FplsrResult {
        forecast,
        fitted,
        ncomp,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::functional_mse;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Deterministic pseudo-white-noise in [-0.5, 0.5] via a linear congruential
    /// generator (no RNG dependency; reproducible across runs).
    fn lcg_white(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((state >> 11) as f64) / ((1u64 << 53) as f64); // in [0,1)
            out.push(u - 0.5);
        }
        out
    }

    /// Build a synthetic curve series whose dominant FPC score follows a genuine
    /// AR(1) process with coefficient `phi` (near-white LCG innovations).
    /// curve_t(u) = a_t·f1(u) + b_t·f2(u), with a_t the AR(1) sequence and b_t a
    /// small deterministic secondary signal.
    fn ar1_curve_series(n: usize, m: usize, phi: f64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let f1: Vec<f64> = argvals
            .iter()
            .map(|u| (std::f64::consts::PI * u).sin())
            .collect();
        let f2: Vec<f64> = argvals
            .iter()
            .map(|u| (2.0 * std::f64::consts::PI * u).sin())
            .collect();
        let e = lcg_white(n, 0x5eed_1234_abcd_0001);
        let mut a = vec![0.0f64; n];
        a[0] = e[0];
        for t in 1..n {
            a[t] = phi * a[t - 1] + e[t];
        }
        let e2 = lcg_white(n, 0x5eed_1234_abcd_0002);
        let mut data = FdMatrix::zeros(n, m);
        for t in 0..n {
            let b = 0.1 * e2[t];
            for j in 0..m {
                data[(t, j)] = a[t] * f1[j] + b * f2[j];
            }
        }
        (data, argvals)
    }

    #[test]
    fn scalar_acov_variance_and_decay() {
        let series = [1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let g = scalar_acov(&series, mean, 3);
        assert!(g[0] > 0.0);
        // gamma(0) is the variance.
        let var = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
        assert!((g[0] - var).abs() < 1e-12);
    }

    #[test]
    fn levinson_durbin_recovers_ar1() {
        // gamma from a strong AR(1): gamma(h) = phi^h * gamma(0).
        let phi: f64 = 0.8;
        let g0: f64 = 1.0;
        let gamma: Vec<f64> = (0..=1i32).map(|h| g0 * phi.powi(h)).collect();
        let (phi_hat, sigma2) = levinson_durbin_yw(&gamma).unwrap();
        assert!((phi_hat[0] - 0.8).abs() < 1e-9);
        assert!(sigma2 > 0.0);
    }

    #[test]
    fn levinson_durbin_rejects_zero_variance() {
        let gamma = [0.0, 0.0];
        let err = levinson_durbin_yw(&gamma).unwrap_err();
        matches!(err, FdarError::ComputationFailed { operation, .. } if operation == "levinson_durbin_yw")
            .then_some(())
            .expect("expected ComputationFailed");
    }

    #[test]
    fn ar_model_fit_and_forecast_ar1() {
        let phi = 0.8;
        let n = 200;
        let mut a = vec![0.0f64; n];
        a[0] = 1.0;
        for t in 1..n {
            let e = 0.4 * (1.3 * t as f64).sin() + 0.3 * (2.7 * t as f64).cos();
            a[t] = phi * a[t - 1] + e;
        }
        let ar = ArModel::fit(&a, n).unwrap();
        assert!(ar.order >= 1);
        // Yule-Walker phi_1 close to the true AR coefficient.
        assert!((ar.phi[0] - 0.8).abs() < 0.1, "phi[0] = {}", ar.phi[0]);
        let f = ar.forecast(1);
        assert_eq!(f.len(), 1);
        assert!(f[0].is_finite());
    }

    #[test]
    fn ftsm_fitted_recovers_input() {
        let (data, argvals) = ar1_curve_series(120, 25, 0.7);
        let fit = ftsm(&data, 3, &argvals).unwrap();
        // Fitted curves recover the input within 1% relative-L2 (T-02).
        let mse = functional_mse(&data, &fit.fitted, &argvals).unwrap();
        // data variance proxy: MSE of data vs its column means.
        let mean_mat = {
            let (n, m) = data.shape();
            let mut mm = FdMatrix::zeros(n, m);
            for j in 0..m {
                let mu = (0..n).map(|i| data[(i, j)]).sum::<f64>() / n as f64;
                for i in 0..n {
                    mm[(i, j)] = mu;
                }
            }
            mm
        };
        let var = functional_mse(&data, &mean_mat, &argvals).unwrap();
        assert!(mse < 0.01 * var, "mse = {mse}, var = {var}");
    }

    #[test]
    fn ftsm_deterministic() {
        let (data, argvals) = ar1_curve_series(80, 20, 0.6);
        let a = ftsm(&data, 2, &argvals).unwrap();
        let b = ftsm(&data, 2, &argvals).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn ftsm_rejects_ncomp_ge_n() {
        let argvals = uniform_grid(10);
        let mut data = FdMatrix::zeros(3, 10);
        for i in 0..3 {
            for j in 0..10 {
                data[(i, j)] = (i + j) as f64;
            }
        }
        let err = ftsm(&data, 5, &argvals).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { parameter, .. } if parameter == "ncomp")
        );
    }

    #[test]
    fn ftsm_rejects_empty() {
        let data = FdMatrix::zeros(0, 0);
        let err = ftsm(&data, 1, &[]).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn ftsm_rejects_argvals_mismatch() {
        let (data, _argvals) = ar1_curve_series(30, 20, 0.5);
        let err = ftsm(&data, 2, &uniform_grid(19)).unwrap_err();
        assert!(matches!(err, FdarError::InvalidDimension { .. }));
    }

    #[test]
    fn forecast_recovers_ar_one_step() {
        let (data, argvals) = ar1_curve_series(400, 25, 0.8);
        let fit = ftsm(&data, 3, &argvals).unwrap();
        let fc = ftsm_forecast(&fit, 1, &argvals).unwrap();
        assert_eq!(fc.forecast.shape(), (1, 25));
        // Dominant-component AR fit recovers the known AR(1) coefficient near 0.8
        // (Yule-Walker point estimate; finite-sample slack).
        let ar0 = ar_model_from_fit(&fit, 0);
        assert!((ar0.phi[0] - 0.8).abs() < 0.12, "phi[0] = {}", ar0.phi[0]);
        // The one-step score forecast tracks the AR one-step prediction
        // 0.8·(last-score) (scores are FPCA-centered, so mean ≈ 0); allow 25%
        // relative slack for the coefficient estimate + any retained higher lags.
        let n = fit.scores.nrows();
        let last = fit.scores[(n - 1, 0)];
        let fscore = ar0.forecast(1)[0];
        let target = 0.8 * last;
        assert!(
            (fscore - target).abs() < 0.25 * target.abs().max(1.0),
            "fscore = {fscore}, 0.8*last = {target}"
        );
    }

    #[test]
    fn forecast_beats_naive_baseline() {
        // Fit on the first n-1 curves, forecast the n-th, compare to naive last-curve.
        let (data, argvals) = ar1_curve_series(140, 25, 0.75);
        let (n, m) = data.shape();
        let mut train = FdMatrix::zeros(n - 1, m);
        for i in 0..n - 1 {
            for j in 0..m {
                train[(i, j)] = data[(i, j)];
            }
        }
        let mut truth = FdMatrix::zeros(1, m);
        let mut naive = FdMatrix::zeros(1, m);
        for j in 0..m {
            truth[(0, j)] = data[(n - 1, j)];
            naive[(0, j)] = data[(n - 2, j)];
        }
        let fit = ftsm(&train, 3, &argvals).unwrap();
        let fc = ftsm_forecast(&fit, 1, &argvals).unwrap();
        let model_mse = functional_mse(&truth, &fc.forecast, &argvals).unwrap();
        let naive_mse = functional_mse(&truth, &naive, &argvals).unwrap();
        assert!(
            model_mse < naive_mse,
            "model_mse = {model_mse}, naive_mse = {naive_mse}"
        );
    }

    #[test]
    fn forecast_rejects_h_zero() {
        let (data, argvals) = ar1_curve_series(40, 20, 0.5);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let err = ftsm_forecast(&fit, 0, &argvals).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { parameter, .. } if parameter == "h"));
    }

    // ── Plan 02: multi-step + dynamic update ──────────────────────────────────

    #[test]
    fn multistep_h1_equals_single_step() {
        let (data, argvals) = ar1_curve_series(120, 25, 0.7);
        let fit = ftsm(&data, 3, &argvals).unwrap();
        let single = ftsm_forecast(&fit, 1, &argvals).unwrap();
        let multi = ftsm_forecast_multistep(&fit, 1, &argvals).unwrap();
        assert_eq!(single.forecast.shape(), multi.forecast.shape());
        for j in 0..25 {
            assert!(
                (single.forecast[(0, j)] - multi.forecast[(0, j)]).abs() < 1e-12,
                "mismatch at j={j}"
            );
        }
    }

    #[test]
    fn multistep_returns_h_rows() {
        let (data, argvals) = ar1_curve_series(120, 20, 0.6);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let fc = ftsm_forecast_multistep(&fit, 5, &argvals).unwrap();
        assert_eq!(fc.forecast.shape(), (5, 20));
        assert_eq!(fc.h, 5);
    }

    #[test]
    fn multistep_rejects_h_zero() {
        let (data, argvals) = ar1_curve_series(40, 20, 0.5);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let err = ftsm_forecast_multistep(&fit, 0, &argvals).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { parameter, .. } if parameter == "h"));
    }

    #[test]
    fn update_agrees_with_refit() {
        let (data, argvals) = ar1_curve_series(120, 25, 0.75);
        let (n, m) = data.shape();
        // Fit on data[0..n-1].
        let mut train = FdMatrix::zeros(n - 1, m);
        for i in 0..n - 1 {
            for j in 0..m {
                train[(i, j)] = data[(i, j)];
            }
        }
        let fit = ftsm(&train, 3, &argvals).unwrap();
        // The new observation = last row of data.
        let mut new_curve = FdMatrix::zeros(1, m);
        for j in 0..m {
            new_curve[(0, j)] = data[(n - 1, j)];
        }
        let updated = ftsm_update(&fit, &new_curve, &argvals).unwrap();
        // Full refit on data[0..n].
        let full = ftsm(&data, 3, &argvals).unwrap();
        let upd_fc = ftsm_forecast(&updated, 1, &argvals).unwrap();
        let full_fc = ftsm_forecast(&full, 1, &argvals).unwrap();
        // Relative-L2 between the two 1-step forecasts < 1% (frozen-mean divergence).
        let err = functional_mse(&full_fc.forecast, &upd_fc.forecast, &argvals).unwrap();
        let scale = {
            let mut zero = FdMatrix::zeros(1, m);
            for j in 0..m {
                zero[(0, j)] = 0.0;
            }
            functional_mse(&full_fc.forecast, &zero, &argvals).unwrap()
        };
        assert!(err < 0.01 * scale.max(1e-9), "err = {err}, scale = {scale}");
    }

    #[test]
    fn update_freezes_loadings() {
        let (data, argvals) = ar1_curve_series(80, 20, 0.6);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let mut new_curve = FdMatrix::zeros(1, 20);
        for j in 0..20 {
            new_curve[(0, j)] = data[(0, j)];
        }
        let updated = ftsm_update(&fit, &new_curve, &argvals).unwrap();
        assert_eq!(updated.mean, fit.mean);
        assert_eq!(updated.rotation, fit.rotation);
        assert_eq!(updated.weights, fit.weights);
    }

    #[test]
    fn update_extends_scores() {
        let (data, argvals) = ar1_curve_series(80, 20, 0.6);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let mut new_curve = FdMatrix::zeros(1, 20);
        for j in 0..20 {
            new_curve[(0, j)] = data[(0, j)];
        }
        let updated = ftsm_update(&fit, &new_curve, &argvals).unwrap();
        assert_eq!(updated.scores.nrows(), fit.scores.nrows() + 1);
        assert_eq!(updated.fitted.nrows(), fit.fitted.nrows() + 1);
    }

    #[test]
    fn update_rejects_bad_shape() {
        let (data, argvals) = ar1_curve_series(40, 20, 0.5);
        let fit = ftsm(&data, 2, &argvals).unwrap();
        let bad = FdMatrix::zeros(1, 19);
        let err = ftsm_update(&fit, &bad, &argvals).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { parameter, .. } if parameter == "new_curve")
        );
    }

    // ── Plan 03: functional PLS forecasting ───────────────────────────────────

    /// Full-rank AR-driven curve series for PLS tests: three AR(1) components on
    /// three basis functions plus small broadband noise, so the lag-1 predictor is
    /// well-conditioned (not confined to a low-dimensional subspace).
    fn pls_curve_series(n: usize, m: usize, phi: f64) -> (FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let basis: Vec<Vec<f64>> = (1..=3)
            .map(|c| {
                argvals
                    .iter()
                    .map(|u| (c as f64 * std::f64::consts::PI * u).sin())
                    .collect::<Vec<f64>>()
            })
            .collect();
        let phis = [phi, phi * 0.7, phi * 0.5];
        let mut comps = vec![vec![0.0f64; n]; 3];
        for (c, comp) in comps.iter_mut().enumerate() {
            let e = lcg_white(n, 0xC0FFEE00 + c as u64);
            comp[0] = e[0];
            for t in 1..n {
                comp[t] = phis[c] * comp[t - 1] + e[t];
            }
        }
        let noise = lcg_white(n * m, 0xBEEF_1234);
        let mut data = FdMatrix::zeros(n, m);
        for t in 0..n {
            for j in 0..m {
                let mut v = 0.05 * noise[t * m + j];
                for c in 0..3 {
                    v += comps[c][t] * basis[c][j];
                }
                data[(t, j)] = v;
            }
        }
        (data, argvals)
    }

    #[test]
    fn fplsr_produces_finite_forecast() {
        let (data, argvals) = pls_curve_series(60, 25, 0.7);
        let res = fplsr(&data, 2, &argvals).unwrap();
        assert_eq!(res.forecast.shape(), (1, 25));
        assert_eq!(res.fitted.shape(), (59, 25));
        for j in 0..25 {
            assert!(res.forecast[(0, j)].is_finite());
        }
    }

    #[test]
    fn fplsr_no_worse_than_naive() {
        // Fit on data[0..n-1], forecast the n-th curve, compare to naive last-curve.
        let (data, argvals) = pls_curve_series(80, 25, 0.75);
        let (n, m) = data.shape();
        let mut train = FdMatrix::zeros(n - 1, m);
        for i in 0..n - 1 {
            for j in 0..m {
                train[(i, j)] = data[(i, j)];
            }
        }
        let mut truth = FdMatrix::zeros(1, m);
        let mut naive = FdMatrix::zeros(1, m);
        for j in 0..m {
            truth[(0, j)] = data[(n - 1, j)];
            naive[(0, j)] = data[(n - 2, j)];
        }
        let res = fplsr(&train, 3, &argvals).unwrap();
        let model_mse = functional_mse(&truth, &res.forecast, &argvals).unwrap();
        let naive_mse = functional_mse(&truth, &naive, &argvals).unwrap();
        assert!(model_mse.is_finite());
        assert!(
            model_mse <= naive_mse,
            "model_mse = {model_mse}, naive_mse = {naive_mse}"
        );
    }

    #[test]
    fn fplsr_deterministic() {
        let (data, argvals) = pls_curve_series(50, 20, 0.6);
        let a = fplsr(&data, 2, &argvals).unwrap();
        let b = fplsr(&data, 2, &argvals).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fplsr_rejects_bad_input() {
        let (data, argvals) = ar1_curve_series(40, 20, 0.5);
        // empty
        let empty = FdMatrix::zeros(0, 0);
        assert!(matches!(
            fplsr(&empty, 1, &[]).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
        // ncomp == 0
        assert!(matches!(
            fplsr(&data, 0, &argvals).unwrap_err(),
            FdarError::InvalidParameter { parameter, .. } if parameter == "ncomp"
        ));
        // n < 3
        let (short, short_argvals) = ar1_curve_series(2, 20, 0.5);
        assert!(matches!(
            fplsr(&short, 1, &short_argvals).unwrap_err(),
            FdarError::InvalidParameter { parameter, .. } if parameter == "data"
        ));
        // argvals mismatch
        assert!(matches!(
            fplsr(&data, 2, &uniform_grid(19)).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
    }
}
