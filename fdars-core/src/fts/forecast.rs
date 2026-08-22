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

use super::{ArModelResult, FtsmForecastResult, FtsmResult};
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

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
/// For `h = 1` this is the single-step convenience forecast; the
/// `ftsm_forecast_multistep` entry point provides the equivalent multi-step path.
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
}
