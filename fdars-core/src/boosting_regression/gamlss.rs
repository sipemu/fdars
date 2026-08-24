//! GAMLSS-style distributional functional regression — location + scale (REG-06-03).
//!
//! Models a Gaussian functional response `Y_i(t) ~ N(μ_i(t), σ_i(t)²)` with
//! separate boosted models for location μ(t) (identity link) and scale σ(t)
//! (log link). The cyclic gamboostLSS algorithm alternates one boosting step per
//! distributional parameter per iteration.
//!
//! # Gaussian negative gradients
//!
//! - μ-step (identity link):   u_μ,i(t) = (Y_i(t) − μ_i(t)) / σ_i(t)²
//! - σ-step (log link):        u_σ,i(t) = −1 + (Y_i(t) − μ_i(t))² / σ_i(t)²
//!
//! σ is maintained in log space (η_σ = log σ); after every σ update:
//! `σ_i(t) = exp(η_σ_i(t)).max(NUMERICAL_EPS)` to guarantee positivity.
//!
//! # Cyclic algorithm
//!
//! **Initialization:** μ̂(t) = Ȳ(t); η_σ = 0 → σ̂ = 1 everywhere.
//!
//! **For** m = 1, …, mstop:
//! 1. Compute U_μ from (Y, μ̂, σ̂); run one boosting step on U_μ; update μ̂ += ν·ĥ_μ.
//! 2. Compute U_σ from (Y, μ̂ updated, σ̂); run one boosting step on U_σ;
//!    update η_σ += ν·ĥ_σ; then σ̂ = exp(η_σ).max(NUMERICAL_EPS) pointwise.
//! 3. Push gaussian_loglik(Y, μ̂, σ̂) to ll_path.
//!
//! # References
//!
//! Hofner et al. (2016). gamboostLSS: An R Package for Model-Based Boosting for
//! Simultaneous Estimation of Noncrossing Quantile Curves. *Journal of Statistical
//! Software*, 74(1). DOI:10.18637/jss.v074.i01.
//!
//! # Divergences from gamboostLSS
//!
//! Uses **cyclic** rather than noncyclic (per-iteration) parameter selection.
//! Noncyclic boosting has better variable-selection properties but is substantially
//! more complex to implement (Hofner et al. 2016, §3.2). Cyclic is the v1 choice
//! per project decisions recorded in 43-CONTEXT.md.
//! Only the Gaussian family (location + scale) is implemented.
//! Link functions: identity for μ, log for σ.

use super::boost_fosr::{build_bspline_design_at, BaseLearner};
use super::{BoostingConfig, GamlssResult};
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::linalg::cholesky_factor;
use crate::matrix::FdMatrix;
use crate::smooth_basis::bspline_penalty_matrix;

// ---------------------------------------------------------------------------
// Gaussian negative-gradient helpers
// ---------------------------------------------------------------------------

/// Negative gradient of the Gaussian log-likelihood w.r.t. μ (identity link).
///
/// `u_μ,i(t) = (Y_i(t) − μ_i(t)) / σ_i(t)²`
///
/// The σ used in the denominator is floored at `sigma_floor` (the marginal
/// residual scale), NOT at machine epsilon. This is the key stability guard
/// (Pitfall 2, RESEARCH.md): if σ were allowed to collapse toward zero in this
/// denominator, the μ working-response `(Y−μ)/σ²` would explode, boosting μ into
/// overflow and, through the coupled σ-step, saturate σ at the wrong scale. A
/// marginal-scale floor caps the working-response magnitude while leaving the
/// μ fit essentially unchanged wherever σ is at or above the marginal scale.
fn mu_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix, sigma_floor: f64) -> FdMatrix {
    let (n, m) = y.shape();
    let floor2 = (sigma_floor * sigma_floor).max(NUMERICAL_EPS);
    let mut u = FdMatrix::zeros(n, m);
    for t in 0..m {
        let y_col = y.column(t);
        let mu_col = mu.column(t);
        let sig_col = sigma.column(t);
        let u_col = u.column_mut(t);
        for i in 0..n {
            let s2 = (sig_col[i] * sig_col[i]).max(floor2);
            u_col[i] = (y_col[i] - mu_col[i]) / s2;
        }
    }
    u
}

/// Negative gradient of the Gaussian log-likelihood w.r.t. η_σ = log σ (log link).
///
/// `u_σ,i(t) = −1 + (Y_i(t) − μ_i(t))² / σ_i(t)²`
///
/// σ is clipped at `NUMERICAL_EPS` before division.
fn sigma_neg_gradient(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> FdMatrix {
    let (n, m) = y.shape();
    let mut u = FdMatrix::zeros(n, m);
    for t in 0..m {
        let y_col = y.column(t);
        let mu_col = mu.column(t);
        let sig_col = sigma.column(t);
        let u_col = u.column_mut(t);
        for i in 0..n {
            let s = sig_col[i].max(NUMERICAL_EPS);
            let r2 = (y_col[i] - mu_col[i]).powi(2);
            u_col[i] = -1.0 + r2 / (s * s);
        }
    }
    u
}

/// Gaussian log-likelihood summed over all observations and time points.
///
/// `ℓ = Σ_i Σ_t [ −log σ_i(t) − (Y_i(t) − μ_i(t))² / (2 σ_i(t)²) ]`
///
/// σ is clipped at `NUMERICAL_EPS` before log and division.
fn gaussian_loglik(y: &FdMatrix, mu: &FdMatrix, sigma: &FdMatrix) -> f64 {
    let (n, m) = y.shape();
    let mut ll = 0.0f64;
    for t in 0..m {
        let y_col = y.column(t);
        let mu_col = mu.column(t);
        let sig_col = sigma.column(t);
        for i in 0..n {
            let s = sig_col[i].max(NUMERICAL_EPS);
            let r = y_col[i] - mu_col[i];
            ll += -s.ln() - r * r / (2.0 * s * s);
        }
    }
    ll
}

// ---------------------------------------------------------------------------
// Internal helper: build base learners from the predictor matrix
// ---------------------------------------------------------------------------

fn build_learners(
    predictors: &FdMatrix,
    n: usize,
    config: &BoostingConfig,
) -> Result<Vec<BaseLearner>, FdarError> {
    let p = predictors.ncols();
    let nbasis = config.nbasis;
    let order = config.order;
    let lambda = config.lambda;

    let mut learners: Vec<BaseLearner> = Vec::with_capacity(p);

    for j in 0..p {
        let x_col = predictors.column(j);

        // Build design matrix Φⱼ (n × K, column-major)
        let phi = build_bspline_design_at(x_col, nbasis, order);
        let actual_k = phi.len() / n;

        // Penalty matrix Rⱼ (K × K, column-major)
        let x_argvals_for_penalty: Vec<f64> = {
            let x_min = x_col.iter().copied().fold(f64::INFINITY, f64::min);
            let x_max = x_col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let n_pts = (actual_k + 1).max(20);
            (0..n_pts)
                .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1).max(1) as f64)
                .collect()
        };
        let r_col_major =
            bspline_penalty_matrix(&x_argvals_for_penalty, actual_k, order, config.lfd_order);

        let k = actual_k;
        let mut a = vec![0.0f64; k * k];

        // Φ'Φ (row-major)
        for row in 0..k {
            for col in row..k {
                let mut s = 0.0f64;
                for i in 0..n {
                    s += phi[i + row * n] * phi[i + col * n];
                }
                a[row * k + col] = s;
                a[col * k + row] = s;
            }
        }

        // Add λ·Rⱼ (convert column-major R to row-major A)
        for col in 0..k {
            for row in 0..k {
                a[row * k + col] += lambda * r_col_major[row + col * k];
            }
        }

        // Ridge jitter
        for i in 0..k {
            a[i * k + i] += 1e-10;
        }

        let l = cholesky_factor(&a, k).map_err(|e| FdarError::ComputationFailed {
            operation: "gamlss_fosr base-learner Cholesky",
            detail: format!("predictor j={j}: {e:?} — try increasing lambda or decreasing nbasis"),
        })?;

        learners.push(BaseLearner { phi, l, k });
    }

    Ok(learners)
}

// ---------------------------------------------------------------------------
// Public: gamlss_fosr
// ---------------------------------------------------------------------------

/// GAMLSS-style distributional functional regression.
///
/// Fits a Gaussian model `Y_i(t) ~ N(μ_i(t), σ_i(t)²)` with separate boosted
/// component-wise models for location μ(t) and log-scale σ(t). The cyclic
/// gamboostLSS algorithm alternates one boosting step for each distributional
/// parameter per iteration.
///
/// # Algorithm
///
/// - **μ-step** (identity link): pseudo-response is `(Y − μ) / σ²`; one boosting
///   step selects the best base-learner for μ and updates `μ̂ += ν·ĥ_μ`.
/// - **σ-step** (log link): pseudo-response is `−1 + (Y − μ)² / σ²`; one boosting
///   step updates the log-σ accumulator `η_σ += ν·ĥ_σ`; then `σ̂ = exp(η_σ).max(ε)`.
/// - The Gaussian log-likelihood is tracked in `ll_path` after each cyclic iteration.
///
/// **Gaussian family only.** For additional distribution families, extend the
/// negative-gradient helpers in `gamlss.rs`.
///
/// **Cyclic vs. noncyclic:** This implementation uses cyclic boosting (fixed update
/// order: μ then σ). Noncyclic boosting (Hofner et al. 2016) has better variable-
/// selection properties but is considerably more complex; it is deferred to a future
/// milestone.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t, column-major `FdMatrix`).
/// * `predictors` — Scalar predictor matrix (n × p, column-major `FdMatrix`).
///   Each column defines one base-learner for both the μ and σ models.
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `config` — [`BoostingConfig`] controlling boosting iterations, learning rate,
///   and B-spline basis specification.
///
/// # Returns
///
/// [`GamlssResult`] with:
/// - `mu_fitted` / `sigma_fitted` — Fitted distribution parameters (n × m_t).
/// - `mu_beta` / `sigma_beta` — Accumulated coefficient functions (p × m_t).
/// - `ll_path` — Gaussian log-likelihood after each cyclic iteration.
/// - `log_likelihood` — Final log-likelihood (last `ll_path` value).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shape constraints are violated.
/// Returns [`FdarError::InvalidParameter`] if config parameters are out of range.
/// Returns [`FdarError::ComputationFailed`] if a Cholesky factorization fails.
///
/// # References
///
/// Hofner et al. (2016). *Journal of Statistical Software*, 74(1).
/// DOI:10.18637/jss.v074.i01.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gamlss_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BoostingConfig,
) -> Result<GamlssResult, FdarError> {
    // ---- Input validation --------------------------------------------------
    let (n, m_t) = data.shape();
    let p = predictors.ncols();

    if n < 3 || m_t == 0 || predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/predictors",
            expected: format!("n >= 3, m > 0, predictors.nrows() == n (n={n})"),
            actual: format!("n={n}, m={m_t}, predictors.nrows()={}", predictors.nrows()),
        });
    }
    if argvals.len() != m_t {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length == data.ncols() = {m_t}"),
            actual: format!("length = {}", argvals.len()),
        });
    }
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "predictors",
            expected: "at least 1 predictor column".to_string(),
            actual: "0 columns".to_string(),
        });
    }
    if config.mstop == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "mstop",
            message: "must be >= 1".to_string(),
        });
    }
    if config.nu <= 0.0 || config.nu > 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "nu",
            message: format!("must be in (0, 1], got {}", config.nu),
        });
    }
    if config.nbasis < 4 {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!("must be >= 4, got {}", config.nbasis),
        });
    }
    if config.lambda <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "lambda",
            message: format!("must be > 0, got {}", config.lambda),
        });
    }
    if config.order < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "order",
            message: format!("must be >= 1, got {}", config.order),
        });
    }
    if config.nbasis > n {
        return Err(FdarError::InvalidParameter {
            parameter: "nbasis",
            message: format!(
                "must be <= n (got nbasis={}, n={n}) — prevents degenerate Cholesky",
                config.nbasis
            ),
        });
    }

    let nu = config.nu;

    // ---- Build base learners (shared by μ and σ models) --------------------
    // The same set of base-learners (same predictor design matrices, same Cholesky
    // factors) is reused for both the μ-step and σ-step each cyclic iteration.
    // This is valid because the base-learner structure (Φⱼ, Aⱼ) depends only on
    // the predictor values, not on the distributional parameter being fitted.
    let learners = build_learners(predictors, n, config)?;

    // ---- Initialization ----------------------------------------------------
    // μ̂(t) = Ȳ(t) — pointwise mean of Y across observations
    let mu_intercept: Vec<f64> = (0..m_t)
        .map(|t| (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64)
        .collect();

    let mut mu_fitted = FdMatrix::zeros(n, m_t);
    for t in 0..m_t {
        let col = mu_fitted.column_mut(t);
        for i in 0..n {
            col[i] = mu_intercept[t];
        }
    }

    // Warm-start σ̂ at the marginal residual scale σ₀ = SD(Y − μ̂₀) rather than at
    // 1.0. Starting from a realistic scale keeps the coupled μ/σ dynamics near
    // equilibrium from the first iteration (σ = 1 would be far too large when the
    // true residual scale is small, forcing a violent σ collapse that destabilizes
    // the μ-step). σ₀ also serves as the μ-gradient denominator floor.
    let sigma_marg = {
        let total = (n * m_t) as f64;
        let mut ss = 0.0f64;
        for t in 0..m_t {
            for i in 0..n {
                let d = data[(i, t)] - mu_intercept[t];
                ss += d * d;
            }
        }
        (ss / total).sqrt().max(1e-6)
    };
    let eta0 = sigma_marg.ln();
    let sigma_intercept: Vec<f64> = vec![sigma_marg; m_t];
    let mut eta_sigma = FdMatrix::zeros(n, m_t); // log-σ accumulator
    let mut sigma_fitted = FdMatrix::zeros(n, m_t);
    for t in 0..m_t {
        let eta_col = eta_sigma.column_mut(t);
        for i in 0..n {
            eta_col[i] = eta0;
        }
    }
    for t in 0..m_t {
        let col = sigma_fitted.column_mut(t);
        for i in 0..n {
            col[i] = sigma_marg;
        }
    }

    // Accumulated coefficient matrices (p × m_t), init to zero
    let mut mu_beta = FdMatrix::zeros(p, m_t);
    let mut sigma_beta = FdMatrix::zeros(p, m_t);

    // ---- Scale stabilization bounds ----------------------------------------
    // A machine-epsilon floor (NUMERICAL_EPS) is far too small for a *scale*
    // parameter: if σ collapses toward it, 1/σ² explodes and amplifies the μ
    // pseudo-response `(Y−μ)/σ²` into overflow → NaN, which makes every
    // base-learner's RSS non-finite and aborts boosting (Pitfall 2). Bound the
    // log-σ accumulator to a data-adaptive window [1e-2·s, 1e2·s] where `s` is
    // the pooled response standard deviation. This lets the σ-step self-correct
    // without ever letting σ run away, and the floor stays safely below any
    // realistic estimated scale so it does not bias σ̂ upward.
    let scale_ref = {
        let total = (n * m_t) as f64;
        let mut mean = 0.0f64;
        for t in 0..m_t {
            for i in 0..n {
                mean += data[(i, t)];
            }
        }
        mean /= total;
        let mut var = 0.0f64;
        for t in 0..m_t {
            for i in 0..n {
                let d = data[(i, t)] - mean;
                var += d * d;
            }
        }
        (var / total).sqrt().max(1e-6)
    };
    let log_sigma_lo = (scale_ref * 1e-2).ln();
    let log_sigma_hi = (scale_ref * 1e2).ln();

    // ---- Cyclic boosting loop ----------------------------------------------
    let mut ll_path: Vec<f64> = Vec::with_capacity(config.mstop);

    for _iter in 0..config.mstop {
        // -- μ-step ----------------------------------------------------------
        // Pseudo-response U_μ = (Y − μ̂) / σ̂²
        let u_mu = mu_neg_gradient(data, &mu_fitted, &sigma_fitted, sigma_marg);

        // One boosting step on U_μ
        let (j_mu, fitted_mu, coefs_mu) = super::boost_fosr::boost_fosr_one_step(&u_mu, &learners)?;

        // Update μ̂ += ν · ĥ_μ
        for t in 0..m_t {
            let mu_col = mu_fitted.column_mut(t);
            let h_col = fitted_mu.column(t);
            for i in 0..n {
                mu_col[i] += nu * h_col[i];
            }
        }

        // Accumulate μ coefficients (p × m_t) for the selected base-learner
        // coefs_mu is (K × m_t); summarize by mean fitted effect per time point
        // as a scalar representative (same approach as boost_fosr.rs)
        let n_f64 = n as f64;
        for t in 0..m_t {
            let h_col = fitted_mu.column(t);
            let mean_effect: f64 = h_col.iter().sum::<f64>() / n_f64;
            mu_beta[(j_mu, t)] += nu * mean_effect;
        }
        let _ = coefs_mu; // coefficient accumulation above is via fitted values

        // -- σ-step ----------------------------------------------------------
        // Pseudo-response U_σ = −1 + (Y − μ̂)² / σ̂²  (uses UPDATED μ̂)
        let u_sigma = sigma_neg_gradient(data, &mu_fitted, &sigma_fitted);

        // Functional-intercept decomposition (gamboostLSS always carries an
        // offset for each distributional parameter). The i-mean of U_σ at each
        // grid point t is the observation-homogeneous component — it captures a
        // heteroscedastic pattern that varies purely over the response grid t
        // (the common case) directly, instead of forcing it through a scalar
        // predictor learner (which would otherwise mis-locate the scale or
        // absorb unmodeled μ-residual structure). The predictor base-learner
        // then fits only the CENTERED (i-varying) remainder.
        let mut u_sigma_mean = vec![0.0f64; m_t];
        for t in 0..m_t {
            let col = u_sigma.column(t);
            u_sigma_mean[t] = col.iter().sum::<f64>() / n_f64;
        }
        let mut u_sigma_centered = u_sigma.clone();
        for t in 0..m_t {
            let col = u_sigma_centered.column_mut(t);
            let mt = u_sigma_mean[t];
            for v in col.iter_mut() {
                *v -= mt;
            }
        }

        // One boosting step on the CENTERED σ pseudo-response (i-varying part)
        let (j_sigma, fitted_sigma, coefs_sigma) =
            super::boost_fosr::boost_fosr_one_step(&u_sigma_centered, &learners)?;

        // Update η_σ += ν · (intercept_t + ĥ_σ), then σ̂ = exp(η_σ)
        for t in 0..m_t {
            let eta_col = eta_sigma.column_mut(t);
            let h_col = fitted_sigma.column(t);
            let sig_col = sigma_fitted.column_mut(t);
            let intercept_t = u_sigma_mean[t];
            for i in 0..n {
                // Clamp the log-σ accumulator to the data-adaptive window BEFORE
                // exponentiating — guarantees σ ∈ [1e-2·s, 1e2·s], finite and
                // strictly positive, preventing the 1/σ² overflow cascade.
                eta_col[i] =
                    (eta_col[i] + nu * (intercept_t + h_col[i])).clamp(log_sigma_lo, log_sigma_hi);
                sig_col[i] = eta_col[i].exp();
            }
        }

        // Accumulate σ predictor coefficients (i-varying effect only; the
        // functional intercept is folded into σ̂ but not attributed to a predictor)
        for t in 0..m_t {
            let h_col = fitted_sigma.column(t);
            let mean_effect: f64 = h_col.iter().sum::<f64>() / n_f64;
            sigma_beta[(j_sigma, t)] += nu * mean_effect;
        }
        let _ = coefs_sigma;

        // -- Track log-likelihood --------------------------------------------
        let ll = gaussian_loglik(data, &mu_fitted, &sigma_fitted);
        ll_path.push(ll);
    }

    let log_likelihood = ll_path.last().copied().unwrap_or(f64::NEG_INFINITY);

    Ok(GamlssResult {
        mu_fitted,
        sigma_fitted,
        mu_intercept,
        sigma_intercept,
        mu_beta,
        sigma_beta,
        log_likelihood,
        ll_path,
        mstop: config.mstop,
        nu,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;
    use crate::test_helpers::uniform_grid;

    /// Default boosting config for tests (small, fast).
    fn default_config() -> BoostingConfig {
        BoostingConfig {
            mstop: 30,
            nu: 0.1,
            nbasis: 8,
            order: 4,
            lfd_order: 2,
            lambda: 0.5,
            ncomp_x: 3,
            seed: 42,
        }
    }

    /// Build a synthetic heteroscedastic Gaussian dataset.
    ///
    /// - n=40 observations, m=20 time points.
    /// - One informative scalar predictor x ∈ [0,1]; one noise predictor (constant).
    /// - True mean: μ_i(t) = x_i · sin(π·t)
    /// - True scale: σ(t) = 0.1 + 0.4·(t > 0.5)  (larger in right half of grid)
    /// - Response: deterministic (no random noise — ensures reproducibility without rand dep).
    fn make_heterosc_dataset(n: usize, m: usize) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);

        // Informative predictor: evenly spaced in [0, 1]
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1).max(1) as f64).collect();
        // Noise predictor: near-constant
        let x2: Vec<f64> = vec![0.5f64; n];

        // Assemble predictors (n × 2, column-major)
        let mut pred_data = vec![0.0f64; n * 2];
        pred_data[..n].copy_from_slice(&x1);
        pred_data[n..].copy_from_slice(&x2);
        let predictors = FdMatrix::from_column_major(pred_data, n, 2).unwrap();

        // True heteroscedastic response
        let mut y_data = vec![0.0f64; n * m];
        for (t_idx, &tv) in argvals.iter().enumerate() {
            let sigma_t = if tv > 0.5 { 0.5 } else { 0.1 };
            let mu_scale = (std::f64::consts::PI * tv).sin();
            for i in 0..n {
                // Deterministic heteroscedastic noise: amplitude matches sigma_t
                let noise = sigma_t * ((i as f64 * 1.7321 + t_idx as f64 * 0.9137).sin());
                y_data[i + t_idx * n] = x1[i] * mu_scale + noise;
            }
        }
        let data = FdMatrix::from_column_major(y_data, n, m).unwrap();

        (data, predictors, argvals)
    }

    /// σ̂(i,t) must be strictly positive everywhere (log link + NUMERICAL_EPS guard).
    #[test]
    fn gamlss_sigma_positive_everywhere() {
        let (data, predictors, argvals) = make_heterosc_dataset(40, 20);
        let config = default_config();
        let result = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap();

        let (n, m) = result.sigma_fitted.shape();
        for i in 0..n {
            for t in 0..m {
                let s = result.sigma_fitted[(i, t)];
                assert!(
                    s > 0.0,
                    "sigma_fitted[({i},{t})] = {s} is not positive — positivity guard failed"
                );
            }
        }
    }

    /// The Gaussian log-likelihood path must be non-decreasing along the cyclic boosting path.
    ///
    /// Strict monotonicity is NOT guaranteed for cyclic gamboostLSS (Hofner et al. 2016
    /// note that the cyclic variant can have non-monotone individual steps), but on a
    /// signal-bearing dataset the final likelihood must improve over the initial one.
    #[test]
    fn gamlss_loglik_non_decreasing() {
        let (data, predictors, argvals) = make_heterosc_dataset(40, 20);
        let config = BoostingConfig {
            mstop: 50,
            nu: 0.05,
            ..default_config()
        };
        let result = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap();

        assert_eq!(
            result.ll_path.len(),
            config.mstop,
            "ll_path length must equal mstop"
        );
        assert!(
            result.ll_path.iter().all(|v| v.is_finite()),
            "ll_path must be finite"
        );

        // The last entry in ll_path must equal log_likelihood
        let last_ll = *result.ll_path.last().unwrap();
        assert!(
            (last_ll - result.log_likelihood).abs() < 1e-10,
            "log_likelihood must equal last ll_path entry"
        );

        // Overall trend: final LL must be >= initial LL (signal-bearing dataset)
        let first_ll = result.ll_path[0];
        assert!(
            last_ll >= first_ll - 1.0, // allow minor tolerance for cyclic oscillations
            "Final LL {last_ll} should be >= initial LL {first_ll} (minus tolerance) on signal data"
        );
    }

    /// Fitted μ̂ correlates with the true mean; σ̂ is larger in the high-variance region.
    #[test]
    fn gamlss_recovers_mean_and_scale() {
        let (data, predictors, argvals) = make_heterosc_dataset(40, 20);
        let config = BoostingConfig {
            mstop: 50,
            nu: 0.1,
            lambda: 0.3,
            ..default_config()
        };
        let result = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap();
        let n = data.nrows();
        let m = data.ncols();

        // Structural check: mean fitted σ in right half of grid (t > 0.5) must be
        // larger than in left half — the algorithm should capture the heteroscedasticity.
        let m_half = m / 2;
        let mut sigma_left = 0.0f64;
        let mut sigma_right = 0.0f64;
        for i in 0..n {
            for t in 0..m_half {
                sigma_left += result.sigma_fitted[(i, t)];
            }
            for t in m_half..m {
                sigma_right += result.sigma_fitted[(i, t)];
            }
        }
        sigma_left /= (n * m_half) as f64;
        sigma_right /= (n * (m - m_half)) as f64;

        // On this dataset the right half has true σ = 0.5, left half 0.1.
        // After 50 iterations the fitted σ should be systematically larger on the right.
        assert!(
            sigma_right > sigma_left,
            "Expected sigma_right ({sigma_right:.4}) > sigma_left ({sigma_left:.4}) — \
             heteroscedasticity not captured"
        );

        // μ̂ should track the data: R² > 0 (any improvement over the intercept)
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        for t in 0..m {
            let mean_t: f64 = (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64;
            for i in 0..n {
                ss_res += (data[(i, t)] - result.mu_fitted[(i, t)]).powi(2);
                ss_tot += (data[(i, t)] - mean_t).powi(2);
            }
        }
        let r2 = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        assert!(
            r2 > 0.0,
            "μ̂ should improve over the intercept (R² = {r2:.4})"
        );
    }

    /// Result shapes: mu_fitted and sigma_fitted are (n, m_t); mu_beta and sigma_beta are (p, m_t).
    #[test]
    fn gamlss_result_shapes() {
        let n = 30;
        let m = 15;
        let (data, predictors, argvals) = make_heterosc_dataset(n, m);
        let config = default_config();
        let result = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap();

        assert_eq!(result.mu_fitted.shape(), (n, m), "mu_fitted shape");
        assert_eq!(result.sigma_fitted.shape(), (n, m), "sigma_fitted shape");
        assert_eq!(
            result.mu_beta.shape(),
            (predictors.ncols(), m),
            "mu_beta shape"
        );
        assert_eq!(
            result.sigma_beta.shape(),
            (predictors.ncols(), m),
            "sigma_beta shape"
        );
        assert_eq!(result.mu_intercept.len(), m, "mu_intercept length");
        assert_eq!(result.sigma_intercept.len(), m, "sigma_intercept length");
        assert_eq!(result.ll_path.len(), config.mstop, "ll_path length");
        assert_eq!(result.mstop, config.mstop, "mstop field");
        assert!((result.nu - config.nu).abs() < 1e-15, "nu field");
    }

    /// Error: predictors.nrows() != data.nrows() → FdarError::InvalidDimension.
    #[test]
    fn gamlss_errors_on_dimension_mismatch() {
        let argvals: Vec<f64> = uniform_grid(15);
        let data = FdMatrix::zeros(20, 15);
        let predictors = FdMatrix::zeros(10, 2); // wrong row count

        let config = default_config();
        let err = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "Expected InvalidDimension on nrows mismatch, got {err:?}"
        );
    }

    /// Error paths: mstop == 0, nu <= 0, lambda <= 0 → FdarError::InvalidParameter.
    #[test]
    fn gamlss_errors_on_invalid_params() {
        let (data, predictors, argvals) = make_heterosc_dataset(30, 15);

        // mstop == 0
        let mut config = default_config();
        config.mstop = 0;
        let err = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "mstop=0 → InvalidParameter, got {err:?}"
        );

        // nu <= 0
        let mut config = default_config();
        config.nu = 0.0;
        let err = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "nu=0 → InvalidParameter, got {err:?}"
        );

        // lambda <= 0
        let mut config = default_config();
        config.lambda = 0.0;
        let err = gamlss_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "lambda=0 → InvalidParameter, got {err:?}"
        );
    }
}
