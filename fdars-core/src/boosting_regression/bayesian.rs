//! Bayesian function-on-scalar regression via conjugate Gibbs sampler (REG-06-04).
//!
//! Fits `Y_i(t) = μ(t) + Σ_j x̃_ij · β_j(t) + ε_i(t)` where `x̃` are the
//! mean-centered scalar predictors and `β_j(t)` are functional coefficients. The
//! functional dimension is compressed with FPCA (`fdata_to_pc`): the response
//! is projected onto its top-`K` functional principal components, and for each
//! component `k` the FPC scores are regressed on the predictors with a conjugate
//! Normal / Inverse-Gamma Gibbs sampler. Each retained draw reconstructs the
//! coefficient functions `β_j(t) = Σ_k b_{jk} · φ_k(t)` from the FPCA rotation, so
//! posterior summaries (mean + **pointwise** credible bands) are obtained directly
//! in the functional domain.
//!
//! # Full conditionals (per FPC component `k`)
//!
//! Model: `ξ_k = X̃ · b_k + ε_k`, `ξ_k ∈ Rⁿ` (scores of component k), `X̃ ∈ R^{n×p}`.
//! Priors: `b_k | σ²_k ~ N(0, τ²·I_p)`, `σ²_k ~ IG(a₀, b₀)`.
//!
//! - `b_k | · ~ N(μ_post, A⁻¹)` with precision `A = X̃'X̃/σ²_k + I_p/τ²`,
//!   `μ_post = A⁻¹ · X̃'ξ_k / σ²_k`. Sampled via Cholesky `A = LLᵀ`:
//!   `b_k = μ_post + Lᵀ⁻¹ z`, `z ~ N(0, I_p)` (Rue 2001) — `Cov = (LLᵀ)⁻¹ = A⁻¹`.
//! - `σ²_k | · ~ IG(a₀ + n/2, b₀ + RSS_k/2)`, drawn as `1 / Gamma(α, 1/β)`.
//!
//! The chain is fully deterministic given `config.seed`
//! (`StdRng::seed_from_u64(seed)`).
//!
//! # References
//!
//! Rue (2001). Fast sampling of Gaussian Markov random fields. *JRSS-B* 63(2).
//! Goldsmith et al. (2015). *JCGS* 23(1). Jiang et al. (2025), arXiv:2505.05633.
//!
//! # Divergences from refund
//!
//! `refund`'s Bayesian FOSR uses spline basis priors with random effects; this
//! implementation uses FPCA score compression (`fdata_to_pc`) for simplicity and
//! zero new dependencies. Pointwise credible bands only (no simultaneous bands).

use super::{BayesianConfig, BayesianFosrResult};
use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back, compute_xtx};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc, FpcaResult};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Gamma, StandardNormal};

/// Solve `Lᵀ v = z` (back substitution) for a lower-triangular `L` stored flat
/// row-major (`L[i*p + j]`, `i ≥ j`). Used to draw `N(0, (LLᵀ)⁻¹)` as `Lᵀ⁻¹ z`.
fn back_solve_lt(l: &[f64], z: &[f64], p: usize) -> Vec<f64> {
    let mut v = z.to_vec();
    for j in (0..p).rev() {
        for k in (j + 1)..p {
            // (Lᵀ)_{jk} = L_{kj} = l[k*p + j]
            v[j] -= l[k * p + j] * v[k];
        }
        v[j] /= l[j * p + j];
    }
    v
}

/// Linear-interpolated quantile of an already-sorted slice.
fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n as f64 - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

// ---------------------------------------------------------------------------
// Private helpers for bayesian_fosr
// ---------------------------------------------------------------------------

/// Validate dimension arguments; returns `(n, m_t, p)` on success.
fn bayesian_fosr_validate_dims(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
) -> Result<(usize, usize, usize), FdarError> {
    let (n, m_t) = data.shape();
    let p = predictors.ncols();
    if n < 2 || m_t == 0 || predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/predictors",
            expected: format!("n >= 2, m_t > 0, predictors.nrows() == n (n={n})"),
            actual: format!(
                "n={n}, m_t={m_t}, predictors.nrows()={}",
                predictors.nrows()
            ),
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
    Ok((n, m_t, p))
}

/// Validate config parameters for `bayesian_fosr`.
fn bayesian_fosr_validate_config(config: &BayesianConfig) -> Result<(), FdarError> {
    if config.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }
    if config.tau2 <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "tau2",
            message: format!("must be > 0, got {}", config.tau2),
        });
    }
    if config.ig_a0 <= 0.0 || config.ig_b0 <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ig_a0/ig_b0",
            message: format!("must be > 0, got a0={}, b0={}", config.ig_a0, config.ig_b0),
        });
    }
    if config.n_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_iter",
            message: "must be >= 1".to_string(),
        });
    }
    if config.thin == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "thin",
            message: "must be >= 1".to_string(),
        });
    }
    Ok(())
}

/// FPCA + predictor centering + X'X + X'ξ_k precompute.
///
/// Returns `(fpca, xc, xtx, xt_xi)` where:
/// - `fpca`: FpcaResult with k retained components
/// - `xc`: centered design matrix (n × p, column-major)
/// - `xtx`: X̃'X̃ (p × p, row-major flat) for the Gibbs precision matrix
/// - `xt_xi[kk][j]` = Σ_i x̃_ij ξ_ik (X̃'ξ_k for each component)
fn bayesian_fosr_precompute(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BayesianConfig,
    n: usize,
    p: usize,
) -> Result<(FpcaResult, FdMatrix, Vec<f64>, Vec<Vec<f64>>), FdarError> {
    let fpca = fdata_to_pc(data, config.ncomp, argvals)?;
    let k = fpca.scores.ncols();

    // Center predictors: β_j(t) is the slope of the centered predictor.
    let mut xbar = vec![0.0f64; p];
    for j in 0..p {
        let col = predictors.column(j);
        xbar[j] = col.iter().sum::<f64>() / n as f64;
    }
    let mut xc = FdMatrix::zeros(n, p);
    for j in 0..p {
        let src = predictors.column(j);
        let dst = xc.column_mut(j);
        for i in 0..n {
            dst[i] = src[i] - xbar[j];
        }
    }

    // Precompute X̃'X̃ (p × p, row-major) and X̃'ξ_k (p-vector per component).
    let xtx = compute_xtx(&xc);
    let mut xt_xi: Vec<Vec<f64>> = vec![vec![0.0f64; p]; k];
    for kk in 0..k {
        let score_col = fpca.scores.column(kk);
        for j in 0..p {
            let xj = xc.column(j);
            let mut s = 0.0f64;
            for i in 0..n {
                s += xj[i] * score_col[i];
            }
            xt_xi[kk][j] = s;
        }
    }

    Ok((fpca, xc, xtx, xt_xi))
}

/// One Gibbs step for FPC component `kk`.
///
/// Updates `b_state[kk]` and `sigma2[kk]` in place.
/// CRITICAL: Called for each kk in 0..k inside the unchanged outer iter loop.
/// The rng draw sequence (StandardNormal z-draw, Gamma sample) must be in the
/// same order as the original to preserve numeric identity.
#[allow(clippy::too_many_arguments)]
fn bayesian_fosr_gibbs_step(
    kk: usize,
    b_state: &mut [Vec<f64>],
    sigma2: &mut [f64],
    xtx: &[f64],
    xt_xi: &[Vec<f64>],
    fpca: &FpcaResult,
    xc: &FdMatrix,
    config: &BayesianConfig,
    rng: &mut StdRng,
    n: usize,
    p: usize,
    a_post_shape: f64,
    inv_tau2: f64,
) -> Result<(), FdarError> {
    let s2 = sigma2[kk];
    // Precision A = X̃'X̃ / σ²_k + I_p / τ²  (row-major p×p).
    let mut a = vec![0.0f64; p * p];
    for idx in 0..p * p {
        a[idx] = xtx[idx] / s2;
    }
    for d in 0..p {
        a[d * p + d] += inv_tau2;
    }
    let l = cholesky_factor(&a, p)?;
    // μ_post = A⁻¹ (X̃'ξ_k / σ²_k)
    let mut rhs = vec![0.0f64; p];
    for j in 0..p {
        rhs[j] = xt_xi[kk][j] / s2;
    }
    let mu_post = cholesky_forward_back(&l, &rhs, p);
    // Draw z ~ N(0, I_p) and v = Lᵀ⁻¹ z, then b_k = μ_post + v.
    let z: Vec<f64> = (0..p)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();
    let v = back_solve_lt(&l, &z, p);
    for j in 0..p {
        b_state[kk][j] = mu_post[j] + v[j];
    }
    // RSS_k = ||ξ_k - X̃ b_k||²
    let score_col = fpca.scores.column(kk);
    let mut rss = 0.0f64;
    for i in 0..n {
        let mut fit = 0.0f64;
        for j in 0..p {
            fit += xc.column(j)[i] * b_state[kk][j];
        }
        let r = score_col[i] - fit;
        rss += r * r;
    }
    // σ²_k ~ IG(a_post, b0 + RSS/2)  drawn as 1 / Gamma(shape, scale = 1/rate)
    let rate = config.ig_b0 + rss / 2.0;
    let gamma = Gamma::new(a_post_shape, 1.0 / rate).map_err(|e| FdarError::ComputationFailed {
        operation: "bayesian_fosr Inverse-Gamma draw",
        detail: format!("Gamma::new failed (shape={a_post_shape}, rate={rate}): {e}"),
    })?;
    let g = gamma.sample(rng);
    sigma2[kk] = 1.0 / g.max(f64::MIN_POSITIVE);
    Ok(())
}

/// Record one thinned post-burn-in draw into beta_draws and beta_sum.
///
/// Reconstructs β_j(t) = Σ_k b_{kj} · φ_k(t) for each predictor j and time t,
/// appending to beta_draws and accumulating beta_sum.
fn bayesian_fosr_record_draw(
    b_state: &[Vec<f64>],
    fpca: &FpcaResult,
    beta_draws: &mut [Vec<f64>],
    beta_sum: &mut [f64],
    k: usize,
    m_t: usize,
    p: usize,
) {
    for j in 0..p {
        for t in 0..m_t {
            let mut beta = 0.0f64;
            for kk in 0..k {
                beta += b_state[kk][j] * fpca.rotation[(t, kk)];
            }
            beta_draws[j * m_t + t].push(beta);
            beta_sum[j * m_t + t] += beta;
        }
    }
}

/// Compute posterior summaries (beta_mean/lower/upper), fitted values, residuals,
/// and sigma2_mean from the retained draws.
///
/// Takes `beta_draws` by move (for in-place sorting per cell).
/// Returns `(beta_mean, beta_lower, beta_upper, fitted, residuals, sigma2_mean)`.
fn bayesian_fosr_summaries(
    mut beta_draws: Vec<Vec<f64>>,
    beta_sum: &[f64],
    n: usize,
    p: usize,
    m_t: usize,
    fpca: &FpcaResult,
    xc: &FdMatrix,
    data: &FdMatrix,
    q: usize,
) -> Result<(FdMatrix, FdMatrix, FdMatrix, FdMatrix, FdMatrix, Vec<f64>), FdarError> {
    let mut beta_mean = FdMatrix::zeros(p, m_t);
    let mut beta_lower = FdMatrix::zeros(p, m_t);
    let mut beta_upper = FdMatrix::zeros(p, m_t);
    for j in 0..p {
        for t in 0..m_t {
            let cell = &mut beta_draws[j * m_t + t];
            beta_mean[(j, t)] = beta_sum[j * m_t + t] / q as f64;
            cell.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            beta_lower[(j, t)] = quantile_sorted(cell, 0.025);
            beta_upper[(j, t)] = quantile_sorted(cell, 0.975);
        }
    }

    // Fitted / residuals / σ²(t)
    // fitted(i,t) = mean_Y(t) + Σ_j x̃_ij · β̄_j(t)
    let mut fitted = FdMatrix::zeros(n, m_t);
    let mut residuals = FdMatrix::zeros(n, m_t);
    let mut sigma2_mean = vec![0.0f64; m_t];
    for t in 0..m_t {
        for i in 0..n {
            let mut val = fpca.mean[t];
            for j in 0..p {
                val += xc.column(j)[i] * beta_mean[(j, t)];
            }
            fitted[(i, t)] = val;
            let r = data[(i, t)] - val;
            residuals[(i, t)] = r;
            sigma2_mean[t] += r * r;
        }
        sigma2_mean[t] /= n as f64;
    }

    Ok((
        beta_mean,
        beta_lower,
        beta_upper,
        fitted,
        residuals,
        sigma2_mean,
    ))
}

/// Bayesian function-on-scalar regression via conjugate Gibbs sampler.
///
/// Compresses the functional response with FPCA, runs a conjugate Normal /
/// Inverse-Gamma Gibbs sampler on the FPC-score regression coefficients, and
/// reconstructs posterior summaries of the coefficient functions `β_j(t)`.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t, column-major).
/// * `predictors` — Scalar predictor matrix X (n × p). Centered internally.
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `config` — [`BayesianConfig`] (FPC components, priors, iterations, seed).
///
/// # Returns
///
/// [`BayesianFosrResult`] with posterior-mean coefficient functions `beta_mean`
/// (p × m_t), pointwise 2.5% / 97.5% credible bands, posterior-mean fitted values
/// and residuals, and the posterior-mean residual variance `sigma2_mean(t)`.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] on shape mismatch; [`FdarError::InvalidParameter`]
/// on out-of-range config; [`FdarError::ComputationFailed`] if FPCA or a Cholesky
/// draw fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn bayesian_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BayesianConfig,
) -> Result<BayesianFosrResult, FdarError> {
    // ---- Validation --------------------------------------------------------
    let (n, m_t, p) = bayesian_fosr_validate_dims(data, predictors, argvals)?;
    bayesian_fosr_validate_config(config)?;

    // ---- Precompute: FPCA + centering + X'X + X'ξ_k -----------------------
    let (fpca, xc, xtx, xt_xi) = bayesian_fosr_precompute(data, predictors, argvals, config, n, p)?;
    let k = fpca.scores.ncols();

    // ---- Gibbs state init --------------------------------------------------
    let mut b_state: Vec<Vec<f64>> = vec![vec![0.0f64; p]; k];
    let mut sigma2: Vec<f64> = vec![1.0f64; k];

    let inv_tau2 = 1.0 / config.tau2;
    let a_post_shape = config.ig_a0 + n as f64 / 2.0;

    let mut rng = StdRng::seed_from_u64(config.seed);

    let total = config.burn_in + config.n_iter * config.thin;
    let q_retained = config.n_iter;

    // Per-(j,t) storage of reconstructed β draws for pointwise quantiles.
    let mut beta_draws: Vec<Vec<f64>> = vec![Vec::with_capacity(q_retained); p * m_t];
    let mut beta_sum = vec![0.0f64; p * m_t];

    // ---- Gibbs loop --------------------------------------------------------
    // CRITICAL: outer `iter` loop, inner `kk` loop — preserves RNG draw order.
    for iter in 0..total {
        for kk in 0..k {
            bayesian_fosr_gibbs_step(
                kk,
                &mut b_state,
                &mut sigma2,
                &xtx,
                &xt_xi,
                &fpca,
                &xc,
                config,
                &mut rng,
                n,
                p,
                a_post_shape,
                inv_tau2,
            )?;
        }

        // Retain thinned post-burn-in draws.
        if iter >= config.burn_in && (iter - config.burn_in) % config.thin == 0 {
            bayesian_fosr_record_draw(&b_state, &fpca, &mut beta_draws, &mut beta_sum, k, m_t, p);
        }
    }

    let q = beta_draws[0].len().max(1);

    // ---- Posterior summaries + fitted/residuals/sigma2_mean ----------------
    let (beta_mean, beta_lower, beta_upper, fitted, residuals, sigma2_mean) =
        bayesian_fosr_summaries(beta_draws, &beta_sum, n, p, m_t, &fpca, &xc, data, q)?;

    Ok(BayesianFosrResult {
        beta_mean,
        beta_lower,
        beta_upper,
        fitted,
        residuals,
        sigma2_mean,
        n_iter: config.n_iter,
        burn_in: config.burn_in,
        thin: config.thin,
        ncomp: k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

    fn default_config() -> BayesianConfig {
        BayesianConfig {
            ncomp: 4,
            tau2: 100.0,
            ig_a0: 0.001,
            ig_b0: 0.001,
            n_iter: 400,
            burn_in: 200,
            thin: 1,
            seed: 20260824,
        }
    }

    /// Y_i(t) = a(t) + x_i · β(t) + small noise, β(t) = sin(π t).
    fn make_fosr_dataset(n: usize, m: usize) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);
        let x1: Vec<f64> = (0..n)
            .map(|i| -1.0 + 2.0 * i as f64 / (n - 1).max(1) as f64)
            .collect();
        let predictors = FdMatrix::from_column_major(x1.clone(), n, 1).unwrap();

        let mut y = vec![0.0f64; n * m];
        for (t_idx, &tv) in argvals.iter().enumerate() {
            let a_t = 0.5 * (2.0 * PI * tv).cos(); // intercept function
            let beta_t = (PI * tv).sin(); // true coefficient
            for i in 0..n {
                let noise = 0.02 * ((i as f64 * 1.2345 + t_idx as f64 * 0.678).sin());
                y[i + t_idx * n] = a_t + x1[i] * beta_t + noise;
            }
        }
        (
            FdMatrix::from_column_major(y, n, m).unwrap(),
            predictors,
            argvals,
        )
    }

    #[test]
    fn bayesian_fosr_recovers_beta() {
        let (data, predictors, argvals) = make_fosr_dataset(60, 25);
        let result = bayesian_fosr(&data, &predictors, &argvals, &default_config()).unwrap();
        assert_eq!(result.beta_mean.shape(), (1, 25));
        // Posterior mean β̄(t) correlates strongly with the true sin(π t).
        let m = 25;
        let mut dot = 0.0;
        let mut nb = 0.0;
        let mut nt = 0.0;
        for t in 0..m {
            let tv = argvals[t];
            let truth = (PI * tv).sin();
            let est = result.beta_mean[(0, t)];
            dot += truth * est;
            nb += est * est;
            nt += truth * truth;
        }
        let corr = dot / (nb.sqrt() * nt.sqrt());
        assert!(
            corr > 0.9,
            "posterior mean β should track the true coefficient (corr={corr:.3})"
        );
    }

    #[test]
    fn bayesian_fosr_credible_bands_bracket_mean() {
        let (data, predictors, argvals) = make_fosr_dataset(50, 20);
        let result = bayesian_fosr(&data, &predictors, &argvals, &default_config()).unwrap();
        for t in 0..20 {
            let lo = result.beta_lower[(0, t)];
            let hi = result.beta_upper[(0, t)];
            let mean = result.beta_mean[(0, t)];
            assert!(lo <= mean + 1e-9, "lower band must be <= mean at t={t}");
            assert!(hi >= mean - 1e-9, "upper band must be >= mean at t={t}");
            assert!(lo.is_finite() && hi.is_finite());
        }
    }

    #[test]
    fn bayesian_fosr_is_deterministic_under_seed() {
        let (data, predictors, argvals) = make_fosr_dataset(40, 15);
        let cfg = default_config();
        let r1 = bayesian_fosr(&data, &predictors, &argvals, &cfg).unwrap();
        let r2 = bayesian_fosr(&data, &predictors, &argvals, &cfg).unwrap();
        assert_eq!(
            r1.beta_mean, r2.beta_mean,
            "same seed → identical posterior mean"
        );
        assert_eq!(r1.beta_lower, r2.beta_lower);
        assert_eq!(r1.beta_upper, r2.beta_upper);
        assert_eq!(r1.sigma2_mean, r2.sigma2_mean);
    }

    #[test]
    fn bayesian_fosr_sigma2_positive_and_shapes() {
        let (data, predictors, argvals) = make_fosr_dataset(40, 18);
        let result = bayesian_fosr(&data, &predictors, &argvals, &default_config()).unwrap();
        assert_eq!(result.fitted.shape(), (40, 18));
        assert_eq!(result.residuals.shape(), (40, 18));
        assert_eq!(result.sigma2_mean.len(), 18);
        assert!(result.sigma2_mean.iter().all(|&s| s > 0.0 && s.is_finite()));
    }

    #[test]
    fn bayesian_fosr_errors_on_dimension_mismatch() {
        let (data, _predictors, argvals) = make_fosr_dataset(30, 12);
        // predictors with wrong row count
        let bad = FdMatrix::from_column_major(vec![0.0; 10], 10, 1).unwrap();
        assert!(bayesian_fosr(&data, &bad, &argvals, &default_config()).is_err());
    }

    #[test]
    fn bayesian_fosr_errors_on_invalid_params() {
        let (data, predictors, argvals) = make_fosr_dataset(30, 12);
        let mut cfg = default_config();
        cfg.tau2 = -1.0;
        assert!(bayesian_fosr(&data, &predictors, &argvals, &cfg).is_err());
        let mut cfg2 = default_config();
        cfg2.ncomp = 0;
        assert!(bayesian_fosr(&data, &predictors, &argvals, &cfg2).is_err());
    }
}
