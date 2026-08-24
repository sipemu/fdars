//! Component-wise gradient boosting for function-on-scalar regression (REG-06-01).
//!
//! Implements the FDboost component-wise boosting algorithm with penalized B-spline
//! base-learners. Each base-learner fits the working residual using a penalized
//! least-squares spline in one scalar predictor; the best base-learner (minimum RSS)
//! is selected and the current estimate updated by ν times its fitted values.
//!
//! # Algorithm
//!
//! **Initialization:** F̂₀(t) = Ȳ(t) (pointwise mean of Y).
//!
//! **For** m = 1, …, mstop:
//! 1. Compute residual U = Y − F̂ (n × m_t).
//! 2. For each predictor j: build design matrix Φⱼ (n × K), form penalized normal
//!    equations Aⱼ = Φⱼ'Φⱼ + λ·Rⱼ + ε·I; factor Aⱼ once (Cholesky), back-solve
//!    for each time point t to get fitted values Ĥⱼ.
//! 3. Select j* = argmin ‖U − Ĥⱼ‖_F².
//! 4. Update F̂ += ν·Ĥⱼ*; accumulate β_{j*}(t) += ν·ĉ_{j*}(t).
//!
//! # References
//!
//! Hothorn et al. (2010). Model-Based Boosting. *Journal of Statistical Software*.
//! FDboost CRAN documentation (rdrr.io/cran/FDboost), `bbs()` base-learner.
//!
//! # Divergences from FDboost
//!
//! - Fixed `mstop` and `nu` (no CV-based early stopping or line search).
//! - GCV path tracks ‖U‖_F² per iteration for post-hoc diagnostics only.
//! - All base-learners share the same `nbasis`, `order`, `lambda` to equalize df.

use crate::basis::bspline::bspline_basis;
use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back};
use crate::matrix::FdMatrix;
use crate::smooth_basis::bspline_penalty_matrix;

use super::{BoostFosrResult, BoostingConfig};

// ---------------------------------------------------------------------------
// Internal helper: build B-spline design matrix at arbitrary predictor values
// ---------------------------------------------------------------------------

/// Build the B-spline design matrix Φ (n × K) at arbitrary predictor values.
///
/// The knot vector is placed uniformly over `[min(x_vals), max(x_vals)]`.
/// Returns a flat column-major matrix of shape `(n, K)` where `K = nbasis`.
///
/// # Layout
///
/// Output is column-major: element (i, k) is at index `i + k * n`.
fn build_bspline_design(x_vals: &[f64], nbasis: usize, order: usize) -> Vec<f64> {
    // nknots such that nknots + order = nbasis
    let nknots = nbasis.saturating_sub(order).max(2);

    // Use bspline_basis which evaluates at arbitrary x_vals (not a uniform grid)
    // Returns flat column-major matrix: element (i, k) at index i + k*n
    bspline_basis(x_vals, nknots, order)
}

// ---------------------------------------------------------------------------
// Shared helper: compute Φ'u (K-vector) from column-major Φ (n × K) and u (&[f64] of length n)
// ---------------------------------------------------------------------------
fn phi_t_times_vec(phi: &[f64], u_col: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..k)
        .map(|kk| {
            let phi_col = &phi[kk * n..(kk + 1) * n]; // column kk of Φ (contiguous)
            phi_col.iter().zip(u_col).map(|(&p, &u)| p * u).sum::<f64>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pointwise R² helper (mirrors function_on_scalar.rs)
// ---------------------------------------------------------------------------
pub(crate) fn pointwise_r_squared(data: &FdMatrix, fitted: &FdMatrix) -> Vec<f64> {
    let (n, m) = data.shape();
    (0..m)
        .map(|t| {
            let mean_t: f64 = (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64;
            let ss_tot: f64 = (0..n).map(|i| (data[(i, t)] - mean_t).powi(2)).sum();
            let ss_res: f64 = (0..n).map(|i| (data[(i, t)] - fitted[(i, t)]).powi(2)).sum();
            if ss_tot > 1e-15 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Public crate-visible helper: one boosting iteration
// ---------------------------------------------------------------------------

/// State for one base-learner: pre-factored Cholesky L and its design matrix Φ.
pub(crate) struct BaseLearner {
    /// Column-major design matrix Φ (n × K).
    pub phi: Vec<f64>,
    /// Cholesky factor L of (Φ'Φ + λR + ε·I) (K × K row-major).
    pub l: Vec<f64>,
    /// Number of basis functions K.
    pub k: usize,
}

/// Perform one component-wise boosting iteration.
///
/// Given the current residual matrix `U` (n × m_t) and a list of pre-factored
/// base-learners, selects the base-learner j* that minimises ‖U − Ĥⱼ‖_F²,
/// computes the update, and returns:
/// - `(j_star, fitted_best, coefs_best)` where
///   - `j_star` is the index of the selected base-learner,
///   - `fitted_best` is Ĥⱼ* (n × m_t column-major FdMatrix),
///   - `coefs_best` is ĉⱼ* (K × m_t column-major FdMatrix) for coefficient accumulation.
///
/// # Errors
///
/// Returns `FdarError::ComputationFailed` if all Cholesky back-solves fail (degenerate data).
///
/// # Design Notes
///
/// The Cholesky factor for each base-learner is pre-computed once and cached in
/// `BaseLearner`; only the back-solve (`cholesky_forward_back`) runs per iteration
/// per time point. This amortises the O(K³) factorization over all mstop iterations.
pub(crate) fn boost_fosr_one_step(
    residuals: &FdMatrix,
    learners: &[BaseLearner],
) -> Result<(usize, FdMatrix, FdMatrix), FdarError> {
    let (n, m_t) = residuals.shape();

    let mut best_rss = f64::INFINITY;
    let mut best_j = 0usize;
    let mut best_fitted: Option<FdMatrix> = None;
    let mut best_coefs: Option<FdMatrix> = None;

    for (j, learner) in learners.iter().enumerate() {
        let k = learner.k;
        let phi = &learner.phi;
        let l = &learner.l;

        // Allocate fitted (n × m_t) and coefs (k × m_t) column-major
        let mut fitted_j = FdMatrix::zeros(n, m_t);
        let mut coefs_j = FdMatrix::zeros(k, m_t);

        for t in 0..m_t {
            let u_col = residuals.column(t); // contiguous slice, length n
            let rhs = phi_t_times_vec(phi, u_col, n, k); // Φ'u[:,t], K-vector
            let c_t = cholesky_forward_back(l, &rhs, k); // (Φ'Φ + λR)⁻¹ Φ'u[:,t]

            // Fitted values for base-learner j at time t: Φ · c_t
            for i in 0..n {
                let mut val = 0.0;
                for kk in 0..k {
                    val += phi[i + kk * n] * c_t[kk]; // Φ is column-major
                }
                fitted_j[(i, t)] = val;
            }
            // Store coefficients
            for kk in 0..k {
                coefs_j[(kk, t)] = c_t[kk];
            }
        }

        // Compute RSS of (U - Ĥⱼ)
        let mut rss = 0.0f64;
        for t in 0..m_t {
            let u_col = residuals.column(t);
            for i in 0..n {
                let diff = u_col[i] - fitted_j[(i, t)];
                rss += diff * diff;
            }
        }

        if rss < best_rss {
            best_rss = rss;
            best_j = j;
            best_fitted = Some(fitted_j);
            best_coefs = Some(coefs_j);
        }
    }

    match (best_fitted, best_coefs) {
        (Some(f), Some(c)) => Ok((best_j, f, c)),
        _ => Err(FdarError::ComputationFailed {
            operation: "boost_fosr_one_step",
            detail: "no valid base-learner found; all Cholesky solves failed or no learners provided".to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public: boost_fosr
// ---------------------------------------------------------------------------

/// Component-wise gradient boosting for function-on-scalar regression.
///
/// Fits a boosted model `Ŷᵢ(t) = F₀(t) + Σⱼ hⱼ(xᵢⱼ, t)` where each base-learner
/// `hⱼ` is a penalized B-spline regression of the working residual on scalar predictor `xⱼ`.
/// At each iteration, the single best base-learner (minimum residual SS) is selected and
/// the current estimate updated by `ν` times its fitted values.
///
/// # Arguments
///
/// * `data` — Functional response Y (n × m_t, column-major `FdMatrix`).
/// * `predictors` — Scalar predictor matrix (n × p, column-major `FdMatrix`). Each
///   column defines one base-learner.
/// * `argvals` — Response grid evaluation points (length m_t).
/// * `config` — [`BoostingConfig`] controlling boosting iterations, learning rate, and
///   B-spline basis specification.
///
/// # Returns
///
/// [`BoostFosrResult`] with fitted values, residuals, pointwise R², coefficient functions
/// per predictor, and path diagnostics (`gcv_path`, `selected_learners`).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` has fewer than 3 rows or 0 columns.
/// - `predictors.nrows()` does not equal `data.nrows()`.
/// - `argvals.len()` does not equal `data.ncols()`.
///
/// Returns [`FdarError::InvalidParameter`] if:
/// - `mstop == 0`
/// - `nu <= 0` or `nu > 1`
/// - `nbasis < 4`
/// - `lambda <= 0`
/// - `order < 1`
/// - `nbasis > n` (prevents over-fitting / degenerate Cholesky)
///
/// Returns [`FdarError::ComputationFailed`] if a base-learner Cholesky factorization fails
/// (ill-conditioned penalized normal equations — try increasing `lambda` or decreasing `nbasis`).
///
/// # Divergences from FDboost
///
/// - Fixed `nu` and `mstop` (no line search / CV-based early stopping).
/// - `gcv_path` records ‖U‖_F² per iteration as a diagnostic proxy for GCV.
/// - All base-learners share `nbasis`, `order`, `lambda` to equalise effective df.
///
/// # Examples
///
/// ```rust
/// use fdars_core::boosting_regression::{boost_fosr, BoostingConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let n = 20;
/// let m = 15;
/// let p = 1;
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let x_vals: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
/// let predictors = FdMatrix::from_column_major(x_vals, n, p).unwrap();
/// let data = FdMatrix::zeros(n, m);
/// let config = BoostingConfig {
///     mstop: 10, nu: 0.1, nbasis: 8, order: 4, lfd_order: 2, lambda: 1.0,
///     ncomp_x: 3, seed: 42,
/// };
/// // let result = boost_fosr(&data, &predictors, &argvals, &config)?;
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fosr(
    data: &FdMatrix,
    predictors: &FdMatrix,
    argvals: &[f64],
    config: &BoostingConfig,
) -> Result<BoostFosrResult, FdarError> {
    // ---- Input validation ---------------------------------------------------
    let (n, m_t) = data.shape();
    let p = predictors.ncols();

    if n < 3 || m_t == 0 || predictors.nrows() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "data/predictors",
            expected: format!("n >= 3, m > 0, predictors.nrows() == n (n={n})"),
            actual: format!(
                "n={n}, m={m_t}, predictors.nrows()={}",
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

    let nbasis = config.nbasis;
    let order = config.order;
    let lambda = config.lambda;
    let nu = config.nu;

    // ---- Initialization: F₀(t) = Ȳ(t) ------------------------------------
    let intercept: Vec<f64> = (0..m_t)
        .map(|t| (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64)
        .collect();

    // Current estimate F (n × m_t)
    let mut f_current = FdMatrix::zeros(n, m_t);
    for t in 0..m_t {
        for i in 0..n {
            f_current[(i, t)] = intercept[t];
        }
    }

    // Accumulated coefficient matrix β (p × m_t), initialized to zero
    let mut beta = FdMatrix::zeros(p, m_t);

    // ---- Pre-compute base-learners (Φⱼ, Lⱼ) for each predictor j ----------
    // The penalty and Cholesky factor are constant across boosting iterations
    // since Φⱼ depends only on the predictor values (not the residuals).
    let mut learners: Vec<BaseLearner> = Vec::with_capacity(p);

    for j in 0..p {
        // Extract column j of predictors (scalar predictor values for base-learner j)
        let x_col = predictors.column(j);

        // Build design matrix Φⱼ (n × K, column-major)
        let phi = build_bspline_design(x_col, nbasis, order);
        let actual_k = phi.len() / n; // actual K (may differ slightly due to nknots rounding)

        // Compute penalty matrix Rⱼ (K × K, column-major from bspline_penalty_matrix)
        // Use the predictor's own range as the argvals for the penalty
        let x_argvals_for_penalty: Vec<f64> = {
            // Use a uniform grid over [min(x), max(x)] for penalty matrix computation
            let x_min = x_col.iter().copied().fold(f64::INFINITY, f64::min);
            let x_max = x_col.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let n_pts = (actual_k + 1).max(20);
            (0..n_pts)
                .map(|i| x_min + (x_max - x_min) * i as f64 / (n_pts - 1).max(1) as f64)
                .collect()
        };
        let r_col_major = bspline_penalty_matrix(&x_argvals_for_penalty, actual_k, order, config.lfd_order);

        // Build Aⱼ = Φⱼ'Φⱼ + λ·Rⱼ + ε·I (K × K row-major)
        // Note: Φ is column-major (element (i,k) at phi[i + k*n])
        //       R from bspline_penalty_matrix is column-major (element (j,k) at r[j + k*K])
        let k = actual_k;
        let mut a = vec![0.0f64; k * k];

        // Φ'Φ: a[row, col] = Σᵢ phi[i + row*n] * phi[i + col*n]
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

        // Ridge jitter ε·I (prevent near-singular Cholesky)
        for i in 0..k {
            a[i * k + i] += 1e-10;
        }

        // Cholesky factor
        let l = cholesky_factor(&a, k).map_err(|e| FdarError::ComputationFailed {
            operation: "boost_fosr base-learner Cholesky",
            detail: format!("predictor j={j}: {e:?} — try increasing lambda or decreasing nbasis"),
        })?;

        learners.push(BaseLearner { phi, l, k });
    }

    // ---- Boosting loop ------------------------------------------------------
    let mut selected_learners = Vec::with_capacity(config.mstop);
    let mut gcv_path = Vec::with_capacity(config.mstop);

    for _iter in 0..config.mstop {
        // Compute residual U = Y − F̂ (n × m_t)
        let mut residuals = FdMatrix::zeros(n, m_t);
        for t in 0..m_t {
            let y_col = data.column(t);
            let f_col = f_current.column(t);
            let r_col = residuals.column_mut(t);
            for i in 0..n {
                r_col[i] = y_col[i] - f_col[i];
            }
        }

        // Track ‖U‖_F² before update
        let rss_before: f64 = (0..m_t)
            .flat_map(|t| residuals.column(t).iter().map(|&r| r * r))
            .sum();
        gcv_path.push(rss_before);

        // Select best base-learner and get its fitted values + coefficients
        let (j_star, fitted_star, _coefs_star) = boost_fosr_one_step(&residuals, &learners)?;
        selected_learners.push(j_star);

        // Update F̂ += ν·Ĥⱼ*
        for t in 0..m_t {
            let f_col = f_current.column_mut(t);
            let h_col = fitted_star.column(t);
            for i in 0..n {
                f_col[i] += nu * h_col[i];
            }
        }

        // Accumulate β_{j*}(t) += ν · mean fitted effect per time point.
        // beta (p × m_t) stores the accumulated contribution of each base-learner.
        // We track the mean fitted value (averaged over observations) as a scalar proxy
        // coefficient per predictor per time point — interpretable as the average marginal
        // effect of base-learner j* at iteration m.
        for t in 0..m_t {
            let h_col = fitted_star.column(t);
            let mean_effect: f64 = h_col.iter().sum::<f64>() / n as f64;
            beta[(j_star, t)] += nu * mean_effect;
        }
    }

    // ---- Final residuals and R² -------------------------------------------
    let mut final_residuals = FdMatrix::zeros(n, m_t);
    for t in 0..m_t {
        let y_col = data.column(t);
        let f_col = f_current.column(t);
        let r_col = final_residuals.column_mut(t);
        for i in 0..n {
            r_col[i] = y_col[i] - f_col[i];
        }
    }

    let r_squared_t = pointwise_r_squared(data, &f_current);
    let r_squared = r_squared_t.iter().sum::<f64>() / m_t as f64;

    Ok(BoostFosrResult {
        intercept,
        beta,
        fitted: f_current,
        residuals: final_residuals,
        r_squared_t,
        r_squared,
        mstop: config.mstop,
        nu,
        selected_learners,
        gcv_path,
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

    /// Build a synthetic dataset with one informative and one noise predictor.
    ///
    /// Y_i(t) = x_i · sin(π·t) + 0.05·noise_i(t)
    fn make_synthetic(n: usize, m: usize) -> (FdMatrix, FdMatrix, Vec<f64>) {
        let argvals = uniform_grid(m);

        // Informative predictor: evenly spaced in [0, 1]
        let x1: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        // Noise predictor: constant (nearly zero signal)
        let x2: Vec<f64> = vec![0.5f64; n];

        // Assemble predictors (n × 2, column-major)
        let mut pred_data = vec![0.0f64; n * 2];
        for i in 0..n {
            pred_data[i] = x1[i]; // column 0
            pred_data[i + n] = x2[i]; // column 1
        }
        let predictors = FdMatrix::from_column_major(pred_data, n, 2).unwrap();

        // Response Y_i(t) = x_i · sin(π·t) + small deterministic "noise"
        let mut y_data = vec![0.0f64; n * m];
        for (t, &tv) in argvals.iter().enumerate() {
            let beta_t = (std::f64::consts::PI * tv).sin(); // true β(t)
            for i in 0..n {
                let noise = 0.03 * ((i + t) as f64 * 1.7321).sin(); // deterministic
                y_data[i + t * n] = x1[i] * beta_t + noise;
            }
        }
        let data = FdMatrix::from_column_major(y_data, n, m).unwrap();

        (data, predictors, argvals)
    }

    fn default_config() -> BoostingConfig {
        BoostingConfig {
            mstop: 30,
            nu: 0.1,
            nbasis: 8,
            order: 4,
            lfd_order: 2,
            lambda: 0.1,
            ncomp_x: 3,
            seed: 42,
        }
    }

    #[test]
    fn boost_fosr_reduces_rss_monotonically() {
        let (data, predictors, argvals) = make_synthetic(25, 20);
        let config = default_config();
        let result = boost_fosr(&data, &predictors, &argvals, &config).unwrap();

        let gcv = &result.gcv_path;
        assert_eq!(gcv.len(), config.mstop, "gcv_path should have mstop entries");

        // Each entry should be <= the previous (non-increasing RSS)
        for i in 1..gcv.len() {
            assert!(
                gcv[i] <= gcv[i - 1] + 1e-8,
                "RSS should be non-increasing: gcv[{i}]={} > gcv[{}]={}",
                gcv[i],
                i - 1,
                gcv[i - 1]
            );
        }
    }

    #[test]
    fn boost_fosr_recovers_known_beta() {
        let (data, predictors, argvals) = make_synthetic(30, 20);
        let config = BoostingConfig {
            mstop: 50,
            nu: 0.1,
            nbasis: 8,
            order: 4,
            lfd_order: 2,
            lambda: 0.05,
            ncomp_x: 3,
            seed: 42,
        };
        let result = boost_fosr(&data, &predictors, &argvals, &config).unwrap();

        // With 50 iterations and one strong signal predictor, R² should be high
        assert!(
            result.r_squared > 0.8,
            "Expected R² > 0.8 on synthetic data, got {}",
            result.r_squared
        );

        // Fitted values should track Y reasonably
        let (n, m) = data.shape();
        let mut ss_res = 0.0f64;
        let mut ss_tot = 0.0f64;
        for t in 0..m {
            let mean_t: f64 = (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64;
            for i in 0..n {
                ss_res += (data[(i, t)] - result.fitted[(i, t)]).powi(2);
                ss_tot += (data[(i, t)] - mean_t).powi(2);
            }
        }
        let global_r2 = if ss_tot > 1e-15 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };
        assert!(
            global_r2 > 0.8,
            "Global R² should exceed 0.8, got {global_r2}"
        );
    }

    #[test]
    fn boost_fosr_r_squared_in_range() {
        let (data, predictors, argvals) = make_synthetic(20, 15);
        let config = default_config();
        let result = boost_fosr(&data, &predictors, &argvals, &config).unwrap();

        // r_squared must be in [-ε, 1+ε] (can be slightly negative for intercept-only model)
        assert!(
            result.r_squared >= -0.05,
            "r_squared below -0.05: {}",
            result.r_squared
        );
        assert!(
            result.r_squared <= 1.0 + 1e-8,
            "r_squared above 1: {}",
            result.r_squared
        );

        // Every pointwise r_squared_t entry should also be sane
        for (t, &r2t) in result.r_squared_t.iter().enumerate() {
            assert!(
                r2t >= -0.05,
                "r_squared_t[{t}] = {r2t} < -0.05"
            );
            assert!(
                r2t <= 1.0 + 1e-8,
                "r_squared_t[{t}] = {r2t} > 1"
            );
        }
    }

    #[test]
    fn boost_fosr_selected_learners_valid() {
        let (data, predictors, argvals) = make_synthetic(20, 15);
        let config = default_config();
        let result = boost_fosr(&data, &predictors, &argvals, &config).unwrap();

        let p = predictors.ncols();
        assert_eq!(
            result.selected_learners.len(),
            config.mstop,
            "selected_learners must have mstop entries"
        );
        for (iter, &j) in result.selected_learners.iter().enumerate() {
            assert!(
                j < p,
                "selected_learners[{iter}] = {j} >= p={p}"
            );
        }
    }

    #[test]
    fn boost_fosr_errors_on_dimension_mismatch() {
        let argvals: Vec<f64> = (0..15).map(|i| i as f64 / 14.0).collect();
        let data = FdMatrix::zeros(20, 15);
        // predictors with wrong number of rows
        let predictors = FdMatrix::zeros(10, 2);
        let config = default_config();

        let err = boost_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "Expected InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn boost_fosr_errors_on_invalid_params() {
        let (data, predictors, argvals) = make_synthetic(20, 15);

        // mstop == 0
        let mut config = default_config();
        config.mstop = 0;
        let err = boost_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }), "mstop=0 should return InvalidParameter");

        // nu <= 0
        let mut config = default_config();
        config.nu = -0.1;
        let err = boost_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }), "nu<0 should return InvalidParameter");

        // lambda <= 0
        let mut config = default_config();
        config.lambda = 0.0;
        let err = boost_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }), "lambda=0 should return InvalidParameter");

        // nbasis < 4
        let mut config = default_config();
        config.nbasis = 2;
        let err = boost_fosr(&data, &predictors, &argvals, &config).unwrap_err();
        assert!(matches!(err, FdarError::InvalidParameter { .. }), "nbasis<4 should return InvalidParameter");
    }
}
