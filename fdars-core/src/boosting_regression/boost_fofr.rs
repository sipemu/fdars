//! Component-wise gradient boosting for function-on-function regression (REG-06-02).
//!
//! Implements the **bfpc** (FPC-compression) variant of boosted FoFR: each functional
//! predictor is compressed via `fdata_to_pc_1d` (truncated KL expansion), and the
//! resulting FPC-score matrices serve as the design matrices for the boosting base-learners.
//!
//! # Model
//!
//! ```text
//! Y_i(t) = F_0(t) + Σ_j ∫ X_{j,i}(s) β_j(s,t) ds + ε_i(t)
//! ```
//!
//! For the bfpc variant, the integral is approximated as:
//!
//! ```text
//! ∫ X_{j,i}(s) β_j(s,t) ds ≈ Σ_k ξ_{ijk} γ_{jk}(t)
//! ```
//!
//! where `ξ_{ijk}` is the k-th FPC score of curve `X_{j,i}` and `γ_{jk}(t)` is a
//! smooth coefficient function. The design matrix for base-learner `j` is thus the
//! `n × K_j` FPC-score matrix `S_j = fpca_j.scores`.
//!
//! # Algorithm
//!
//! **Preprocessing:** Compute `FpcaResult_j = fdata_to_pc_1d(X_j, ncomp_x, argvals_j)` for
//! each predictor `j`. Score matrices `S_j ∈ R^{n × K_j}` serve as base-learner designs.
//!
//! **Initialization:** `F̂_0(t) = Ȳ(t)` (pointwise column mean of Y, length m_y).
//!
//! **For** m = 1, …, mstop:
//! 1. Residual `U = Y − F̂` (n × m_y).
//! 2. For each base-learner j: solve `(S_j'S_j + ε·I) c_j(t) = S_j'·U[:,t]` (Cholesky,
//!    factored once per learner outside the time-point loop), giving `Ĥ_j = S_j · c_j`.
//! 3. Select `j* = argmin_j ‖U − Ĥ_j‖_F²`.
//! 4. Update `F̂ += ν·Ĥ_{j*}`; accumulate `score_coefs[j*] += ν·c_{j*}`.
//!
//! **Reconstruction:** For predictor j, `β_j(s,t) = rotation_j (m_x × K_j) · score_coefs_j
//! (K_j × m_y)` gives the `(m_x × m_y)` coefficient surface.
//!
//! # Divergences from FDboost
//!
//! FDboost's `bsignal` base-learner uses trapezoidal-rule integration over a B-spline basis
//! for `β(s,t)` jointly, creating a full bivariate coefficient surface from a B-spline tensor
//! product. The bfpc variant used here compresses via truncated KL expansion (`fdata_to_pc_1d`),
//! which is simpler to implement without new dependencies and is an accepted equivalent
//! for smooth functional predictors. The reconstruction step `rotation · score_coefs` recovers
//! the coefficient surface in the original functional data space.
//!
//! # References
//!
//! FDboost CRAN documentation (rdrr.io/cran/FDboost), `bfpc` / `bsignal` base-learners.
//! Hothorn et al. (2010). Model-Based Boosting. *Journal of Statistical Software*.

use super::{BoostFofrResult, BoostingConfig};
use crate::error::FdarError;
use crate::linalg::{cholesky_factor, cholesky_forward_back};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

// ---------------------------------------------------------------------------
// Internal helper: S'u (K-vector) from column-major S (n × K) and u (&[f64] of length n)
// ---------------------------------------------------------------------------
fn st_times_vec(scores: &FdMatrix, u_col: &[f64]) -> Vec<f64> {
    let (_n, k) = scores.shape();
    (0..k)
        .map(|kk| {
            // scores.column(kk) is contiguous (column-major): element (i,kk) at i + kk*n
            let s_col = scores.column(kk);
            s_col
                .iter()
                .zip(u_col.iter())
                .map(|(&s, &u)| s * u)
                .sum::<f64>()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Pointwise R² helper (mirrors boost_fosr.rs / function_on_scalar.rs)
// ---------------------------------------------------------------------------
fn pointwise_r_squared(data: &FdMatrix, fitted: &FdMatrix) -> Vec<f64> {
    let (n, m) = data.shape();
    (0..m)
        .map(|t| {
            let mean_t: f64 = (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64;
            let ss_tot: f64 = (0..n).map(|i| (data[(i, t)] - mean_t).powi(2)).sum();
            let ss_res: f64 = (0..n)
                .map(|i| (data[(i, t)] - fitted[(i, t)]).powi(2))
                .sum();
            if ss_tot > 1e-15 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Internal: pre-factored score-space base-learner
// ---------------------------------------------------------------------------

/// Score-space base-learner for one functional predictor.
///
/// Caches the Cholesky factor L of (S'S + ε·I) (K × K row-major) so the
/// factorization is amortised across all mstop iterations. Only the
/// back-solve `cholesky_forward_back(L, S'U[:,t], K)` runs per iteration
/// per time point — O(K²) per time point instead of O(K³).
struct ScoreLearner {
    /// FPC score matrix S (n × K), stored as FdMatrix (column-major).
    scores: FdMatrix,
    /// Cholesky factor L of (S'S + ε·I) (K × K row-major flat).
    l: Vec<f64>,
    /// Number of FPC components K.
    k: usize,
}

// ---------------------------------------------------------------------------
// Public: boost_fofr
// ---------------------------------------------------------------------------

/// Component-wise gradient boosting for function-on-function regression (bfpc variant).
///
/// Fits a boosted model `Ŷᵢ(t) = F₀(t) + Σⱼ Sⱼ · γ̂ⱼ(t)` where each base-learner j
/// operates on the FPC score matrix `Sⱼ ∈ R^{n × K_j}` of functional predictor j.
/// At each iteration, the single best base-learner (minimum residual SS) is selected
/// and the current estimate updated by `ν` times its fitted values.
///
/// The coefficient surface `β̂_j(s,t) = rotation_j (m_x × K_j) · score_coefs_j (K_j × m_y)`
/// is reconstructed after the boosting loop for each predictor that was selected at
/// least once. Predictors never selected receive a zero coefficient surface.
///
/// # Arguments
///
/// * `x_data` — Slice of functional predictor matrices (one per predictor, each n × m_x).
/// * `x_argvals` — Evaluation grids for each predictor (one per predictor, length m_x).
/// * `y_data` — Functional response matrix (n × m_y, column-major `FdMatrix`).
/// * `y_argvals` — Response grid evaluation points (length m_y).
/// * `config` — [`BoostingConfig`] controlling boosting iterations, learning rate ν,
///   and FPC compression (`ncomp_x` components per predictor).
///
/// # Returns
///
/// [`BoostFofrResult`] with fitted values `(n × m_y)`, residuals, pointwise R²,
/// accumulated FPC-space score coefficient matrices per predictor, reconstructed
/// coefficient surfaces β_j(s,t) per predictor, and path diagnostics.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `x_data` is empty.
/// - Any predictor or the response has a different number of rows (n mismatch).
/// - `x_argvals[j].len()` does not equal predictor j's column count.
/// - `y_argvals.len()` does not equal `y_data.ncols()`.
/// - `n < 3`.
///
/// Returns [`FdarError::InvalidParameter`] if:
/// - `mstop == 0`
/// - `nu <= 0` or `nu > 1`
/// - `ncomp_x == 0`
///
/// Returns [`FdarError::ComputationFailed`] if FPCA or a base-learner Cholesky
/// factorization fails (ill-conditioned scores — try reducing `ncomp_x`).
///
/// # Divergences from FDboost
///
/// Uses FPC score compression (`fdata_to_pc_1d`) rather than FDboost's `bsignal`
/// B-spline joint expansion for β(s,t). The bfpc truncated-KL variant is simpler
/// to implement without new dependencies and equivalent for smooth functional predictors.
/// Fixed `nu` and `mstop` (no line search or CV-based early stopping).
///
/// # Examples
///
/// ```rust
/// use fdars_core::boosting_regression::{boost_fofr, BoostingConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let n = 20;
/// let m_x = 15;
/// let m_y = 12;
/// let x_argvals: Vec<f64> = (0..m_x).map(|i| i as f64 / (m_x - 1) as f64).collect();
/// let y_argvals: Vec<f64> = (0..m_y).map(|i| i as f64 / (m_y - 1) as f64).collect();
/// let x = FdMatrix::zeros(n, m_x);
/// let y = FdMatrix::zeros(n, m_y);
/// let config = BoostingConfig {
///     mstop: 10, nu: 0.1, nbasis: 8, order: 4, lfd_order: 2, lambda: 1.0,
///     ncomp_x: 3, seed: 42,
/// };
/// // let result = boost_fofr(&[&x], &[x_argvals.as_slice()], &y, &y_argvals, &config)?;
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fofr(
    x_data: &[&FdMatrix],
    x_argvals: &[&[f64]],
    y_data: &FdMatrix,
    y_argvals: &[f64],
    config: &BoostingConfig,
) -> Result<BoostFofrResult, FdarError> {
    // ---- Input validation ---------------------------------------------------
    let p = x_data.len();
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "x_data",
            expected: "at least 1 functional predictor".to_string(),
            actual: "0 predictors".to_string(),
        });
    }
    if x_argvals.len() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "x_argvals",
            expected: format!("length == x_data.len() = {p}"),
            actual: format!("length = {}", x_argvals.len()),
        });
    }

    let (n, m_y) = y_data.shape();

    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_data",
            expected: "at least 3 observations (n >= 3)".to_string(),
            actual: format!("n = {n}"),
        });
    }
    if m_y == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "y_data",
            expected: "at least 1 response grid point (m_y > 0)".to_string(),
            actual: "m_y = 0".to_string(),
        });
    }
    if y_argvals.len() != m_y {
        return Err(FdarError::InvalidDimension {
            parameter: "y_argvals",
            expected: format!("length == y_data.ncols() = {m_y}"),
            actual: format!("length = {}", y_argvals.len()),
        });
    }

    // Check each predictor's dimensions
    for j in 0..p {
        if x_data[j].nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "x_data[j].nrows()",
                expected: format!("== n = {n} (matching y_data)"),
                actual: format!("x_data[{j}].nrows() = {}", x_data[j].nrows()),
            });
        }
        if x_argvals[j].len() != x_data[j].ncols() {
            return Err(FdarError::InvalidDimension {
                parameter: "x_argvals[j]",
                expected: format!("length == x_data[{j}].ncols() = {}", x_data[j].ncols()),
                actual: format!("length = {}", x_argvals[j].len()),
            });
        }
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
    if config.ncomp_x == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp_x",
            message: "must be >= 1".to_string(),
        });
    }

    let nu = config.nu;

    // ---- Preprocessing: compute FPC scores for each functional predictor ----
    // For each predictor j, fdata_to_pc_1d compresses X_j (n × m_x) to
    // FPC scores S_j (n × K_j) where K_j = min(ncomp_x, n-1, m_x).
    let mut fpca_x: Vec<FpcaResult> = Vec::with_capacity(p);
    for j in 0..p {
        let ncomp = config.ncomp_x.min(n - 1).min(x_data[j].ncols());
        let fpca_j = fdata_to_pc_1d(x_data[j], ncomp, x_argvals[j]).map_err(|e| {
            FdarError::ComputationFailed {
                operation: "boost_fofr FPCA preprocessing",
                detail: format!("predictor j={j}: {e:?}"),
            }
        })?;
        fpca_x.push(fpca_j);
    }

    // ---- Build pre-factored score-space base-learners ----------------------
    // For each predictor j:
    //   S_j = fpca_x[j].scores (n × K_j)
    //   A_j = S_j'S_j + ε·I (K_j × K_j, row-major)
    //   L_j = cholesky_factor(A_j, K_j)
    //
    // The factorization is cached; only back-solve runs per iteration per t.
    let mut learners: Vec<ScoreLearner> = Vec::with_capacity(p);
    for j in 0..p {
        let scores = fpca_x[j].scores.clone(); // n × K_j
        let (n_s, k) = scores.shape();
        debug_assert_eq!(n_s, n, "scores.nrows() must equal n");

        // Build S'S (K × K row-major)
        let mut sts = vec![0.0f64; k * k];
        for row in 0..k {
            for col in row..k {
                let mut s = 0.0;
                // scores is column-major: element (i, col) at i + col*n
                let s_col_r = scores.column(row);
                let s_col_c = scores.column(col);
                for i in 0..n {
                    s += s_col_r[i] * s_col_c[i];
                }
                sts[row * k + col] = s;
                sts[col * k + row] = s;
            }
        }

        // Ridge jitter ε·I for numerical stability (T-43-02a)
        for i in 0..k {
            sts[i * k + i] += 1e-10;
        }

        let l = cholesky_factor(&sts, k).map_err(|e| FdarError::ComputationFailed {
            operation: "boost_fofr score-space Cholesky",
            detail: format!("predictor j={j}: {e:?} — try reducing ncomp_x"),
        })?;

        learners.push(ScoreLearner { scores, l, k });
    }

    // ---- Initialization: F₀(t) = Ȳ(t) ------------------------------------
    let intercept: Vec<f64> = (0..m_y)
        .map(|t| (0..n).map(|i| y_data[(i, t)]).sum::<f64>() / n as f64)
        .collect();

    // Current estimate F (n × m_y)
    let mut f_current = FdMatrix::zeros(n, m_y);
    for t in 0..m_y {
        let f_col = f_current.column_mut(t);
        for i in 0..n {
            f_col[i] = intercept[t];
        }
    }

    // Accumulated score coefficients per predictor: score_coefs[j] is K_j × m_y
    let mut score_coefs: Vec<FdMatrix> = learners
        .iter()
        .map(|lrn| FdMatrix::zeros(lrn.k, m_y))
        .collect();

    // ---- Boosting loop ------------------------------------------------------
    let mut selected_learners: Vec<usize> = Vec::with_capacity(config.mstop);
    let mut gcv_path: Vec<f64> = Vec::with_capacity(config.mstop);

    // Scratch buffers: reused each iteration to avoid repeated allocation
    // (fitted_j, coefs_j allocated once per iteration per learner below)

    for _iter in 0..config.mstop {
        // Compute residual U = Y − F̂ (n × m_y)
        let mut residuals = FdMatrix::zeros(n, m_y);
        for t in 0..m_y {
            let y_col = y_data.column(t);
            let f_col = f_current.column(t);
            let r_col = residuals.column_mut(t);
            for i in 0..n {
                r_col[i] = y_col[i] - f_col[i];
            }
        }

        // Track ‖U‖_F² before update (GCV proxy)
        let rss_before: f64 = (0..m_y)
            .flat_map(|t| residuals.column(t).iter().map(|&r| r * r))
            .sum();
        gcv_path.push(rss_before);

        // Find best base-learner
        let mut best_rss = f64::INFINITY;
        let mut best_j = 0usize;
        let mut best_fitted: Option<FdMatrix> = None;
        let mut best_coefs: Option<FdMatrix> = None;

        for (j, learner) in learners.iter().enumerate() {
            let k = learner.k;
            let scores = &learner.scores; // n × K (column-major FdMatrix)

            let mut fitted_j = FdMatrix::zeros(n, m_y);
            let mut coefs_j = FdMatrix::zeros(k, m_y);

            // Solve per response time point t
            for t in 0..m_y {
                let u_col = residuals.column(t); // contiguous, length n
                let rhs = st_times_vec(scores, u_col); // S'·u[:,t], K-vector
                let c_t = cholesky_forward_back(&learner.l, &rhs, k); // K-vector

                // Fitted: Ĥ_j[:,t] = S_j · c_t
                // S_j is column-major: element (i, kk) at scores[(i, kk)]
                let f_col_j = fitted_j.column_mut(t);
                for i in 0..n {
                    let mut val = 0.0;
                    for kk in 0..k {
                        val += scores[(i, kk)] * c_t[kk];
                    }
                    f_col_j[i] = val;
                }

                // Store coefficients
                for kk in 0..k {
                    coefs_j[(kk, t)] = c_t[kk];
                }
            }

            // RSS of (U − Ĥ_j)
            let rss: f64 = (0..m_y)
                .flat_map(|t| {
                    let u_col = residuals.column(t);
                    let h_col = fitted_j.column(t);
                    (0..n).map(move |i| {
                        let d = u_col[i] - h_col[i];
                        d * d
                    })
                })
                .sum();

            if rss < best_rss {
                best_rss = rss;
                best_j = j;
                best_fitted = Some(fitted_j);
                best_coefs = Some(coefs_j);
            }
        }

        // best_fitted / best_coefs are always Some (we validated p >= 1 above)
        let fitted_star = best_fitted.expect("at least one learner must exist");
        let coefs_star = best_coefs.expect("at least one learner must exist");

        selected_learners.push(best_j);

        // Update F̂ += ν·Ĥ_{j*}
        for t in 0..m_y {
            let f_col = f_current.column_mut(t);
            let h_col = fitted_star.column(t);
            for i in 0..n {
                f_col[i] += nu * h_col[i];
            }
        }

        // Accumulate score coefficients for j*: score_coefs[j*] (K_{j*} × m_y) += ν·c_{j*}
        for t in 0..m_y {
            let k = learners[best_j].k;
            for kk in 0..k {
                score_coefs[best_j][(kk, t)] += nu * coefs_star[(kk, t)];
            }
        }
    }

    // ---- Coefficient-surface reconstruction --------------------------------
    // For each predictor j:
    //   β_j(s, t) = rotation_j (m_x × K_j)  ·  score_coefs_j (K_j × m_y)
    //             = (m_x × m_y) matrix
    //
    // rotation_j is stored as FdMatrix (m_x × K_j, column-major):
    //   element (s, k) at rotation[(s, k)]
    // score_coefs_j is (K_j × m_y, column-major):
    //   element (k, t) at score_coefs[j][(k, t)]
    let beta_surfaces: Vec<FdMatrix> = fpca_x
        .iter()
        .zip(score_coefs.iter())
        .map(|(fpca_j, coef_j)| {
            let m_x = fpca_j.rotation.nrows(); // m_x
            let k = fpca_j.rotation.ncols(); // K_j
            let mut beta = FdMatrix::zeros(m_x, m_y);
            for s in 0..m_x {
                for t in 0..m_y {
                    let mut val = 0.0;
                    for kk in 0..k {
                        val += fpca_j.rotation[(s, kk)] * coef_j[(kk, t)];
                    }
                    beta[(s, t)] = val;
                }
            }
            beta
        })
        .collect();

    // ---- Final residuals and R² -------------------------------------------
    let mut final_residuals = FdMatrix::zeros(n, m_y);
    for t in 0..m_y {
        let y_col = y_data.column(t);
        let f_col = f_current.column(t);
        let r_col = final_residuals.column_mut(t);
        for i in 0..n {
            r_col[i] = y_col[i] - f_col[i];
        }
    }

    let r_squared_t = pointwise_r_squared(y_data, &f_current);
    let r_squared = r_squared_t.iter().sum::<f64>() / m_y as f64;

    Ok(BoostFofrResult {
        intercept,
        fitted: f_current,
        residuals: final_residuals,
        r_squared_t,
        r_squared,
        fpca_x,
        score_coefs,
        beta_surfaces,
        selected_learners,
        gcv_path,
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
    use crate::test_helpers::uniform_grid;

    /// Build a synthetic function-on-function dataset.
    ///
    /// Response: `Y_i(t) = Σ_k ξ_{ik} · cos(k·π·t)` where ξ_{ik} are the FPC
    /// scores of the functional predictor X_i(s).
    /// Predictor: `X_i(s) = a_i · sin(π·s) + b_i · sin(2π·s)`
    ///
    /// Returns (x_predictors, y_data, x_argvals, y_argvals, noise_predictor).
    fn make_fofr_data(
        n: usize,
        m_x: usize,
        m_y: usize,
    ) -> (FdMatrix, FdMatrix, Vec<f64>, Vec<f64>, FdMatrix) {
        let tx = uniform_grid(m_x);
        let ty = uniform_grid(m_y);

        // Deterministic amplitude coefficients for each observation
        let as_: Vec<f64> = (0..n)
            .map(|i| (i as f64) / (n - 1) as f64 * 2.0 - 1.0)
            .collect();
        let bs: Vec<f64> = (0..n)
            .map(|i| ((i + 1) as f64 * 1.618) % 1.0 * 2.0 - 1.0)
            .collect();

        // Build X (n × m_x): X_i(s) = a_i*sin(π·s) + b_i*sin(2π·s)
        let mut x_data = vec![0.0f64; n * m_x];
        for (t, &sv) in tx.iter().enumerate() {
            for i in 0..n {
                x_data[i + t * n] = as_[i] * (std::f64::consts::PI * sv).sin()
                    + bs[i] * (2.0 * std::f64::consts::PI * sv).sin();
            }
        }
        let x = FdMatrix::from_column_major(x_data, n, m_x).unwrap();

        // Build Y (n × m_y): Y_i(t) = a_i*cos(π·t) + b_i*cos(2π·t) + small noise
        let mut y_data = vec![0.0f64; n * m_y];
        for (t, &tv) in ty.iter().enumerate() {
            for i in 0..n {
                let noise = 0.05 * ((i + t + 7) as f64 * std::f64::consts::E).sin();
                y_data[i + t * n] = as_[i] * (std::f64::consts::PI * tv).cos()
                    + bs[i] * (2.0 * std::f64::consts::PI * tv).cos()
                    + noise;
            }
        }
        let y = FdMatrix::from_column_major(y_data, n, m_y).unwrap();

        // A noise predictor (constant across observations → no signal)
        let noise_x_data: Vec<f64> = (0..n * m_x)
            .map(|k| {
                let i = k % n;
                let t = k / n;
                0.02 * ((i + t * 3) as f64 * 1.234).sin()
            })
            .collect();
        let noise_x = FdMatrix::from_column_major(noise_x_data, n, m_x).unwrap();

        (x, y, tx, ty, noise_x)
    }

    fn default_config() -> BoostingConfig {
        BoostingConfig {
            mstop: 30,
            nu: 0.1,
            nbasis: 8,
            order: 4,
            lfd_order: 2,
            lambda: 1.0,
            ncomp_x: 3,
            seed: 42,
        }
    }

    #[test]
    fn boost_fofr_fitted_shape() {
        let (x, y, tx, ty, noise_x) = make_fofr_data(25, 20, 18);
        let config = default_config();
        let result = boost_fofr(
            &[&x, &noise_x],
            &[tx.as_slice(), tx.as_slice()],
            &y,
            &ty,
            &config,
        )
        .unwrap();

        assert_eq!(result.fitted.shape(), (25, 18), "fitted must be (n, m_y)");
        assert_eq!(
            result.residuals.shape(),
            (25, 18),
            "residuals must be (n, m_y)"
        );
        assert_eq!(
            result.r_squared_t.len(),
            18,
            "r_squared_t must have m_y entries"
        );
        assert_eq!(
            result.intercept.len(),
            18,
            "intercept must have m_y entries"
        );
        assert_eq!(result.mstop, config.mstop);
        assert!((result.nu - config.nu).abs() < 1e-15);
        assert_eq!(result.selected_learners.len(), config.mstop);
        assert_eq!(result.gcv_path.len(), config.mstop);
    }

    #[test]
    fn boost_fofr_residuals_decrease() {
        // gcv_path[i] = ‖U‖_F² before iteration i+1.
        // With a signal-bearing predictor, the GCV path must be non-increasing.
        let (x, y, tx, ty, _noise_x) = make_fofr_data(25, 20, 18);
        let config = BoostingConfig {
            mstop: 40,
            ..default_config()
        };
        let result = boost_fofr(&[&x], &[tx.as_slice()], &y, &ty, &config).unwrap();

        let gcv = &result.gcv_path;
        assert_eq!(gcv.len(), config.mstop);

        // Allow a tiny tolerance for floating-point rounding
        for i in 1..gcv.len() {
            assert!(
                gcv[i] <= gcv[i - 1] + 1e-6,
                "GCV path should be non-increasing: gcv[{i}]={} > gcv[{}]={}",
                gcv[i],
                i - 1,
                gcv[i - 1]
            );
        }
    }

    #[test]
    fn boost_fofr_r_squared_in_range() {
        let (x, y, tx, ty, _noise_x) = make_fofr_data(25, 20, 18);
        let config = default_config();
        let result = boost_fofr(&[&x], &[tx.as_slice()], &y, &ty, &config).unwrap();

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

        for (t, &r2t) in result.r_squared_t.iter().enumerate() {
            assert!(r2t >= -0.05, "r_squared_t[{t}] = {r2t} < -0.05");
            assert!(r2t <= 1.0 + 1e-8, "r_squared_t[{t}] = {r2t} > 1");
        }
    }

    #[test]
    fn boost_fofr_beta_surface_shape() {
        let (x, y, tx, ty, noise_x) = make_fofr_data(25, 20, 18);
        let config = default_config();
        let result = boost_fofr(
            &[&x, &noise_x],
            &[tx.as_slice(), tx.as_slice()],
            &y,
            &ty,
            &config,
        )
        .unwrap();

        // Two predictors → two beta surfaces, each (m_x, m_y)
        assert_eq!(
            result.beta_surfaces.len(),
            2,
            "one beta surface per predictor"
        );
        assert_eq!(
            result.score_coefs.len(),
            2,
            "one score_coefs entry per predictor"
        );

        for j in 0..2 {
            assert_eq!(
                result.beta_surfaces[j].shape(),
                (20, 18),
                "beta_surfaces[{j}] must be (m_x, m_y)"
            );
        }

        // fpca_x: one entry per predictor
        assert_eq!(result.fpca_x.len(), 2);
    }

    #[test]
    fn boost_fofr_errors_on_dimension_mismatch() {
        let (x, y, tx, ty, _noise_x) = make_fofr_data(25, 20, 18);
        let config = default_config();

        // Predictor with wrong number of rows
        let bad_x = FdMatrix::zeros(10, 20);
        let err = boost_fofr(&[&bad_x], &[tx.as_slice()], &y, &ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "row mismatch must return InvalidDimension, got {err:?}"
        );

        // x_argvals wrong length
        let bad_tx: Vec<f64> = vec![0.0, 0.5, 1.0]; // length 3, not 20
        let err = boost_fofr(&[&x], &[bad_tx.as_slice()], &y, &ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "argvals mismatch must return InvalidDimension, got {err:?}"
        );

        // y_argvals wrong length
        let bad_ty: Vec<f64> = vec![0.0, 1.0]; // length 2, not 18
        let err = boost_fofr(&[&x], &[tx.as_slice()], &y, &bad_ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "y_argvals mismatch must return InvalidDimension, got {err:?}"
        );
    }

    #[test]
    fn boost_fofr_errors_on_invalid_params() {
        let (x, y, tx, ty, _noise_x) = make_fofr_data(25, 20, 18);

        // mstop == 0
        let config = BoostingConfig {
            mstop: 0,
            ..default_config()
        };
        let err = boost_fofr(&[&x], &[tx.as_slice()], &y, &ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "mstop=0 must return InvalidParameter, got {err:?}"
        );

        // ncomp_x == 0
        let config = BoostingConfig {
            ncomp_x: 0,
            ..default_config()
        };
        let err = boost_fofr(&[&x], &[tx.as_slice()], &y, &ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "ncomp_x=0 must return InvalidParameter, got {err:?}"
        );

        // nu > 1
        let config = BoostingConfig {
            nu: 1.5,
            ..default_config()
        };
        let err = boost_fofr(&[&x], &[tx.as_slice()], &y, &ty, &config).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidParameter { .. }),
            "nu>1 must return InvalidParameter, got {err:?}"
        );

        // empty x_data
        let err = boost_fofr(&[], &[], &y, &ty, &default_config()).unwrap_err();
        assert!(
            matches!(err, FdarError::InvalidDimension { .. }),
            "empty predictors must return InvalidDimension, got {err:?}"
        );
    }
}
