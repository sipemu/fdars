//! Functional Generalized Linear Model (GLM) over FPC scores.
//!
//! Implements `functional_glm` — a scalar-on-function GLM that covers the four
//! mainstream exponential-family distributions through a [`GlmFamily`] enum
//! (canonical link + variance function per family).  The IRLS loop runs over
//! Functional Principal Component (FPC) scores produced by [`fdata_to_pc_1d`],
//! reusing the same weighted-normal-equations solver as [`functional_logistic`].
//!
//! # Supported families and canonical links
//!
//! | Family | Link g(μ) | Variance V(μ) |
//! |--------|-----------|--------------|
//! | [`GlmFamily::Binomial`]  | logit     | μ(1−μ)       |
//! | [`GlmFamily::Poisson`]   | log       | μ             |
//! | [`GlmFamily::Gamma`]     | inverse   | μ²            |
//! | [`GlmFamily::Gaussian`]  | identity  | 1             |
//!
//! # IRLS convergence
//!
//! The loop converges when the absolute change in deviance between consecutive
//! iterations is below `tol`, or when `max_iter` is reached.  Gaussian with
//! identity link converges in a single IRLS step (weights ≡ 1 → OLS).
//!
//! # Convention divergences from R `glm()`
//!
//! - **Convergence criterion:** deviance-change `< tol`, not coefficient-change
//!   (more scale-invariant; recommended for multi-family code).
//! - **Canonical links only:** Gamma uses inverse link (g(μ)=1/μ), NOT log-link.
//! - **AIC/BIC:** computed as `−2·log_likelihood + 2p` and `−2·log_likelihood + p·ln(n)`
//!   using the log-likelihood kernel per family. The dispersion φ is **not** folded into
//!   the Gamma/Gaussian AIC/BIC log-likelihood kernel, so Gamma and Gaussian AIC magnitudes
//!   are **not** directly comparable to R's `glm()` / `lm()` output.
//! - **Standard errors:** the dispersion φ (φ = 1 for Binomial/Poisson; Pearson χ²/dof for
//!   Gaussian/Gamma) IS applied to the reported coefficient standard errors:
//!   `Var(β̂) = φ·(XᵀWX)⁻¹`.
//! - **μ/η clamping:** Poisson clamps η ≤ 500 before `exp`; Gamma clamps η ≥ 1e-10 so
//!   μ = 1/η remains finite; all families clamp μ ≥ 1e-10 in weight/deviance computations.
//! - **Gamma intercept initialisation:** β₀ = 1/mean(y) so η₀ > 0 (μ₀ = mean(y)), preventing
//!   a divide-by-zero on the very first IRLS step.
//!
//! [`functional_logistic`]: crate::scalar_on_function::functional_logistic

use super::{
    build_design_matrix, cholesky_factor, cholesky_solve, compute_beta_se, compute_fitted,
    compute_ols_std_errors, recover_beta_t, sigmoid, FunctionalGlmResult, GlmFamily,
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};

// ---------------------------------------------------------------------------
// GlmFamily methods — per-family link / variance / deviance / log-likelihood
// ---------------------------------------------------------------------------

impl GlmFamily {
    /// Inverse link function: η → μ = g⁻¹(η), clamped to a valid range.
    pub(crate) fn inv_link(self, eta: f64) -> f64 {
        match self {
            GlmFamily::Binomial => sigmoid(eta),
            GlmFamily::Poisson => eta.min(500.0_f64).exp().max(1e-10),
            GlmFamily::Gamma => (1.0 / eta.max(1e-10)).max(1e-10),
            GlmFamily::Gaussian => eta,
        }
    }

    /// Link derivative: dη/dμ = g′(μ).
    ///
    /// **Stored separately from [`irls_weight`].**  The working response
    /// `z = η + (y − μ) · g′(μ)` must use this value directly — never derive
    /// the working response from `1/weight`.  For Gamma, g′(μ) = −1/μ² is
    /// **negative**, which is required for IRLS to converge.
    pub(crate) fn link_deriv(self, mu: f64) -> f64 {
        match self {
            GlmFamily::Binomial => 1.0 / (mu * (1.0 - mu)).max(1e-10),
            GlmFamily::Poisson => 1.0 / mu.max(1e-10),
            GlmFamily::Gamma => -1.0 / mu.max(1e-10).powi(2), // NEGATIVE — do not confuse with irls_weight
            GlmFamily::Gaussian => 1.0,
        }
    }

    /// IRLS weight: w_i = (dμ/dη)² / V(μ) = 1 / (V(μ) · g′(μ)²).
    ///
    /// For canonical links this simplifies to:
    /// Binomial = μ(1−μ), Poisson = μ, Gamma = μ², Gaussian = 1.
    ///
    /// **Gamma derivation (inverse link):**
    /// - dμ/dη = −μ² (from μ = 1/η → dμ/dη = −1/η² = −μ²)
    /// - V(μ) = μ²
    /// - w = (dμ/dη)² / V(μ) = μ⁴ / μ² = μ²
    pub(crate) fn irls_weight(self, mu: f64) -> f64 {
        match self {
            GlmFamily::Binomial => (mu * (1.0 - mu)).max(1e-10),
            GlmFamily::Poisson => mu.max(1e-10),
            // w = μ² (NOT 1/μ²) — see derivation in doc comment above
            GlmFamily::Gamma => mu.max(1e-10).powi(2),
            GlmFamily::Gaussian => 1.0,
        }
    }

    /// Total deviance D = 2 Σ d(y_i, μ_i).
    ///
    /// Uses the `0·log(0) = 0` convention (Pitfall 4 in RESEARCH.md) via the
    /// private `xlogy` helper.
    pub(crate) fn deviance(self, y: &[f64], mu: &[f64]) -> f64 {
        fn xlogy(x: f64, y: f64) -> f64 {
            if x == 0.0 {
                0.0
            } else {
                x * y.ln()
            }
        }
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| match self {
                GlmFamily::Binomial => {
                    2.0 * (xlogy(yi, yi / mi.max(1e-15))
                        + xlogy(1.0 - yi, (1.0 - yi) / (1.0 - mi).max(1e-15)))
                }
                GlmFamily::Poisson => 2.0 * (xlogy(yi, yi / mi.max(1e-15)) - (yi - mi)),
                GlmFamily::Gamma => 2.0 * ((yi - mi) / mi.max(1e-15) - (yi / mi.max(1e-15)).ln()),
                GlmFamily::Gaussian => (yi - mi).powi(2),
            })
            .sum()
    }

    /// Log-likelihood kernel sufficient for AIC/BIC (excludes normalising constants).
    ///
    /// Note: for Gamma and Gaussian the dispersion parameter φ is not estimated
    /// separately.  See module-level documentation for the resulting AIC
    /// comparability caveat.
    ///
    /// For the Poisson family, `log(y!) = ln Γ(y+1)` is computed via the private
    /// [`ln_gamma`] Lanczos helper — O(1) and overflow-free (the earlier
    /// `Σ_{k=1}^{y} ln(k)` form was O(y) and, on a saturating `y as u64`, unbounded).
    pub(crate) fn log_likelihood(self, y: &[f64], mu: &[f64]) -> f64 {
        y.iter()
            .zip(mu)
            .map(|(&yi, &mi)| match self {
                GlmFamily::Binomial => {
                    let mi = mi.clamp(1e-15, 1.0 - 1e-15);
                    yi * mi.ln() + (1.0 - yi) * (1.0 - mi).ln()
                }
                GlmFamily::Poisson => {
                    let mi = mi.max(1e-300);
                    // log(y!) = ln Γ(y+1) — O(1), overflow-free (yi is a validated
                    // finite non-negative integer, so yi + 1.0 >= 1.0).
                    let ln_y_fact = ln_gamma(yi + 1.0);
                    yi * mi.ln() - mi - ln_y_fact
                }
                GlmFamily::Gamma => {
                    let mi = mi.max(1e-300);
                    -yi / mi - mi.ln()
                }
                GlmFamily::Gaussian => {
                    // Kernel only: −(y−μ)² (scale by −1/(2σ²) for absolute LL)
                    -(yi - mi).powi(2)
                }
            })
            .sum()
    }
}

// ---------------------------------------------------------------------------
// Response-domain validation
// ---------------------------------------------------------------------------

fn validate_response(y: &[f64], family: GlmFamily) -> Result<(), FdarError> {
    // Reject non-finite responses for ALL families FIRST. IEEE 754 makes
    // `NaN <= 0.0` false and `f64::INFINITY.floor() == f64::INFINITY`, so a
    // non-finite value would otherwise slip past the per-family guards below —
    // producing an all-NaN result (Gamma NaN) or, for Poisson, a `yi as u64`
    // saturation to u64::MAX driving an unbounded log-factorial loop.
    if let Some(&bad) = y.iter().find(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "y",
            message: format!("response contains a non-finite value ({bad})"),
        });
    }
    match family {
        GlmFamily::Binomial => {
            if y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be 0.0 or 1.0 for Binomial family".to_string(),
                });
            }
        }
        GlmFamily::Poisson => {
            if y.iter().any(|&yi| yi < 0.0 || yi != yi.floor()) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be non-negative integers for Poisson family"
                        .to_string(),
                });
            }
        }
        GlmFamily::Gamma => {
            if y.iter().any(|&yi| yi <= 0.0) {
                return Err(FdarError::InvalidParameter {
                    parameter: "y",
                    message: "all values must be strictly positive for Gamma family".to_string(),
                });
            }
        }
        GlmFamily::Gaussian => {} // unrestricted
    }
    Ok(())
}

/// Natural log of the Gamma function via the Lanczos approximation (g = 7, n = 9).
///
/// Used for the Poisson `log(y!) = ln Γ(y+1)` term. O(1) and overflow-free — it
/// replaces an O(y) running `Σ ln(k)` sum, and adds no crate dependency (statrs
/// is not vendored). Accurate to ~15 significant digits for the arguments used
/// here (`x = y + 1 >= 1`); the reflection branch covers `x < 0.5` for completeness.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // Reflection: ln Γ(x) = ln(π / sin(πx)) − ln Γ(1 − x)
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().abs().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

// ---------------------------------------------------------------------------
// Generic IRLS step
// ---------------------------------------------------------------------------

/// One IRLS step: compute working response and solve weighted normal equations.
/// Returns updated beta or `None` if the system is singular.
fn irls_step_glm(
    design: &FdMatrix,
    y: &[f64],
    beta: &[f64],
    family: GlmFamily,
) -> Option<Vec<f64>> {
    let (n, p) = design.shape();

    // Linear predictor η = Xβ
    let eta: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| design[(i, j)] * beta[j]).sum())
        .collect();

    // μ = g⁻¹(η), w = IRLS weight, z = working response
    let mu: Vec<f64> = eta.iter().map(|&e| family.inv_link(e)).collect();
    let w: Vec<f64> = mu.iter().map(|&m| family.irls_weight(m)).collect();
    // z_i = η_i + (y_i − μ_i) · g′(μ_i)   [MUST use link_deriv, NOT 1/weight]
    let z: Vec<f64> = (0..n)
        .map(|i| eta[i] + (y[i] - mu[i]) * family.link_deriv(mu[i]))
        .collect();

    // Weighted normal equations: (X′WX)β = X′Wz
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let s: f64 = (0..n).map(|i| design[(i, k)] * w[i] * design[(i, j)]).sum();
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    let xtwz: Vec<f64> = (0..p)
        .map(|k| (0..n).map(|i| design[(i, k)] * w[i] * z[i]).sum())
        .collect();

    cholesky_solve(&xtwx, &xtwz, p).ok()
}

// ---------------------------------------------------------------------------
// IRLS loop
// ---------------------------------------------------------------------------

/// Run IRLS until deviance-change < tol or max_iter is reached.
/// Returns (beta, iterations).
fn irls_loop_glm(
    design: &FdMatrix,
    y: &[f64],
    family: GlmFamily,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, usize) {
    let p_total = design.ncols();
    let mut beta = init_beta(p_total, y, family);
    let mut iterations = 0;

    // Initial deviance
    let mu_init: Vec<f64> = {
        let (n, p) = design.shape();
        (0..n)
            .map(|i| {
                let eta: f64 = (0..p).map(|j| design[(i, j)] * beta[j]).sum();
                family.inv_link(eta)
            })
            .collect()
    };
    let mut dev_old = family.deviance(y, &mu_init);

    for iter in 0..max_iter {
        iterations = iter + 1;
        let Some(beta_new) = irls_step_glm(design, y, &beta, family) else {
            break;
        };
        // Compute new deviance
        let (n, p) = design.shape();
        let mu_new: Vec<f64> = (0..n)
            .map(|i| {
                let eta: f64 = (0..p).map(|j| design[(i, j)] * beta_new[j]).sum();
                family.inv_link(eta)
            })
            .collect();
        let dev_new = family.deviance(y, &mu_new);
        beta = beta_new;
        if (dev_new - dev_old).abs() < tol {
            break;
        }
        dev_old = dev_new;
    }
    (beta, iterations)
}

/// Initialise β for the IRLS loop.
///
/// Zero-initialisation works for Binomial (η=0 → μ=0.5), Poisson (η=0 → μ=1),
/// and Gaussian.  For Gamma, zero β → η=0 → μ=1/0 = ∞ on the first step, so
/// the intercept is initialised to 1/mean(y) so that η₀ > 0 and μ₀ = mean(y).
fn init_beta(p: usize, y: &[f64], family: GlmFamily) -> Vec<f64> {
    let mut beta = vec![0.0_f64; p];
    if let GlmFamily::Gamma = family {
        let mean_y = y.iter().sum::<f64>() / y.len() as f64;
        beta[0] = 1.0 / mean_y.max(1e-10);
    }
    beta
}

// ---------------------------------------------------------------------------
// Result assembly
// ---------------------------------------------------------------------------

fn build_glm_result(
    design: &FdMatrix,
    beta: Vec<f64>,
    y: &[f64],
    fpca: FpcaResult,
    ncomp: usize,
    m: usize,
    iterations: usize,
    family: GlmFamily,
) -> FunctionalGlmResult {
    let (n, p) = design.shape();
    let linear_predictors = compute_fitted(design, &beta);
    let fitted_values: Vec<f64> = linear_predictors
        .iter()
        .map(|&e| family.inv_link(e))
        .collect();

    let beta_t = recover_beta_t(&beta[1..=ncomp], &fpca.rotation, m);
    let gamma: Vec<f64> = beta[1 + ncomp..].to_vec();

    // SE from Fisher information matrix (X′WX)⁻¹ evaluated at converged β
    let w_final: Vec<f64> = fitted_values
        .iter()
        .map(|&mu| family.irls_weight(mu))
        .collect();
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let s: f64 = (0..n)
                .map(|i| design[(i, k)] * w_final[i] * design[(i, j)])
                .sum();
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    // Dispersion φ scales the coefficient covariance: Var(β̂) = φ·(XᵀWX)⁻¹.
    // φ = 1 for Binomial/Poisson (fixed by the family); for Gaussian/Gamma it is
    // estimated by the Pearson χ² statistic over residual dof, so the reported
    // standard errors are not systematically too small.
    let dispersion = match family {
        GlmFamily::Binomial | GlmFamily::Poisson => 1.0,
        GlmFamily::Gaussian => {
            let dof = n.saturating_sub(p).max(1) as f64;
            let rss: f64 = y
                .iter()
                .zip(&fitted_values)
                .map(|(&yi, &mi)| (yi - mi).powi(2))
                .sum();
            rss / dof
        }
        GlmFamily::Gamma => {
            let dof = n.saturating_sub(p).max(1) as f64;
            // Pearson χ² with V(μ) = μ²: Σ ((yᵢ − μᵢ)/μᵢ)²
            let chi2: f64 = y
                .iter()
                .zip(&fitted_values)
                .map(|(&yi, &mi)| ((yi - mi) / mi.max(1e-10)).powi(2))
                .sum();
            chi2 / dof
        }
    };
    let std_errors = cholesky_factor(&xtwx, p).map_or_else(
        |_| vec![f64::NAN; p],
        |l| compute_ols_std_errors(&l, p, dispersion),
    );
    let beta_se = compute_beta_se(&std_errors[1..=ncomp], &fpca.rotation, m);

    let ll = family.log_likelihood(y, &fitted_values);
    let deviance = family.deviance(y, &fitted_values);
    let nf = n as f64;
    let pf = p as f64;
    let aic = -2.0 * ll + 2.0 * pf;
    let bic = -2.0 * ll + nf.ln() * pf;

    FunctionalGlmResult {
        intercept: beta[0],
        beta_t,
        beta_se,
        gamma,
        fitted_values,
        linear_predictors,
        ncomp,
        coefficients: beta,
        std_errors,
        log_likelihood: ll,
        deviance,
        iterations,
        fpca,
        aic,
        bic,
        family,
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fit a functional GLM for a scalar response over a functional predictor.
///
/// Models: g(E[Y | X]) = α + ∫β(t)X(t)dt + γᵀz
///
/// via IRLS (iteratively reweighted least squares) on FPC scores, where g is
/// the canonical link function for the chosen [`GlmFamily`].
///
/// # Arguments
///
/// * `data` — Functional predictor matrix (n × m, column-major `FdMatrix`)
/// * `y` — Scalar response vector (length n); must satisfy the family's domain
///   constraint (Binomial: {0,1}; Poisson: non-negative integers; Gamma: > 0)
/// * `family` — Exponential-family distribution; determines link and variance
/// * `scalar_covariates` — Optional scalar covariate matrix (n × p)
/// * `ncomp` — Number of FPC components (clamped to min(n−1, m))
/// * `max_iter` — Maximum IRLS iterations (pass 0 for default of 25)
/// * `tol` — Deviance-change convergence tolerance (pass ≤ 0.0 for default 1e-6)
///
/// # Returns
///
/// A [`FunctionalGlmResult`] containing: intercept, functional coefficient
/// β(t), FPC score coefficients γ, fitted values μ = g⁻¹(η), linear
/// predictors η, deviance, log-likelihood, AIC, BIC, standard errors, and
/// the embedded [`FpcaResult`] for projecting new data.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` has fewer than 3 rows or zero columns
/// - `y.len() != n`
/// - `scalar_covariates` is provided but its row count differs from `n`
///
/// Returns [`FdarError::InvalidParameter`] if any response value violates the
/// family's domain constraint (Binomial y ∉ {0,1}; Poisson y < 0 or non-integer;
/// Gamma y ≤ 0).
///
/// Returns [`FdarError::ComputationFailed`] if the SVD inside FPCA fails.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::scalar_on_function::{functional_glm, GlmFamily};
///
/// let data = FdMatrix::from_column_major(
///     (0..600).map(|i| (i as f64 * 0.05).sin()).collect(),
///     20, 30,
/// ).unwrap();
/// let y: Vec<f64> = (0..20).map(|i| (i as f64) * 0.4).collect();
/// let fit = functional_glm(&data, &y, GlmFamily::Gaussian, None, 3, 25, 1e-6).unwrap();
/// assert_eq!(fit.fitted_values.len(), 20);
/// assert_eq!(fit.beta_t.len(), 30);
/// assert!(fit.iterations >= 1);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_glm(
    data: &FdMatrix,
    y: &[f64],
    family: GlmFamily,
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    max_iter: usize,
    tol: f64,
) -> Result<FunctionalGlmResult, FdarError> {
    let (n, m) = data.shape();

    // --- Dimension checks (fire before FPCA) ---
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n}"),
        });
    }
    if m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 column".to_string(),
            actual: "0".to_string(),
        });
    }
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: format!("{n}"),
            actual: format!("{}", y.len()),
        });
    }
    if let Some(sc) = scalar_covariates {
        let sc_rows = sc.shape().0;
        if sc_rows != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows (matching data)"),
                actual: format!("{sc_rows}"),
            });
        }
    }

    // --- Response-domain guard (fires before FPCA) ---
    validate_response(y, family)?;

    let ncomp = ncomp.min(n - 1).min(m);
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);

    let max_iter = if max_iter == 0 { 25 } else { max_iter };
    let tol = if tol <= 0.0 { 1e-6 } else { tol };

    let (beta, iterations) = irls_loop_glm(&design, y, family, max_iter, tol);
    Ok(build_glm_result(
        &design, beta, y, fpca, ncomp, m, iterations, family,
    ))
}

/// Predict response for new functional data using a fitted GLM.
///
/// Projects new curves through the stored FPCA, computes the linear predictor,
/// and applies the inverse link: μ = g⁻¹(η).
///
/// # Arguments
///
/// * `fit` — A fitted [`FunctionalGlmResult`]
/// * `new_data` — New functional predictor matrix (n_new × m), where `m` MUST
///   equal the training grid length
/// * `new_scalar` — Optional new scalar covariates (n_new × p)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `new_data`'s column count differs
/// from the training grid length, or if `new_scalar`'s shape does not match the
/// fitted model's scalar-covariate count (or is missing when the model has
/// scalar covariates). This prevents out-of-bounds indexing / silent truncation.
pub fn predict_functional_glm(
    fit: &FunctionalGlmResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Result<Vec<f64>, FdarError> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();
    let m_train = fit.fpca.mean.len();

    if m != m_train {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m_train} columns (training grid length)"),
            actual: format!("{m}"),
        });
    }
    match new_scalar {
        Some(sc) => {
            let (sc_rows, sc_cols) = sc.shape();
            if sc_rows != n_new {
                return Err(FdarError::InvalidDimension {
                    parameter: "new_scalar",
                    expected: format!("{n_new} rows (matching new_data)"),
                    actual: format!("{sc_rows}"),
                });
            }
            if sc_cols != p_scalar {
                return Err(FdarError::InvalidDimension {
                    parameter: "new_scalar",
                    expected: format!("{p_scalar} columns (model scalar covariates)"),
                    actual: format!("{sc_cols}"),
                });
            }
        }
        None if p_scalar > 0 => {
            return Err(FdarError::InvalidDimension {
                parameter: "new_scalar",
                expected: format!("{p_scalar} columns (model was fit with scalar covariates)"),
                actual: "None".to_string(),
            });
        }
        None => {}
    }

    Ok((0..n_new)
        .map(|i| {
            let mut eta = fit.coefficients[0]; // intercept
            for k in 0..ncomp {
                let mut s = 0.0;
                for j in 0..m {
                    s += (new_data[(i, j)] - fit.fpca.mean[j])
                        * fit.fpca.rotation[(j, k)]
                        * fit.fpca.weights[j];
                }
                eta += fit.coefficients[1 + k] * s;
            }
            if let Some(sc) = new_scalar {
                for j in 0..p_scalar {
                    eta += fit.gamma[j] * sc[(i, j)];
                }
            }
            fit.family.inv_link(eta)
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_on_function::functional_logistic;

    fn make_data(n: usize, m: usize) -> FdMatrix {
        FdMatrix::from_column_major(
            (0..n * m)
                .map(|i| ((i as f64) * 0.07).sin() + 0.01 * (i as f64))
                .collect(),
            n,
            m,
        )
        .unwrap()
    }

    // ------------------------------------------------------------------
    // Task 1: Gaussian tracer smoke test
    // ------------------------------------------------------------------

    #[test]
    fn test_gaussian_smoke() {
        let n = 30;
        let m = 40;
        let data = make_data(n, m);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();

        let fit = functional_glm(&data, &y, GlmFamily::Gaussian, None, 3, 25, 1e-6).unwrap();

        assert_eq!(fit.fitted_values.len(), n, "fitted_values len");
        assert_eq!(fit.beta_t.len(), m, "beta_t len");
        assert!(fit.iterations >= 1, "at least one iteration");
        assert!(
            fit.fitted_values.iter().all(|v| v.is_finite()),
            "all fitted_values finite"
        );
    }

    // ------------------------------------------------------------------
    // Task 2: Binomial parity with functional_logistic
    // ------------------------------------------------------------------

    #[test]
    fn test_binomial_parity_with_logistic() {
        let n = 30;
        let m = 50;
        let data = make_data(n, m);
        // Binary labels: first half 0, second half 1
        let y_bin: Vec<f64> = (0..n).map(|i| if i < n / 2 { 0.0 } else { 1.0 }).collect();

        // functional_logistic stops on max-coefficient-change while functional_glm
        // stops on deviance-change. The per-step IRLS update is identical, so with a
        // tight tol and ample iterations BOTH fully converge to the same fixed point
        // — making the parity comparison deterministic (not criterion-timing dependent).
        let fit_logistic = functional_logistic(&data, &y_bin, None, 3, 100, 1e-12).unwrap();
        let fit_glm =
            functional_glm(&data, &y_bin, GlmFamily::Binomial, None, 3, 100, 1e-12).unwrap();

        // Coefficient parity
        for (i, (a, b)) in fit_logistic
            .coefficients
            .iter()
            .zip(&fit_glm.coefficients)
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "coefficient[{i}] mismatch: logistic={a}, glm={b}"
            );
        }
        // Fitted value (probability) parity
        for (i, (a, b)) in fit_logistic
            .probabilities
            .iter()
            .zip(&fit_glm.fitted_values)
            .enumerate()
        {
            assert!(
                (a - b).abs() < 1e-6,
                "fitted_value[{i}] mismatch: logistic={a}, glm={b}"
            );
        }
    }

    #[test]
    fn test_binomial_out_of_range_guard() {
        let n = 10;
        let m = 20;
        let data = make_data(n, m);
        let mut y = vec![0.0f64; n];
        y[3] = 0.5; // invalid

        let result = functional_glm(&data, &y, GlmFamily::Binomial, None, 3, 25, 1e-6);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected InvalidParameter for out-of-range Binomial y"
        );
    }

    // ------------------------------------------------------------------
    // Task 3: Poisson recovery
    // ------------------------------------------------------------------

    /// Build a rich functional dataset with K orthogonal components, each with varying amplitude.
    ///
    /// Curve i = Σ_k scores_k[i] * basis_k(t) where basis_k = sin(k*π*t).
    /// This ensures FPCA picks up K independent components and the design matrix is full-rank.
    fn make_rich_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        // Generate 3 components with decorrelated, linearly-spaced scores.
        // Curve i = s0*sin(πt) + 0.5*s1*sin(2πt) + 0.25*s2*sin(3πt), where
        // s0, s1, s2 are index-based quasi-random permutations so the 3 FPCA
        // components are well-separated and X'WX is non-singular.
        let mut vals = vec![0.0f64; n * m];
        let mut first_scores = vec![0.0f64; n];
        for i in 0..n {
            let s0 = (i as f64 / (n - 1) as f64) * 2.0 - 1.0; // in [-1, 1]
            let s1 = ((i * 3 % n) as f64 / (n - 1) as f64) * 1.6 - 0.8;
            let s2 = ((i * 7 % n) as f64 / (n - 1) as f64) * 1.4 - 0.7;
            first_scores[i] = s0;
            for j in 0..m {
                let t = j as f64 / (m - 1) as f64;
                let b0 = (std::f64::consts::PI * t).sin();
                let b1 = (2.0 * std::f64::consts::PI * t).sin();
                let b2 = (3.0 * std::f64::consts::PI * t).sin();
                vals[i + j * n] = s0 * b0 + 0.5 * s1 * b1 + 0.25 * s2 * b2;
            }
        }
        let data = FdMatrix::from_column_major(vals, n, m).unwrap();
        (data, first_scores)
    }

    #[test]
    fn test_poisson_recovery() {
        // Deterministic: true Poisson log(mu_i) = 1.0 + 1.5 * first_score_i.
        // y_i = round(mu_i) (integer counts).
        let n = 100;
        let m = 30;

        let (data, first_scores) = make_rich_data(n, m);
        let true_mu: Vec<f64> = first_scores
            .iter()
            .map(|&s| (1.0 + 1.5 * s).exp())
            .collect();
        let y: Vec<f64> = true_mu.iter().map(|&mu| mu.round().max(0.0)).collect();

        let fit = functional_glm(&data, &y, GlmFamily::Poisson, None, 3, 100, 1e-6).unwrap();

        assert!(
            fit.fitted_values.iter().all(|&v| v.is_finite() && v > 0.0),
            "all fitted_values finite and positive"
        );

        // Pearson correlation between fit.fitted_values and true_mu
        let corr = pearson_corr(&fit.fitted_values, &true_mu);
        assert!(corr > 0.9, "Pearson corr={corr} should be > 0.9");
    }

    #[test]
    fn test_gamma_recovery() {
        // True model: 1/μ_i = 2.0 + 1.0 * first_score_i (Gamma inverse link).
        // first_scores from make_rich_data are in [-1, 1], so η_i ∈ [1.0, 3.0] > 0,
        // and μ_i = 1/η_i ∈ [0.33, 1.0] — strictly positive throughout.
        //
        // This test verifies that the Gamma GLM with CORRECT IRLS weight (w = μ²)
        // recovers the true generating mean with Pearson correlation > 0.9.
        // The 3-component functional data from make_rich_data ensures the FPCA
        // scores are not trivially aligned with the true score, providing a
        // meaningful regression test.
        //
        // Note: for this noiseless multi-component fixture the IRLS weight choice
        // (μ² vs 1/μ²) both converge to the same point estimates; the primary
        // value of the corr > 0.9 assertion is to confirm the overall GLM algorithm
        // is producing a sensible Gamma fit, not just finite values.
        let n = 100;
        let m = 30;

        let (data, first_scores) = make_rich_data(n, m);
        // η_i = 2.0 + 1.0 * s_i, s_i ∈ [-1, 1] → η_i ∈ [1.0, 3.0] > 0
        let true_mu: Vec<f64> = first_scores
            .iter()
            .map(|&s| 1.0 / (2.0 + 1.0 * s))
            .collect();
        let y = true_mu.clone();

        let fit = functional_glm(&data, &y, GlmFamily::Gamma, None, 3, 100, 1e-6).unwrap();

        assert!(
            fit.fitted_values.iter().all(|&v| v.is_finite() && v > 0.0),
            "all Gamma fitted_values finite and positive"
        );

        // Sanity: fitted means correlate with true means
        let corr = pearson_corr(&fit.fitted_values, &true_mu);
        assert!(
            corr > 0.9,
            "Gamma recovery: Pearson corr={corr:.4} should be > 0.9"
        );
    }

    // ------------------------------------------------------------------
    // Task 3: domain guard tests
    // ------------------------------------------------------------------

    #[test]
    fn test_poisson_negative_guard() {
        let n = 10;
        let m = 20;
        let data = make_data(n, m);
        let mut y = vec![1.0f64; n];
        y[2] = -1.0;

        let result = functional_glm(&data, &y, GlmFamily::Poisson, None, 3, 25, 1e-6);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected InvalidParameter for negative Poisson y"
        );
    }

    #[test]
    fn test_poisson_noninteger_guard() {
        let n = 10;
        let m = 20;
        let data = make_data(n, m);
        let mut y = vec![1.0f64; n];
        y[5] = 1.5;

        let result = functional_glm(&data, &y, GlmFamily::Poisson, None, 3, 25, 1e-6);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected InvalidParameter for non-integer Poisson y"
        );
    }

    #[test]
    fn test_gamma_nonpositive_guard() {
        let n = 10;
        let m = 20;
        let data = make_data(n, m);
        let mut y = vec![1.0f64; n];
        y[4] = 0.0;

        let result = functional_glm(&data, &y, GlmFamily::Gamma, None, 3, 25, 1e-6);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "expected InvalidParameter for non-positive Gamma y"
        );
    }

    #[test]
    fn test_dimension_mismatch_guard() {
        let n = 10;
        let m = 20;
        let data = make_data(n, m);
        let y = vec![1.0f64; n + 1]; // wrong length

        let result = functional_glm(&data, &y, GlmFamily::Gaussian, None, 3, 25, 1e-6);
        assert!(
            matches!(result, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension for y.len() mismatch"
        );
    }

    #[test]
    fn test_nonfinite_response_guard() {
        // IN-01 / CR-02a: NaN (Gamma) and +Inf (Poisson) must be rejected rather
        // than slipping past the per-family guards into an all-NaN result / an
        // unbounded log-factorial loop.
        let n = 10;
        let m = 20;
        let data = make_data(n, m);

        let mut y_nan = vec![1.0f64; n];
        y_nan[3] = f64::NAN;
        assert!(
            matches!(
                functional_glm(&data, &y_nan, GlmFamily::Gamma, None, 3, 25, 1e-6),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for NaN Gamma response"
        );

        let mut y_inf = vec![1.0f64; n];
        y_inf[5] = f64::INFINITY;
        assert!(
            matches!(
                functional_glm(&data, &y_inf, GlmFamily::Poisson, None, 3, 25, 1e-6),
                Err(FdarError::InvalidParameter { .. })
            ),
            "expected InvalidParameter for +Inf Poisson response"
        );
    }

    #[test]
    fn test_predict_dimension_guard() {
        // CR-03: predict must reject a new_data grid length that differs from the
        // training grid instead of panicking / silently truncating.
        let n = 30;
        let m = 40;
        let data = make_data(n, m);
        let y: Vec<f64> = (0..n).map(|i| (i as f64) * 0.5 + 1.0).collect();
        let fit = functional_glm(&data, &y, GlmFamily::Gaussian, None, 3, 25, 1e-6).unwrap();

        // Correct grid length succeeds.
        assert!(predict_functional_glm(&fit, &data, None).is_ok());

        // Wrong grid length → InvalidDimension (no panic).
        let wrong = make_data(5, m + 3);
        assert!(
            matches!(
                predict_functional_glm(&fit, &wrong, None),
                Err(FdarError::InvalidDimension { .. })
            ),
            "expected InvalidDimension for mismatched predict grid length"
        );
    }

    // ------------------------------------------------------------------
    // Helper: Pearson correlation
    // ------------------------------------------------------------------

    fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let num: f64 = x
            .iter()
            .zip(y)
            .map(|(&xi, &yi)| (xi - mx) * (yi - my))
            .sum();
        let dx: f64 = x.iter().map(|&xi| (xi - mx).powi(2)).sum::<f64>().sqrt();
        let dy: f64 = y.iter().map(|&yi| (yi - my).powi(2)).sum::<f64>().sqrt();
        if dx == 0.0 || dy == 0.0 {
            0.0
        } else {
            num / (dx * dy)
        }
    }
}
