//! Scalar-on-function regression with mixed scalar/functional covariates.
//!
//! Implements models of the form:
//! ```text
//! y = α + ∫β(t)X(t)dt + γᵀz + ε
//! ```
//! where X(t) is a functional predictor, z is a vector of scalar covariates,
//! β(t) is the functional coefficient, and γ is the vector of scalar coefficients.
//!
//! # Methods
//!
//! - [`fregre_lm`]: FPC-based functional linear model with optional scalar covariates
//! - [`fregre_l1`]: L1 (median) robust functional regression via IRLS
//! - [`fregre_huber`]: Huber M-estimation robust functional regression via IRLS
//! - [`fregre_np_mixed`]: Nonparametric kernel regression with product kernels
//! - [`functional_logistic`]: Logistic regression for binary outcomes
//! - [`fregre_cv`]: Cross-validation for number of FPC components

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{FpcaResult, PlsResult};

mod bootstrap;
mod cv;
mod fregre_lm;
mod logistic;
mod nonparametric;
mod pls;
mod robust;
#[cfg(test)]
mod tests;

// Re-export all public items from submodules
pub use bootstrap::{bootstrap_ci_fregre_lm, bootstrap_ci_functional_logistic};
pub use cv::{fregre_basis_cv, fregre_np_cv};
pub use fregre_lm::{fregre_cv, fregre_lm, model_selection_ncomp, predict_fregre_lm};
pub use logistic::{functional_logistic, predict_functional_logistic};
pub use nonparametric::{fregre_np_mixed, predict_fregre_np};
pub use pls::{fregre_pls, predict_fregre_pls};
pub use robust::{fregre_huber, fregre_l1, predict_fregre_robust};

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of functional linear regression.
#[derive(Debug, Clone, PartialEq)]
pub struct FregreLmResult {
    /// Intercept α
    pub intercept: f64,
    /// Functional coefficient β(t), evaluated on the original grid (length m)
    pub beta_t: Vec<f64>,
    /// Pointwise standard errors of β(t) (length m)
    pub beta_se: Vec<f64>,
    /// Scalar coefficients γ (one per scalar covariate)
    pub gamma: Vec<f64>,
    /// Fitted values ŷ (length n)
    pub fitted_values: Vec<f64>,
    /// Residuals y - ŷ (length n)
    pub residuals: Vec<f64>,
    /// R² statistic
    pub r_squared: f64,
    /// Adjusted R²
    pub r_squared_adj: f64,
    /// Standard errors of all coefficients (intercept, FPC scores, scalar covariates)
    pub std_errors: Vec<f64>,
    /// Number of FPC components used
    pub ncomp: usize,
    /// FPCA result (for projecting new data)
    pub fpca: FpcaResult,
    /// Regression coefficients on (FPC scores, scalar covariates) — internal
    pub coefficients: Vec<f64>,
    /// Residual standard error
    pub residual_se: f64,
    /// GCV criterion value (if computed)
    pub gcv: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// Result of nonparametric functional regression with mixed predictors.
#[derive(Debug, Clone, PartialEq)]
pub struct FregreNpResult {
    /// Fitted values ŷ (length n)
    pub fitted_values: Vec<f64>,
    /// Residuals y - ŷ (length n)
    pub residuals: Vec<f64>,
    /// R² statistic
    pub r_squared: f64,
    /// Bandwidth for functional distance kernel
    pub h_func: f64,
    /// Bandwidth for scalar covariates kernel
    pub h_scalar: f64,
    /// Leave-one-out CV error
    pub cv_error: f64,
}

/// Result of robust (L1 or Huber) functional regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FregreRobustResult {
    /// Intercept
    pub intercept: f64,
    /// Functional coefficient β(t), evaluated on the original grid (length m)
    pub beta_t: Vec<f64>,
    /// Fitted values ŷ (length n)
    pub fitted_values: Vec<f64>,
    /// Residuals y - ŷ (length n)
    pub residuals: Vec<f64>,
    /// Regression coefficients (intercept, FPC scores, scalar covariates)
    pub coefficients: Vec<f64>,
    /// Number of FPC components used
    pub ncomp: usize,
    /// FPCA result (for projecting new data)
    pub fpca: FpcaResult,
    /// Number of IRLS iterations performed
    pub iterations: usize,
    /// Whether the IRLS algorithm converged
    pub converged: bool,
    /// Final IRLS weights (length n)
    pub weights: Vec<f64>,
    /// R² statistic
    pub r_squared: f64,
}

/// Result of functional logistic regression.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionalLogisticResult {
    /// Intercept α
    pub intercept: f64,
    /// Functional coefficient β(t), evaluated on the original grid (length m)
    pub beta_t: Vec<f64>,
    /// Pointwise standard errors of β(t) (length m)
    pub beta_se: Vec<f64>,
    /// Scalar coefficients γ (one per scalar covariate)
    pub gamma: Vec<f64>,
    /// Predicted probabilities P(Y=1) (length n)
    pub probabilities: Vec<f64>,
    /// Predicted class labels (0 or 1)
    pub predicted_classes: Vec<usize>,
    /// Number of FPC components used
    pub ncomp: usize,
    /// Classification accuracy on training data
    pub accuracy: f64,
    /// Standard errors of all coefficients (intercept, FPC scores, scalar covariates)
    pub std_errors: Vec<f64>,
    /// Regression coefficients on (FPC scores, scalar covariates) — internal
    pub coefficients: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// Number of IRLS iterations
    pub iterations: usize,
    /// FPCA result (for projecting new data)
    pub fpca: FpcaResult,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// Result of cross-validation for K selection.
#[derive(Debug, Clone, PartialEq)]
pub struct FregreCvResult {
    /// Candidate K values tested
    pub k_values: Vec<usize>,
    /// CV error for each K
    pub cv_errors: Vec<f64>,
    /// Optimal K (minimizing CV error)
    pub optimal_k: usize,
    /// Minimum CV error
    pub min_cv_error: f64,
}

/// Result of PLS-based scalar-on-function regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PlsRegressionResult {
    /// Intercept α
    pub intercept: f64,
    /// Functional coefficient β(t), evaluated on the original grid (length m)
    pub beta_t: Vec<f64>,
    /// Scalar coefficients γ (one per scalar covariate)
    pub gamma: Vec<f64>,
    /// Fitted values ŷ (length n)
    pub fitted_values: Vec<f64>,
    /// Residuals y - ŷ (length n)
    pub residuals: Vec<f64>,
    /// R² statistic
    pub r_squared: f64,
    /// Adjusted R²
    pub r_squared_adj: f64,
    /// Number of PLS components used
    pub ncomp: usize,
    /// PLS result (for projecting new data)
    pub pls: PlsResult,
    /// Regression coefficients on (intercept, PLS scores, scalar covariates)
    pub coefficients: Vec<f64>,
    /// Residual standard error
    pub residual_se: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
}

/// Criterion used for model selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion
    Aic,
    /// Bayesian Information Criterion
    Bic,
    /// Generalized Cross-Validation
    Gcv,
}

/// Result of ncomp model selection.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSelectionResult {
    /// Best number of FPC components by the chosen criterion
    pub best_ncomp: usize,
    /// (ncomp, AIC, BIC, GCV) for each candidate
    pub criteria: Vec<(usize, f64, f64, f64)>,
}

/// Result of bootstrap confidence intervals for β(t).
#[derive(Debug, Clone, PartialEq)]
pub struct BootstrapCiResult {
    /// Pointwise lower bound (length m).
    pub lower: Vec<f64>,
    /// Pointwise upper bound (length m).
    pub upper: Vec<f64>,
    /// Original β(t) estimate (length m).
    pub center: Vec<f64>,
    /// Simultaneous lower bound (sup-norm adjusted, length m).
    pub sim_lower: Vec<f64>,
    /// Simultaneous upper bound (sup-norm adjusted, length m).
    pub sim_upper: Vec<f64>,
    /// Number of bootstrap replicates that converged.
    pub n_boot_success: usize,
}

/// Result of lambda selection for basis regression via cross-validation.
#[derive(Debug, Clone, PartialEq)]
pub struct FregreBasisCvResult {
    /// Optimal smoothing parameter lambda.
    pub optimal_lambda: f64,
    /// Mean CV error for each lambda.
    pub cv_errors: Vec<f64>,
    /// SE of CV error across folds for each lambda.
    pub cv_se: Vec<f64>,
    /// Lambda values tested.
    pub lambda_values: Vec<f64>,
    /// Minimum mean CV error.
    pub min_cv_error: f64,
}

/// Result of bandwidth selection for nonparametric regression via CV.
#[derive(Debug, Clone, PartialEq)]
pub struct FregreNpCvResult {
    /// Optimal bandwidth.
    pub optimal_h: f64,
    /// Mean CV error for each bandwidth.
    pub cv_errors: Vec<f64>,
    /// SE of CV error across folds for each bandwidth.
    pub cv_se: Vec<f64>,
    /// Bandwidth values tested.
    pub h_values: Vec<f64>,
    /// Minimum mean CV error.
    pub min_cv_error: f64,
}

// ---------------------------------------------------------------------------
// Shared linear algebra helpers
// ---------------------------------------------------------------------------

/// Compute X'X (symmetric, p×p stored flat row-major).
pub(crate) fn compute_xtx(x: &FdMatrix) -> Vec<f64> {
    let (n, p) = x.shape();
    let mut xtx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, k)] * x[(i, j)];
            }
            xtx[k * p + j] = s;
            xtx[j * p + k] = s;
        }
    }
    xtx
}

/// Compute X'y (length p).
fn compute_xty(x: &FdMatrix, y: &[f64]) -> Vec<f64> {
    let (n, p) = x.shape();
    (0..p)
        .map(|k| {
            let mut s = 0.0;
            for i in 0..n {
                s += x[(i, k)] * y[i];
            }
            s
        })
        .collect()
}

/// Cholesky factorization: A = LL'. Returns L (p×p flat row-major) or error if singular.
pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut diag = a[j * p + j];
        for k in 0..j {
            diag -= l[j * p + k] * l[j * p + k];
        }
        if diag <= 1e-12 {
            return Err(FdarError::ComputationFailed {
                operation: "Cholesky factorization",
                detail: "matrix is singular or near-singular; try reducing ncomp or check for collinear FPC scores".into(),
            });
        }
        l[j * p + j] = diag.sqrt();
        for i in (j + 1)..p {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = s / l[j * p + j];
        }
    }
    Ok(l)
}

/// Solve Lz = b (forward) then L'x = z (back). L is p×p flat row-major.
pub(crate) fn cholesky_forward_back(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut z = b.to_vec();
    for j in 0..p {
        for k in 0..j {
            z[j] -= l[j * p + k] * z[k];
        }
        z[j] /= l[j * p + j];
    }
    for j in (0..p).rev() {
        for k in (j + 1)..p {
            z[j] -= l[k * p + j] * z[k];
        }
        z[j] /= l[j * p + j];
    }
    z
}

/// Solve Ax = b via Cholesky decomposition (A must be symmetric positive definite).
pub(super) fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let l = cholesky_factor(a, p)?;
    Ok(cholesky_forward_back(&l, b, p))
}

/// Compute hat matrix diagonal: H_ii = x_i' (X'X)^{-1} x_i, given Cholesky factor L of X'X.
pub(crate) fn compute_hat_diagonal(x: &FdMatrix, l: &[f64]) -> Vec<f64> {
    let (n, p) = x.shape();
    let mut hat_diag = vec![0.0; n];
    for i in 0..n {
        let mut v = vec![0.0; p];
        for j in 0..p {
            v[j] = x[(i, j)];
            for k in 0..j {
                v[j] -= l[j * p + k] * v[k];
            }
            v[j] /= l[j * p + j];
        }
        hat_diag[i] = v.iter().map(|vi| vi * vi).sum();
    }
    hat_diag
}

/// Compute diagonal of (X'X)^{-1} given Cholesky factor L, then SE = sqrt(sigma² * diag).
fn compute_ols_std_errors(l: &[f64], p: usize, sigma2: f64) -> Vec<f64> {
    let mut se = vec![0.0; p];
    for j in 0..p {
        let mut v = vec![0.0; p];
        v[j] = 1.0;
        for k in 0..p {
            for kk in 0..k {
                v[k] -= l[k * p + kk] * v[kk];
            }
            v[k] /= l[k * p + k];
        }
        se[j] = (sigma2 * v.iter().map(|vi| vi * vi).sum::<f64>()).sqrt();
    }
    se
}

// ---------------------------------------------------------------------------
// Design matrix and coefficient recovery
// ---------------------------------------------------------------------------

/// Build design matrix: \[1, ξ_1, ..., ξ_K, z_1, ..., z_p\].
/// Validate inputs for fregre_lm / functional_logistic.
fn validate_fregre_inputs(
    n: usize,
    m: usize,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
) -> Result<(), FdarError> {
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
        if sc.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "scalar_covariates",
                expected: format!("{n} rows"),
                actual: format!("{} rows", sc.nrows()),
            });
        }
    }
    Ok(())
}

/// Resolve ncomp: auto-select via CV if 0, otherwise clamp.
fn resolve_ncomp(
    ncomp: usize,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    m: usize,
) -> Result<usize, FdarError> {
    if ncomp == 0 {
        let cv = fregre_cv(data, y, scalar_covariates, 1, m.min(n - 1).min(20), 5)?;
        Ok(cv.optimal_k)
    } else {
        Ok(ncomp.min(n - 1).min(m))
    }
}

pub(crate) fn build_design_matrix(
    fpca_scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> FdMatrix {
    let p_scalar = scalar_covariates.map_or(0, super::matrix::FdMatrix::ncols);
    let p_total = 1 + ncomp + p_scalar;
    let mut design = FdMatrix::zeros(n, p_total);
    for i in 0..n {
        design[(i, 0)] = 1.0;
        for k in 0..ncomp {
            design[(i, 1 + k)] = fpca_scores[(i, k)];
        }
        if let Some(sc) = scalar_covariates {
            for j in 0..p_scalar {
                design[(i, 1 + ncomp + j)] = sc[(i, j)];
            }
        }
    }
    design
}

/// Recover functional coefficient β(t) = Σ_k γ_k φ_k(t).
fn recover_beta_t(fpc_coeffs: &[f64], rotation: &FdMatrix, m: usize) -> Vec<f64> {
    let ncomp = fpc_coeffs.len();
    let mut beta_t = vec![0.0; m];
    for k in 0..ncomp {
        for j in 0..m {
            beta_t[j] += fpc_coeffs[k] * rotation[(j, k)];
        }
    }
    beta_t
}

/// Pointwise standard error of β(t) via error propagation through FPCA rotation.
///
/// SE[β(t_j)]² = Σ_k φ_k(t_j)² · SE[γ_k]²
fn compute_beta_se(gamma_se: &[f64], rotation: &FdMatrix, m: usize) -> Vec<f64> {
    let ncomp = gamma_se.len();
    let mut beta_se = vec![0.0; m];
    for j in 0..m {
        let mut var_j = 0.0;
        for k in 0..ncomp {
            var_j += rotation[(j, k)].powi(2) * gamma_se[k].powi(2);
        }
        beta_se[j] = var_j.sqrt();
    }
    beta_se
}

/// Compute fitted values ŷ = X β.
fn compute_fitted(design: &FdMatrix, coeffs: &[f64]) -> Vec<f64> {
    let (n, p) = design.shape();
    (0..n)
        .map(|i| {
            let mut yhat = 0.0;
            for j in 0..p {
                yhat += design[(i, j)] * coeffs[j];
            }
            yhat
        })
        .collect()
}

/// Compute R² and adjusted R².
fn compute_r_squared(y: &[f64], residuals: &[f64], p_total: usize) -> (f64, f64) {
    let n = y.len();
    let y_mean = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };
    let df_model = (p_total - 1) as f64;
    let r_squared_adj = if n as f64 - df_model - 1.0 > 0.0 {
        1.0 - (1.0 - r_squared) * (n as f64 - 1.0) / (n as f64 - df_model - 1.0)
    } else {
        r_squared
    };
    (r_squared, r_squared_adj)
}

// ---------------------------------------------------------------------------
// OLS solver
// ---------------------------------------------------------------------------

/// Solve ordinary least squares: min ||Xb - y||² via normal equations with Cholesky.
/// Returns (coefficients, hat_matrix_diagonal) or error if singular.
fn ols_solve(x: &FdMatrix, y: &[f64]) -> Result<(Vec<f64>, Vec<f64>), FdarError> {
    let (n, p) = x.shape();
    if n < p || p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "design matrix",
            expected: format!("n >= p and p > 0 (p={p})"),
            actual: format!("n={n}, p={p}"),
        });
    }
    let xtx = compute_xtx(x);
    let xty = compute_xty(x, y);
    let l = cholesky_factor(&xtx, p)?;
    let b = cholesky_forward_back(&l, &xty, p);
    let hat_diag = compute_hat_diagonal(x, &l);
    Ok((b, hat_diag))
}

/// Sigmoid function: 1 / (1 + exp(-x))
pub(crate) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

// ---------------------------------------------------------------------------
// Predict methods on result structs
// ---------------------------------------------------------------------------

impl FregreLmResult {
    /// Predict new responses. Delegates to [`predict_fregre_lm`].
    pub fn predict(&self, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64> {
        predict_fregre_lm(self, new_data, new_scalar)
    }
}

impl FregreRobustResult {
    /// Predict new responses. Delegates to [`predict_fregre_robust`].
    pub fn predict(&self, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64> {
        predict_fregre_robust(self, new_data, new_scalar)
    }
}

impl FunctionalLogisticResult {
    /// Predict P(Y=1) for new data. Delegates to [`predict_functional_logistic`].
    pub fn predict(&self, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64> {
        predict_functional_logistic(self, new_data, new_scalar)
    }
}
