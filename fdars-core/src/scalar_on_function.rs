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
//! - [`fregre_np_mixed`]: Nonparametric kernel regression with product kernels
//! - [`functional_logistic`]: Logistic regression for binary outcomes
//! - [`fregre_cv`]: Cross-validation for number of FPC components

use crate::cv::create_folds;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use rand::prelude::*;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of functional linear regression.
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

/// Result of functional logistic regression.
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
    pub predicted_classes: Vec<u8>,
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
pub struct ModelSelectionResult {
    /// Best number of FPC components by the chosen criterion
    pub best_ncomp: usize,
    /// (ncomp, AIC, BIC, GCV) for each candidate
    pub criteria: Vec<(usize, f64, f64, f64)>,
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

/// Cholesky factorization: A = LL'. Returns L (p×p flat row-major) or None if singular.
pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut diag = a[j * p + j];
        for k in 0..j {
            diag -= l[j * p + k] * l[j * p + k];
        }
        if diag <= 1e-12 {
            return None;
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
    Some(l)
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
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let l = cholesky_factor(a, p)?;
    Some(cholesky_forward_back(&l, b, p))
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
) -> Option<()> {
    if n < 3 || m == 0 || y.len() != n {
        return None;
    }
    if let Some(sc) = scalar_covariates {
        if sc.nrows() != n {
            return None;
        }
    }
    Some(())
}

/// Resolve ncomp: auto-select via CV if 0, otherwise clamp.
fn resolve_ncomp(
    ncomp: usize,
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
    m: usize,
) -> Option<usize> {
    if ncomp == 0 {
        let cv = fregre_cv(data, y, scalar_covariates, 1, m.min(n - 1).min(20), 5)?;
        Some(cv.optimal_k)
    } else {
        Some(ncomp.min(n - 1).min(m))
    }
}

pub(crate) fn build_design_matrix(
    fpca_scores: &FdMatrix,
    ncomp: usize,
    scalar_covariates: Option<&FdMatrix>,
    n: usize,
) -> FdMatrix {
    let p_scalar = scalar_covariates.map_or(0, |sc| sc.ncols());
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
/// Returns (coefficients, hat_matrix_diagonal) or None if singular.
fn ols_solve(x: &FdMatrix, y: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let (n, p) = x.shape();
    if n < p || p == 0 {
        return None;
    }
    let xtx = compute_xtx(x);
    let xty = compute_xty(x, y);
    let l = cholesky_factor(&xtx, p)?;
    let b = cholesky_forward_back(&l, &xty, p);
    let hat_diag = compute_hat_diagonal(x, &l);
    Some((b, hat_diag))
}

// ---------------------------------------------------------------------------
// fregre_lm: FPC-based functional linear model
// ---------------------------------------------------------------------------

/// Functional linear model with optional scalar covariates.
///
/// Fits the model: `y = α + Σ_k γ_k ξ_k + γ_z' z + ε`
/// where ξ_k are FPC scores of the functional predictor X(t) and z are scalar covariates.
/// The functional coefficient is recovered as `β(t) = Σ_k γ_k φ_k(t)`.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix (n × p), `None` for pure functional model
/// * `ncomp` - Number of FPC components (if 0, selected by GCV)
///
/// # Returns
/// [`FregreLmResult`] with estimated coefficients, fitted values, and diagnostics
pub fn fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<FregreLmResult> {
    let (n, m) = data.shape();
    validate_fregre_inputs(n, m, y, scalar_covariates)?;

    let ncomp = resolve_ncomp(ncomp, data, y, scalar_covariates, n, m)?;

    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);
    let p_total = design.ncols();
    let (coeffs, hat_diag) = ols_solve(&design, y)?;

    let fitted_values = compute_fitted(&design, &coeffs);
    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, r_squared_adj) = compute_r_squared(y, &residuals, p_total);

    let ss_res: f64 = residuals.iter().map(|r| r * r).sum();
    let df_resid = (n as f64 - p_total as f64).max(1.0);
    let residual_se = (ss_res / df_resid).sqrt();
    let sigma2 = ss_res / df_resid;

    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p_total).unwrap_or_else(|| vec![1.0; p_total * p_total]);
    let std_errors = compute_ols_std_errors(&l, p_total, sigma2);

    let gcv = hat_diag
        .iter()
        .zip(&residuals)
        .map(|(&h, &r)| (r / (1.0 - h).max(1e-10)).powi(2))
        .sum::<f64>()
        / n as f64;

    let beta_t = recover_beta_t(&coeffs[1..1 + ncomp], &fpca.rotation, m);
    let beta_se = compute_beta_se(&std_errors[1..1 + ncomp], &fpca.rotation, m);
    let gamma: Vec<f64> = coeffs[1 + ncomp..].to_vec();

    let nf = n as f64;
    let rss = ss_res;
    let aic = nf * (rss / nf).ln() + 2.0 * p_total as f64;
    let bic = nf * (rss / nf).ln() + (nf).ln() * p_total as f64;

    Some(FregreLmResult {
        intercept: coeffs[0],
        beta_t,
        beta_se,
        gamma,
        fitted_values,
        residuals,
        r_squared,
        r_squared_adj,
        std_errors,
        ncomp,
        fpca,
        coefficients: coeffs,
        residual_se,
        gcv,
        aic,
        bic,
    })
}

// ---------------------------------------------------------------------------
// fregre_cv: Cross-validation for K selection
// ---------------------------------------------------------------------------

/// Copy a subset of rows from src into dst.
fn copy_rows(dst: &mut FdMatrix, src: &FdMatrix, src_rows: &[usize]) {
    let ncols = src.ncols();
    for (dst_i, &src_i) in src_rows.iter().enumerate() {
        for j in 0..ncols {
            dst[(dst_i, j)] = src[(src_i, j)];
        }
    }
}

/// Compute CV error for a single K across all folds (randomized assignment).
fn cv_error_for_k(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    k: usize,
    n_folds: usize,
    folds: &[usize],
) -> Option<f64> {
    let n = data.nrows();
    let ncols = data.ncols();
    let mut total_error = 0.0;
    let mut count = 0;

    for fold in 0..n_folds {
        let train_idx: Vec<usize> = (0..n).filter(|&i| folds[i] != fold).collect();
        let test_idx: Vec<usize> = (0..n).filter(|&i| folds[i] == fold).collect();
        let n_test = test_idx.len();
        if n_test == 0 || train_idx.len() < k + 2 {
            continue;
        }

        let mut train_data = FdMatrix::zeros(train_idx.len(), ncols);
        let mut test_data = FdMatrix::zeros(n_test, ncols);
        copy_rows(&mut train_data, data, &train_idx);
        copy_rows(&mut test_data, data, &test_idx);

        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();
        let test_y: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

        let train_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(train_idx.len(), sc.ncols());
            copy_rows(&mut m, sc, &train_idx);
            m
        });
        let test_sc = scalar_covariates.map(|sc| {
            let mut m = FdMatrix::zeros(n_test, sc.ncols());
            copy_rows(&mut m, sc, &test_idx);
            m
        });

        let fit = match fregre_lm(&train_data, &train_y, train_sc.as_ref(), k) {
            Some(f) => f,
            None => continue,
        };

        let predictions = predict_fregre_lm(&fit, &test_data, test_sc.as_ref());
        let fold_mse: f64 = predictions
            .iter()
            .zip(&test_y)
            .map(|(&yhat, &yi)| (yhat - yi).powi(2))
            .sum::<f64>()
            / n_test as f64;

        total_error += fold_mse * n_test as f64;
        count += n_test;
    }

    if count > 0 {
        Some(total_error / count as f64)
    } else {
        None
    }
}

/// K-fold cross-validation for selecting the number of FPC components.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector (length n)
/// * `scalar_covariates` - Optional scalar covariates matrix
/// * `k_min` - Minimum number of components to test
/// * `k_max` - Maximum number of components to test
/// * `n_folds` - Number of CV folds
pub fn fregre_cv(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    k_min: usize,
    k_max: usize,
    n_folds: usize,
) -> Option<FregreCvResult> {
    let n = data.nrows();
    if n < n_folds || k_min < 1 || k_min > k_max {
        return None;
    }

    let k_max = k_max.min(n - 2).min(data.ncols());

    // Use randomized fold assignment (consistent seed for reproducibility)
    let folds = create_folds(n, n_folds, 42);

    let mut k_values = Vec::new();
    let mut cv_errors = Vec::new();

    for k in k_min..=k_max {
        if let Some(err) = cv_error_for_k(data, y, scalar_covariates, k, n_folds, &folds) {
            k_values.push(k);
            cv_errors.push(err);
        }
    }

    if k_values.is_empty() {
        return None;
    }

    let (optimal_idx, &min_cv_error) = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    Some(FregreCvResult {
        k_values: k_values.clone(),
        cv_errors,
        optimal_k: k_values[optimal_idx],
        min_cv_error,
    })
}

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

/// Select optimal ncomp for `fregre_lm` using AIC, BIC, or GCV.
///
/// Fits models for `ncomp = 1..=max_ncomp`, collects information criteria,
/// and returns the best ncomp under the chosen criterion.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar responses (length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `max_ncomp` — Maximum number of FPC components to try
/// * `criterion` — Which criterion to minimise
pub fn model_selection_ncomp(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    max_ncomp: usize,
    criterion: SelectionCriterion,
) -> Option<ModelSelectionResult> {
    if max_ncomp == 0 {
        return None;
    }

    let mut criteria = Vec::with_capacity(max_ncomp);

    for k in 1..=max_ncomp {
        if let Some(fit) = fregre_lm(data, y, scalar_covariates, k) {
            criteria.push((k, fit.aic, fit.bic, fit.gcv));
        }
    }

    if criteria.is_empty() {
        return None;
    }

    let best_idx = match criterion {
        SelectionCriterion::Aic => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
        SelectionCriterion::Bic => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
        SelectionCriterion::Gcv => criteria
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0),
    };

    Some(ModelSelectionResult {
        best_ncomp: criteria[best_idx].0,
        criteria,
    })
}

/// Predict new responses using a fitted functional linear model.
///
/// # Arguments
/// * `fit` - A fitted [`FregreLmResult`]
/// * `new_data` - New functional predictor matrix (n_new × m)
/// * `new_scalar` - Optional new scalar covariates (n_new × p)
pub fn predict_fregre_lm(
    fit: &FregreLmResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();

    let mut predictions = vec![0.0; n_new];
    for i in 0..n_new {
        let mut yhat = fit.intercept;
        for k in 0..ncomp {
            let mut s = 0.0;
            for j in 0..m {
                s += (new_data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, k)];
            }
            yhat += fit.coefficients[1 + k] * s;
        }
        if let Some(sc) = new_scalar {
            for j in 0..p_scalar {
                yhat += fit.gamma[j] * sc[(i, j)];
            }
        }
        predictions[i] = yhat;
    }
    predictions
}

// ---------------------------------------------------------------------------
// Nonparametric kernel regression
// ---------------------------------------------------------------------------

/// Gaussian kernel: K(d, h) = exp(-d² / (2h²))
fn gaussian_kernel(d: f64, h: f64) -> f64 {
    (-d * d / (2.0 * h * h)).exp()
}

/// Compute symmetric pairwise distance matrix (flat n×n).
fn compute_pairwise_distances(data: &FdMatrix, argvals: &[f64]) -> Vec<f64> {
    let n = data.nrows();
    let weights = crate::helpers::simpsons_weights(argvals);
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = crate::helpers::l2_distance(&data.row(i), &data.row(j), &weights);
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Compute pairwise Euclidean distance matrix for scalar covariates.
fn compute_scalar_distances(sc: &FdMatrix) -> Vec<f64> {
    let n = sc.nrows();
    let p = sc.ncols();
    let mut dists = vec![0.0; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0.0;
            for k in 0..p {
                let diff = sc[(i, k)] - sc[(j, k)];
                d2 += diff * diff;
            }
            let d = d2.sqrt();
            dists[i * n + j] = d;
            dists[j * n + i] = d;
        }
    }
    dists
}

/// Nadaraya-Watson LOO prediction for one observation.
fn nw_loo_predict(
    i: usize,
    n: usize,
    y: &[f64],
    func_dists: &[f64],
    scalar_dists: &[f64],
    h_func: f64,
    h_scalar: f64,
    has_scalar: bool,
) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for j in 0..n {
        if i == j {
            continue;
        }
        let kf = gaussian_kernel(func_dists[i * n + j], h_func);
        let ks = if has_scalar {
            gaussian_kernel(scalar_dists[i * n + j], h_scalar)
        } else {
            1.0
        };
        let w = kf * ks;
        num += w * y[j];
        den += w;
    }
    if den > 1e-15 {
        num / den
    } else {
        y[i]
    }
}

/// LOO-CV error for Nadaraya-Watson with a single bandwidth.
fn loo_cv_error(dists: &[f64], y: &[f64], n: usize, h: f64) -> f64 {
    (0..n)
        .map(|i| {
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let w = gaussian_kernel(dists[i * n + j], h);
                num += w * y[j];
                den += w;
            }
            let yhat = if den > 1e-15 { num / den } else { y[i] };
            (y[i] - yhat).powi(2)
        })
        .sum::<f64>()
        / n as f64
}

/// Select bandwidth by minimizing LOO-CV error on a grid of distance quantiles.
fn select_bandwidth_loo(dists: &[f64], y: &[f64], n: usize, _other_dists: Option<&[f64]>) -> f64 {
    let mut nonzero_dists: Vec<f64> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| dists[i * n + j]))
        .filter(|&d| d > 0.0)
        .collect();
    nonzero_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if nonzero_dists.is_empty() {
        return 1.0;
    }

    let n_cand = 20;
    let mut best_h = nonzero_dists[nonzero_dists.len() / 2];
    let mut best_cv = f64::INFINITY;

    for qi in 1..=n_cand {
        let q = qi as f64 / (n_cand + 1) as f64;
        let idx = ((nonzero_dists.len() as f64 * q) as usize).min(nonzero_dists.len() - 1);
        let h = nonzero_dists[idx].max(1e-10);
        let cv = loo_cv_error(dists, y, n, h);
        if cv < best_cv {
            best_cv = cv;
            best_h = h;
        }
    }
    best_h
}

/// Nonparametric kernel regression with mixed functional and scalar predictors.
///
/// Uses product kernels:
/// ```text
/// ŷ(x, z) = Σᵢ K_func(Xᵢ, x) · K_scalar(zᵢ, z) · yᵢ / Σᵢ K_func(Xᵢ, x) · K_scalar(zᵢ, z)
/// ```
///
/// Bandwidths are selected via leave-one-out CV if set to 0.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Scalar response vector
/// * `argvals` - Grid points for integration (length m)
/// * `scalar_covariates` - Optional scalar covariates (n × p)
/// * `h_func` - Bandwidth for functional kernel (0 for automatic)
/// * `h_scalar` - Bandwidth for scalar kernel (0 for automatic)
pub fn fregre_np_mixed(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    h_func: f64,
    h_scalar: f64,
) -> Option<FregreNpResult> {
    let n = data.nrows();
    if n < 3 || data.ncols() == 0 || y.len() != n || argvals.len() != data.ncols() {
        return None;
    }

    let func_dists = compute_pairwise_distances(data, argvals);
    let has_scalar = scalar_covariates.is_some();
    let scalar_dists = scalar_covariates
        .map(compute_scalar_distances)
        .unwrap_or_default();

    let h_func = if h_func <= 0.0 {
        select_bandwidth_loo(&func_dists, y, n, None)
    } else {
        h_func
    };

    let h_scalar = if has_scalar && h_scalar <= 0.0 {
        select_bandwidth_loo(&scalar_dists, y, n, Some(&func_dists))
    } else {
        h_scalar
    };

    let mut fitted_values = vec![0.0; n];
    let mut cv_error = 0.0;
    for i in 0..n {
        fitted_values[i] = nw_loo_predict(
            i,
            n,
            y,
            &func_dists,
            &scalar_dists,
            h_func,
            h_scalar,
            has_scalar,
        );
        cv_error += (y[i] - fitted_values[i]).powi(2);
    }
    cv_error /= n as f64;

    let residuals: Vec<f64> = y
        .iter()
        .zip(&fitted_values)
        .map(|(&yi, &yh)| yi - yh)
        .collect();
    let (r_squared, _) = compute_r_squared(y, &residuals, 1);

    Some(FregreNpResult {
        fitted_values,
        residuals,
        r_squared,
        h_func,
        h_scalar,
        cv_error,
    })
}

/// Predict new responses using a fitted nonparametric model.
pub fn predict_fregre_np(
    train_data: &FdMatrix,
    y: &[f64],
    train_scalar: Option<&FdMatrix>,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
    argvals: &[f64],
    h_func: f64,
    h_scalar: f64,
) -> Vec<f64> {
    let n_train = train_data.nrows();
    let n_new = new_data.nrows();
    let weights = crate::helpers::simpsons_weights(argvals);

    (0..n_new)
        .map(|i| {
            let new_row = new_data.row(i);
            let mut num = 0.0;
            let mut den = 0.0;
            for j in 0..n_train {
                let d_func = crate::helpers::l2_distance(&new_row, &train_data.row(j), &weights);
                let kf = gaussian_kernel(d_func, h_func);
                let ks = match (new_scalar, train_scalar) {
                    (Some(ns), Some(ts)) => {
                        let d2: f64 = (0..ns.ncols())
                            .map(|k| (ns[(i, k)] - ts[(j, k)]).powi(2))
                            .sum();
                        gaussian_kernel(d2.sqrt(), h_scalar)
                    }
                    _ => 1.0,
                };
                let w = kf * ks;
                num += w * y[j];
                den += w;
            }
            if den > 1e-15 {
                num / den
            } else {
                0.0
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Functional logistic regression
// ---------------------------------------------------------------------------

/// One IRLS step: compute working response and solve weighted least squares.
/// Returns updated beta or None if system is singular.
fn irls_step(design: &FdMatrix, y: &[f64], beta: &[f64]) -> Option<Vec<f64>> {
    let (n, p) = design.shape();

    // Linear predictor η = Xβ, probabilities μ = sigmoid(η)
    let eta: Vec<f64> = (0..n)
        .map(|i| (0..p).map(|j| design[(i, j)] * beta[j]).sum())
        .collect();
    let mu: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
    let w: Vec<f64> = mu.iter().map(|&p| (p * (1.0 - p)).max(1e-10)).collect();
    let z_work: Vec<f64> = (0..n).map(|i| eta[i] + (y[i] - mu[i]) / w[i]).collect();

    // Weighted normal equations: (X'WX) β = X'Wz
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += design[(i, k)] * w[i] * design[(i, j)];
            }
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }

    let mut xtwz = vec![0.0; p];
    for k in 0..p {
        let mut s = 0.0;
        for i in 0..n {
            s += design[(i, k)] * w[i] * z_work[i];
        }
        xtwz[k] = s;
    }

    cholesky_solve(&xtwx, &xtwz, p)
}

/// Compute log-likelihood of logistic model.
fn logistic_log_likelihood(probabilities: &[f64], y: &[f64]) -> f64 {
    probabilities
        .iter()
        .zip(y)
        .map(|(&p, &yi)| {
            let p = p.clamp(1e-15, 1.0 - 1e-15);
            yi * p.ln() + (1.0 - yi) * (1.0 - p).ln()
        })
        .sum()
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

/// Run IRLS iteration loop and return (beta, iterations).
fn irls_loop(design: &FdMatrix, y: &[f64], max_iter: usize, tol: f64) -> (Vec<f64>, usize) {
    let p_total = design.ncols();
    let mut beta = vec![0.0; p_total];
    let mut iterations = 0;
    for iter in 0..max_iter {
        iterations = iter + 1;
        let beta_new = match irls_step(design, y, &beta) {
            Some(b) => b,
            None => break,
        };
        let change: f64 = beta_new
            .iter()
            .zip(&beta)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        beta = beta_new;
        if change < tol {
            break;
        }
    }
    (beta, iterations)
}

/// Build logistic result from converged beta.
fn build_logistic_result(
    design: &FdMatrix,
    beta: Vec<f64>,
    y: &[f64],
    fpca: FpcaResult,
    ncomp: usize,
    m: usize,
    iterations: usize,
) -> FunctionalLogisticResult {
    let (n, p) = design.shape();
    let eta = compute_fitted(design, &beta);
    let probabilities: Vec<f64> = eta.iter().map(|&e| sigmoid(e)).collect();
    let predicted_classes: Vec<u8> = probabilities
        .iter()
        .map(|&p| if p >= 0.5 { 1 } else { 0 })
        .collect();
    let correct: usize = predicted_classes
        .iter()
        .zip(y)
        .filter(|(&pred, &actual)| pred as f64 == actual)
        .count();
    let beta_t = recover_beta_t(&beta[1..1 + ncomp], &fpca.rotation, m);
    let gamma: Vec<f64> = beta[1 + ncomp..].to_vec();

    // Compute coefficient SEs from Fisher information matrix (X'WX)^{-1}
    let w: Vec<f64> = probabilities
        .iter()
        .map(|&mu| (mu * (1.0 - mu)).max(1e-10))
        .collect();
    let mut xtwx = vec![0.0; p * p];
    for k in 0..p {
        for j in k..p {
            let mut s = 0.0;
            for i in 0..n {
                s += design[(i, k)] * w[i] * design[(i, j)];
            }
            xtwx[k * p + j] = s;
            xtwx[j * p + k] = s;
        }
    }
    let std_errors = cholesky_factor(&xtwx, p)
        .map(|l| compute_ols_std_errors(&l, p, 1.0))
        .unwrap_or_else(|| vec![f64::NAN; p]);
    let beta_se = compute_beta_se(&std_errors[1..1 + ncomp], &fpca.rotation, m);

    let ll = logistic_log_likelihood(&probabilities, y);
    let deviance = -2.0 * ll;
    let nf = n as f64;
    let pf = p as f64;
    let aic = deviance + 2.0 * pf;
    let bic = deviance + nf.ln() * pf;

    FunctionalLogisticResult {
        intercept: beta[0],
        beta_t,
        beta_se,
        gamma,
        accuracy: correct as f64 / nf,
        log_likelihood: ll,
        probabilities,
        predicted_classes,
        ncomp,
        std_errors,
        coefficients: beta,
        iterations,
        fpca,
        aic,
        bic,
    }
}

/// Functional logistic regression for binary outcomes.
///
/// Fits: `log(P(Y=1)/P(Y=0)) = α + ∫β(t)X(t)dt + γᵀz`
/// using IRLS (iteratively reweighted least squares) on FPC scores.
///
/// # Arguments
/// * `data` - Functional predictor matrix (n × m)
/// * `y` - Binary response vector (0.0 or 1.0, length n)
/// * `scalar_covariates` - Optional scalar covariates (n × p)
/// * `ncomp` - Number of FPC components
/// * `max_iter` - Maximum IRLS iterations (default: 25)
/// * `tol` - Convergence tolerance (default: 1e-6)
pub fn functional_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    max_iter: usize,
    tol: f64,
) -> Option<FunctionalLogisticResult> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || y.len() != n {
        return None;
    }
    if y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
        return None;
    }

    let ncomp = ncomp.min(n - 1).min(m);
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let design = build_design_matrix(&fpca.scores, ncomp, scalar_covariates, n);

    let max_iter = if max_iter == 0 { 25 } else { max_iter };
    let tol = if tol <= 0.0 { 1e-6 } else { tol };

    let (beta, iterations) = irls_loop(&design, y, max_iter, tol);
    Some(build_logistic_result(
        &design, beta, y, fpca, ncomp, m, iterations,
    ))
}

// ---------------------------------------------------------------------------
// Bootstrap CIs for β(t)
// ---------------------------------------------------------------------------

/// Result of bootstrap confidence intervals for β(t).
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

/// Gather rows from `src` by index (with replacement), returning a new matrix.
fn subsample_rows(src: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let ncols = src.ncols();
    let mut out = FdMatrix::zeros(indices.len(), ncols);
    for (dst_i, &src_i) in indices.iter().enumerate() {
        for j in 0..ncols {
            out[(dst_i, j)] = src[(src_i, j)];
        }
    }
    out
}

/// Compute the q-th quantile of a sorted slice (linear interpolation).
fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = q * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = lo + 1;
    let frac = pos - lo as f64;
    if hi >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Bootstrap confidence intervals for β(t) from a functional linear model.
///
/// Uses cases bootstrap (resampling observation indices with replacement) to
/// build pointwise and simultaneous confidence bands for the functional
/// coefficient β(t).
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar response vector (length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
/// * `n_boot` — Number of bootstrap replicates
/// * `alpha` — Significance level (e.g., 0.05 for 95% CI)
/// * `seed` — RNG seed for reproducibility
pub fn bootstrap_ci_fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    alpha: f64,
    seed: u64,
) -> Option<BootstrapCiResult> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || y.len() != n || n_boot == 0 || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }

    // Fit original model
    let original = fregre_lm(data, y, scalar_covariates, ncomp)?;
    let center = original.beta_t.clone();

    // Bootstrap replicates
    let boot_betas: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_boot)
        .filter_map(|b| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let boot_data = subsample_rows(data, &indices);
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &indices));

            fregre_lm(&boot_data, &boot_y, boot_sc.as_ref(), ncomp).map(|fit| fit.beta_t)
        })
        .collect();

    let n_boot_success = boot_betas.len();
    if n_boot_success < 3 {
        return None;
    }

    // Pointwise bands: sort each grid point across replicates
    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let mut lower = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut boot_se = vec![0.0; m];

    for j in 0..m {
        let mut vals: Vec<f64> = boot_betas.iter().map(|b| b[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lower[j] = quantile(&vals, lo_q);
        upper[j] = quantile(&vals, hi_q);

        // Bootstrap SE at this grid point
        let mean_j: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var_j: f64 =
            vals.iter().map(|&v| (v - mean_j).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        boot_se[j] = var_j.sqrt().max(1e-15);
    }

    // Simultaneous bands: sup-norm bootstrap
    let mut t_stats: Vec<f64> = boot_betas
        .iter()
        .map(|b| {
            (0..m)
                .map(|j| ((b[j] - center[j]) / boot_se[j]).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    t_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let c_alpha = quantile(&t_stats, 1.0 - alpha);
    let sim_lower: Vec<f64> = (0..m).map(|j| center[j] - c_alpha * boot_se[j]).collect();
    let sim_upper: Vec<f64> = (0..m).map(|j| center[j] + c_alpha * boot_se[j]).collect();

    Some(BootstrapCiResult {
        lower,
        upper,
        center,
        sim_lower,
        sim_upper,
        n_boot_success,
    })
}

/// Bootstrap confidence intervals for β(t) from a functional logistic model.
///
/// Same algorithm as [`bootstrap_ci_fregre_lm`] but using [`functional_logistic`]
/// as the inner estimator. Degenerate resamples (all-0 or all-1 y) fail naturally.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Binary response vector (0.0 or 1.0, length n)
/// * `scalar_covariates` — Optional scalar covariates (n × p)
/// * `ncomp` — Number of FPC components
/// * `n_boot` — Number of bootstrap replicates
/// * `alpha` — Significance level
/// * `seed` — RNG seed
/// * `max_iter` — Maximum IRLS iterations per replicate
/// * `tol` — IRLS convergence tolerance
pub fn bootstrap_ci_functional_logistic(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
    n_boot: usize,
    alpha: f64,
    seed: u64,
    max_iter: usize,
    tol: f64,
) -> Option<BootstrapCiResult> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || y.len() != n || n_boot == 0 || alpha <= 0.0 || alpha >= 1.0 {
        return None;
    }

    // Fit original model
    let original = functional_logistic(data, y, scalar_covariates, ncomp, max_iter, tol)?;
    let center = original.beta_t.clone();

    // Bootstrap replicates
    let boot_betas: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_boot)
        .filter_map(|b| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            let boot_data = subsample_rows(data, &indices);
            let boot_y: Vec<f64> = indices.iter().map(|&i| y[i]).collect();
            let boot_sc = scalar_covariates.map(|sc| subsample_rows(sc, &indices));

            functional_logistic(&boot_data, &boot_y, boot_sc.as_ref(), ncomp, max_iter, tol)
                .map(|fit| fit.beta_t)
        })
        .collect();

    let n_boot_success = boot_betas.len();
    if n_boot_success < 3 {
        return None;
    }

    let lo_q = alpha / 2.0;
    let hi_q = 1.0 - alpha / 2.0;
    let mut lower = vec![0.0; m];
    let mut upper = vec![0.0; m];
    let mut boot_se = vec![0.0; m];

    for j in 0..m {
        let mut vals: Vec<f64> = boot_betas.iter().map(|b| b[j]).collect();
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        lower[j] = quantile(&vals, lo_q);
        upper[j] = quantile(&vals, hi_q);

        let mean_j: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        let var_j: f64 =
            vals.iter().map(|&v| (v - mean_j).powi(2)).sum::<f64>() / (vals.len() - 1) as f64;
        boot_se[j] = var_j.sqrt().max(1e-15);
    }

    let mut t_stats: Vec<f64> = boot_betas
        .iter()
        .map(|b| {
            (0..m)
                .map(|j| ((b[j] - center[j]) / boot_se[j]).abs())
                .fold(0.0_f64, f64::max)
        })
        .collect();
    t_stats.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let c_alpha = quantile(&t_stats, 1.0 - alpha);
    let sim_lower: Vec<f64> = (0..m).map(|j| center[j] - c_alpha * boot_se[j]).collect();
    let sim_upper: Vec<f64> = (0..m).map(|j| center[j] + c_alpha * boot_se[j]).collect();

    Some(BootstrapCiResult {
        lower,
        upper,
        center,
        sim_lower,
        sim_upper,
        n_boot_success,
    })
}

// ---------------------------------------------------------------------------
// Basis regression CV (R's fregre.basis.cv)
// ---------------------------------------------------------------------------

/// Result of lambda selection for basis regression via cross-validation.
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

/// K-fold CV for selecting the regularization parameter lambda
/// in basis-regression (R's `fregre.basis.cv`).
///
/// Projects functional predictors onto a B-spline or Fourier basis,
/// fits a ridge regression on the resulting coefficients for each lambda,
/// and selects the lambda minimizing mean CV error.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar response vector (length n)
/// * `argvals` — Evaluation grid (length m)
/// * `n_folds` — Number of CV folds
/// * `lambda_range` — Lambda values to test. If `None`, uses 20 log-spaced
///   values from 1e-4 to 1e4.
/// * `nbasis` — Number of basis functions
/// * `basis_type` — Basis type (B-spline or Fourier)
pub fn fregre_basis_cv(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    lambda_range: Option<&[f64]>,
    nbasis: usize,
    basis_type: &crate::smooth_basis::BasisType,
) -> Option<FregreBasisCvResult> {
    use crate::smooth_basis::{smooth_basis, BasisType, FdPar};

    let (n, m) = data.shape();
    if n < n_folds || m == 0 || y.len() != n || argvals.len() != m || nbasis < 2 {
        return None;
    }

    // Default lambda grid
    let default_lambdas: Vec<f64> = if lambda_range.is_none() {
        (0..20)
            .map(|i| {
                let log_lam = -4.0 + 8.0 * i as f64 / 19.0;
                10.0_f64.powf(log_lam)
            })
            .collect()
    } else {
        Vec::new()
    };
    let lambdas = lambda_range.unwrap_or(&default_lambdas);

    if lambdas.is_empty() {
        return None;
    }

    // Compute penalty matrix once
    let penalty = match basis_type {
        BasisType::Bspline { order } => {
            crate::smooth_basis::bspline_penalty_matrix(argvals, nbasis, *order, 2)
        }
        BasisType::Fourier { period } => {
            crate::smooth_basis::fourier_penalty_matrix(nbasis, *period, 2)
        }
    };

    // Create folds
    let folds = crate::cv::create_folds(n, n_folds, 42);

    let mut cv_errors = vec![0.0; lambdas.len()];
    let mut cv_fold_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(n_folds); lambdas.len()];

    for fold in 0..n_folds {
        let (train_idx, test_idx) = crate::cv::fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let train_data = crate::cv::subset_rows(data, &train_idx);
        let train_y = crate::cv::subset_vec(y, &train_idx);
        let test_data = crate::cv::subset_rows(data, &test_idx);
        let test_y = crate::cv::subset_vec(y, &test_idx);

        for (li, &lam) in lambdas.iter().enumerate() {
            let fdpar = FdPar {
                basis_type: basis_type.clone(),
                nbasis,
                lambda: lam,
                lfd_order: 2,
                penalty_matrix: penalty.clone(),
            };

            // Project training curves to basis coefficients
            let train_result = match smooth_basis(&train_data, argvals, &fdpar) {
                Some(r) => r,
                None => {
                    cv_fold_errors[li].push(f64::INFINITY);
                    continue;
                }
            };

            // Project test curves to basis coefficients
            let test_result = match smooth_basis(&test_data, argvals, &fdpar) {
                Some(r) => r,
                None => {
                    cv_fold_errors[li].push(f64::INFINITY);
                    continue;
                }
            };

            // Fit linear model on training coefficients -> y
            let train_coefs = &train_result.coefficients;
            let test_coefs = &test_result.coefficients;
            let n_train = train_idx.len();
            let n_test = test_idx.len();
            let k = train_coefs.ncols();

            // Penalized OLS: (X'X + lam*P) \ X'y
            let mut xtx = vec![0.0; k * k];
            let mut xty_vec = vec![0.0; k];
            for i in 0..n_train {
                for j in 0..k {
                    xty_vec[j] += train_coefs[(i, j)] * train_y[i];
                    for l in 0..k {
                        xtx[j * k + l] += train_coefs[(i, j)] * train_coefs[(i, l)];
                    }
                }
            }
            // Add roughness penalty: lam * P (not ridge lam * I)
            for j in 0..k {
                for l in 0..k {
                    xtx[j * k + l] += lam * penalty[j * k + l];
                }
            }

            // Solve via Gaussian elimination
            let beta = {
                let mut a = xtx;
                let mut b = xty_vec;
                crate::smoothing::solve_gaussian_pub(&mut a, &mut b, k)
            };

            // Predict on test set
            let mut fold_mse = 0.0;
            for i in 0..n_test {
                let mut yhat = 0.0;
                for j in 0..k {
                    yhat += test_coefs[(i, j)] * beta[j];
                }
                fold_mse += (test_y[i] - yhat).powi(2);
            }
            fold_mse /= n_test as f64;
            cv_fold_errors[li].push(fold_mse);
        }
    }

    // Compute mean and SE across folds
    let mut cv_se = vec![0.0; lambdas.len()];
    for li in 0..lambdas.len() {
        let errs = &cv_fold_errors[li];
        if errs.is_empty() {
            cv_errors[li] = f64::INFINITY;
            continue;
        }
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        cv_errors[li] = mean;
        if errs.len() > 1 {
            let var =
                errs.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (errs.len() - 1) as f64;
            cv_se[li] = (var / errs.len() as f64).sqrt();
        }
    }

    let (best_idx, &min_cv_error) = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    Some(FregreBasisCvResult {
        optimal_lambda: lambdas[best_idx],
        cv_errors,
        cv_se,
        lambda_values: lambdas.to_vec(),
        min_cv_error,
    })
}

// ---------------------------------------------------------------------------
// Nonparametric regression bandwidth CV (R's fregre.np.cv)
// ---------------------------------------------------------------------------

/// Result of bandwidth selection for nonparametric regression via CV.
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

/// K-fold CV for selecting the bandwidth in functional nonparametric
/// regression (R's `fregre.np.cv`).
///
/// Computes a full n×n L2 distance matrix once, then for each candidate
/// bandwidth and each fold, does Nadaraya-Watson prediction.
///
/// # Arguments
/// * `data` — Functional predictor matrix (n × m)
/// * `y` — Scalar response vector (length n)
/// * `argvals` — Evaluation grid (length m)
/// * `n_folds` — Number of CV folds
/// * `h_range` — Bandwidth values to test. If `None`, uses 20 quantiles
///   (5th to 95th percentile) of the pairwise L2 distances.
/// * `scalar_covariates` — Optional scalar covariates (n × p)
pub fn fregre_np_cv(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    n_folds: usize,
    h_range: Option<&[f64]>,
    scalar_covariates: Option<&FdMatrix>,
) -> Option<FregreNpCvResult> {
    let (n, m) = data.shape();
    if n < n_folds || m == 0 || y.len() != n || argvals.len() != m || n < 3 {
        return None;
    }

    // Compute full distance matrix
    let func_dists = compute_pairwise_distances(data, argvals);
    let has_scalar = scalar_covariates.is_some();
    let scalar_dists = scalar_covariates
        .map(compute_scalar_distances)
        .unwrap_or_default();

    // Default h grid: 20 quantiles of pairwise distances
    let default_h: Vec<f64> = if h_range.is_none() {
        let mut nonzero: Vec<f64> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let d = func_dists[i * n + j];
                if d > 0.0 {
                    nonzero.push(d);
                }
            }
        }
        nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if nonzero.is_empty() {
            return None;
        }
        (1..=20)
            .map(|qi| {
                let q = 0.05 + 0.90 * (qi - 1) as f64 / 19.0;
                let idx = ((nonzero.len() as f64 * q) as usize).min(nonzero.len() - 1);
                nonzero[idx].max(1e-10)
            })
            .collect()
    } else {
        Vec::new()
    };
    let h_values = h_range.unwrap_or(&default_h);

    if h_values.is_empty() {
        return None;
    }

    let folds = crate::cv::create_folds(n, n_folds, 42);

    let mut cv_errors = vec![0.0; h_values.len()];
    let mut cv_fold_errors: Vec<Vec<f64>> = vec![Vec::with_capacity(n_folds); h_values.len()];

    // Select scalar bandwidth once (from full data)
    let h_scalar = if has_scalar {
        select_bandwidth_loo(&scalar_dists, y, n, Some(&func_dists))
    } else {
        1.0
    };

    for fold in 0..n_folds {
        let (train_idx, test_idx) = crate::cv::fold_indices(&folds, fold);
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }

        let train_y: Vec<f64> = train_idx.iter().map(|&i| y[i]).collect();

        for (hi, &h) in h_values.iter().enumerate() {
            let mut fold_mse = 0.0;
            for &ti in &test_idx {
                // NW prediction using only training data
                let mut num = 0.0;
                let mut den = 0.0;
                for (local_j, &tj) in train_idx.iter().enumerate() {
                    let kf = gaussian_kernel(func_dists[ti * n + tj], h);
                    let ks = if has_scalar {
                        gaussian_kernel(scalar_dists[ti * n + tj], h_scalar)
                    } else {
                        1.0
                    };
                    let w = kf * ks;
                    num += w * train_y[local_j];
                    den += w;
                }
                let y_hat = if den > 1e-15 { num / den } else { y[ti] };
                fold_mse += (y[ti] - y_hat).powi(2);
            }
            fold_mse /= test_idx.len() as f64;
            cv_fold_errors[hi].push(fold_mse);
        }
    }

    // Compute mean and SE
    let mut cv_se = vec![0.0; h_values.len()];
    for hi in 0..h_values.len() {
        let errs = &cv_fold_errors[hi];
        if errs.is_empty() {
            cv_errors[hi] = f64::INFINITY;
            continue;
        }
        let mean = errs.iter().sum::<f64>() / errs.len() as f64;
        cv_errors[hi] = mean;
        if errs.len() > 1 {
            let var =
                errs.iter().map(|&e| (e - mean).powi(2)).sum::<f64>() / (errs.len() - 1) as f64;
            cv_se[hi] = (var / errs.len() as f64).sqrt();
        }
    }

    let (best_idx, &min_cv_error) = cv_errors
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))?;

    Some(FregreNpCvResult {
        optimal_h: h_values[best_idx],
        cv_errors,
        cv_se,
        h_values: h_values.to_vec(),
        min_cv_error,
    })
}

/// Predict probabilities P(Y=1) for new data using a fitted functional logistic model.
///
/// Projects new curves through the stored FPCA, computes linear predictor,
/// and applies sigmoid.
///
/// # Arguments
/// * `fit` — A fitted [`FunctionalLogisticResult`]
/// * `new_data` — New functional predictor matrix (n_new × m)
/// * `new_scalar` — Optional new scalar covariates (n_new × p)
pub fn predict_functional_logistic(
    fit: &FunctionalLogisticResult,
    new_data: &FdMatrix,
    new_scalar: Option<&FdMatrix>,
) -> Vec<f64> {
    let (n_new, m) = new_data.shape();
    let ncomp = fit.ncomp;
    let p_scalar = fit.gamma.len();

    (0..n_new)
        .map(|i| {
            let mut eta = fit.coefficients[0]; // intercept
            for k in 0..ncomp {
                let mut s = 0.0;
                for j in 0..m {
                    s += (new_data[(i, j)] - fit.fpca.mean[j]) * fit.fpca.rotation[(j, k)];
                }
                eta += fit.coefficients[1 + k] * s;
            }
            if let Some(sc) = new_scalar {
                for j in 0..p_scalar {
                    eta += fit.gamma[j] * sc[(i, j)];
                }
            }
            sigmoid(eta)
        })
        .collect()
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

impl FunctionalLogisticResult {
    /// Predict P(Y=1) for new data. Delegates to [`predict_functional_logistic`].
    pub fn predict(&self, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64> {
        predict_functional_logistic(self, new_data, new_scalar)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize, seed: u64) -> (FdMatrix, Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];

        for i in 0..n {
            let phase =
                (seed.wrapping_mul(17).wrapping_add(i as u64 * 31) % 1000) as f64 / 1000.0 * PI;
            let amplitude =
                ((seed.wrapping_mul(13).wrapping_add(i as u64 * 7) % 100) as f64 / 100.0) - 0.5;
            for j in 0..m {
                data[(i, j)] =
                    (2.0 * PI * t[j] + phase).sin() + amplitude * (4.0 * PI * t[j]).cos();
            }
            y[i] = 2.0 * phase + 3.0 * amplitude + 0.05 * (seed.wrapping_add(i as u64) % 10) as f64;
        }
        (data, y, t)
    }

    // ----- fregre_lm tests -----

    #[test]
    fn test_fregre_lm_basic() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let result = fregre_lm(&data, &y, None, 3);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.fitted_values.len(), 30);
        assert_eq!(fit.residuals.len(), 30);
        assert_eq!(fit.beta_t.len(), 50);
        assert_eq!(fit.ncomp, 3);
        assert!(fit.r_squared >= 0.0 && fit.r_squared <= 1.0 + 1e-10);
    }

    #[test]
    fn test_fregre_lm_with_scalar_covariates() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let mut sc = FdMatrix::zeros(30, 2);
        for i in 0..30 {
            sc[(i, 0)] = i as f64 / 30.0;
            sc[(i, 1)] = (i as f64 * 0.7).sin();
        }
        let result = fregre_lm(&data, &y, Some(&sc), 3);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.gamma.len(), 2);
    }

    #[test]
    fn test_fregre_lm_residuals_sum_near_zero() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let resid_sum: f64 = fit.residuals.iter().sum::<f64>();
        assert!(
            resid_sum.abs() < 1e-8,
            "Residuals should sum to ~0 with intercept, got {}",
            resid_sum
        );
    }

    #[test]
    fn test_fregre_lm_fitted_plus_residuals_equals_y() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        for i in 0..30 {
            let reconstructed = fit.fitted_values[i] + fit.residuals[i];
            assert!(
                (reconstructed - y[i]).abs() < 1e-10,
                "ŷ + r should equal y at index {}",
                i
            );
        }
    }

    #[test]
    fn test_fregre_lm_more_components_higher_r2() {
        let (data, y, _t) = generate_test_data(50, 50, 42);
        let fit1 = fregre_lm(&data, &y, None, 1).unwrap();
        let fit3 = fregre_lm(&data, &y, None, 3).unwrap();
        assert!(
            fit3.r_squared >= fit1.r_squared - 1e-10,
            "More components should give >= R²: {} vs {}",
            fit3.r_squared,
            fit1.r_squared
        );
    }

    #[test]
    fn test_fregre_lm_invalid_input() {
        let data = FdMatrix::zeros(2, 50);
        let y = vec![1.0, 2.0];
        assert!(fregre_lm(&data, &y, None, 1).is_none());

        let data = FdMatrix::zeros(10, 50);
        let y = vec![1.0; 5];
        assert!(fregre_lm(&data, &y, None, 2).is_none());
    }

    #[test]
    fn test_fregre_lm_std_errors_positive() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        for (i, &se) in fit.std_errors.iter().enumerate() {
            assert!(
                se > 0.0 && se.is_finite(),
                "Std error {} should be positive finite, got {}",
                i,
                se
            );
        }
    }

    // ----- predict_fregre_lm tests -----

    #[test]
    fn test_predict_fregre_lm_on_training_data() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        let preds = predict_fregre_lm(&fit, &data, None);
        for i in 0..30 {
            assert!(
                (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
                "Prediction on training data should match fitted values"
            );
        }
    }

    // ----- fregre_cv tests -----

    #[test]
    fn test_fregre_cv_returns_result() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let result = fregre_cv(&data, &y, None, 1, 8, 5);
        assert!(result.is_some());
        let cv = result.unwrap();
        assert!(cv.optimal_k >= 1 && cv.optimal_k <= 8);
        assert!(cv.min_cv_error >= 0.0);
    }

    #[test]
    fn test_fregre_cv_with_scalar_covariates() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let mut sc = FdMatrix::zeros(30, 1);
        for i in 0..30 {
            sc[(i, 0)] = i as f64;
        }
        let result = fregre_cv(&data, &y, Some(&sc), 1, 5, 3);
        assert!(result.is_some());
    }

    // ----- fregre_np_mixed tests -----

    #[test]
    fn test_fregre_np_mixed_basic() {
        let (data, y, t) = generate_test_data(30, 50, 42);
        let result = fregre_np_mixed(&data, &y, &t, None, 0.0, 0.0);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.fitted_values.len(), 30);
        assert!(fit.h_func > 0.0);
        assert!(fit.cv_error >= 0.0);
    }

    #[test]
    fn test_fregre_np_mixed_with_scalars() {
        let (data, y, t) = generate_test_data(30, 50, 42);
        let mut sc = FdMatrix::zeros(30, 1);
        for i in 0..30 {
            sc[(i, 0)] = i as f64 / 30.0;
        }
        let result = fregre_np_mixed(&data, &y, &t, Some(&sc), 0.0, 0.0);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert!(fit.h_scalar > 0.0);
    }

    #[test]
    fn test_fregre_np_mixed_manual_bandwidth() {
        let (data, y, t) = generate_test_data(30, 50, 42);
        let result = fregre_np_mixed(&data, &y, &t, None, 0.5, 0.0);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert!(
            (fit.h_func - 0.5).abs() < 1e-10,
            "Should use provided bandwidth"
        );
    }

    // ----- functional_logistic tests -----

    #[test]
    fn test_functional_logistic_basic() {
        let (data, y_cont, _t) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let result = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.probabilities.len(), 30);
        assert_eq!(fit.predicted_classes.len(), 30);
        assert!(fit.accuracy >= 0.0 && fit.accuracy <= 1.0);
        for &p in &fit.probabilities {
            assert!((0.0..=1.0).contains(&p), "Probability out of range: {}", p);
        }
    }

    #[test]
    fn test_functional_logistic_with_scalar_covariates() {
        let (data, y_cont, _t) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let mut sc = FdMatrix::zeros(30, 1);
        for i in 0..30 {
            sc[(i, 0)] = i as f64 / 30.0;
        }
        let result = functional_logistic(&data, &y_bin, Some(&sc), 3, 25, 1e-6);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.gamma.len(), 1);
    }

    #[test]
    fn test_functional_logistic_invalid_response() {
        let (data, _, _) = generate_test_data(30, 50, 42);
        let y: Vec<f64> = (0..30).map(|i| i as f64).collect();
        assert!(functional_logistic(&data, &y, None, 3, 25, 1e-6).is_none());
    }

    #[test]
    fn test_functional_logistic_convergence() {
        let (data, y_cont, _t) = generate_test_data(40, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let fit = functional_logistic(&data, &y_bin, None, 3, 100, 1e-8).unwrap();
        assert!(fit.iterations <= 100, "Should converge within max_iter");
    }

    // ----- Edge cases -----

    #[test]
    fn test_fregre_lm_single_component() {
        let (data, y, _t) = generate_test_data(20, 50, 42);
        let result = fregre_lm(&data, &y, None, 1);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.ncomp, 1);
    }

    #[test]
    fn test_fregre_lm_auto_k_selection() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let result = fregre_lm(&data, &y, None, 0);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert!(fit.ncomp >= 1);
    }

    #[test]
    fn test_predict_fregre_np_basic() {
        let (data, y, t) = generate_test_data(30, 50, 42);
        let preds = predict_fregre_np(&data, &y, None, &data, None, &t, 0.5, 0.5);
        assert_eq!(preds.len(), 30);
        for &p in &preds {
            assert!(p.is_finite(), "Prediction should be finite");
        }
    }

    #[test]
    fn test_sigmoid_properties() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
        assert!((sigmoid(3.0) + sigmoid(-3.0) - 1.0).abs() < 1e-10);
    }

    // ----- beta_se tests -----

    #[test]
    fn test_fregre_lm_beta_se() {
        let (data, y, _t) = generate_test_data(30, 50, 42);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert_eq!(fit.beta_se.len(), 50, "beta_se should have length m");
        for (j, &se) in fit.beta_se.iter().enumerate() {
            assert!(
                se > 0.0 && se.is_finite(),
                "beta_se[{}] should be positive finite, got {}",
                j,
                se
            );
        }
    }

    #[test]
    fn test_fregre_lm_beta_se_coverage() {
        // Use generate_test_data which is known to produce valid FPCA, then check SE properties
        let (data, y, _t) = generate_test_data(50, 50, 99);
        let fit = fregre_lm(&data, &y, None, 3).unwrap();
        assert_eq!(fit.beta_se.len(), 50);
        // With valid data, beta_se should be positive everywhere
        for (j, &se) in fit.beta_se.iter().enumerate() {
            assert!(
                se > 0.0 && se.is_finite(),
                "beta_se[{}] should be positive finite, got {}",
                j,
                se
            );
        }
        // The CI band [beta_t ± 2·SE] should have non-zero width everywhere
        for j in 0..50 {
            let width = 4.0 * fit.beta_se[j];
            assert!(width > 0.0, "CI width should be positive at j={}", j);
        }
    }

    #[test]
    fn test_functional_logistic_beta_se() {
        let (data, y_cont, _t) = generate_test_data(30, 50, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let fit = functional_logistic(&data, &y_bin, None, 3, 25, 1e-6).unwrap();
        assert_eq!(fit.beta_se.len(), 50, "beta_se should have length m");
        assert_eq!(
            fit.std_errors.len(),
            1 + 3,
            "std_errors should have length 1 + ncomp"
        );
        for (j, &se) in fit.beta_se.iter().enumerate() {
            assert!(
                se > 0.0 && se.is_finite(),
                "beta_se[{}] should be positive finite, got {}",
                j,
                se
            );
        }
        for (j, &se) in fit.std_errors.iter().enumerate() {
            assert!(
                se > 0.0 && se.is_finite(),
                "std_errors[{}] should be positive finite, got {}",
                j,
                se
            );
        }
    }

    #[test]
    fn test_beta_se_zero_for_constant() {
        // When all curves are nearly identical, β(t) ≈ 0 but SE should still be finite/positive
        let n = 30;
        let m = 20;
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..m {
                // Nearly identical curves with tiny variation
                data[(i, j)] = 1.0 + 0.001 * (i as f64 / n as f64);
            }
            y[i] = i as f64 / n as f64;
        }
        let fit = fregre_lm(&data, &y, None, 1).unwrap();
        assert_eq!(fit.beta_se.len(), m);
        for (j, &se) in fit.beta_se.iter().enumerate() {
            assert!(
                se.is_finite() && se >= 0.0,
                "beta_se[{}] should be finite non-negative, got {}",
                j,
                se
            );
        }
    }

    // ----- Bootstrap CI tests -----

    #[test]
    fn test_bootstrap_ci_fregre_lm_shape() {
        let (data, y, _t) = generate_test_data(30, 20, 42);
        let result = bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 123);
        assert!(result.is_some(), "bootstrap_ci_fregre_lm should succeed");
        let ci = result.unwrap();
        assert_eq!(ci.lower.len(), 20);
        assert_eq!(ci.upper.len(), 20);
        assert_eq!(ci.center.len(), 20);
        assert_eq!(ci.sim_lower.len(), 20);
        assert_eq!(ci.sim_upper.len(), 20);
        assert!(ci.n_boot_success > 0);
    }

    #[test]
    fn test_bootstrap_ci_fregre_lm_ordering() {
        let (data, y, _t) = generate_test_data(30, 20, 42);
        let ci = bootstrap_ci_fregre_lm(&data, &y, None, 2, 100, 0.05, 42).unwrap();
        for j in 0..20 {
            // Pointwise: lower ≤ center ≤ upper
            assert!(
                ci.lower[j] <= ci.center[j] + 1e-10,
                "lower <= center at j={}: {} > {}",
                j,
                ci.lower[j],
                ci.center[j]
            );
            assert!(
                ci.center[j] <= ci.upper[j] + 1e-10,
                "center <= upper at j={}: {} > {}",
                j,
                ci.center[j],
                ci.upper[j]
            );
            // Simultaneous: sim_lower ≤ center ≤ sim_upper
            assert!(
                ci.sim_lower[j] <= ci.center[j] + 1e-10,
                "sim_lower <= center at j={}: {} > {}",
                j,
                ci.sim_lower[j],
                ci.center[j]
            );
            assert!(
                ci.center[j] <= ci.sim_upper[j] + 1e-10,
                "center <= sim_upper at j={}: {} > {}",
                j,
                ci.center[j],
                ci.sim_upper[j]
            );
        }
        // Simultaneous band should be wider on average
        let pw_width: f64 = (0..20).map(|j| ci.upper[j] - ci.lower[j]).sum::<f64>() / 20.0;
        let sim_width: f64 = (0..20)
            .map(|j| ci.sim_upper[j] - ci.sim_lower[j])
            .sum::<f64>()
            / 20.0;
        assert!(
            sim_width >= pw_width - 1e-10,
            "Simultaneous band should be wider on average: sim={}, pw={}",
            sim_width,
            pw_width
        );
    }

    #[test]
    fn test_bootstrap_ci_fregre_lm_center_matches_fit() {
        let (data, y, _t) = generate_test_data(30, 20, 42);
        let fit = fregre_lm(&data, &y, None, 2).unwrap();
        let ci = bootstrap_ci_fregre_lm(&data, &y, None, 2, 50, 0.05, 42).unwrap();
        for j in 0..20 {
            assert!(
                (ci.center[j] - fit.beta_t[j]).abs() < 1e-12,
                "center should match original beta_t at j={}",
                j
            );
        }
    }

    #[test]
    fn test_bootstrap_ci_functional_logistic_shape() {
        let (data, y_cont, _t) = generate_test_data(40, 20, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let result =
            bootstrap_ci_functional_logistic(&data, &y_bin, None, 2, 50, 0.05, 123, 25, 1e-6);
        assert!(
            result.is_some(),
            "bootstrap_ci_functional_logistic should succeed"
        );
        let ci = result.unwrap();
        assert_eq!(ci.lower.len(), 20);
        assert_eq!(ci.upper.len(), 20);
        assert_eq!(ci.center.len(), 20);
        assert!(ci.n_boot_success > 0);
    }

    #[test]
    fn test_bootstrap_ci_logistic_ordering() {
        let (data, y_cont, _t) = generate_test_data(40, 20, 42);
        let y_median = {
            let mut sorted = y_cont.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };
        let y_bin: Vec<f64> = y_cont
            .iter()
            .map(|&v| if v >= y_median { 1.0 } else { 0.0 })
            .collect();

        let ci = bootstrap_ci_functional_logistic(&data, &y_bin, None, 2, 100, 0.05, 42, 25, 1e-6)
            .unwrap();
        for j in 0..20 {
            assert!(
                ci.lower[j] <= ci.upper[j] + 1e-10,
                "lower <= upper at j={}",
                j
            );
        }
    }

    // ----- fregre_basis_cv tests -----

    #[test]
    fn test_fregre_basis_cv_returns_result() {
        let (data, y, t) = generate_test_data(30, 20, 42);
        let result = fregre_basis_cv(
            &data,
            &y,
            &t,
            5,
            None,
            7,
            &crate::smooth_basis::BasisType::Bspline { order: 4 },
        );
        assert!(result.is_some(), "fregre_basis_cv should succeed");
        let res = result.unwrap();
        assert!(res.optimal_lambda > 0.0);
        assert_eq!(res.cv_errors.len(), 20); // default 20 lambdas
        assert!(res.min_cv_error >= 0.0);
    }

    #[test]
    fn test_fregre_basis_cv_custom_lambdas() {
        let (data, y, t) = generate_test_data(25, 15, 42);
        let lambdas = vec![0.001, 0.01, 0.1, 1.0, 10.0];
        let result = fregre_basis_cv(
            &data,
            &y,
            &t,
            5,
            Some(&lambdas),
            7,
            &crate::smooth_basis::BasisType::Bspline { order: 4 },
        );
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.lambda_values.len(), 5);
        assert!(lambdas.contains(&res.optimal_lambda));
    }

    // ----- fregre_np_cv tests -----

    #[test]
    fn test_fregre_np_cv_returns_result() {
        let (data, y, t) = generate_test_data(25, 15, 42);
        let result = fregre_np_cv(&data, &y, &t, 5, None, None);
        assert!(result.is_some(), "fregre_np_cv should succeed");
        let res = result.unwrap();
        assert!(res.optimal_h > 0.0);
        assert_eq!(res.cv_errors.len(), 20); // default 20 quantiles
        assert!(res.min_cv_error >= 0.0);
    }

    #[test]
    fn test_fregre_np_cv_custom_h() {
        let (data, y, t) = generate_test_data(20, 10, 42);
        let h_vals = vec![0.1, 0.5, 1.0, 2.0];
        let result = fregre_np_cv(&data, &y, &t, 3, Some(&h_vals), None);
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.h_values.len(), 4);
        assert!(h_vals.contains(&res.optimal_h));
    }
}
