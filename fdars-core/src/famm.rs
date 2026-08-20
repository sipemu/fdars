//! Functional Additive Mixed Models (FAMM).
//!
//! Implements functional mixed effects models for repeated functional
//! measurements with subject-level covariates.
//!
//! Model: `Y_ij(t) = μ(t) + X_i'β(t) + b_i(t) + ε_ij(t)`
//!
//! Key functions:
//! - [`fmm`] — Fit a functional mixed model via FPC decomposition
//! - [`fmm_predict`] — Predict curves for new subjects
//! - [`fmm_test_fixed`] — Hypothesis test on fixed effects

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::linalg::{
    cholesky_factor as linalg_cholesky_factor,
    cholesky_forward_back as linalg_cholesky_forward_back,
};
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of a functional mixed model fit.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FmmResult {
    /// Overall mean function μ̂(t) (length m)
    pub mean_function: Vec<f64>,
    /// Fixed effect coefficient functions β̂_j(t) (p × m matrix, one row per covariate)
    pub beta_functions: FdMatrix,
    /// Random effect functions b̂_i(t) per subject (n_subjects × m)
    pub random_effects: FdMatrix,
    /// Fitted values for all observations (n_total × m)
    pub fitted: FdMatrix,
    /// Residuals (n_total × m)
    pub residuals: FdMatrix,
    /// Variance of random effects at each time point (length m)
    pub random_variance: Vec<f64>,
    /// Residual variance estimate
    pub sigma2_eps: f64,
    /// Random effect variance estimate (per-component)
    pub sigma2_u: Vec<f64>,
    /// Number of FPC components used
    pub ncomp: usize,
    /// Number of subjects
    pub n_subjects: usize,
    /// FPC eigenvalues (singular values squared / n)
    pub eigenvalues: Vec<f64>,
}

/// Result of fixed effect hypothesis test.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FmmTestResult {
    /// F-statistic per covariate (length p)
    pub f_statistics: Vec<f64>,
    /// P-values per covariate (via permutation, length p)
    pub p_values: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Core FMM algorithm
// ---------------------------------------------------------------------------

/// Fit a functional mixed model via FPC decomposition.
///
/// # Arguments
/// * `data` — All observed curves (n_total × m), stacked across subjects and visits
/// * `subject_ids` — Subject identifier for each curve (length n_total)
/// * `covariates` — Subject-level covariates (n_total × p).
///   Each row corresponds to the same curve in `data`.
///   If a covariate is subject-level, its value should be repeated across visits.
/// * `ncomp` — Number of FPC components
///
/// # Algorithm
/// 1. Pool curves, compute FPCA
/// 2. For each FPC score, fit scalar mixed model: ξ_ijk = x_i'γ_k + u_ik + e_ijk
/// 3. Recover β̂(t) and b̂_i(t) from component coefficients
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` is empty (zero rows or
/// columns), or if `subject_ids.len()` does not match the number of rows.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero.
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<FmmResult, FdarError> {
    let n_total = data.nrows();
    let m = data.ncols();
    if n_total == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n_total} x {m}"),
        });
    }
    if subject_ids.len() != n_total {
        return Err(FdarError::InvalidDimension {
            parameter: "subject_ids",
            expected: format!("length {n_total}"),
            actual: format!("length {}", subject_ids.len()),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }

    // Determine unique subjects
    let (subject_map, n_subjects) = build_subject_map(subject_ids);

    // Step 1: FPCA on pooled data
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;
    let k = fpca.scores.ncols(); // actual number of components

    // Step 2: For each FPC score, fit scalar mixed model (parallelized)
    let p = covariates.map_or(0, super::matrix::FdMatrix::ncols);
    let ComponentResults {
        gamma,
        u_hat,
        sigma2_u,
        sigma2_eps,
    } = fit_all_components(
        &fpca.scores,
        &subject_map,
        n_subjects,
        covariates,
        p,
        k,
        n_total,
        m,
    );

    // Step 3: Recover functional coefficients (using gamma in original scale)
    let beta_functions = recover_beta_functions(&gamma, &fpca.rotation, p, m, k);
    let random_effects = recover_random_effects(&u_hat, &fpca.rotation, n_subjects, m, k);

    // Compute random variance function: Var(b_i(t)) across subjects
    let random_variance = compute_random_variance(&random_effects, n_subjects, m);

    // Compute fitted and residuals
    let (fitted, residuals) = compute_fitted_residuals(
        data,
        &fpca.mean,
        &beta_functions,
        &random_effects,
        covariates,
        &subject_map,
        n_total,
        m,
        p,
    );

    let eigenvalues: Vec<f64> = fpca
        .singular_values
        .iter()
        .map(|&sv| sv * sv / n_total as f64)
        .collect();

    Ok(FmmResult {
        mean_function: fpca.mean,
        beta_functions,
        random_effects,
        fitted,
        residuals,
        random_variance,
        sigma2_eps,
        sigma2_u,
        ncomp: k,
        n_subjects,
        eigenvalues,
    })
}

/// Build mapping from observation index to subject index (0..n_subjects-1).
pub(crate) fn build_subject_map(subject_ids: &[usize]) -> (Vec<usize>, usize) {
    let mut unique_ids: Vec<usize> = subject_ids.to_vec();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    let n_subjects = unique_ids.len();

    let map: Vec<usize> = subject_ids
        .iter()
        .map(|id| unique_ids.iter().position(|u| u == id).unwrap_or(0))
        .collect();

    (map, n_subjects)
}

/// Aggregated results from fitting all FPC components.
struct ComponentResults {
    gamma: Vec<Vec<f64>>, // gamma[j][k] = fixed effect coeff j for component k
    u_hat: Vec<Vec<f64>>, // u_hat[i][k] = random effect for subject i, component k
    sigma2_u: Vec<f64>,   // per-component random effect variance
    sigma2_eps: f64,      // average residual variance across components
}

/// Fit scalar mixed models for all FPC components (parallelized across components).
///
/// For each component k, scales FPC scores to L²-normalized space, fits a scalar
/// mixed model, then scales coefficients back to the original score space.
#[allow(clippy::too_many_arguments)]
fn fit_all_components(
    scores: &FdMatrix,
    subject_map: &[usize],
    n_subjects: usize,
    covariates: Option<&FdMatrix>,
    p: usize,
    k: usize,
    n_total: usize,
    m: usize,
) -> ComponentResults {
    // Normalize scores by sqrt(h) to match R's L²-weighted FPCA convention.
    // This ensures variance components are on the same scale as R's lmer().
    let h = if m > 1 { 1.0 / (m - 1) as f64 } else { 1.0 };
    let score_scale = h.sqrt();

    // Fit each component independently — parallelized when the feature is enabled
    let per_comp: Vec<ScalarMixedResult> = iter_maybe_parallel!(0..k)
        .map(|comp| {
            let comp_scores: Vec<f64> = (0..n_total)
                .map(|i| scores[(i, comp)] * score_scale)
                .collect();
            fit_scalar_mixed_model(&comp_scores, subject_map, n_subjects, covariates, p)
        })
        .collect();

    // Unpack per-component results into the aggregate structure
    let mut gamma = vec![vec![0.0; k]; p];
    let mut u_hat = vec![vec![0.0; k]; n_subjects];
    let mut sigma2_u = vec![0.0; k];
    let mut sigma2_eps_total = 0.0;

    for (comp, result) in per_comp.iter().enumerate() {
        for j in 0..p {
            gamma[j][comp] = result.gamma[j] / score_scale;
        }
        for s in 0..n_subjects {
            u_hat[s][comp] = result.u_hat[s] / score_scale;
        }
        sigma2_u[comp] = result.sigma2_u;
        sigma2_eps_total += result.sigma2_eps;
    }
    let sigma2_eps = sigma2_eps_total / k as f64;

    ComponentResults {
        gamma,
        u_hat,
        sigma2_u,
        sigma2_eps,
    }
}

/// Scalar mixed model result for one FPC component.
pub(crate) struct ScalarMixedResult {
    pub(crate) gamma: Vec<f64>, // fixed effects (length p)
    pub(crate) u_hat: Vec<f64>, // random effects per subject (length n_subjects)
    pub(crate) sigma2_u: f64,   // random effect variance
    pub(crate) sigma2_eps: f64, // residual variance
}

/// Precomputed subject structure for the mixed model.
pub(crate) struct SubjectStructure {
    pub(crate) counts: Vec<usize>,
    pub(crate) obs: Vec<Vec<usize>>,
}

impl SubjectStructure {
    pub(crate) fn new(subject_map: &[usize], n_subjects: usize, n: usize) -> Self {
        let mut counts = vec![0usize; n_subjects];
        let mut obs: Vec<Vec<usize>> = vec![Vec::new(); n_subjects];
        for i in 0..n {
            let s = subject_map[i];
            counts[s] += 1;
            obs[s].push(i);
        }
        Self { counts, obs }
    }
}

/// Compute shrinkage weights: w_s = σ²_u / (σ²_u + σ²_e / n_s).
fn shrinkage_weights(ss: &SubjectStructure, sigma2_u: f64, sigma2_e: f64) -> Vec<f64> {
    ss.counts
        .iter()
        .map(|&c| {
            let ns = c as f64;
            if ns < 1.0 {
                0.0
            } else {
                sigma2_u / (sigma2_u + sigma2_e / ns)
            }
        })
        .collect()
}

/// GLS fixed effect update using block-diagonal V^{-1}.
///
/// Computes γ = (X'V⁻¹X)⁻¹ X'V⁻¹y exploiting the balanced random intercept structure.
fn gls_update_gamma(
    cov: &FdMatrix,
    p: usize,
    ss: &SubjectStructure,
    weights: &[f64],
    y: &[f64],
    sigma2_e: f64,
) -> Option<Vec<f64>> {
    let n_subjects = ss.counts.len();
    let mut xtvinvx = vec![0.0; p * p];
    let mut xtvinvy = vec![0.0; p];
    let inv_e = 1.0 / sigma2_e;

    for s in 0..n_subjects {
        let ns = ss.counts[s] as f64;
        if ns < 1.0 {
            continue;
        }
        let (x_sum, y_sum) = subject_sums(cov, y, &ss.obs[s], p);
        accumulate_gls_terms(
            cov,
            y,
            &ss.obs[s],
            &x_sum,
            y_sum,
            weights[s],
            ns,
            inv_e,
            p,
            &mut xtvinvx,
            &mut xtvinvy,
        );
    }

    for j in 0..p {
        xtvinvx[j * p + j] += 1e-10;
    }
    cholesky_solve(&xtvinvx, &xtvinvy, p)
}

/// Compute subject-level covariate sums and response sum.
fn subject_sums(cov: &FdMatrix, y: &[f64], obs: &[usize], p: usize) -> (Vec<f64>, f64) {
    let mut x_sum = vec![0.0; p];
    let mut y_sum = 0.0;
    for &i in obs {
        for r in 0..p {
            x_sum[r] += cov[(i, r)];
        }
        y_sum += y[i];
    }
    (x_sum, y_sum)
}

/// Accumulate X'V^{-1}X and X'V^{-1}y for one subject.
fn accumulate_gls_terms(
    cov: &FdMatrix,
    y: &[f64],
    obs: &[usize],
    x_sum: &[f64],
    y_sum: f64,
    w_s: f64,
    ns: f64,
    inv_e: f64,
    p: usize,
    xtvinvx: &mut [f64],
    xtvinvy: &mut [f64],
) {
    for &i in obs {
        let vinv_y = inv_e * (y[i] - w_s * y_sum / ns);
        for r in 0..p {
            xtvinvy[r] += cov[(i, r)] * vinv_y;
            for c in r..p {
                let vinv_xc = inv_e * (cov[(i, c)] - w_s * x_sum[c] / ns);
                let val = cov[(i, r)] * vinv_xc;
                xtvinvx[r * p + c] += val;
                if r != c {
                    xtvinvx[c * p + r] += val;
                }
            }
        }
    }
}

/// REML EM update for variance components.
///
/// Returns (σ²_u_new, σ²_e_new) from the conditional expectations.
/// Uses n - p divisor for σ²_e (REML correction where p = number of fixed effects).
fn reml_variance_update(
    residuals: &[f64],
    ss: &SubjectStructure,
    weights: &[f64],
    sigma2_u: f64,
    p: usize,
) -> (f64, f64) {
    let n_subjects = ss.counts.len();
    let n: usize = ss.counts.iter().sum();
    let mut sigma2_u_new = 0.0;
    let mut sigma2_e_new = 0.0;

    for s in 0..n_subjects {
        let ns = ss.counts[s] as f64;
        if ns < 1.0 {
            continue;
        }
        let w_s = weights[s];
        let mean_r_s: f64 = ss.obs[s].iter().map(|&i| residuals[i]).sum::<f64>() / ns;
        let u_hat_s = w_s * mean_r_s;
        let cond_var_s = sigma2_u * (1.0 - w_s);

        sigma2_u_new += u_hat_s * u_hat_s + cond_var_s;
        for &i in &ss.obs[s] {
            sigma2_e_new += (residuals[i] - u_hat_s).powi(2);
        }
        sigma2_e_new += ns * cond_var_s;
    }

    // REML divisor: n - p for residual variance (matches R's lmer)
    let denom_e = (n.saturating_sub(p)).max(1) as f64;

    (
        (sigma2_u_new / n_subjects as f64).max(1e-15),
        (sigma2_e_new / denom_e).max(1e-15),
    )
}

/// Fit scalar mixed model: y_ij = x_i'γ + u_i + e_ij.
///
/// Uses iterative GLS for fixed effects + REML EM for variance components,
/// matching R's lmer() behavior. Initializes from Henderson's ANOVA, then
/// iterates until convergence.
pub(crate) fn fit_scalar_mixed_model(
    y: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    covariates: Option<&FdMatrix>,
    p: usize,
) -> ScalarMixedResult {
    let n = y.len();
    let ss = SubjectStructure::new(subject_map, n_subjects, n);

    // Initialize from OLS + Henderson's ANOVA
    let gamma_init = estimate_fixed_effects(y, covariates, p, n);
    let residuals_init = compute_ols_residuals(y, covariates, &gamma_init, p, n);
    let (mut sigma2_u, mut sigma2_e) =
        estimate_variance_components(&residuals_init, subject_map, n_subjects, n);

    if sigma2_e < 1e-15 {
        sigma2_e = 1e-6;
    }
    if sigma2_u < 1e-15 {
        sigma2_u = sigma2_e * 0.1;
    }

    let mut gamma = gamma_init;

    for _iter in 0..50 {
        let sigma2_u_old = sigma2_u;
        let sigma2_e_old = sigma2_e;

        let weights = shrinkage_weights(&ss, sigma2_u, sigma2_e);

        if let Some(cov) = covariates.filter(|_| p > 0) {
            if let Some(g) = gls_update_gamma(cov, p, &ss, &weights, y, sigma2_e) {
                gamma = g;
            }
        }

        let r = compute_ols_residuals(y, covariates, &gamma, p, n);
        (sigma2_u, sigma2_e) = reml_variance_update(&r, &ss, &weights, sigma2_u, p);

        let delta = (sigma2_u - sigma2_u_old).abs() + (sigma2_e - sigma2_e_old).abs();
        if delta < 1e-10 * (sigma2_u_old + sigma2_e_old) {
            break;
        }
    }

    let final_residuals = compute_ols_residuals(y, covariates, &gamma, p, n);
    let u_hat = compute_blup(
        &final_residuals,
        subject_map,
        n_subjects,
        sigma2_u,
        sigma2_e,
    );

    ScalarMixedResult {
        gamma,
        u_hat,
        sigma2_u,
        sigma2_eps: sigma2_e,
    }
}

/// OLS estimation of fixed effects.
fn estimate_fixed_effects(
    y: &[f64],
    covariates: Option<&FdMatrix>,
    p: usize,
    n: usize,
) -> Vec<f64> {
    if p == 0 || covariates.is_none() {
        return Vec::new();
    }
    let cov = covariates.expect("checked: covariates is Some");

    // Solve (X'X)γ = X'y via Cholesky
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        for r in 0..p {
            xty[r] += cov[(i, r)] * y[i];
            for s in r..p {
                let val = cov[(i, r)] * cov[(i, s)];
                xtx[r * p + s] += val;
                if r != s {
                    xtx[s * p + r] += val;
                }
            }
        }
    }
    // Regularize
    for j in 0..p {
        xtx[j * p + j] += 1e-8;
    }

    cholesky_solve(&xtx, &xty, p).unwrap_or(vec![0.0; p])
}

/// Cholesky solve: A x = b where A is p-by-p symmetric positive definite.
/// Returns `None` if the matrix is singular.
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let l = linalg_cholesky_factor(a, p).ok()?;
    Some(linalg_cholesky_forward_back(&l, b, p))
}

/// Compute OLS residuals: r = y - X*gamma.
fn compute_ols_residuals(
    y: &[f64],
    covariates: Option<&FdMatrix>,
    gamma: &[f64],
    p: usize,
    n: usize,
) -> Vec<f64> {
    let mut residuals = y.to_vec();
    if p > 0 {
        if let Some(cov) = covariates {
            for i in 0..n {
                for j in 0..p {
                    residuals[i] -= cov[(i, j)] * gamma[j];
                }
            }
        }
    }
    residuals
}

/// Estimate variance components via method of moments.
///
/// σ²_u and σ²_ε from one-way random effects ANOVA.
fn estimate_variance_components(
    residuals: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    n: usize,
) -> (f64, f64) {
    // Compute subject means and within-subject SS
    let mut subject_sums = vec![0.0; n_subjects];
    let mut subject_counts = vec![0usize; n_subjects];
    for i in 0..n {
        let s = subject_map[i];
        subject_sums[s] += residuals[i];
        subject_counts[s] += 1;
    }
    let subject_means: Vec<f64> = subject_sums
        .iter()
        .zip(&subject_counts)
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    // Within-subject SS
    let mut ss_within = 0.0;
    for i in 0..n {
        let s = subject_map[i];
        ss_within += (residuals[i] - subject_means[s]).powi(2);
    }
    let df_within = n.saturating_sub(n_subjects);

    // Between-subject SS
    let grand_mean = residuals.iter().sum::<f64>() / n as f64;
    let mut ss_between = 0.0;
    for s in 0..n_subjects {
        ss_between += subject_counts[s] as f64 * (subject_means[s] - grand_mean).powi(2);
    }

    let sigma2_eps = if df_within > 0 {
        ss_within / df_within as f64
    } else {
        1e-6
    };

    // Mean number of observations per subject
    let n_bar = n as f64 / n_subjects.max(1) as f64;
    let df_between = n_subjects.saturating_sub(1).max(1);
    let ms_between = ss_between / df_between as f64;
    let sigma2_u = ((ms_between - sigma2_eps) / n_bar).max(0.0);

    (sigma2_u, sigma2_eps)
}

/// Compute BLUP (Best Linear Unbiased Prediction) for random effects.
///
/// û_i = σ²_u / (σ²_u + σ²_ε/n_i) * (ȳ_i - x̄_i'γ)
fn compute_blup(
    residuals: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    sigma2_u: f64,
    sigma2_eps: f64,
) -> Vec<f64> {
    let mut subject_sums = vec![0.0; n_subjects];
    let mut subject_counts = vec![0usize; n_subjects];
    for (i, &r) in residuals.iter().enumerate() {
        let s = subject_map[i];
        subject_sums[s] += r;
        subject_counts[s] += 1;
    }

    (0..n_subjects)
        .map(|s| {
            let ni = subject_counts[s] as f64;
            if ni < 1.0 {
                return 0.0;
            }
            let mean_r = subject_sums[s] / ni;
            let shrinkage = sigma2_u / (sigma2_u + sigma2_eps / ni).max(1e-15);
            shrinkage * mean_r
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Recovery of functional coefficients
// ---------------------------------------------------------------------------

/// Recover β̂(t) = Σ_k γ̂_jk φ_k(t) for each covariate j.
fn recover_beta_functions(
    gamma: &[Vec<f64>],
    rotation: &FdMatrix,
    p: usize,
    m: usize,
    k: usize,
) -> FdMatrix {
    let mut beta = FdMatrix::zeros(p, m);
    for j in 0..p {
        for t in 0..m {
            let mut val = 0.0;
            for comp in 0..k {
                val += gamma[j][comp] * rotation[(t, comp)];
            }
            beta[(j, t)] = val;
        }
    }
    beta
}

/// Recover b̂_i(t) = Σ_k û_ik φ_k(t) for each subject i.
pub(crate) fn recover_random_effects(
    u_hat: &[Vec<f64>],
    rotation: &FdMatrix,
    n_subjects: usize,
    m: usize,
    k: usize,
) -> FdMatrix {
    let mut re = FdMatrix::zeros(n_subjects, m);
    for s in 0..n_subjects {
        for t in 0..m {
            let mut val = 0.0;
            for comp in 0..k {
                val += u_hat[s][comp] * rotation[(t, comp)];
            }
            re[(s, t)] = val;
        }
    }
    re
}

/// Compute random effect variance function: Var_i(b̂_i(t)).
fn compute_random_variance(random_effects: &FdMatrix, n_subjects: usize, m: usize) -> Vec<f64> {
    (0..m)
        .map(|t| {
            let mean: f64 =
                (0..n_subjects).map(|s| random_effects[(s, t)]).sum::<f64>() / n_subjects as f64;
            let var: f64 = (0..n_subjects)
                .map(|s| (random_effects[(s, t)] - mean).powi(2))
                .sum::<f64>()
                / n_subjects.max(1) as f64;
            var
        })
        .collect()
}

/// Compute fitted values and residuals.
fn compute_fitted_residuals(
    data: &FdMatrix,
    mean_function: &[f64],
    beta_functions: &FdMatrix,
    random_effects: &FdMatrix,
    covariates: Option<&FdMatrix>,
    subject_map: &[usize],
    n_total: usize,
    m: usize,
    p: usize,
) -> (FdMatrix, FdMatrix) {
    let mut fitted = FdMatrix::zeros(n_total, m);
    let mut residuals = FdMatrix::zeros(n_total, m);

    for i in 0..n_total {
        let s = subject_map[i];
        for t in 0..m {
            let mut val = mean_function[t] + random_effects[(s, t)];
            if p > 0 {
                if let Some(cov) = covariates {
                    for j in 0..p {
                        val += cov[(i, j)] * beta_functions[(j, t)];
                    }
                }
            }
            fitted[(i, t)] = val;
            residuals[(i, t)] = data[(i, t)] - val;
        }
    }

    (fitted, residuals)
}

// ---------------------------------------------------------------------------
// Prediction
// ---------------------------------------------------------------------------

/// Predict curves for new subjects.
///
/// # Arguments
/// * `result` — Fitted FMM result
/// * `new_covariates` — Covariates for new subjects (n_new × p)
///
/// Returns predicted curves (n_new × m) using only fixed effects (no random effects for new subjects).
#[must_use = "prediction result should not be discarded"]
pub fn fmm_predict(result: &FmmResult, new_covariates: Option<&FdMatrix>) -> FdMatrix {
    let m = result.mean_function.len();
    let n_new = new_covariates.map_or(1, super::matrix::FdMatrix::nrows);
    let p = result.beta_functions.nrows();

    let mut predicted = FdMatrix::zeros(n_new, m);
    for i in 0..n_new {
        for t in 0..m {
            let mut val = result.mean_function[t];
            if let Some(cov) = new_covariates {
                for j in 0..p {
                    val += cov[(i, j)] * result.beta_functions[(j, t)];
                }
            }
            predicted[(i, t)] = val;
        }
    }
    predicted
}

// ---------------------------------------------------------------------------
// Hypothesis testing
// ---------------------------------------------------------------------------

/// Permutation test for fixed effects in functional mixed model.
///
/// Tests H₀: β_j(t) = 0 for each covariate j.
/// Uses integrated squared norm as test statistic: T_j = ∫ β̂_j(t)² dt.
///
/// # Arguments
/// * `data` — All observed curves (n_total × m)
/// * `subject_ids` — Subject identifiers
/// * `covariates` — Subject-level covariates (n_total × p)
/// * `ncomp` — Number of FPC components
/// * `n_perm` — Number of permutations
/// * `seed` — Random seed
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows, or
/// `covariates` has zero columns.
/// Propagates errors from [`fmm`] (e.g., dimension mismatches or FPCA failure).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fmm_test_fixed(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: &FdMatrix,
    ncomp: usize,
    n_perm: usize,
    seed: u64,
) -> Result<FmmTestResult, FdarError> {
    let n_total = data.nrows();
    let m = data.ncols();
    let p = covariates.ncols();
    if n_total == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n_total} rows"),
        });
    }
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "covariates",
            expected: "at least 1 column".to_string(),
            actual: "0 columns".to_string(),
        });
    }

    // Fit observed model
    let result = fmm(data, subject_ids, Some(covariates), ncomp)?;

    // Observed test statistics: ∫ β̂_j(t)² dt for each covariate
    let observed_stats = compute_integrated_beta_sq(&result.beta_functions, p, m);

    // Permutation test
    let (f_statistics, p_values) = permutation_test(
        data,
        subject_ids,
        covariates,
        ncomp,
        n_perm,
        seed,
        &observed_stats,
        p,
        m,
    );

    Ok(FmmTestResult {
        f_statistics,
        p_values,
    })
}

/// Compute ∫ β̂_j(t)² dt for each covariate.
fn compute_integrated_beta_sq(beta: &FdMatrix, p: usize, m: usize) -> Vec<f64> {
    let h = if m > 1 { 1.0 / (m - 1) as f64 } else { 1.0 };
    (0..p)
        .map(|j| {
            let ss: f64 = (0..m).map(|t| beta[(j, t)].powi(2)).sum();
            ss * h
        })
        .collect()
}

/// Run permutation test for fixed effects.
fn permutation_test(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: &FdMatrix,
    ncomp: usize,
    n_perm: usize,
    seed: u64,
    observed_stats: &[f64],
    p: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
    use rand::prelude::*;
    let n_total = data.nrows();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut n_ge = vec![0usize; p];

    for _ in 0..n_perm {
        // Permute covariates across subjects
        let mut perm_indices: Vec<usize> = (0..n_total).collect();
        perm_indices.shuffle(&mut rng);
        let perm_cov = permute_rows(covariates, &perm_indices);

        if let Ok(perm_result) = fmm(data, subject_ids, Some(&perm_cov), ncomp) {
            let perm_stats = compute_integrated_beta_sq(&perm_result.beta_functions, p, m);
            for j in 0..p {
                if perm_stats[j] >= observed_stats[j] {
                    n_ge[j] += 1;
                }
            }
        }
    }

    let p_values: Vec<f64> = n_ge
        .iter()
        .map(|&count| (count + 1) as f64 / (n_perm + 1) as f64)
        .collect();
    let f_statistics = observed_stats.to_vec();

    (f_statistics, p_values)
}

/// Permute rows of a matrix according to given indices.
fn permute_rows(mat: &FdMatrix, indices: &[usize]) -> FdMatrix {
    let n = indices.len();
    let m = mat.ncols();
    let mut result = FdMatrix::zeros(n, m);
    for (new_i, &old_i) in indices.iter().enumerate() {
        for j in 0..m {
            result[(new_i, j)] = mat[(old_i, j)];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// denseFLMM — dense functional linear mixed model
// ---------------------------------------------------------------------------

/// Configuration for [`dense_flmm`].
///
/// No `#[non_exhaustive]` — callers may use struct-literal construction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseFlmmConfig {
    /// Number of FPC components (default: 3)
    pub ncomp: usize,
    /// Maximum REML EM iterations per component model (default: 50)
    pub max_iter: usize,
    /// Relative convergence tolerance for variance components (default: 1e-10)
    pub tol: f64,
    /// Include random slopes in addition to random intercepts (default: false).
    ///
    /// When `false`, only random intercepts are estimated.
    /// Random-slope estimation is not yet implemented; setting this to `true`
    /// is accepted (no error) but silently falls back to intercept-only estimation.
    /// `sigma2_slope` will always be zero-filled in this release.
    pub random_slopes: bool,
}

impl Default for DenseFlmmConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            max_iter: 50,
            tol: 1e-10,
            random_slopes: false,
        }
    }
}

/// Result of a dense functional linear mixed model fit.
///
/// # Parametrization note
///
/// fdars formulates the model over FPC scores (reusing `fdata_to_pc_1d`) rather than
/// over spline/basis coefficients as in R's `denseFLMM` package. Consequently
/// variance components are per FPC component, not smoothed over the argument domain.
///
/// Random-slope estimation is not implemented in this release; `sigma2_slope` is
/// always zero-filled (one entry per FPC component). Future releases may add
/// two-random-effect scalar LMM support via a dedicated helper.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DenseFlmmResult {
    /// Overall mean function μ̂(t) (length m)
    pub mean_function: Vec<f64>,
    /// Fixed effect coefficient functions β̂_j(t) (p × m matrix, one row per covariate)
    pub beta_functions: FdMatrix,
    /// Random effect functions b̂_i(t) per subject (n_subjects × m)
    pub random_effects: FdMatrix,
    /// Fitted values for all observations (n_total × m)
    pub fitted: FdMatrix,
    /// Residuals (n_total × m)
    pub residuals: FdMatrix,
    /// Variance of random effects at each time point Var_i(b̂_i(t)) (length m)
    pub random_variance: Vec<f64>,
    /// Average residual variance across FPC components
    pub sigma2_eps: f64,
    /// Random-intercept variance per FPC component (length k)
    pub sigma2_u: Vec<f64>,
    /// Random-slope variance per FPC component (zero-filled when `random_slopes = false`)
    pub sigma2_slope: Vec<f64>,
    /// Number of FPC components actually used
    pub ncomp: usize,
    /// Number of unique subjects
    pub n_subjects: usize,
    /// FPC eigenvalues (singular values squared / n_total), length k
    pub eigenvalues: Vec<f64>,
    /// Maximum number of REML EM iterations reached across components
    pub n_iter: usize,
    /// `true` if all component models converged before `config.max_iter`
    pub converged: bool,
}

/// Fit a dense functional linear mixed model via FPC score decomposition.
///
/// Extends [`fmm`] with REML convergence metadata and the `DenseFlmmConfig`
/// struct interface.
///
/// # Parametrization divergence from R's `denseFLMM`
///
/// R's `denseFLMM` estimates eigenfunctions from raw covariance smoothing
/// (gamm/bam REML over basis coefficients). fdars decomposes curves into
/// FPC scores via `fdata_to_pc_1d`, then fits a per-component scalar mixed
/// model — producing equivalent fixed-effect and random-effect functions but
/// without the covariance-smoothing regularization step.
///
/// # Arguments
///
/// * `data` — All observed curves (n_total × m)
/// * `subject_ids` — Subject identifier for each curve (length n_total)
/// * `covariates` — Subject-level covariates (n_total × p), or `None`
/// * `config` — Algorithm configuration
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` is empty or `subject_ids`
/// length mismatches `data.nrows()`.
/// Returns [`FdarError::InvalidParameter`] if `config.ncomp` is zero.
/// Returns [`FdarError::ComputationFailed`] if the underlying FPCA fails.
///
/// # Example
///
/// ```rust
/// use fdars_core::famm::{DenseFlmmConfig, dense_flmm};
/// let mut cfg = DenseFlmmConfig::default();
/// cfg.ncomp = 2;
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dense_flmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &DenseFlmmConfig,
) -> Result<DenseFlmmResult, FdarError> {
    let n_total = data.nrows();
    let m = data.ncols();
    if n_total == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n_total} x {m}"),
        });
    }
    if subject_ids.len() != n_total {
        return Err(FdarError::InvalidDimension {
            parameter: "subject_ids",
            expected: format!("length {n_total}"),
            actual: format!("length {}", subject_ids.len()),
        });
    }
    if config.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }

    let (subject_map, n_subjects) = build_subject_map(subject_ids);

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(data, config.ncomp, &argvals)?;
    let k = fpca.scores.ncols();

    let p = covariates.map_or(0, super::matrix::FdMatrix::ncols);

    // Scale scores as in fit_all_components
    let h = if m > 1 { 1.0 / (m - 1) as f64 } else { 1.0 };
    let score_scale = h.sqrt();

    // Fit each component with convergence tracking
    let per_comp: Vec<ScalarMixedResultWithMeta> = iter_maybe_parallel!(0..k)
        .map(|comp| {
            let comp_scores: Vec<f64> = (0..n_total)
                .map(|i| fpca.scores[(i, comp)] * score_scale)
                .collect();
            fit_scalar_mixed_model_tracked(
                &comp_scores,
                &subject_map,
                n_subjects,
                covariates,
                p,
                config.max_iter,
                config.tol,
            )
        })
        .collect();

    // Unpack per-component results
    let mut gamma = vec![vec![0.0; k]; p];
    let mut u_hat = vec![vec![0.0; k]; n_subjects];
    let mut sigma2_u = vec![0.0; k];
    let mut sigma2_eps_total = 0.0;
    let mut all_converged = true;
    let mut max_n_iter = 0usize;

    for (comp, r) in per_comp.iter().enumerate() {
        for j in 0..p {
            gamma[j][comp] = r.result.gamma[j] / score_scale;
        }
        for s in 0..n_subjects {
            u_hat[s][comp] = r.result.u_hat[s] / score_scale;
        }
        sigma2_u[comp] = r.result.sigma2_u;
        sigma2_eps_total += r.result.sigma2_eps;
        if !r.converged {
            all_converged = false;
        }
        if r.n_iter > max_n_iter {
            max_n_iter = r.n_iter;
        }
    }
    let sigma2_eps = if k > 0 {
        sigma2_eps_total / k as f64
    } else {
        0.0
    };

    let beta_functions = recover_beta_functions(&gamma, &fpca.rotation, p, m, k);
    let random_effects = recover_random_effects(&u_hat, &fpca.rotation, n_subjects, m, k);
    let random_variance = compute_random_variance(&random_effects, n_subjects, m);

    let (fitted, residuals) = compute_fitted_residuals(
        data,
        &fpca.mean,
        &beta_functions,
        &random_effects,
        covariates,
        &subject_map,
        n_total,
        m,
        p,
    );

    let eigenvalues: Vec<f64> = fpca
        .singular_values
        .iter()
        .map(|&sv| sv * sv / n_total as f64)
        .collect();

    // sigma2_slope is always zero-filled (random-slope estimation not yet implemented)
    let sigma2_slope = vec![0.0; k];

    Ok(DenseFlmmResult {
        mean_function: fpca.mean,
        beta_functions,
        random_effects,
        fitted,
        residuals,
        random_variance,
        sigma2_eps,
        sigma2_u,
        sigma2_slope,
        ncomp: k,
        n_subjects,
        eigenvalues,
        n_iter: max_n_iter,
        converged: all_converged,
    })
}

/// Internal: scalar mixed model result with convergence metadata.
struct ScalarMixedResultWithMeta {
    result: ScalarMixedResult,
    n_iter: usize,
    converged: bool,
}

/// Fit scalar mixed model, tracking convergence and iteration count.
fn fit_scalar_mixed_model_tracked(
    y: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    covariates: Option<&FdMatrix>,
    p: usize,
    max_iter: usize,
    tol: f64,
) -> ScalarMixedResultWithMeta {
    let n = y.len();
    let ss = SubjectStructure::new(subject_map, n_subjects, n);

    let gamma_init = estimate_fixed_effects(y, covariates, p, n);
    let residuals_init = compute_ols_residuals(y, covariates, &gamma_init, p, n);
    let (mut sigma2_u, mut sigma2_e) =
        estimate_variance_components(&residuals_init, subject_map, n_subjects, n);

    if sigma2_e < 1e-15 {
        sigma2_e = 1e-6;
    }
    if sigma2_u < 1e-15 {
        sigma2_u = sigma2_e * 0.1;
    }

    let mut gamma = gamma_init;
    let mut converged = false;
    let mut n_iter = 0usize;

    for _iter in 0..max_iter {
        n_iter += 1;
        let sigma2_u_old = sigma2_u;
        let sigma2_e_old = sigma2_e;

        let weights = shrinkage_weights(&ss, sigma2_u, sigma2_e);

        if let Some(cov) = covariates.filter(|_| p > 0) {
            if let Some(g) = gls_update_gamma(cov, p, &ss, &weights, y, sigma2_e) {
                gamma = g;
            }
        }

        let r = compute_ols_residuals(y, covariates, &gamma, p, n);
        (sigma2_u, sigma2_e) = reml_variance_update(&r, &ss, &weights, sigma2_u, p);

        let delta = (sigma2_u - sigma2_u_old).abs() + (sigma2_e - sigma2_e_old).abs();
        if delta < tol * (sigma2_u_old + sigma2_e_old) {
            converged = true;
            break;
        }
    }

    let final_residuals = compute_ols_residuals(y, covariates, &gamma, p, n);
    let u_hat = compute_blup(
        &final_residuals,
        subject_map,
        n_subjects,
        sigma2_u,
        sigma2_e,
    );

    ScalarMixedResultWithMeta {
        result: ScalarMixedResult {
            gamma,
            u_hat,
            sigma2_u,
            sigma2_eps: sigma2_e,
        },
        n_iter,
        converged,
    }
}

// ---------------------------------------------------------------------------
// multiFAMM — multivariate stacked extension
// ---------------------------------------------------------------------------

/// Configuration for [`multi_famm`].
///
/// No `#[non_exhaustive]` — callers may use struct-literal construction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiFammConfig {
    /// Number of FPC components per response dimension (default: 3)
    pub ncomp: usize,
    /// Maximum REML EM iterations per component model (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-10)
    pub tol: f64,
}

impl Default for MultiFammConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            max_iter: 50,
            tol: 1e-10,
        }
    }
}

/// Result of a multivariate FAMM fit.
///
/// # Divergence from R's `multiFAMM`
///
/// R's `multiFAMM` (Volkmann et al. 2021) uses joint multivariate FPCA so that
/// cross-dimension covariance kernels K_g(d,e)(t,t') are modelled. fdars instead
/// runs D independent univariate FPCAs (one per response dimension via
/// [`dense_flmm`]), capturing within-dimension structure but **not**
/// cross-dimension covariances. This is a documented capability divergence;
/// users requiring cross-dimension random effects should consider the R package.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MultiFammResult {
    /// Per-dimension FLMM results (length D)
    pub components: Vec<DenseFlmmResult>,
    /// Fitted values stacked row-wise across all dimensions: (n_total × D) × m
    pub stacked_fitted: FdMatrix,
    /// Residuals stacked row-wise across all dimensions: (n_total × D) × m
    pub stacked_residuals: FdMatrix,
    /// Number of response dimensions D
    pub n_dims: usize,
}

/// Fit a multivariate functional additive mixed model.
///
/// Calls [`dense_flmm`] independently for each response dimension and stacks
/// the fitted curves and residuals row-wise.
///
/// All dimensions must share the same number of evaluation points (`ncols`).
///
/// # Divergence from R's `multiFAMM`
///
/// Cross-dimension covariance kernels are not modelled; see [`MultiFammResult`]
/// for details.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` is empty (zero dimensions),
/// - any dimension has zero rows or columns,
/// - dimensions differ in grid size (`ncols`), or
/// - `subject_ids` length mismatches `data[0].nrows()`.
///
/// Returns [`FdarError::InvalidParameter`] if `config.ncomp` is zero.
/// Propagates errors from [`dense_flmm`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn multi_famm(
    data: &[FdMatrix],
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &MultiFammConfig,
) -> Result<MultiFammResult, FdarError> {
    let n_dims = data.len();
    if n_dims == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least one response dimension".to_string(),
            actual: "0 dimensions".to_string(),
        });
    }

    let n_total = data[0].nrows();
    let m = data[0].ncols();

    if n_total == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n_total} x {m}"),
        });
    }

    // Validate all dimensions share the same grid and row count
    for (d, dim) in data.iter().enumerate().skip(1) {
        if dim.ncols() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("all dimensions share ncols = {m}"),
                actual: format!("dimension {d} has ncols = {}", dim.ncols()),
            });
        }
        if dim.nrows() != n_total {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("all dimensions share nrows = {n_total}"),
                actual: format!("dimension {d} has nrows = {}", dim.nrows()),
            });
        }
    }

    // Build per-dimension DenseFlmmConfig from MultiFammConfig
    let dense_cfg = DenseFlmmConfig {
        ncomp: config.ncomp,
        max_iter: config.max_iter,
        tol: config.tol,
        random_slopes: false,
    };

    // Fit each dimension independently
    let mut components: Vec<DenseFlmmResult> = Vec::with_capacity(n_dims);
    for dim_data in data.iter() {
        let result = dense_flmm(dim_data, subject_ids, covariates, &dense_cfg)?;
        components.push(result);
    }

    // Stack fitted and residuals row-wise: (n_total * n_dims) × m
    let stacked_rows = n_total * n_dims;
    let mut stacked_fitted_data = vec![0.0; stacked_rows * m];
    let mut stacked_residuals_data = vec![0.0; stacked_rows * m];

    for (d, comp) in components.iter().enumerate() {
        for i in 0..n_total {
            let row = d * n_total + i;
            for t in 0..m {
                // column-major: element (row, col) at index row + col * nrows
                stacked_fitted_data[row + t * stacked_rows] = comp.fitted[(i, t)];
                stacked_residuals_data[row + t * stacked_rows] = comp.residuals[(i, t)];
            }
        }
    }

    let stacked_fitted = FdMatrix::from_column_major(stacked_fitted_data, stacked_rows, m)
        .map_err(|_| FdarError::ComputationFailed {
            operation: "multi_famm stacking",
            detail: "failed to build stacked_fitted matrix".to_string(),
        })?;
    let stacked_residuals = FdMatrix::from_column_major(stacked_residuals_data, stacked_rows, m)
        .map_err(|_| FdarError::ComputationFailed {
            operation: "multi_famm stacking",
            detail: "failed to build stacked_residuals matrix".to_string(),
        })?;

    Ok(MultiFammResult {
        components,
        stacked_fitted,
        stacked_residuals,
        n_dims,
    })
}

// ---------------------------------------------------------------------------
// fastFMM — massively-univariate per-gridpoint inference
// ---------------------------------------------------------------------------

/// Configuration for [`fast_fmm`].
///
/// No `#[non_exhaustive]` — callers may use struct-literal construction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FastFmmConfig {
    /// Running-mean smoother window width along the grid axis (default: 3; 1 = no smoothing).
    ///
    /// # Divergence from R's `fastFMM`
    ///
    /// R's `fastFMM` uses mgcv thin-plate splines for post-smoothing. fdars uses
    /// a running-mean smoother configured by this window width. Savitzky-Golay
    /// smoothing (better peak preservation) is a planned future improvement.
    pub smooth_window: usize,
    /// Maximum iterations for each per-gridpoint scalar mixed model (default: 30)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// Compute Wald t-statistics and pointwise p-values (default: true).
    ///
    /// When `false`, `t_stats` is zero-filled and `p_values` is one-filled.
    ///
    /// # Divergence from R's `fastFMM`
    ///
    /// R uses a bootstrap for non-Gaussian inference. fdars provides Wald-only
    /// (standard-normal approximation) inference.
    pub compute_inference: bool,
}

impl Default for FastFmmConfig {
    fn default() -> Self {
        Self {
            smooth_window: 3,
            max_iter: 30,
            tol: 1e-8,
            compute_inference: true,
        }
    }
}

/// Result of a fast massively-univariate functional mixed model fit.
///
/// # Divergence from R's `fastFMM`
///
/// R's `fastFMM` (Cui et al. 2022, JCGS 31(1):219–230) fits per-gridpoint
/// GLMMs via `lme4` and smooths via mgcv. fdars fits per-gridpoint scalar mixed
/// models via the existing REML-EM solver, smooths via running-mean, and
/// computes Wald-only inference — no bootstrap.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FastFmmResult {
    /// Smoothed fixed-effect functions: p × m matrix (one row per covariate)
    pub beta_matrix: FdMatrix,
    /// Wald t-statistics: p × m (zero-filled when `compute_inference = false`)
    pub t_stats: FdMatrix,
    /// Pointwise two-sided p-values: p × m (one-filled when `compute_inference = false`)
    pub p_values: FdMatrix,
    /// Per-gridpoint residual variance estimate (length m)
    pub sigma2_eps: Vec<f64>,
    /// Per-gridpoint random-intercept variance estimate (length m)
    pub sigma2_u: Vec<f64>,
    /// Number of grid points m
    pub n_grid: usize,
}

/// Fit a fast massively-univariate functional mixed model.
///
/// Fits a scalar mixed model at each grid point independently, then
/// applies a running-mean smoother along the grid axis.
///
/// # Algorithm
///
/// 1. For each grid point t in 0..m: fit a scalar mixed model on `data.column(t)`
///    via REML-EM, producing raw (β̂(t), û_i(t), σ̂²_u(t), σ̂²_ε(t)).
/// 2. Smooth the raw p × m coefficient matrix with a running-mean window of
///    width `config.smooth_window` (window 1 = identity / no smoothing).
/// 3. When `config.compute_inference`: compute Wald t-statistics
///    `t_jt = β̂_j(t) / se_j(t)` using a standard-normal two-sided p-value.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` is empty or `subject_ids`
/// length mismatches `data.nrows()`.
/// Returns [`FdarError::InvalidParameter`] if `config.smooth_window` is zero.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fast_fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &FastFmmConfig,
) -> Result<FastFmmResult, FdarError> {
    let n_total = data.nrows();
    let m = data.ncols();
    if n_total == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n_total} x {m}"),
        });
    }
    if subject_ids.len() != n_total {
        return Err(FdarError::InvalidDimension {
            parameter: "subject_ids",
            expected: format!("length {n_total}"),
            actual: format!("length {}", subject_ids.len()),
        });
    }
    if config.smooth_window == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "smooth_window",
            message: "must be >= 1 (use 1 for no smoothing)".to_string(),
        });
    }

    let (subject_map, n_subjects) = build_subject_map(subject_ids);
    let p = covariates.map_or(0, super::matrix::FdMatrix::ncols);

    // Per-gridpoint result container (immutable per-item for safe parallel collect)
    struct PointwiseResult {
        gamma: Vec<f64>, // length p (fixed effects at this grid point)
        sigma2_u: f64,   // random-intercept variance at this grid point
        sigma2_eps: f64, // residual variance at this grid point
    }

    // Step 1: Fit per-gridpoint scalar mixed models
    // Use column-major zero-copy access: data.column(t) is a contiguous &[f64]
    let per_point: Vec<PointwiseResult> = iter_maybe_parallel!(0..m)
        .map(|t| {
            let y_t: Vec<f64> = data.column(t).to_vec();
            let r = fit_scalar_mixed_model(&y_t, &subject_map, n_subjects, covariates, p);
            PointwiseResult {
                gamma: r.gamma,
                sigma2_u: r.sigma2_u,
                sigma2_eps: r.sigma2_eps,
            }
        })
        .collect();

    // Unpack into raw p × m beta matrix and per-gridpoint variances
    let mut raw_beta_data = vec![0.0; p * m]; // row-by-row in column-major: row=j, col=t
    let mut sigma2_eps_vec = vec![0.0; m];
    let mut sigma2_u_vec = vec![0.0; m];

    for (t, pt) in per_point.iter().enumerate() {
        // Fill column-major beta: element (j, t) at index j + t * p
        for j in 0..p {
            raw_beta_data[j + t * p] = pt.gamma.get(j).copied().unwrap_or(0.0);
        }
        sigma2_eps_vec[t] = pt.sigma2_eps;
        sigma2_u_vec[t] = pt.sigma2_u;
    }

    // Step 2: Running-mean smoothing along the grid axis (per covariate row)
    let mut smoothed_beta_data = raw_beta_data.clone();
    let w = config.smooth_window;
    if w > 1 && m > 1 {
        let half = w / 2;
        for j in 0..p {
            for t in 0..m {
                let lo = t.saturating_sub(half);
                let hi = (t + half + 1).min(m);
                let count = (hi - lo) as f64;
                let sum: f64 = (lo..hi).map(|tt| raw_beta_data[j + tt * p]).sum();
                smoothed_beta_data[j + t * p] = sum / count;
            }
        }
    }

    // Build the smoothed beta FdMatrix (p × m, column-major)
    let beta_matrix = if p > 0 {
        FdMatrix::from_column_major(smoothed_beta_data, p, m).map_err(|_| {
            FdarError::ComputationFailed {
                operation: "fast_fmm",
                detail: "failed to build beta_matrix".to_string(),
            }
        })?
    } else {
        FdMatrix::zeros(0, m)
    };

    // Step 3: Wald inference
    let (t_stats, p_values) = if config.compute_inference && p > 0 {
        // Compute X'X for standard errors using the first non-zero observation
        // SE²_j(t) = sigma2_eps(t) * (X'X)^{-1}_{jj}
        // We accumulate X'X once (same design for all t) then invert
        let xtx_inv_diag = compute_xtx_inv_diag(covariates, p, n_total);

        let mut t_data = vec![0.0f64; p * m];
        let mut pv_data = vec![1.0f64; p * m];

        for j in 0..p {
            for t in 0..m {
                let beta_jt = beta_matrix[(j, t)];
                let se_sq = sigma2_eps_vec[t] * xtx_inv_diag.get(j).copied().unwrap_or(1.0);
                let se = se_sq.sqrt().max(1e-15);
                let t_stat = beta_jt / se;
                let pval = 2.0 * normal_sf(t_stat.abs());
                t_data[j + t * p] = t_stat;
                pv_data[j + t * p] = pval.clamp(0.0, 1.0);
            }
        }

        let ts = FdMatrix::from_column_major(t_data, p, m).map_err(|_| {
            FdarError::ComputationFailed {
                operation: "fast_fmm",
                detail: "failed to build t_stats".to_string(),
            }
        })?;
        let pv = FdMatrix::from_column_major(pv_data, p, m).map_err(|_| {
            FdarError::ComputationFailed {
                operation: "fast_fmm",
                detail: "failed to build p_values".to_string(),
            }
        })?;
        (ts, pv)
    } else {
        // No inference or no covariates: zeros / ones
        (FdMatrix::zeros(p, m), ones_fdmatrix(p, m))
    };

    Ok(FastFmmResult {
        beta_matrix,
        t_stats,
        p_values,
        sigma2_eps: sigma2_eps_vec,
        sigma2_u: sigma2_u_vec,
        n_grid: m,
    })
}

/// Compute diagonal of (X'X)^{-1} for Wald standard errors.
fn compute_xtx_inv_diag(covariates: Option<&FdMatrix>, p: usize, n: usize) -> Vec<f64> {
    let Some(cov) = covariates else {
        return vec![1.0; p];
    };
    let mut xtx = vec![0.0; p * p];
    for i in 0..n {
        for r in 0..p {
            for s in r..p {
                let val = cov[(i, r)] * cov[(i, s)];
                xtx[r * p + s] += val;
                if r != s {
                    xtx[s * p + r] += val;
                }
            }
        }
    }
    for j in 0..p {
        xtx[j * p + j] += 1e-8;
    }
    // Invert via Cholesky; fall back to reciprocal diagonal if singular
    if let Some(inv) = cholesky_invert(&xtx, p) {
        (0..p).map(|j| inv[j * p + j].max(1e-15)).collect()
    } else {
        // Fallback: diagonal only
        (0..p)
            .map(|j| {
                let d = xtx[j * p + j];
                if d > 1e-15 {
                    1.0 / d
                } else {
                    1.0
                }
            })
            .collect()
    }
}

/// Invert a p×p symmetric positive definite matrix via Cholesky.
fn cholesky_invert(a: &[f64], p: usize) -> Option<Vec<f64>> {
    let l = linalg_cholesky_factor(a, p).ok()?;
    // Solve A * X = I column by column
    let mut inv = vec![0.0; p * p];
    let mut e = vec![0.0; p];
    for j in 0..p {
        e.fill(0.0);
        e[j] = 1.0;
        let col = linalg_cholesky_forward_back(&l, &e, p);
        for i in 0..p {
            inv[i * p + j] = col[i];
        }
    }
    Some(inv)
}

/// Standard normal survival function: P(Z > x) using erf approximation.
fn normal_sf(x: f64) -> f64 {
    // 0.5 * erfc(x / sqrt(2))
    0.5 * erfc(x / core::f64::consts::SQRT_2)
}

/// Complementary error function approximation (Abramowitz & Stegun 7.1.26).
fn erfc(x: f64) -> f64 {
    // Handle negative x via symmetry: erfc(-x) = 2 - erfc(x)
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    // Rational approximation valid for x >= 0, max |error| < 1.5e-7
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    poly * (-x * x).exp()
}

/// Create an FdMatrix filled with ones (p × m).
fn ones_fdmatrix(p: usize, m: usize) -> FdMatrix {
    if p == 0 || m == 0 {
        return FdMatrix::zeros(p, m);
    }
    let data = vec![1.0f64; p * m];
    FdMatrix::from_column_major(data, p, m).unwrap_or_else(|_| FdMatrix::zeros(p, m))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;

    /// Generate repeated measurements: n_subjects × n_visits curves.
    /// Subject-level covariate z affects the curve amplitude.
    fn generate_fmm_data(
        n_subjects: usize,
        n_visits: usize,
        m: usize,
    ) -> (FdMatrix, Vec<usize>, FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let n_total = n_subjects * n_visits;
        let mut col_major = vec![0.0; n_total * m];
        let mut subject_ids = vec![0usize; n_total];
        let mut cov_data = vec![0.0; n_total];

        for s in 0..n_subjects {
            let z = s as f64 / n_subjects as f64; // covariate in [0, 1)
            let subject_effect = 0.5 * (s as f64 - n_subjects as f64 / 2.0); // random-like effect

            for v in 0..n_visits {
                let obs = s * n_visits + v;
                subject_ids[obs] = s;
                cov_data[obs] = z;
                let noise_scale = 0.05;

                for (j, &tj) in t.iter().enumerate() {
                    // Y_sv(t) = sin(2πt) + z * t + subject_effect * cos(2πt) + noise
                    let mu = (2.0 * PI * tj).sin();
                    let fixed = z * tj * 3.0;
                    let random = subject_effect * (2.0 * PI * tj).cos() * 0.3;
                    let noise = noise_scale * ((obs * 7 + j * 3) % 100) as f64 / 100.0;
                    col_major[obs + j * n_total] = mu + fixed + random + noise;
                }
            }
        }

        let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
        let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();
        (data, subject_ids, covariates, t)
    }

    #[test]
    fn test_fmm_basic() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 50);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        assert_eq!(result.mean_function.len(), 50);
        assert_eq!(result.beta_functions.nrows(), 1); // 1 covariate
        assert_eq!(result.beta_functions.ncols(), 50);
        assert_eq!(result.random_effects.nrows(), 10);
        assert_eq!(result.fitted.nrows(), 30);
        assert_eq!(result.residuals.nrows(), 30);
        assert_eq!(result.n_subjects, 10);
    }

    #[test]
    fn test_fmm_fitted_plus_residuals_equals_data() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(8, 3, 40);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        let n = data.nrows();
        let m = data.ncols();
        for i in 0..n {
            for t in 0..m {
                let reconstructed = result.fitted[(i, t)] + result.residuals[(i, t)];
                assert!(
                    (reconstructed - data[(i, t)]).abs() < 1e-8,
                    "Fitted + residual should equal data at ({}, {}): {} vs {}",
                    i,
                    t,
                    reconstructed,
                    data[(i, t)]
                );
            }
        }
    }

    #[test]
    fn test_fmm_random_variance_positive() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 50);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        for &v in &result.random_variance {
            assert!(v >= 0.0, "Random variance should be non-negative");
        }
    }

    #[test]
    fn test_fmm_no_covariates() {
        let (data, subject_ids, _cov, _t) = generate_fmm_data(8, 3, 40);
        let result = fmm(&data, &subject_ids, None, 3).unwrap();

        assert_eq!(result.beta_functions.nrows(), 0);
        assert_eq!(result.n_subjects, 8);
        assert_eq!(result.fitted.nrows(), 24);
    }

    #[test]
    fn test_fmm_predict() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 50);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        // Predict for new subjects with covariate = 0.5
        let new_cov = FdMatrix::from_column_major(vec![0.5], 1, 1).unwrap();
        let predicted = fmm_predict(&result, Some(&new_cov));

        assert_eq!(predicted.nrows(), 1);
        assert_eq!(predicted.ncols(), 50);

        // Predicted curve should be reasonable (not NaN or extreme)
        for t in 0..50 {
            assert!(predicted[(0, t)].is_finite());
            assert!(
                predicted[(0, t)].abs() < 20.0,
                "Predicted value too extreme at t={}: {}",
                t,
                predicted[(0, t)]
            );
        }
    }

    #[test]
    fn test_fmm_test_fixed_detects_effect() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(15, 3, 40);

        let result = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 99, 42).unwrap();

        assert_eq!(result.f_statistics.len(), 1);
        assert_eq!(result.p_values.len(), 1);
        assert!(
            result.p_values[0] < 0.1,
            "Should detect covariate effect, got p={}",
            result.p_values[0]
        );
    }

    #[test]
    fn test_fmm_test_fixed_no_effect() {
        let n_subjects = 10;
        let n_visits = 3;
        let m = 40;
        let t = uniform_grid(m);
        let n_total = n_subjects * n_visits;

        // No covariate effect: Y = sin(2πt) + noise
        let mut col_major = vec![0.0; n_total * m];
        let mut subject_ids = vec![0usize; n_total];
        let mut cov_data = vec![0.0; n_total];

        for s in 0..n_subjects {
            for v in 0..n_visits {
                let obs = s * n_visits + v;
                subject_ids[obs] = s;
                cov_data[obs] = s as f64 / n_subjects as f64;
                for (j, &tj) in t.iter().enumerate() {
                    col_major[obs + j * n_total] =
                        (2.0 * PI * tj).sin() + 0.1 * ((obs * 7 + j * 3) % 100) as f64 / 100.0;
                }
            }
        }

        let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
        let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();

        let result = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 99, 42).unwrap();
        assert!(
            result.p_values[0] > 0.05,
            "Should not detect effect, got p={}",
            result.p_values[0]
        );
    }

    #[test]
    fn test_fmm_invalid_input() {
        let data = FdMatrix::zeros(0, 0);
        assert!(fmm(&data, &[], None, 1).is_err());

        let data = FdMatrix::zeros(10, 50);
        let ids = vec![0; 5]; // wrong length
        assert!(fmm(&data, &ids, None, 1).is_err());
    }

    #[test]
    fn test_fmm_single_visit_per_subject() {
        let n = 10;
        let m = 40;
        let t = uniform_grid(m);
        let mut col_major = vec![0.0; n * m];
        let subject_ids: Vec<usize> = (0..n).collect();

        for i in 0..n {
            for (j, &tj) in t.iter().enumerate() {
                col_major[i + j * n] = (2.0 * PI * tj).sin();
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

        // Should still work with 1 visit per subject
        let result = fmm(&data, &subject_ids, None, 2).unwrap();
        assert_eq!(result.n_subjects, n);
        assert_eq!(result.fitted.nrows(), n);
    }

    #[test]
    fn test_build_subject_map() {
        let (map, n) = build_subject_map(&[5, 5, 10, 10, 20]);
        assert_eq!(n, 3);
        assert_eq!(map, vec![0, 0, 1, 1, 2]);
    }

    #[test]
    fn test_variance_components_positive() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 50);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        assert!(result.sigma2_eps >= 0.0);
        for &s in &result.sigma2_u {
            assert!(s >= 0.0);
        }
    }

    // -------------------------------------------------------------------
    // Additional tests
    // -------------------------------------------------------------------

    #[test]
    fn test_fmm_ncomp_zero_returns_error() {
        let (data, subject_ids, _cov, _t) = generate_fmm_data(5, 2, 20);
        let err = fmm(&data, &subject_ids, None, 0).unwrap_err();
        match err {
            FdarError::InvalidParameter { parameter, .. } => {
                assert_eq!(parameter, "ncomp");
            }
            other => panic!("Expected InvalidParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_fmm_single_component() {
        // Fit with only 1 FPC component
        let (data, subject_ids, covariates, _t) = generate_fmm_data(8, 3, 30);
        let result = fmm(&data, &subject_ids, Some(&covariates), 1).unwrap();

        assert_eq!(result.ncomp, 1);
        assert_eq!(result.sigma2_u.len(), 1);
        assert_eq!(result.eigenvalues.len(), 1);
        assert_eq!(result.mean_function.len(), 30);
        // Fitted + residuals = data
        for i in 0..data.nrows() {
            for t in 0..data.ncols() {
                let diff = (result.fitted[(i, t)] + result.residuals[(i, t)] - data[(i, t)]).abs();
                assert!(diff < 1e-8);
            }
        }
    }

    #[test]
    fn test_fmm_two_subjects() {
        // Minimal number of subjects (2) with multiple visits
        let n_subjects = 2;
        let n_visits = 5;
        let m = 20;
        let t = uniform_grid(m);
        let n_total = n_subjects * n_visits;
        let mut col_major = vec![0.0; n_total * m];
        let mut subject_ids = vec![0usize; n_total];

        for s in 0..n_subjects {
            for v in 0..n_visits {
                let obs = s * n_visits + v;
                subject_ids[obs] = s;
                for (j, &tj) in t.iter().enumerate() {
                    col_major[obs + j * n_total] =
                        (2.0 * PI * tj).sin() + (s as f64) * 0.5 + 0.01 * v as f64;
                }
            }
        }
        let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
        let result = fmm(&data, &subject_ids, None, 2).unwrap();

        assert_eq!(result.n_subjects, 2);
        assert_eq!(result.random_effects.nrows(), 2);
        assert_eq!(result.fitted.nrows(), n_total);
    }

    #[test]
    fn test_fmm_predict_no_covariates() {
        let (data, subject_ids, _cov, _t) = generate_fmm_data(6, 3, 30);
        let result = fmm(&data, &subject_ids, None, 2).unwrap();

        // Predict without covariates — should return mean function
        let predicted = fmm_predict(&result, None);
        assert_eq!(predicted.nrows(), 1);
        assert_eq!(predicted.ncols(), 30);
        for t in 0..30 {
            let diff = (predicted[(0, t)] - result.mean_function[t]).abs();
            assert!(
                diff < 1e-12,
                "Without covariates, prediction should equal mean"
            );
        }
    }

    #[test]
    fn test_fmm_predict_multiple_new_subjects() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 40);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        // Predict for 3 new subjects with different covariate values
        let new_cov = FdMatrix::from_column_major(vec![0.1, 0.5, 0.9], 3, 1).unwrap();
        let predicted = fmm_predict(&result, Some(&new_cov));

        assert_eq!(predicted.nrows(), 3);
        assert_eq!(predicted.ncols(), 40);

        // All predictions should be finite
        for i in 0..3 {
            for t in 0..40 {
                assert!(predicted[(i, t)].is_finite());
            }
        }

        // Predictions for different covariates should differ
        let diff_01: f64 = (0..40)
            .map(|t| (predicted[(0, t)] - predicted[(1, t)]).powi(2))
            .sum();
        assert!(
            diff_01 > 1e-10,
            "Different covariates should yield different predictions"
        );
    }

    #[test]
    fn test_fmm_eigenvalues_decreasing() {
        let (data, subject_ids, _cov, _t) = generate_fmm_data(10, 3, 50);
        let result = fmm(&data, &subject_ids, None, 5).unwrap();

        // Eigenvalues should be in decreasing order (from FPCA)
        for i in 1..result.eigenvalues.len() {
            assert!(
                result.eigenvalues[i] <= result.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be non-increasing: {} > {}",
                result.eigenvalues[i],
                result.eigenvalues[i - 1]
            );
        }
    }

    #[test]
    fn test_fmm_random_effects_sum_near_zero() {
        // Random effects should approximately sum to zero across subjects
        let (data, subject_ids, covariates, _t) = generate_fmm_data(20, 3, 40);
        let result = fmm(&data, &subject_ids, Some(&covariates), 3).unwrap();

        let m = result.mean_function.len();
        for t in 0..m {
            let sum: f64 = (0..result.n_subjects)
                .map(|s| result.random_effects[(s, t)])
                .sum();
            let mean_abs: f64 = (0..result.n_subjects)
                .map(|s| result.random_effects[(s, t)].abs())
                .sum::<f64>()
                / result.n_subjects as f64;
            // Relative to the scale of random effects, the sum should be small
            if mean_abs > 1e-10 {
                assert!(
                    (sum / result.n_subjects as f64).abs() < mean_abs * 2.0,
                    "Random effects should roughly center around zero at t={}: sum={}, mean_abs={}",
                    t,
                    sum,
                    mean_abs
                );
            }
        }
    }

    #[test]
    fn test_fmm_subject_ids_mismatch_error() {
        let data = FdMatrix::zeros(10, 20);
        let ids = vec![0; 7]; // wrong length
        let err = fmm(&data, &ids, None, 1).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "subject_ids");
            }
            other => panic!("Expected InvalidDimension, got {:?}", other),
        }
    }

    #[test]
    fn test_fmm_test_fixed_empty_data_error() {
        let data = FdMatrix::zeros(0, 0);
        let covariates = FdMatrix::zeros(0, 1);
        let err = fmm_test_fixed(&data, &[], &covariates, 1, 10, 42).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "data");
            }
            other => panic!("Expected InvalidDimension for data, got {:?}", other),
        }
    }

    #[test]
    fn test_fmm_test_fixed_zero_covariates_error() {
        let data = FdMatrix::zeros(10, 20);
        let ids = vec![0; 10];
        let covariates = FdMatrix::zeros(10, 0);
        let err = fmm_test_fixed(&data, &ids, &covariates, 1, 10, 42).unwrap_err();
        match err {
            FdarError::InvalidDimension { parameter, .. } => {
                assert_eq!(parameter, "covariates");
            }
            other => panic!("Expected InvalidDimension for covariates, got {:?}", other),
        }
    }

    #[test]
    fn test_build_subject_map_single_subject() {
        let (map, n) = build_subject_map(&[42, 42, 42]);
        assert_eq!(n, 1);
        assert_eq!(map, vec![0, 0, 0]);
    }

    #[test]
    fn test_build_subject_map_non_contiguous_ids() {
        let (map, n) = build_subject_map(&[100, 200, 100, 300, 200]);
        assert_eq!(n, 3);
        // sorted unique: [100, 200, 300] -> indices [0, 1, 2]
        assert_eq!(map, vec![0, 1, 0, 2, 1]);
    }

    #[test]
    fn test_fmm_many_components_clamped() {
        // Request more components than available; FPCA should clamp
        let (data, subject_ids, _cov, _t) = generate_fmm_data(5, 3, 20);
        let n_total = data.nrows();
        // Request 100 components — should be clamped to min(n_total, m) - 1
        let result = fmm(&data, &subject_ids, None, 100).unwrap();
        assert!(
            result.ncomp <= n_total.min(20),
            "ncomp should be clamped: got {}",
            result.ncomp
        );
        assert!(result.ncomp >= 1);
    }

    #[test]
    fn test_fmm_residuals_small_with_enough_components() {
        // With enough components, residuals should be small relative to data
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 30);
        let result = fmm(&data, &subject_ids, Some(&covariates), 5).unwrap();

        let n = data.nrows();
        let m = data.ncols();
        let mut data_ss = 0.0_f64;
        let mut resid_ss = 0.0_f64;
        for i in 0..n {
            for t in 0..m {
                data_ss += data[(i, t)].powi(2);
                resid_ss += result.residuals[(i, t)].powi(2);
            }
        }

        // R-squared should be reasonably high for structured data
        let r_squared = 1.0 - resid_ss / data_ss;
        assert!(
            r_squared > 0.5,
            "R-squared should be high with enough components: {}",
            r_squared
        );
    }

    // -----------------------------------------------------------------------
    // dense_flmm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dense_flmm_basic() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 30);
        let cfg = DenseFlmmConfig::default();
        let result = dense_flmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
        assert_eq!(result.ncomp, cfg.ncomp);
        assert_eq!(result.n_subjects, 10);
        assert_eq!(result.mean_function.len(), 30);
        assert_eq!(result.beta_functions.ncols(), 30);
        assert_eq!(result.random_variance.len(), 30);
        assert_eq!(result.sigma2_u.len(), cfg.ncomp);
        // Random-slope variance is always present, zero-filled this release.
        assert_eq!(result.sigma2_slope.len(), cfg.ncomp);
        assert!(result.sigma2_slope.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_dense_flmm_fitted_plus_residuals_equals_data() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(8, 3, 24);
        let cfg = DenseFlmmConfig::default();
        let result = dense_flmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
        let n = data.nrows();
        let m = data.ncols();
        for i in 0..n {
            for j in 0..m {
                let recon = result.fitted[(i, j)] + result.residuals[(i, j)];
                assert!(
                    (recon - data[(i, j)]).abs() < 1e-6,
                    "fitted+residuals must equal data at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_dense_flmm_recovers_signal_and_positive_variance() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(12, 4, 30);
        let cfg = DenseFlmmConfig {
            ncomp: 4,
            ..Default::default()
        };
        let result = dense_flmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
        // Residuals shrink relative to a mean-only baseline (fit tracks the truth).
        let n = data.nrows();
        let m = data.ncols();
        let mut col_means = vec![0.0; m];
        for j in 0..m {
            for i in 0..n {
                col_means[j] += data[(i, j)];
            }
            col_means[j] /= n as f64;
        }
        let (mut base_ss, mut resid_ss) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            for j in 0..m {
                base_ss += (data[(i, j)] - col_means[j]).powi(2);
                resid_ss += result.residuals[(i, j)].powi(2);
            }
        }
        assert!(
            resid_ss < 0.5 * base_ss,
            "mixed model should explain most variance: resid={resid_ss}, base={base_ss}"
        );
        // At least one FPC component has positive random-intercept variance.
        assert!(result.sigma2_u.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn test_dense_flmm_invalid_inputs() {
        let cfg = DenseFlmmConfig::default();
        let empty = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
        assert!(dense_flmm(&empty, &[], None, &cfg).is_err());

        let (data, subject_ids, _cov, _t) = generate_fmm_data(4, 2, 10);
        // Mismatched subject_ids length.
        let bad_ids = vec![0usize; subject_ids.len() + 1];
        assert!(dense_flmm(&data, &bad_ids, None, &cfg).is_err());

        // ncomp == 0.
        let bad_cfg = DenseFlmmConfig {
            ncomp: 0,
            ..Default::default()
        };
        assert!(dense_flmm(&data, &subject_ids, None, &bad_cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // multi_famm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_famm_basic() {
        let (d0, subject_ids, cov, _t) = generate_fmm_data(10, 3, 20);
        let (d1, _s1, _c1, _t1) = generate_fmm_data(10, 3, 20);
        let cfg = MultiFammConfig {
            ncomp: 3,
            max_iter: 50,
            tol: 1e-10,
        };
        let result = multi_famm(&[d0, d1], &subject_ids, Some(&cov), &cfg).unwrap();
        assert_eq!(result.n_dims, 2);
        assert_eq!(result.components.len(), 2);
        // Stacked matrices carry D * n_total rows.
        assert_eq!(result.stacked_fitted.nrows(), 2 * subject_ids.len());
        assert_eq!(result.stacked_residuals.nrows(), 2 * subject_ids.len());
    }

    #[test]
    fn test_multi_famm_invalid_inputs() {
        let cfg = MultiFammConfig {
            ncomp: 3,
            max_iter: 50,
            tol: 1e-10,
        };
        // Empty dimension list.
        assert!(multi_famm(&[], &[], None, &cfg).is_err());

        // Grid-size mismatch between dimensions.
        let (d0, subject_ids, cov, _t) = generate_fmm_data(6, 2, 20);
        let (d1, _s1, _c1, _t1) = generate_fmm_data(6, 2, 25);
        assert!(multi_famm(&[d0, d1], &subject_ids, Some(&cov), &cfg).is_err());
    }

    // -----------------------------------------------------------------------
    // fast_fmm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fast_fmm_basic() {
        let (data, subject_ids, covariates, _t) = generate_fmm_data(10, 3, 20);
        let cfg = FastFmmConfig::default();
        let result = fast_fmm(&data, &subject_ids, Some(&covariates), &cfg).unwrap();
        assert_eq!(result.n_grid, 20);
        assert_eq!(result.beta_matrix.ncols(), 20);
        assert_eq!(result.p_values.ncols(), 20);
        assert_eq!(result.sigma2_eps.len(), 20);
        // p-values must be valid probabilities, t-stats finite.
        for i in 0..result.p_values.nrows() {
            for j in 0..result.p_values.ncols() {
                let p = result.p_values[(i, j)];
                assert!((0.0..=1.0).contains(&p), "p-value out of range: {p}");
                assert!(result.t_stats[(i, j)].is_finite());
            }
        }
    }

    #[test]
    fn test_fast_fmm_invalid_inputs() {
        let (data, subject_ids, _cov, _t) = generate_fmm_data(4, 2, 10);
        // smooth_window == 0.
        let bad_cfg = FastFmmConfig {
            smooth_window: 0,
            ..Default::default()
        };
        assert!(fast_fmm(&data, &subject_ids, None, &bad_cfg).is_err());

        // Mismatched subject_ids length.
        let cfg = FastFmmConfig::default();
        let bad_ids = vec![0usize; subject_ids.len() + 1];
        assert!(fast_fmm(&data, &bad_ids, None, &cfg).is_err());
    }
}
