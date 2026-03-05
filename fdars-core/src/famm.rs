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

use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

/// Result of a functional mixed model fit.
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
pub fn fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Option<FmmResult> {
    let n_total = data.nrows();
    let m = data.ncols();
    if n_total == 0 || m == 0 || subject_ids.len() != n_total || ncomp == 0 {
        return None;
    }

    // Determine unique subjects
    let (subject_map, n_subjects) = build_subject_map(subject_ids);

    // Step 1: FPCA on pooled data
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let k = fpca.scores.ncols(); // actual number of components

    // Step 2: For each FPC score, fit scalar mixed model
    let p = covariates.map_or(0, |c| c.ncols());
    let mut gamma = vec![vec![0.0; k]; p]; // gamma[j][k] = fixed effect coeff j for component k
    let mut u_hat = vec![vec![0.0; k]; n_subjects]; // u_hat[i][k] = random effect for subject i, component k
    let mut sigma2_u = vec![0.0; k];
    let mut sigma2_eps_total = 0.0;

    for comp in 0..k {
        let scores: Vec<f64> = (0..n_total).map(|i| fpca.scores[(i, comp)]).collect();
        let result = fit_scalar_mixed_model(&scores, &subject_map, n_subjects, covariates, p);
        for j in 0..p {
            gamma[j][comp] = result.gamma[j];
        }
        for s in 0..n_subjects {
            u_hat[s][comp] = result.u_hat[s];
        }
        sigma2_u[comp] = result.sigma2_u;
        sigma2_eps_total += result.sigma2_eps;
    }
    let sigma2_eps = sigma2_eps_total / k as f64;

    // Step 3: Recover functional coefficients
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

    Some(FmmResult {
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
fn build_subject_map(subject_ids: &[usize]) -> (Vec<usize>, usize) {
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

/// Scalar mixed model result for one FPC component.
struct ScalarMixedResult {
    gamma: Vec<f64>, // fixed effects (length p)
    u_hat: Vec<f64>, // random effects per subject (length n_subjects)
    sigma2_u: f64,   // random effect variance
    sigma2_eps: f64, // residual variance
}

/// Fit scalar mixed model: y_ij = x_i'γ + u_i + e_ij.
///
/// Uses Henderson's mixed model equations with REML-like variance estimation.
fn fit_scalar_mixed_model(
    y: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    covariates: Option<&FdMatrix>,
    p: usize,
) -> ScalarMixedResult {
    let n = y.len();

    // Step 1: Estimate fixed effects via OLS (ignoring random effects initially)
    let gamma = estimate_fixed_effects(y, covariates, p, n);

    // Step 2: Compute residuals from fixed effects
    let residuals = compute_ols_residuals(y, covariates, &gamma, p, n);

    // Step 3: Estimate variance components
    let (sigma2_u, sigma2_eps) =
        estimate_variance_components(&residuals, subject_map, n_subjects, n);

    // Step 4: Compute BLUP for random effects
    let u_hat = compute_blup(&residuals, subject_map, n_subjects, sigma2_u, sigma2_eps);

    ScalarMixedResult {
        gamma,
        u_hat,
        sigma2_u,
        sigma2_eps,
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
    let cov = covariates.unwrap();

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

/// Cholesky solve: A x = b where A is p×p symmetric positive definite.
fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    // Factor A = L L'
    let mut l = vec![0.0; p * p];
    for j in 0..p {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * p + k] * l[j * p + k];
        }
        let diag = a[j * p + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j * p + j] = diag.sqrt();
        for i in (j + 1)..p {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * p + k] * l[j * p + k];
            }
            l[i * p + j] = (a[i * p + j] - s) / l[j * p + j];
        }
    }

    // Forward solve: L z = b
    let mut z = vec![0.0; p];
    for i in 0..p {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * p + j] * z[j];
        }
        z[i] = (b[i] - s) / l[i * p + i];
    }

    // Back solve: L' x = z
    let mut x = vec![0.0; p];
    for i in (0..p).rev() {
        let mut s = 0.0;
        for j in (i + 1)..p {
            s += l[j * p + i] * x[j];
        }
        x[i] = (z[i] - s) / l[i * p + i];
    }

    Some(x)
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
fn recover_random_effects(
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
pub fn fmm_predict(result: &FmmResult, new_covariates: Option<&FdMatrix>) -> FdMatrix {
    let m = result.mean_function.len();
    let n_new = new_covariates.map_or(1, |c| c.nrows());
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
pub fn fmm_test_fixed(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: &FdMatrix,
    ncomp: usize,
    n_perm: usize,
    seed: u64,
) -> Option<FmmTestResult> {
    let n_total = data.nrows();
    let m = data.ncols();
    let p = covariates.ncols();
    if n_total == 0 || p == 0 {
        return None;
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

    Some(FmmTestResult {
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

        if let Some(perm_result) = fmm(data, subject_ids, Some(&perm_cov), ncomp) {
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
    }

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
        assert!(fmm(&data, &[], None, 1).is_none());

        let data = FdMatrix::zeros(10, 50);
        let ids = vec![0; 5]; // wrong length
        assert!(fmm(&data, &ids, None, 1).is_none());
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
}
