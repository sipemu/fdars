//! Function-on-scalar regression and functional ANOVA.
//!
//! Predicts a **functional response** from scalar/categorical predictors:
//! ```text
//! X_i(t) = μ(t) + Σⱼ βⱼ(t) · z_ij + ε_i(t)
//! ```
//!
//! # Methods
//!
//! - [`fosr`]: Penalized function-on-scalar regression (pointwise OLS + smoothing)
//! - [`fanova`]: Functional ANOVA with permutation-based global test
//! - [`predict_fosr`]: Predict new curves from fitted model

use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;

// ---------------------------------------------------------------------------
// Linear algebra helpers (self-contained)
// ---------------------------------------------------------------------------

/// Cholesky factorization: A = LL'. Returns L (p×p flat row-major) or None if singular.
fn cholesky_factor(a: &[f64], p: usize) -> Option<Vec<f64>> {
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

/// Solve Lz = b (forward) then L'x = z (back).
fn cholesky_forward_back(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
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

/// Compute X'X (symmetric, p×p stored flat row-major).
fn compute_xtx(x: &FdMatrix) -> Vec<f64> {
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

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Result of function-on-scalar regression.
pub struct FosrResult {
    /// Intercept function μ(t) (length m)
    pub intercept: Vec<f64>,
    /// Coefficient functions β_j(t), one per predictor (p × m matrix, row j = βⱼ(t))
    pub beta: FdMatrix,
    /// Fitted functional values (n × m matrix)
    pub fitted: FdMatrix,
    /// Residual functions (n × m matrix)
    pub residuals: FdMatrix,
    /// Pointwise R² across the domain (length m)
    pub r_squared_t: Vec<f64>,
    /// Global R² (integrated)
    pub r_squared: f64,
    /// Pointwise standard errors for each βⱼ(t) (p × m matrix)
    pub beta_se: FdMatrix,
    /// Smoothing parameter λ used
    pub lambda: f64,
    /// GCV value
    pub gcv: f64,
}

/// Result of FPC-based function-on-scalar regression.
pub struct FosrFpcResult {
    /// Intercept function μ(t) (length m)
    pub intercept: Vec<f64>,
    /// Coefficient functions β_j(t), one per predictor (p × m matrix, row j = βⱼ(t))
    pub beta: FdMatrix,
    /// Fitted functional values (n × m matrix)
    pub fitted: FdMatrix,
    /// Residual functions (n × m matrix)
    pub residuals: FdMatrix,
    /// Pointwise R² across the domain (length m)
    pub r_squared_t: Vec<f64>,
    /// Global R² (integrated)
    pub r_squared: f64,
    /// FPC-space regression coefficients gamma\[j\]\[k\] (one `Vec<f64>` per predictor)
    pub beta_scores: Vec<Vec<f64>>,
    /// Number of FPC components used
    pub ncomp: usize,
}

/// Result of functional ANOVA.
pub struct FanovaResult {
    /// Group mean functions (k × m matrix, row g = mean curve of group g)
    pub group_means: FdMatrix,
    /// Overall mean function (length m)
    pub overall_mean: Vec<f64>,
    /// Pointwise F-statistic across the domain (length m)
    pub f_statistic_t: Vec<f64>,
    /// Global test statistic (integrated F)
    pub global_statistic: f64,
    /// P-value from permutation test
    pub p_value: f64,
    /// Number of permutations performed
    pub n_perm: usize,
    /// Number of groups
    pub n_groups: usize,
    /// Group labels (sorted unique values)
    pub group_labels: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Build second-order difference penalty matrix D'D (p×p, flat row-major).
fn penalty_matrix(m: usize) -> Vec<f64> {
    if m < 3 {
        return vec![0.0; m * m];
    }
    // D is (m-2)×m second-difference operator
    // D'D is m×m symmetric banded matrix
    let mut dtd = vec![0.0; m * m];
    for i in 0..m - 2 {
        // D[i,:] = [0..0, 1, -2, 1, 0..0] at positions i, i+1, i+2
        let coeffs = [(i, 1.0), (i + 1, -2.0), (i + 2, 1.0)];
        for &(r, cr) in &coeffs {
            for &(c, cc) in &coeffs {
                dtd[r * m + c] += cr * cc;
            }
        }
    }
    dtd
}

/// Solve (A + λP)x = b for each column of B (pointwise regression at each t).
/// A is X'X (p×p), B is X'Y (p×m), P is penalty matrix (p×p).
/// Returns coefficient matrix (p×m).
fn penalized_solve(xtx: &[f64], xty: &FdMatrix, penalty: &[f64], lambda: f64) -> Option<FdMatrix> {
    let p = xty.nrows();
    let m = xty.ncols();

    // Build (X'X + λP)
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }

    // Cholesky factor
    let l = cholesky_factor(&a, p)?;

    // Solve for each grid point
    let mut beta = FdMatrix::zeros(p, m);
    for t in 0..m {
        let b: Vec<f64> = (0..p).map(|j| xty[(j, t)]).collect();
        let x = cholesky_forward_back(&l, &b, p);
        for j in 0..p {
            beta[(j, t)] = x[j];
        }
    }
    Some(beta)
}

/// Compute pointwise R² at each grid point.
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

/// GCV for penalized regression: (1/nm) Σ_{i,t} (r_{it} / (1 - tr(H)/n))²
fn compute_fosr_gcv(residuals: &FdMatrix, trace_h: f64) -> f64 {
    let (n, m) = residuals.shape();
    let denom = (1.0 - trace_h / n as f64).max(1e-10);
    let ss_res: f64 = (0..n)
        .flat_map(|i| (0..m).map(move |t| residuals[(i, t)].powi(2)))
        .sum();
    ss_res / (n as f64 * m as f64 * denom * denom)
}

// ---------------------------------------------------------------------------
// fosr: Function-on-scalar regression
// ---------------------------------------------------------------------------

/// Penalized function-on-scalar regression.
///
/// Fits pointwise OLS at each grid point t, then smooths the coefficient
/// functions β_j(t) using a second-order roughness penalty.
///
/// # Arguments
/// * `data` - Functional response matrix (n × m)
/// * `predictors` - Scalar predictor matrix (n × p)
/// * `lambda` - Smoothing parameter (0 for no smoothing, negative for GCV selection)
///
/// # Returns
/// [`FosrResult`] with coefficient functions, fitted values, and diagnostics
/// Build design matrix with intercept: \[1, z_1, ..., z_p\].
fn build_fosr_design(predictors: &FdMatrix, n: usize) -> FdMatrix {
    let p = predictors.ncols();
    let p_total = p + 1;
    let mut design = FdMatrix::zeros(n, p_total);
    for i in 0..n {
        design[(i, 0)] = 1.0;
        for j in 0..p {
            design[(i, 1 + j)] = predictors[(i, j)];
        }
    }
    design
}

/// Compute X'Y (p_total × m).
fn compute_xty_matrix(design: &FdMatrix, data: &FdMatrix) -> FdMatrix {
    let (n, m) = data.shape();
    let p_total = design.ncols();
    let mut xty = FdMatrix::zeros(p_total, m);
    for j in 0..p_total {
        for t in 0..m {
            let mut s = 0.0;
            for i in 0..n {
                s += design[(i, j)] * data[(i, t)];
            }
            xty[(j, t)] = s;
        }
    }
    xty
}

/// Extract rows 1..p+1 from a (p+1)×m matrix, dropping the intercept row.
fn drop_intercept_rows(full: &FdMatrix, p: usize, m: usize) -> FdMatrix {
    let mut out = FdMatrix::zeros(p, m);
    for j in 0..p {
        for t in 0..m {
            out[(j, t)] = full[(j + 1, t)];
        }
    }
    out
}

pub fn fosr(data: &FdMatrix, predictors: &FdMatrix, lambda: f64) -> Option<FosrResult> {
    let (n, m) = data.shape();
    let p = predictors.ncols();
    if n < p + 2 || m == 0 || predictors.nrows() != n {
        return None;
    }

    let design = build_fosr_design(predictors, n);
    let p_total = design.ncols();
    let xtx = compute_xtx(&design);
    let xty = compute_xty_matrix(&design, data);
    let penalty = penalty_matrix(p_total);

    let lambda = if lambda < 0.0 {
        select_lambda_gcv(&xtx, &xty, &penalty, data, &design)
    } else {
        lambda
    };

    let beta = penalized_solve(&xtx, &xty, &penalty, lambda)?;
    let (fitted, residuals) = compute_fosr_fitted(&design, &beta, data);

    let r_squared_t = pointwise_r_squared(data, &fitted);
    let r_squared = r_squared_t.iter().sum::<f64>() / m as f64;
    let beta_se = compute_beta_se(&xtx, &penalty, lambda, &residuals, p_total, n);
    let trace_h = compute_trace_hat(&xtx, &penalty, lambda, p_total, n);
    let gcv = compute_fosr_gcv(&residuals, trace_h);

    let intercept: Vec<f64> = (0..m).map(|t| beta[(0, t)]).collect();

    Some(FosrResult {
        intercept,
        beta: drop_intercept_rows(&beta, p, m),
        fitted,
        residuals,
        r_squared_t,
        r_squared,
        beta_se: drop_intercept_rows(&beta_se, p, m),
        lambda,
        gcv,
    })
}

/// Compute fitted values Ŷ = X β and residuals.
fn compute_fosr_fitted(
    design: &FdMatrix,
    beta: &FdMatrix,
    data: &FdMatrix,
) -> (FdMatrix, FdMatrix) {
    let (n, m) = data.shape();
    let p_total = design.ncols();
    let mut fitted = FdMatrix::zeros(n, m);
    let mut residuals = FdMatrix::zeros(n, m);
    for i in 0..n {
        for t in 0..m {
            let mut yhat = 0.0;
            for j in 0..p_total {
                yhat += design[(i, j)] * beta[(j, t)];
            }
            fitted[(i, t)] = yhat;
            residuals[(i, t)] = data[(i, t)] - yhat;
        }
    }
    (fitted, residuals)
}

/// Select smoothing parameter λ via GCV on a grid.
fn select_lambda_gcv(
    xtx: &[f64],
    xty: &FdMatrix,
    penalty: &[f64],
    data: &FdMatrix,
    design: &FdMatrix,
) -> f64 {
    let lambdas = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0];
    let p_total = design.ncols();
    let n = design.nrows();

    let mut best_lambda = 0.0;
    let mut best_gcv = f64::INFINITY;

    for &lam in &lambdas {
        let beta = match penalized_solve(xtx, xty, penalty, lam) {
            Some(b) => b,
            None => continue,
        };
        let (_, residuals) = compute_fosr_fitted(design, &beta, data);
        let trace_h = compute_trace_hat(xtx, penalty, lam, p_total, n);
        let gcv = compute_fosr_gcv(&residuals, trace_h);
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lambda = lam;
        }
    }
    best_lambda
}

/// Compute trace of hat matrix: tr(H) = tr(X (X'X + λP)^{-1} X') = Σ_j h_jj.
fn compute_trace_hat(xtx: &[f64], penalty: &[f64], lambda: f64, p: usize, n: usize) -> f64 {
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }
    // tr(H) = tr(X A^{-1} X') = Σ_{j=0..p} a^{-1}_{jj} * xtx_{jj}
    // More precisely: tr(X (X'X+λP)^{-1} X') = tr((X'X+λP)^{-1} X'X)
    let l = match cholesky_factor(&a, p) {
        Some(l) => l,
        None => return p as f64, // fallback
    };

    // Compute A^{-1} X'X via solving A Z = X'X column by column, then trace
    let mut trace = 0.0;
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| xtx[i * p + j]).collect();
        let z = cholesky_forward_back(&l, &col, p);
        trace += z[j]; // diagonal element of A^{-1} X'X
    }
    trace.min(n as f64)
}

/// Compute pointwise standard errors for β(t).
fn compute_beta_se(
    xtx: &[f64],
    penalty: &[f64],
    lambda: f64,
    residuals: &FdMatrix,
    p: usize,
    n: usize,
) -> FdMatrix {
    let m = residuals.ncols();
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }
    let l = match cholesky_factor(&a, p) {
        Some(l) => l,
        None => return FdMatrix::zeros(p, m),
    };

    // Diagonal of A^{-1}
    let a_inv_diag: Vec<f64> = (0..p)
        .map(|j| {
            let mut ej = vec![0.0; p];
            ej[j] = 1.0;
            let v = cholesky_forward_back(&l, &ej, p);
            v[j]
        })
        .collect();

    let df = (n - p).max(1) as f64;
    let mut se = FdMatrix::zeros(p, m);
    for t in 0..m {
        let sigma2_t: f64 = (0..n).map(|i| residuals[(i, t)].powi(2)).sum::<f64>() / df;
        for j in 0..p {
            se[(j, t)] = (sigma2_t * a_inv_diag[j]).max(0.0).sqrt();
        }
    }
    se
}

// ---------------------------------------------------------------------------
// fosr_fpc: FPC-based function-on-scalar regression (matches R's fda.usc approach)
// ---------------------------------------------------------------------------

/// FPC-based function-on-scalar regression.
///
/// Reduces the functional response to FPC scores, regresses each score on the
/// scalar predictors via OLS, then reconstructs β(t) from the loadings.
/// This matches R's `fdata2pc` + `lm(scores ~ x)` approach.
///
/// # Arguments
/// * `data` - Functional response matrix (n × m)
/// * `predictors` - Scalar predictor matrix (n × p)
/// * `ncomp` - Number of FPC components to use
pub fn fosr_fpc(data: &FdMatrix, predictors: &FdMatrix, ncomp: usize) -> Option<FosrFpcResult> {
    let (n, m) = data.shape();
    let p = predictors.ncols();
    if n < p + 2 || m == 0 || predictors.nrows() != n || ncomp == 0 {
        return None;
    }

    // Step 1: FPCA on functional response
    let fpca = fdata_to_pc_1d(data, ncomp)?;
    let k = fpca.scores.ncols(); // actual components (may be < ncomp)

    // Step 2: Build design matrix [1 | predictors] for OLS on scores
    let p_total = p + 1; // intercept + p predictors
    let design = build_fosr_design(predictors, n);

    // Solve OLS: for each component, regress scores[:,comp] on design
    // gamma_all[comp] = (X'X)^{-1} X' scores[:,comp]  — length p_total
    let xtx = compute_xtx(&design);
    let l = cholesky_factor(&xtx, p_total)?;

    let mut gamma_all = vec![vec![0.0; p_total]; k]; // gamma_all[comp][coef]
    for comp in 0..k {
        let mut xts = vec![0.0; p_total];
        for j in 0..p_total {
            for i in 0..n {
                xts[j] += design[(i, j)] * fpca.scores[(i, comp)];
            }
        }
        gamma_all[comp] = cholesky_forward_back(&l, &xts, p_total);
    }

    // Step 3: Reconstruct beta functions
    // beta_j(t) = Σ_k gamma[comp][1+j] * rotation[t][comp]
    let mut beta = FdMatrix::zeros(p, m);
    for j in 0..p {
        for t in 0..m {
            let mut val = 0.0;
            for comp in 0..k {
                val += gamma_all[comp][1 + j] * fpca.rotation[(t, comp)];
            }
            beta[(j, t)] = val;
        }
    }

    // Step 4: Intercept function = mean(t) + Σ_k gamma_intercept[k] * rotation[t][k]
    let mut intercept = fpca.mean.clone();
    for t in 0..m {
        for comp in 0..k {
            intercept[t] += gamma_all[comp][0] * fpca.rotation[(t, comp)];
        }
    }

    // Step 5: Fitted values and residuals
    let mut fitted = FdMatrix::zeros(n, m);
    let mut residuals = FdMatrix::zeros(n, m);
    for i in 0..n {
        for t in 0..m {
            let mut yhat = intercept[t];
            for j in 0..p {
                yhat += predictors[(i, j)] * beta[(j, t)];
            }
            fitted[(i, t)] = yhat;
            residuals[(i, t)] = data[(i, t)] - yhat;
        }
    }

    let r_squared_t = pointwise_r_squared(data, &fitted);
    let r_squared = r_squared_t.iter().sum::<f64>() / m as f64;

    // Extract beta_scores: gamma[j][k] = gamma_all[k][1+j]
    // Scale by sqrt(h) to report in L²-normalized space (matching R's fdata2pc convention)
    let h = if m > 1 { 1.0 / (m - 1) as f64 } else { 1.0 };
    let score_scale = h.sqrt();
    let beta_scores: Vec<Vec<f64>> = (0..p)
        .map(|j| {
            (0..k)
                .map(|comp| gamma_all[comp][1 + j] * score_scale)
                .collect()
        })
        .collect();

    Some(FosrFpcResult {
        intercept,
        beta,
        fitted,
        residuals,
        r_squared_t,
        r_squared,
        beta_scores,
        ncomp: k,
    })
}

/// Predict new functional responses from a fitted FOSR model.
///
/// # Arguments
/// * `result` - Fitted [`FosrResult`]
/// * `new_predictors` - New scalar predictors (n_new × p)
pub fn predict_fosr(result: &FosrResult, new_predictors: &FdMatrix) -> FdMatrix {
    let n_new = new_predictors.nrows();
    let m = result.intercept.len();
    let p = result.beta.nrows();

    let mut predicted = FdMatrix::zeros(n_new, m);
    for i in 0..n_new {
        for t in 0..m {
            let mut yhat = result.intercept[t];
            for j in 0..p {
                yhat += new_predictors[(i, j)] * result.beta[(j, t)];
            }
            predicted[(i, t)] = yhat;
        }
    }
    predicted
}

// ---------------------------------------------------------------------------
// fanova: Functional ANOVA
// ---------------------------------------------------------------------------

/// Compute group means and overall mean.
fn compute_group_means(
    data: &FdMatrix,
    groups: &[usize],
    labels: &[usize],
) -> (FdMatrix, Vec<f64>) {
    let (n, m) = data.shape();
    let k = labels.len();
    let mut group_means = FdMatrix::zeros(k, m);
    let mut counts = vec![0usize; k];

    for i in 0..n {
        let g = labels.iter().position(|&l| l == groups[i]).unwrap_or(0);
        counts[g] += 1;
        for t in 0..m {
            group_means[(g, t)] += data[(i, t)];
        }
    }
    for g in 0..k {
        if counts[g] > 0 {
            for t in 0..m {
                group_means[(g, t)] /= counts[g] as f64;
            }
        }
    }

    let overall_mean: Vec<f64> = (0..m)
        .map(|t| (0..n).map(|i| data[(i, t)]).sum::<f64>() / n as f64)
        .collect();

    (group_means, overall_mean)
}

/// Compute pointwise F-statistic.
fn pointwise_f_statistic(
    data: &FdMatrix,
    groups: &[usize],
    labels: &[usize],
    group_means: &FdMatrix,
    overall_mean: &[f64],
) -> Vec<f64> {
    let (n, m) = data.shape();
    let k = labels.len();
    let mut counts = vec![0usize; k];
    for &g in groups {
        let idx = labels.iter().position(|&l| l == g).unwrap_or(0);
        counts[idx] += 1;
    }

    (0..m)
        .map(|t| {
            let ss_between: f64 = (0..k)
                .map(|g| counts[g] as f64 * (group_means[(g, t)] - overall_mean[t]).powi(2))
                .sum();
            let ss_within: f64 = (0..n)
                .map(|i| {
                    let g = labels.iter().position(|&l| l == groups[i]).unwrap_or(0);
                    (data[(i, t)] - group_means[(g, t)]).powi(2)
                })
                .sum();
            let ms_between = ss_between / (k as f64 - 1.0).max(1.0);
            let ms_within = ss_within / (n as f64 - k as f64).max(1.0);
            if ms_within > 1e-15 {
                ms_between / ms_within
            } else {
                0.0
            }
        })
        .collect()
}

/// Compute global test statistic (integrated F).
fn global_f_statistic(f_t: &[f64]) -> f64 {
    f_t.iter().sum::<f64>() / f_t.len() as f64
}

/// Functional ANOVA: test whether groups have different mean curves.
///
/// Uses a permutation-based global test with the integrated F-statistic.
///
/// # Arguments
/// * `data` - Functional response matrix (n × m)
/// * `groups` - Group labels for each observation (length n, integer-coded)
/// * `n_perm` - Number of permutations for the global test
///
/// # Returns
/// [`FanovaResult`] with group means, F-statistics, and permutation p-value
pub fn fanova(data: &FdMatrix, groups: &[usize], n_perm: usize) -> Option<FanovaResult> {
    let (n, m) = data.shape();
    if n < 3 || m == 0 || groups.len() != n {
        return None;
    }

    let mut labels: Vec<usize> = groups.to_vec();
    labels.sort();
    labels.dedup();
    let n_groups = labels.len();
    if n_groups < 2 {
        return None;
    }

    let (group_means, overall_mean) = compute_group_means(data, groups, &labels);
    let f_t = pointwise_f_statistic(data, groups, &labels, &group_means, &overall_mean);
    let observed_stat = global_f_statistic(&f_t);

    // Permutation test
    let n_perm = n_perm.max(1);
    let mut n_ge = 0usize;
    let mut perm_groups = groups.to_vec();

    // Simple LCG for reproducibility without requiring rand
    let mut rng_state: u64 = 42;
    for _ in 0..n_perm {
        // Fisher-Yates shuffle with LCG
        for i in (1..n).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state >> 33) as usize % (i + 1);
            perm_groups.swap(i, j);
        }

        let (perm_means, perm_overall) = compute_group_means(data, &perm_groups, &labels);
        let perm_f = pointwise_f_statistic(data, &perm_groups, &labels, &perm_means, &perm_overall);
        let perm_stat = global_f_statistic(&perm_f);
        if perm_stat >= observed_stat {
            n_ge += 1;
        }
    }

    let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);

    Some(FanovaResult {
        group_means,
        overall_mean,
        f_statistic_t: f_t,
        global_statistic: observed_stat,
        p_value,
        n_perm,
        n_groups,
        group_labels: labels,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    fn generate_fosr_data(n: usize, m: usize) -> (FdMatrix, FdMatrix) {
        let t = uniform_grid(m);
        let mut y = FdMatrix::zeros(n, m);
        let mut z = FdMatrix::zeros(n, 2);

        for i in 0..n {
            let age = (i as f64) / (n as f64);
            let group = if i % 2 == 0 { 1.0 } else { 0.0 };
            z[(i, 0)] = age;
            z[(i, 1)] = group;
            for j in 0..m {
                // True model: μ(t) + age * β₁(t) + group * β₂(t)
                let mu = (2.0 * PI * t[j]).sin();
                let beta1 = t[j]; // Linear coefficient for age
                let beta2 = (4.0 * PI * t[j]).cos(); // Oscillating for group
                y[(i, j)] = mu
                    + age * beta1
                    + group * beta2
                    + 0.05 * ((i * 13 + j * 7) % 100) as f64 / 100.0;
            }
        }
        (y, z)
    }

    // ----- FOSR tests -----

    #[test]
    fn test_fosr_basic() {
        let (y, z) = generate_fosr_data(30, 50);
        let result = fosr(&y, &z, 0.0);
        assert!(result.is_some());
        let fit = result.unwrap();
        assert_eq!(fit.intercept.len(), 50);
        assert_eq!(fit.beta.shape(), (2, 50));
        assert_eq!(fit.fitted.shape(), (30, 50));
        assert_eq!(fit.residuals.shape(), (30, 50));
        assert!(fit.r_squared >= 0.0);
    }

    #[test]
    fn test_fosr_with_penalty() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit0 = fosr(&y, &z, 0.0).unwrap();
        let fit1 = fosr(&y, &z, 1.0).unwrap();
        // Both should produce valid results
        assert_eq!(fit0.beta.shape(), (2, 50));
        assert_eq!(fit1.beta.shape(), (2, 50));
    }

    #[test]
    fn test_fosr_auto_lambda() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit = fosr(&y, &z, -1.0).unwrap();
        assert!(fit.lambda >= 0.0);
    }

    #[test]
    fn test_fosr_fitted_plus_residuals_equals_y() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit = fosr(&y, &z, 0.0).unwrap();
        for i in 0..30 {
            for t in 0..50 {
                let reconstructed = fit.fitted[(i, t)] + fit.residuals[(i, t)];
                assert!(
                    (reconstructed - y[(i, t)]).abs() < 1e-10,
                    "ŷ + r should equal y at ({}, {})",
                    i,
                    t
                );
            }
        }
    }

    #[test]
    fn test_fosr_pointwise_r_squared_valid() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit = fosr(&y, &z, 0.0).unwrap();
        for &r2 in &fit.r_squared_t {
            assert!(
                (-0.01..=1.0 + 1e-10).contains(&r2),
                "R²(t) out of range: {}",
                r2
            );
        }
    }

    #[test]
    fn test_fosr_se_positive() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit = fosr(&y, &z, 0.0).unwrap();
        for j in 0..2 {
            for t in 0..50 {
                assert!(
                    fit.beta_se[(j, t)] >= 0.0 && fit.beta_se[(j, t)].is_finite(),
                    "SE should be non-negative finite"
                );
            }
        }
    }

    #[test]
    fn test_fosr_invalid_input() {
        let y = FdMatrix::zeros(2, 50);
        let z = FdMatrix::zeros(2, 1);
        assert!(fosr(&y, &z, 0.0).is_none());
    }

    // ----- predict_fosr tests -----

    #[test]
    fn test_predict_fosr_on_training_data() {
        let (y, z) = generate_fosr_data(30, 50);
        let fit = fosr(&y, &z, 0.0).unwrap();
        let preds = predict_fosr(&fit, &z);
        assert_eq!(preds.shape(), (30, 50));
        for i in 0..30 {
            for t in 0..50 {
                assert!(
                    (preds[(i, t)] - fit.fitted[(i, t)]).abs() < 1e-8,
                    "Prediction on training data should match fitted"
                );
            }
        }
    }

    // ----- FANOVA tests -----

    #[test]
    fn test_fanova_two_groups() {
        let n = 40;
        let m = 50;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        let mut groups = vec![0usize; n];
        for i in 0..n {
            groups[i] = if i < n / 2 { 0 } else { 1 };
            for j in 0..m {
                let base = (2.0 * PI * t[j]).sin();
                let effect = if groups[i] == 1 { 0.5 * t[j] } else { 0.0 };
                data[(i, j)] = base + effect + 0.01 * (i as f64 * 0.1).sin();
            }
        }

        let result = fanova(&data, &groups, 200);
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.n_groups, 2);
        assert_eq!(res.group_means.shape(), (2, m));
        assert_eq!(res.f_statistic_t.len(), m);
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
        // With a real group effect, p should be small
        assert!(
            res.p_value < 0.1,
            "Should detect group effect, got p={}",
            res.p_value
        );
    }

    #[test]
    fn test_fanova_no_effect() {
        let n = 40;
        let m = 50;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        let mut groups = vec![0usize; n];
        for i in 0..n {
            groups[i] = if i < n / 2 { 0 } else { 1 };
            for j in 0..m {
                // Same distribution for both groups
                data[(i, j)] =
                    (2.0 * PI * t[j]).sin() + 0.1 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }

        let result = fanova(&data, &groups, 200);
        assert!(result.is_some());
        let res = result.unwrap();
        // Without group effect, p should be large
        assert!(
            res.p_value > 0.05,
            "Should not detect effect, got p={}",
            res.p_value
        );
    }

    #[test]
    fn test_fanova_three_groups() {
        let n = 30;
        let m = 50;
        let t = uniform_grid(m);

        let mut data = FdMatrix::zeros(n, m);
        let mut groups = vec![0usize; n];
        for i in 0..n {
            groups[i] = i % 3;
            for j in 0..m {
                let effect = match groups[i] {
                    0 => 0.0,
                    1 => 0.5 * t[j],
                    _ => -0.3 * (2.0 * PI * t[j]).cos(),
                };
                data[(i, j)] = (2.0 * PI * t[j]).sin() + effect + 0.01 * (i as f64 * 0.1).sin();
            }
        }

        let result = fanova(&data, &groups, 200);
        assert!(result.is_some());
        let res = result.unwrap();
        assert_eq!(res.n_groups, 3);
    }

    #[test]
    fn test_fanova_invalid_input() {
        let data = FdMatrix::zeros(10, 50);
        let groups = vec![0; 10]; // Only one group
        assert!(fanova(&data, &groups, 100).is_none());

        let groups = vec![0; 5]; // Wrong length
        assert!(fanova(&data, &groups, 100).is_none());
    }
}
