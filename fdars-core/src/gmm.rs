//! Model-based functional clustering via Gaussian mixture models.
//!
//! Implements the fdaMocca approach (Arnqvist & Sjöstedt de Luna, 2023):
//! project curves onto a basis, concatenate with scalar covariates, and fit
//! a Gaussian mixture using EM.
//!
//! Key functions:
//! - [`gmm_cluster`] — Main clustering entry point with automatic K selection
//! - [`gmm_em`] — Single-K EM algorithm
//! - [`predict_gmm`] — Assign new observations to clusters

use crate::basis::fdata_to_basis_1d;
use crate::matrix::FdMatrix;
use rand::prelude::*;
use std::f64::consts::PI;

/// Covariance structure for GMM components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CovType {
    /// Full covariance matrix (d² parameters per component)
    Full,
    /// Diagonal covariance (d parameters per component)
    Diagonal,
}

/// Result from a single GMM fit with fixed K.
pub struct GmmResult {
    /// Hard cluster assignments (length n)
    pub cluster: Vec<usize>,
    /// Posterior membership probabilities (n x K)
    pub membership: FdMatrix,
    /// Component means (K x d)
    pub means: Vec<Vec<f64>>,
    /// Component covariances: for Full, each is d×d flattened; for Diagonal, each is length d
    pub covariances: Vec<Vec<f64>>,
    /// Mixing proportions (length K)
    pub weights: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// BIC value
    pub bic: f64,
    /// ICL value (BIC penalized by entropy)
    pub icl: f64,
    /// Number of EM iterations
    pub iterations: usize,
    /// Whether EM converged
    pub converged: bool,
    /// Number of clusters
    pub k: usize,
    /// Feature dimension (basis coefficients + covariates)
    pub d: usize,
}

/// Result from automatic K selection.
pub struct GmmClusterResult {
    /// Best GMM result (by BIC or ICL)
    pub best: GmmResult,
    /// BIC values for each K tried
    pub bic_values: Vec<(usize, f64)>,
    /// ICL values for each K tried
    pub icl_values: Vec<(usize, f64)>,
}

// ---------------------------------------------------------------------------
// Cholesky helpers for GMM covariance operations
// ---------------------------------------------------------------------------

/// Cholesky factorization of a d×d matrix (row-major flat). Returns lower-triangular L.
fn cholesky_d(mat: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; d * d];
    for j in 0..d {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j * d + k] * l[j * d + k];
        }
        let diag = mat[j * d + j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j * d + j] = diag.sqrt();
        for i in (j + 1)..d {
            let mut s = 0.0;
            for k in 0..j {
                s += l[i * d + k] * l[j * d + k];
            }
            l[i * d + j] = (mat[i * d + j] - s) / l[j * d + j];
        }
    }
    Some(l)
}

/// Log-determinant from Cholesky factor L: 2 * sum(log(L_ii)).
fn log_det_from_cholesky(l: &[f64], d: usize) -> f64 {
    let mut s = 0.0;
    for i in 0..d {
        s += l[i * d + i].ln();
    }
    2.0 * s
}

/// Solve L * x = b (forward substitution).
fn forward_solve(l: &[f64], b: &[f64], d: usize) -> Vec<f64> {
    let mut x = vec![0.0; d];
    for i in 0..d {
        let mut s = 0.0;
        for j in 0..i {
            s += l[i * d + j] * x[j];
        }
        x[i] = (b[i] - s) / l[i * d + i];
    }
    x
}

/// Compute (z - mu)^T Sigma^{-1} (z - mu) using Cholesky factor L.
/// Also returns log|Sigma| = log_det_from_cholesky(L).
fn mahalanobis_sq(z: &[f64], mu: &[f64], chol: &[f64], d: usize) -> f64 {
    let diff: Vec<f64> = z.iter().zip(mu.iter()).map(|(&a, &b)| a - b).collect();
    let y = forward_solve(chol, &diff, d);
    y.iter().map(|&v| v * v).sum()
}

// ---------------------------------------------------------------------------
// Feature extraction: basis coefficients + optional covariates
// ---------------------------------------------------------------------------

/// Build feature matrix: project curves onto basis, optionally append covariates.
/// Returns (feature_matrix as Vec<Vec<f64>>, dimension d).
fn build_features(
    data: &FdMatrix,
    argvals: &[f64],
    covariates: Option<&FdMatrix>,
    nbasis: usize,
    basis_type: i32,
    cov_weight: f64,
) -> Option<(Vec<Vec<f64>>, usize)> {
    let n = data.nrows();
    let proj = fdata_to_basis_1d(data, argvals, nbasis, basis_type)?;
    let coef = &proj.coefficients;
    let d_basis = coef.ncols();

    let d_cov = covariates.map_or(0, |c| c.ncols());
    let d = d_basis + d_cov;

    let mut features = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(d);
        for j in 0..d_basis {
            row.push(coef[(i, j)]);
        }
        if let Some(cov) = covariates {
            for j in 0..d_cov {
                row.push(cov[(i, j)] * cov_weight);
            }
        }
        features.push(row);
    }

    Some((features, d))
}

// ---------------------------------------------------------------------------
// GMM initialization: k-means++ on feature vectors
// ---------------------------------------------------------------------------

/// Euclidean distance squared between two feature vectors.
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
}

/// K-means++ initialization on feature vectors. Returns initial means.
fn kmeans_pp_init(features: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<Vec<f64>> {
    let n = features.len();
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    centers.push(features[rng.gen_range(0..n)].clone());

    let mut min_dists = vec![f64::INFINITY; n];
    for c_idx in 1..k {
        let last = &centers[c_idx - 1];
        for i in 0..n {
            let d = dist_sq(&features[i], last);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        let total: f64 = min_dists.iter().sum();
        if total < 1e-15 {
            centers.push(features[rng.gen_range(0..n)].clone());
            continue;
        }
        let r = rng.gen::<f64>() * total;
        let mut cum = 0.0;
        let mut chosen = n - 1;
        for i in 0..n {
            cum += min_dists[i];
            if cum >= r {
                chosen = i;
                break;
            }
        }
        centers.push(features[chosen].clone());
    }
    centers
}

/// Run a few k-means iterations to get initial cluster assignments.
fn kmeans_init_assignments(features: &[Vec<f64>], k: usize, rng: &mut StdRng) -> Vec<usize> {
    let n = features.len();
    let mut centers = kmeans_pp_init(features, k, rng);

    // Run 10 iterations of Lloyd's algorithm
    let mut assignments = vec![0usize; n];
    for _ in 0..10 {
        // Assign
        for i in 0..n {
            let mut best = 0;
            let mut best_d = f64::INFINITY;
            for c in 0..k {
                let d = dist_sq(&features[i], &centers[c]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assignments[i] = best;
        }
        // Update centers
        let d = features[0].len();
        let mut counts = vec![0usize; k];
        let mut new_centers = vec![vec![0.0; d]; k];
        for i in 0..n {
            let c = assignments[i];
            counts[c] += 1;
            for j in 0..d {
                new_centers[c][j] += features[i][j];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d {
                    new_centers[c][j] /= counts[c] as f64;
                }
            } else {
                new_centers[c] = centers[c].clone();
            }
        }
        centers = new_centers;
    }
    assignments
}

// ---------------------------------------------------------------------------
// EM algorithm
// ---------------------------------------------------------------------------

/// Initialize GMM parameters from k-means assignments.
fn init_params_from_assignments(
    features: &[Vec<f64>],
    assignments: &[usize],
    k: usize,
    d: usize,
    cov_type: CovType,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = features.len();
    let mut counts = vec![0usize; k];
    let mut means = vec![vec![0.0; d]; k];

    for i in 0..n {
        let c = assignments[i];
        counts[c] += 1;
        for j in 0..d {
            means[c][j] += features[i][j];
        }
    }
    for c in 0..k {
        if counts[c] > 0 {
            for j in 0..d {
                means[c][j] /= counts[c] as f64;
            }
        }
    }

    let reg = 1e-6; // regularization
    let covariances = compute_covariances(features, assignments, &means, k, d, cov_type, reg);

    let weights: Vec<f64> = counts.iter().map(|&c| c.max(1) as f64 / n as f64).collect();

    (means, covariances, weights)
}

/// Accumulate full covariance from unit-weighted observations.
fn accumulate_full_cov(
    features: &[Vec<f64>],
    indices: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for &i in indices {
        for r in 0..d {
            let dr = features[i][r] - mean[r];
            for s in r..d {
                let val = dr * (features[i][s] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Accumulate diagonal covariance from unit-weighted observations.
fn accumulate_diag_cov(
    features: &[Vec<f64>],
    indices: &[usize],
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; d];
    for &i in indices {
        for j in 0..d {
            diag[j] += (features[i][j] - mean[j]).powi(2);
        }
    }
    diag
}

/// Accumulate weighted full covariance from all observations.
fn accumulate_full_cov_weighted(
    features: &[Vec<f64>],
    resp: &[f64],
    c: usize,
    k: usize,
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut cov = vec![0.0; d * d];
    for i in 0..features.len() {
        let w = resp[i * k + c];
        for r in 0..d {
            let dr = features[i][r] - mean[r];
            for s in r..d {
                let val = w * dr * (features[i][s] - mean[s]);
                cov[r * d + s] += val;
                if r != s {
                    cov[s * d + r] += val;
                }
            }
        }
    }
    cov
}

/// Accumulate weighted diagonal covariance from all observations.
fn accumulate_diag_cov_weighted(
    features: &[Vec<f64>],
    resp: &[f64],
    c: usize,
    k: usize,
    mean: &[f64],
    d: usize,
) -> Vec<f64> {
    let mut diag = vec![0.0; d];
    for i in 0..features.len() {
        let w = resp[i * k + c];
        for j in 0..d {
            diag[j] += w * (features[i][j] - mean[j]).powi(2);
        }
    }
    diag
}

/// Normalize covariance by count and add regularization.
fn regularize_cov(cov: &mut [f64], scale: f64, d: usize, reg: f64, is_full: bool) {
    for v in cov.iter_mut() {
        *v /= scale;
    }
    if is_full {
        for j in 0..d {
            cov[j * d + j] += reg;
        }
    } else {
        for v in cov.iter_mut() {
            *v += reg;
        }
    }
}

/// Identity-like regularization covariance (for degenerate components).
fn identity_cov(d: usize, reg: f64, cov_type: CovType) -> Vec<f64> {
    match cov_type {
        CovType::Full => {
            let mut cov = vec![0.0; d * d];
            for j in 0..d {
                cov[j * d + j] = reg;
            }
            cov
        }
        CovType::Diagonal => vec![reg; d],
    }
}

/// Compute covariances from hard assignments.
fn compute_covariances(
    features: &[Vec<f64>],
    assignments: &[usize],
    means: &[Vec<f64>],
    k: usize,
    d: usize,
    cov_type: CovType,
    reg: f64,
) -> Vec<Vec<f64>> {
    let n = features.len();
    let mut covariances = Vec::with_capacity(k);

    for c in 0..k {
        let members: Vec<usize> = (0..n).filter(|&i| assignments[i] == c).collect();
        let nc = members.len().max(1) as f64;

        let mut cov = match cov_type {
            CovType::Full => accumulate_full_cov(features, &members, &means[c], d),
            CovType::Diagonal => accumulate_diag_cov(features, &members, &means[c], d),
        };
        regularize_cov(&mut cov, nc, d, reg, cov_type == CovType::Full);
        covariances.push(cov);
    }
    covariances
}

/// Compute log-density of z under component c.
fn log_component_density(z: &[f64], mean: &[f64], cov: &[f64], d: usize, cov_type: CovType) -> f64 {
    let log_2pi = (2.0 * PI).ln();

    match cov_type {
        CovType::Full => {
            let chol = match cholesky_d(cov, d) {
                Some(l) => l,
                None => return f64::NEG_INFINITY,
            };
            let log_det = log_det_from_cholesky(&chol, d);
            let maha = mahalanobis_sq(z, mean, &chol, d);
            -0.5 * (d as f64 * log_2pi + log_det + maha)
        }
        CovType::Diagonal => {
            let mut log_det = 0.0;
            let mut maha = 0.0;
            for j in 0..d {
                if cov[j] <= 0.0 {
                    return f64::NEG_INFINITY;
                }
                log_det += cov[j].ln();
                maha += (z[j] - mean[j]).powi(2) / cov[j];
            }
            -0.5 * (d as f64 * log_2pi + log_det + maha)
        }
    }
}

/// E-step: compute log-responsibilities and log-likelihood.
/// Returns (responsibilities as n×k flat row-major, log_likelihood).
fn e_step(
    features: &[Vec<f64>],
    means: &[Vec<f64>],
    covariances: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    d: usize,
    cov_type: CovType,
) -> (Vec<f64>, f64) {
    let n = features.len();
    let mut resp = vec![0.0; n * k];
    let mut ll = 0.0;

    for i in 0..n {
        // Compute log(π_c * N(z_i | μ_c, Σ_c)) for each c
        let mut log_probs = vec![f64::NEG_INFINITY; k];
        for c in 0..k {
            if weights[c] > 1e-15 {
                log_probs[c] = weights[c].ln()
                    + log_component_density(&features[i], &means[c], &covariances[c], d, cov_type);
            }
        }

        // Log-sum-exp for numerical stability
        let max_lp = log_probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if max_lp == f64::NEG_INFINITY {
            // All components give -inf density; assign uniform
            for c in 0..k {
                resp[i * k + c] = 1.0 / k as f64;
            }
            continue;
        }

        let lse = max_lp
            + log_probs
                .iter()
                .map(|&lp| (lp - max_lp).exp())
                .sum::<f64>()
                .ln();
        ll += lse;

        for c in 0..k {
            resp[i * k + c] = (log_probs[c] - lse).exp();
        }
    }

    (resp, ll)
}

/// M-step: update means, covariances, and weights from responsibilities.
fn m_step(
    features: &[Vec<f64>],
    resp: &[f64],
    k: usize,
    d: usize,
    cov_type: CovType,
    reg: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
    let n = features.len();
    let n_f64 = n as f64;

    let mut means = vec![vec![0.0; d]; k];
    let mut weights = vec![0.0; k];

    // Compute effective counts and means
    for c in 0..k {
        let mut nk = 0.0;
        for i in 0..n {
            nk += resp[i * k + c];
        }
        weights[c] = nk / n_f64;

        if nk > 1e-15 {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..n {
                    s += resp[i * k + c] * features[i][j];
                }
                means[c][j] = s / nk;
            }
        }
    }

    // Compute covariances
    let covariances =
        m_step_covariances(features, resp, &means, &weights, k, d, cov_type, reg, n_f64);

    (means, covariances, weights)
}

/// M-step covariance computation using soft responsibilities.
fn m_step_covariances(
    features: &[Vec<f64>],
    resp: &[f64],
    means: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    d: usize,
    cov_type: CovType,
    reg: f64,
    n_f64: f64,
) -> Vec<Vec<f64>> {
    let mut covariances = Vec::with_capacity(k);

    for c in 0..k {
        let nk = weights[c] * n_f64;
        if nk < 1e-15 {
            covariances.push(identity_cov(d, reg, cov_type));
            continue;
        }

        let mut cov = match cov_type {
            CovType::Full => accumulate_full_cov_weighted(features, resp, c, k, &means[c], d),
            CovType::Diagonal => accumulate_diag_cov_weighted(features, resp, c, k, &means[c], d),
        };
        regularize_cov(&mut cov, nk, d, reg, cov_type == CovType::Full);
        covariances.push(cov);
    }
    covariances
}

/// Count free parameters in the GMM.
fn count_params(k: usize, d: usize, cov_type: CovType) -> usize {
    let mean_params = k * d;
    let weight_params = k - 1;
    let cov_params = match cov_type {
        CovType::Full => k * d * (d + 1) / 2,
        CovType::Diagonal => k * d,
    };
    mean_params + weight_params + cov_params
}

/// Compute BIC = -2*LL + p*ln(n).
fn compute_bic(ll: f64, n: usize, n_params: usize) -> f64 {
    -2.0 * ll + n_params as f64 * (n as f64).ln()
}

/// Compute ICL = BIC + 2*entropy(responsibilities).
fn compute_icl(bic: f64, resp: &[f64], n: usize, k: usize) -> f64 {
    let mut entropy = 0.0;
    for i in 0..n {
        for c in 0..k {
            let r = resp[i * k + c];
            if r > 1e-15 {
                entropy -= r * r.ln();
            }
        }
    }
    bic + 2.0 * entropy
}

/// Run EM for a Gaussian mixture model on pre-extracted feature vectors.
///
/// # Arguments
/// * `features` — Feature vectors (n × d)
/// * `k` — Number of clusters
/// * `cov_type` — Covariance structure
/// * `max_iter` — Maximum EM iterations
/// * `tol` — Log-likelihood convergence tolerance
/// * `seed` — Random seed
///
/// # Returns
/// `GmmResult` with cluster assignments, membership, parameters, and model selection criteria.
pub fn gmm_em(
    features: &[Vec<f64>],
    k: usize,
    cov_type: CovType,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Option<GmmResult> {
    let n = features.len();
    if n == 0 || k == 0 || k > n {
        return None;
    }
    let d = features[0].len();
    if d == 0 {
        return None;
    }

    let mut rng = StdRng::seed_from_u64(seed);

    // Initialize via k-means
    let assignments = kmeans_init_assignments(features, k, &mut rng);
    let (mut means, mut covariances, mut weights) =
        init_params_from_assignments(features, &assignments, k, d, cov_type);

    let reg = 1e-6;
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0;
    let mut resp = vec![0.0; n * k];

    for iter in 0..max_iter {
        iterations = iter + 1;

        let (new_resp, ll) = e_step(features, &means, &covariances, &weights, k, d, cov_type);
        resp = new_resp;

        if (ll - prev_ll).abs() < tol && iter > 0 {
            converged = true;
            break;
        }
        prev_ll = ll;

        let (new_means, new_covs, new_weights) = m_step(features, &resp, k, d, cov_type, reg);
        means = new_means;
        covariances = new_covs;
        weights = new_weights;
    }

    // Final E-step for log-likelihood
    let (final_resp, log_likelihood) =
        e_step(features, &means, &covariances, &weights, k, d, cov_type);
    resp = final_resp;

    let n_params = count_params(k, d, cov_type);
    let bic = compute_bic(log_likelihood, n, n_params);
    let icl = compute_icl(bic, &resp, n, k);

    // Hard assignments and membership matrix
    let cluster = hard_assignments(&resp, n, k);
    let membership = resp_to_membership(&resp, n, k);

    Some(GmmResult {
        cluster,
        membership,
        means,
        covariances,
        weights,
        log_likelihood,
        bic,
        icl,
        iterations,
        converged,
        k,
        d,
    })
}

/// Convert flat responsibilities to hard cluster assignments.
fn hard_assignments(resp: &[f64], n: usize, k: usize) -> Vec<usize> {
    (0..n)
        .map(|i| {
            let mut best = 0;
            let mut best_p = f64::NEG_INFINITY;
            for c in 0..k {
                if resp[i * k + c] > best_p {
                    best_p = resp[i * k + c];
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Convert flat row-major responsibilities to FdMatrix (n × k).
fn resp_to_membership(resp: &[f64], n: usize, k: usize) -> FdMatrix {
    // resp is row-major (n × k), FdMatrix is column-major
    let mut col_major = vec![0.0; n * k];
    for i in 0..n {
        for c in 0..k {
            col_major[i + c * n] = resp[i * k + c];
        }
    }
    FdMatrix::from_column_major(col_major, n, k).unwrap()
}

/// Main clustering function: project curves onto basis, concatenate covariates,
/// and fit GMM with automatic K selection.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `covariates` — Optional scalar covariates (n × p)
/// * `k_range` — Range of K values to try (e.g., `2..=5`)
/// * `nbasis` — Number of basis functions for projection
/// * `basis_type` — 0 = B-spline, 1 = Fourier
/// * `cov_type` — Covariance structure
/// * `cov_weight` — Scaling factor for covariates (default 1.0)
/// * `max_iter` — Maximum EM iterations per K
/// * `tol` — Convergence tolerance
/// * `n_init` — Number of random initializations per K
/// * `seed` — Base random seed
/// * `use_icl` — If true, select K by ICL; otherwise by BIC
pub fn gmm_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    covariates: Option<&FdMatrix>,
    k_range: &[usize],
    nbasis: usize,
    basis_type: i32,
    cov_type: CovType,
    cov_weight: f64,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    seed: u64,
    use_icl: bool,
) -> Option<GmmClusterResult> {
    let (features, _d) = build_features(data, argvals, covariates, nbasis, basis_type, cov_weight)?;

    let mut bic_values = Vec::new();
    let mut icl_values = Vec::new();
    let mut best_result: Option<GmmResult> = None;
    let mut best_criterion = f64::INFINITY;

    for &k in k_range {
        let best_for_k = run_multiple_inits(&features, k, cov_type, max_iter, tol, n_init, seed);
        let result = match best_for_k {
            Some(r) => r,
            None => continue,
        };

        bic_values.push((k, result.bic));
        icl_values.push((k, result.icl));

        let criterion = if use_icl { result.icl } else { result.bic };
        if criterion < best_criterion {
            best_criterion = criterion;
            best_result = Some(result);
        }
    }

    best_result.map(|best| GmmClusterResult {
        best,
        bic_values,
        icl_values,
    })
}

/// Run multiple initializations for a single K and return the best by log-likelihood.
fn run_multiple_inits(
    features: &[Vec<f64>],
    k: usize,
    cov_type: CovType,
    max_iter: usize,
    tol: f64,
    n_init: usize,
    base_seed: u64,
) -> Option<GmmResult> {
    let mut best: Option<GmmResult> = None;
    for init in 0..n_init.max(1) {
        let seed = base_seed.wrapping_add(init as u64 * 1000 + k as u64);
        if let Some(result) = gmm_em(features, k, cov_type, max_iter, tol, seed) {
            let is_better = best
                .as_ref()
                .map_or(true, |b| result.log_likelihood > b.log_likelihood);
            if is_better {
                best = Some(result);
            }
        }
    }
    best
}

/// Predict cluster assignments for new observations.
///
/// # Arguments
/// * `new_data` — New functional data (n_new × m)
/// * `argvals` — Evaluation points
/// * `new_covariates` — Optional scalar covariates for new data
/// * `result` — Fitted GMM result
/// * `nbasis` — Number of basis functions (must match training)
/// * `basis_type` — Basis type (must match training)
/// * `cov_weight` — Covariate weight (must match training)
/// * `cov_type` — Covariance type (must match training)
pub fn predict_gmm(
    new_data: &FdMatrix,
    argvals: &[f64],
    new_covariates: Option<&FdMatrix>,
    result: &GmmResult,
    nbasis: usize,
    basis_type: i32,
    cov_weight: f64,
    cov_type: CovType,
) -> Option<(Vec<usize>, FdMatrix)> {
    let (features, _d) = build_features(
        new_data,
        argvals,
        new_covariates,
        nbasis,
        basis_type,
        cov_weight,
    )?;

    let k = result.k;
    let d = result.d;
    let n = features.len();

    let (resp, _ll) = e_step(
        &features,
        &result.means,
        &result.covariances,
        &result.weights,
        k,
        d,
        cov_type,
    );

    let cluster = hard_assignments(&resp, n, k);
    let membership = resp_to_membership(&resp, n, k);

    Some((cluster, membership))
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

    /// Generate two clearly separated clusters of curves.
    fn generate_two_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let n = 2 * n_per;
        let mut col_major = vec![0.0; n * m];

        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                // Cluster 0: sin with small noise
                col_major[i + j * n] =
                    (2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }
        for i in 0..n_per {
            for (j, &tj) in t.iter().enumerate() {
                // Cluster 1: shifted sin
                col_major[(i + n_per) + j * n] =
                    (2.0 * PI * tj).sin() + 5.0 + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
            }
        }
        (FdMatrix::from_column_major(col_major, n, m).unwrap(), t)
    }

    #[test]
    fn test_gmm_em_basic() {
        let (data, t) = generate_two_clusters(15, 50);
        let features =
            build_features(&data, &t, None, 8, 0, 1.0).expect("Feature extraction failed");
        let result = gmm_em(&features.0, 2, CovType::Full, 100, 1e-6, 42).unwrap();

        assert_eq!(result.cluster.len(), 30);
        assert_eq!(result.k, 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_gmm_em_finds_clusters() {
        // Test on synthetic Gaussian features directly (bypasses basis projection)
        let n_per = 30;
        let n = 2 * n_per;
        let mut features = Vec::with_capacity(n);

        // Cluster 0: mean = [0, 0, 0]
        for i in 0..n_per {
            let noise = (i as f64 * 0.1).sin() * 0.3;
            features.push(vec![noise, noise * 0.5, -noise * 0.7]);
        }
        // Cluster 1: mean = [5, 5, 5]
        for i in 0..n_per {
            let noise = (i as f64 * 0.1).sin() * 0.3;
            features.push(vec![5.0 + noise, 5.0 + noise * 0.5, 5.0 - noise * 0.7]);
        }

        let result = gmm_em(&features, 2, CovType::Diagonal, 200, 1e-6, 42).unwrap();

        let c0 = result.cluster[0];
        let c1 = result.cluster[n_per];
        assert_ne!(c0, c1, "Two clusters should be separated");

        let correct_first = (0..n_per).filter(|&i| result.cluster[i] == c0).count();
        let correct_second = (n_per..2 * n_per)
            .filter(|&i| result.cluster[i] == c1)
            .count();
        assert!(
            correct_first >= n_per - 1,
            "First cluster mostly correct: {}/{}",
            correct_first,
            n_per
        );
        assert!(
            correct_second >= n_per - 1,
            "Second cluster mostly correct: {}/{}",
            correct_second,
            n_per
        );
    }

    #[test]
    fn test_gmm_em_diagonal_covariance() {
        let (data, t) = generate_two_clusters(15, 50);
        let (features, _d) = build_features(&data, &t, None, 8, 0, 1.0).unwrap();

        let result = gmm_em(&features, 2, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        assert_eq!(result.cluster.len(), 30);

        // Diagonal should have fewer parameters → BIC advantage for simpler models
        let result_full = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();
        // Both should find clusters
        assert_eq!(result_full.cluster.len(), 30);
    }

    #[test]
    fn test_gmm_membership_sums_to_one() {
        let (data, t) = generate_two_clusters(10, 50);
        let (features, _d) = build_features(&data, &t, None, 6, 0, 1.0).unwrap();

        let result = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

        let n = result.membership.nrows();
        let k = result.membership.ncols();
        for i in 0..n {
            let sum: f64 = (0..k).map(|c| result.membership[(i, c)]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Membership should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_gmm_bic_icl_finite() {
        let (data, t) = generate_two_clusters(10, 50);
        let (features, _d) = build_features(&data, &t, None, 6, 0, 1.0).unwrap();

        let result = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

        assert!(result.bic.is_finite(), "BIC should be finite");
        assert!(result.icl.is_finite(), "ICL should be finite");
        assert!(
            result.icl >= result.bic,
            "ICL >= BIC (ICL adds entropy penalty)"
        );
    }

    #[test]
    fn test_gmm_cluster_auto_k() {
        // Use pseudo-random Gaussian noise via Box-Muller
        let n_per = 50;
        let n = 2 * n_per;
        let mut features = Vec::with_capacity(n);
        let mut rng = StdRng::seed_from_u64(99);

        for _ in 0..n_per {
            let u1: f64 = rng.gen::<f64>().max(1e-15);
            let u2: f64 = rng.gen::<f64>();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
            features.push(vec![z1 * 0.5, z2 * 0.5]);
        }
        for _ in 0..n_per {
            let u1: f64 = rng.gen::<f64>().max(1e-15);
            let u2: f64 = rng.gen::<f64>();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
            features.push(vec![6.0 + z1 * 0.5, 6.0 + z2 * 0.5]);
        }

        let mut best_bic = f64::INFINITY;
        let mut best_k = 0;
        let mut bic_values = Vec::new();
        for k in 1..=4 {
            let r = run_multiple_inits(&features, k, CovType::Diagonal, 100, 1e-6, 3, 42).unwrap();
            bic_values.push((k, r.bic));
            if r.bic < best_bic {
                best_bic = r.bic;
                best_k = k;
            }
        }

        assert_eq!(bic_values.len(), 4);
        assert_eq!(
            best_k, 2,
            "Should select K=2 for well-separated data, BICs: {:?}",
            bic_values
        );
    }

    #[test]
    fn test_gmm_with_covariates() {
        // Test that appending a covariate dimension to features separates clusters
        let n_per = 25;
        let n = 2 * n_per;

        // Base features identical for both groups, covariate separates them
        let mut features = Vec::with_capacity(n);
        for i in 0..n_per {
            let noise = (i as f64 * 0.1).sin() * 0.1;
            features.push(vec![noise, noise * 0.5, 0.0]); // covariate = 0
        }
        for i in 0..n_per {
            let noise = (i as f64 * 0.1).sin() * 0.1;
            features.push(vec![noise, noise * 0.5, 10.0]); // covariate = 10
        }

        let result = gmm_em(&features, 2, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        assert_eq!(result.cluster.len(), n);

        let c0 = result.cluster[0];
        let correct = (0..n_per).filter(|&i| result.cluster[i] == c0).count()
            + (n_per..n).filter(|&i| result.cluster[i] != c0).count();
        assert!(
            correct >= n - 2,
            "Covariate-based separation: {}/{} correct",
            correct,
            n
        );
    }

    #[test]
    fn test_predict_gmm() {
        let n_per = 15;
        let (data, t) = generate_two_clusters(n_per, 50);
        let nbasis = 8;
        let basis_type = 0;

        let result = gmm_cluster(
            &data,
            &t,
            None,
            &[2],
            nbasis,
            basis_type,
            CovType::Diagonal,
            1.0,
            100,
            1e-6,
            1,
            42,
            false,
        )
        .unwrap();

        // Predict on training data — should mostly match
        let (pred_cluster, pred_mem) = predict_gmm(
            &data,
            &t,
            None,
            &result.best,
            nbasis,
            basis_type,
            1.0,
            CovType::Diagonal,
        )
        .unwrap();

        assert_eq!(pred_cluster.len(), 2 * n_per);
        assert_eq!(pred_mem.nrows(), 2 * n_per);
        assert_eq!(pred_mem.ncols(), 2);

        let matching: usize = pred_cluster
            .iter()
            .zip(&result.best.cluster)
            .filter(|(&a, &b)| a == b)
            .count();
        assert!(
            matching >= 2 * n_per - 3,
            "Predict on training data should mostly match: {}/{}",
            matching,
            2 * n_per
        );
    }

    #[test]
    fn test_gmm_em_invalid_input() {
        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(gmm_em(&empty, 2, CovType::Full, 100, 1e-6, 42).is_none());

        let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        // k > n
        assert!(gmm_em(&features, 5, CovType::Full, 100, 1e-6, 42).is_none());
        // k = 0
        assert!(gmm_em(&features, 0, CovType::Full, 100, 1e-6, 42).is_none());
    }

    #[test]
    fn test_gmm_deterministic() {
        let (data, t) = generate_two_clusters(10, 50);
        let (features, _d) = build_features(&data, &t, None, 6, 0, 1.0).unwrap();

        let r1 = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();
        let r2 = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

        assert_eq!(r1.cluster, r2.cluster);
        assert!((r1.log_likelihood - r2.log_likelihood).abs() < 1e-10);
    }

    #[test]
    fn test_count_params() {
        // K=2, d=3, Full: means=6, weights=1, cov=2*(3*4/2)=12 → 19
        assert_eq!(count_params(2, 3, CovType::Full), 19);
        // K=2, d=3, Diagonal: means=6, weights=1, cov=2*3=6 → 13
        assert_eq!(count_params(2, 3, CovType::Diagonal), 13);
    }

    #[test]
    fn test_gmm_k1() {
        let (data, t) = generate_two_clusters(10, 50);
        let (features, _d) = build_features(&data, &t, None, 6, 0, 1.0).unwrap();

        let result = gmm_em(&features, 1, CovType::Full, 100, 1e-6, 42).unwrap();
        assert!(result.cluster.iter().all(|&c| c == 0));
        assert!(result.converged);
    }

    #[test]
    fn test_gmm_weights_sum_to_one() {
        let (data, t) = generate_two_clusters(10, 50);
        let (features, _d) = build_features(&data, &t, None, 6, 0, 1.0).unwrap();

        let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Mixing weights should sum to 1, got {}",
            sum
        );
    }
}
