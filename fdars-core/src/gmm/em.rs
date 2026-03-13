//! EM algorithm for Gaussian mixture models.

use super::covariance::{
    accumulate_diag_cov_weighted, accumulate_full_cov_weighted, identity_cov, regularize_cov,
};
use super::init::{init_params_from_assignments, kmeans_init_assignments};
use super::{CovType, GmmResult};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::linalg::{cholesky_d, log_det_from_cholesky, mahalanobis_sq};
use crate::matrix::FdMatrix;
use rand::prelude::*;
use std::f64::consts::PI;

/// Compute log-density of z under component c.
fn log_component_density(z: &[f64], mean: &[f64], cov: &[f64], d: usize, cov_type: CovType) -> f64 {
    let log_2pi = (2.0 * PI).ln();

    match cov_type {
        CovType::Full => {
            let Ok(chol) = cholesky_d(cov, d) else {
                return f64::NEG_INFINITY;
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

/// Compute log(pi_c * N(z_i | mu_c, Sigma_c)) for each component c.
fn compute_log_component_probs(
    feature: &[f64],
    means: &[Vec<f64>],
    covariances: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    d: usize,
    cov_type: CovType,
) -> Vec<f64> {
    let mut log_probs = vec![f64::NEG_INFINITY; k];
    for c in 0..k {
        if weights[c] > 1e-15 {
            log_probs[c] = weights[c].ln()
                + log_component_density(feature, &means[c], &covariances[c], d, cov_type);
        }
    }
    log_probs
}

/// Normalize log-probabilities to responsibilities via log-sum-exp. Returns log-likelihood contribution.
fn normalizeresponsibilities(log_probs: &[f64], resp: &mut [f64]) -> f64 {
    let k = log_probs.len();
    let max_lp = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_lp == f64::NEG_INFINITY {
        for c in 0..k {
            resp[c] = 1.0 / k as f64;
        }
        return 0.0;
    }
    let lse = max_lp
        + log_probs
            .iter()
            .map(|&lp| (lp - max_lp).exp())
            .sum::<f64>()
            .ln();
    for c in 0..k {
        resp[c] = (log_probs[c] - lse).exp();
    }
    lse
}

/// E-step: compute responsibilities and log-likelihood.
/// Returns (responsibilities as n*k flat row-major, log_likelihood).
pub(super) fn e_step(
    features: &[Vec<f64>],
    means: &[Vec<f64>],
    covariances: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    d: usize,
    cov_type: CovType,
) -> (Vec<f64>, f64) {
    #[cfg(feature = "parallel")]
    use rayon::iter::ParallelIterator;

    let n = features.len();
    let per_obs: Vec<(Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let log_probs = compute_log_component_probs(
                &features[i],
                means,
                covariances,
                weights,
                k,
                d,
                cov_type,
            );
            let mut r = vec![0.0; k];
            let ll_i = normalizeresponsibilities(&log_probs, &mut r);
            (r, ll_i)
        })
        .collect();

    let mut resp = vec![0.0; n * k];
    let mut ll = 0.0;
    for (i, (r, ll_i)) in per_obs.into_iter().enumerate() {
        resp[i * k..(i + 1) * k].copy_from_slice(&r);
        ll += ll_i;
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
pub(super) fn count_params(k: usize, d: usize, cov_type: CovType) -> usize {
    let mean_params = k * d;
    let weight_params = k - 1;
    let cov_params = match cov_type {
        CovType::Full => k * d * (d + 1) / 2,
        CovType::Diagonal => k * d,
    };
    mean_params + weight_params + cov_params
}

/// Compute BIC = -2*LL + p*ln(n).
pub(super) fn compute_bic(ll: f64, n: usize, n_params: usize) -> f64 {
    -2.0 * ll + n_params as f64 * (n as f64).ln()
}

/// Compute ICL = BIC + 2*entropy(responsibilities).
pub(super) fn compute_icl(bic: f64, resp: &[f64], n: usize, k: usize) -> f64 {
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

/// Convert flat responsibilities to hard cluster assignments.
pub(super) fn hard_assignments(resp: &[f64], n: usize, k: usize) -> Vec<usize> {
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

/// Convert flat row-major responsibilities to FdMatrix (n x k).
pub(super) fn resp_to_membership(resp: &[f64], n: usize, k: usize) -> FdMatrix {
    // resp is row-major (n × k), FdMatrix is column-major
    let mut col_major = vec![0.0; n * k];
    for i in 0..n {
        for c in 0..k {
            col_major[i + c * n] = resp[i * k + c];
        }
    }
    FdMatrix::from_column_major(col_major, n, k).expect("dimension invariant: data.len() == n * m")
}

/// Finalize GMM: run final E-step, compute BIC/ICL, extract assignments.
pub(super) fn finalize_gmm(
    features: &[Vec<f64>],
    means: Vec<Vec<f64>>,
    covariances: Vec<Vec<f64>>,
    weights: Vec<f64>,
    k: usize,
    d: usize,
    n: usize,
    cov_type: CovType,
    iterations: usize,
    converged: bool,
) -> GmmResult {
    let (resp, log_likelihood) = e_step(features, &means, &covariances, &weights, k, d, cov_type);
    let n_params = count_params(k, d, cov_type);
    let bic = compute_bic(log_likelihood, n, n_params);
    let icl = compute_icl(bic, &resp, n, k);
    let cluster = hard_assignments(&resp, n, k);
    let membership = resp_to_membership(&resp, n, k);

    GmmResult {
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
    }
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
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `features` is empty or has
/// zero-dimensional feature vectors.
/// Returns [`FdarError::InvalidParameter`] if `k` is zero or exceeds `n`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gmm_em(
    features: &[Vec<f64>],
    k: usize,
    cov_type: CovType,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Result<GmmResult, FdarError> {
    let n = features.len();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "features",
            expected: "non-empty slice".to_string(),
            actual: "0 rows".to_string(),
        });
    }
    if k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "must be >= 1".to_string(),
        });
    }
    if k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("must be <= n ({n}), got {k}"),
        });
    }
    let d = features[0].len();
    if d == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "features",
            expected: "non-zero feature dimension".to_string(),
            actual: "0".to_string(),
        });
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

        let (newresp, ll) = e_step(features, &means, &covariances, &weights, k, d, cov_type);
        resp = newresp;

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

    // Final E-step and model selection
    Ok(finalize_gmm(
        features,
        means,
        covariances,
        weights,
        k,
        d,
        n,
        cov_type,
        iterations,
        converged,
    ))
}
