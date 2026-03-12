//! Clustering wrapper and prediction for GMM.

use super::em::{e_step, gmm_em, hard_assignments, resp_to_membership};
use super::init::build_features;
use super::{CovType, GmmClusterResult, GmmResult};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Run multiple initializations for a single K and return the best by log-likelihood.
pub(super) fn run_multiple_inits(
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
        if let Ok(result) = gmm_em(features, k, cov_type, max_iter, tol, seed) {
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
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] if basis projection fails or no
/// valid GMM fit is found for any K in the given range.
#[must_use = "expensive computation whose result should not be discarded"]
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
) -> Result<GmmClusterResult, FdarError> {
    let (features, _d) = build_features(data, argvals, covariates, nbasis, basis_type, cov_weight)
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "build_features",
            detail: "basis projection failed".to_string(),
        })?;

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

    best_result
        .map(|best| GmmClusterResult {
            best,
            bic_values,
            icl_values,
        })
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "gmm_cluster",
            detail: "no valid GMM fit found for any K in range".to_string(),
        })
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
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] if basis projection fails for the
/// new data.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn predict_gmm(
    new_data: &FdMatrix,
    argvals: &[f64],
    new_covariates: Option<&FdMatrix>,
    result: &GmmResult,
    nbasis: usize,
    basis_type: i32,
    cov_weight: f64,
    cov_type: CovType,
) -> Result<(Vec<usize>, FdMatrix), FdarError> {
    let (features, _d) = build_features(
        new_data,
        argvals,
        new_covariates,
        nbasis,
        basis_type,
        cov_weight,
    )
    .ok_or_else(|| FdarError::ComputationFailed {
        operation: "build_features",
        detail: "basis projection failed for new data".to_string(),
    })?;

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

    Ok((cluster, membership))
}
