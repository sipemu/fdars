//! funHDDC: per-group subspace covariance functional clustering.
#![allow(non_snake_case)]
//!
//! Implements a simplified version of the funHDDC algorithm (Bouveyron &
//! Brunet 2012, "Model-based clustering of high-dimensional data") restricted
//! to the **AkBk** covariance model: each group k has an intrinsic subspace
//! spanned by `d_k` leading eigenvectors (with per-subspace isotropic variance
//! `a_k`) plus an isotropic residual noise variance `b_k` on the orthogonal
//! complement.
//!
//! **Deliberate divergence from the R `funHDDC` package:** The R package
//! implements six covariance model families (`AkjBkQkDk`, `AkBkQkDk`,
//! `ABkQkDk`, `AkBQkDk`, `ABQkDk`, `ABQDk`). This implementation provides
//! only the single `AkBk` model (one shared isotropic within-subspace variance
//! per group and one isotropic noise variance per group). It is intentionally
//! simplified for tractability and to avoid a new crate dependency. Document
//! the model family mismatch in user-facing projects where comparison with R is
//! expected.
//!
//! Key function:
//! - [`funhddC_cluster`] — fit the AkBk funHDDC model to functional data

use super::covariance::data_scaled_reg;
use super::em::{compute_bic, compute_icl, hard_assignments, resp_to_membership};
use super::init::kmeans_init_assignments;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc_1d;
use nalgebra::{DMatrix, SVD};
use rand::prelude::*;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Configuration for funHDDC per-group subspace clustering.
///
/// Implements the `AkBk` model from Bouveyron & Brunet (2012):
/// each group k has `d_k` leading eigenvectors (intrinsic subspace) plus an
/// isotropic residual noise variance on the complement.
///
/// **Note:** This is a single representative model and does **not** implement
/// the full six-model `akjbkqkdk` family offered by the R `funHDDC` package.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FunHddcConfig {
    /// Number of clusters (default: 2).
    pub k: usize,
    /// Intrinsic subspace dimension per group (default: 2).
    pub d_k: usize,
    /// Maximum EM iterations per restart (default: 100).
    pub max_iter: usize,
    /// Log-likelihood convergence tolerance (default: 1e-6).
    pub tol: f64,
    /// Number of random restarts; best result by log-likelihood is returned (default: 3).
    pub n_init: usize,
    /// Base random seed; restart `i` uses `seed + i * 1000` (default: 42).
    pub seed: u64,
    /// Number of global FPCA components used for initialisation features (default: 10).
    pub ncomp_init: usize,
}

impl Default for FunHddcConfig {
    fn default() -> Self {
        FunHddcConfig {
            k: 2,
            d_k: 2,
            max_iter: 100,
            tol: 1e-6,
            n_init: 3,
            seed: 42,
            ncomp_init: 10,
        }
    }
}

/// Result from funHDDC per-group subspace clustering.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FunHddcResult {
    /// Hard cluster assignments (length n).
    pub cluster: Vec<usize>,
    /// Posterior membership probabilities (n × k), column-major FdMatrix.
    pub membership: FdMatrix,
    /// Per-group subspace matrices (length k); each is m × d_k_eff column-major.
    pub subspaces: Vec<FdMatrix>,
    /// Per-group within-subspace variances (length k); each is length d_k_eff.
    pub within_vars: Vec<Vec<f64>>,
    /// Per-group isotropic noise variances (length k).
    pub noise_vars: Vec<f64>,
    /// Per-group mean curves (length k); each is length m.
    pub means: Vec<Vec<f64>>,
    /// Mixing proportions (length k).
    pub weights: Vec<f64>,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// BIC value.
    pub bic: f64,
    /// ICL value.
    pub icl: f64,
    /// Number of EM iterations performed.
    pub iterations: usize,
    /// Whether EM converged within `max_iter`.
    pub converged: bool,
    /// Number of clusters.
    pub k: usize,
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Log Gaussian density for one observation under the AkBk subspace model.
///
/// Parameters:
/// - `diff`: centered observation `x_i - mu_k` (length m)
/// - `u_k`: subspace columns (m × d_k_eff, column-major)
/// - `a_k`: within-subspace variances (length d_k_eff)
/// - `b_k`: isotropic noise variance (scalar > 0)
/// - `m`, `d_k_eff`: dimensions
fn log_density_subspace(
    diff: &[f64],
    u_k: &[f64],
    a_k: &[f64],
    b_k: f64,
    m: usize,
    d_k_eff: usize,
) -> f64 {
    use std::f64::consts::PI;
    // Project diff onto subspace: z = U_k^T diff (column-major U_k)
    let mut z = vec![0.0_f64; d_k_eff];
    for j in 0..d_k_eff {
        for r in 0..m {
            z[j] += u_k[r + j * m] * diff[r];
        }
    }

    // Within-subspace log-likelihood: Σ_j -0.5*(ln(a_k[j]) + z[j]^2/a_k[j])
    let mut ll = 0.0_f64;
    for j in 0..d_k_eff {
        if a_k[j] <= 0.0 {
            return f64::NEG_INFINITY;
        }
        ll -= 0.5 * (a_k[j].ln() + z[j].powi(2) / a_k[j]);
    }

    // Complement squared norm: ||diff||^2 - ||z||^2
    let diff_sq: f64 = diff.iter().map(|v| v * v).sum();
    let z_sq: f64 = z.iter().map(|v| v * v).sum();
    let complement_sq = diff_sq - z_sq;

    if b_k <= 0.0 {
        return f64::NEG_INFINITY;
    }
    let m_minus_dk = (m - d_k_eff) as f64;
    ll -= 0.5 * (m_minus_dk * b_k.ln() + complement_sq / b_k);

    // Normalizing constant: -0.5 * m * ln(2π)
    ll -= 0.5 * (m as f64) * (2.0 * PI).ln();
    ll
}

/// Log-sum-exp normalization of log-probabilities into responsibilities.
/// Returns log-likelihood contribution for this observation.
fn normalize_log_probs(log_probs: &[f64], resp: &mut [f64]) -> f64 {
    let k = log_probs.len();
    let max_lp = log_probs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max_lp == f64::NEG_INFINITY {
        let uniform = 1.0 / k as f64;
        for r in resp.iter_mut() {
            *r = uniform;
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

/// Run the subspace E-step: compute responsibilities and log-likelihood.
///
/// Returns `(resp_flat, log_likelihood)` where `resp_flat` is n*k row-major.
fn e_step_subspace(
    data_rows: &[Vec<f64>], // n rows of length m
    means: &[Vec<f64>],     // k means of length m
    subspaces: &[Vec<f64>], // k subspace matrices (m * d_k_eff, col-major)
    within_vars: &[Vec<f64>], // k within-subspace var vectors
    noise_vars: &[f64],     // k noise variances
    weights: &[f64],        // k mixing proportions
    k: usize,
    m: usize,
) -> (Vec<f64>, f64) {
    let n = data_rows.len();
    let mut resp = vec![0.0_f64; n * k];
    let mut total_ll = 0.0_f64;

    for i in 0..n {
        let x = &data_rows[i];
        let mut log_probs = vec![f64::NEG_INFINITY; k];
        for c in 0..k {
            if weights[c] > 1e-15 {
                let d_k_eff = within_vars[c].len();
                let diff: Vec<f64> = x.iter().zip(means[c].iter()).map(|(&xi, &mi)| xi - mi).collect();
                let ld = log_density_subspace(
                    &diff,
                    &subspaces[c],
                    &within_vars[c],
                    noise_vars[c],
                    m,
                    d_k_eff,
                );
                log_probs[c] = weights[c].ln() + ld;
            }
        }
        let mut r = vec![0.0_f64; k];
        let ll_i = normalize_log_probs(&log_probs, &mut r);
        resp[i * k..(i + 1) * k].copy_from_slice(&r);
        total_ll += ll_i;
    }
    (resp, total_ll)
}

/// Thin SVD of a data slice (n_k × m) to get leading `d_k_eff` right-singular vectors.
///
/// Returns `(u_k_flat, a_k)` where `u_k_flat` is m × d_k_eff column-major
/// and `a_k` is the per-direction variance (singular_value^2 / n_k).
/// Returns `None` if SVD fails.
fn per_group_svd(
    centered_rows: &[Vec<f64>],
    d_k_req: usize,
    m: usize,
    reg: f64,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let n_k = centered_rows.len();
    if n_k == 0 {
        return None;
    }
    let d_k_eff = d_k_req.min(n_k).min(m);
    if d_k_eff == 0 {
        return None;
    }

    // Build n_k × m DMatrix (row-major from our row vecs)
    let mut mat = DMatrix::<f64>::zeros(n_k, m);
    for (i, row) in centered_rows.iter().enumerate() {
        for j in 0..m {
            mat[(i, j)] = row[j];
        }
    }

    let svd = SVD::new(mat, true, true);
    let v_t = svd.v_t?;
    let singular_values = &svd.singular_values;

    // Rotation: m × d_k_eff column-major (columns are right-singular vectors)
    let mut u_k_flat = vec![0.0_f64; m * d_k_eff];
    for j in 0..d_k_eff {
        for r in 0..m {
            // v_t is d × m, column j of V is row j of V^T
            u_k_flat[r + j * m] = v_t[(j, r)];
        }
    }

    // Within-subspace variances: σ_j^2 / n_k, floored at reg
    let n_k_f = n_k as f64;
    let a_k: Vec<f64> = (0..d_k_eff)
        .map(|j| {
            let sv = singular_values[j];
            (sv * sv / n_k_f).max(reg)
        })
        .collect();

    Some((u_k_flat, a_k))
}

/// Run one EM for funHDDC from a given initialisation.
#[allow(clippy::too_many_arguments)]
fn run_one_em(
    data_rows: &[Vec<f64>],
    k: usize,
    m: usize,
    d_k_req: usize,
    max_iter: usize,
    tol: f64,
    init_assignments: &[usize],
    reg: f64,
) -> Option<(
    Vec<f64>,            // resp flat (n*k)
    Vec<Vec<f64>>,       // means
    Vec<Vec<f64>>,       // subspaces (col-major)
    Vec<Vec<f64>>,       // within_vars
    Vec<f64>,            // noise_vars
    Vec<f64>,            // weights
    f64,                 // log_likelihood
    usize,               // iterations
    bool,                // converged
)> {
    let n = data_rows.len();

    // Initialise means, weights from hard assignments
    let mut means: Vec<Vec<f64>> = vec![vec![0.0_f64; m]; k];
    let mut counts = vec![0usize; k];
    for (i, &c) in init_assignments.iter().enumerate() {
        counts[c] += 1;
        for j in 0..m {
            means[c][j] += data_rows[i][j];
        }
    }
    for c in 0..k {
        let nc = counts[c].max(1);
        for j in 0..m {
            means[c][j] /= nc as f64;
        }
    }
    let mut weights: Vec<f64> = counts
        .iter()
        .map(|&c| c.max(1) as f64 / n as f64)
        .collect();

    // Initialise subspaces and noise vars to identity-like fallback
    let mut subspaces: Vec<Vec<f64>> = vec![vec![0.0_f64; m * d_k_req.min(m)]; k];
    let mut within_vars: Vec<Vec<f64>> = vec![vec![reg; d_k_req.min(m)]; k];
    let mut noise_vars: Vec<f64> = vec![reg; k];

    // Initialise subspaces from group data
    for c in 0..k {
        let member_rows: Vec<Vec<f64>> = (0..n)
            .filter(|&i| init_assignments[i] == c)
            .map(|i| {
                data_rows[i]
                    .iter()
                    .zip(means[c].iter())
                    .map(|(&x, &mu)| x - mu)
                    .collect()
            })
            .collect();

        if let Some((u_k, a_k)) = per_group_svd(&member_rows, d_k_req, m, reg) {
            let d_k_eff = a_k.len();
            subspaces[c] = u_k;
            within_vars[c] = a_k.clone();
            // Noise var from total variance minus within-subspace variance
            let total_var: f64 = member_rows
                .iter()
                .flat_map(|r| r.iter())
                .map(|v| v * v)
                .sum::<f64>()
                / member_rows.len().max(1) as f64;
            let subspace_var: f64 = a_k.iter().sum();
            let complement_var = (total_var - subspace_var).max(0.0);
            let m_minus_dk = (m - d_k_eff) as f64;
            noise_vars[c] = if m_minus_dk > 0.0 {
                (complement_var / m_minus_dk).max(reg)
            } else {
                reg
            };
        }
    }

    let mut resp = vec![0.0_f64; n * k];
    let mut prev_ll = f64::NEG_INFINITY;
    let mut converged = false;
    let mut iterations = 0usize;

    for iter in 0..max_iter {
        iterations = iter + 1;

        // E-step
        let (new_resp, ll) =
            e_step_subspace(data_rows, &means, &subspaces, &within_vars, &noise_vars, &weights, k, m);
        resp = new_resp;

        if (ll - prev_ll).abs() < tol && iter > 0 {
            converged = true;
            break;
        }
        prev_ll = ll;

        // M-step
        // --- Update means and weights ---
        let mut new_means = vec![vec![0.0_f64; m]; k];
        let mut nk_vec = vec![0.0_f64; k];
        for i in 0..n {
            for c in 0..k {
                let r = resp[i * k + c];
                nk_vec[c] += r;
                for j in 0..m {
                    new_means[c][j] += r * data_rows[i][j];
                }
            }
        }
        for c in 0..k {
            let nk = nk_vec[c];
            if nk > 1e-15 {
                for j in 0..m {
                    new_means[c][j] /= nk;
                }
            }
        }
        let n_f = n as f64;
        weights = nk_vec.iter().map(|&nk| nk / n_f).collect();
        means = new_means;

        // --- Update subspaces: weighted SVD per group ---
        for c in 0..k {
            let nk = nk_vec[c];
            if nk < 1e-15 {
                // Empty cluster fallback: keep identity-like subspace
                let d_k_eff = d_k_req.min(m);
                subspaces[c] = vec![0.0_f64; m * d_k_eff];
                within_vars[c] = vec![reg; d_k_eff];
                noise_vars[c] = reg;
                continue;
            }

            // Build weighted centered rows (weight = sqrt(resp))
            let mut w_rows: Vec<Vec<f64>> = Vec::with_capacity(n);
            for i in 0..n {
                let sqrt_r = resp[i * k + c].sqrt();
                if sqrt_r > 1e-15 {
                    let row: Vec<f64> = data_rows[i]
                        .iter()
                        .zip(means[c].iter())
                        .map(|(&x, &mu)| sqrt_r * (x - mu))
                        .collect();
                    w_rows.push(row);
                }
            }

            if w_rows.is_empty() {
                let d_k_eff = d_k_req.min(m);
                subspaces[c] = vec![0.0_f64; m * d_k_eff];
                within_vars[c] = vec![reg; d_k_eff];
                noise_vars[c] = reg;
                continue;
            }

            if let Some((u_k, a_k)) = per_group_svd(&w_rows, d_k_req, m, reg) {
                let d_k_eff = a_k.len();
                // Rescale a_k back from weighted rows (SVD of sqrt(r)*diff → a = sigma^2 / nk)
                let a_k_rescaled: Vec<f64> = a_k.iter().map(|&a| a.max(reg)).collect();
                subspaces[c] = u_k;
                within_vars[c] = a_k_rescaled.clone();

                // Noise variance: (total weighted variance - subspace variance) / (m - d_k_eff)
                let total_wvar: f64 = w_rows
                    .iter()
                    .flat_map(|r| r.iter())
                    .map(|v| v * v)
                    .sum::<f64>()
                    / w_rows.len() as f64;
                let subspace_var: f64 = a_k_rescaled.iter().sum();
                let complement_var = (total_wvar - subspace_var).max(0.0);
                let m_minus_dk = (m - d_k_eff) as f64;
                noise_vars[c] = if m_minus_dk > 0.0 {
                    (complement_var / m_minus_dk).max(reg)
                } else {
                    reg
                };
            }
        }
    }

    // Final E-step for accurate LL
    let (final_resp, final_ll) =
        e_step_subspace(data_rows, &means, &subspaces, &within_vars, &noise_vars, &weights, k, m);

    Some((
        final_resp,
        means,
        subspaces,
        within_vars,
        noise_vars,
        weights,
        final_ll,
        iterations,
        converged,
    ))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Cluster functional data using the funHDDC `AkBk` subspace covariance model.
///
/// Each group k is modelled by a `d_k`-dimensional intrinsic subspace (leading
/// eigenvectors of the within-group data) plus an isotropic residual-noise
/// variance on the complement. Parameters are estimated by EM; the best
/// result across `config.n_init` random restarts is returned.
///
/// # Deliberate simplification
///
/// This implements only the **single `AkBk` model**: one shared scalar
/// within-subspace variance and one isotropic noise variance per group.
/// The R `funHDDC` package implements six model families
/// (`AkjBkQkDk`, `AkBkQkDk`, `ABkQkDk`, `AkBQkDk`, `ABQkDk`, `ABQDk`);
/// this implementation is the simplest case for tractability. If you need
/// the full model family, use the R `funHDDC` package.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m), n curves at m evaluation points.
/// * `argvals` — Evaluation grid (length m); used for the initial FPCA projection.
/// * `config` — Algorithm configuration; see [`FunHddcConfig`].
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` is empty or if
/// `argvals.len() != m`.
/// Returns [`FdarError::InvalidParameter`] if `k == 0`, `k > n`, `d_k == 0`,
/// or `d_k >= m`.
/// Returns [`FdarError::ComputationFailed`] if all restarts fail.
///
/// # Examples
///
/// ```no_run
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::gmm::subspace::{funhddC_cluster, FunHddcConfig};
///
/// let mut cfg = FunHddcConfig::default();
/// cfg.k = 2;
/// cfg.d_k = 2;
/// // Provide data and argvals ...
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn funhddC_cluster(
    data: &FdMatrix,
    argvals: &[f64],
    config: &FunHddcConfig,
) -> Result<FunHddcResult, FdarError> {
    let (n, m) = data.shape();

    // Input validation
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    if config.k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: "must be >= 1".to_string(),
        });
    }
    if config.k > n {
        return Err(FdarError::InvalidParameter {
            parameter: "k",
            message: format!("must be <= n ({n}), got {}", config.k),
        });
    }
    if config.d_k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "d_k",
            message: "must be >= 1".to_string(),
        });
    }
    if config.d_k >= m {
        return Err(FdarError::InvalidParameter {
            parameter: "d_k",
            message: format!("must be < m ({m}), got {}", config.d_k),
        });
    }

    // Build row-major data buffer for efficient per-row access
    let data_rm = data.to_row_major();
    let data_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| data_rm[i * m..(i + 1) * m].to_vec())
        .collect();

    // Compute global data-scaled regularization floor
    let reg = data_scaled_reg(&data_rows, m);

    // Global FPCA for initialisation features
    let ncomp_init = config.ncomp_init.min(n).min(m).max(1);
    let fpca = fdata_to_pc_1d(data, ncomp_init, argvals)?;
    let score_mat = &fpca.scores;
    let d_feat = score_mat.ncols();
    let features: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..d_feat).map(|j| score_mat[(i, j)]).collect())
        .collect();

    let k = config.k;

    // Multi-restart EM
    let mut best: Option<(
        Vec<f64>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<Vec<f64>>,
        Vec<f64>,
        Vec<f64>,
        f64,
        usize,
        bool,
    )> = None;

    for init_idx in 0..config.n_init {
        let seed = config.seed.wrapping_add(init_idx as u64 * 1000);
        let mut rng = StdRng::seed_from_u64(seed);
        let init_assignments = kmeans_init_assignments(&features, k, &mut rng);

        if let Some(result) = run_one_em(
            &data_rows,
            k,
            m,
            config.d_k,
            config.max_iter,
            config.tol,
            &init_assignments,
            reg,
        ) {
            let ll = result.6;
            let is_better = best.as_ref().map_or(true, |b| ll > b.6);
            if is_better {
                best = Some(result);
            }
        }
    }

    let (resp, means, subspaces_flat, within_vars, noise_vars, weights, log_likelihood, iterations, converged) =
        best.ok_or_else(|| FdarError::ComputationFailed {
            operation: "funhddC_cluster",
            detail: "all EM restarts failed".to_string(),
        })?;

    // Parameter count for BIC/ICL:
    // per-group: m*d_k_eff (subspace) - d_k_eff*(d_k_eff-1)/2 (Stiefel constraint) + d_k_eff (a_k) + 1 (b_k)
    // plus (k-1) mixing proportions
    let d_k_eff = within_vars.first().map_or(1, |v| v.len());
    let subspace_params = k * (m * d_k_eff - d_k_eff * (d_k_eff.saturating_sub(1)) / 2);
    let var_params = k * d_k_eff + k; // a_k + b_k
    let n_params = subspace_params + var_params + (k - 1);

    let bic = compute_bic(log_likelihood, n, n_params);
    let icl = compute_icl(bic, &resp, n, k);

    let cluster = hard_assignments(&resp, n, k);
    let membership = resp_to_membership(&resp, n, k);

    // Convert flat subspace vectors to FdMatrix (m × d_k_eff)
    let subspaces: Vec<FdMatrix> = subspaces_flat
        .into_iter()
        .zip(within_vars.iter())
        .map(|(flat, av)| {
            let d = av.len();
            if d == 0 || flat.is_empty() {
                FdMatrix::zeros(m, 1)
            } else {
                FdMatrix::from_column_major(flat, m, d)
                    .unwrap_or_else(|_| FdMatrix::zeros(m, d))
            }
        })
        .collect();

    Ok(FunHddcResult {
        cluster,
        membership,
        subspaces,
        within_vars,
        noise_vars,
        means,
        weights,
        log_likelihood,
        bic,
        icl,
        iterations,
        converged,
        k,
    })
}

// ---------------------------------------------------------------------------
// Inline tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::{adjusted_rand_index, uniform_grid};

    /// Generate two vertically separated groups of sinusoidal curves.
    /// Cluster 0: sin(t), Cluster 1: sin(t) + 5.0
    fn two_separated_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<usize>) {
        let argvals = uniform_grid(m);
        let n = 2 * n_per;
        let mut data_rm = vec![0.0_f64; n * m];
        let mut labels = vec![0usize; n];
        for i in 0..n_per {
            for j in 0..m {
                data_rm[i * m + j] = argvals[j].sin();
            }
            labels[i] = 0;
        }
        for i in 0..n_per {
            for j in 0..m {
                data_rm[(n_per + i) * m + j] = argvals[j].sin() + 5.0;
            }
            labels[n_per + i] = 1;
        }
        // Convert row-major to column-major FdMatrix
        let mut col_major = vec![0.0_f64; n * m];
        for ii in 0..n {
            for jj in 0..m {
                col_major[ii + jj * n] = data_rm[ii * m + jj];
            }
        }
        let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
        (data, argvals, labels)
    }

    #[test]
    fn test_funhddC_recovery() {
        let (data, argvals, labels) = two_separated_clusters(15, 20);
        let config = FunHddcConfig {
            k: 2,
            d_k: 2,
            max_iter: 100,
            tol: 1e-6,
            n_init: 3,
            seed: 42,
            ncomp_init: 8,
        };
        let result = funhddC_cluster(&data, &argvals, &config).unwrap();
        let ari = adjusted_rand_index(&labels, &result.cluster);
        assert!(
            ari >= 0.90,
            "Recovery ARI should be >= 0.90, got {ari:.4}"
        );
    }

    #[test]
    fn test_funhddC_bic_finite() {
        let (data, argvals, _) = two_separated_clusters(15, 20);
        let config = FunHddcConfig {
            k: 2,
            d_k: 2,
            max_iter: 100,
            tol: 1e-6,
            n_init: 3,
            seed: 42,
            ncomp_init: 8,
        };
        let result = funhddC_cluster(&data, &argvals, &config).unwrap();
        assert!(result.bic.is_finite(), "BIC should be finite, got {}", result.bic);
        assert!(result.icl.is_finite(), "ICL should be finite, got {}", result.icl);
        assert!(
            result.log_likelihood.is_finite(),
            "log-likelihood should be finite, got {}",
            result.log_likelihood
        );
    }

    #[test]
    fn test_funhddC_deterministic() {
        let (data, argvals, _) = two_separated_clusters(15, 20);
        let config = FunHddcConfig {
            k: 2,
            d_k: 2,
            max_iter: 100,
            tol: 1e-6,
            n_init: 3,
            seed: 99,
            ncomp_init: 8,
        };
        let r1 = funhddC_cluster(&data, &argvals, &config).unwrap();
        let r2 = funhddC_cluster(&data, &argvals, &config).unwrap();
        assert_eq!(r1.cluster, r2.cluster, "Same seed must give identical cluster assignments");
    }

    #[test]
    fn test_funhddC_invalid_empty() {
        let data = FdMatrix::zeros(0, 10);
        let argvals = uniform_grid(10);
        let config = FunHddcConfig { k: 2, ..Default::default() };
        assert!(funhddC_cluster(&data, &argvals, &config).is_err());
    }

    #[test]
    fn test_funhddC_invalid_k_zero() {
        let data = FdMatrix::zeros(5, 10);
        let argvals = uniform_grid(10);
        let config = FunHddcConfig { k: 0, ..Default::default() };
        assert!(funhddC_cluster(&data, &argvals, &config).is_err());
    }

    #[test]
    fn test_funhddC_invalid_k_exceeds_n() {
        let data = FdMatrix::zeros(3, 10);
        let argvals = uniform_grid(10);
        let config = FunHddcConfig { k: 5, ..Default::default() };
        assert!(funhddC_cluster(&data, &argvals, &config).is_err());
    }

    #[test]
    fn test_funhddC_invalid_dk_ge_m() {
        let data = FdMatrix::zeros(5, 10);
        let argvals = uniform_grid(10);
        let config = FunHddcConfig { k: 2, d_k: 10, ..Default::default() };
        assert!(funhddC_cluster(&data, &argvals, &config).is_err());
    }

    #[test]
    fn test_funhddC_invalid_argvals_mismatch() {
        let data = FdMatrix::zeros(5, 10);
        let argvals = uniform_grid(8); // wrong length
        let config = FunHddcConfig { k: 2, ..Default::default() };
        assert!(funhddC_cluster(&data, &argvals, &config).is_err());
    }
}
