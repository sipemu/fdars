//! TSRVF (Transported SRSF) transforms and parallel transport methods.
//!
//! Maps aligned SRSFs to the tangent space of the Karcher mean on the Hilbert
//! sphere. Tangent vectors live in a standard Euclidean space, enabling PCA,
//! regression, and clustering on elastic-aligned curves.

use super::karcher::karcher_mean;
use super::srsf::{srsf_inverse, srsf_transform};
use super::KarcherMeanResult;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::smoothing::nadaraya_watson;
use crate::warping::{exp_map_sphere, inv_exp_map_sphere, l2_norm_l2};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of the TSRVF transform.
#[derive(Debug, Clone, PartialEq)]
pub struct TsrvfResult {
    /// Tangent vectors in Euclidean space (n × m).
    pub tangent_vectors: FdMatrix,
    /// Karcher mean curve (length m).
    pub mean: Vec<f64>,
    /// SRSF of the Karcher mean (length m).
    pub mean_srsf: Vec<f64>,
    /// L2 norm of the mean SRSF.
    pub mean_srsf_norm: f64,
    /// Per-curve aligned SRSF norms (length n).
    pub srsf_norms: Vec<f64>,
    /// Per-curve initial values f_i(0) for SRSF inverse reconstruction (length n).
    pub initial_values: Vec<f64>,
    /// Warping functions from Karcher mean computation (n × m).
    pub gammas: FdMatrix,
    /// Whether the Karcher mean converged.
    pub converged: bool,
}

/// Method for transporting tangent vectors on the Hilbert sphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TransportMethod {
    /// Inverse exponential map (log map) — default, matches existing TSRVF behavior.
    #[default]
    LogMap,
    /// Schild's ladder approximation to parallel transport.
    SchildsLadder,
    /// Pole ladder approximation to parallel transport.
    PoleLadder,
}

// ─── Smoothing ──────────────────────────────────────────────────────────────

/// Smooth aligned SRSFs to remove DP kink artifacts before TSRVF computation.
///
/// Uses Nadaraya-Watson kernel smoothing (Gaussian, bandwidth = 2 grid spacings)
/// on each SRSF row. This removes the derivative spikes from DP warp kinks
/// without affecting alignment results or the Karcher mean.
fn smooth_aligned_srsfs(srsf: &FdMatrix, m: usize) -> FdMatrix {
    let n = srsf.nrows();
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;

    let mut smoothed = FdMatrix::zeros(n, m);
    for i in 0..n {
        let qi = srsf.row(i);
        let qi_smooth = nadaraya_watson(&time, &qi, &time, bandwidth, "gaussian");
        for j in 0..m {
            smoothed[(i, j)] = qi_smooth[j];
        }
    }
    smoothed
}

// ─── Parallel Transport ─────────────────────────────────────────────────────

/// Schild's ladder parallel transport of vector `v` from `from` to `to` on the sphere.
pub(super) fn parallel_transport_schilds(
    v: &[f64],
    from: &[f64],
    to: &[f64],
    time: &[f64],
) -> Vec<f64> {
    let v_norm = l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        return vec![0.0; v.len()];
    }

    // endpoint = exp_from(v)
    let endpoint = exp_map_sphere(from, v, time);

    // midpoint_v = log_to(endpoint) — vector at `to` pointing toward endpoint
    let log_to_ep = inv_exp_map_sphere(to, &endpoint, time);

    // midpoint = exp_to(0.5 * log_to_ep)
    let half_log: Vec<f64> = log_to_ep.iter().map(|&x| 0.5 * x).collect();
    let midpoint = exp_map_sphere(to, &half_log, time);

    // transported = 2 * log_to(midpoint)
    let log_to_mid = inv_exp_map_sphere(to, &midpoint, time);
    log_to_mid.iter().map(|&x| 2.0 * x).collect()
}

/// Pole ladder parallel transport of vector `v` from `from` to `to` on the sphere.
pub(super) fn parallel_transport_pole(
    v: &[f64],
    from: &[f64],
    to: &[f64],
    time: &[f64],
) -> Vec<f64> {
    let v_norm = l2_norm_l2(v, time);
    if v_norm < 1e-10 {
        return vec![0.0; v.len()];
    }

    // pole = exp_from(-v)
    let neg_v: Vec<f64> = v.iter().map(|&x| -x).collect();
    let pole = exp_map_sphere(from, &neg_v, time);

    // midpoint_v = log_to(pole)
    let log_to_pole = inv_exp_map_sphere(to, &pole, time);

    // midpoint = exp_to(0.5 * log_to_pole)
    let half_log: Vec<f64> = log_to_pole.iter().map(|&x| 0.5 * x).collect();
    let midpoint = exp_map_sphere(to, &half_log, time);

    // transported = -2 * log_to(midpoint)
    let log_to_mid = inv_exp_map_sphere(to, &midpoint, time);
    log_to_mid.iter().map(|&x| -2.0 * x).collect()
}

// ─── TSRVF Public API ───────────────────────────────────────────────────────

/// Full TSRVF pipeline: compute Karcher mean, then transport SRSFs to tangent space.
///
/// # Arguments
/// * `data` — Functional data matrix (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `max_iter` — Maximum Karcher mean iterations
/// * `tol` — Convergence tolerance for Karcher mean
///
/// # Returns
/// [`TsrvfResult`] containing tangent vectors and associated metadata.
pub fn tsrvf_transform(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> TsrvfResult {
    let karcher = karcher_mean(data, argvals, max_iter, tol, lambda);
    tsrvf_from_alignment(&karcher, argvals)
}

/// Compute TSRVF from a pre-computed Karcher mean alignment.
///
/// Avoids re-running the expensive Karcher mean computation when the alignment
/// has already been computed.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// [`TsrvfResult`] containing tangent vectors and associated metadata.
pub fn tsrvf_from_alignment(karcher: &KarcherMeanResult, argvals: &[f64]) -> TsrvfResult {
    let (n, m) = karcher.aligned_data.shape();

    // Step 1: Compute SRSFs of aligned data
    let aligned_srsf = srsf_transform(&karcher.aligned_data, argvals);

    // Step 1b: Smooth aligned SRSFs to remove DP kink artifacts.
    //
    // DP alignment produces piecewise-linear warps with kinks at grid transitions.
    // When curves are reparameterized by these warps, the kinks propagate into the
    // aligned curves' derivatives (SRSFs), creating spikes that dominate TSRVF
    // tangent vectors and PCA.
    //
    // R's fdasrvf does not smooth here and suffers from the same spike artifacts.
    // Python's fdasrsf mitigates this via spline smoothing (s=1e-4) in SqrtMean.
    // We smooth the aligned SRSFs before tangent vector computation — this only
    // affects TSRVF output and does not change the alignment or Karcher mean.
    let aligned_srsf = smooth_aligned_srsfs(&aligned_srsf, m);

    // Step 2: Smooth and normalize mean SRSF to unit sphere.
    // The mean SRSF must be smoothed consistently with the aligned SRSFs
    // so that a single curve (which IS the mean) produces a zero tangent vector.
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;
    let mean_srsf_smooth = nadaraya_watson(&time, &karcher.mean_srsf, &time, bandwidth, "gaussian");
    let mean_norm = l2_norm_l2(&mean_srsf_smooth, &time);

    let mu_unit: Vec<f64> = if mean_norm > 1e-10 {
        mean_srsf_smooth.iter().map(|&q| q / mean_norm).collect()
    } else {
        vec![0.0; m]
    };

    // Step 3: For each aligned curve, compute tangent vector via inverse exponential map
    let srsf_norms: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            l2_norm_l2(&qi, &time)
        })
        .collect();

    let tangent_data: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            let qi_norm = srsf_norms[i];

            if qi_norm < 1e-10 || mean_norm < 1e-10 {
                return vec![0.0; m];
            }

            // Normalize to unit sphere
            let qi_unit: Vec<f64> = qi.iter().map(|&q| q / qi_norm).collect();

            // Shooting vector from mu_unit to qi_unit
            inv_exp_map_sphere(&mu_unit, &qi_unit, &time)
        })
        .collect();

    // Assemble tangent vectors into FdMatrix
    let mut tangent_vectors = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            tangent_vectors[(i, j)] = tangent_data[i][j];
        }
    }

    // Store per-curve initial values for SRSF inverse reconstruction.
    // Warping preserves f_i(0) since gamma(0) = 0.
    let initial_values: Vec<f64> = (0..n).map(|i| karcher.aligned_data[(i, 0)]).collect();

    TsrvfResult {
        tangent_vectors,
        mean: karcher.mean.clone(),
        mean_srsf: mean_srsf_smooth,
        mean_srsf_norm: mean_norm,
        srsf_norms,
        initial_values,
        gammas: karcher.gammas.clone(),
        converged: karcher.converged,
    }
}

/// Reconstruct aligned curves from TSRVF tangent vectors.
///
/// Inverts the TSRVF transform: maps tangent vectors back to the Hilbert sphere
/// via the exponential map, rescales, and reconstructs curves via SRSF inverse.
///
/// # Arguments
/// * `tsrvf` — TSRVF result from [`tsrvf_transform`] or [`tsrvf_from_alignment`]
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// FdMatrix of reconstructed aligned curves (n × m).
pub fn tsrvf_inverse(tsrvf: &TsrvfResult, argvals: &[f64]) -> FdMatrix {
    let (n, m) = tsrvf.tangent_vectors.shape();

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Normalize mean SRSF to unit sphere
    let mu_unit: Vec<f64> = if tsrvf.mean_srsf_norm > 1e-10 {
        tsrvf
            .mean_srsf
            .iter()
            .map(|&q| q / tsrvf.mean_srsf_norm)
            .collect()
    } else {
        vec![0.0; m]
    };

    let curves: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let vi = tsrvf.tangent_vectors.row(i);

            // Map back to sphere: exp_map(mu_unit, v_i)
            let qi_unit = exp_map_sphere(&mu_unit, &vi, &time);

            // Rescale by original norm
            let qi: Vec<f64> = qi_unit.iter().map(|&q| q * tsrvf.srsf_norms[i]).collect();

            // Reconstruct curve from SRSF using per-curve initial value
            srsf_inverse(&qi, argvals, tsrvf.initial_values[i])
        })
        .collect();

    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            result[(i, j)] = curves[i][j];
        }
    }
    result
}

/// Full TSRVF pipeline with configurable transport method.
///
/// Like [`tsrvf_transform`] but allows choosing the parallel transport method.
pub fn tsrvf_transform_with_method(
    data: &FdMatrix,
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
    method: TransportMethod,
) -> TsrvfResult {
    let karcher = karcher_mean(data, argvals, max_iter, tol, lambda);
    tsrvf_from_alignment_with_method(&karcher, argvals, method)
}

/// Compute TSRVF from a pre-computed Karcher mean with configurable transport.
///
/// - [`TransportMethod::LogMap`]: Uses `inv_exp_map(mu, qi)` directly (standard TSRVF).
/// - [`TransportMethod::SchildsLadder`]: Computes `v = -log_qi(mu)`, then transports
///   via Schild's ladder from qi to mu.
/// - [`TransportMethod::PoleLadder`]: Same but via pole ladder.
pub fn tsrvf_from_alignment_with_method(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    method: TransportMethod,
) -> TsrvfResult {
    if method == TransportMethod::LogMap {
        return tsrvf_from_alignment(karcher, argvals);
    }

    let (n, m) = karcher.aligned_data.shape();
    let aligned_srsf = srsf_transform(&karcher.aligned_data, argvals);
    let aligned_srsf = smooth_aligned_srsfs(&aligned_srsf, m);
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let bandwidth = 2.0 / (m - 1) as f64;
    let mean_srsf_smooth = nadaraya_watson(&time, &karcher.mean_srsf, &time, bandwidth, "gaussian");
    let mean_norm = l2_norm_l2(&mean_srsf_smooth, &time);

    let mu_unit: Vec<f64> = if mean_norm > 1e-10 {
        mean_srsf_smooth.iter().map(|&q| q / mean_norm).collect()
    } else {
        vec![0.0; m]
    };

    let srsf_norms: Vec<f64> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            l2_norm_l2(&qi, &time)
        })
        .collect();

    let tangent_data: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let qi = aligned_srsf.row(i);
            let qi_norm = srsf_norms[i];

            if qi_norm < 1e-10 || mean_norm < 1e-10 {
                return vec![0.0; m];
            }

            let qi_unit: Vec<f64> = qi.iter().map(|&q| q / qi_norm).collect();

            // Compute v = -log_qi(mu) — vector at qi pointing away from mu
            let v_at_qi = inv_exp_map_sphere(&qi_unit, &mu_unit, &time);
            let neg_v: Vec<f64> = v_at_qi.iter().map(|&x| -x).collect();

            // Transport from qi to mu
            match method {
                TransportMethod::SchildsLadder => {
                    parallel_transport_schilds(&neg_v, &qi_unit, &mu_unit, &time)
                }
                TransportMethod::PoleLadder => {
                    parallel_transport_pole(&neg_v, &qi_unit, &mu_unit, &time)
                }
                TransportMethod::LogMap => unreachable!(),
            }
        })
        .collect();

    let mut tangent_vectors = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            tangent_vectors[(i, j)] = tangent_data[i][j];
        }
    }

    let initial_values: Vec<f64> = (0..n).map(|i| karcher.aligned_data[(i, 0)]).collect();

    TsrvfResult {
        tangent_vectors,
        mean: karcher.mean.clone(),
        mean_srsf: mean_srsf_smooth,
        mean_srsf_norm: mean_norm,
        srsf_norms,
        initial_values,
        gammas: karcher.gammas.clone(),
        converged: karcher.converged,
    }
}
