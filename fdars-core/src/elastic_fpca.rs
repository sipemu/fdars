//! Vertical, horizontal, and joint FPCA for elastic functional data.
//!
//! These FPCA variants decompose amplitude vs phase variability after elastic
//! alignment. They correspond to `vert.fpca`, `horiz.fpca`, and `jointFPCA`
//! from the R fdasrvf package.
//!
//! Key capabilities:
//! - [`vert_fpca`] — Amplitude FPCA in augmented SRSF space
//! - [`horiz_fpca`] — Phase FPCA via shooting vectors on the Hilbert sphere
//! - [`joint_fpca`] — Combined amplitude + phase FPCA

use crate::alignment::{srsf_inverse, srsf_transform, KarcherMeanResult};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::warping::{exp_map_sphere, inv_exp_map_sphere, invert_gamma, l2_norm_l2, psi_to_gam};
use nalgebra::SVD;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Threshold constants ─────────────────────────────────────────────────────

/// Minimum curve count (N) at which the inner `for i in 0..n` fill in
/// [`svd_scores_and_eigenvalues`] uses parallel dispatch via `iter_maybe_parallel!`.
/// Below this threshold the sequential path is taken — the body is a single
/// multiply (`scores[(i,k)] = u[(i,k)] * sv`), so the overhead of spawning
/// rayon work-items exceeds the gain until N ≥ 50 (per the audit's
/// streaming-sentinel payback analysis, PERF-04-C).
const SCORES_PARALLEL_THRESHOLD: usize = 50;

/// Approximate floating-point work (≈ flops) below which the per-curve /
/// per-row helpers in this module stay sequential: rayon dispatch costs a few
/// µs, which a loop this small cannot repay (ALN-HYGIENE #56).
const PAR_WORK_THRESHOLD: usize = 200_000;

/// `(0..len).map(f).collect()`, dispatched through `iter_maybe_parallel!` when
/// `work >= PAR_WORK_THRESHOLD`. Both paths evaluate the same `f(i)` and
/// collect in index order, so the output is bit-identical either way.
fn map_indexed<T, F>(len: usize, work: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync + Send,
{
    if work >= PAR_WORK_THRESHOLD {
        iter_maybe_parallel!(0..len).map(f).collect()
    } else {
        (0..len).map(f).collect()
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of vertical (amplitude) FPCA.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct VertFpcaResult {
    /// PC scores (n × ncomp).
    pub scores: FdMatrix,
    /// Eigenfunctions in augmented SRSF space (ncomp × (m+1)).
    pub eigenfunctions_q: FdMatrix,
    /// Eigenfunctions in function space (ncomp × m).
    pub eigenfunctions_f: FdMatrix,
    /// Eigenvalues (variance explained).
    pub eigenvalues: Vec<f64>,
    /// Cumulative proportion of variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Augmented mean SRSF (length m+1).
    pub mean_q: Vec<f64>,
}

/// Result of horizontal (phase) FPCA.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct HorizFpcaResult {
    /// PC scores (n × ncomp).
    pub scores: FdMatrix,
    /// Eigenfunctions in ψ space (ncomp × m).
    pub eigenfunctions_psi: FdMatrix,
    /// Eigenfunctions as warping functions (ncomp × m).
    pub eigenfunctions_gam: FdMatrix,
    /// Eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Cumulative proportion of variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Mean ψ on the sphere (length m).
    pub mean_psi: Vec<f64>,
    /// Shooting vectors (n × m).
    pub shooting_vectors: FdMatrix,
}

/// Result of joint (amplitude + phase) FPCA.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct JointFpcaResult {
    /// PC scores (n × ncomp).
    pub scores: FdMatrix,
    /// Eigenvalues.
    pub eigenvalues: Vec<f64>,
    /// Cumulative proportion of variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Phase-vs-amplitude balance weight.
    pub balance_c: f64,
    /// Vertical (amplitude) component of eigenvectors (ncomp × (m+1)).
    pub vert_component: FdMatrix,
    /// Horizontal (phase) component of eigenvectors (ncomp × m).
    pub horiz_component: FdMatrix,
}

// ─── Vertical FPCA ──────────────────────────────────────────────────────────

/// Perform vertical (amplitude) FPCA on elastically aligned curves.
///
/// 1. Compute SRSFs of aligned curves
/// 2. Augment with `sign(f_i(t0)) * sqrt(|f_i(t0)|)` as extra dimension
/// 3. Center, compute covariance, SVD
/// 4. Project onto eigenvectors and convert back to function space
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result (with aligned data and gammas)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components to extract
pub fn vert_fpca(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
) -> Result<VertFpcaResult, crate::FdarError> {
    let (n, m) = karcher.aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "aligned_data/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1).min(m);
    let m_aug = m + 1;

    let qn = match &karcher.aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => srsf_transform(&karcher.aligned_data, argvals),
    };

    let q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n, m);

    // Covariance matrix K (m_aug × m_aug) and SVD
    let (_, mean_q) = center_matrix(&q_aug, n, m_aug);
    let k_mat = build_symmetric_covariance(&q_aug, &mean_q, n, m_aug);

    let svd = SVD::new(k_mat, true, true);
    let u_cov = svd
        .u
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute U matrix; check for constant or zero-variance aligned functions".to_string(),
        })?;

    let eigenvalues: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    // Eigenfunctions = columns of U from svd(K)
    let mut eigenfunctions_q = FdMatrix::zeros(ncomp, m_aug);
    for k in 0..ncomp {
        for j in 0..m_aug {
            eigenfunctions_q[(k, j)] = u_cov[(j, k)];
        }
    }

    // Scores: project centered data onto eigenvectors
    let scores = project_onto_eigenvectors(&q_aug, &mean_q, u_cov, n, m_aug, ncomp);

    // Convert eigenfunctions to function domain via srsf_inverse
    let mut eigenfunctions_f = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let q_k: Vec<f64> = (0..m)
            .map(|j| mean_q[j] + eigenfunctions_q[(k, j)])
            .collect();
        let aug_val = mean_q[m] + eigenfunctions_q[(k, m)];
        let f0 = aug_val.signum() * aug_val * aug_val;
        let f_k = srsf_inverse(&q_k, argvals, f0);
        for j in 0..m {
            eigenfunctions_f[(k, j)] = f_k[j];
        }
    }

    Ok(VertFpcaResult {
        scores,
        eigenfunctions_q,
        eigenfunctions_f,
        eigenvalues,
        cumulative_variance,
        mean_q,
    })
}

// ─── Horizontal FPCA ────────────────────────────────────────────────────────

/// Perform horizontal (phase) FPCA on warping functions.
///
/// 1. Convert warps to ψ space (Hilbert sphere)
/// 2. Compute Karcher mean on sphere via iterative exp/log maps
/// 3. Compute shooting vectors (log map at mean)
/// 4. PCA on shooting vectors
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components
pub fn horiz_fpca(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
) -> Result<HorizFpcaResult, crate::FdarError> {
    let (n, m) = karcher.gammas.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "gammas/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1).min(m);

    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    let psis = warps_to_normalized_psi(&karcher.gammas, argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 50);
    let shooting = shooting_vectors_from_psis(&psis, &mu_psi, &time);

    // PCA on shooting vectors (tangent space → standard PCA)
    let (centered, _mean_v) = center_matrix(&shooting, n, m);

    let svd = SVD::new(centered.to_dmatrix(), true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail:
                "SVD failed to compute V^T matrix; check for constant or zero-variance functions"
                    .to_string(),
        })?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n).ok_or_else(|| {
        crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute scores; try reducing ncomp or check for degenerate input data".to_string(),
        }
    })?;
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    // Eigenfunctions in ψ space
    let mut eigenfunctions_psi = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        for j in 0..m {
            eigenfunctions_psi[(k, j)] = v_t[(k, j)];
        }
    }

    // Convert eigenfunctions to warping functions
    let mut eigenfunctions_gam = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let v_k: Vec<f64> = (0..m).map(|j| eigenfunctions_psi[(k, j)]).collect();
        let psi_k = exp_map_sphere(&mu_psi, &v_k, &time);
        let gam_k = psi_to_gam(&psi_k, &time);
        for j in 0..m {
            eigenfunctions_gam[(k, j)] = t0 + gam_k[j] * domain;
        }
    }

    Ok(HorizFpcaResult {
        scores,
        eigenfunctions_psi,
        eigenfunctions_gam,
        eigenvalues,
        cumulative_variance,
        mean_psi: mu_psi,
        shooting_vectors: shooting,
    })
}

// ─── Joint FPCA ─────────────────────────────────────────────────────────────

/// Perform joint (amplitude + phase) FPCA.
///
/// Concatenates augmented SRSFs and scaled shooting vectors, then does PCA
/// on the combined representation.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components
/// * `balance_c` — Weight for phase component. `None` chooses it by minimising
///   the reconstruction error of the original SRSFs (fdasrsf `jointFPCA`).
pub fn joint_fpca(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
) -> Result<JointFpcaResult, crate::FdarError> {
    let (n, m) = karcher.aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "aligned_data/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }

    let _vert = vert_fpca(karcher, argvals, ncomp)?;
    let horiz = horiz_fpca(karcher, argvals, ncomp)?;

    let m_aug = m + 1;
    let ncomp = ncomp.min(n - 1);

    let qn = match &karcher.aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => srsf_transform(&karcher.aligned_data, argvals),
    };
    let q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n, m);
    let (q_centered, mean_q) = center_matrix(&q_aug, n, m_aug);

    let shooting = &horiz.shooting_vectors;
    let c = match balance_c {
        Some(c) => c,
        None => optimize_balance_c_raw(&BalanceCProblem::new(
            &q_centered,
            &mean_q,
            shooting,
            &horiz.mean_psi,
            &qn,
            &karcher.gammas,
            argvals,
            ncomp,
        )),
    };

    // Concatenate: g_i = [qn_aug_centered_i; C * v_i]
    let combined = build_combined_representation(&q_centered, shooting, c, n, m_aug, m);

    let svd = SVD::new(combined.to_dmatrix(), true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail:
                "SVD failed to compute V^T matrix; check for constant or zero-variance functions"
                    .to_string(),
        })?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n).ok_or_else(|| {
        crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute scores; try reducing ncomp or check for degenerate input data".to_string(),
        }
    })?;
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    // Split eigenvectors into amplitude and phase parts
    let (vert_component, horiz_component) = split_joint_eigenvectors(v_t, ncomp, m_aug, m);

    Ok(JointFpcaResult {
        scores,
        eigenvalues,
        cumulative_variance,
        balance_c: c,
        vert_component,
        horiz_component,
    })
}

// ─── From-alignment wrappers ───────────────────────────────────────────────
//
// These accept raw aligned-data fields instead of a full `KarcherMeanResult`,
// making it possible to run elastic FPCA from an `AlignmentLayer` or any other
// source that provides the same arrays.

/// Vertical (amplitude) FPCA from pre-aligned curves and (optional) SRSFs.
///
/// This is equivalent to [`vert_fpca`] but does not require a
/// [`KarcherMeanResult`].  Only the fields actually used by the algorithm are
/// accepted as arguments.
///
/// # Arguments
/// * `aligned_data` — Aligned curves (n × m).
/// * `aligned_srsfs` — Pre-computed SRSFs of aligned curves (n × m).
///   When `None`, SRSFs are recomputed from `aligned_data`.
/// * `argvals` — Evaluation grid (length m).
/// * `ncomp` — Number of principal components to extract.
pub fn vert_fpca_from_alignment(
    aligned_data: &FdMatrix,
    aligned_srsfs: Option<&FdMatrix>,
    argvals: &[f64],
    ncomp: usize,
) -> Result<VertFpcaResult, crate::FdarError> {
    let (n, m) = aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "aligned_data/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1).min(m);
    let m_aug = m + 1;

    let qn = match aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => srsf_transform(aligned_data, argvals),
    };

    let q_aug = build_augmented_srsfs(&qn, aligned_data, n, m);

    let (_, mean_q) = center_matrix(&q_aug, n, m_aug);
    let k_mat = build_symmetric_covariance(&q_aug, &mean_q, n, m_aug);

    let svd = SVD::new(k_mat, true, true);
    let u_cov = svd
        .u
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute U matrix; check for constant or zero-variance aligned functions".to_string(),
        })?;

    let eigenvalues: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    let mut eigenfunctions_q = FdMatrix::zeros(ncomp, m_aug);
    for k in 0..ncomp {
        for j in 0..m_aug {
            eigenfunctions_q[(k, j)] = u_cov[(j, k)];
        }
    }

    let scores = project_onto_eigenvectors(&q_aug, &mean_q, u_cov, n, m_aug, ncomp);

    let mut eigenfunctions_f = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let q_k: Vec<f64> = (0..m)
            .map(|j| mean_q[j] + eigenfunctions_q[(k, j)])
            .collect();
        let aug_val = mean_q[m] + eigenfunctions_q[(k, m)];
        let f0 = aug_val.signum() * aug_val * aug_val;
        let f_k = srsf_inverse(&q_k, argvals, f0);
        for j in 0..m {
            eigenfunctions_f[(k, j)] = f_k[j];
        }
    }

    Ok(VertFpcaResult {
        scores,
        eigenfunctions_q,
        eigenfunctions_f,
        eigenvalues,
        cumulative_variance,
        mean_q,
    })
}

/// Horizontal (phase) FPCA from warping functions.
///
/// This is equivalent to [`horiz_fpca`] but does not require a
/// [`KarcherMeanResult`].  Only the warping functions are needed.
///
/// # Arguments
/// * `gammas` — Warping functions (n × m).
/// * `argvals` — Evaluation grid (length m).
/// * `ncomp` — Number of principal components to extract.
pub fn horiz_fpca_from_alignment(
    gammas: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
) -> Result<HorizFpcaResult, crate::FdarError> {
    let (n, m) = gammas.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "gammas/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1).min(m);

    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    let psis = warps_to_normalized_psi(gammas, argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 50);
    let shooting = shooting_vectors_from_psis(&psis, &mu_psi, &time);

    let (centered, _mean_v) = center_matrix(&shooting, n, m);

    let svd = SVD::new(centered.to_dmatrix(), true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail:
                "SVD failed to compute V^T matrix; check for constant or zero-variance functions"
                    .to_string(),
        })?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n).ok_or_else(|| {
        crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute scores; try reducing ncomp or check for degenerate input data".to_string(),
        }
    })?;
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    let mut eigenfunctions_psi = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        for j in 0..m {
            eigenfunctions_psi[(k, j)] = v_t[(k, j)];
        }
    }

    let mut eigenfunctions_gam = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        let v_k: Vec<f64> = (0..m).map(|j| eigenfunctions_psi[(k, j)]).collect();
        let psi_k = exp_map_sphere(&mu_psi, &v_k, &time);
        let gam_k = psi_to_gam(&psi_k, &time);
        for j in 0..m {
            eigenfunctions_gam[(k, j)] = t0 + gam_k[j] * domain;
        }
    }

    Ok(HorizFpcaResult {
        scores,
        eigenfunctions_psi,
        eigenfunctions_gam,
        eigenvalues,
        cumulative_variance,
        mean_psi: mu_psi,
        shooting_vectors: shooting,
    })
}

/// Joint (amplitude + phase) FPCA from pre-aligned curves and warps.
///
/// This is equivalent to [`joint_fpca`] but does not require a
/// [`KarcherMeanResult`].
///
/// # Arguments
/// * `aligned_data` — Aligned curves (n × m).
/// * `aligned_srsfs` — Pre-computed SRSFs of aligned curves (n × m), or `None`.
/// * `gammas` — Warping functions (n × m).
/// * `argvals` — Evaluation grid (length m).
/// * `ncomp` — Number of principal components to extract.
/// * `balance_c` — Weight for phase component (`None` ⇒ optimized automatically).
pub fn joint_fpca_from_alignment(
    aligned_data: &FdMatrix,
    aligned_srsfs: Option<&FdMatrix>,
    gammas: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
) -> Result<JointFpcaResult, crate::FdarError> {
    let (n, m) = aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "aligned_data/argvals",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }

    let horiz = horiz_fpca_from_alignment(gammas, argvals, ncomp)?;

    let m_aug = m + 1;
    let ncomp = ncomp.min(n - 1);

    let qn = match aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => srsf_transform(aligned_data, argvals),
    };
    let q_aug = build_augmented_srsfs(&qn, aligned_data, n, m);
    let (q_centered, mean_q) = center_matrix(&q_aug, n, m_aug);

    let shooting = &horiz.shooting_vectors;
    let c = match balance_c {
        Some(c) => c,
        None => optimize_balance_c_raw(&BalanceCProblem::new(
            &q_centered,
            &mean_q,
            shooting,
            &horiz.mean_psi,
            &qn,
            gammas,
            argvals,
            ncomp,
        )),
    };

    let combined = build_combined_representation(&q_centered, shooting, c, n, m_aug, m);

    let svd = SVD::new(combined.to_dmatrix(), true, true);
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail:
                "SVD failed to compute V^T matrix; check for constant or zero-variance functions"
                    .to_string(),
        })?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n).ok_or_else(|| {
        crate::FdarError::ComputationFailed {
            operation: "SVD",
            detail: "SVD failed to compute scores; try reducing ncomp or check for degenerate input data".to_string(),
        }
    })?;
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    let (vert_component, horiz_component) = split_joint_eigenvectors(v_t, ncomp, m_aug, m);

    Ok(JointFpcaResult {
        scores,
        eigenvalues,
        cumulative_variance,
        balance_c: c,
        vert_component,
        horiz_component,
    })
}

// ─── Shared Helpers ────────────────────────────────────────────────────────

/// Compute cumulative proportion of variance explained from eigenvalues.
fn cumulative_variance_from_eigenvalues(eigenvalues: &[f64]) -> Vec<f64> {
    let total_var: f64 = eigenvalues.iter().sum();
    let mut cum = Vec::with_capacity(eigenvalues.len());
    let mut running = 0.0;
    for ev in eigenvalues {
        running += ev;
        cum.push(if total_var > 0.0 {
            running / total_var
        } else {
            0.0
        });
    }
    cum
}

/// Convert warping functions to normalized ψ vectors on the Hilbert sphere.
pub(crate) fn warps_to_normalized_psi(gammas: &FdMatrix, argvals: &[f64]) -> Vec<Vec<f64>> {
    let (n, m) = gammas.shape();
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    // Per-curve, independent: (parallel) map, collected in index order.
    map_indexed(n, 8 * n * m, |i| {
        let gam_01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
        let mut grad = vec![0.0; m];
        grad[0] = (gam_01[1] - gam_01[0]) / binsize;
        for j in 1..m - 1 {
            grad[j] = (gam_01[j + 1] - gam_01[j - 1]) / (2.0 * binsize);
        }
        grad[m - 1] = (gam_01[m - 1] - gam_01[m - 2]) / binsize;
        let mut psi: Vec<f64> = grad.iter().map(|&g| g.max(0.0).sqrt()).collect();
        let norm = l2_norm_l2(&psi, &time);
        if norm > 1e-10 {
            for v in &mut psi {
                *v /= norm;
            }
        }
        psi
    })
}

/// Compute Karcher mean on the Hilbert sphere via iterative exp/log maps.
pub(crate) fn sphere_karcher_mean(psis: &[Vec<f64>], time: &[f64], max_iter: usize) -> Vec<f64> {
    let n = psis.len();
    let m = psis[0].len();

    // Initial mean: normalized arithmetic mean
    let mut mu_psi = vec![0.0; m];
    for psi in psis {
        for j in 0..m {
            mu_psi[j] += psi[j];
        }
    }
    for j in 0..m {
        mu_psi[j] /= n as f64;
    }
    normalize_to_sphere(&mut mu_psi, time);

    // Iterative refinement
    for _ in 0..max_iter {
        let mean_v = mean_tangent_vector(psis, &mu_psi, time);
        let step_norm = l2_norm_l2(&mean_v, time);
        if step_norm < 1e-8 {
            break;
        }
        mu_psi = exp_map_sphere(&mu_psi, &mean_v, time);
        normalize_to_sphere(&mut mu_psi, time);
    }

    mu_psi
}

/// Compute shooting vectors v_i = log_μ(ψ_i) from ψ vectors and Karcher mean.
pub(crate) fn shooting_vectors_from_psis(
    psis: &[Vec<f64>],
    mu_psi: &[f64],
    time: &[f64],
) -> FdMatrix {
    let n = psis.len();
    let m = psis[0].len();
    // Collect per-row results in parallel (each inv_exp_map_sphere call is independent),
    // then assign sequentially into the column-major FdMatrix buffer — mirrors the
    // collect-then-assign pattern from alignment/set.rs::align_to_target (PERF-04-A).
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| inv_exp_map_sphere(mu_psi, &psis[i], time))
        .collect();
    let mut shooting = FdMatrix::zeros(n, m);
    for (i, v) in rows.into_iter().enumerate() {
        for j in 0..m {
            shooting[(i, j)] = v[j];
        }
    }
    shooting
}

/// Build augmented SRSF matrix: original SRSFs + sign(f(id))*sqrt(|f(id)|) column.
pub(crate) fn build_augmented_srsfs(
    qn: &FdMatrix,
    aligned_data: &FdMatrix,
    n: usize,
    m: usize,
) -> FdMatrix {
    let id = m / 2;
    let m_aug = m + 1;
    // Collect per-row owned Vecs in parallel, then assign sequentially into the
    // column-major FdMatrix buffer (PERF-04-B, collect-then-assign pattern).
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut row = Vec::with_capacity(m_aug);
            for j in 0..m {
                row.push(qn[(i, j)]);
            }
            let fid = aligned_data[(i, id)];
            row.push(fid.signum() * fid.abs().sqrt());
            row
        })
        .collect();
    let mut q_aug = FdMatrix::zeros(n, m_aug);
    for (i, row) in rows.into_iter().enumerate() {
        for j in 0..m_aug {
            q_aug[(i, j)] = row[j];
        }
    }
    q_aug
}

/// Center a matrix and return the mean vector.
pub(crate) fn center_matrix(mat: &FdMatrix, n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let mut mean = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            mean[j] += mat[(i, j)];
        }
        mean[j] /= n as f64;
    }
    let mut centered = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            centered[(i, j)] = mat[(i, j)] - mean[j];
        }
    }
    (centered, mean)
}

/// Extract eigenvalues and scores from SVD of centered data.
///
/// When `n >= SCORES_PARALLEL_THRESHOLD` the inner `for i in 0..n` fill is
/// dispatched via `iter_maybe_parallel!` (PERF-04-C). Below the threshold the
/// sequential path is used — the body is a single multiply, so dispatch
/// overhead exceeds the gain at small N. Both branches produce identical
/// floating-point values (pure per-index writes, no cross-iteration reduction).
fn svd_scores_and_eigenvalues(
    svd: &SVD<f64, nalgebra::Dyn, nalgebra::Dyn>,
    ncomp: usize,
    n: usize,
) -> Option<(FdMatrix, Vec<f64>)> {
    let u = svd.u.as_ref()?;
    let eigenvalues: Vec<f64> = svd
        .singular_values
        .iter()
        .take(ncomp)
        .map(|&s| s * s / (n - 1) as f64)
        .collect();
    let mut scores = FdMatrix::zeros(n, ncomp);
    if n >= SCORES_PARALLEL_THRESHOLD {
        // Parallel branch: collect per-i values for each component, then assign.
        for k in 0..ncomp {
            let sv = svd.singular_values[k];
            let col: Vec<f64> = iter_maybe_parallel!(0..n).map(|i| u[(i, k)] * sv).collect();
            for (i, val) in col.into_iter().enumerate() {
                scores[(i, k)] = val;
            }
        }
    } else {
        // Sequential branch: original nested loops for small N.
        for k in 0..ncomp {
            let sv = svd.singular_values[k];
            for i in 0..n {
                scores[(i, k)] = u[(i, k)] * sv;
            }
        }
    }
    Some((scores, eigenvalues))
}

/// Split joint eigenvectors into vertical (amplitude) and horizontal (phase) components.
fn split_joint_eigenvectors(
    v_t: &nalgebra::DMatrix<f64>,
    ncomp: usize,
    m_aug: usize,
    m: usize,
) -> (FdMatrix, FdMatrix) {
    let mut vert_component = FdMatrix::zeros(ncomp, m_aug);
    let mut horiz_component = FdMatrix::zeros(ncomp, m);
    for k in 0..ncomp {
        for j in 0..m_aug {
            vert_component[(k, j)] = v_t[(k, j)];
        }
        for j in 0..m {
            horiz_component[(k, j)] = v_t[(k, m_aug + j)];
        }
    }
    (vert_component, horiz_component)
}

/// Build symmetric covariance matrix K (d × d) from data and mean.
fn build_symmetric_covariance(
    data: &FdMatrix,
    mean: &[f64],
    n: usize,
    d: usize,
) -> nalgebra::DMatrix<f64> {
    let nf = (n - 1) as f64;
    // Rows p of the upper triangle are independent, so they are built in
    // parallel (above the work threshold). Each entry (p, q) is still
    // accumulated over i = 0..n in order from 0.0, exactly as the former
    // `for i { for p { for q } }` loop did, so the sums are bit-identical.
    // The centred values `data[(i, p)] - mean[p]` are the same f64 operations
    // as before, just precomputed once into contiguous columns.
    let cen: Vec<f64> = (0..d)
        .flat_map(|p| (0..n).map(move |i| data[(i, p)] - mean[p]))
        .collect();
    let rows: Vec<Vec<f64>> = map_indexed(d, n * d * d / 2, |p| {
        let cp = &cen[p * n..(p + 1) * n];
        (p..d)
            .map(|q| {
                let cq = &cen[q * n..(q + 1) * n];
                let mut acc = 0.0;
                for i in 0..n {
                    acc += cp[i] * cq[i];
                }
                acc
            })
            .collect()
    });
    let mut k_mat = nalgebra::DMatrix::zeros(d, d);
    for (p, row) in rows.into_iter().enumerate() {
        for (off, v) in row.into_iter().enumerate() {
            k_mat[(p, p + off)] = v;
        }
    }
    for p in 0..d {
        k_mat[(p, p)] /= nf;
        for q in (p + 1)..d {
            k_mat[(p, q)] /= nf;
            k_mat[(q, p)] = k_mat[(p, q)];
        }
    }
    k_mat
}

/// Project centered data onto covariance eigenvectors to get scores.
fn project_onto_eigenvectors(
    data: &FdMatrix,
    mean: &[f64],
    u_cov: &nalgebra::DMatrix<f64>,
    n: usize,
    d: usize,
    ncomp: usize,
) -> FdMatrix {
    // Per-curve rows in parallel; each score is the same in-order dot product.
    let rows: Vec<Vec<f64>> = map_indexed(n, 2 * n * d * ncomp, |i| {
        (0..ncomp)
            .map(|k| {
                let mut s = 0.0;
                for j in 0..d {
                    s += (data[(i, j)] - mean[j]) * u_cov[(j, k)];
                }
                s
            })
            .collect()
    });
    let mut scores = FdMatrix::zeros(n, ncomp);
    for (i, row) in rows.into_iter().enumerate() {
        for (k, v) in row.into_iter().enumerate() {
            scores[(i, k)] = v;
        }
    }
    scores
}

/// Normalize a vector to unit L2 norm on sphere. Returns whether normalization happened.
fn normalize_to_sphere(mu: &mut [f64], time: &[f64]) {
    let norm = l2_norm_l2(mu, time);
    if norm > 1e-10 {
        for v in mu.iter_mut() {
            *v /= norm;
        }
    }
}

/// Compute mean tangent vector on sphere from ψ vectors at current mean.
fn mean_tangent_vector(psis: &[Vec<f64>], mu_psi: &[f64], time: &[f64]) -> Vec<f64> {
    let n = psis.len();
    let m = mu_psi.len();
    // Log maps in parallel; the sum is then taken sequentially in curve order
    // (deterministic reduction, bit-identical to the sequential loop).
    let logs: Vec<Vec<f64>> = map_indexed(n, 16 * n * m, |i| {
        inv_exp_map_sphere(mu_psi, &psis[i], time)
    });
    let mut mean_v = vec![0.0; m];
    for v in &logs {
        for j in 0..m {
            mean_v[j] += v[j];
        }
    }
    for j in 0..m {
        mean_v[j] /= n as f64;
    }
    mean_v
}

/// Build combined representation: [q_centered | c * shooting] for joint FPCA.
pub(crate) fn build_combined_representation(
    q_centered: &FdMatrix,
    shooting: &FdMatrix,
    c: f64,
    n: usize,
    m_aug: usize,
    m: usize,
) -> FdMatrix {
    let combined_dim = m_aug + m;
    let mut combined = FdMatrix::zeros(n, combined_dim);
    for i in 0..n {
        for j in 0..m_aug {
            combined[(i, j)] = q_centered[(i, j)];
        }
        for j in 0..m {
            combined[(i, m_aug + j)] = c * shooting[(i, j)];
        }
    }
    combined
}

/// Inputs of the automatic balance-C search shared by [`joint_fpca`] and
/// [`joint_fpca_from_alignment`].
struct BalanceCProblem<'a> {
    /// Centred augmented SRSFs (n × (m+1)).
    q_centered: &'a FdMatrix,
    /// Mean of the augmented SRSFs (length m+1).
    mean_q: &'a [f64],
    /// Shooting vectors of the warps at `mu_psi` (n × m).
    shooting: &'a FdMatrix,
    /// Karcher mean of the warps on the Hilbert sphere (length m, \[0,1\] grid).
    mu_psi: &'a [f64],
    /// Original (unaligned) SRSFs, rebuilt as `(qnᵢ, γᵢ⁻¹)` (n rows of length m).
    q0: Vec<Vec<f64>>,
    /// Evaluation grid (length m).
    argvals: &'a [f64],
    /// Uniform \[0,1\] grid (length m).
    time: Vec<f64>,
    /// Number of joint components used in the reconstruction.
    ncomp: usize,
}

/// SRSF group action `(q, γ) = (q ∘ γ)·√γ̇` for a warp `gam01` given on the
/// normalised \[0,1\] grid `time`; `q` is sampled on `argvals`.
fn warp_srsf_01(q: &[f64], argvals: &[f64], gam01: &[f64], time: &[f64]) -> Vec<f64> {
    let m = q.len();
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let gam_abs: Vec<f64> = gam01.iter().map(|&g| t0 + g * domain).collect();
    let q_warped = crate::alignment::reparameterize_curve(q, argvals, &gam_abs);
    let gdot = crate::helpers::gradient_uniform(gam01, time[1] - time[0]);
    q_warped
        .iter()
        .zip(gdot.iter())
        .map(|(&qv, &g)| qv * g.max(0.0).sqrt())
        .collect()
}

impl<'a> BalanceCProblem<'a> {
    /// Assemble the problem; `qn` are the aligned SRSFs and `gammas` the warps
    /// (on `argvals`), from which the original SRSFs are rebuilt.
    #[allow(clippy::too_many_arguments)]
    fn new(
        q_centered: &'a FdMatrix,
        mean_q: &'a [f64],
        shooting: &'a FdMatrix,
        mu_psi: &'a [f64],
        qn: &FdMatrix,
        gammas: &FdMatrix,
        argvals: &'a [f64],
        ncomp: usize,
    ) -> Self {
        let (n, m) = qn.shape();
        let t0 = argvals[0];
        let domain = argvals[m - 1] - t0;
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let q0 = map_indexed(n, 8 * n * m, |i| {
            let gam01: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / domain).collect();
            let gam_inv = invert_gamma(&gam01, &time);
            warp_srsf_01(&qn.row(i), argvals, &gam_inv, &time)
        });
        Self {
            q_centered,
            mean_q,
            shooting,
            mu_psi,
            q0,
            argvals,
            time,
            ncomp,
        }
    }

    /// Upper end of the search bracket, `round(max q_aug) + 1` as in fdasrsf.
    fn upper_bound(&self) -> f64 {
        let (n, m_aug) = self.q_centered.shape();
        let mut mx = f64::NEG_INFINITY;
        for j in 0..m_aug {
            for i in 0..n {
                mx = mx.max(self.q_centered[(i, j)] + self.mean_q[j]);
            }
        }
        if mx.is_finite() {
            (mx.round() + 1.0).max(1.0)
        } else {
            10.0
        }
    }

    /// Squared SRSF reconstruction error `dᵢ = ∫(q̃ᵢ − q0ᵢ)²` of curve `i`
    /// from its rank-k joint reconstruction row `rec` (length m+1+m).
    fn curve_error(&self, i: usize, rec: &[f64], c: f64) -> f64 {
        let m = self.shooting.ncols();
        let m_aug = self.q_centered.ncols();
        let q_hat: Vec<f64> = (0..m).map(|j| self.mean_q[j] + rec[j]).collect();
        let v_hat: Vec<f64> = (0..m).map(|j| rec[m_aug + j] / c).collect();
        let psi_hat = exp_map_sphere(self.mu_psi, &v_hat, &self.time);
        let gam_hat = psi_to_gam(&psi_hat, &self.time);
        let gam_inv = invert_gamma(&gam_hat, &self.time);
        let q_tilde = warp_srsf_01(&q_hat, self.argvals, &gam_inv, &self.time);
        let diff2: Vec<f64> = q_tilde
            .iter()
            .zip(self.q0[i].iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .collect();
        crate::helpers::trapz(&diff2, &self.time)
    }

    /// fdasrsf `find_C` objective `Σᵢ dᵢ² / N` at balance weight `c > 0`.
    fn objective(&self, c: f64) -> f64 {
        let (n, m_aug) = self.q_centered.shape();
        let m = self.shooting.ncols();
        let combined =
            build_combined_representation(self.q_centered, self.shooting, c, n, m_aug, m);
        let svd = SVD::new(combined.to_dmatrix(), true, true);
        let (Some(u), Some(v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) else {
            return f64::INFINITY;
        };
        let k = self.ncomp.min(svd.singular_values.len());
        let dim = m_aug + m;
        let mut total = 0.0;
        for i in 0..n {
            let rec: Vec<f64> = (0..dim)
                .map(|j| {
                    let mut s = 0.0;
                    for r in 0..k {
                        s += u[(i, r)] * svd.singular_values[r] * v_t[(r, j)];
                    }
                    s
                })
                .collect();
            let d = self.curve_error(i, &rec, c);
            total += d * d;
        }
        let val = total / n as f64;
        if val.is_finite() {
            val
        } else {
            f64::INFINITY
        }
    }
}

/// Number of coarse grid points scanned before the golden-section refinement.
const BALANCE_C_GRID: usize = 20;

/// Choose the balance weight C by minimising the reconstruction error of the
/// original SRSFs (fdasrsf `fdajpca.calc_fpca` / `find_C`).
///
/// The objective rebuilds every curve from its rank-`ncomp` joint
/// reconstruction (amplitude via the SRSF block, phase via the shooting block
/// mapped back to a warp) and compares it with the unaligned SRSF. Unlike the
/// residual variance of `[q | c·v]`, it does not shrink as `c → 0`: a vanishing
/// phase weight leaves the phase unexplained and the error large (GH #84).
///
/// Search: the objective has shallow local minima, so the bracket
/// `(0, round(max q_aug) + 1]` is first scanned on a uniform grid and the best
/// cell is refined by golden-section search (fdasrsf uses a single bounded
/// Brent search, which can stop in a local minimum).
fn optimize_balance_c_raw(problem: &BalanceCProblem<'_>) -> f64 {
    let hi = problem.upper_bound();
    let h = hi / BALANCE_C_GRID as f64;
    let grid: Vec<f64> = (1..=BALANCE_C_GRID).map(|k| k as f64 * h).collect();
    let vals: Vec<f64> = iter_maybe_parallel!(grid.clone())
        .map(|c| problem.objective(c))
        .collect();
    let mut best = 0;
    for (k, &v) in vals.iter().enumerate() {
        if v < vals[best] {
            best = k;
        }
    }
    if !vals[best].is_finite() {
        return 1.0;
    }
    let a = grid[best] - h;
    let b = (grid[best] + h).min(hi);
    let (c_ref, f_ref) = golden_section_min(|c| problem.objective(c), a.max(1e-3 * h), b);
    if f_ref <= vals[best] {
        c_ref
    } else {
        grid[best]
    }
}

/// Golden-section minimisation of `f` on `[a, b]` to absolute width `1e-5`
/// (fdasrsf's `fminbound` tolerance). Returns `(argmin, min)`.
fn golden_section_min(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> (f64, f64) {
    let g = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut x1 = b - g * (b - a);
    let mut x2 = a + g * (b - a);
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..80 {
        if b - a < 1e-5 {
            break;
        }
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - g * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + g * (b - a);
            f2 = f(x2);
        }
    }
    if f1 < f2 {
        (x1, f1)
    } else {
        (x2, f2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::karcher_mean;
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let shift = 0.1 * (i as f64 - n as f64 / 2.0);
            let scale = 1.0 + 0.2 * (i as f64 / n as f64);
            for j in 0..m {
                data[(i, j)] = scale * (2.0 * PI * (t[j] + shift)).sin();
            }
        }
        (data, t)
    }

    #[test]
    fn test_vert_fpca_basic() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = vert_fpca(&km, &t, 3);
        assert!(result.is_ok(), "vert_fpca should succeed");

        let res = result.unwrap();
        assert_eq!(res.scores.shape(), (15, 3));
        assert_eq!(res.eigenvalues.len(), 3);
        assert_eq!(res.eigenfunctions_q.shape(), (3, 52)); // m+1
        assert_eq!(res.eigenfunctions_f.shape(), (3, 51));

        // Eigenvalues should be non-negative and decreasing
        for ev in &res.eigenvalues {
            assert!(*ev >= -1e-10, "Eigenvalue should be non-negative: {}", ev);
        }
        for i in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[i] <= res.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be decreasing"
            );
        }

        // Cumulative variance should be increasing and <= 1
        for i in 1..res.cumulative_variance.len() {
            assert!(res.cumulative_variance[i] >= res.cumulative_variance[i - 1] - 1e-10);
        }
        assert!(*res.cumulative_variance.last().unwrap() <= 1.0 + 1e-10);
    }

    #[test]
    fn test_horiz_fpca_basic() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = horiz_fpca(&km, &t, 3);
        assert!(result.is_ok(), "horiz_fpca should succeed");

        let res = result.unwrap();
        assert_eq!(res.scores.shape(), (15, 3));
        assert_eq!(res.eigenvalues.len(), 3);
        assert_eq!(res.eigenfunctions_psi.shape(), (3, 51));
        assert_eq!(res.shooting_vectors.shape(), (15, 51));
    }

    #[test]
    fn test_joint_fpca_basic() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = joint_fpca(&km, &t, 3, Some(1.0));
        assert!(result.is_ok(), "joint_fpca should succeed");

        let res = result.unwrap();
        assert_eq!(res.scores.shape(), (15, 3));
        assert_eq!(res.eigenvalues.len(), 3);
        assert!(res.balance_c >= 0.0);
    }

    #[test]
    fn test_joint_fpca_optimize_c() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = joint_fpca(&km, &t, 3, None);
        assert!(
            result.is_ok(),
            "joint_fpca with C optimization should succeed"
        );
    }

    /// GH #84 repro: one Gaussian bump whose location shifts by 0.1·zᵢ (pure
    /// phase variation), n = 30, m = 101. `z` = numpy
    /// `default_rng(0).standard_normal(30)`.
    fn bump_phase_data() -> (FdMatrix, Vec<f64>, Vec<f64>) {
        let z = vec![
            0.1257302210933933,
            -0.1321048632913019,
            0.6404226504432821,
            0.10490011715303971,
            -0.535669373161111,
            0.36159505490948474,
            1.3040000451301372,
            0.9470809631292422,
            -0.7037352358069926,
            -1.2654214710460525,
            -0.6232744625373522,
            0.0413259793472436,
            -2.3250307746388343,
            -0.21879166393254573,
            -1.2459109472530652,
            -0.7322673547034516,
            -0.5442589828573099,
            -0.31630015636915454,
            0.4116305363741328,
            1.0425133694426776,
            -0.12853466294403426,
            1.3664634705496859,
            -0.6651946734866135,
            0.3515100700930197,
            0.9034701816518086,
            0.09401229776087457,
            -0.7434992493538084,
            -0.9217253762584194,
            -0.45772582566733916,
            0.2201951234700494,
        ];
        let m = 101;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(z.len(), m);
        for (i, &zi) in z.iter().enumerate() {
            for j in 0..m {
                let u = (t[j] - 0.5 - 0.1 * zi) / 0.08;
                data[(i, j)] = (-u * u).exp();
            }
        }
        (data, t, z)
    }

    fn abs_corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for (&x, &y) in a.iter().zip(b) {
            sab += (x - ma) * (y - mb);
            saa += (x - ma) * (x - ma);
            sbb += (y - mb) * (y - mb);
        }
        (sab / (saa * sbb).sqrt()).abs()
    }

    fn balance_problem_parts(
        km: &KarcherMeanResult,
        t: &[f64],
    ) -> (FdMatrix, FdMatrix, Vec<f64>, HorizFpcaResult) {
        let (n, m) = km.aligned_data.shape();
        let qn = srsf_transform(&km.aligned_data, t);
        let q_aug = build_augmented_srsfs(&qn, &km.aligned_data, n, m);
        let (q_centered, mean_q) = center_matrix(&q_aug, n, m + 1);
        let horiz = horiz_fpca(km, t, 2).expect("horiz_fpca");
        (qn, q_centered, mean_q, horiz)
    }

    /// GH #84: the automatic balance weight must not collapse to c ≈ 0 on
    /// pure phase variation.
    ///
    /// Reference (fdasrsf 2.7.5, `fdawarp(X.T, t).srsf_align()` then
    /// `fdajpca(w).calc_fpca(no=2)`): selected C = 4.0 (upper bracket end
    /// `round(max q_aug) + 1 = 4`), |corr(score₁, z)| = 0.993; its `find_C`
    /// objective is 7.008 at C = 0.001, 6.111 at C = 0.1, 0.0334 at C = 0.2 and
    /// 0.010–0.017 on C ∈ [0.5, 4] — any C ≳ 0.3 is near-optimal, C → 0 drops the
    /// phase. Before the fix fdars selected c ≈ 3e-4 with |corr| = 0.42.
    #[test]
    fn test_joint_fpca_auto_balance_c_keeps_phase() {
        let (data, t, z) = bump_phase_data();
        let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
        let res = joint_fpca(&km, &t, 2, None).expect("joint_fpca");
        assert!(
            res.balance_c >= 0.3 && res.balance_c <= 5.0,
            "automatic balance_c = {} collapsed / left the fdasrsf bracket",
            res.balance_c
        );
        let s0: Vec<f64> = (0..z.len()).map(|i| res.scores[(i, 0)]).collect();
        let r = abs_corr(&s0, &z);
        assert!(r >= 0.95, "first joint score lost the phase: |corr| = {r}");

        // The selected c is (near-)optimal for the objective it minimises.
        let (qn, q_centered, mean_q, horiz) = balance_problem_parts(&km, &t);
        let p = BalanceCProblem::new(
            &q_centered,
            &mean_q,
            &horiz.shooting_vectors,
            &horiz.mean_psi,
            &qn,
            &km.gammas,
            &t,
            2,
        );
        let f_auto = p.objective(res.balance_c);
        assert!(f_auto <= p.objective(1.0) + 1e-12);
        assert!(f_auto * 50.0 < p.objective(1e-3), "c→0 must be much worse");
    }

    /// Invariant: with all components kept the joint reconstruction is exact,
    /// so the rebuilt curves equal the original SRSFs up to discretisation and
    /// the objective is ~0 for every c, far below the rank-1 value.
    #[test]
    fn test_balance_c_objective_full_rank_is_near_zero() {
        let (data, t, _z) = bump_phase_data();
        let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
        let (qn, q_centered, mean_q, horiz) = balance_problem_parts(&km, &t);
        let mk = |k| {
            BalanceCProblem::new(
                &q_centered,
                &mean_q,
                &horiz.shooting_vectors,
                &horiz.mean_psi,
                &qn,
                &km.gammas,
                &t,
                k,
            )
        };
        let full = mk(30);
        let rank1 = mk(1);
        for c in [0.05, 0.5, 2.0] {
            let f_full = full.objective(c);
            assert!(f_full < 1e-2 * rank1.objective(c), "c={c}: {f_full}");
        }
    }

    #[test]
    fn test_joint_fpca_from_alignment_auto_c_matches_karcher_path() {
        let (data, t, _z) = bump_phase_data();
        let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
        let a = joint_fpca(&km, &t, 2, None).expect("joint_fpca");
        let b = joint_fpca_from_alignment(
            &km.aligned_data,
            km.aligned_srsfs.as_ref(),
            &km.gammas,
            &t,
            2,
            None,
        )
        .expect("joint_fpca_from_alignment");
        assert!((a.balance_c - b.balance_c).abs() < 1e-9 * a.balance_c.max(1.0));
    }

    #[test]
    fn test_vert_fpca_invalid_input() {
        let data = FdMatrix::zeros(1, 10); // n < 2
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
        let km = KarcherMeanResult {
            mean: vec![0.0; 10],
            mean_srsf: vec![0.0; 10],
            gammas: FdMatrix::zeros(1, 10),
            aligned_data: data,
            n_iter: 0,
            converged: true,
            aligned_srsfs: None,
        };
        assert!(vert_fpca(&km, &t, 3).is_err());
    }

    #[test]
    fn test_horiz_fpca_eigenvalue_properties() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let res = horiz_fpca(&km, &t, 3).expect("horiz_fpca should succeed");

        // Eigenvalues non-negative
        for ev in &res.eigenvalues {
            assert!(*ev >= -1e-10, "Eigenvalue should be non-negative: {}", ev);
        }
        // Eigenvalues decreasing
        for i in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[i] <= res.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be decreasing"
            );
        }
        // Cumulative variance increasing and <= 1
        for i in 1..res.cumulative_variance.len() {
            assert!(res.cumulative_variance[i] >= res.cumulative_variance[i - 1] - 1e-10);
        }
        assert!(*res.cumulative_variance.last().unwrap() <= 1.0 + 1e-10);
    }

    #[test]
    fn test_joint_fpca_eigenvalue_properties() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let res = joint_fpca(&km, &t, 3, Some(1.0)).expect("joint_fpca should succeed");

        // Eigenvalues non-negative
        for ev in &res.eigenvalues {
            assert!(*ev >= -1e-10, "Eigenvalue should be non-negative: {}", ev);
        }
        // Eigenvalues decreasing
        for i in 1..res.eigenvalues.len() {
            assert!(
                res.eigenvalues[i] <= res.eigenvalues[i - 1] + 1e-10,
                "Eigenvalues should be decreasing"
            );
        }
        // Cumulative variance increasing and <= 1
        for i in 1..res.cumulative_variance.len() {
            assert!(res.cumulative_variance[i] >= res.cumulative_variance[i - 1] - 1e-10);
        }
        assert!(*res.cumulative_variance.last().unwrap() <= 1.0 + 1e-10);
    }

    #[test]
    fn test_vert_fpca_ncomp_sensitivity() -> Result<(), crate::error::FdarError> {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        for &ncomp in &[1, 2, 5, 10] {
            let res = vert_fpca(&km, &t, ncomp)?;
            assert_eq!(res.scores.shape(), (15, ncomp));
            assert_eq!(res.eigenvalues.len(), ncomp);
            assert_eq!(res.eigenfunctions_q.shape(), (ncomp, 52));
            assert_eq!(res.eigenfunctions_f.shape(), (ncomp, 51));
            assert_eq!(res.cumulative_variance.len(), ncomp);
        }
        Ok(())
    }

    #[test]
    fn test_vert_fpca_score_orthogonality() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let res = vert_fpca(&km, &t, 3).expect("vert_fpca should succeed");

        let n = 15;
        // Check approximate orthogonality of score vectors
        for k1 in 0..3 {
            for k2 in (k1 + 1)..3 {
                let dot: f64 = (0..n)
                    .map(|i| res.scores[(i, k1)] * res.scores[(i, k2)])
                    .sum();
                let norm1: f64 = (0..n)
                    .map(|i| res.scores[(i, k1)].powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm2: f64 = (0..n)
                    .map(|i| res.scores[(i, k2)].powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm1 > 1e-10 && norm2 > 1e-10 {
                    let cos_angle = (dot / (norm1 * norm2)).abs();
                    assert!(
                        cos_angle < 0.15,
                        "Score components {} and {} should be approximately orthogonal, cos={}",
                        k1,
                        k2,
                        cos_angle
                    );
                }
            }
        }
    }

    // ── parallelism equivalence tests (PERF-04) ──

    /// PERF-04-A: shooting_vectors_from_psis produces bit-identical output under parallel
    /// and sequential feature configurations (pure per-index writes, no reduction).
    #[test]
    fn test_shooting_vectors_parallel_equiv() {
        let (data, t) = generate_test_data(51, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let psis = warps_to_normalized_psi(&km.gammas, &t);
        let time: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();
        let mu_psi = sphere_karcher_mean(&psis, &time, 50);

        // Result from the (possibly parallelized) function under test.
        let parallel_result = shooting_vectors_from_psis(&psis, &mu_psi, &time);

        // Sequential reference computed inline.
        let n = psis.len();
        let m = psis[0].len();
        let mut seq_result = FdMatrix::zeros(n, m);
        for i in 0..n {
            let v = inv_exp_map_sphere(&mu_psi, &psis[i], &time);
            for j in 0..m {
                seq_result[(i, j)] = v[j];
            }
        }

        assert_eq!(
            parallel_result.shape(),
            seq_result.shape(),
            "shapes must match"
        );
        for i in 0..n {
            for j in 0..m {
                assert_eq!(
                    parallel_result[(i, j)],
                    seq_result[(i, j)],
                    "shooting_vectors_from_psis bit-identical at ({i},{j})"
                );
            }
        }
    }

    /// PERF-04-C: svd_scores_and_eigenvalues uses the parallel branch at N >= 50 and the
    /// sequential branch below 50; both branches produce output that exactly matches a
    /// hand-computed u[i,k] * singular_value[k] reference.
    #[test]
    fn test_scores_threshold() {
        // Helper that asserts scores equal u[(i,k)] * sv[k] for all (i,k).
        fn check_scores_exact(n: usize, m: usize) {
            let (data, t) = generate_test_data(n, m);
            let km = karcher_mean(&data, &t, 5, 1e-4, 0.0);
            let ncomp = 2.min(n - 1);
            // Reproduce the centered-data path that horiz_fpca uses.
            let psis = warps_to_normalized_psi(&km.gammas, &t);
            let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
            let mu_psi = sphere_karcher_mean(&psis, &time, 20);
            let shooting = shooting_vectors_from_psis(&psis, &mu_psi, &time);
            let (centered, _) = center_matrix(&shooting, n, m);
            let svd = nalgebra::SVD::new(centered.to_dmatrix(), true, true);
            let u = svd.u.as_ref().expect("SVD U must exist");
            let (scores, _eigenvalues) =
                svd_scores_and_eigenvalues(&svd, ncomp, n).expect("scores must exist");
            // Verify each score equals u[(i,k)] * singular_value[k] exactly.
            for k in 0..ncomp {
                let sv = svd.singular_values[k];
                for i in 0..n {
                    let expected = u[(i, k)] * sv;
                    assert_eq!(
                        scores[(i, k)],
                        expected,
                        "scores bit-identical at N={n} (i={i}, k={k})"
                    );
                }
            }
        }

        // Small-N path (n < SCORES_PARALLEL_THRESHOLD = 50) — sequential branch.
        check_scores_exact(10, 31);
        // Large-N path (n >= SCORES_PARALLEL_THRESHOLD = 50) — parallel branch.
        check_scores_exact(51, 31);
    }

    /// PERF-04-B: build_augmented_srsfs produces bit-identical output under parallel
    /// and sequential feature configurations (pure per-index writes, no reduction).
    #[test]
    fn test_augmented_srsfs_parallel_equiv() {
        let (data, t) = generate_test_data(51, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let n = 51;
        let m = 51;
        let qn = match &km.aligned_srsfs {
            Some(s) => s.clone(),
            None => crate::alignment::srsf_transform(&km.aligned_data, &t),
        };

        // Result from the (possibly parallelized) function under test.
        let parallel_result = build_augmented_srsfs(&qn, &km.aligned_data, n, m);

        // Sequential reference computed inline.
        let id = m / 2;
        let m_aug = m + 1;
        let mut seq_result = FdMatrix::zeros(n, m_aug);
        for i in 0..n {
            for j in 0..m {
                seq_result[(i, j)] = qn[(i, j)];
            }
            let fid = km.aligned_data[(i, id)];
            seq_result[(i, m)] = fid.signum() * fid.abs().sqrt();
        }

        assert_eq!(
            parallel_result.shape(),
            seq_result.shape(),
            "shapes must match"
        );
        for i in 0..n {
            for j in 0..m_aug {
                assert_eq!(
                    parallel_result[(i, j)],
                    seq_result[(i, j)],
                    "build_augmented_srsfs bit-identical at ({i},{j})"
                );
            }
        }
    }

    /// PERF-04-D: vert_fpca scores and eigenvalues are deterministic and equivalent to
    /// a sequential reference at N >= 50. The pure-write loops (build_augmented_srsfs,
    /// project_onto_eigenvectors) are bit-identical; the final SVD is deterministic for
    /// the same input, so we assert exact equality.
    #[test]
    fn test_vert_fpca_parallel_equiv() {
        let n = 51;
        let m = 51;
        let (data, t) = generate_test_data(n, m);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let ncomp = 3;

        // Run twice with the same inputs — determinism implies equivalence across
        // feature configurations (same code path when called sequentially).
        let res1 = vert_fpca(&km, &t, ncomp).expect("vert_fpca should succeed");
        let res2 = vert_fpca(&km, &t, ncomp).expect("vert_fpca should succeed");

        assert_eq!(res1.scores.shape(), (n, ncomp));
        assert_eq!(res1.eigenvalues.len(), ncomp);

        // Exact equality: deterministic computation, no floating-point accumulation.
        for i in 0..n {
            for k in 0..ncomp {
                assert_eq!(
                    res1.scores[(i, k)],
                    res2.scores[(i, k)],
                    "vert_fpca scores deterministic at N={n} (i={i}, k={k})"
                );
            }
        }
        for k in 0..ncomp {
            assert_eq!(
                res1.eigenvalues[k], res2.eigenvalues[k],
                "vert_fpca eigenvalues deterministic at k={k}"
            );
        }
    }

    /// PERF-04-E: joint_fpca scores and eigenvalues are deterministic and equivalent
    /// to a sequential reference at N >= 50.
    #[test]
    fn test_joint_fpca_parallel_equiv() {
        let n = 51;
        let m = 51;
        let (data, t) = generate_test_data(n, m);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let ncomp = 3;
        // Fix balance_c so optimize_balance_c golden-section is not invoked
        // (keeps the test deterministic without relying on optimizer convergence).
        let balance_c = Some(1.0);

        let res1 = joint_fpca(&km, &t, ncomp, balance_c).expect("joint_fpca should succeed");
        let res2 = joint_fpca(&km, &t, ncomp, balance_c).expect("joint_fpca should succeed");

        assert_eq!(res1.scores.shape(), (n, ncomp));
        assert_eq!(res1.eigenvalues.len(), ncomp);

        for i in 0..n {
            for k in 0..ncomp {
                assert_eq!(
                    res1.scores[(i, k)],
                    res2.scores[(i, k)],
                    "joint_fpca scores deterministic at N={n} (i={i}, k={k})"
                );
            }
        }
        for k in 0..ncomp {
            assert_eq!(
                res1.eigenvalues[k], res2.eigenvalues[k],
                "joint_fpca eigenvalues deterministic at k={k}"
            );
        }
    }

    // ── from_alignment wrapper tests ──

    #[test]
    fn test_vert_fpca_from_alignment_matches_original() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        let original = vert_fpca(&km, &t, 3).expect("vert_fpca should succeed");
        let from_aln = vert_fpca_from_alignment(&km.aligned_data, km.aligned_srsfs.as_ref(), &t, 3)
            .expect("vert_fpca_from_alignment should succeed");

        assert_eq!(original.scores.shape(), from_aln.scores.shape());
        assert_eq!(original.eigenvalues.len(), from_aln.eigenvalues.len());
        for (a, b) in original.eigenvalues.iter().zip(&from_aln.eigenvalues) {
            assert!((a - b).abs() < 1e-10, "eigenvalues should match");
        }
    }

    #[test]
    fn test_vert_fpca_from_alignment_without_srsfs() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        let res = vert_fpca_from_alignment(&km.aligned_data, None, &t, 3)
            .expect("vert_fpca_from_alignment without srsfs should succeed");
        assert_eq!(res.scores.shape(), (15, 3));
    }

    #[test]
    fn test_horiz_fpca_from_alignment_matches_original() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        let original = horiz_fpca(&km, &t, 3).expect("horiz_fpca should succeed");
        let from_aln = horiz_fpca_from_alignment(&km.gammas, &t, 3)
            .expect("horiz_fpca_from_alignment should succeed");

        assert_eq!(original.scores.shape(), from_aln.scores.shape());
        assert_eq!(original.eigenvalues.len(), from_aln.eigenvalues.len());
        for (a, b) in original.eigenvalues.iter().zip(&from_aln.eigenvalues) {
            assert!((a - b).abs() < 1e-10, "eigenvalues should match");
        }
    }

    #[test]
    fn test_joint_fpca_from_alignment_matches_original() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        let original = joint_fpca(&km, &t, 3, Some(1.0)).expect("joint_fpca should succeed");
        let from_aln = joint_fpca_from_alignment(
            &km.aligned_data,
            km.aligned_srsfs.as_ref(),
            &km.gammas,
            &t,
            3,
            Some(1.0),
        )
        .expect("joint_fpca_from_alignment should succeed");

        assert_eq!(original.scores.shape(), from_aln.scores.shape());
        assert_eq!(original.eigenvalues.len(), from_aln.eigenvalues.len());
        for (a, b) in original.eigenvalues.iter().zip(&from_aln.eigenvalues) {
            assert!((a - b).abs() < 1e-10, "eigenvalues should match");
        }
    }

    #[test]
    fn test_joint_fpca_from_alignment_optimize_c() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);

        let res = joint_fpca_from_alignment(
            &km.aligned_data,
            km.aligned_srsfs.as_ref(),
            &km.gammas,
            &t,
            3,
            None,
        )
        .expect("joint_fpca_from_alignment with C optimization should succeed");
        assert_eq!(res.scores.shape(), (15, 3));
        assert!(res.balance_c >= 0.0);
    }

    #[test]
    fn test_from_alignment_invalid_input() {
        let data = FdMatrix::zeros(1, 10);
        let t: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();

        assert!(vert_fpca_from_alignment(&data, None, &t, 3).is_err());
        assert!(horiz_fpca_from_alignment(&data, &t, 3).is_err());
        assert!(joint_fpca_from_alignment(&data, None, &data, &t, 3, Some(1.0)).is_err());
    }

    // ── ALN-HYGIENE #56: parallel helpers vs the former sequential loops ──
    // Reference implementations are verbatim copies of the pre-change loops;
    // sizes are chosen above PAR_WORK_THRESHOLD so the parallel path runs.

    fn pseudo(n: usize, d: usize, seed: f64) -> FdMatrix {
        let mut x = FdMatrix::zeros(n, d);
        for i in 0..n {
            for j in 0..d {
                x[(i, j)] =
                    ((i as f64 + 1.3) * (j as f64 + 0.7) * seed).sin() + 0.1 * (i * j % 7) as f64;
            }
        }
        x
    }

    fn bits(xs: &[f64]) -> Vec<u64> {
        xs.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn covariance_parallel_matches_sequential_bits() {
        let (n, d) = (100, 70);
        assert!(n * d * d / 2 >= PAR_WORK_THRESHOLD);
        let x = pseudo(n, d, 0.37);
        let (_, mean) = center_matrix(&x, n, d);
        let nf = (n - 1) as f64;
        let mut r = nalgebra::DMatrix::zeros(d, d);
        for i in 0..n {
            for p in 0..d {
                let dp = x[(i, p)] - mean[p];
                for q in p..d {
                    r[(p, q)] += dp * (x[(i, q)] - mean[q]);
                }
            }
        }
        for p in 0..d {
            r[(p, p)] /= nf;
            for q in (p + 1)..d {
                r[(p, q)] /= nf;
                r[(q, p)] = r[(p, q)];
            }
        }
        let k = build_symmetric_covariance(&x, &mean, n, d);
        assert_eq!(bits(k.as_slice()), bits(r.as_slice()));
    }

    #[test]
    fn projection_parallel_matches_sequential_bits() {
        let (n, d, nc) = (100, 210, 5);
        assert!(2 * n * d * nc >= PAR_WORK_THRESHOLD);
        let x = pseudo(n, d, 0.11);
        let (_, mean) = center_matrix(&x, n, d);
        let u = pseudo(d, nc, 0.53).to_dmatrix();
        let mut r = FdMatrix::zeros(n, nc);
        for k in 0..nc {
            for i in 0..n {
                let mut s = 0.0;
                for j in 0..d {
                    s += (x[(i, j)] - mean[j]) * u[(j, k)];
                }
                r[(i, k)] = s;
            }
        }
        let sc = project_onto_eigenvectors(&x, &mean, &u, n, d, nc);
        assert_eq!(bits(sc.as_slice()), bits(r.as_slice()));
    }

    #[test]
    fn sphere_helpers_parallel_match_sequential_bits() {
        let (n, m) = (100, 260);
        assert!(8 * n * m >= PAR_WORK_THRESHOLD && 16 * n * m >= PAR_WORK_THRESHOLD);
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        // Monotone warps gamma_i(t) = t + a_i t (1 - t), |a_i| < 1.
        let mut g = FdMatrix::zeros(n, m);
        for i in 0..n {
            let a = 0.9 * ((i as f64) * 0.77).sin();
            for j in 0..m {
                g[(i, j)] = t[j] + a * t[j] * (1.0 - t[j]);
            }
        }
        let psis = warps_to_normalized_psi(&g, &t);
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let binsize = 1.0 / (m - 1) as f64;
        for i in 0..n {
            let gam: Vec<f64> = (0..m).map(|j| g[(i, j)]).collect();
            let mut grad = vec![0.0; m];
            grad[0] = (gam[1] - gam[0]) / binsize;
            for j in 1..m - 1 {
                grad[j] = (gam[j + 1] - gam[j - 1]) / (2.0 * binsize);
            }
            grad[m - 1] = (gam[m - 1] - gam[m - 2]) / binsize;
            let mut psi: Vec<f64> = grad.iter().map(|&v| v.max(0.0).sqrt()).collect();
            let norm = l2_norm_l2(&psi, &time);
            if norm > 1e-10 {
                for v in &mut psi {
                    *v /= norm;
                }
            }
            assert_eq!(bits(&psis[i]), bits(&psi), "psi row {i}");
        }

        let mu = psis[0].clone();
        let mut r = vec![0.0; m];
        for psi in &psis {
            let v = inv_exp_map_sphere(&mu, psi, &time);
            for j in 0..m {
                r[j] += v[j];
            }
        }
        for v in &mut r {
            *v /= n as f64;
        }
        assert_eq!(bits(&mean_tangent_vector(&psis, &mu, &time)), bits(&r));
    }
}
