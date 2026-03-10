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

use crate::matrix::FdMatrix;
use crate::warping::{exp_map_sphere, inv_exp_map_sphere, l2_norm_l2, psi_to_gam};
use nalgebra::SVD;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of vertical (amplitude) FPCA.
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
#[derive(Debug, Clone)]
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
) -> Option<VertFpcaResult> {
    let (n, m) = karcher.aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return None;
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
    let u_cov = svd.u.as_ref()?;

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

    Some(VertFpcaResult {
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
) -> Option<HorizFpcaResult> {
    let (n, m) = karcher.gammas.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return None;
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
    let v_t = svd.v_t.as_ref()?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n)?;
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

    Some(HorizFpcaResult {
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
/// * `balance_c` — Weight for phase component (if None, optimized via golden section)
pub fn joint_fpca(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
) -> Option<JointFpcaResult> {
    let (n, m) = karcher.aligned_data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return None;
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
    let (q_centered, _mean_q) = center_matrix(&q_aug, n, m_aug);

    let shooting = &horiz.shooting_vectors;
    let c = match balance_c {
        Some(c) => c,
        None => optimize_balance_c(karcher, argvals, &q_centered, shooting, ncomp),
    };

    // Concatenate: g_i = [qn_aug_centered_i; C * v_i]
    let combined = build_combined_representation(&q_centered, shooting, c, n, m_aug, m);

    let svd = SVD::new(combined.to_dmatrix(), true, true);
    let v_t = svd.v_t.as_ref()?;
    let (scores, eigenvalues) = svd_scores_and_eigenvalues(&svd, ncomp, n)?;
    let cumulative_variance = cumulative_variance_from_eigenvalues(&eigenvalues);

    // Split eigenvectors into amplitude and phase parts
    let (vert_component, horiz_component) = split_joint_eigenvectors(v_t, ncomp, m_aug, m);

    Some(JointFpcaResult {
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

    (0..n)
        .map(|i| {
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
        .collect()
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
    let mut shooting = FdMatrix::zeros(n, m);
    for i in 0..n {
        let v = inv_exp_map_sphere(mu_psi, &psis[i], time);
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
    let mut q_aug = FdMatrix::zeros(n, m_aug);
    for i in 0..n {
        for j in 0..m {
            q_aug[(i, j)] = qn[(i, j)];
        }
        let fid = aligned_data[(i, id)];
        q_aug[(i, m)] = fid.signum() * fid.abs().sqrt();
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
    for k in 0..ncomp {
        let sv = svd.singular_values[k];
        for i in 0..n {
            scores[(i, k)] = u[(i, k)] * sv;
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
    let mut k_mat = nalgebra::DMatrix::zeros(d, d);
    for i in 0..n {
        for p in 0..d {
            let dp = data[(i, p)] - mean[p];
            for q in p..d {
                k_mat[(p, q)] += dp * (data[(i, q)] - mean[q]);
            }
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
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..d {
                s += (data[(i, j)] - mean[j]) * u_cov[(j, k)];
            }
            scores[(i, k)] = s;
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
    let mut mean_v = vec![0.0; m];
    for psi in psis {
        let v = inv_exp_map_sphere(mu_psi, psi, time);
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
fn build_combined_representation(
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

/// Optimize the balance parameter C via golden section search.
///
/// Minimizes reconstruction error of the joint representation.
fn optimize_balance_c(
    _karcher: &KarcherMeanResult,
    _argvals: &[f64],
    q_centered: &FdMatrix,
    shooting: &FdMatrix,
    ncomp: usize,
) -> f64 {
    let (n, m) = shooting.shape();
    let m_aug = q_centered.ncols();
    let combined_dim = m_aug + m;

    let golden_ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut a = 0.0_f64;
    let mut b = 10.0_f64;

    let eval_c = |c: f64| -> f64 {
        let mut combined = FdMatrix::zeros(n, combined_dim);
        for i in 0..n {
            for j in 0..m_aug {
                combined[(i, j)] = q_centered[(i, j)];
            }
            for j in 0..m {
                combined[(i, m_aug + j)] = c * shooting[(i, j)];
            }
        }

        let svd = SVD::new(combined.to_dmatrix(), true, true);
        if let (Some(_u), Some(_v_t)) = (svd.u.as_ref(), svd.v_t.as_ref()) {
            let nc = ncomp.min(svd.singular_values.len());
            // Reconstruction error = total variance - explained variance
            let total_var: f64 = svd.singular_values.iter().map(|&s| s * s).sum();
            let explained: f64 = svd.singular_values.iter().take(nc).map(|&s| s * s).sum();
            total_var - explained
        } else {
            f64::INFINITY
        }
    };

    for _ in 0..20 {
        let c1 = b - golden_ratio * (b - a);
        let c2 = a + golden_ratio * (b - a);
        if eval_c(c1) < eval_c(c2) {
            b = c2;
        } else {
            a = c1;
        }
    }

    (a + b) / 2.0
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
        assert!(result.is_some(), "vert_fpca should succeed");

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
        assert!(result.is_some(), "horiz_fpca should succeed");

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
        assert!(result.is_some(), "joint_fpca should succeed");

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
            result.is_some(),
            "joint_fpca with C optimization should succeed"
        );
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
        assert!(vert_fpca(&km, &t, 3).is_none());
    }
}
