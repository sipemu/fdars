//! Gaussian generative model for random curve synthesis from aligned data.

use super::srsf::{reparameterize_curve, srsf_inverse, srsf_single};
use super::KarcherMeanResult;
use crate::elastic_fpca::{horiz_fpca, sphere_karcher_mean, vert_fpca, warps_to_normalized_psi};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::warping::{exp_map_sphere, normalize_warp, psi_to_gam};

use rand::prelude::*;
use rand_distr::StandardNormal;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of Gaussian generative model sampling.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GenerativeModelResult {
    /// Generated function samples (n_samples x m).
    pub samples: FdMatrix,
    /// Generated warping functions (n_samples x m).
    pub warps: FdMatrix,
    /// FPCA scores used for generation (n_samples x ncomp).
    pub scores: FdMatrix,
}

// ─── Gaussian Generative Model ──────────────────────────────────────────────

/// Generate random curves from a fitted Gaussian model on aligned data.
///
/// Samples amplitude and phase components independently from their
/// respective FPCA score distributions (Gaussian with covariance = diag(eigenvalues)),
/// then combines them to produce synthetic functional data.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result (with aligned data and gammas)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components for both amplitude and phase
/// * `n_samples` — Number of curves to generate
/// * `seed` — RNG seed for reproducibility
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if dimensions are inconsistent or
/// `FdarError::ComputationFailed` if FPCA fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gauss_model(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    n_samples: usize,
    seed: u64,
) -> Result<GenerativeModelResult, FdarError> {
    let (n, m) = karcher.aligned_data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if n < 2 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_data",
            expected: "n >= 2, m >= 2".to_string(),
            actual: format!("n={n}, m={m}"),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
    }
    if n_samples < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be >= 1".to_string(),
        });
    }

    // Amplitude FPCA
    let vert = vert_fpca(karcher, argvals, ncomp)?;
    let vert_ncomp = vert.eigenvalues.len();
    let m_aug = m + 1;

    // Phase FPCA
    let horiz = horiz_fpca(karcher, argvals, ncomp)?;
    let horiz_ncomp = horiz.eigenvalues.len();

    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Get the mean psi on the sphere for phase generation
    let psis = warps_to_normalized_psi(&karcher.gammas, argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 50);

    // Mean SRSF (augmented)
    let mean_q = &vert.mean_q;

    let total_ncomp = vert_ncomp + horiz_ncomp;
    let mut samples = FdMatrix::zeros(n_samples, m);
    let mut warps = FdMatrix::zeros(n_samples, m);
    let mut scores = FdMatrix::zeros(n_samples, total_ncomp);

    for i in 0..n_samples {
        let mut rng = StdRng::seed_from_u64(seed + i as u64);

        // Generate amplitude scores and reconstruct SRSF
        let mut q_new = vec![0.0; m_aug];
        q_new[..m_aug].copy_from_slice(&mean_q[..m_aug]);
        for k in 0..vert_ncomp {
            let std_dev = vert.eigenvalues[k].max(0.0).sqrt();
            let z: f64 = rng.sample(StandardNormal);
            let score_k = z * std_dev;
            scores[(i, k)] = score_k;
            for j in 0..m_aug {
                q_new[j] += score_k * vert.eigenfunctions_q[(k, j)];
            }
        }

        // Reconstruct curve from SRSF
        let aug_val = q_new[m];
        let f0 = aug_val.signum() * aug_val * aug_val;
        let f_new = srsf_inverse(&q_new[..m], argvals, f0);

        // Generate phase scores and reconstruct warping function
        let mut v = vec![0.0; m];
        for k in 0..horiz_ncomp {
            let std_dev = horiz.eigenvalues[k].max(0.0).sqrt();
            let z: f64 = rng.sample(StandardNormal);
            let score_k = z * std_dev;
            scores[(i, vert_ncomp + k)] = score_k;
            for j in 0..m {
                v[j] += score_k * horiz.eigenfunctions_psi[(k, j)];
            }
        }

        // Map shooting vector to sphere via exp map at mean psi
        let psi_new = exp_map_sphere(&mu_psi, &v, &time);
        let gam_01 = psi_to_gam(&psi_new, &time);

        // Rescale gamma to original domain
        let mut gamma: Vec<f64> = gam_01.iter().map(|&g| t0 + g * domain).collect();
        normalize_warp(&mut gamma, argvals);

        // Apply warp to generate final sample
        let sample = reparameterize_curve(&f_new, argvals, &gamma);

        for j in 0..m {
            samples[(i, j)] = sample[j];
            warps[(i, j)] = gamma[j];
        }
    }

    Ok(GenerativeModelResult {
        samples,
        warps,
        scores,
    })
}

/// Generate random curves from a joint Gaussian model preserving amplitude-phase
/// correlation.
///
/// Computes amplitude and phase FPCA separately, concatenates their scores to
/// form a joint score vector, estimates the joint covariance, and samples from
/// the joint distribution. This preserves cross-correlation between amplitude
/// and phase variability.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components per domain (amplitude and phase)
/// * `n_samples` — Number of curves to generate
/// * `balance_c` — Weight for balancing phase vs amplitude variance
/// * `seed` — RNG seed for reproducibility
///
/// # Errors
/// Returns `FdarError` on dimension mismatch or FPCA failure.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn joint_gauss_model(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    n_samples: usize,
    balance_c: f64,
    seed: u64,
) -> Result<GenerativeModelResult, FdarError> {
    let (_n, m) = karcher.aligned_data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if ncomp < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".to_string(),
        });
    }
    if n_samples < 1 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_samples",
            message: "n_samples must be >= 1".to_string(),
        });
    }

    // Amplitude FPCA
    let vert = vert_fpca(karcher, argvals, ncomp)?;
    let vert_ncomp = vert.eigenvalues.len();
    let m_aug = m + 1;

    // Phase FPCA
    let horiz = horiz_fpca(karcher, argvals, ncomp)?;
    let horiz_ncomp = horiz.eigenvalues.len();

    let total_ncomp = vert_ncomp + horiz_ncomp;
    let n = karcher.aligned_data.nrows();

    // Build joint score matrix: [vert_scores | balance_c * horiz_scores]
    let mut joint_scores = FdMatrix::zeros(n, total_ncomp);
    for i in 0..n {
        for k in 0..vert_ncomp {
            joint_scores[(i, k)] = vert.scores[(i, k)];
        }
        for k in 0..horiz_ncomp {
            joint_scores[(i, vert_ncomp + k)] = balance_c * horiz.scores[(i, k)];
        }
    }

    // Estimate joint covariance (diagonal for sampling)
    let mut joint_mean = vec![0.0; total_ncomp];
    for k in 0..total_ncomp {
        for i in 0..n {
            joint_mean[k] += joint_scores[(i, k)];
        }
        joint_mean[k] /= n as f64;
    }

    let mut joint_var = vec![0.0; total_ncomp];
    for k in 0..total_ncomp {
        for i in 0..n {
            let diff = joint_scores[(i, k)] - joint_mean[k];
            joint_var[k] += diff * diff;
        }
        joint_var[k] /= (n - 1).max(1) as f64;
    }

    // Sphere/warping setup
    let t0 = argvals[0];
    let t1 = argvals[m - 1];
    let domain = t1 - t0;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    let psis = warps_to_normalized_psi(&karcher.gammas, argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 50);
    let mean_q = &vert.mean_q;

    let mut samples = FdMatrix::zeros(n_samples, m);
    let mut warps_out = FdMatrix::zeros(n_samples, m);
    let mut scores_out = FdMatrix::zeros(n_samples, total_ncomp);

    for i in 0..n_samples {
        let mut rng = StdRng::seed_from_u64(seed + i as u64);

        // Sample from joint distribution
        let mut joint_z = vec![0.0; total_ncomp];
        for k in 0..total_ncomp {
            let z: f64 = rng.sample(StandardNormal);
            joint_z[k] = joint_mean[k] + z * joint_var[k].max(0.0).sqrt();
            scores_out[(i, k)] = joint_z[k];
        }

        // Reconstruct amplitude from SRSF
        let mut q_new = vec![0.0; m_aug];
        q_new[..m_aug].copy_from_slice(&mean_q[..m_aug]);
        for k in 0..vert_ncomp {
            let score_k = joint_z[k];
            for j in 0..m_aug {
                q_new[j] += score_k * vert.eigenfunctions_q[(k, j)];
            }
        }
        let aug_val = q_new[m];
        let f0 = aug_val.signum() * aug_val * aug_val;
        let f_new = srsf_inverse(&q_new[..m], argvals, f0);

        // Reconstruct phase from shooting vector
        let mut v = vec![0.0; m];
        for k in 0..horiz_ncomp {
            // Undo balance_c scaling
            let score_k = if balance_c.abs() > 1e-15 {
                joint_z[vert_ncomp + k] / balance_c
            } else {
                0.0
            };
            for j in 0..m {
                v[j] += score_k * horiz.eigenfunctions_psi[(k, j)];
            }
        }

        let psi_new = exp_map_sphere(&mu_psi, &v, &time);
        let gam_01 = psi_to_gam(&psi_new, &time);
        let mut gamma: Vec<f64> = gam_01.iter().map(|&g| t0 + g * domain).collect();
        normalize_warp(&mut gamma, argvals);

        let sample = reparameterize_curve(&f_new, argvals, &gamma);
        for j in 0..m {
            samples[(i, j)] = sample[j];
            warps_out[(i, j)] = gamma[j];
        }
    }

    Ok(GenerativeModelResult {
        samples,
        warps: warps_out,
        scores: scores_out,
    })
}

// ─── Helper ─────────────────────────────────────────────────────────────────

/// Compute SRSF of a single curve (delegates to crate-internal helper).
#[allow(dead_code)]
fn _srsf_single_wrapper(f: &[f64], argvals: &[f64]) -> Vec<f64> {
    srsf_single(f, argvals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::karcher_mean;
    use std::f64::consts::PI;

    fn make_test_karcher(n: usize, m: usize) -> (KarcherMeanResult, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let shift = 0.1 * (i as f64 - n as f64 / 2.0);
            let scale = 1.0 + 0.2 * (i as f64 / n as f64);
            for j in 0..m {
                data[(i, j)] = scale * (2.0 * PI * (t[j] + shift)).sin();
            }
        }
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        (km, t)
    }

    #[test]
    fn gauss_model_correct_shapes() {
        let (km, t) = make_test_karcher(15, 51);
        let ncomp = 3;
        let n_samples = 10;
        let result = gauss_model(&km, &t, ncomp, n_samples, 42).unwrap();

        assert_eq!(result.samples.shape(), (n_samples, 51));
        assert_eq!(result.warps.shape(), (n_samples, 51));
        // scores is n_samples x (vert_ncomp + horiz_ncomp)
        let (_, score_cols) = result.scores.shape();
        assert!(
            score_cols >= ncomp,
            "scores should have at least ncomp columns, got {score_cols}"
        );
        assert_eq!(result.scores.nrows(), n_samples);
    }

    #[test]
    fn gauss_model_reproducible() {
        let (km, t) = make_test_karcher(15, 51);
        let r1 = gauss_model(&km, &t, 3, 5, 42).unwrap();
        let r2 = gauss_model(&km, &t, 3, 5, 42).unwrap();

        assert_eq!(r1.samples, r2.samples);
        assert_eq!(r1.warps, r2.warps);
        assert_eq!(r1.scores, r2.scores);
    }

    #[test]
    fn gauss_model_warps_valid() {
        let (km, t) = make_test_karcher(15, 51);
        let result = gauss_model(&km, &t, 3, 10, 99).unwrap();
        let m = t.len();

        for i in 0..result.warps.nrows() {
            let warp = result.warps.row(i);

            // Monotone non-decreasing
            for j in 1..m {
                assert!(
                    warp[j] >= warp[j - 1] - 1e-12,
                    "warp {i} not monotone at j={j}: {} < {}",
                    warp[j],
                    warp[j - 1]
                );
            }

            // Correct boundary values
            assert!(
                (warp[0] - t[0]).abs() < 1e-10,
                "warp {i} start: {} != {}",
                warp[0],
                t[0]
            );
            assert!(
                (warp[m - 1] - t[m - 1]).abs() < 1e-10,
                "warp {i} end: {} != {}",
                warp[m - 1],
                t[m - 1]
            );
        }
    }

    #[test]
    fn joint_gauss_model_smoke() {
        let (km, t) = make_test_karcher(15, 51);
        let ncomp = 3;
        let n_samples = 8;
        let result = joint_gauss_model(&km, &t, ncomp, n_samples, 1.0, 42).unwrap();

        assert_eq!(result.samples.shape(), (n_samples, 51));
        assert_eq!(result.warps.shape(), (n_samples, 51));
        assert_eq!(result.scores.nrows(), n_samples);

        // All samples should be finite
        for i in 0..n_samples {
            for j in 0..51 {
                assert!(
                    result.samples[(i, j)].is_finite(),
                    "sample ({i},{j}) is not finite"
                );
            }
        }
    }
}
