//! Gaussian generative model for random curve synthesis from aligned data.

use super::srsf::{reparameterize_curve, srsf_inverse};
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

// ─── Shared helpers ─────────────────────────────────────────────────────────

/// Reconstruct a curve value vector from an augmented SRSF coefficient vector.
///
/// The augmented coordinate `q_new[m]` encodes the curve level at the domain
/// midpoint. Reconstruct from zero start and shift so the midpoint matches
/// (passing it as `srsf_inverse`'s `f0` would offset the whole curve — GH #34).
fn reconstruct_curve_from_aug_srsf(q_new: &[f64], argvals: &[f64], m: usize) -> Vec<f64> {
    let aug_val = q_new[m];
    let f_mid = aug_val.signum() * aug_val * aug_val;
    let mut f_new = srsf_inverse(&q_new[..m], argvals, 0.0);
    let shift = f_mid - f_new[m / 2];
    for val in f_new.iter_mut() {
        *val += shift;
    }
    f_new
}

/// Map a phase shooting vector to a warping function on the original domain.
///
/// Applies exp-map from `mu_psi`, converts psi -> gamma on [0,1], then rescales
/// to [t0, t1] and normalises boundary conditions.
fn phase_shooting_to_warp(
    v: &[f64],
    mu_psi: &[f64],
    time: &[f64],
    argvals: &[f64],
    t0: f64,
    domain: f64,
) -> Vec<f64> {
    let psi_new = exp_map_sphere(mu_psi, v, time);
    let gam_01 = psi_to_gam(&psi_new, time);
    let mut gamma: Vec<f64> = gam_01.iter().map(|&g| t0 + g * domain).collect();
    normalize_warp(&mut gamma, argvals);
    gamma
}

// ─── Gaussian Generative Model ──────────────────────────────────────────────

/// Validate common inputs for gauss_model and return (n, m).
fn gauss_validate(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
    n_samples: usize,
) -> Result<(usize, usize), FdarError> {
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
    Ok((n, m))
}

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
    let (_n, m) = gauss_validate(karcher, argvals, ncomp, n_samples)?;

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
        let mut rng = crate::helpers::seed_for_thread(seed, i);

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

        let f_new = reconstruct_curve_from_aug_srsf(&q_new, argvals, m);

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

        let gamma = phase_shooting_to_warp(&v, &mu_psi, &time, argvals, t0, domain);

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

// ─── Joint Gaussian Generative Model ────────────────────────────────────────

/// Build the joint score matrix [vert_scores | balance_c * horiz_scores].
fn build_joint_scores(
    vert_scores: &FdMatrix,
    horiz_scores: &FdMatrix,
    n: usize,
    vert_ncomp: usize,
    horiz_ncomp: usize,
    balance_c: f64,
) -> FdMatrix {
    let total_ncomp = vert_ncomp + horiz_ncomp;
    let mut joint = FdMatrix::zeros(n, total_ncomp);
    for i in 0..n {
        for k in 0..vert_ncomp {
            joint[(i, k)] = vert_scores[(i, k)];
        }
        for k in 0..horiz_ncomp {
            joint[(i, vert_ncomp + k)] = balance_c * horiz_scores[(i, k)];
        }
    }
    joint
}

/// Estimate column means and variances of a joint score matrix.
fn joint_score_mean_var(
    joint_scores: &FdMatrix,
    n: usize,
    total_ncomp: usize,
) -> (Vec<f64>, Vec<f64>) {
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
    (joint_mean, joint_var)
}

/// Sample one joint score vector from a diagonal Gaussian.
///
/// Consumes one `rng.sample(StandardNormal)` per component.
fn sample_joint_z(
    rng: &mut impl Rng,
    joint_mean: &[f64],
    joint_var: &[f64],
    total_ncomp: usize,
    scores_out: &mut FdMatrix,
    i: usize,
) -> Vec<f64> {
    let mut joint_z = vec![0.0; total_ncomp];
    for k in 0..total_ncomp {
        let z: f64 = rng.sample(StandardNormal);
        joint_z[k] = joint_mean[k] + z * joint_var[k].max(0.0).sqrt();
        scores_out[(i, k)] = joint_z[k];
    }
    joint_z
}

/// Reconstruct amplitude curve from joint_z scores and mean SRSF.
fn reconstruct_amplitude(
    joint_z: &[f64],
    mean_q: &[f64],
    vert_eigenfunctions_q: &FdMatrix,
    vert_ncomp: usize,
    m_aug: usize,
    argvals: &[f64],
    m: usize,
) -> Vec<f64> {
    let mut q_new = vec![0.0; m_aug];
    q_new[..m_aug].copy_from_slice(&mean_q[..m_aug]);
    for k in 0..vert_ncomp {
        let score_k = joint_z[k];
        for j in 0..m_aug {
            q_new[j] += score_k * vert_eigenfunctions_q[(k, j)];
        }
    }
    reconstruct_curve_from_aug_srsf(&q_new, argvals, m)
}

/// Reconstruct phase shooting vector from joint_z, undoing balance_c scaling.
fn reconstruct_phase_shooting(
    joint_z: &[f64],
    horiz_eigenfunctions_psi: &FdMatrix,
    vert_ncomp: usize,
    horiz_ncomp: usize,
    m: usize,
    balance_c: f64,
) -> Vec<f64> {
    let mut v = vec![0.0; m];
    for k in 0..horiz_ncomp {
        let score_k = if balance_c.abs() > 1e-15 {
            joint_z[vert_ncomp + k] / balance_c
        } else {
            0.0
        };
        for j in 0..m {
            v[j] += score_k * horiz_eigenfunctions_psi[(k, j)];
        }
    }
    v
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
    let (_n_ignored, m) = gauss_validate(karcher, argvals, ncomp, n_samples)?;

    // Amplitude FPCA
    let vert = vert_fpca(karcher, argvals, ncomp)?;
    let vert_ncomp = vert.eigenvalues.len();
    let m_aug = m + 1;

    // Phase FPCA
    let horiz = horiz_fpca(karcher, argvals, ncomp)?;
    let horiz_ncomp = horiz.eigenvalues.len();

    let total_ncomp = vert_ncomp + horiz_ncomp;
    let n = karcher.aligned_data.nrows();

    let joint_scores = build_joint_scores(
        &vert.scores,
        &horiz.scores,
        n,
        vert_ncomp,
        horiz_ncomp,
        balance_c,
    );
    let (joint_mean, joint_var) = joint_score_mean_var(&joint_scores, n, total_ncomp);

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
        let mut rng = crate::helpers::seed_for_thread(seed, i);

        let joint_z = sample_joint_z(
            &mut rng,
            &joint_mean,
            &joint_var,
            total_ncomp,
            &mut scores_out,
            i,
        );

        let f_new = reconstruct_amplitude(
            &joint_z,
            mean_q,
            &vert.eigenfunctions_q,
            vert_ncomp,
            m_aug,
            argvals,
            m,
        );

        let v = reconstruct_phase_shooting(
            &joint_z,
            &horiz.eigenfunctions_psi,
            vert_ncomp,
            horiz_ncomp,
            m,
            balance_c,
        );

        let gamma = phase_shooting_to_warp(&v, &mu_psi, &time, argvals, t0, domain);

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

    /// Regression test for GH #34: generated samples must reproduce the data
    /// mean, not sit a constant offset above it (previously ~+1 for curves whose
    /// midpoint level was ~1 because the midpoint-anchored augmented level was
    /// wrongly passed as `srsf_inverse`'s start value).
    #[test]
    fn gauss_model_sample_mean_no_constant_offset() {
        // Single-bump curves on a zero baseline (the issue's repro shape).
        let m = 60;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let n = 20;
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let amp = 0.8 + 0.02 * i as f64;
            for j in 0..m {
                let x = t[j] - 0.5;
                data[(i, j)] = amp * (-(x * x) / 0.02).exp();
            }
        }
        let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
        let result = gauss_model(&km, &t, 3, 300, 42).unwrap();
        let samples = &result.samples;
        let ns = samples.nrows();

        // Average pointwise offset between sample mean and data mean must be
        // near zero (the bug produced a uniform shift of ~+1.0).
        let mut offset = 0.0;
        for j in 0..m {
            let dm = (0..n).map(|i| data[(i, j)]).sum::<f64>() / n as f64;
            let sm = (0..ns).map(|i| samples[(i, j)]).sum::<f64>() / ns as f64;
            offset += sm - dm;
        }
        offset /= m as f64;
        assert!(
            offset.abs() < 0.3,
            "sample mean has a constant offset of {offset:.3} from the data mean (GH #34)"
        );
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
