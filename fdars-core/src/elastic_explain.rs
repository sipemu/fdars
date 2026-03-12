//! Elastic shape explainability: amplitude vs phase attribution.
//!
//! Decomposes elastic PCR predictions into "shape" (amplitude) and "timing" (phase)
//! contributions, with permutation-based variable importance.
//!
//! - [`elastic_pcr_attribution`] — Decompose predictions and compute importance scores

use crate::alignment::srsf_transform;
use crate::elastic_fpca::{
    build_augmented_srsfs, center_matrix, shooting_vectors_from_psis, sphere_karcher_mean,
    warps_to_normalized_psi,
};
use crate::elastic_regression::{ElasticPcrResult, PcaMethod};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use rand::prelude::*;

/// Result of elastic amplitude/phase attribution.
#[derive(Debug, Clone)]
pub struct ElasticAttributionResult {
    /// Per-observation amplitude contribution (length n).
    pub amplitude_contribution: Vec<f64>,
    /// Per-observation phase contribution (length n).
    pub phase_contribution: Vec<f64>,
    /// R² drop from permuting amplitude scores.
    pub amplitude_importance: f64,
    /// R² drop from permuting phase scores.
    pub phase_importance: f64,
}

/// Decompose elastic PCR predictions into amplitude and phase contributions.
///
/// For joint FPCA, the joint eigenvectors split into vertical (amplitude) and
/// horizontal (phase) parts. Each observation's score can be decomposed into
/// amplitude and phase sub-scores based on these parts.
///
/// For vertical-only or horizontal-only models, the missing component
/// contributes zero.
///
/// # Arguments
/// * `result` — A fitted [`ElasticPcrResult`] (must have stored FPCA results)
/// * `y` — Original scalar responses (length n)
/// * `ncomp` — Number of components to use for attribution
/// * `n_perm` — Number of permutation replicates for importance
/// * `seed` — RNG seed for permutation reproducibility
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `y.len()` does not match the
/// number of fitted values in `result`.
/// Returns [`FdarError::InvalidParameter`] if `ncomp` is zero or there are
/// fewer than 2 observations.
/// Returns [`FdarError::ComputationFailed`] if the joint FPCA result is
/// missing from a `PcaMethod::Joint` model.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pcr_attribution(
    result: &ElasticPcrResult,
    y: &[f64],
    ncomp: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ElasticAttributionResult, FdarError> {
    let n = result.fitted_values.len();
    if y.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "y",
            expected: n.to_string(),
            actual: y.len().to_string(),
        });
    }
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "ncomp must be >= 1".into(),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n",
            message: "need at least 2 observations".into(),
        });
    }
    let actual_ncomp = ncomp.min(result.coefficients.len());

    match result.pca_method {
        PcaMethod::Joint => attribution_joint(result, y, actual_ncomp, n_perm, seed),
        PcaMethod::Vertical => {
            // All contribution is amplitude, phase is zero
            let amp: Vec<f64> = result
                .fitted_values
                .iter()
                .map(|&f| f - result.alpha)
                .collect();
            let phase = vec![0.0; n];
            let amp_imp = permutation_importance_single(
                y,
                &result.fitted_values,
                result.alpha,
                &result.coefficients,
                actual_ncomp,
                n_perm,
                seed,
            );
            Ok(ElasticAttributionResult {
                amplitude_contribution: amp,
                phase_contribution: phase,
                amplitude_importance: amp_imp,
                phase_importance: 0.0,
            })
        }
        PcaMethod::Horizontal => {
            // All contribution is phase, amplitude is zero
            let phase: Vec<f64> = result
                .fitted_values
                .iter()
                .map(|&f| f - result.alpha)
                .collect();
            let amp = vec![0.0; n];
            let phase_imp = permutation_importance_single(
                y,
                &result.fitted_values,
                result.alpha,
                &result.coefficients,
                actual_ncomp,
                n_perm,
                seed,
            );
            Ok(ElasticAttributionResult {
                amplitude_contribution: amp,
                phase_contribution: phase,
                amplitude_importance: 0.0,
                phase_importance: phase_imp,
            })
        }
    }
}

/// Joint FPCA attribution: decompose scores into amp and phase parts.
fn attribution_joint(
    result: &ElasticPcrResult,
    y: &[f64],
    ncomp: usize,
    n_perm: usize,
    seed: u64,
) -> Result<ElasticAttributionResult, FdarError> {
    let joint = result
        .joint_fpca
        .as_ref()
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "elastic_pcr_attribution",
            detail: "joint_fpca result missing from ElasticPcrResult".into(),
        })?;
    let km = &result.karcher;
    let (n, m) = km.aligned_data.shape();
    let m_aug = m + 1;

    let qn = match &km.aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => {
            let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
            srsf_transform(&km.aligned_data, &argvals)
        }
    };

    let q_aug = build_augmented_srsfs(&qn, &km.aligned_data, n, m);
    let (_, mean_q) = center_matrix(&q_aug, n, m_aug);

    // Compute shooting vectors using shared helpers
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let psis = warps_to_normalized_psi(&km.gammas, &argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 50);
    let shooting = shooting_vectors_from_psis(&psis, &mu_psi, &time);

    let c = joint.balance_c;
    let (amp_scores, phase_scores) = decompose_joint_scores(
        &q_aug,
        &mean_q,
        &shooting,
        &joint.vert_component,
        &joint.horiz_component,
        c,
        n,
        m_aug,
        m,
        ncomp,
    );

    let (amplitude_contribution, phase_contribution) =
        compute_contributions(&amp_scores, &phase_scores, &result.coefficients, n, ncomp);

    // Permutation importance
    let r2_orig = compute_r2(y, &result.fitted_values);
    let amplitude_importance = permutation_importance(
        y,
        result.alpha,
        &result.coefficients,
        &amp_scores,
        &phase_scores,
        ncomp,
        n_perm,
        seed,
        true,
    );
    let phase_importance = permutation_importance(
        y,
        result.alpha,
        &result.coefficients,
        &amp_scores,
        &phase_scores,
        ncomp,
        n_perm,
        seed + 1_000_000,
        false,
    );

    Ok(ElasticAttributionResult {
        amplitude_contribution,
        phase_contribution,
        amplitude_importance: (r2_orig - amplitude_importance).max(0.0),
        phase_importance: (r2_orig - phase_importance).max(0.0),
    })
}

/// Decompose joint scores into amplitude and phase sub-scores.
fn decompose_joint_scores(
    q_aug: &FdMatrix,
    mean_q: &[f64],
    shooting: &FdMatrix,
    vert_component: &FdMatrix,
    horiz_component: &FdMatrix,
    c: f64,
    n: usize,
    m_aug: usize,
    m: usize,
    ncomp: usize,
) -> (FdMatrix, FdMatrix) {
    let mut amp_scores = FdMatrix::zeros(n, ncomp);
    let mut phase_scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        for i in 0..n {
            let mut amp_s = 0.0;
            for j in 0..m_aug {
                amp_s += (q_aug[(i, j)] - mean_q[j]) * vert_component[(k, j)];
            }
            amp_scores[(i, k)] = amp_s;

            let mut phase_s = 0.0;
            for j in 0..m {
                phase_s += c * shooting[(i, j)] * horiz_component[(k, j)];
            }
            phase_scores[(i, k)] = phase_s;
        }
    }
    (amp_scores, phase_scores)
}

/// Compute amplitude and phase contributions from decomposed scores.
fn compute_contributions(
    amp_scores: &FdMatrix,
    phase_scores: &FdMatrix,
    coefficients: &[f64],
    n: usize,
    ncomp: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut amplitude_contribution = vec![0.0; n];
    let mut phase_contribution = vec![0.0; n];
    for i in 0..n {
        for k in 0..ncomp {
            amplitude_contribution[i] += coefficients[k] * amp_scores[(i, k)];
            phase_contribution[i] += coefficients[k] * phase_scores[(i, k)];
        }
    }
    (amplitude_contribution, phase_contribution)
}

/// Compute R² statistic.
fn compute_r2(y: &[f64], fitted: &[f64]) -> f64 {
    let n = y.len();
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = y
        .iter()
        .zip(fitted)
        .map(|(&yi, &fi)| (yi - fi).powi(2))
        .sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}

/// Permutation importance: shuffle one component's scores, recompute fitted, return avg R².
fn permutation_importance(
    y: &[f64],
    alpha: f64,
    coefficients: &[f64],
    amp_scores: &FdMatrix,
    phase_scores: &FdMatrix,
    ncomp: usize,
    n_perm: usize,
    seed: u64,
    permute_amplitude: bool,
) -> f64 {
    let n = y.len();
    if n_perm == 0 {
        return compute_r2(y, &vec![alpha; n]);
    }

    let mut total_r2 = 0.0;
    for p in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(p as u64));
        let mut perm_idx: Vec<usize> = (0..n).collect();
        perm_idx.shuffle(&mut rng);

        let fitted = fitted_with_permuted_scores(
            alpha,
            coefficients,
            amp_scores,
            phase_scores,
            &perm_idx,
            n,
            ncomp,
            permute_amplitude,
        );
        total_r2 += compute_r2(y, &fitted);
    }
    total_r2 / n_perm as f64
}

/// Compute fitted values with one component's scores permuted.
fn fitted_with_permuted_scores(
    alpha: f64,
    coefficients: &[f64],
    amp_scores: &FdMatrix,
    phase_scores: &FdMatrix,
    perm_idx: &[usize],
    n: usize,
    ncomp: usize,
    permute_amplitude: bool,
) -> Vec<f64> {
    let mut fitted = vec![0.0; n];
    for i in 0..n {
        fitted[i] = alpha;
        for k in 0..ncomp {
            let amp_i = if permute_amplitude {
                amp_scores[(perm_idx[i], k)]
            } else {
                amp_scores[(i, k)]
            };
            let phase_i = if !permute_amplitude {
                phase_scores[(perm_idx[i], k)]
            } else {
                phase_scores[(i, k)]
            };
            fitted[i] += coefficients[k] * (amp_i + phase_i);
        }
    }
    fitted
}

/// Permutation importance for single-component models (vert-only or horiz-only).
fn permutation_importance_single(
    y: &[f64],
    fitted_values: &[f64],
    alpha: f64,
    _coefficients: &[f64],
    _ncomp: usize,
    n_perm: usize,
    seed: u64,
) -> f64 {
    let n = y.len();
    let r2_orig = compute_r2(y, fitted_values);
    if n_perm == 0 {
        return r2_orig;
    }

    // Extract per-obs contribution = fitted - alpha, then permute
    let contribs: Vec<f64> = fitted_values.iter().map(|&f| f - alpha).collect();
    let mut total_r2 = 0.0;
    for p in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(p as u64));
        let mut perm_idx: Vec<usize> = (0..n).collect();
        perm_idx.shuffle(&mut rng);

        let fitted_perm: Vec<f64> = (0..n).map(|i| alpha + contribs[perm_idx[i]]).collect();
        total_r2 += compute_r2(y, &fitted_perm);
    }
    let avg_r2 = total_r2 / n_perm as f64;
    (r2_orig - avg_r2).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elastic_regression::{elastic_pcr, PcaMethod};
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0; n];
        for i in 0..n {
            let amp = 1.0 + 0.5 * (i as f64 / n as f64);
            let shift = 0.1 * (i as f64 - n as f64 / 2.0);
            for j in 0..m {
                data[(i, j)] = amp * (2.0 * PI * (t[j] + shift)).sin();
            }
            y[i] = amp;
        }
        (data, y, t)
    }

    #[test]
    fn test_elastic_attribution_joint_decomposition() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
        let attr = elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

        assert_eq!(attr.amplitude_contribution.len(), 15);
        assert_eq!(attr.phase_contribution.len(), 15);

        // Verify: amp + phase ≈ fitted - alpha
        for i in 0..15 {
            let sum = attr.amplitude_contribution[i] + attr.phase_contribution[i];
            let expected = result.fitted_values[i] - result.alpha;
            assert!(
                (sum - expected).abs() < 1e-6,
                "amp + phase should ≈ fitted - alpha at i={}: {} vs {}",
                i,
                sum,
                expected
            );
        }
    }

    #[test]
    fn test_elastic_attribution_vertical_only() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Vertical, 0.0, 5, 1e-3).unwrap();
        let attr = elastic_pcr_attribution(&result, &y, 3, 10, 42).unwrap();

        // Phase contribution should all be zero
        for i in 0..15 {
            assert!(
                attr.phase_contribution[i].abs() < 1e-12,
                "phase_contribution should be 0 for vertical-only at i={}",
                i
            );
        }
        assert!(
            attr.phase_importance.abs() < 1e-12,
            "phase_importance should be 0 for vertical-only"
        );
    }

    #[test]
    fn test_elastic_attribution_importance_nonnegative() {
        let (data, y, t) = generate_test_data(15, 51);
        let result = elastic_pcr(&data, &y, &t, 3, PcaMethod::Joint, 0.0, 5, 1e-3).unwrap();
        let attr = elastic_pcr_attribution(&result, &y, 3, 20, 42).unwrap();

        assert!(
            attr.amplitude_importance >= 0.0,
            "amplitude_importance should be >= 0, got {}",
            attr.amplitude_importance
        );
        assert!(
            attr.phase_importance >= 0.0,
            "phase_importance should be >= 0, got {}",
            attr.phase_importance
        );
    }
}
