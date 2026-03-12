//! Elastic changepoint detection for functional data streams.
//!
//! Implements the methods from Tucker & Yarger (2023) for detecting changepoints
//! in the amplitude, phase, or joint structure of functional data.
//!
//! Key capabilities:
//! - [`elastic_amp_changepoint`] — Amplitude changepoint via CUSUM on aligned curves
//! - [`elastic_ph_changepoint`] — Phase changepoint via CUSUM on shooting vectors
//! - [`elastic_fpca_changepoint`] — FPCA-based changepoint via Hotelling CUSUM

use crate::alignment::{karcher_mean, KarcherMeanResult};
use crate::elastic_fpca::{
    horiz_fpca, joint_fpca, shooting_vectors_from_psis, sphere_karcher_mean, vert_fpca,
    warps_to_normalized_psi,
};
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of changepoint detection.
#[derive(Debug, Clone)]
pub struct ChangepointResult {
    /// Estimated changepoint location (index in 1..n-1).
    pub changepoint: usize,
    /// Test statistic value.
    pub test_statistic: f64,
    /// Monte Carlo p-value.
    pub p_value: f64,
    /// CUSUM values for k=1..n-1.
    pub cusum_values: Vec<f64>,
}

/// Type of changepoint to detect.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChangepointType {
    /// Amplitude changepoint (on aligned curves).
    Amplitude,
    /// Phase changepoint (on warping functions).
    Phase,
    /// FPCA-based changepoint.
    Fpca(FpcaChangepointMethod),
}

/// FPCA method for changepoint detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FpcaChangepointMethod {
    Vertical,
    Horizontal,
    Joint,
}

/// Kernel for long-run covariance estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CovKernel {
    Bartlett,
    Parzen,
    FlatTop,
    Simple,
}

// ─── Amplitude Changepoint ──────────────────────────────────────────────────

/// Detect a changepoint in the amplitude of functional data.
///
/// 1. Karcher mean alignment
/// 2. CUSUM statistic on aligned curves
/// 3. k* = argmax(S_n), T_n = max(S_n)
/// 4. Monte Carlo p-value via permutation testing
///
/// # Arguments
/// * `data` — Functional data (n × m), ordered by time/index
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Alignment penalty
/// * `max_iter` — Karcher mean iterations
/// * `n_mc` — Number of Monte Carlo simulations for p-value
/// * `_cov_kernel` — Unused (kept for backward compatibility)
/// * `_cov_bandwidth` — Unused (kept for backward compatibility)
/// * `seed` — Random seed for reproducibility
pub fn elastic_amp_changepoint(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_iter: usize,
    n_mc: usize,
    _cov_kernel: CovKernel,
    _cov_bandwidth: Option<usize>,
    seed: u64,
) -> Result<ChangepointResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 4, m >= 2, argvals.len() == m".to_string(),
            actual: format!("n={}, m={}, argvals.len()={}", n, m, argvals.len()),
        });
    }

    // Alignment
    let km = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // CUSUM on aligned curves
    let weights = simpsons_weights(argvals);
    let cusum_values = functional_cusum(&km.aligned_data, &weights);

    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    // Monte Carlo p-value via permutation
    let p_value = mc_pvalue_permutation(test_statistic, &km.aligned_data, &weights, n_mc, seed);

    Ok(ChangepointResult {
        changepoint,
        test_statistic,
        p_value,
        cusum_values,
    })
}

// ─── Phase Changepoint ──────────────────────────────────────────────────────

/// Detect a changepoint in the phase (warping) of functional data.
///
/// Same as amplitude changepoint but on shooting vectors from the warping functions.
pub fn elastic_ph_changepoint(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_iter: usize,
    n_mc: usize,
    _cov_kernel: CovKernel,
    _cov_bandwidth: Option<usize>,
    seed: u64,
) -> Result<ChangepointResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 4, m >= 2, argvals.len() == m".to_string(),
            actual: format!("n={}, m={}, argvals.len()={}", n, m, argvals.len()),
        });
    }

    let km = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // Compute shooting vectors from warps (live on uniform [0,1] grid, not argvals)
    let shooting = compute_shooting_vectors(&km, argvals).ok_or_else(|| {
        crate::FdarError::ComputationFailed {
            operation: "compute_shooting_vectors",
            detail: "failed to compute shooting vectors from warps".to_string(),
        }
    })?;

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let weights = simpsons_weights(&time);
    let cusum_values = functional_cusum(&shooting, &weights);
    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    // Monte Carlo p-value via permutation
    let p_value = mc_pvalue_permutation(test_statistic, &shooting, &weights, n_mc, seed);

    Ok(ChangepointResult {
        changepoint,
        test_statistic,
        p_value,
        cusum_values,
    })
}

// ─── FPCA Changepoint ───────────────────────────────────────────────────────

/// Detect a changepoint using FPCA scores with Hotelling CUSUM.
///
/// Uses PC scores from vert/horiz/joint FPCA for a Hotelling-type test.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `argvals` — Evaluation points (length m)
/// * `pca_method` — Which FPCA variant to use
/// * `ncomp` — Number of principal components
/// * `lambda` — Alignment penalty
/// * `max_iter` — Karcher mean iterations
/// * `n_mc` — Monte Carlo simulations
/// * `seed` — Random seed
pub fn elastic_fpca_changepoint(
    data: &FdMatrix,
    argvals: &[f64],
    pca_method: FpcaChangepointMethod,
    ncomp: usize,
    lambda: f64,
    max_iter: usize,
    n_mc: usize,
    seed: u64,
) -> Result<ChangepointResult, crate::FdarError> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m || ncomp < 1 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data",
            expected: "n >= 4, m >= 2, argvals.len() == m, ncomp >= 1".to_string(),
            actual: format!(
                "n={}, m={}, argvals.len()={}, ncomp={}",
                n,
                m,
                argvals.len(),
                ncomp
            ),
        });
    }

    let km = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // Get PC scores
    let scores = match pca_method {
        FpcaChangepointMethod::Vertical => {
            let fpca = vert_fpca(&km, argvals, ncomp)?;
            fpca.scores
        }
        FpcaChangepointMethod::Horizontal => {
            let fpca = horiz_fpca(&km, argvals, ncomp)?;
            fpca.scores
        }
        FpcaChangepointMethod::Joint => {
            let fpca = joint_fpca(&km, argvals, ncomp, None)?;
            fpca.scores
        }
    };

    // Hotelling CUSUM on scores
    let cusum_values = hotelling_cusum(&scores);
    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    // Monte Carlo p-value via permutation
    let p_value = mc_pvalue_permutation_hotelling(test_statistic, &scores, n_mc, seed);

    Ok(ChangepointResult {
        changepoint,
        test_statistic,
        p_value,
        cusum_values,
    })
}

// ─── Internal Helpers ───────────────────────────────────────────────────────

/// Compute shooting vectors from warping functions.
fn compute_shooting_vectors(km: &KarcherMeanResult, argvals: &[f64]) -> Option<FdMatrix> {
    let (n, m) = km.gammas.shape();
    if n < 2 || m < 2 {
        return None;
    }
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let psis = warps_to_normalized_psi(&km.gammas, argvals);
    let mu_psi = sphere_karcher_mean(&psis, &time, 30);
    Some(shooting_vectors_from_psis(&psis, &mu_psi, &time))
}

/// Functional CUSUM: S_n(k) = (1/n²) Σ_j w_j * (cumsum_j - (k/n)*total_j)²
///
/// Uses Simpson's quadrature weights for spatial integration.
fn functional_cusum(data: &FdMatrix, weights: &[f64]) -> Vec<f64> {
    let (n, m) = data.shape();
    let mut cusum_values = Vec::with_capacity(n - 1);

    // Running sum
    let mut cumsum = vec![0.0; m];
    let mut total = vec![0.0; m];
    for i in 0..n {
        for j in 0..m {
            total[j] += data[(i, j)];
        }
    }

    for k in 1..n {
        for j in 0..m {
            cumsum[j] += data[(k - 1, j)];
        }

        // S_n(k) = (1/n²) Σ_j w_j * (cumsum_j - (k/n)*total_j)²
        let ratio = k as f64 / n as f64;
        let mut s = 0.0;
        for j in 0..m {
            let diff = cumsum[j] - ratio * total[j];
            s += weights[j] * diff * diff;
        }
        s /= (n * n) as f64;
        cusum_values.push(s);
    }

    cusum_values
}

/// Compute sample covariance matrix with regularization, returning its inverse.
fn regularized_cov_inverse(scores: &FdMatrix, mean: &[f64], n: usize, p: usize) -> DMatrix<f64> {
    let mut cov = DMatrix::zeros(p, p);
    for i in 0..n {
        for j1 in 0..p {
            for j2 in 0..p {
                cov[(j1, j2)] += (scores[(i, j1)] - mean[j1]) * (scores[(i, j2)] - mean[j2]);
            }
        }
    }
    cov /= (n - 1) as f64;
    for j in 0..p {
        cov[(j, j)] += 1e-6;
    }
    if let Some(chol) = cov.cholesky() {
        chol.inverse()
    } else {
        DMatrix::identity(p, p)
    }
}

/// Hotelling CUSUM on multivariate scores: S_n(k) = (1/n) Δ_k' Σ^{-1} Δ_k
fn hotelling_cusum(scores: &FdMatrix) -> Vec<f64> {
    let (n, p) = scores.shape();

    let mut mean = vec![0.0; p];
    for k in 0..p {
        for i in 0..n {
            mean[k] += scores[(i, k)];
        }
        mean[k] /= n as f64;
    }

    let cov_inv = regularized_cov_inverse(scores, &mean, n, p);

    let mut cumsum = vec![0.0; p];
    let mut total = vec![0.0; p];
    for i in 0..n {
        for k in 0..p {
            total[k] += scores[(i, k)];
        }
    }

    let mut cusum_values = Vec::with_capacity(n - 1);
    for k in 1..n {
        for j in 0..p {
            cumsum[j] += scores[(k - 1, j)];
        }
        let ratio = k as f64 / n as f64;
        let mut delta = nalgebra::DVector::zeros(p);
        for j in 0..p {
            delta[j] = (cumsum[j] - ratio * total[j]) / n as f64;
        }
        let hotelling = (&delta.transpose() * &cov_inv * &delta)[(0, 0)];
        cusum_values.push(hotelling);
    }

    cusum_values
}

/// Find index and value of maximum CUSUM.
fn find_max_cusum(cusum_values: &[f64]) -> (usize, f64) {
    let mut max_val = f64::NEG_INFINITY;
    let mut max_idx = 0;
    for (i, &v) in cusum_values.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i + 1; // 1-indexed changepoint
        }
    }
    (max_idx, max_val)
}

/// Monte Carlo p-value via permutation testing for functional CUSUM.
///
/// Shuffles the row order of `data`, recomputes `functional_cusum`, and compares
/// max(cusum) to the observed test statistic. Scale-invariant by construction.
fn mc_pvalue_permutation(
    test_stat: f64,
    data: &FdMatrix,
    weights: &[f64],
    n_mc: usize,
    seed: u64,
) -> f64 {
    let (n, m) = data.shape();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut count_exceed = 0usize;

    let mut shuffled = FdMatrix::zeros(n, m);
    for _ in 0..n_mc {
        indices.shuffle(&mut rng);
        for (new_i, &orig_i) in indices.iter().enumerate() {
            for j in 0..m {
                shuffled[(new_i, j)] = data[(orig_i, j)];
            }
        }
        let cusum = functional_cusum(&shuffled, weights);
        let (_, max_val) = find_max_cusum(&cusum);
        if max_val >= test_stat {
            count_exceed += 1;
        }
    }

    (count_exceed as f64 + 1.0) / (n_mc as f64 + 1.0)
}

/// Monte Carlo p-value via permutation testing for Hotelling CUSUM.
///
/// Shuffles row order of scores, recomputes `hotelling_cusum`, and compares.
fn mc_pvalue_permutation_hotelling(
    test_stat: f64,
    scores: &FdMatrix,
    n_mc: usize,
    seed: u64,
) -> f64 {
    let (n, p) = scores.shape();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n).collect();
    let mut count_exceed = 0usize;

    let mut shuffled = FdMatrix::zeros(n, p);
    for _ in 0..n_mc {
        indices.shuffle(&mut rng);
        for (new_i, &orig_i) in indices.iter().enumerate() {
            for j in 0..p {
                shuffled[(new_i, j)] = scores[(orig_i, j)];
            }
        }
        let cusum = hotelling_cusum(&shuffled);
        let (_, max_val) = find_max_cusum(&cusum);
        if max_val >= test_stat {
            count_exceed += 1;
        }
    }

    (count_exceed as f64 + 1.0) / (n_mc as f64 + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_changepoint_data(n: usize, m: usize, cp: usize) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);

        for i in 0..n {
            let amp = if i < cp { 1.0 } else { 2.0 };
            for j in 0..m {
                data[(i, j)] = amp * (2.0 * PI * t[j]).sin();
            }
        }
        (data, t)
    }

    #[test]
    fn test_amp_changepoint_detects_shift() {
        let n = 30;
        let m = 51;
        let cp = 15;
        let (data, t) = generate_changepoint_data(n, m, cp);

        let result = elastic_amp_changepoint(&data, &t, 0.0, 5, 199, CovKernel::Bartlett, None, 42);
        assert!(result.is_ok(), "amp changepoint should succeed");

        let res = result.unwrap();
        assert!(
            res.cusum_values.len() == n - 1,
            "Should have n-1 CUSUM values"
        );
        assert!(
            (res.changepoint as i64 - cp as i64).abs() <= 5,
            "Detected changepoint {} should be near true cp {}",
            res.changepoint,
            cp
        );
        assert!(
            res.p_value < 0.05,
            "Clear amplitude shift should be significant, got p={}",
            res.p_value
        );
    }

    #[test]
    fn test_ph_changepoint_basic() {
        let n = 30;
        let m = 51;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);

        for i in 0..n {
            let shift = if i < 15 { 0.0 } else { 0.15 };
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * (t[j] + shift)).sin();
            }
        }

        let result = elastic_ph_changepoint(&data, &t, 0.0, 5, 100, CovKernel::Bartlett, None, 42);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fpca_changepoint_basic() {
        let n = 30;
        let m = 51;
        let cp = 15;
        let (data, t) = generate_changepoint_data(n, m, cp);

        let result = elastic_fpca_changepoint(
            &data,
            &t,
            FpcaChangepointMethod::Vertical,
            3,
            0.0,
            5,
            100,
            42,
        );
        assert!(result.is_ok(), "fpca changepoint should succeed");
    }

    #[test]
    fn test_changepoint_no_change() {
        // All curves identical → weak test statistic, high p-value
        let n = 20;
        let m = 51;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }

        let result = elastic_amp_changepoint(&data, &t, 0.0, 5, 200, CovKernel::Bartlett, None, 42);
        if let Ok(res) = result {
            assert!(
                res.p_value > 0.1,
                "No change should not be significant, got p={}",
                res.p_value
            );
        }
    }

    #[test]
    fn test_pvalue_permutation_discriminates() {
        let m = 51;
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        // Strong signal: amplitude doubles at midpoint
        let (data_signal, _) = generate_changepoint_data(30, m, 15);
        let res_signal =
            elastic_amp_changepoint(&data_signal, &t, 0.0, 5, 199, CovKernel::Bartlett, None, 99)
                .expect("should succeed");
        assert!(
            res_signal.p_value < 0.05,
            "Strong signal should give small p, got {}",
            res_signal.p_value
        );

        // No signal: all curves identical
        let mut data_null = FdMatrix::zeros(30, m);
        for i in 0..30 {
            for j in 0..m {
                data_null[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }
        let res_null =
            elastic_amp_changepoint(&data_null, &t, 0.0, 5, 199, CovKernel::Bartlett, None, 99)
                .expect("should succeed");
        assert!(
            res_null.p_value > 0.1,
            "No signal should give large p, got {}",
            res_null.p_value
        );
    }

    #[test]
    fn test_invalid_input() {
        let data = FdMatrix::zeros(2, 5);
        let t: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
        // n=2 < 4 → should return None
        assert!(
            elastic_amp_changepoint(&data, &t, 0.0, 5, 100, CovKernel::Simple, None, 42).is_err()
        );
    }

    #[test]
    fn test_fpca_changepoint_horizontal() {
        let n = 30;
        let m = 51;
        let cp = 15;
        let (data, t) = generate_changepoint_data(n, m, cp);

        let result = elastic_fpca_changepoint(
            &data,
            &t,
            FpcaChangepointMethod::Horizontal,
            3,
            0.0,
            5,
            100,
            42,
        );
        assert!(
            result.is_ok(),
            "fpca changepoint (horizontal) should succeed"
        );

        let res = result.unwrap();
        assert_eq!(res.cusum_values.len(), n - 1);
    }

    #[test]
    fn test_fpca_changepoint_joint() {
        let n = 30;
        let m = 51;
        let cp = 15;
        let (data, t) = generate_changepoint_data(n, m, cp);

        let result =
            elastic_fpca_changepoint(&data, &t, FpcaChangepointMethod::Joint, 3, 0.0, 5, 100, 42);
        assert!(result.is_ok(), "fpca changepoint (joint) should succeed");

        let res = result.unwrap();
        assert_eq!(res.cusum_values.len(), n - 1);
    }

    #[test]
    fn test_changepoint_seed_determinism() {
        let n = 30;
        let m = 51;
        let cp = 15;
        let (data, t) = generate_changepoint_data(n, m, cp);

        let res1 = elastic_amp_changepoint(&data, &t, 0.0, 5, 199, CovKernel::Bartlett, None, 42)
            .expect("should succeed");
        let res2 = elastic_amp_changepoint(&data, &t, 0.0, 5, 199, CovKernel::Bartlett, None, 42)
            .expect("should succeed");

        assert_eq!(res1.changepoint, res2.changepoint);
        assert!((res1.p_value - res2.p_value).abs() < 1e-10);
    }
}
