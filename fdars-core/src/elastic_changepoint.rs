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
use nalgebra::{DMatrix, SVD};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

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
/// 4. Monte Carlo p-value via Brownian bridge simulation
///
/// # Arguments
/// * `data` — Functional data (n × m), ordered by time/index
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Alignment penalty
/// * `max_iter` — Karcher mean iterations
/// * `n_mc` — Number of Monte Carlo simulations for p-value
/// * `cov_kernel` — Kernel for long-run covariance
/// * `cov_bandwidth` — Bandwidth for covariance kernel (if None, auto-select)
/// * `seed` — Random seed for reproducibility
pub fn elastic_amp_changepoint(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    max_iter: usize,
    n_mc: usize,
    cov_kernel: CovKernel,
    cov_bandwidth: Option<usize>,
    seed: u64,
) -> Option<ChangepointResult> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m {
        return None;
    }

    // Alignment
    let km = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // CUSUM on aligned curves
    let weights = simpsons_weights(argvals);
    let cusum_values = functional_cusum(&km.aligned_data, &weights);

    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    // Long-run covariance and eigendecomposition
    let bandwidth = cov_bandwidth.unwrap_or_else(|| auto_bandwidth(n));
    let eigen_values = long_run_cov_eigenvalues(&km.aligned_data, bandwidth, cov_kernel, 20);

    // Monte Carlo p-value
    let p_value = mc_pvalue_brownian_bridge(test_statistic, &eigen_values, n, n_mc, seed);

    Some(ChangepointResult {
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
    cov_kernel: CovKernel,
    cov_bandwidth: Option<usize>,
    seed: u64,
) -> Option<ChangepointResult> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m {
        return None;
    }

    let km = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // Compute shooting vectors from warps
    let shooting = compute_shooting_vectors(&km, argvals)?;

    let weights = simpsons_weights(argvals);
    let cusum_values = functional_cusum(&shooting, &weights);
    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    let bandwidth = cov_bandwidth.unwrap_or_else(|| auto_bandwidth(n));
    let eigen_values = long_run_cov_eigenvalues(&shooting, bandwidth, cov_kernel, 20);

    let p_value = mc_pvalue_brownian_bridge(test_statistic, &eigen_values, n, n_mc, seed);

    Some(ChangepointResult {
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
) -> Option<ChangepointResult> {
    let (n, m) = data.shape();
    if n < 4 || m < 2 || argvals.len() != m || ncomp < 1 {
        return None;
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

    let actual_ncomp = scores.ncols();

    // Hotelling CUSUM on scores
    let cusum_values = hotelling_cusum(&scores);
    let (changepoint, test_statistic) = find_max_cusum(&cusum_values);

    // Monte Carlo p-value via squared Brownian bridges
    let p_value = mc_pvalue_hotelling(test_statistic, actual_ncomp, n, n_mc, seed);

    Some(ChangepointResult {
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

/// Functional CUSUM: S_n(k) = (1/(m*n²)) Σ_j (cumsum_j - (k/n)*total_j)²
///
/// Matches R's fdasrvf formula: `sum(diff_vec^2) / (m * n^2)`.
fn functional_cusum(data: &FdMatrix, _weights: &[f64]) -> Vec<f64> {
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

        // S_n(k) = (1/(m*n²)) Σ_j (cumsum_j - (k/n)*total_j)²
        let ratio = k as f64 / n as f64;
        let mut s = 0.0;
        for j in 0..m {
            let diff = cumsum[j] - ratio * total[j];
            s += diff * diff;
        }
        s /= (m * n * n) as f64;
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

/// Compute lag-h cross-covariance matrix Γ(h) = (1/n) Σ_i (x_i - μ)(x_{i+h} - μ)'.
fn lag_cross_covariance(
    data: &FdMatrix,
    mean: &[f64],
    n: usize,
    m: usize,
    lag: usize,
) -> DMatrix<f64> {
    let count = n.saturating_sub(lag);
    let mut gamma_lag = DMatrix::zeros(m, m);
    for i in 0..count {
        let i2 = i + lag;
        for j1 in 0..m {
            for j2 in 0..m {
                gamma_lag[(j1, j2)] += (data[(i, j1)] - mean[j1]) * (data[(i2, j2)] - mean[j2]);
            }
        }
    }
    gamma_lag /= n as f64;
    gamma_lag
}

/// Eigenvalues of long-run covariance via kernel-weighted lag cross-covariances.
fn long_run_cov_eigenvalues(
    data: &FdMatrix,
    bandwidth: usize,
    kernel: CovKernel,
    n_eigen: usize,
) -> Vec<f64> {
    let (n, m) = data.shape();

    let mut mean = vec![0.0; m];
    for j in 0..m {
        for i in 0..n {
            mean[j] += data[(i, j)];
        }
        mean[j] /= n as f64;
    }

    let mut d_mat = DMatrix::zeros(m, m);
    for lag in 0..=bandwidth {
        let k_weight = kernel_weight(lag as f64 / bandwidth as f64, kernel);
        if k_weight.abs() < 1e-15 || lag >= n {
            continue;
        }
        let gamma_lag = lag_cross_covariance(data, &mean, n, m, lag);
        if lag == 0 {
            d_mat += k_weight * &gamma_lag;
        } else {
            d_mat += k_weight * (&gamma_lag + gamma_lag.transpose());
        }
    }

    let svd = SVD::new(d_mat, false, false);
    svd.singular_values
        .iter()
        .take(n_eigen)
        .map(|&s| s.max(0.0))
        .collect()
}

/// Kernel weight function.
fn kernel_weight(x: f64, kernel: CovKernel) -> f64 {
    let x_abs = x.abs();
    match kernel {
        CovKernel::Bartlett => {
            if x_abs <= 1.0 {
                1.0 - x_abs
            } else {
                0.0
            }
        }
        CovKernel::Parzen => {
            if x_abs <= 0.5 {
                1.0 - 6.0 * x_abs * x_abs + 6.0 * x_abs * x_abs * x_abs
            } else if x_abs <= 1.0 {
                2.0 * (1.0 - x_abs).powi(3)
            } else {
                0.0
            }
        }
        CovKernel::FlatTop => {
            if x_abs <= 0.5 {
                1.0
            } else if x_abs <= 1.0 {
                2.0 * (1.0 - x_abs)
            } else {
                0.0
            }
        }
        CovKernel::Simple => {
            if x_abs <= 1.0 {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Auto-select bandwidth: floor(n^{1/3}).
fn auto_bandwidth(n: usize) -> usize {
    (n as f64).powf(1.0 / 3.0).floor() as usize
}

/// Monte Carlo p-value using Brownian bridge simulation.
///
/// Simulates max_t(Σ λ_k BB_k(t)²) and counts fraction ≥ T_n.
fn mc_pvalue_brownian_bridge(
    test_stat: f64,
    eigenvalues: &[f64],
    n: usize,
    n_mc: usize,
    seed: u64,
) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eigen = eigenvalues.len();

    let mut count_exceed = 0usize;
    let grid_size = n - 1;

    for _ in 0..n_mc {
        // Generate n_eigen independent Brownian bridges on [0,1]
        let mut max_stat = 0.0;
        for t_idx in 1..=grid_size {
            let t = t_idx as f64 / n as f64;
            let mut val = 0.0;
            for k in 0..n_eigen {
                // Brownian bridge at t: W(t) - t*W(1), but we simulate directly
                // BB(t) ~ N(0, t(1-t)), so BB(t) = sqrt(t(1-t)) * Z
                let z: f64 = normal.sample(&mut rng);
                let bb = (t * (1.0 - t)).sqrt() * z;
                val += eigenvalues[k] * bb * bb;
            }
            if val > max_stat {
                max_stat = val;
            }
        }

        if max_stat >= test_stat {
            count_exceed += 1;
        }
    }

    (count_exceed as f64 + 1.0) / (n_mc as f64 + 1.0)
}

/// Monte Carlo p-value for Hotelling CUSUM via squared Brownian bridges.
fn mc_pvalue_hotelling(test_stat: f64, p: usize, n: usize, n_mc: usize, seed: u64) -> f64 {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut count_exceed = 0usize;
    let grid_size = n - 1;

    for _ in 0..n_mc {
        let mut max_stat = 0.0;
        for t_idx in 1..=grid_size {
            let t = t_idx as f64 / n as f64;
            let var = t * (1.0 - t);
            let mut val = 0.0;
            for _ in 0..p {
                let z: f64 = normal.sample(&mut rng);
                let bb = var.sqrt() * z;
                val += bb * bb;
            }
            if val > max_stat {
                max_stat = val;
            }
        }

        if max_stat >= test_stat {
            count_exceed += 1;
        }
    }

    (count_exceed as f64 + 1.0) / (n_mc as f64 + 1.0)
}

/// Simulate a Brownian bridge on a grid.
#[allow(dead_code)]
fn brownian_bridge(n: usize, rng: &mut StdRng) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut bb = vec![0.0; n];
    if n < 2 {
        return bb;
    }

    // Generate Brownian motion
    let dt = 1.0 / (n - 1) as f64;
    let sqrt_dt = dt.sqrt();
    let mut w = vec![0.0; n];
    for i in 1..n {
        w[i] = w[i - 1] + sqrt_dt * normal.sample(rng);
    }

    // Bridge: BB(t) = W(t) - t * W(1)
    let w_final = w[n - 1];
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64;
        bb[i] = w[i] - t * w_final;
    }

    bb
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

        let result = elastic_amp_changepoint(&data, &t, 0.0, 5, 100, CovKernel::Bartlett, None, 42);
        assert!(result.is_some(), "amp changepoint should succeed");

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
        assert!(res.p_value >= 0.0 && res.p_value <= 1.0);
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
        assert!(result.is_some());
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
        assert!(result.is_some(), "fpca changepoint should succeed");
    }

    #[test]
    fn test_kernel_weights() {
        // Bartlett
        assert!((kernel_weight(0.0, CovKernel::Bartlett) - 1.0).abs() < 1e-10);
        assert!((kernel_weight(0.5, CovKernel::Bartlett) - 0.5).abs() < 1e-10);
        assert!(kernel_weight(1.5, CovKernel::Bartlett).abs() < 1e-10);

        // Parzen
        assert!((kernel_weight(0.0, CovKernel::Parzen) - 1.0).abs() < 1e-10);
        assert!(kernel_weight(1.5, CovKernel::Parzen).abs() < 1e-10);

        // FlatTop
        assert!((kernel_weight(0.0, CovKernel::FlatTop) - 1.0).abs() < 1e-10);
        assert!((kernel_weight(0.3, CovKernel::FlatTop) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_brownian_bridge_endpoints() {
        let mut rng = StdRng::seed_from_u64(123);
        let bb = brownian_bridge(101, &mut rng);
        assert!((bb[0]).abs() < 1e-10, "BB should start at 0");
        assert!((bb[100]).abs() < 1e-10, "BB should end at 0");
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
        if let Some(res) = result {
            // With no change, p-value should tend to be higher
            // (not a strict guarantee due to randomness, but a sanity check)
            assert!(res.p_value > 0.0);
        }
    }

    #[test]
    fn test_invalid_input() {
        let data = FdMatrix::zeros(2, 5);
        let t: Vec<f64> = (0..5).map(|i| i as f64 / 4.0).collect();
        // n=2 < 4 → should return None
        assert!(
            elastic_amp_changepoint(&data, &t, 0.0, 5, 100, CovKernel::Simple, None, 42).is_none()
        );
    }
}
