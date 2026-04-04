//! Robust alternatives to the Karcher mean: median and trimmed mean.
//!
//! The standard Karcher mean is sensitive to outlier curves. This module
//! provides two robust alternatives:
//!
//! - [`karcher_median`] — Geometric median via iteratively reweighted
//!   Karcher mean (Weiszfeld algorithm on the elastic manifold).
//! - [`robust_karcher_mean`] — Trimmed Karcher mean that removes the
//!   most distant curves before averaging.

use super::karcher::karcher_mean;
use super::pairwise::elastic_distance;
use super::set::align_to_target;
use super::srsf::srsf_single;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Configuration for robust Karcher estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct RobustKarcherConfig {
    /// Maximum number of outer iterations.
    pub max_iter: usize,
    /// Convergence tolerance (relative change in SRSF).
    pub tol: f64,
    /// Roughness penalty for elastic alignment (0.0 = no penalty).
    pub lambda: f64,
    /// Fraction of most-distant curves to trim (for trimmed mean).
    pub trim_fraction: f64,
}

impl Default for RobustKarcherConfig {
    fn default() -> Self {
        Self {
            max_iter: 20,
            tol: 1e-3,
            lambda: 0.0,
            trim_fraction: 0.1,
        }
    }
}

/// Result of robust Karcher estimation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RobustKarcherResult {
    /// Robust mean/median curve.
    pub mean: Vec<f64>,
    /// SRSF of the robust mean/median.
    pub mean_srsf: Vec<f64>,
    /// Warping functions for all curves (n x m).
    pub gammas: FdMatrix,
    /// All curves aligned to the robust mean/median (n x m).
    pub aligned_data: FdMatrix,
    /// Per-curve weights (1/distance for median, 0/1 for trimmed).
    pub weights: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Compute the Karcher median via the Weiszfeld algorithm on the elastic manifold.
///
/// The geometric median minimizes the sum of elastic distances to all curves,
/// rather than the sum of squared distances (as with the mean). This makes it
/// robust to outlier curves.
///
/// # Algorithm
/// 1. Initialize with standard Karcher mean (1 iteration) as starting point.
/// 2. Iterative Weiszfeld loop:
///    a. Align all curves to the current median estimate.
///    b. Compute elastic distances.
///    c. Set weights w_i = 1 / max(d_i, epsilon), normalize.
///    d. Compute weighted pointwise mean of aligned curves.
///    e. Check convergence (relative change in SRSF).
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `config`  — Configuration parameters.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_median(
    data: &FdMatrix,
    argvals: &[f64],
    config: &RobustKarcherConfig,
) -> Result<RobustKarcherResult, FdarError> {
    let (n, m) = data.shape();
    validate_inputs(n, m, argvals)?;

    // Step 1: Initialize with a quick Karcher mean (1 iteration).
    let init = karcher_mean(data, argvals, 1, config.tol, config.lambda);
    let mut current_mean = init.mean;

    let mut converged = false;
    let mut n_iter = 0;
    let mut weights = vec![1.0 / n as f64; n];
    let mut alignment_result = align_to_target(data, &current_mean, argvals, config.lambda);

    // Step 2: Weiszfeld iterations.
    for iter in 0..config.max_iter {
        n_iter = iter + 1;

        // Compute elastic distances.
        let distances: Vec<f64> = (0..n)
            .map(|i| {
                let fi = data.row(i);
                elastic_distance(&current_mean, &fi, argvals, config.lambda)
            })
            .collect();

        // Compute weights: w_i = 1 / max(d_i, epsilon).
        let epsilon = 1e-10;
        let raw_weights: Vec<f64> = distances.iter().map(|&d| 1.0 / d.max(epsilon)).collect();
        let w_sum: f64 = raw_weights.iter().sum();
        weights = raw_weights.iter().map(|&w| w / w_sum).collect();

        // Weighted pointwise mean of aligned curves.
        let mut new_mean = vec![0.0; m];
        for i in 0..n {
            for j in 0..m {
                new_mean[j] += weights[i] * alignment_result.aligned_data[(i, j)];
            }
        }

        // Check convergence.
        let old_srsf = srsf_single(&current_mean, argvals);
        let new_srsf = srsf_single(&new_mean, argvals);
        let rel = relative_srsf_change(&old_srsf, &new_srsf);

        current_mean = new_mean;

        if rel < config.tol {
            converged = true;
            // Final alignment to converged median.
            alignment_result = align_to_target(data, &current_mean, argvals, config.lambda);
            break;
        }

        // Re-align to updated median.
        alignment_result = align_to_target(data, &current_mean, argvals, config.lambda);
    }

    let mean_srsf = srsf_single(&current_mean, argvals);

    Ok(RobustKarcherResult {
        mean: current_mean,
        mean_srsf,
        gammas: alignment_result.gammas,
        aligned_data: alignment_result.aligned_data,
        weights,
        n_iter,
        converged,
    })
}

/// Compute a trimmed Karcher mean by removing the most distant curves.
///
/// Computes the standard Karcher mean, identifies and removes the top
/// `trim_fraction` of curves by elastic distance, then recomputes the
/// Karcher mean on the remaining curves. All curves (including trimmed
/// ones) are re-aligned to the robust mean for the final output.
///
/// # Arguments
/// * `data`    — Functional data matrix (n x m).
/// * `argvals` — Evaluation points (length m).
/// * `config`  — Configuration parameters.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`.
/// Returns [`FdarError::InvalidParameter`] if `trim_fraction` is not in \[0, 1).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn robust_karcher_mean(
    data: &FdMatrix,
    argvals: &[f64],
    config: &RobustKarcherConfig,
) -> Result<RobustKarcherResult, FdarError> {
    let (n, m) = data.shape();
    validate_inputs(n, m, argvals)?;

    if !(0.0..1.0).contains(&config.trim_fraction) {
        return Err(FdarError::InvalidParameter {
            parameter: "trim_fraction",
            message: format!("must be in [0, 1), got {}", config.trim_fraction),
        });
    }

    // Step 1: Compute standard Karcher mean.
    let initial_mean = karcher_mean(data, argvals, config.max_iter, config.tol, config.lambda);

    // Step 2: Compute elastic distances from the mean.
    let distances: Vec<f64> = (0..n)
        .map(|i| {
            let fi = data.row(i);
            elastic_distance(&initial_mean.mean, &fi, argvals, config.lambda)
        })
        .collect();

    // Step 3: Sort by distance, identify curves to trim.
    let n_trim = ((n as f64) * config.trim_fraction).ceil() as usize;
    let n_keep = n.saturating_sub(n_trim).max(2); // Keep at least 2 curves.

    let mut indexed_distances: Vec<(usize, f64)> =
        distances.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let kept_indices: Vec<usize> = indexed_distances
        .iter()
        .take(n_keep)
        .map(|&(i, _)| i)
        .collect();

    // Step 4: Set weights.
    let mut weights = vec![0.0; n];
    for &idx in &kept_indices {
        weights[idx] = 1.0;
    }

    // Step 5: Recompute Karcher mean on the kept subset.
    let kept_data = subset_rows_from_indices(data, &kept_indices);
    let robust_mean = karcher_mean(
        &kept_data,
        argvals,
        config.max_iter,
        config.tol,
        config.lambda,
    );

    // Step 6: Re-align ALL curves (including trimmed) to the robust mean.
    let final_alignment = align_to_target(data, &robust_mean.mean, argvals, config.lambda);

    let mean_srsf = srsf_single(&robust_mean.mean, argvals);

    Ok(RobustKarcherResult {
        mean: robust_mean.mean,
        mean_srsf,
        gammas: final_alignment.gammas,
        aligned_data: final_alignment.aligned_data,
        weights,
        n_iter: robust_mean.n_iter,
        converged: robust_mean.converged,
    })
}

/// Validate common input dimensions.
fn validate_inputs(n: usize, m: usize, argvals: &[f64]) -> Result<(), FdarError> {
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    Ok(())
}

/// Compute relative change between successive SRSFs.
fn relative_srsf_change(q_old: &[f64], q_new: &[f64]) -> f64 {
    let diff_norm: f64 = q_old
        .iter()
        .zip(q_new.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let old_norm: f64 = q_old.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
    diff_norm / old_norm
}

use crate::cv::subset_rows as subset_rows_from_indices;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn make_sine_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let mut data_vec = vec![0.0; n * m];
        for i in 0..n {
            let phase = 0.03 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        (data, t)
    }

    #[test]
    fn karcher_median_basic() {
        let (data, t) = make_sine_data(5, 20);
        let config = RobustKarcherConfig {
            max_iter: 5,
            ..Default::default()
        };
        let result = karcher_median(&data, &t, &config).unwrap();
        assert_eq!(result.mean.len(), 20);
        assert_eq!(result.mean_srsf.len(), 20);
        assert_eq!(result.gammas.shape(), (5, 20));
        assert_eq!(result.aligned_data.shape(), (5, 20));
        assert_eq!(result.weights.len(), 5);
        assert!(result.n_iter >= 1);
    }

    #[test]
    fn karcher_median_robust_to_outlier() {
        let m = 20;
        let t = uniform_grid(m);
        let n = 6;
        let mut data_vec = vec![0.0; n * m];

        // 5 clean curves (slight phase shifts).
        for i in 0..5 {
            let phase = 0.02 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        // 1 extreme outlier.
        for j in 0..m {
            data_vec[5 + j * n] = (t[j] * 20.0).cos() * 5.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

        // Compute standard mean and median.
        let std_mean = karcher_mean(&data, &t, 5, 1e-3, 0.0);
        let median_config = RobustKarcherConfig {
            max_iter: 5,
            ..Default::default()
        };
        let median_result = karcher_median(&data, &t, &median_config).unwrap();

        // Compute a clean reference (mean of just the clean curves).
        let clean_data = subset_rows_from_indices(&data, &[0, 1, 2, 3, 4]);
        let clean_mean = karcher_mean(&clean_data, &t, 5, 1e-3, 0.0);

        // Median should be closer to the clean mean than the standard mean is.
        let d_std = pointwise_l2(&std_mean.mean, &clean_mean.mean);
        let d_median = pointwise_l2(&median_result.mean, &clean_mean.mean);
        assert!(
            d_median <= d_std + 1e-6,
            "median distance to clean ({d_median:.4}) should be <= standard mean distance ({d_std:.4})"
        );
    }

    #[test]
    fn robust_trimmed_removes_outliers() {
        let m = 20;
        let t = uniform_grid(m);
        let n = 6;
        let mut data_vec = vec![0.0; n * m];

        // 5 clean curves.
        for i in 0..5 {
            let phase = 0.02 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        // 1 extreme outlier.
        for j in 0..m {
            data_vec[5 + j * n] = (t[j] * 20.0).cos() * 5.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

        let config = RobustKarcherConfig {
            max_iter: 5,
            trim_fraction: 0.2, // Trim top 20% (= 2 curves out of 6).
            ..Default::default()
        };
        let result = robust_karcher_mean(&data, &t, &config).unwrap();

        // The outlier (index 5) should have weight 0.0.
        assert!(
            result.weights[5] < 1e-10,
            "outlier weight should be 0, got {}",
            result.weights[5]
        );

        // At least some curves should have weight 1.0.
        let n_kept: usize = result.weights.iter().filter(|&&w| w > 0.5).count();
        assert!(n_kept >= 4, "should keep at least 4 curves, got {n_kept}");
    }

    #[test]
    fn robust_config_default() {
        let cfg = RobustKarcherConfig::default();
        assert_eq!(cfg.max_iter, 20);
        assert!((cfg.tol - 1e-3).abs() < f64::EPSILON);
        assert!((cfg.lambda - 0.0).abs() < f64::EPSILON);
        assert!((cfg.trim_fraction - 0.1).abs() < f64::EPSILON);
    }

    /// Simple pointwise L2 distance between two curves.
    fn pointwise_l2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}
