//! Outlier detection for functional data.
//!
//! This module provides methods for detecting outliers in functional data
//! based on depth measures and likelihood ratio tests.

use crate::depth::fraiman_muniz_1d;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;

/// Compute FM depth for internal use.
fn compute_fm_depth_internal(data: &[f64], n: usize, m: usize) -> Vec<f64> {
    fraiman_muniz_1d(data, data, n, n, m, true)
}

/// Compute bootstrap threshold for LRT outlier detection.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `nb` - Number of bootstrap iterations
/// * `smo` - Smoothing parameter for bootstrap
/// * `trim` - Trimming proportion
/// * `seed` - Random seed
/// * `percentile` - Percentile for threshold (e.g., 0.99 for 99th percentile)
///
/// # Returns
/// Threshold at specified percentile for outlier detection
pub fn outliers_threshold_lrt(
    data: &[f64],
    n: usize,
    m: usize,
    nb: usize,
    smo: f64,
    trim: f64,
    seed: u64,
    percentile: f64,
) -> f64 {
    if n < 3 || m == 0 || data.len() != n * m {
        return 0.0;
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil() as usize;

    // Compute column standard deviations for smoothing
    let col_vars: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n {
                let val = data[i + j * n];
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            var.max(0.0).sqrt()
        })
        .collect();

    // Run bootstrap iterations in parallel
    let max_dists: Vec<f64> = (0..nb)
        .into_par_iter()
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed + b as u64);

            // Resample with replacement
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();

            // Build resampled data with smoothing noise
            let mut boot_data = vec![0.0; n * m];
            for (new_i, &old_i) in indices.iter().enumerate() {
                for j in 0..m {
                    let noise: f64 = rng.sample::<f64, _>(StandardNormal) * smo * col_vars[j];
                    boot_data[new_i + j * n] = data[old_i + j * n] + noise;
                }
            }

            // Compute FM depth for bootstrap sample
            let depths = compute_fm_depth_internal(&boot_data, n, m);

            // Get indices of top n_keep curves by depth
            let mut depth_idx: Vec<(usize, f64)> =
                depths.iter().enumerate().map(|(i, &d)| (i, d)).collect();
            depth_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let keep_idx: Vec<usize> = depth_idx.iter().take(n_keep).map(|(i, _)| *i).collect();

            // Compute trimmed mean and variance
            let mut trimmed_mean = vec![0.0; m];
            for j in 0..m {
                for &i in &keep_idx {
                    trimmed_mean[j] += boot_data[i + j * n];
                }
                trimmed_mean[j] /= n_keep as f64;
            }

            let mut trimmed_var = vec![0.0; m];
            for j in 0..m {
                for &i in &keep_idx {
                    let diff = boot_data[i + j * n] - trimmed_mean[j];
                    trimmed_var[j] += diff * diff;
                }
                trimmed_var[j] /= n_keep as f64;
                trimmed_var[j] = trimmed_var[j].max(1e-10);
            }

            // Compute max normalized distance to trimmed mean
            let mut max_dist = 0.0;
            for i in 0..n {
                let mut dist = 0.0;
                for j in 0..m {
                    let diff = boot_data[i + j * n] - trimmed_mean[j];
                    dist += diff * diff / trimmed_var[j];
                }
                dist = (dist / m as f64).sqrt();
                if dist > max_dist {
                    max_dist = dist;
                }
            }

            max_dist
        })
        .collect();

    // Return threshold at specified percentile
    let mut sorted_dists = max_dists;
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((nb as f64 * percentile) as usize).min(nb.saturating_sub(1));
    sorted_dists.get(idx).copied().unwrap_or(0.0)
}

/// Detect outliers using LRT method.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `threshold` - Outlier threshold
/// * `trim` - Trimming proportion
///
/// # Returns
/// Vector of booleans indicating outliers
pub fn detect_outliers_lrt(
    data: &[f64],
    n: usize,
    m: usize,
    threshold: f64,
    trim: f64,
) -> Vec<bool> {
    if n < 3 || m == 0 || data.len() != n * m {
        return vec![false; n];
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil() as usize;

    // Compute FM depth
    let depths = compute_fm_depth_internal(data, n, m);

    // Get indices of top n_keep curves by depth
    let mut depth_idx: Vec<(usize, f64)> = depths.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    depth_idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let keep_idx: Vec<usize> = depth_idx.iter().take(n_keep).map(|(i, _)| *i).collect();

    // Compute trimmed mean and variance
    let mut trimmed_mean = vec![0.0; m];
    for j in 0..m {
        for &i in &keep_idx {
            trimmed_mean[j] += data[i + j * n];
        }
        trimmed_mean[j] /= n_keep as f64;
    }

    let mut trimmed_var = vec![0.0; m];
    for j in 0..m {
        for &i in &keep_idx {
            let diff = data[i + j * n] - trimmed_mean[j];
            trimmed_var[j] += diff * diff;
        }
        trimmed_var[j] /= n_keep as f64;
        trimmed_var[j] = trimmed_var[j].max(1e-10);
    }

    // Compute normalized distance for each observation
    (0..n)
        .into_par_iter()
        .map(|i| {
            let mut dist = 0.0;
            for j in 0..m {
                let diff = data[i + j * n] - trimmed_mean[j];
                dist += diff * diff / trimmed_var[j];
            }
            dist = (dist / m as f64).sqrt();
            dist > threshold
        })
        .collect()
}
