//! Outlier detection for functional data.
//!
//! This module provides methods for detecting outliers in functional data
//! based on depth measures and likelihood ratio tests.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz};
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute trimmed mean and variance from data using depth-based trimming.
///
/// Returns (trimmed_mean, trimmed_var) each of length m.
fn compute_trimmed_stats(data: &FdMatrix, depths: &[f64], n_keep: usize) -> (Vec<f64>, Vec<f64>) {
    let m = data.ncols();

    let mut depth_idx: Vec<(usize, f64)> =
        depths.iter().enumerate().map(|(i, &d)| (i, d)).collect();
    // O(n) partial sort instead of O(n log n) full sort — we only need the top n_keep elements
    if n_keep < depth_idx.len() {
        depth_idx.select_nth_unstable_by(n_keep - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let keep_idx: Vec<usize> = depth_idx[..n_keep].iter().map(|(i, _)| *i).collect();

    let mut trimmed_mean = vec![0.0; m];
    for j in 0..m {
        for &i in &keep_idx {
            trimmed_mean[j] += data[(i, j)];
        }
        trimmed_mean[j] /= n_keep as f64;
    }

    let mut trimmed_var = vec![0.0; m];
    for j in 0..m {
        for &i in &keep_idx {
            let diff = data[(i, j)] - trimmed_mean[j];
            trimmed_var[j] += diff * diff;
        }
        trimmed_var[j] /= n_keep as f64;
        trimmed_var[j] = trimmed_var[j].max(1e-10);
    }

    (trimmed_mean, trimmed_var)
}

/// Compute normalized Mahalanobis-like distance for a single observation.
fn normalized_distance(
    data: &FdMatrix,
    i: usize,
    trimmed_mean: &[f64],
    trimmed_var: &[f64],
) -> f64 {
    let m = data.ncols();
    let mut dist = 0.0;
    for j in 0..m {
        let diff = data[(i, j)] - trimmed_mean[j];
        dist += diff * diff / trimmed_var[j];
    }
    (dist / m as f64).sqrt()
}

/// Compute bootstrap threshold for LRT outlier detection.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `nb` - Number of bootstrap iterations
/// * `smo` - Smoothing parameter for bootstrap
/// * `trim` - Trimming proportion
/// * `seed` - Random seed
/// * `percentile` - Percentile for threshold (e.g., 0.99 for 99th percentile)
///
/// # Returns
/// Threshold at specified percentile for outlier detection
pub fn outliers_threshold_lrt(
    data: &FdMatrix,
    nb: usize,
    smo: f64,
    trim: f64,
    seed: u64,
    percentile: f64,
) -> f64 {
    outliers_threshold_lrt_with_dist(data, nb, smo, trim, seed, percentile).0
}

/// Compute bootstrap threshold and full null distribution for LRT outlier detection.
///
/// Same as [`outliers_threshold_lrt`] but also returns the sorted bootstrap
/// distribution of max-distances, enabling per-curve p-value computation:
/// `p = (sum(boot_dist >= d) + 1) / (B + 1)`.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `nb` - Number of bootstrap iterations
/// * `smo` - Smoothing parameter for bootstrap
/// * `trim` - Trimming proportion
/// * `seed` - Random seed
/// * `percentile` - Percentile for threshold (e.g., 0.99 for 99th percentile)
///
/// # Returns
/// `(threshold, sorted_distribution)` — threshold at specified percentile and the
/// full sorted bootstrap null distribution of max-distances (length `nb`).
pub fn outliers_threshold_lrt_with_dist(
    data: &FdMatrix,
    nb: usize,
    smo: f64,
    trim: f64,
    seed: u64,
    percentile: f64,
) -> (f64, Vec<f64>) {
    let n = data.nrows();
    let m = data.ncols();

    if n < 3 || m == 0 {
        return (0.0, vec![]);
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
    let n_keep = n_keep.min(n);

    // Compute column standard deviations for smoothing
    let col_vars: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n {
                let val = data[(i, j)];
                sum += val;
                sum_sq += val * val;
            }
            let mean = sum / n as f64;
            let var = sum_sq / n as f64 - mean * mean;
            var.max(0.0).sqrt()
        })
        .collect();

    // Run bootstrap iterations in parallel
    let max_dists: Vec<f64> = iter_maybe_parallel!(0..nb)
        .map(|b| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(b as u64));

            // Resample with replacement and add smoothing noise
            let indices: Vec<usize> = (0..n).map(|_| rng.gen_range(0..n)).collect();
            // Pre-generate all noise values to preserve RNG sequence
            let noise_vals: Vec<f64> = (0..n * m)
                .map(|_| rng.sample::<f64, _>(StandardNormal))
                .collect();
            let mut boot_data = FdMatrix::zeros(n, m);
            // Column-first iteration for cache-friendly writes in column-major layout
            for j in 0..m {
                let smo_var = smo * col_vars[j];
                for (new_i, &old_i) in indices.iter().enumerate() {
                    let noise = noise_vals[new_i * m + j] * smo_var;
                    boot_data[(new_i, j)] = data[(old_i, j)] + noise;
                }
            }

            // Compute trimmed stats from bootstrap sample
            let state = SortedReferenceState::from_reference(&boot_data);
            let streaming_fm = StreamingFraimanMuniz::new(state, true);
            let depths = streaming_fm.depth_batch(&boot_data);
            let (trimmed_mean, trimmed_var) = compute_trimmed_stats(&boot_data, &depths, n_keep);

            // Find max normalized distance across all observations
            (0..n)
                .map(|i| normalized_distance(&boot_data, i, &trimmed_mean, &trimmed_var))
                .fold(0.0_f64, f64::max)
        })
        .collect();

    // Sort and extract threshold at specified percentile
    let mut sorted_dists = max_dists;
    sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((nb as f64 * percentile) as usize).min(nb.saturating_sub(1));
    let threshold = sorted_dists.get(idx).copied().unwrap_or(0.0);
    (threshold, sorted_dists)
}

/// Detect outliers using LRT method.
///
/// # Arguments
/// * `data` - Functional data matrix (n observations x m evaluation points)
/// * `threshold` - Outlier threshold
/// * `trim` - Trimming proportion
///
/// # Returns
/// Vector of booleans indicating outliers
pub fn detect_outliers_lrt(data: &FdMatrix, threshold: f64, trim: f64) -> Vec<bool> {
    let n = data.nrows();
    let m = data.ncols();

    if n < 3 || m == 0 {
        return vec![false; n];
    }

    let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
    let n_keep = n_keep.min(n);

    let state = SortedReferenceState::from_reference(data);
    let streaming_fm = StreamingFraimanMuniz::new(state, true);
    let depths = streaming_fm.depth_batch(data);
    let (trimmed_mean, trimmed_var) = compute_trimmed_stats(data, &depths, n_keep);

    iter_maybe_parallel!(0..n)
        .map(|i| normalized_distance(data, i, &trimmed_mean, &trimmed_var) > threshold)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Generate homogeneous functional data
    fn generate_normal_fdata(n: usize, m: usize, seed: u64) -> FdMatrix {
        let mut rng = StdRng::seed_from_u64(seed);
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let phase: f64 = rng.gen::<f64>() * 0.2;
            let amp: f64 = 1.0 + rng.gen::<f64>() * 0.1;
            for j in 0..m {
                let noise: f64 = rng.sample::<f64, _>(StandardNormal) * 0.05;
                data[(i, j)] = amp * (2.0 * PI * t[j] + phase).sin() + noise;
            }
        }
        data
    }

    /// Generate data with obvious outliers
    fn generate_data_with_outlier(n: usize, m: usize, n_outliers: usize) -> FdMatrix {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

        let mut data = FdMatrix::zeros(n, m);

        // Normal curves
        for i in 0..(n - n_outliers) {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin();
            }
        }

        // Outlier curves (shifted up by 10)
        for i in (n - n_outliers)..n {
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j]).sin() + 10.0;
            }
        }

        data
    }

    // ============== Threshold tests ==============

    #[test]
    fn test_outliers_threshold_lrt_returns_positive() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);

        assert!(threshold > 0.0, "Threshold should be positive");
    }

    #[test]
    fn test_outliers_threshold_lrt_deterministic() {
        let n = 15;
        let m = 25;
        let data = generate_normal_fdata(n, m, 42);

        let t1 = outliers_threshold_lrt(&data, 30, 0.1, 0.1, 123, 0.95);
        let t2 = outliers_threshold_lrt(&data, 30, 0.1, 0.1, 123, 0.95);

        assert!(
            (t1 - t2).abs() < 1e-10,
            "Same seed should give same threshold"
        );
    }

    #[test]
    fn test_outliers_threshold_lrt_percentile_effect() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        let t_low = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.50);
        let t_high = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.99);

        assert!(
            t_high >= t_low,
            "Higher percentile should give higher or equal threshold"
        );
    }

    #[test]
    fn test_outliers_threshold_lrt_invalid_input() {
        // Too few observations
        let data = FdMatrix::zeros(2, 30);
        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10, "Should return 0 for n < 3");

        // Empty m
        let data = FdMatrix::zeros(10, 0);
        let threshold = outliers_threshold_lrt(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
    }

    // ============== Detection tests ==============

    #[test]
    fn test_detect_outliers_lrt_finds_obvious_outlier() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 1);

        // Use a reasonable threshold
        let outliers = detect_outliers_lrt(&data, 3.0, 0.1);

        assert_eq!(outliers.len(), n);

        // The last curve (outlier) should be detected
        assert!(outliers[n - 1], "Obvious outlier should be detected");

        // Most normal curves should not be outliers
        let n_detected: usize = outliers.iter().filter(|&&x| x).count();
        assert!(n_detected <= 3, "Should not detect too many outliers");
    }

    #[test]
    fn test_detect_outliers_lrt_homogeneous_data() {
        let n = 20;
        let m = 30;
        let data = generate_normal_fdata(n, m, 42);

        // With very high threshold, no outliers
        let outliers = detect_outliers_lrt(&data, 100.0, 0.1);

        let n_detected: usize = outliers.iter().filter(|&&x| x).count();
        assert_eq!(
            n_detected, 0,
            "Very high threshold should detect no outliers"
        );
    }

    #[test]
    fn test_detect_outliers_lrt_threshold_effect() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 3);

        let low_thresh = detect_outliers_lrt(&data, 2.0, 0.1);
        let high_thresh = detect_outliers_lrt(&data, 10.0, 0.1);

        let n_low: usize = low_thresh.iter().filter(|&&x| x).count();
        let n_high: usize = high_thresh.iter().filter(|&&x| x).count();

        assert!(
            n_low >= n_high,
            "Lower threshold should detect more or equal outliers"
        );
    }

    #[test]
    fn test_detect_outliers_lrt_invalid_input() {
        // Too few observations
        let data = FdMatrix::zeros(2, 30);
        let outliers = detect_outliers_lrt(&data, 3.0, 0.1);
        assert_eq!(outliers.len(), 2);
        assert!(
            outliers.iter().all(|&x| !x),
            "Should return all false for n < 3"
        );
    }

    #[test]
    fn test_identical_data_outliers() {
        let n = 10;
        let m = 20;
        let data = FdMatrix::from_column_major(vec![1.0; n * m], n, m).unwrap();
        let flags = detect_outliers_lrt(&data, 1.0, 0.15);
        assert_eq!(flags.len(), n);
        // All identical → no outliers
        for &f in &flags {
            assert!(!f);
        }
    }

    #[test]
    fn test_n3_minimal_outliers() {
        // Minimum viable: 3 curves
        let n = 3;
        let m = 10;
        let mut data_vec = vec![0.0; n * m];
        // Third curve is an outlier
        for j in 0..m {
            data_vec[j * n] = 0.0;
            data_vec[1 + j * n] = 0.1;
            data_vec[2 + j * n] = 100.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        let flags = detect_outliers_lrt(&data, 0.5, 0.15);
        assert_eq!(flags.len(), n);
    }

    // ============== With-distribution tests ==============

    #[test]
    fn test_with_dist_returns_sorted_distribution() {
        let data = generate_normal_fdata(20, 30, 42);
        let nb = 50;
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);

        assert_eq!(dist.len(), nb, "Distribution length should equal nb");
        for w in dist.windows(2) {
            assert!(w[0] <= w[1], "Distribution should be sorted");
        }
        let idx = ((nb as f64 * 0.95) as usize).min(nb - 1);
        assert!(
            (threshold - dist[idx]).abs() < 1e-10,
            "Threshold should match distribution at percentile index"
        );
    }

    #[test]
    fn test_with_dist_matches_scalar() {
        let data = generate_normal_fdata(15, 25, 99);
        let scalar = outliers_threshold_lrt(&data, 40, 0.1, 0.1, 123, 0.95);
        let (with_dist, _) = outliers_threshold_lrt_with_dist(&data, 40, 0.1, 0.1, 123, 0.95);
        assert!(
            (scalar - with_dist).abs() < 1e-10,
            "Scalar version should match with_dist version"
        );
    }

    #[test]
    fn test_bootstrap_dist_enables_pvalue() {
        let n = 20;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 1);
        let trim = 0.1;

        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 200, 0.1, trim, 42, 0.99);
        let nb = dist.len();

        // Compute per-curve distances
        let n_keep = ((1.0 - trim) * n as f64).ceil() as usize;
        let state = SortedReferenceState::from_reference(&data);
        let streaming_fm = StreamingFraimanMuniz::new(state, true);
        let depths = streaming_fm.depth_batch(&data);
        let (tmean, tvar) = compute_trimmed_stats(&data, &depths, n_keep);

        // p-value for the outlier curve (last one)
        let d_outlier = normalized_distance(&data, n - 1, &tmean, &tvar);
        let p_outlier =
            (dist.iter().filter(|&&v| v >= d_outlier).count() as f64 + 1.0) / (nb as f64 + 1.0);

        // p-value for a normal curve (first one)
        let d_normal = normalized_distance(&data, 0, &tmean, &tvar);
        let p_normal =
            (dist.iter().filter(|&&v| v >= d_normal).count() as f64 + 1.0) / (nb as f64 + 1.0);

        assert!(
            p_outlier < 0.05,
            "Outlier should have small p-value, got {p_outlier}"
        );
        assert!(
            p_normal > 0.05,
            "Normal curve should have large p-value, got {p_normal}"
        );
    }

    #[test]
    fn test_with_dist_invalid_input() {
        let data = FdMatrix::zeros(2, 30);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 50, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
        assert!(dist.is_empty(), "Should return empty dist for n < 3");
    }

    #[test]
    fn test_all_false_high_threshold() {
        let n = 10;
        let m = 20;
        let data_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        // Very high threshold → no outliers
        let flags = detect_outliers_lrt(&data, 1e10, 0.15);
        for &f in &flags {
            assert!(!f, "High threshold should produce no outliers");
        }
    }

    // ============== n_keep clamping tests ==============

    #[test]
    fn test_trim_zero_no_trimming() {
        // trim=0 → n_keep=n, exercises the skip-partial-sort branch in compute_trimmed_stats
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 0.0, 42, 0.95);
        assert!(threshold > 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 0.0);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_near_one_heavy_trimming() {
        // trim=0.9 → n_keep=1, exercises minimal trim set (single deepest curve)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 0.9, 42, 0.95);
        assert!(threshold >= 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 0.9);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_one_clamps_to_one() {
        // trim=1.0 → n_keep would be 0, must clamp to 1 (was a panic before fix)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, 1.0, 42, 0.95);
        assert!(threshold >= 0.0);
        let flags = detect_outliers_lrt(&data, threshold, 1.0);
        assert_eq!(flags.len(), 10);
    }

    #[test]
    fn test_trim_negative_clamps_to_n() {
        // trim=-0.5 → n_keep would exceed n, must clamp to n (was a panic before fix)
        let data = generate_normal_fdata(10, 20, 42);
        let threshold = outliers_threshold_lrt(&data, 30, 0.1, -0.5, 42, 0.95);
        assert!(threshold > 0.0);
        let flags = detect_outliers_lrt(&data, threshold, -0.5);
        assert_eq!(flags.len(), 10);
    }

    // ============== Bootstrap parameter edge cases ==============

    #[test]
    fn test_smo_zero_no_noise() {
        // smo=0 → bootstrap resamples without smoothing noise
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 30, 0.0, 0.1, 42, 0.95);
        assert!(threshold > 0.0);
        assert_eq!(dist.len(), 30);
    }

    #[test]
    fn test_nb_zero_empty_bootstrap() {
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 0, 0.1, 0.1, 42, 0.95);
        assert!(threshold.abs() < 1e-10);
        assert!(dist.is_empty());
    }

    #[test]
    fn test_nb_one_single_bootstrap() {
        let data = generate_normal_fdata(10, 20, 42);
        let (threshold, dist) = outliers_threshold_lrt_with_dist(&data, 1, 0.1, 0.1, 42, 0.95);
        assert_eq!(dist.len(), 1);
        // With 1 iteration, threshold must equal the single value
        assert!((threshold - dist[0]).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_zero_returns_minimum() {
        let data = generate_normal_fdata(15, 20, 42);
        let nb = 50;
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);
        let t_zero = outliers_threshold_lrt(&data, nb, 0.1, 0.1, 42, 0.0);
        assert!(
            (t_zero - dist[0]).abs() < 1e-10,
            "percentile=0 should return the minimum of the distribution"
        );
    }

    #[test]
    fn test_percentile_one_returns_maximum() {
        let data = generate_normal_fdata(15, 20, 42);
        let nb = 50;
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, nb, 0.1, 0.1, 42, 0.95);
        let t_one = outliers_threshold_lrt(&data, nb, 0.1, 0.1, 42, 1.0);
        assert!(
            (t_one - *dist.last().unwrap()).abs() < 1e-10,
            "percentile=1 should return the maximum of the distribution"
        );
    }

    // ============== Distribution invariant tests ==============

    #[test]
    fn test_distribution_values_non_negative() {
        let data = generate_normal_fdata(15, 20, 42);
        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 50, 0.1, 0.1, 42, 0.95);
        for &v in &dist {
            assert!(v >= 0.0, "Max-distances must be non-negative, got {v}");
        }
    }

    // ============== detect_outliers_lrt edge cases ==============

    #[test]
    fn test_detect_m_zero_returns_all_false() {
        let data = FdMatrix::zeros(10, 0);
        let flags = detect_outliers_lrt(&data, 3.0, 0.1);
        assert_eq!(flags.len(), 10);
        assert!(flags.iter().all(|&f| !f));
    }

    #[test]
    fn test_detect_multiple_outliers() {
        let data = generate_data_with_outlier(20, 30, 3);
        let flags = detect_outliers_lrt(&data, 3.0, 0.1);
        // All three outlier curves (indices 17, 18, 19) should be detected
        let outlier_count = flags[17..20].iter().filter(|&&x| x).count();
        assert!(
            outlier_count >= 2,
            "At least 2 of 3 outliers should be detected, got {outlier_count}"
        );
    }

    // ============== End-to-end integration ==============

    #[test]
    fn test_end_to_end_threshold_then_detect() {
        let data = generate_data_with_outlier(20, 30, 2);
        let threshold = outliers_threshold_lrt(&data, 100, 0.1, 0.1, 42, 0.99);
        let flags = detect_outliers_lrt(&data, threshold, 0.1);

        // Outlier curves (last 2) should be flagged
        assert!(
            flags[18] || flags[19],
            "At least one outlier should be detected in end-to-end flow"
        );
        // Normal curves should mostly not be flagged
        let false_positives = flags[..18].iter().filter(|&&x| x).count();
        assert!(
            false_positives <= 2,
            "False positive count should be low, got {false_positives}"
        );
    }

    #[test]
    fn test_end_to_end_with_dist_pvalues_all_curves() {
        // Full pipeline: bootstrap dist → per-curve p-values → outlier classification
        let n = 25;
        let m = 30;
        let data = generate_data_with_outlier(n, m, 2);
        let trim = 0.1;

        let (_, dist) = outliers_threshold_lrt_with_dist(&data, 200, 0.1, trim, 42, 0.99);
        let nb = dist.len();

        let n_keep = ((1.0 - trim) * n as f64).ceil().max(1.0) as usize;
        let n_keep = n_keep.min(n);
        let state = SortedReferenceState::from_reference(&data);
        let streaming_fm = StreamingFraimanMuniz::new(state, true);
        let depths = streaming_fm.depth_batch(&data);
        let (tmean, tvar) = compute_trimmed_stats(&data, &depths, n_keep);

        // Compute p-values for all curves
        let pvalues: Vec<f64> = (0..n)
            .map(|i| {
                let d = normalized_distance(&data, i, &tmean, &tvar);
                (dist.iter().filter(|&&v| v >= d).count() as f64 + 1.0) / (nb as f64 + 1.0)
            })
            .collect();

        // Normal curves (0..23) should have large p-values
        let normal_small_p = pvalues[..23].iter().filter(|&&p| p < 0.01).count();
        assert_eq!(
            normal_small_p, 0,
            "Normal curves should not have tiny p-values"
        );

        // Outlier curves (23, 24) should have small p-values
        for &i in &[23, 24] {
            assert!(
                pvalues[i] < 0.05,
                "Outlier curve {i} should have small p-value, got {}",
                pvalues[i]
            );
        }
    }
}
