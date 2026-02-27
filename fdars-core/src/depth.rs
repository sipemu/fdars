//! Depth measures for functional data.
//!
//! This module provides various depth measures for assessing the centrality
//! of functional observations within a reference sample.

use crate::helpers::simpsons_weights;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::streaming_depth::{
    FullReferenceState, SortedReferenceState, StreamingBd, StreamingDepth, StreamingFraimanMuniz,
    StreamingMbd,
};
use rand::prelude::*;
use rand_distr::StandardNormal;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute Fraiman-Muniz depth for 1D functional data.
///
/// Uses the FM1 formula: d = 1 - |0.5 - Fn(x)|
/// With scale=true: d = 2 * min(Fn(x), 1-Fn(x))
///
/// # Arguments
/// * `data_obj` - Data to compute depth for (nobj x n_points)
/// * `data_ori` - Reference data (nori x n_points)
/// * `scale` - Whether to scale the depth values
pub fn fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() == 0 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = SortedReferenceState::from_reference(data_ori);
    let streaming = StreamingFraimanMuniz::new(state, scale);
    streaming.depth_batch(data_obj)
}

/// Compute Fraiman-Muniz depth for 2D functional data (surfaces).
pub fn fraiman_muniz_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // Same implementation as 1D - iterate over all grid points
    fraiman_muniz_1d(data_obj, data_ori, scale)
}

/// Compute modal depth for 1D functional data.
///
/// Uses a Gaussian kernel to measure density around each curve.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `h` - Bandwidth parameter
pub fn modal_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut depth = 0.0;

            for j in 0..nori {
                let mut dist_sq = 0.0;
                for t in 0..n_points {
                    let diff = data_obj[(i, t)] - data_ori[(j, t)];
                    dist_sq += diff * diff;
                }
                let dist = (dist_sq / n_points as f64).sqrt();
                let kernel_val = (-0.5 * (dist / h).powi(2)).exp();
                depth += kernel_val;
            }

            depth / nori as f64
        })
        .collect()
}

/// Compute modal depth for 2D functional data.
pub fn modal_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    modal_1d(data_obj, data_ori, h)
}

/// Compute random projection depth for 1D functional data.
///
/// Projects curves to scalars using random projections and computes
/// average univariate depth.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nproj` - Number of random projections
pub fn random_projection_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 || nproj == 0 {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let projections: Vec<Vec<f64>> = (0..nproj)
        .map(|_| {
            let mut proj: Vec<f64> = (0..n_points).map(|_| rng.sample(StandardNormal)).collect();
            let norm: f64 = proj.iter().map(|x| x * x).sum::<f64>().sqrt();
            proj.iter_mut().for_each(|x| *x /= norm);
            proj
        })
        .collect();

    // Pre-compute and pre-sort reference projections (identical for every query curve)
    let sorted_proj_ori: Vec<Vec<f64>> = projections
        .iter()
        .map(|proj| {
            let mut proj_ori: Vec<f64> = (0..nori)
                .map(|j| {
                    let mut p = 0.0;
                    for t in 0..n_points {
                        p += data_ori[(j, t)] * proj[t];
                    }
                    p
                })
                .collect();
            proj_ori.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            proj_ori
        })
        .collect();

    let denom = nori as f64 + 1.0;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total_depth = 0.0;

            for (proj, sorted_ori) in projections.iter().zip(sorted_proj_ori.iter()) {
                let mut proj_i = 0.0;
                for t in 0..n_points {
                    proj_i += data_obj[(i, t)] * proj[t];
                }

                // O(log N) rank lookup via binary search
                let below = sorted_ori.partition_point(|&v| v < proj_i);
                let above = nori - sorted_ori.partition_point(|&v| v <= proj_i);
                let depth = (below.min(above) as f64 + 1.0) / denom;

                total_depth += depth;
            }

            total_depth / nproj as f64
        })
        .collect()
}

/// Compute random projection depth for 2D functional data.
pub fn random_projection_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_projection_1d(data_obj, data_ori, nproj)
}

/// Compute random Tukey depth for 1D functional data.
///
/// Takes the minimum over all random projections (more conservative than RP depth).
pub fn random_tukey_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 || nproj == 0 {
        return Vec::new();
    }

    let mut rng = rand::thread_rng();
    let projections: Vec<Vec<f64>> = (0..nproj)
        .map(|_| {
            let mut proj: Vec<f64> = (0..n_points).map(|_| rng.sample(StandardNormal)).collect();
            let norm: f64 = proj.iter().map(|x| x * x).sum::<f64>().sqrt();
            proj.iter_mut().for_each(|x| *x /= norm);
            proj
        })
        .collect();

    // Pre-compute and pre-sort reference projections (identical for every query curve)
    let sorted_proj_ori: Vec<Vec<f64>> = projections
        .iter()
        .map(|proj| {
            let mut proj_ori: Vec<f64> = (0..nori)
                .map(|j| {
                    let mut p = 0.0;
                    for t in 0..n_points {
                        p += data_ori[(j, t)] * proj[t];
                    }
                    p
                })
                .collect();
            proj_ori.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            proj_ori
        })
        .collect();

    let denom = nori as f64 + 1.0;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut min_depth = f64::INFINITY;

            for (proj, sorted_ori) in projections.iter().zip(sorted_proj_ori.iter()) {
                let mut proj_i = 0.0;
                for t in 0..n_points {
                    proj_i += data_obj[(i, t)] * proj[t];
                }

                // O(log N) rank lookup via binary search
                let below = sorted_ori.partition_point(|&v| v < proj_i);
                let above = nori - sorted_ori.partition_point(|&v| v <= proj_i);
                let depth = (below.min(above) as f64 + 1.0) / denom;

                min_depth = min_depth.min(depth);
            }

            min_depth
        })
        .collect()
}

/// Compute random Tukey depth for 2D functional data.
pub fn random_tukey_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_tukey_1d(data_obj, data_ori, nproj)
}

/// Compute Functional Spatial Depth for 1D functional data.
pub fn functional_spatial_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut sum_unit = vec![0.0; n_points];

            for j in 0..nori {
                // First pass: compute norm without allocating a direction buffer
                let mut norm_sq = 0.0;
                for t in 0..n_points {
                    let d = data_ori[(j, t)] - data_obj[(i, t)];
                    norm_sq += d * d;
                }

                let norm = norm_sq.sqrt();
                if norm > 1e-10 {
                    // Second pass: accumulate unit-vector sum using the known norm
                    let inv_norm = 1.0 / norm;
                    for t in 0..n_points {
                        sum_unit[t] += (data_ori[(j, t)] - data_obj[(i, t)]) * inv_norm;
                    }
                }
            }

            let mut avg_norm_sq = 0.0;
            for t in 0..n_points {
                let avg = sum_unit[t] / nori as f64;
                avg_norm_sq += avg * avg;
            }

            1.0 - avg_norm_sq.sqrt()
        })
        .collect()
}

/// Compute Functional Spatial Depth for 2D functional data.
pub fn functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    functional_spatial_1d(data_obj, data_ori)
}

/// Compute kernel distance contribution for a single (j,k) pair.
fn kernel_pair_contribution(j: usize, k: usize, m1: &[Vec<f64>], m2: &[f64]) -> Option<f64> {
    let denom_j_sq = 2.0 - 2.0 * m2[j];
    if denom_j_sq < 1e-20 {
        return None;
    }
    let denom_k_sq = 2.0 - 2.0 * m2[k];
    if denom_k_sq < 1e-20 {
        return None;
    }
    let denom = denom_j_sq.sqrt() * denom_k_sq.sqrt();
    if denom <= 1e-20 {
        return None;
    }
    let m_ijk = (1.0 + m1[j][k] - m2[j] - m2[k]) / denom;
    if m_ijk.is_finite() {
        Some(m_ijk)
    } else {
        None
    }
}

/// Accumulate the kernel spatial depth statistic for a single observation.
/// Returns (total_sum, valid_count) from the double sum over reference pairs.
///
/// Exploits symmetry: `kernel_pair_contribution(j, k, m1, m2) == kernel_pair_contribution(k, j, m1, m2)`
/// since m1 is symmetric and the formula `(1 + m1[j][k] - m2[j] - m2[k]) / (sqrt(2-2*m2[j]) * sqrt(2-2*m2[k]))`
/// is symmetric in j and k. Loops over the upper triangle only.
fn kfsd_accumulate(m2: &[f64], m1: &[Vec<f64>], nori: usize) -> (f64, usize) {
    let mut total_sum = 0.0;
    let mut valid_count = 0;

    // Diagonal contributions (j == k)
    for j in 0..nori {
        if let Some(val) = kernel_pair_contribution(j, j, m1, m2) {
            total_sum += val;
            valid_count += 1;
        }
    }

    // Upper triangle contributions (j < k), counted twice by symmetry
    for j in 0..nori {
        for k in (j + 1)..nori {
            if let Some(val) = kernel_pair_contribution(j, k, m1, m2) {
                total_sum += 2.0 * val;
                valid_count += 2;
            }
        }
    }

    (total_sum, valid_count)
}

/// Shared implementation for kernel functional spatial depth.
/// Uses weighted L2 norm: sum_t weights[t] * (f(t) - g(t))^2.
fn kfsd_weighted(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64, weights: &[f64]) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();
    let h_sq = h * h;

    // Pre-compute M1[j,k] = K(X_j, X_k) for reference data
    let m1_upper: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..nori)
        .flat_map(|j| {
            ((j + 1)..nori)
                .map(|k| {
                    let mut sum = 0.0;
                    for t in 0..n_points {
                        let diff = data_ori[(j, t)] - data_ori[(k, t)];
                        sum += weights[t] * diff * diff;
                    }
                    (j, k, (-sum / h_sq).exp())
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut m1 = vec![vec![0.0; nori]; nori];
    for j in 0..nori {
        m1[j][j] = 1.0;
    }
    for (j, k, kval) in m1_upper {
        m1[j][k] = kval;
        m1[k][j] = kval;
    }

    let nori_f64 = nori as f64;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let m2: Vec<f64> = (0..nori)
                .map(|j| {
                    let mut sum = 0.0;
                    for t in 0..n_points {
                        let diff = data_obj[(i, t)] - data_ori[(j, t)];
                        sum += weights[t] * diff * diff;
                    }
                    (-sum / h_sq).exp()
                })
                .collect();

            let (total_sum, valid_count) = kfsd_accumulate(&m2, &m1, nori);

            if valid_count > 0 && total_sum >= 0.0 {
                1.0 - total_sum.sqrt() / nori_f64
            } else if total_sum < 0.0 {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}


/// Compute Kernel Functional Spatial Depth (KFSD) for 1D functional data.
///
/// Implements the RKHS-based formulation.
pub fn kernel_functional_spatial_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: &[f64],
    h: f64,
) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    kfsd_weighted(data_obj, data_ori, h, &weights)
}

/// Compute Kernel Functional Spatial Depth (KFSD) for 2D functional data.
pub fn kernel_functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    let weights = vec![1.0; n_points];
    kfsd_weighted(data_obj, data_ori, h, &weights)
}

/// Compute Band Depth (BD) for 1D functional data.
///
/// BD(x) = proportion of pairs (i,j) where x lies within the band formed by curves i and j.
pub fn band_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() < 2 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = FullReferenceState::from_reference(data_ori);
    let streaming = StreamingBd::new(state);
    streaming.depth_batch(data_obj)
}

/// Compute Modified Band Depth (MBD) for 1D functional data.
///
/// MBD(x) = average over pairs of the proportion of the domain where x is inside the band.
pub fn modified_band_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() < 2 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = SortedReferenceState::from_reference(data_ori);
    let streaming = StreamingMbd::new(state);
    streaming.depth_batch(data_obj)
}

/// Compute Modified Epigraph Index (MEI) for 1D functional data.
///
/// MEI measures the proportion of time a curve is below other curves.
pub fn modified_epigraph_index_1d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0;

            for j in 0..nori {
                let mut count = 0.0;

                for t in 0..n_points {
                    let xi = data_obj[(i, t)];
                    let xj = data_ori[(j, t)];

                    if xi < xj {
                        count += 1.0;
                    } else if (xi - xj).abs() < 1e-12 {
                        count += 0.5;
                    }
                }

                total += count / n_points as f64;
            }

            total / nori as f64
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn uniform_grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    fn generate_centered_data(n: usize, m: usize) -> FdMatrix {
        let argvals = uniform_grid(m);
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
            }
        }
        FdMatrix::from_column_major(data, n, m).unwrap()
    }

    // ============== Fraiman-Muniz tests ==============

    #[test]
    fn test_fraiman_muniz() {
        // Simple test: identical data should give maximum depth
        let data = FdMatrix::from_column_major(vec![1.0, 1.0, 2.0, 2.0], 2, 2).unwrap(); // 2 identical curves, 2 points each
        let depths = fraiman_muniz_1d(&data, &data, true);
        assert_eq!(depths.len(), 2);
    }

    #[test]
    fn test_fraiman_muniz_central_deeper() {
        let n = 20;
        let m = 30;
        let data = generate_centered_data(n, m);
        let depths = fraiman_muniz_1d(&data, &data, true);

        // Central curve (index n/2) should have higher depth than extreme curves
        let central_depth = depths[n / 2];
        let edge_depth = depths[0];
        assert!(
            central_depth > edge_depth,
            "Central curve should be deeper: {} > {}",
            central_depth,
            edge_depth
        );
    }

    #[test]
    fn test_fraiman_muniz_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = fraiman_muniz_1d(&data, &data, true);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_fraiman_muniz_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(fraiman_muniz_1d(&empty, &empty, true).is_empty());
    }

    // ============== Modal depth tests ==============

    #[test]
    fn test_modal_central_deeper() {
        let n = 20;
        let m = 30;
        let data = generate_centered_data(n, m);
        let depths = modal_1d(&data, &data, 0.5);

        let central_depth = depths[n / 2];
        let edge_depth = depths[0];
        assert!(central_depth > edge_depth, "Central curve should be deeper");
    }

    #[test]
    fn test_modal_positive() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = modal_1d(&data, &data, 0.5);

        for d in &depths {
            assert!(*d > 0.0, "Modal depth should be positive");
        }
    }

    #[test]
    fn test_modal_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(modal_1d(&empty, &empty, 0.5).is_empty());
    }

    // ============== Random projection depth tests ==============

    #[test]
    fn test_rp_depth_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = random_projection_1d(&data, &data, 50);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "RP depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_rp_depth_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(random_projection_1d(&empty, &empty, 10).is_empty());
    }

    // ============== Random Tukey depth tests ==============

    #[test]
    fn test_random_tukey_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = random_tukey_1d(&data, &data, 50);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Tukey depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_random_tukey_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(random_tukey_1d(&empty, &empty, 10).is_empty());
    }

    // ============== Functional spatial depth tests ==============

    #[test]
    fn test_functional_spatial_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = functional_spatial_1d(&data, &data);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Spatial depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_functional_spatial_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(functional_spatial_1d(&empty, &empty).is_empty());
    }

    // ============== Band depth tests ==============

    #[test]
    fn test_band_depth_central_deeper() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = band_1d(&data, &data);

        // Central curve should be in more bands
        let central_depth = depths[n / 2];
        let edge_depth = depths[0];
        assert!(
            central_depth >= edge_depth,
            "Central curve should have higher band depth"
        );
    }

    #[test]
    fn test_band_depth_range() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = band_1d(&data, &data);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Band depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_band_depth_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(band_1d(&empty, &empty).is_empty());
        let single = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
        assert!(band_1d(&single, &single).is_empty()); // need at least 2 ref curves
    }

    // ============== Modified band depth tests ==============

    #[test]
    fn test_modified_band_depth_range() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = modified_band_1d(&data, &data);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "MBD should be in [0, 1]");
        }
    }

    #[test]
    fn test_modified_band_depth_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(modified_band_1d(&empty, &empty).is_empty());
    }

    // ============== Modified epigraph index tests ==============

    #[test]
    fn test_mei_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let mei = modified_epigraph_index_1d(&data, &data);

        for d in &mei {
            assert!(*d >= 0.0 && *d <= 1.0, "MEI should be in [0, 1]");
        }
    }

    #[test]
    fn test_mei_invalid() {
        let empty = FdMatrix::zeros(0, 0);
        assert!(modified_epigraph_index_1d(&empty, &empty).is_empty());
    }

    // ============== KFSD 1D tests ==============

    #[test]
    fn test_kfsd_1d_range() {
        let n = 10;
        let m = 20;
        let argvals = uniform_grid(m);
        let data = generate_centered_data(n, m);
        let depths = kernel_functional_spatial_1d(&data, &data, &argvals, 0.5);

        assert_eq!(depths.len(), n);
        for d in &depths {
            assert!(
                *d >= -0.01 && *d <= 1.01,
                "KFSD depth should be near [0, 1], got {}",
                d
            );
            assert!(d.is_finite(), "KFSD depth should be finite");
        }

        // Central curve should have higher depth
        let central_depth = depths[n / 2];
        let edge_depth = depths[0];
        assert!(
            central_depth > edge_depth,
            "Central KFSD depth {} should be > edge depth {}",
            central_depth,
            edge_depth
        );
    }

    #[test]
    fn test_kfsd_1d_identical() {
        // All identical curves should exercise the denom_j_sq < 1e-20 path
        let n = 5;
        let m = 10;
        let argvals = uniform_grid(m);
        let data_vec: Vec<f64> = (0..n * m)
            .map(|i| (2.0 * PI * (i % m) as f64 / (m - 1) as f64).sin())
            .collect();
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

        // When all curves are identical, kernel distances are all 1.0
        // and denom_j_sq = K(x,x) + K(y,y) - 2*K(x,y) = 1 + 1 - 2*1 = 0
        let depths = kernel_functional_spatial_1d(&data, &data, &argvals, 0.5);

        assert_eq!(depths.len(), n);
        for d in &depths {
            assert!(
                d.is_finite(),
                "KFSD depth should be finite for identical curves"
            );
        }
    }

    #[test]
    fn test_kfsd_1d_invalid() {
        let argvals = uniform_grid(10);
        let empty = FdMatrix::zeros(0, 0);
        assert!(kernel_functional_spatial_1d(&empty, &empty, &argvals, 0.5).is_empty());
        let empty_obj = FdMatrix::zeros(0, 0);
        assert!(kernel_functional_spatial_1d(&empty_obj, &empty_obj, &argvals, 0.5).is_empty());
    }

    // ============== KFSD 2D tests ==============

    #[test]
    fn test_kfsd_2d_range() {
        let n = 8;
        let m = 15;
        let data = generate_centered_data(n, m);
        let depths = kernel_functional_spatial_2d(&data, &data, 0.5);

        assert_eq!(depths.len(), n);
        for d in &depths {
            assert!(
                *d >= -0.01 && *d <= 1.01,
                "KFSD 2D depth should be near [0, 1], got {}",
                d
            );
            assert!(d.is_finite(), "KFSD 2D depth should be finite");
        }
    }

    // ============== 2D delegation tests ==============

    #[test]
    fn test_fraiman_muniz_2d_delegates() {
        let n = 10;
        let m = 15;
        let data = generate_centered_data(n, m);
        let depths_1d = fraiman_muniz_1d(&data, &data, true);
        let depths_2d = fraiman_muniz_2d(&data, &data, true);
        assert_eq!(depths_1d, depths_2d);
    }

    #[test]
    fn test_modal_2d_delegates() {
        let n = 10;
        let m = 15;
        let data = generate_centered_data(n, m);
        let depths_1d = modal_1d(&data, &data, 0.5);
        let depths_2d = modal_2d(&data, &data, 0.5);
        assert_eq!(depths_1d, depths_2d);
    }

    #[test]
    fn test_functional_spatial_2d_delegates() {
        let n = 10;
        let m = 15;
        let data = generate_centered_data(n, m);
        let depths_1d = functional_spatial_1d(&data, &data);
        let depths_2d = functional_spatial_2d(&data, &data);
        assert_eq!(depths_1d, depths_2d);
    }

    #[test]
    fn test_random_projection_2d_returns_valid() {
        let n = 10;
        let m = 15;
        let data = generate_centered_data(n, m);
        let depths = random_projection_2d(&data, &data, 20);
        assert_eq!(depths.len(), n);
        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "RP 2D depth should be in [0, 1]");
        }
    }

    // ============== Golden-value regression tests ==============

    /// Fixed small dataset: 5 curves, 10 time points (deterministic)
    fn golden_data() -> FdMatrix {
        let n = 5;
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
            }
        }
        FdMatrix::from_column_major(data, n, m).unwrap()
    }

    #[test]
    fn test_fm_golden_values_scaled() {
        let data = golden_data();
        let depths = fraiman_muniz_1d(&data, &data, true);
        let expected = [0.4, 0.8, 0.8, 0.4, 0.0];
        assert_eq!(depths.len(), expected.len());
        for (d, e) in depths.iter().zip(expected.iter()) {
            assert!(
                (d - e).abs() < 1e-10,
                "FM scaled golden mismatch: got {}, expected {}",
                d,
                e
            );
        }
    }

    #[test]
    fn test_fm_golden_values_unscaled() {
        let data = golden_data();
        let depths = fraiman_muniz_1d(&data, &data, false);
        let expected = [0.2, 0.4, 0.4, 0.2, 0.0];
        assert_eq!(depths.len(), expected.len());
        for (d, e) in depths.iter().zip(expected.iter()) {
            assert!(
                (d - e).abs() < 1e-10,
                "FM unscaled golden mismatch: got {}, expected {}",
                d,
                e
            );
        }
    }

    #[test]
    fn test_mbd_golden_values() {
        let data = golden_data();
        let depths = modified_band_1d(&data, &data);
        let expected = [0.4, 0.7, 0.8, 0.7, 0.4];
        assert_eq!(depths.len(), expected.len());
        for (d, e) in depths.iter().zip(expected.iter()) {
            assert!(
                (d - e).abs() < 1e-10,
                "MBD golden mismatch: got {}, expected {}",
                d,
                e
            );
        }
    }
}
