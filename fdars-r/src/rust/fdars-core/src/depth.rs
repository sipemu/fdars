//! Depth measures for functional data.
//!
//! This module provides various depth measures for assessing the centrality
//! of functional observations within a reference sample.

use crate::helpers::simpsons_weights;
use crate::iter_maybe_parallel;
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
/// * `data_obj` - Data to compute depth for (column-major, nobj x n_points)
/// * `data_ori` - Reference data (column-major, nori x n_points)
/// * `nobj` - Number of objects
/// * `nori` - Number of reference observations
/// * `n_points` - Number of evaluation points
/// * `scale` - Whether to scale the depth values
pub fn fraiman_muniz_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    scale: bool,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    let scale_factor = if scale { 2.0 } else { 1.0 };

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut depth_sum = 0.0;

            for t in 0..n_points {
                let x_t = data_obj[i + t * nobj];
                let mut le_count = 0;

                for j in 0..nori {
                    let y_t = data_ori[j + t * nori];
                    if y_t <= x_t {
                        le_count += 1;
                    }
                }

                let fn_x = le_count as f64 / nori as f64;
                let univariate_depth = fn_x.min(1.0 - fn_x) * scale_factor;
                depth_sum += univariate_depth;
            }

            depth_sum / n_points as f64
        })
        .collect()
}

/// Compute Fraiman-Muniz depth for 2D functional data (surfaces).
pub fn fraiman_muniz_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    scale: bool,
) -> Vec<f64> {
    // Same implementation as 1D - iterate over all grid points
    fraiman_muniz_1d(data_obj, data_ori, nobj, nori, n_points, scale)
}

/// Compute modal depth for 1D functional data.
///
/// Uses a Gaussian kernel to measure density around each curve.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nobj` - Number of objects
/// * `nori` - Number of reference observations
/// * `n_points` - Number of evaluation points
/// * `h` - Bandwidth parameter
pub fn modal_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    h: f64,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut depth = 0.0;

            for j in 0..nori {
                let mut dist_sq = 0.0;
                for t in 0..n_points {
                    let diff = data_obj[i + t * nobj] - data_ori[j + t * nori];
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
pub fn modal_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    h: f64,
) -> Vec<f64> {
    modal_1d(data_obj, data_ori, nobj, nori, n_points, h)
}

/// Compute random projection depth for 1D functional data.
///
/// Projects curves to scalars using random projections and computes
/// average univariate depth.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nobj` - Number of objects
/// * `nori` - Number of reference observations
/// * `n_points` - Number of evaluation points
/// * `nproj` - Number of random projections
pub fn random_projection_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    nproj: usize,
) -> Vec<f64> {
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

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total_depth = 0.0;

            for proj in &projections {
                let mut proj_i = 0.0;
                for t in 0..n_points {
                    proj_i += data_obj[i + t * nobj] * proj[t];
                }

                let mut proj_ori: Vec<f64> = (0..nori)
                    .map(|j| {
                        let mut p = 0.0;
                        for t in 0..n_points {
                            p += data_ori[j + t * nori] * proj[t];
                        }
                        p
                    })
                    .collect();

                proj_ori.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let below = proj_ori.iter().filter(|&&x| x < proj_i).count();
                let above = proj_ori.iter().filter(|&&x| x > proj_i).count();
                let depth = (below.min(above) as f64 + 1.0) / (nori as f64 + 1.0);

                total_depth += depth;
            }

            total_depth / nproj as f64
        })
        .collect()
}

/// Compute random projection depth for 2D functional data.
pub fn random_projection_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    nproj: usize,
) -> Vec<f64> {
    random_projection_1d(data_obj, data_ori, nobj, nori, n_points, nproj)
}

/// Compute random Tukey depth for 1D functional data.
///
/// Takes the minimum over all random projections (more conservative than RP depth).
pub fn random_tukey_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    nproj: usize,
) -> Vec<f64> {
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

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut min_depth = f64::INFINITY;

            for proj in &projections {
                let mut proj_i = 0.0;
                for t in 0..n_points {
                    proj_i += data_obj[i + t * nobj] * proj[t];
                }

                let proj_ori: Vec<f64> = (0..nori)
                    .map(|j| {
                        let mut p = 0.0;
                        for t in 0..n_points {
                            p += data_ori[j + t * nori] * proj[t];
                        }
                        p
                    })
                    .collect();

                let below = proj_ori.iter().filter(|&&x| x < proj_i).count();
                let above = proj_ori.iter().filter(|&&x| x > proj_i).count();
                let depth = (below.min(above) as f64 + 1.0) / (nori as f64 + 1.0);

                min_depth = min_depth.min(depth);
            }

            min_depth
        })
        .collect()
}

/// Compute random Tukey depth for 2D functional data.
pub fn random_tukey_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    nproj: usize,
) -> Vec<f64> {
    random_tukey_1d(data_obj, data_ori, nobj, nori, n_points, nproj)
}

/// Compute Functional Spatial Depth for 1D functional data.
pub fn functional_spatial_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut sum_unit = vec![0.0; n_points];

            for j in 0..nori {
                let mut direction = vec![0.0; n_points];
                let mut norm_sq = 0.0;

                for t in 0..n_points {
                    direction[t] = data_ori[j + t * nori] - data_obj[i + t * nobj];
                    norm_sq += direction[t] * direction[t];
                }

                let norm = norm_sq.sqrt();
                if norm > 1e-10 {
                    for t in 0..n_points {
                        sum_unit[t] += direction[t] / norm;
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
pub fn functional_spatial_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
) -> Vec<f64> {
    functional_spatial_1d(data_obj, data_ori, nobj, nori, n_points)
}

/// Compute Kernel Functional Spatial Depth (KFSD) for 1D functional data.
///
/// Implements the RKHS-based formulation.
pub fn kernel_functional_spatial_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    argvals: &[f64],
    h: f64,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);
    let h_sq = h * h;

    // Helper function to compute integrated L2 norm squared
    let norm_sq_integrated = |data1: &[f64],
                              row1: usize,
                              nrow1: usize,
                              data2: &[f64],
                              row2: usize,
                              nrow2: usize|
     -> f64 {
        let mut sum = 0.0;
        for t in 0..n_points {
            let diff = data1[row1 + t * nrow1] - data2[row2 + t * nrow2];
            sum += weights[t] * diff * diff;
        }
        sum
    };

    let kern = |dist_sq: f64| -> f64 { (-dist_sq / h_sq).exp() };

    // Pre-compute M1[j,k] = K(X_j, X_k) for all pairs in reference data
    let m1_upper: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..nori)
        .flat_map(|j| {
            ((j + 1)..nori)
                .map(|k| {
                    let mut sum = 0.0;
                    for t in 0..n_points {
                        let diff = data_ori[j + t * nori] - data_ori[k + t * nori];
                        sum += weights[t] * diff * diff;
                    }
                    let kval = (-sum / h_sq).exp();
                    (j, k, kval)
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
            let k02_i = 1.0;

            let m2: Vec<f64> = (0..nori)
                .map(|j| {
                    let d_sq = norm_sq_integrated(data_obj, i, nobj, data_ori, j, nori);
                    kern(d_sq)
                })
                .collect();

            let mut total_sum = 0.0;
            let mut valid_count = 0;

            for j in 0..nori {
                let k01_j = 1.0;
                let denom_j_sq = k02_i + k01_j - 2.0 * m2[j];

                if denom_j_sq < 1e-20 {
                    continue;
                }
                let denom_j = denom_j_sq.sqrt();

                for k in 0..nori {
                    let k01_k = 1.0;
                    let denom_k_sq = k02_i + k01_k - 2.0 * m2[k];

                    if denom_k_sq < 1e-20 {
                        continue;
                    }
                    let denom_k = denom_k_sq.sqrt();

                    let numerator = k02_i + m1[j][k] - m2[j] - m2[k];
                    let denom = denom_j * denom_k;

                    if denom > 1e-20 {
                        let m_ijk = numerator / denom;
                        if m_ijk.is_finite() {
                            total_sum += m_ijk;
                            valid_count += 1;
                        }
                    }
                }
            }

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

/// Compute Kernel Functional Spatial Depth (KFSD) for 2D functional data.
pub fn kernel_functional_spatial_2d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
    h: f64,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    let h_sq = h * h;

    let norm_sq = |data1: &[f64],
                   row1: usize,
                   nrow1: usize,
                   data2: &[f64],
                   row2: usize,
                   nrow2: usize|
     -> f64 {
        let mut sum = 0.0;
        for t in 0..n_points {
            let diff = data1[row1 + t * nrow1] - data2[row2 + t * nrow2];
            sum += diff * diff;
        }
        sum
    };

    let kern = |dist_sq: f64| -> f64 { (-dist_sq / h_sq).exp() };

    // Pre-compute M1 matrix
    let m1_upper: Vec<(usize, usize, f64)> = iter_maybe_parallel!(0..nori)
        .flat_map(|j| {
            ((j + 1)..nori)
                .map(|k| {
                    let d_sq = norm_sq(data_ori, j, nori, data_ori, k, nori);
                    let kval = kern(d_sq);
                    (j, k, kval)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut m1_mat = vec![vec![0.0; nori]; nori];
    for j in 0..nori {
        m1_mat[j][j] = 1.0;
    }
    for (j, k, kval) in m1_upper {
        m1_mat[j][k] = kval;
        m1_mat[k][j] = kval;
    }

    let nori_f64 = nori as f64;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let k02_i = 1.0;

            let m2: Vec<f64> = (0..nori)
                .map(|j| {
                    let d_sq = norm_sq(data_obj, i, nobj, data_ori, j, nori);
                    kern(d_sq)
                })
                .collect();

            let mut total_sum = 0.0;
            let mut valid_count = 0;

            for j in 0..nori {
                let k01_j = 1.0;
                let denom_j_sq = k02_i + k01_j - 2.0 * m2[j];

                if denom_j_sq < 1e-20 {
                    continue;
                }
                let denom_j = denom_j_sq.sqrt();

                for k in 0..nori {
                    let k01_k = 1.0;
                    let denom_k_sq = k02_i + k01_k - 2.0 * m2[k];

                    if denom_k_sq < 1e-20 {
                        continue;
                    }
                    let denom_k = denom_k_sq.sqrt();

                    let numerator = k02_i + m1_mat[j][k] - m2[j] - m2[k];
                    let denom = denom_j * denom_k;

                    if denom > 1e-20 {
                        let m_ijk = numerator / denom;
                        if m_ijk.is_finite() {
                            total_sum += m_ijk;
                            valid_count += 1;
                        }
                    }
                }
            }

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

/// Compute Band Depth (BD) for 1D functional data.
///
/// BD(x) = proportion of pairs (i,j) where x lies within the band formed by curves i and j.
pub fn band_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
) -> Vec<f64> {
    if nobj == 0 || nori < 2 || n_points == 0 {
        return Vec::new();
    }

    let n_pairs = (nori * (nori - 1)) / 2;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut count_in_band = 0usize;

            for j in 0..nori {
                for k in (j + 1)..nori {
                    let mut inside_band = true;

                    for t in 0..n_points {
                        let x_t = data_obj[i + t * nobj];
                        let y_j_t = data_ori[j + t * nori];
                        let y_k_t = data_ori[k + t * nori];

                        let band_min = y_j_t.min(y_k_t);
                        let band_max = y_j_t.max(y_k_t);

                        if x_t < band_min || x_t > band_max {
                            inside_band = false;
                            break;
                        }
                    }

                    if inside_band {
                        count_in_band += 1;
                    }
                }
            }

            count_in_band as f64 / n_pairs as f64
        })
        .collect()
}

/// Compute Modified Band Depth (MBD) for 1D functional data.
///
/// MBD(x) = average over pairs of the proportion of the domain where x is inside the band.
pub fn modified_band_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
) -> Vec<f64> {
    if nobj == 0 || nori < 2 || n_points == 0 {
        return Vec::new();
    }

    let n_pairs = (nori * (nori - 1)) / 2;

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total_proportion = 0.0;

            for j in 0..nori {
                for k in (j + 1)..nori {
                    let mut count_inside = 0usize;

                    for t in 0..n_points {
                        let x_t = data_obj[i + t * nobj];
                        let y_j_t = data_ori[j + t * nori];
                        let y_k_t = data_ori[k + t * nori];

                        let band_min = y_j_t.min(y_k_t);
                        let band_max = y_j_t.max(y_k_t);

                        if x_t >= band_min && x_t <= band_max {
                            count_inside += 1;
                        }
                    }

                    total_proportion += count_inside as f64 / n_points as f64;
                }
            }

            total_proportion / n_pairs as f64
        })
        .collect()
}

/// Compute Modified Epigraph Index (MEI) for 1D functional data.
///
/// MEI measures the proportion of time a curve is below other curves.
pub fn modified_epigraph_index_1d(
    data_obj: &[f64],
    data_ori: &[f64],
    nobj: usize,
    nori: usize,
    n_points: usize,
) -> Vec<f64> {
    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut total = 0.0;

            for j in 0..nori {
                let mut count = 0.0;

                for t in 0..n_points {
                    let xi = data_obj[i + t * nobj];
                    let xj = data_ori[j + t * nori];

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

    fn generate_centered_data(n: usize, m: usize) -> Vec<f64> {
        let argvals = uniform_grid(m);
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
            }
        }
        data
    }

    // ============== Fraiman-Muniz tests ==============

    #[test]
    fn test_fraiman_muniz() {
        // Simple test: identical data should give maximum depth
        let data = vec![1.0, 1.0, 2.0, 2.0]; // 2 identical curves, 2 points each
        let depths = fraiman_muniz_1d(&data, &data, 2, 2, 2, true);
        assert_eq!(depths.len(), 2);
    }

    #[test]
    fn test_fraiman_muniz_central_deeper() {
        let n = 20;
        let m = 30;
        let data = generate_centered_data(n, m);
        let depths = fraiman_muniz_1d(&data, &data, n, n, m, true);

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
        let depths = fraiman_muniz_1d(&data, &data, n, n, m, true);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_fraiman_muniz_invalid() {
        assert!(fraiman_muniz_1d(&[], &[], 0, 0, 0, true).is_empty());
    }

    // ============== Modal depth tests ==============

    #[test]
    fn test_modal_central_deeper() {
        let n = 20;
        let m = 30;
        let data = generate_centered_data(n, m);
        let depths = modal_1d(&data, &data, n, n, m, 0.5);

        let central_depth = depths[n / 2];
        let edge_depth = depths[0];
        assert!(central_depth > edge_depth, "Central curve should be deeper");
    }

    #[test]
    fn test_modal_positive() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = modal_1d(&data, &data, n, n, m, 0.5);

        for d in &depths {
            assert!(*d > 0.0, "Modal depth should be positive");
        }
    }

    #[test]
    fn test_modal_invalid() {
        assert!(modal_1d(&[], &[], 0, 0, 0, 0.5).is_empty());
    }

    // ============== Random projection depth tests ==============

    #[test]
    fn test_rp_depth_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = random_projection_1d(&data, &data, n, n, m, 50);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "RP depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_rp_depth_invalid() {
        assert!(random_projection_1d(&[], &[], 0, 0, 0, 10).is_empty());
    }

    // ============== Random Tukey depth tests ==============

    #[test]
    fn test_random_tukey_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = random_tukey_1d(&data, &data, n, n, m, 50);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Tukey depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_random_tukey_invalid() {
        assert!(random_tukey_1d(&[], &[], 0, 0, 0, 10).is_empty());
    }

    // ============== Functional spatial depth tests ==============

    #[test]
    fn test_functional_spatial_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = functional_spatial_1d(&data, &data, n, n, m);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Spatial depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_functional_spatial_invalid() {
        assert!(functional_spatial_1d(&[], &[], 0, 0, 0).is_empty());
    }

    // ============== Band depth tests ==============

    #[test]
    fn test_band_depth_central_deeper() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = band_1d(&data, &data, n, n, m);

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
        let depths = band_1d(&data, &data, n, n, m);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "Band depth should be in [0, 1]");
        }
    }

    #[test]
    fn test_band_depth_invalid() {
        assert!(band_1d(&[], &[], 0, 0, 0).is_empty());
        assert!(band_1d(&[1.0, 2.0], &[1.0, 2.0], 1, 1, 2).is_empty()); // need at least 2 ref curves
    }

    // ============== Modified band depth tests ==============

    #[test]
    fn test_modified_band_depth_range() {
        let n = 10;
        let m = 20;
        let data = generate_centered_data(n, m);
        let depths = modified_band_1d(&data, &data, n, n, m);

        for d in &depths {
            assert!(*d >= 0.0 && *d <= 1.0, "MBD should be in [0, 1]");
        }
    }

    #[test]
    fn test_modified_band_depth_invalid() {
        assert!(modified_band_1d(&[], &[], 0, 0, 0).is_empty());
    }

    // ============== Modified epigraph index tests ==============

    #[test]
    fn test_mei_range() {
        let n = 15;
        let m = 20;
        let data = generate_centered_data(n, m);
        let mei = modified_epigraph_index_1d(&data, &data, n, n, m);

        for d in &mei {
            assert!(*d >= 0.0 && *d <= 1.0, "MEI should be in [0, 1]");
        }
    }

    #[test]
    fn test_mei_invalid() {
        assert!(modified_epigraph_index_1d(&[], &[], 0, 0, 0).is_empty());
    }
}
