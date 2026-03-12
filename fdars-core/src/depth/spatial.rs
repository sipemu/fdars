//! Functional spatial depth measures (FSD and KFSD).

use crate::helpers::simpsons_weights;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Compute Functional Spatial Depth for 1D functional data.
///
/// Uses L2 norm with Simpson's integration weights to match R's `depth.FSD()`.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for (nobj x n_points)
/// * `data_ori` - Reference data (nori x n_points)
/// * `argvals` - Optional evaluation grid; if None, uses uniform \[0,1\] grid
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_spatial_1d(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    argvals: Option<&[f64]>,
) -> Vec<f64> {
    let nobj = data_obj.nrows();
    let nori = data_ori.nrows();
    let n_points = data_obj.ncols();

    if nobj == 0 || nori == 0 || n_points == 0 {
        return Vec::new();
    }

    // Build integration weights from argvals
    let default_argvals: Vec<f64>;
    let weights = if let Some(av) = argvals {
        simpsons_weights(av)
    } else {
        default_argvals = (0..n_points)
            .map(|i| i as f64 / (n_points - 1).max(1) as f64)
            .collect();
        simpsons_weights(&default_argvals)
    };

    iter_maybe_parallel!(0..nobj)
        .map(|i| {
            let mut sum_unit = vec![0.0; n_points];

            for j in 0..nori {
                // Compute L2 norm with integration weights
                let mut norm_sq = 0.0;
                for t in 0..n_points {
                    let d = data_ori[(j, t)] - data_obj[(i, t)];
                    norm_sq += weights[t] * d * d;
                }

                let norm = norm_sq.sqrt();
                if norm > 1e-10 {
                    let inv_norm = 1.0 / norm;
                    for t in 0..n_points {
                        sum_unit[t] += (data_ori[(j, t)] - data_obj[(i, t)]) * inv_norm;
                    }
                }
            }

            // Compute L2 norm of average unit vector with integration weights
            let mut avg_norm_sq = 0.0;
            for t in 0..n_points {
                let avg = sum_unit[t] / nori as f64;
                avg_norm_sq += weights[t] * avg * avg;
            }

            1.0 - avg_norm_sq.sqrt()
        })
        .collect()
}

/// Compute Functional Spatial Depth for 2D functional data.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn functional_spatial_2d(data_obj: &FdMatrix, data_ori: &FdMatrix) -> Vec<f64> {
    functional_spatial_1d(data_obj, data_ori, None)
}

/// Compute kernel distance contribution for a single (j,k) pair.
fn kernel_pair_contribution(j: usize, k: usize, m1: &FdMatrix, m2: &[f64]) -> Option<f64> {
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
    let m_ijk = (1.0 + m1[(j, k)] - m2[j] - m2[k]) / denom;
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
fn kfsd_accumulate(m2: &[f64], m1: &FdMatrix, nori: usize) -> (f64, usize) {
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

    let mut m1 = FdMatrix::zeros(nori, nori);
    for j in 0..nori {
        m1[(j, j)] = 1.0;
    }
    for (j, k, kval) in m1_upper {
        m1[(j, k)] = kval;
        m1[(k, j)] = kval;
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
#[must_use = "expensive computation whose result should not be discarded"]
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
#[must_use = "expensive computation whose result should not be discarded"]
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
