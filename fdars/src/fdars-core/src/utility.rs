//! Utility functions for functional data analysis.

use crate::helpers::simpsons_weights;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Compute Simpson's rule integration for a single function.
///
/// # Arguments
/// * `values` - Function values at evaluation points
/// * `argvals` - Evaluation points
pub fn integrate_simpson(values: &[f64], argvals: &[f64]) -> f64 {
    if values.len() != argvals.len() || values.is_empty() {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum()
}

/// Compute inner product between two functional data curves.
///
/// # Arguments
/// * `curve1` - First curve values
/// * `curve2` - Second curve values
/// * `argvals` - Evaluation points
pub fn inner_product(curve1: &[f64], curve2: &[f64], argvals: &[f64]) -> f64 {
    if curve1.len() != curve2.len() || curve1.len() != argvals.len() || curve1.is_empty() {
        return 0.0;
    }

    let weights = simpsons_weights(argvals);
    curve1
        .iter()
        .zip(curve2.iter())
        .zip(weights.iter())
        .map(|((&c1, &c2), &w)| c1 * c2 * w)
        .sum()
}

/// Compute inner product matrix for functional data.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of observations
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
///
/// # Returns
/// Symmetric inner product matrix (n x n), column-major
pub fn inner_product_matrix(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> Vec<f64> {
    if n == 0 || m == 0 || argvals.len() != m || data.len() != n * m {
        return Vec::new();
    }

    let weights = simpsons_weights(argvals);

    // Compute upper triangle in parallel
    let upper_triangle: Vec<(usize, usize, f64)> = (0..n)
        .into_par_iter()
        .flat_map(|i| {
            (i..n)
                .map(|j| {
                    let mut ip = 0.0;
                    for k in 0..m {
                        ip += data[i + k * n] * data[j + k * n] * weights[k];
                    }
                    (i, j, ip)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    // Build symmetric matrix
    let mut result = vec![0.0; n * n];
    for (i, j, ip) in upper_triangle {
        result[i + j * n] = ip;
        result[j + i * n] = ip;
    }

    result
}

/// Compute the Adot matrix used in PCvM statistic.
pub fn compute_adot(n: usize, inprod: &[f64]) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }

    let expected_len = (n * n + n) / 2;
    if inprod.len() != expected_len {
        return Vec::new();
    }

    let out_len = (n * n - n + 2) / 2;
    let mut adot_vec = vec![0.0; out_len];

    adot_vec[0] = PI * (n + 1) as f64;

    // Collect all (i, j) pairs for parallel processing
    let pairs: Vec<(usize, usize)> = (2..=n).flat_map(|i| (1..i).map(move |j| (i, j))).collect();

    // Compute adot values in parallel
    let results: Vec<(usize, f64)> = pairs
        .into_par_iter()
        .map(|(i, j)| {
            let mut sumr = 0.0;

            for r in 1..=n {
                if i == r || j == r {
                    sumr += PI;
                } else {
                    let auxi = i * (i - 1) / 2;
                    let auxj = j * (j - 1) / 2;
                    let auxr = r * (r - 1) / 2;

                    let ij = auxi + j - 1;
                    let ii = auxi + i - 1;
                    let jj = auxj + j - 1;
                    let rr = auxr + r - 1;

                    let ir = if i > r { auxi + r - 1 } else { auxr + i - 1 };
                    let rj = if r > j { auxr + j - 1 } else { auxj + r - 1 };
                    let jr = rj;

                    let num = inprod[ij] - inprod[ir] - inprod[rj] + inprod[rr];
                    let aux1 = (inprod[ii] - 2.0 * inprod[ir] + inprod[rr]).sqrt();
                    let aux2 = (inprod[jj] - 2.0 * inprod[jr] + inprod[rr]).sqrt();
                    let den = aux1 * aux2;

                    let mut quo = if den.abs() > 1e-10 { num / den } else { 0.0 };
                    quo = quo.clamp(-1.0, 1.0);

                    sumr += (PI - quo.acos()).abs();
                }
            }

            let idx = 1 + ((i - 1) * (i - 2) / 2) + j - 1;
            (idx, sumr)
        })
        .collect();

    // Fill in the results
    for (idx, val) in results {
        if idx < adot_vec.len() {
            adot_vec[idx] = val;
        }
    }

    adot_vec
}

/// Compute the PCvM statistic.
pub fn pcvm_statistic(adot_vec: &[f64], residuals: &[f64]) -> f64 {
    let n = residuals.len();

    if n == 0 || adot_vec.is_empty() {
        return 0.0;
    }

    let mut sums = 0.0;
    for i in 2..=n {
        for j in 1..i {
            let idx = 1 + ((i - 1) * (i - 2) / 2) + j - 1;
            if idx < adot_vec.len() {
                sums += residuals[i - 1] * adot_vec[idx] * residuals[j - 1];
            }
        }
    }

    let diag_sum: f64 = residuals.iter().map(|r| r * r).sum();
    adot_vec[0] * diag_sum + 2.0 * sums
}

/// Result of random projection statistics.
pub struct RpStatResult {
    /// CvM statistics for each projection
    pub cvm: Vec<f64>,
    /// KS statistics for each projection
    pub ks: Vec<f64>,
}

/// Compute random projection statistics.
pub fn rp_stat(proj_x_ord: &[i32], residuals: &[f64], n_proj: usize) -> RpStatResult {
    let n = residuals.len();

    if n == 0 || n_proj == 0 || proj_x_ord.len() != n * n_proj {
        return RpStatResult {
            cvm: Vec::new(),
            ks: Vec::new(),
        };
    }

    // Process projections in parallel
    let stats: Vec<(f64, f64)> = (0..n_proj)
        .into_par_iter()
        .map(|p| {
            let mut y = vec![0.0; n];
            let mut cumsum = 0.0;

            for i in 0..n {
                let idx = proj_x_ord[p * n + i] as usize;
                if idx > 0 && idx <= n {
                    cumsum += residuals[idx - 1];
                }
                y[i] = cumsum;
            }

            let sum_y_sq: f64 = y.iter().map(|yi| yi * yi).sum();
            let cvm = sum_y_sq / (n * n) as f64;

            let max_abs_y = y.iter().map(|yi| yi.abs()).fold(0.0, f64::max);
            let ks = max_abs_y / (n as f64).sqrt();

            (cvm, ks)
        })
        .collect();

    let cvm_stats: Vec<f64> = stats.iter().map(|(cvm, _)| *cvm).collect();
    let ks_stats: Vec<f64> = stats.iter().map(|(_, ks)| *ks).collect();

    RpStatResult {
        cvm: cvm_stats,
        ks: ks_stats,
    }
}

/// k-NN prediction for functional regression.
pub fn knn_predict(
    distance_matrix: &[f64],
    y: &[f64],
    n_train: usize,
    n_test: usize,
    k: usize,
) -> Vec<f64> {
    if n_train == 0 || n_test == 0 || k == 0 || y.len() != n_train {
        return vec![0.0; n_test];
    }

    let k = k.min(n_train);

    (0..n_test)
        .into_par_iter()
        .map(|i| {
            // Get distances from test point i to all training points
            let mut distances: Vec<(usize, f64)> = (0..n_train)
                .map(|j| (j, distance_matrix[i + j * n_test]))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Average of k nearest neighbors
            let sum: f64 = distances.iter().take(k).map(|(j, _)| y[*j]).sum();
            sum / k as f64
        })
        .collect()
}

/// Compute leave-one-out cross-validation error for k-NN.
pub fn knn_loocv(distance_matrix: &[f64], y: &[f64], n: usize, k: usize) -> f64 {
    if n == 0 || k == 0 || y.len() != n || distance_matrix.len() != n * n {
        return f64::INFINITY;
    }

    let k = k.min(n - 1);

    let errors: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Get distances from point i to all other points
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance_matrix[i + j * n]))
                .collect();

            // Sort by distance
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Prediction
            let pred: f64 = distances.iter().take(k).map(|(j, _)| y[*j]).sum::<f64>() / k as f64;

            // Squared error
            (y[i] - pred).powi(2)
        })
        .collect();

    errors.iter().sum::<f64>() / n as f64
}
