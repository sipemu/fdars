//! Smoothing functions for functional data.
//!
//! This module provides kernel-based smoothing methods including
//! Nadaraya-Watson, local linear, and local polynomial regression.

use rayon::prelude::*;

/// Gaussian kernel function.
fn gaussian_kernel(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Epanechnikov kernel function.
fn epanechnikov_kernel(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Get kernel function by name.
fn get_kernel(kernel_type: &str) -> fn(f64) -> f64 {
    match kernel_type.to_lowercase().as_str() {
        "epanechnikov" | "epan" => epanechnikov_kernel,
        _ => gaussian_kernel,
    }
}

/// Nadaraya-Watson kernel smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `kernel` - Kernel type ("gaussian" or "epanechnikov")
///
/// # Returns
/// Smoothed values at x_new
pub fn nadaraya_watson(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    bandwidth: f64,
    kernel: &str,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; x_new.len()];
    }

    let kernel_fn = get_kernel(kernel);

    x_new
        .par_iter()
        .map(|&x0| {
            let mut num = 0.0;
            let mut denom = 0.0;

            for i in 0..n {
                let u = (x[i] - x0) / bandwidth;
                let w = kernel_fn(u);
                num += w * y[i];
                denom += w;
            }

            if denom > 1e-10 {
                num / denom
            } else {
                0.0
            }
        })
        .collect()
}

/// Local linear regression smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `kernel` - Kernel type
///
/// # Returns
/// Smoothed values at x_new
pub fn local_linear(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    bandwidth: f64,
    kernel: &str,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 {
        return vec![0.0; x_new.len()];
    }

    let kernel_fn = get_kernel(kernel);

    x_new
        .par_iter()
        .map(|&x0| {
            // Compute weighted moments
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;

            for i in 0..n {
                let u = (x[i] - x0) / bandwidth;
                let w = kernel_fn(u);
                let d = x[i] - x0;

                s0 += w;
                s1 += w * d;
                s2 += w * d * d;
                t0 += w * y[i];
                t1 += w * y[i] * d;
            }

            // Solve local linear regression
            let det = s0 * s2 - s1 * s1;
            if det.abs() > 1e-10 {
                (s2 * t0 - s1 * t1) / det
            } else if s0 > 1e-10 {
                t0 / s0
            } else {
                0.0
            }
        })
        .collect()
}

/// Local polynomial regression smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `bandwidth` - Kernel bandwidth
/// * `degree` - Polynomial degree
/// * `kernel` - Kernel type
///
/// # Returns
/// Smoothed values at x_new
pub fn local_polynomial(
    x: &[f64],
    y: &[f64],
    x_new: &[f64],
    bandwidth: f64,
    degree: usize,
    kernel: &str,
) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || bandwidth <= 0.0 || degree == 0 {
        return vec![0.0; x_new.len()];
    }

    if degree == 1 {
        return local_linear(x, y, x_new, bandwidth, kernel);
    }

    let kernel_fn = get_kernel(kernel);
    let p = degree + 1; // Number of coefficients

    x_new
        .par_iter()
        .map(|&x0| {
            // Build weighted design matrix and response
            let mut xtx = vec![0.0; p * p];
            let mut xty = vec![0.0; p];

            for i in 0..n {
                let u = (x[i] - x0) / bandwidth;
                let w = kernel_fn(u);
                let d = x[i] - x0;

                // Build powers of d
                let mut powers = vec![1.0; p];
                for j in 1..p {
                    powers[j] = powers[j - 1] * d;
                }

                // Accumulate X'WX and X'Wy
                for j in 0..p {
                    for k in 0..p {
                        xtx[j * p + k] += w * powers[j] * powers[k];
                    }
                    xty[j] += w * powers[j] * y[i];
                }
            }

            // Solve using simple Gaussian elimination (for small p)
            // For numerical stability in real applications, use proper linear algebra
            // This is a simplified implementation
            let mut a = xtx.clone();
            let mut b = xty.clone();

            for i in 0..p {
                // Find pivot
                let mut max_idx = i;
                for j in (i + 1)..p {
                    if a[j * p + i].abs() > a[max_idx * p + i].abs() {
                        max_idx = j;
                    }
                }

                // Swap rows
                if max_idx != i {
                    for k in 0..p {
                        a.swap(i * p + k, max_idx * p + k);
                    }
                    b.swap(i, max_idx);
                }

                let pivot = a[i * p + i];
                if pivot.abs() < 1e-10 {
                    continue;
                }

                // Eliminate
                for j in (i + 1)..p {
                    let factor = a[j * p + i] / pivot;
                    for k in i..p {
                        a[j * p + k] -= factor * a[i * p + k];
                    }
                    b[j] -= factor * b[i];
                }
            }

            // Back substitution
            let mut coefs = vec![0.0; p];
            for i in (0..p).rev() {
                let mut sum = b[i];
                for j in (i + 1)..p {
                    sum -= a[i * p + j] * coefs[j];
                }
                if a[i * p + i].abs() > 1e-10 {
                    coefs[i] = sum / a[i * p + i];
                }
            }

            coefs[0] // Return intercept (fitted value at x0)
        })
        .collect()
}

/// k-Nearest Neighbors smoother.
///
/// # Arguments
/// * `x` - Predictor values
/// * `y` - Response values
/// * `x_new` - Points at which to evaluate the smoother
/// * `k` - Number of neighbors
///
/// # Returns
/// Smoothed values at x_new
pub fn knn_smoother(x: &[f64], y: &[f64], x_new: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    if n == 0 || y.len() != n || x_new.is_empty() || k == 0 {
        return vec![0.0; x_new.len()];
    }

    let k = k.min(n);

    x_new
        .par_iter()
        .map(|&x0| {
            // Compute distances
            let mut distances: Vec<(usize, f64)> = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| (i, (xi - x0).abs()))
                .collect();

            // Partial sort to get k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Average of k nearest neighbors
            let sum: f64 = distances.iter().take(k).map(|(i, _)| y[*i]).sum();
            sum / k as f64
        })
        .collect()
}

/// Compute smoothing matrix for Nadaraya-Watson.
///
/// Returns the smoother matrix S such that y_hat = S * y.
pub fn smoothing_matrix_nw(x: &[f64], bandwidth: f64, kernel: &str) -> Vec<f64> {
    let n = x.len();
    if n == 0 || bandwidth <= 0.0 {
        return Vec::new();
    }

    let kernel_fn = get_kernel(kernel);
    let mut s = vec![0.0; n * n];

    for i in 0..n {
        let mut row_sum = 0.0;
        for j in 0..n {
            let u = (x[j] - x[i]) / bandwidth;
            s[i + j * n] = kernel_fn(u);
            row_sum += s[i + j * n];
        }
        if row_sum > 1e-10 {
            for j in 0..n {
                s[i + j * n] /= row_sum;
            }
        }
    }

    s
}
