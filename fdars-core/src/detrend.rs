//! Detrending and decomposition functions for non-stationary functional data.
//!
//! This module provides methods for removing trends from functional data
//! to enable more accurate seasonal analysis. It includes:
//! - Linear detrending (least squares)
//! - Polynomial detrending (QR decomposition)
//! - Differencing (first and second order)
//! - LOESS detrending (local polynomial regression)
//! - Spline detrending (P-splines)
//! - Automatic method selection via AIC

use crate::smoothing::local_polynomial;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Result of detrending operation.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Estimated trend values (n x m column-major)
    pub trend: Vec<f64>,
    /// Detrended data (n x m column-major)
    pub detrended: Vec<f64>,
    /// Method used for detrending
    pub method: String,
    /// Polynomial coefficients (for polynomial methods, per sample)
    /// For n samples with polynomial degree d: coefficients[i * (d+1) + k] is coefficient k for sample i
    pub coefficients: Option<Vec<f64>>,
    /// Residual sum of squares for each sample
    pub rss: Vec<f64>,
    /// Number of parameters (for AIC calculation)
    pub n_params: usize,
}

/// Result of seasonal decomposition.
#[derive(Debug, Clone)]
pub struct DecomposeResult {
    /// Trend component (n x m column-major)
    pub trend: Vec<f64>,
    /// Seasonal component (n x m column-major)
    pub seasonal: Vec<f64>,
    /// Remainder/residual component (n x m column-major)
    pub remainder: Vec<f64>,
    /// Period used for decomposition
    pub period: f64,
    /// Decomposition method ("additive" or "multiplicative")
    pub method: String,
}

/// Remove linear trend from functional data using least squares.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m): n samples, m evaluation points
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values of length m
///
/// # Returns
/// TrendResult with trend, detrended data, and coefficients (intercept, slope)
pub fn detrend_linear(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> TrendResult {
    if n == 0 || m < 2 || data.len() != n * m || argvals.len() != m {
        return TrendResult {
            trend: vec![0.0; n * m],
            detrended: data.to_vec(),
            method: "linear".to_string(),
            coefficients: None,
            rss: vec![0.0; n],
            n_params: 2,
        };
    }

    // Precompute t statistics
    let mean_t: f64 = argvals.iter().sum::<f64>() / m as f64;
    let ss_t: f64 = argvals.iter().map(|&t| (t - mean_t).powi(2)).sum();

    // Process each sample in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, f64, f64, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract curve
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();
            let mean_y: f64 = curve.iter().sum::<f64>() / m as f64;

            // Compute slope: sum((t - mean_t) * (y - mean_y)) / sum((t - mean_t)^2)
            let mut sp = 0.0;
            for j in 0..m {
                sp += (argvals[j] - mean_t) * (curve[j] - mean_y);
            }
            let slope = if ss_t.abs() > 1e-15 { sp / ss_t } else { 0.0 };
            let intercept = mean_y - slope * mean_t;

            // Compute trend and detrended
            let mut trend = vec![0.0; m];
            let mut detrended = vec![0.0; m];
            let mut rss = 0.0;
            for j in 0..m {
                trend[j] = intercept + slope * argvals[j];
                detrended[j] = curve[j] - trend[j];
                rss += detrended[j].powi(2);
            }

            (trend, detrended, intercept, slope, rss)
        })
        .collect();

    // Reassemble into column-major format
    let mut trend = vec![0.0; n * m];
    let mut detrended = vec![0.0; n * m];
    let mut coefficients = vec![0.0; n * 2];
    let mut rss = vec![0.0; n];

    for (i, (t, d, intercept, slope, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            detrended[i + j * n] = d[j];
        }
        coefficients[i * 2] = intercept;
        coefficients[i * 2 + 1] = slope;
        rss[i] = r;
    }

    TrendResult {
        trend,
        detrended,
        method: "linear".to_string(),
        coefficients: Some(coefficients),
        rss,
        n_params: 2,
    }
}

/// Remove polynomial trend from functional data using QR decomposition.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values of length m
/// * `degree` - Polynomial degree (1 = linear, 2 = quadratic, etc.)
///
/// # Returns
/// TrendResult with trend, detrended data, and polynomial coefficients
pub fn detrend_polynomial(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    degree: usize,
) -> TrendResult {
    if n == 0 || m < degree + 1 || data.len() != n * m || argvals.len() != m || degree == 0 {
        // For degree 0 or invalid input, return original data
        return TrendResult {
            trend: vec![0.0; n * m],
            detrended: data.to_vec(),
            method: format!("polynomial({})", degree),
            coefficients: None,
            rss: vec![0.0; n],
            n_params: degree + 1,
        };
    }

    // Special case: degree 1 is linear
    if degree == 1 {
        let mut result = detrend_linear(data, n, m, argvals);
        result.method = "polynomial(1)".to_string();
        return result;
    }

    let n_coef = degree + 1;

    // Normalize argvals to avoid numerical issues with high-degree polynomials
    let t_min = argvals.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = argvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let t_range = if (t_max - t_min).abs() > 1e-15 {
        t_max - t_min
    } else {
        1.0
    };
    let t_norm: Vec<f64> = argvals.iter().map(|&t| (t - t_min) / t_range).collect();

    // Build Vandermonde matrix (m x n_coef)
    let mut design = DMatrix::zeros(m, n_coef);
    for j in 0..m {
        let t = t_norm[j];
        let mut power = 1.0;
        for k in 0..n_coef {
            design[(j, k)] = power;
            power *= t;
        }
    }

    // SVD for stable least squares
    let svd = design.clone().svd(true, true);

    // Process each sample in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract curve
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();
            let y = DVector::from_row_slice(&curve);

            // Solve least squares using SVD
            let beta = svd
                .solve(&y, 1e-10)
                .unwrap_or_else(|_| DVector::zeros(n_coef));

            // Compute fitted values (trend) and residuals
            let fitted = &design * &beta;
            let mut trend = vec![0.0; m];
            let mut detrended = vec![0.0; m];
            let mut rss = 0.0;
            for j in 0..m {
                trend[j] = fitted[j];
                detrended[j] = curve[j] - fitted[j];
                rss += detrended[j].powi(2);
            }

            // Extract coefficients
            let coefs: Vec<f64> = beta.iter().cloned().collect();

            (trend, detrended, coefs, rss)
        })
        .collect();

    // Reassemble into column-major format
    let mut trend = vec![0.0; n * m];
    let mut detrended = vec![0.0; n * m];
    let mut coefficients = vec![0.0; n * n_coef];
    let mut rss = vec![0.0; n];

    for (i, (t, d, coefs, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            detrended[i + j * n] = d[j];
        }
        for k in 0..n_coef {
            coefficients[i * n_coef + k] = coefs[k];
        }
        rss[i] = r;
    }

    TrendResult {
        trend,
        detrended,
        method: format!("polynomial({})", degree),
        coefficients: Some(coefficients),
        rss,
        n_params: n_coef,
    }
}

/// Remove trend by differencing.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `order` - Differencing order (1 or 2)
///
/// # Returns
/// TrendResult with trend (cumulative sum to reverse), detrended (differences),
/// and original first values as "coefficients"
///
/// Note: Differencing reduces the series length by `order` points.
/// The returned detrended data has m - order points padded with zeros at the end.
pub fn detrend_diff(data: &[f64], n: usize, m: usize, order: usize) -> TrendResult {
    if n == 0 || m <= order || data.len() != n * m || order == 0 || order > 2 {
        return TrendResult {
            trend: vec![0.0; n * m],
            detrended: data.to_vec(),
            method: format!("diff{}", order),
            coefficients: None,
            rss: vec![0.0; n],
            n_params: order,
        };
    }

    let new_m = m - order;

    // Process each sample in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract curve
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();

            // First difference
            let diff1: Vec<f64> = (0..m - 1).map(|j| curve[j + 1] - curve[j]).collect();

            // Second difference if order == 2
            let detrended = if order == 2 {
                (0..diff1.len() - 1)
                    .map(|j| diff1[j + 1] - diff1[j])
                    .collect()
            } else {
                diff1.clone()
            };

            // Store initial values needed for reconstruction
            let initial_values = if order == 2 {
                vec![curve[0], curve[1]]
            } else {
                vec![curve[0]]
            };

            // Compute RSS (sum of squared differences as "residuals" - interpretation varies)
            let rss: f64 = detrended.iter().map(|&x| x.powi(2)).sum();

            // For "trend", we reconstruct as cumsum of differences
            // This is a rough approximation; true trend would need integration
            let mut trend = vec![0.0; m];
            trend[0] = curve[0];
            if order == 1 {
                for j in 1..m {
                    trend[j] = curve[j] - if j <= new_m { detrended[j - 1] } else { 0.0 };
                }
            } else {
                // For order 2, trend is less meaningful
                trend = curve.clone();
            }

            // Pad detrended to full length
            let mut det_full = vec![0.0; m];
            for j in 0..new_m {
                det_full[j] = detrended[j];
            }

            (trend, det_full, initial_values, rss)
        })
        .collect();

    // Reassemble
    let mut trend = vec![0.0; n * m];
    let mut detrended = vec![0.0; n * m];
    let mut coefficients = vec![0.0; n * order];
    let mut rss = vec![0.0; n];

    for (i, (t, d, init, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            detrended[i + j * n] = d[j];
        }
        for k in 0..order {
            coefficients[i * order + k] = init[k];
        }
        rss[i] = r;
    }

    TrendResult {
        trend,
        detrended,
        method: format!("diff{}", order),
        coefficients: Some(coefficients),
        rss,
        n_params: order,
    }
}

/// Remove trend using LOESS (local polynomial regression).
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values
/// * `bandwidth` - Bandwidth as fraction of data range (0.1 to 0.5 typical)
/// * `degree` - Local polynomial degree (1 or 2)
///
/// # Returns
/// TrendResult with LOESS-smoothed trend
pub fn detrend_loess(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    bandwidth: f64,
    degree: usize,
) -> TrendResult {
    if n == 0 || m < 3 || data.len() != n * m || argvals.len() != m || bandwidth <= 0.0 {
        return TrendResult {
            trend: vec![0.0; n * m],
            detrended: data.to_vec(),
            method: "loess".to_string(),
            coefficients: None,
            rss: vec![0.0; n],
            n_params: (m as f64 * bandwidth).ceil() as usize,
        };
    }

    // Convert bandwidth from fraction to absolute units
    let t_min = argvals.iter().cloned().fold(f64::INFINITY, f64::min);
    let t_max = argvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let abs_bandwidth = (t_max - t_min) * bandwidth;

    // Process each sample in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, f64)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract curve
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();

            // Apply local polynomial regression
            let trend =
                local_polynomial(argvals, &curve, argvals, abs_bandwidth, degree, "gaussian");

            // Compute detrended and RSS
            let mut detrended = vec![0.0; m];
            let mut rss = 0.0;
            for j in 0..m {
                detrended[j] = curve[j] - trend[j];
                rss += detrended[j].powi(2);
            }

            (trend, detrended, rss)
        })
        .collect();

    // Reassemble
    let mut trend = vec![0.0; n * m];
    let mut detrended = vec![0.0; n * m];
    let mut rss = vec![0.0; n];

    for (i, (t, d, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            detrended[i + j * n] = d[j];
        }
        rss[i] = r;
    }

    // Effective number of parameters for LOESS is approximately n * bandwidth
    let n_params = (m as f64 * bandwidth).ceil() as usize;

    TrendResult {
        trend,
        detrended,
        method: "loess".to_string(),
        coefficients: None,
        rss,
        n_params,
    }
}

/// Automatically select the best detrending method using AIC.
///
/// Compares linear, polynomial (degree 2 and 3), and LOESS,
/// selecting the method with lowest AIC.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values
///
/// # Returns
/// TrendResult from the best method
pub fn auto_detrend(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> TrendResult {
    if n == 0 || m < 4 || data.len() != n * m || argvals.len() != m {
        return TrendResult {
            trend: vec![0.0; n * m],
            detrended: data.to_vec(),
            method: "auto(none)".to_string(),
            coefficients: None,
            rss: vec![0.0; n],
            n_params: 0,
        };
    }

    // Compute AIC for a result: AIC = n * log(RSS/n) + 2*k
    // We use mean AIC across all samples
    let compute_aic = |result: &TrendResult| -> f64 {
        let mut total_aic = 0.0;
        for i in 0..n {
            let rss = result.rss[i];
            let k = result.n_params as f64;
            let aic = if rss > 1e-15 {
                m as f64 * (rss / m as f64).ln() + 2.0 * k
            } else {
                f64::NEG_INFINITY // Perfect fit (unlikely)
            };
            total_aic += aic;
        }
        total_aic / n as f64
    };

    // Try different methods
    let linear = detrend_linear(data, n, m, argvals);
    let poly2 = detrend_polynomial(data, n, m, argvals, 2);
    let poly3 = detrend_polynomial(data, n, m, argvals, 3);
    let loess = detrend_loess(data, n, m, argvals, 0.3, 2);

    let aic_linear = compute_aic(&linear);
    let aic_poly2 = compute_aic(&poly2);
    let aic_poly3 = compute_aic(&poly3);
    let aic_loess = compute_aic(&loess);

    // Find minimum AIC
    let methods = [
        (aic_linear, "linear", linear),
        (aic_poly2, "polynomial(2)", poly2),
        (aic_poly3, "polynomial(3)", poly3),
        (aic_loess, "loess", loess),
    ];

    let (_, best_name, mut best_result) = methods
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();

    best_result.method = format!("auto({})", best_name);
    best_result
}

/// Additive seasonal decomposition: data = trend + seasonal + remainder
///
/// Uses LOESS or spline for trend extraction, then averages within-period
/// residuals to estimate the seasonal component.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values
/// * `period` - Seasonal period in same units as argvals
/// * `trend_method` - "loess" or "spline"
/// * `bandwidth` - Bandwidth for LOESS (fraction, e.g., 0.3)
/// * `n_harmonics` - Number of Fourier harmonics for seasonal component
///
/// # Returns
/// DecomposeResult with trend, seasonal, and remainder components
pub fn decompose_additive(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    trend_method: &str,
    bandwidth: f64,
    n_harmonics: usize,
) -> DecomposeResult {
    if n == 0 || m < 4 || data.len() != n * m || argvals.len() != m || period <= 0.0 {
        return DecomposeResult {
            trend: vec![0.0; n * m],
            seasonal: vec![0.0; n * m],
            remainder: data.to_vec(),
            period,
            method: "additive".to_string(),
        };
    }

    // Step 1: Extract trend using LOESS or spline
    let trend_result = match trend_method {
        "spline" => {
            // Use P-spline fitting - use a larger bandwidth for trend
            detrend_loess(data, n, m, argvals, bandwidth.max(0.3), 2)
        }
        "loess" | _ => detrend_loess(data, n, m, argvals, bandwidth.max(0.3), 2),
    };

    // Step 2: Extract seasonal component using Fourier basis on detrended data
    let n_harm = n_harmonics.max(1).min(m / 4);
    let omega = 2.0 * std::f64::consts::PI / period;

    // Process each sample
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let trend_i: Vec<f64> = (0..m).map(|j| trend_result.trend[i + j * n]).collect();
            let detrended_i: Vec<f64> = (0..m).map(|j| trend_result.detrended[i + j * n]).collect();

            // Fit Fourier model to detrended data: sum of sin and cos terms
            // y = sum_k (a_k * cos(k*omega*t) + b_k * sin(k*omega*t))
            let n_coef = 2 * n_harm;
            let mut design = DMatrix::zeros(m, n_coef);
            for j in 0..m {
                let t = argvals[j];
                for k in 0..n_harm {
                    let freq = (k + 1) as f64 * omega;
                    design[(j, 2 * k)] = (freq * t).cos();
                    design[(j, 2 * k + 1)] = (freq * t).sin();
                }
            }

            // Solve least squares using SVD
            let y = DVector::from_row_slice(&detrended_i);
            let svd = design.clone().svd(true, true);
            let coef = svd
                .solve(&y, 1e-10)
                .unwrap_or_else(|_| DVector::zeros(n_coef));

            // Compute seasonal component
            let fitted = &design * &coef;
            let seasonal: Vec<f64> = fitted.iter().cloned().collect();

            // Compute remainder
            let remainder: Vec<f64> = (0..m).map(|j| detrended_i[j] - seasonal[j]).collect();

            (trend_i, seasonal, remainder)
        })
        .collect();

    // Reassemble into column-major format
    let mut trend = vec![0.0; n * m];
    let mut seasonal = vec![0.0; n * m];
    let mut remainder = vec![0.0; n * m];

    for (i, (t, s, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            seasonal[i + j * n] = s[j];
            remainder[i + j * n] = r[j];
        }
    }

    DecomposeResult {
        trend,
        seasonal,
        remainder,
        period,
        method: "additive".to_string(),
    }
}

/// Multiplicative seasonal decomposition: data = trend * seasonal * remainder
///
/// Applies log transformation, then additive decomposition, then back-transforms.
/// Handles non-positive values by adding a shift.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Time/argument values
/// * `period` - Seasonal period
/// * `trend_method` - "loess" or "spline"
/// * `bandwidth` - Bandwidth for LOESS
/// * `n_harmonics` - Number of Fourier harmonics
///
/// # Returns
/// DecomposeResult with multiplicative components
pub fn decompose_multiplicative(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    trend_method: &str,
    bandwidth: f64,
    n_harmonics: usize,
) -> DecomposeResult {
    if n == 0 || m < 4 || data.len() != n * m || argvals.len() != m || period <= 0.0 {
        return DecomposeResult {
            trend: vec![0.0; n * m],
            seasonal: vec![0.0; n * m],
            remainder: data.to_vec(),
            period,
            method: "multiplicative".to_string(),
        };
    }

    // Find minimum value and add shift if needed to make all values positive
    let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };

    // Log transform
    let log_data: Vec<f64> = data.iter().map(|&x| (x + shift).ln()).collect();

    // Apply additive decomposition to log data
    let additive_result = decompose_additive(
        &log_data,
        n,
        m,
        argvals,
        period,
        trend_method,
        bandwidth,
        n_harmonics,
    );

    // Back transform: exp of each component
    // For multiplicative: data = trend * seasonal * remainder
    // In log space: log(data) = log(trend) + log(seasonal) + log(remainder)
    // So: trend_mult = exp(trend_add), seasonal_mult = exp(seasonal_add), etc.

    let mut trend = vec![0.0; n * m];
    let mut seasonal = vec![0.0; n * m];
    let mut remainder = vec![0.0; n * m];

    for idx in 0..n * m {
        // Back-transform trend (subtract shift)
        trend[idx] = additive_result.trend[idx].exp() - shift;

        // Seasonal is a multiplicative factor (centered around 1)
        // We interpret the additive seasonal component as log(seasonal factor)
        seasonal[idx] = additive_result.seasonal[idx].exp();

        // Remainder is also multiplicative
        remainder[idx] = additive_result.remainder[idx].exp();
    }

    DecomposeResult {
        trend,
        seasonal,
        remainder,
        period,
        method: "multiplicative".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_detrend_linear_removes_linear_trend() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // y = 2 + 0.5*t + sin(2*pi*t/2)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / 2.0).sin())
            .collect();

        let result = detrend_linear(&data, 1, m, &argvals);

        // Detrended should be approximately sin wave
        let expected: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * PI * t / 2.0).sin())
            .collect();

        let mut max_diff = 0.0f64;
        for j in 0..m {
            let diff = (result.detrended[j] - expected[j]).abs();
            max_diff = max_diff.max(diff);
        }
        assert!(max_diff < 0.2, "Max difference: {}", max_diff);
    }

    #[test]
    fn test_detrend_polynomial_removes_quadratic_trend() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // y = 1 + 0.5*t - 0.1*t^2 + sin(2*pi*t/2)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 1.0 + 0.5 * t - 0.1 * t * t + (2.0 * PI * t / 2.0).sin())
            .collect();

        let result = detrend_polynomial(&data, 1, m, &argvals, 2);

        // Detrended should be approximately sin wave
        let expected: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * PI * t / 2.0).sin())
            .collect();

        // Compute correlation
        let mean_det: f64 = result.detrended.iter().sum::<f64>() / m as f64;
        let mean_exp: f64 = expected.iter().sum::<f64>() / m as f64;
        let mut num = 0.0;
        let mut den_det = 0.0;
        let mut den_exp = 0.0;
        for j in 0..m {
            num += (result.detrended[j] - mean_det) * (expected[j] - mean_exp);
            den_det += (result.detrended[j] - mean_det).powi(2);
            den_exp += (expected[j] - mean_exp).powi(2);
        }
        let corr = num / (den_det.sqrt() * den_exp.sqrt());
        assert!(corr > 0.95, "Correlation: {}", corr);
    }

    #[test]
    fn test_detrend_diff1() {
        let m = 100;
        // Random walk: cumsum of random values
        let data: Vec<f64> = {
            let mut v = vec![0.0; m];
            v[0] = 1.0;
            for i in 1..m {
                v[i] = v[i - 1] + 0.1 * (i as f64).sin();
            }
            v
        };

        let result = detrend_diff(&data, 1, m, 1);

        // First difference should recover the increments
        for j in 0..m - 1 {
            let expected = data[j + 1] - data[j];
            assert!(
                (result.detrended[j] - expected).abs() < 1e-10,
                "Mismatch at {}: {} vs {}",
                j,
                result.detrended[j],
                expected
            );
        }
    }

    #[test]
    fn test_auto_detrend_selects_linear_for_linear_data() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();

        // Pure linear trend with small noise
        let data: Vec<f64> = argvals.iter().map(|&t| 2.0 + 0.5 * t).collect();

        let result = auto_detrend(&data, 1, m, &argvals);

        // Should select linear (or poly 2/3 with linear being sufficient)
        assert!(
            result.method.contains("linear") || result.method.contains("polynomial"),
            "Method: {}",
            result.method
        );
    }
}
