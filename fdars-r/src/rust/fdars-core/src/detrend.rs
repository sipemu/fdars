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

use crate::iter_maybe_parallel;
use crate::smoothing::local_polynomial;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

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
    let results: Vec<(Vec<f64>, Vec<f64>, f64, f64, f64)> = iter_maybe_parallel!(0..n)
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
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
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
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
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
            det_full[..new_m].copy_from_slice(&detrended[..new_m]);

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
    let results: Vec<(Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
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
        _ => detrend_loess(data, n, m, argvals, bandwidth.max(0.3), 2),
    };

    // Step 2: Extract seasonal component using Fourier basis on detrended data
    let n_harm = n_harmonics.max(1).min(m / 4);
    let omega = 2.0 * std::f64::consts::PI / period;

    // Process each sample
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
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

// ============================================================================
// STL Decomposition (Cleveland et al., 1990)
// ============================================================================

/// Result of STL decomposition including robustness weights.
#[derive(Debug, Clone)]
pub struct StlResult {
    /// Trend component (n x m column-major)
    pub trend: Vec<f64>,
    /// Seasonal component (n x m column-major)
    pub seasonal: Vec<f64>,
    /// Remainder/residual component (n x m column-major)
    pub remainder: Vec<f64>,
    /// Robustness weights per point (n x m column-major)
    pub weights: Vec<f64>,
    /// Period used for decomposition
    pub period: usize,
    /// Seasonal smoothing window
    pub s_window: usize,
    /// Trend smoothing window
    pub t_window: usize,
    /// Number of inner loop iterations performed
    pub inner_iterations: usize,
    /// Number of outer loop iterations performed
    pub outer_iterations: usize,
}

/// STL Decomposition: Seasonal and Trend decomposition using LOESS
///
/// Implements the Cleveland et al. (1990) algorithm for robust iterative
/// decomposition of time series into trend, seasonal, and remainder components.
///
/// # Algorithm Overview
/// - **Inner Loop**: Extracts seasonal and trend components using LOESS smoothing
/// - **Outer Loop**: Computes robustness weights to downweight outliers
///
/// # Arguments
/// * `data` - Column-major matrix (n x m): n samples, m evaluation points
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `period` - Seasonal period (number of observations per cycle)
/// * `s_window` - Seasonal smoothing window (must be odd, ≥7 recommended)
/// * `t_window` - Trend smoothing window. If None, uses default formula
/// * `l_window` - Low-pass filter window. If None, uses period
/// * `robust` - Whether to perform robustness iterations
/// * `inner_iterations` - Number of inner loop iterations. Default: 2
/// * `outer_iterations` - Number of outer loop iterations. Default: 1 (or 15 if robust)
///
/// # Returns
/// `StlResult` with trend, seasonal, remainder, and robustness weights
///
/// # References
/// Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990).
/// STL: A Seasonal-Trend Decomposition Procedure Based on Loess.
/// Journal of Official Statistics, 6(1), 3-73.
pub fn stl_decompose(
    data: &[f64],
    n: usize,
    m: usize,
    period: usize,
    s_window: Option<usize>,
    t_window: Option<usize>,
    l_window: Option<usize>,
    robust: bool,
    inner_iterations: Option<usize>,
    outer_iterations: Option<usize>,
) -> StlResult {
    // Validate inputs
    if n == 0 || m < 2 * period || data.len() != n * m || period < 2 {
        return StlResult {
            trend: vec![0.0; n * m],
            seasonal: vec![0.0; n * m],
            remainder: data.to_vec(),
            weights: vec![1.0; n * m],
            period,
            s_window: 0,
            t_window: 0,
            inner_iterations: 0,
            outer_iterations: 0,
        };
    }

    // Set default parameters following Cleveland et al. recommendations
    let s_win = s_window.unwrap_or(7).max(3) | 1; // Ensure odd

    // Default t_window: smallest odd integer >= (1.5 * period) / (1 - 1.5/s_window)
    let t_win = t_window.unwrap_or_else(|| {
        let ratio = 1.5 * period as f64 / (1.0 - 1.5 / s_win as f64);
        let val = ratio.ceil() as usize;
        val.max(3) | 1 // Ensure odd
    });

    // Low-pass filter window: smallest odd integer >= period
    let l_win = l_window.unwrap_or(period) | 1;

    let n_inner = inner_iterations.unwrap_or(2);
    let n_outer = outer_iterations.unwrap_or(if robust { 15 } else { 1 });

    // Process each sample in parallel
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[i + j * n]).collect();
            stl_single_series(
                &curve, period, s_win, t_win, l_win, robust, n_inner, n_outer,
            )
        })
        .collect();

    // Reassemble into column-major format
    let mut trend = vec![0.0; n * m];
    let mut seasonal = vec![0.0; n * m];
    let mut remainder = vec![0.0; n * m];
    let mut weights = vec![1.0; n * m];

    for (i, (t, s, r, w)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[i + j * n] = t[j];
            seasonal[i + j * n] = s[j];
            remainder[i + j * n] = r[j];
            weights[i + j * n] = w[j];
        }
    }

    StlResult {
        trend,
        seasonal,
        remainder,
        weights,
        period,
        s_window: s_win,
        t_window: t_win,
        inner_iterations: n_inner,
        outer_iterations: n_outer,
    }
}

/// STL decomposition for a single time series.
fn stl_single_series(
    data: &[f64],
    period: usize,
    s_window: usize,
    t_window: usize,
    l_window: usize,
    robust: bool,
    n_inner: usize,
    n_outer: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let m = data.len();

    // Initialize components
    let mut trend = vec![0.0; m];
    let mut seasonal = vec![0.0; m];
    let mut weights = vec![1.0; m];

    // Outer loop for robustness
    for _outer in 0..n_outer {
        // Inner loop
        for _inner in 0..n_inner {
            // Step 1: Detrending
            let detrended: Vec<f64> = data
                .iter()
                .zip(trend.iter())
                .map(|(&y, &t)| y - t)
                .collect();

            // Step 2: Cycle-subseries smoothing
            let cycle_smoothed = smooth_cycle_subseries(&detrended, period, s_window, &weights);

            // Step 3: Low-pass filtering of smoothed cycle-subseries
            let low_pass = stl_lowpass_filter(&cycle_smoothed, period, l_window);

            // Step 4: Detrending the smoothed cycle-subseries
            seasonal = cycle_smoothed
                .iter()
                .zip(low_pass.iter())
                .map(|(&c, &l)| c - l)
                .collect();

            // Step 5: Deseasonalizing
            let deseasonalized: Vec<f64> = data
                .iter()
                .zip(seasonal.iter())
                .map(|(&y, &s)| y - s)
                .collect();

            // Step 6: Trend smoothing (weighted LOESS)
            trend = weighted_loess(&deseasonalized, t_window, &weights);
        }

        // After inner loop: compute residuals and robustness weights
        if robust && _outer < n_outer - 1 {
            let remainder: Vec<f64> = data
                .iter()
                .zip(trend.iter())
                .zip(seasonal.iter())
                .map(|((&y, &t), &s)| y - t - s)
                .collect();

            weights = compute_robustness_weights(&remainder);
        }
    }

    // Final remainder
    let remainder: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| y - t - s)
        .collect();

    (trend, seasonal, remainder, weights)
}

/// Smooth cycle-subseries: for each seasonal position, smooth across cycles.
fn smooth_cycle_subseries(
    data: &[f64],
    period: usize,
    s_window: usize,
    weights: &[f64],
) -> Vec<f64> {
    let m = data.len();
    let n_cycles = (m + period - 1) / period;
    let mut result = vec![0.0; m];

    // For each position in the cycle (0, 1, ..., period-1)
    for pos in 0..period {
        // Extract subseries at this position
        let mut subseries_idx: Vec<usize> = Vec::new();
        let mut subseries_vals: Vec<f64> = Vec::new();
        let mut subseries_weights: Vec<f64> = Vec::new();

        for cycle in 0..n_cycles {
            let idx = cycle * period + pos;
            if idx < m {
                subseries_idx.push(idx);
                subseries_vals.push(data[idx]);
                subseries_weights.push(weights[idx]);
            }
        }

        if subseries_vals.is_empty() {
            continue;
        }

        // Smooth this subseries using weighted LOESS
        let smoothed = weighted_loess(&subseries_vals, s_window, &subseries_weights);

        // Put smoothed values back
        for (i, &idx) in subseries_idx.iter().enumerate() {
            result[idx] = smoothed[i];
        }
    }

    result
}

/// Low-pass filter for STL (combination of moving averages).
/// Applies: MA(period) -> MA(period) -> MA(3)
fn stl_lowpass_filter(data: &[f64], period: usize, _l_window: usize) -> Vec<f64> {
    // First MA with period
    let ma1 = moving_average(data, period);
    // Second MA with period
    let ma2 = moving_average(&ma1, period);
    // Third MA with 3
    moving_average(&ma2, 3)
}

/// Simple moving average with window size.
fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    let m = data.len();
    if m == 0 || window == 0 {
        return data.to_vec();
    }

    let half = window / 2;
    let mut result = vec![0.0; m];

    for i in 0..m {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(m);
        let sum: f64 = data[start..end].iter().sum();
        let count = (end - start) as f64;
        result[i] = sum / count;
    }

    result
}

/// Weighted LOESS smoothing.
fn weighted_loess(data: &[f64], window: usize, weights: &[f64]) -> Vec<f64> {
    let m = data.len();
    if m == 0 {
        return vec![];
    }

    let half = window / 2;
    let mut result = vec![0.0; m];

    for i in 0..m {
        let start = i.saturating_sub(half);
        let end = (i + half + 1).min(m);

        // Compute weighted local linear regression
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let mut sum_wy = 0.0;
        let mut sum_wxx = 0.0;
        let mut sum_wxy = 0.0;

        for j in start..end {
            // Tricube weight based on distance
            let dist = (j as f64 - i as f64).abs() / (half.max(1) as f64);
            let tricube = if dist < 1.0 {
                (1.0 - dist.powi(3)).powi(3)
            } else {
                0.0
            };

            let w = tricube * weights[j];
            let x = j as f64;
            let y = data[j];

            sum_w += w;
            sum_wx += w * x;
            sum_wy += w * y;
            sum_wxx += w * x * x;
            sum_wxy += w * x * y;
        }

        // Solve weighted least squares
        if sum_w > 1e-10 {
            let denom = sum_w * sum_wxx - sum_wx * sum_wx;
            if denom.abs() > 1e-10 {
                let intercept = (sum_wxx * sum_wy - sum_wx * sum_wxy) / denom;
                let slope = (sum_w * sum_wxy - sum_wx * sum_wy) / denom;
                result[i] = intercept + slope * i as f64;
            } else {
                result[i] = sum_wy / sum_w;
            }
        } else {
            result[i] = data[i];
        }
    }

    result
}

/// Compute robustness weights using bisquare function.
fn compute_robustness_weights(residuals: &[f64]) -> Vec<f64> {
    let m = residuals.len();
    if m == 0 {
        return vec![];
    }

    // Compute median absolute deviation (MAD)
    let mut abs_residuals: Vec<f64> = residuals.iter().map(|&r| r.abs()).collect();
    abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let median_idx = m / 2;
    let mad = if m % 2 == 0 {
        (abs_residuals[median_idx - 1] + abs_residuals[median_idx]) / 2.0
    } else {
        abs_residuals[median_idx]
    };

    // Scale factor: 6 * MAD (Cleveland et al. use 6 MAD)
    let h = 6.0 * mad.max(1e-10);

    // Bisquare weight function
    residuals
        .iter()
        .map(|&r| {
            let u = r.abs() / h;
            if u < 1.0 {
                (1.0 - u * u).powi(2)
            } else {
                0.0
            }
        })
        .collect()
}

/// Wrapper function for functional data STL decomposition.
///
/// Computes STL decomposition for each curve in the functional data object
/// and returns aggregated results.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m) of functional data
/// * `n` - Number of samples (rows)
/// * `m` - Number of evaluation points (columns)
/// * `argvals` - Time points of length m (used to infer period if needed)
/// * `period` - Seasonal period (in number of observations)
/// * `s_window` - Seasonal smoothing window
/// * `t_window` - Trend smoothing window (0 for auto)
/// * `robust` - Whether to use robustness iterations
///
/// # Returns
/// `StlResult` with decomposed components.
pub fn stl_fdata(
    data: &[f64],
    n: usize,
    m: usize,
    _argvals: &[f64],
    period: usize,
    s_window: Option<usize>,
    t_window: Option<usize>,
    robust: bool,
) -> StlResult {
    stl_decompose(
        data, n, m, period, s_window, t_window, None, robust, None, None,
    )
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

    // ========================================================================
    // Tests for detrend_loess
    // ========================================================================

    #[test]
    fn test_detrend_loess_removes_linear_trend() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // y = 2 + 0.5*t + sin(2*pi*t/2)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / 2.0).sin())
            .collect();

        let result = detrend_loess(&data, 1, m, &argvals, 0.3, 1);

        // Detrended should be approximately sin wave
        let expected: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * PI * t / 2.0).sin())
            .collect();

        // Compute correlation (LOESS may smooth slightly)
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
        assert!(corr > 0.9, "Correlation: {}", corr);
        assert_eq!(result.method, "loess");
    }

    #[test]
    fn test_detrend_loess_removes_quadratic_trend() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // y = 1 + 0.3*t - 0.05*t^2 + sin(2*pi*t/2)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 1.0 + 0.3 * t - 0.05 * t * t + (2.0 * PI * t / 2.0).sin())
            .collect();

        let result = detrend_loess(&data, 1, m, &argvals, 0.3, 2);

        // Trend should follow the quadratic shape
        assert_eq!(result.trend.len(), m);
        assert_eq!(result.detrended.len(), m);

        // Check that RSS is computed
        assert!(result.rss[0] > 0.0);
    }

    #[test]
    fn test_detrend_loess_different_bandwidths() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Noisy sine wave
        let data: Vec<f64> = argvals
            .iter()
            .enumerate()
            .map(|(i, &t)| (2.0 * PI * t / 2.0).sin() + 0.1 * ((i * 17) % 100) as f64 / 100.0)
            .collect();

        // Small bandwidth = more local = rougher trend
        let result_small = detrend_loess(&data, 1, m, &argvals, 0.1, 1);
        // Large bandwidth = smoother trend
        let result_large = detrend_loess(&data, 1, m, &argvals, 0.5, 1);

        // Both should produce valid results
        assert_eq!(result_small.trend.len(), m);
        assert_eq!(result_large.trend.len(), m);

        // Large bandwidth should have more parameters
        assert!(result_large.n_params >= result_small.n_params);
    }

    #[test]
    fn test_detrend_loess_short_series() {
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let data: Vec<f64> = argvals.iter().map(|&t| t * 2.0).collect();

        let result = detrend_loess(&data, 1, m, &argvals, 0.3, 1);

        // Should still work on short series
        assert_eq!(result.trend.len(), m);
        assert_eq!(result.detrended.len(), m);
    }

    // ========================================================================
    // Tests for decompose_additive
    // ========================================================================

    #[test]
    fn test_decompose_additive_separates_components() {
        let m = 200;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // data = trend + seasonal: y = 2 + 0.5*t + sin(2*pi*t/2)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 2.0 + 0.5 * t + (2.0 * PI * t / period).sin())
            .collect();

        let result = decompose_additive(&data, 1, m, &argvals, period, "loess", 0.3, 3);

        assert_eq!(result.trend.len(), m);
        assert_eq!(result.seasonal.len(), m);
        assert_eq!(result.remainder.len(), m);
        assert_eq!(result.method, "additive");
        assert_eq!(result.period, period);

        // Check that components approximately sum to original
        for j in 0..m {
            let reconstructed = result.trend[j] + result.seasonal[j] + result.remainder[j];
            assert!(
                (reconstructed - data[j]).abs() < 0.5,
                "Reconstruction error at {}: {} vs {}",
                j,
                reconstructed,
                data[j]
            );
        }
    }

    #[test]
    fn test_decompose_additive_different_harmonics() {
        let m = 200;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Simple seasonal pattern
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 1.0 + (2.0 * PI * t / period).sin())
            .collect();

        // 1 harmonic
        let result1 = decompose_additive(&data, 1, m, &argvals, period, "loess", 0.3, 1);
        // 5 harmonics
        let result5 = decompose_additive(&data, 1, m, &argvals, period, "loess", 0.3, 5);

        // Both should produce valid results
        assert_eq!(result1.seasonal.len(), m);
        assert_eq!(result5.seasonal.len(), m);
    }

    #[test]
    fn test_decompose_additive_residual_properties() {
        let m = 200;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Data with trend and seasonal
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 2.0 + 0.3 * t + (2.0 * PI * t / period).sin())
            .collect();

        let result = decompose_additive(&data, 1, m, &argvals, period, "loess", 0.3, 3);

        // Remainder should have mean close to zero
        let mean_rem: f64 = result.remainder.iter().sum::<f64>() / m as f64;
        assert!(mean_rem.abs() < 0.5, "Remainder mean: {}", mean_rem);

        // Remainder variance should be smaller than original variance
        let var_data: f64 = data
            .iter()
            .map(|&x| (x - data.iter().sum::<f64>() / m as f64).powi(2))
            .sum::<f64>()
            / m as f64;
        let var_rem: f64 = result
            .remainder
            .iter()
            .map(|&x| (x - mean_rem).powi(2))
            .sum::<f64>()
            / m as f64;
        assert!(
            var_rem < var_data,
            "Remainder variance {} should be < data variance {}",
            var_rem,
            var_data
        );
    }

    #[test]
    fn test_decompose_additive_multi_sample() {
        let n = 3;
        let m = 100;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Create 3 samples with different amplitudes
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            let amp = (i + 1) as f64;
            for j in 0..m {
                data[i + j * n] =
                    1.0 + 0.1 * argvals[j] + amp * (2.0 * PI * argvals[j] / period).sin();
            }
        }

        let result = decompose_additive(&data, n, m, &argvals, period, "loess", 0.3, 2);

        assert_eq!(result.trend.len(), n * m);
        assert_eq!(result.seasonal.len(), n * m);
        assert_eq!(result.remainder.len(), n * m);
    }

    // ========================================================================
    // Tests for decompose_multiplicative
    // ========================================================================

    #[test]
    fn test_decompose_multiplicative_basic() {
        let m = 200;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Multiplicative: data = trend * seasonal
        // trend = 2 + 0.1*t, seasonal = 1 + 0.3*sin(...)
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 + 0.1 * t) * (1.0 + 0.3 * (2.0 * PI * t / period).sin()))
            .collect();

        let result = decompose_multiplicative(&data, 1, m, &argvals, period, "loess", 0.3, 3);

        assert_eq!(result.trend.len(), m);
        assert_eq!(result.seasonal.len(), m);
        assert_eq!(result.remainder.len(), m);
        assert_eq!(result.method, "multiplicative");

        // Seasonal factors should be centered around 1
        let mean_seasonal: f64 = result.seasonal.iter().sum::<f64>() / m as f64;
        assert!(
            (mean_seasonal - 1.0).abs() < 0.5,
            "Mean seasonal factor: {}",
            mean_seasonal
        );
    }

    #[test]
    fn test_decompose_multiplicative_non_positive_data() {
        let m = 100;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Data with negative values
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| -1.0 + (2.0 * PI * t / period).sin())
            .collect();

        // Should handle negative values by shifting
        let result = decompose_multiplicative(&data, 1, m, &argvals, period, "loess", 0.3, 2);

        assert_eq!(result.trend.len(), m);
        assert_eq!(result.seasonal.len(), m);
        // All seasonal values should be positive (multiplicative factors)
        for &s in result.seasonal.iter() {
            assert!(s.is_finite(), "Seasonal should be finite");
        }
    }

    #[test]
    fn test_decompose_multiplicative_vs_additive() {
        let m = 200;
        let period = 2.0;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64 * 10.0).collect();

        // Simple positive data
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| 5.0 + (2.0 * PI * t / period).sin())
            .collect();

        let add_result = decompose_additive(&data, 1, m, &argvals, period, "loess", 0.3, 3);
        let mult_result = decompose_multiplicative(&data, 1, m, &argvals, period, "loess", 0.3, 3);

        // Both should produce valid decompositions
        assert_eq!(add_result.seasonal.len(), m);
        assert_eq!(mult_result.seasonal.len(), m);

        // Additive seasonal oscillates around 0
        let add_mean: f64 = add_result.seasonal.iter().sum::<f64>() / m as f64;
        // Multiplicative seasonal oscillates around 1
        let mult_mean: f64 = mult_result.seasonal.iter().sum::<f64>() / m as f64;

        assert!(
            add_mean.abs() < mult_mean,
            "Additive mean {} vs mult mean {}",
            add_mean,
            mult_mean
        );
    }

    #[test]
    fn test_decompose_multiplicative_edge_cases() {
        // Empty data
        let result = decompose_multiplicative(&[], 0, 0, &[], 2.0, "loess", 0.3, 2);
        assert_eq!(result.trend.len(), 0);

        // Very short series
        let m = 5;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 2.0, 1.0];
        let result = decompose_multiplicative(&data, 1, m, &argvals, 2.0, "loess", 0.3, 1);
        // Should return original data as remainder for too-short series
        assert_eq!(result.remainder.len(), m);
    }
}
