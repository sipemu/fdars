use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::borrow::Cow;

use super::TrendResult;

/// Remove linear trend from functional data using least squares.
///
/// # Arguments
/// * `data` - Matrix (n x m): n samples, m evaluation points
/// * `argvals` - Time/argument values of length m
///
/// # Returns
/// TrendResult with trend, detrended data, and coefficients (intercept, slope)
#[must_use = "returns the detrending result which should not be discarded"]
pub fn detrend_linear(data: &FdMatrix, argvals: &[f64]) -> TrendResult {
    let (n, m) = data.shape();
    if n == 0 || m < 2 || argvals.len() != m {
        return TrendResult::empty(data, n, m, Cow::Borrowed("linear"), 2);
    }

    let mean_t: f64 = argvals.iter().sum::<f64>() / m as f64;
    let ss_t: f64 = argvals.iter().map(|&t| (t - mean_t).powi(2)).sum();

    // Compute only scalar coefficients in parallel (no Vec allocations per sample)
    let scalars: Vec<(f64, f64, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut sum_y = 0.0;
            for j in 0..m {
                sum_y += data[(i, j)];
            }
            let mean_y = sum_y / m as f64;
            let mut sp = 0.0;
            for j in 0..m {
                sp += (argvals[j] - mean_t) * (data[(i, j)] - mean_y);
            }
            let slope = if ss_t.abs() > 1e-15 { sp / ss_t } else { 0.0 };
            let intercept = mean_y - slope * mean_t;
            let mut rss = 0.0;
            for j in 0..m {
                let residual = data[(i, j)] - (intercept + slope * argvals[j]);
                rss += residual * residual;
            }
            (intercept, slope, rss)
        })
        .collect();

    // Write directly into pre-allocated output matrices
    let mut trend = FdMatrix::zeros(n, m);
    let mut detrended = FdMatrix::zeros(n, m);
    let mut coefficients = FdMatrix::zeros(n, 2);
    let mut rss = vec![0.0; n];

    for (i, &(intercept, slope, r)) in scalars.iter().enumerate() {
        coefficients[(i, 0)] = intercept;
        coefficients[(i, 1)] = slope;
        rss[i] = r;
        for j in 0..m {
            let trend_val = intercept + slope * argvals[j];
            trend[(i, j)] = trend_val;
            detrended[(i, j)] = data[(i, j)] - trend_val;
        }
    }

    TrendResult {
        trend,
        detrended,
        method: Cow::Borrowed("linear"),
        coefficients: Some(coefficients),
        rss,
        n_params: 2,
    }
}
