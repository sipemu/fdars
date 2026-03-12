use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::borrow::Cow;

use super::{detrend_loess, DecomposeResult};

fn fit_fourier_seasonal(
    detrended_i: &[f64],
    argvals: &[f64],
    omega: f64,
    n_harm: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
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
    let y = DVector::from_row_slice(detrended_i);
    let svd = design.clone().svd(true, true);
    let coef = svd
        .solve(&y, 1e-10)
        .unwrap_or_else(|_| DVector::zeros(n_coef));
    let fitted = &design * &coef;
    let seasonal: Vec<f64> = fitted.iter().copied().collect();
    let remainder: Vec<f64> = (0..m).map(|j| detrended_i[j] - seasonal[j]).collect();
    (seasonal, remainder)
}

/// Additive seasonal decomposition: data = trend + seasonal + remainder
pub fn decompose_additive(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    trend_method: &str,
    bandwidth: f64,
    n_harmonics: usize,
) -> DecomposeResult {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return DecomposeResult {
            trend: FdMatrix::zeros(n, m),
            seasonal: FdMatrix::zeros(n, m),
            remainder: FdMatrix::from_slice(data.as_slice(), n, m)
                .unwrap_or_else(|_| FdMatrix::zeros(n, m)),
            period,
            method: Cow::Borrowed("additive"),
        };
    }
    let _ = trend_method;
    let trend_result = detrend_loess(data, argvals, bandwidth.max(0.3), 2);
    let n_harm = n_harmonics.max(1).min(m / 4);
    let omega = 2.0 * std::f64::consts::PI / period;
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let trend_i: Vec<f64> = (0..m).map(|j| trend_result.trend[(i, j)]).collect();
            let detrended_i: Vec<f64> = (0..m).map(|j| trend_result.detrended[(i, j)]).collect();
            let (seasonal, remainder) =
                fit_fourier_seasonal(&detrended_i, argvals, omega, n_harm, m);
            (trend_i, seasonal, remainder)
        })
        .collect();
    let mut trend = FdMatrix::zeros(n, m);
    let mut seasonal = FdMatrix::zeros(n, m);
    let mut remainder = FdMatrix::zeros(n, m);
    for (i, (t, s, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[(i, j)] = t[j];
            seasonal[(i, j)] = s[j];
            remainder[(i, j)] = r[j];
        }
    }
    DecomposeResult {
        trend,
        seasonal,
        remainder,
        period,
        method: Cow::Borrowed("additive"),
    }
}

/// Multiplicative seasonal decomposition: data = trend * seasonal * remainder
pub fn decompose_multiplicative(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    trend_method: &str,
    bandwidth: f64,
    n_harmonics: usize,
) -> DecomposeResult {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return DecomposeResult {
            trend: FdMatrix::zeros(n, m),
            seasonal: FdMatrix::zeros(n, m),
            remainder: FdMatrix::from_slice(data.as_slice(), n, m)
                .unwrap_or_else(|_| FdMatrix::zeros(n, m)),
            period,
            method: Cow::Borrowed("multiplicative"),
        };
    }
    let min_val = data
        .as_slice()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let shift = if min_val <= 0.0 { -min_val + 1.0 } else { 0.0 };
    let log_data_vec: Vec<f64> = data.as_slice().iter().map(|&x| (x + shift).ln()).collect();
    let log_data = FdMatrix::from_column_major(log_data_vec, n, m)
        .expect("dimension invariant: data.len() == n * m");
    let additive_result = decompose_additive(
        &log_data,
        argvals,
        period,
        trend_method,
        bandwidth,
        n_harmonics,
    );
    let mut trend = FdMatrix::zeros(n, m);
    let mut seasonal = FdMatrix::zeros(n, m);
    let mut remainder = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            trend[(i, j)] = additive_result.trend[(i, j)].exp() - shift;
            seasonal[(i, j)] = additive_result.seasonal[(i, j)].exp();
            remainder[(i, j)] = additive_result.remainder[(i, j)].exp();
        }
    }
    DecomposeResult {
        trend,
        seasonal,
        remainder,
        period,
        method: Cow::Borrowed("multiplicative"),
    }
}
