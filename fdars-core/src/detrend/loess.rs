use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use crate::smoothing::local_polynomial;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::borrow::Cow;

use super::{reassemble_trend_results, TrendResult};

/// Remove trend using LOESS (local polynomial regression).
pub fn detrend_loess(
    data: &FdMatrix,
    argvals: &[f64],
    bandwidth: f64,
    degree: usize,
) -> TrendResult {
    let (n, m) = data.shape();
    if n == 0 || m < 3 || argvals.len() != m || bandwidth <= 0.0 {
        return TrendResult::empty(
            data,
            n,
            m,
            Cow::Borrowed("loess"),
            (m as f64 * bandwidth.max(0.0)).ceil() as usize,
        );
    }
    let t_min = argvals.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = argvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let abs_bandwidth = (t_max - t_min) * bandwidth;
    let results: Vec<(Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            let trend =
                local_polynomial(argvals, &curve, argvals, abs_bandwidth, degree, "tricube");
            let mut detrended = vec![0.0; m];
            let mut rss = 0.0;
            for j in 0..m {
                detrended[j] = curve[j] - trend[j];
                rss += detrended[j].powi(2);
            }
            (trend, detrended, rss)
        })
        .collect();
    let (trend, detrended, rss) = reassemble_trend_results(results, n, m);
    let n_params = (m as f64 * bandwidth).ceil() as usize;
    TrendResult {
        trend,
        detrended,
        method: Cow::Borrowed("loess"),
        coefficients: None,
        rss,
        n_params,
    }
}
