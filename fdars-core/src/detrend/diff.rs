use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::borrow::Cow;

use super::{reassemble_polynomial_results, TrendResult};

pub(super) fn diff_single_curve(
    curve: &[f64],
    m: usize,
    order: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let diff1: Vec<f64> = (0..m - 1).map(|j| curve[j + 1] - curve[j]).collect();
    let detrended = if order == 2 {
        (0..diff1.len() - 1)
            .map(|j| diff1[j + 1] - diff1[j])
            .collect()
    } else {
        diff1.clone()
    };
    let initial_values = if order == 2 {
        vec![curve[0], curve[1]]
    } else {
        vec![curve[0]]
    };
    let rss: f64 = detrended.iter().map(|&x| x.powi(2)).sum();
    let new_m = m - order;
    let mut trend = vec![0.0; m];
    trend[0] = curve[0];
    if order == 1 {
        for j in 1..m {
            trend[j] = curve[j] - if j <= new_m { detrended[j - 1] } else { 0.0 };
        }
    } else {
        trend = curve.to_vec();
    }
    let mut det_full = vec![0.0; m];
    det_full[..new_m].copy_from_slice(&detrended[..new_m]);
    (trend, det_full, initial_values, rss)
}

/// Remove trend by differencing.
#[must_use = "returns the detrending result which should not be discarded"]
pub fn detrend_diff(data: &FdMatrix, order: usize) -> TrendResult {
    let (n, m) = data.shape();
    if n == 0 || m <= order || order == 0 || order > 2 {
        return TrendResult::empty(data, n, m, Cow::Owned(format!("diff{order}")), order);
    }
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            diff_single_curve(&curve, m, order)
        })
        .collect();
    let (trend, detrended, coefficients, rss) = reassemble_polynomial_results(results, n, m, order);
    TrendResult {
        trend,
        detrended,
        method: Cow::Owned(format!("diff{order}")),
        coefficients: Some(coefficients),
        rss,
        n_params: order,
    }
}
