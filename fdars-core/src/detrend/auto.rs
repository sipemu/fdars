use std::borrow::Cow;

use crate::matrix::FdMatrix;

use super::{detrend_linear, detrend_loess, detrend_polynomial, TrendResult};

/// Automatically select the best detrending method using AIC.
#[must_use = "returns the detrending result which should not be discarded"]
pub fn auto_detrend(data: &FdMatrix, argvals: &[f64]) -> TrendResult {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m {
        return TrendResult::empty(data, n, m, Cow::Borrowed("auto(none)"), 0);
    }
    let compute_aic = |result: &TrendResult| -> f64 {
        let mut total_aic = 0.0;
        for i in 0..n {
            let rss = result.rss[i];
            let k = result.n_params as f64;
            let aic = if rss > 1e-15 {
                m as f64 * (rss / m as f64).ln() + 2.0 * k
            } else {
                f64::NEG_INFINITY
            };
            total_aic += aic;
        }
        total_aic / n as f64
    };
    let linear = detrend_linear(data, argvals);
    let poly2 = detrend_polynomial(data, argvals, 2);
    let poly3 = detrend_polynomial(data, argvals, 3);
    let loess = detrend_loess(data, argvals, 0.3, 2);
    let aic_linear = compute_aic(&linear);
    let aic_poly2 = compute_aic(&poly2);
    let aic_poly3 = compute_aic(&poly3);
    let aic_loess = compute_aic(&loess);
    let methods = [
        (aic_linear, "linear", linear),
        (aic_poly2, "polynomial(2)", poly2),
        (aic_poly3, "polynomial(3)", poly3),
        (aic_loess, "loess", loess),
    ];
    let (_, best_name, mut best_result) = methods
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .expect("non-empty iterator");
    best_result.method = Cow::Owned(format!("auto({best_name})"));
    best_result
}
