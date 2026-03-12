use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector, Dyn, SVD};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::borrow::Cow;

use super::{reassemble_polynomial_results, TrendResult};

pub(super) fn build_vandermonde_matrix(t_norm: &[f64], m: usize, n_coef: usize) -> DMatrix<f64> {
    let mut design = DMatrix::zeros(m, n_coef);
    for j in 0..m {
        let t = t_norm[j];
        let mut power = 1.0;
        for k in 0..n_coef {
            design[(j, k)] = power;
            power *= t;
        }
    }
    design
}

pub(super) fn fit_polynomial_single_curve(
    curve: &[f64],
    svd: &SVD<f64, Dyn, Dyn>,
    design: &DMatrix<f64>,
    n_coef: usize,
    m: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let y = DVector::from_row_slice(curve);
    let beta = svd
        .solve(&y, 1e-10)
        .unwrap_or_else(|_| DVector::zeros(n_coef));
    let fitted = design * &beta;
    let mut trend = vec![0.0; m];
    let mut detrended = vec![0.0; m];
    let mut rss = 0.0;
    for j in 0..m {
        trend[j] = fitted[j];
        detrended[j] = curve[j] - fitted[j];
        rss += detrended[j].powi(2);
    }
    let coefs: Vec<f64> = beta.iter().copied().collect();
    (trend, detrended, coefs, rss)
}

/// Remove polynomial trend from functional data using QR decomposition.
#[must_use = "returns the detrending result which should not be discarded"]
pub fn detrend_polynomial(data: &FdMatrix, argvals: &[f64], degree: usize) -> TrendResult {
    let (n, m) = data.shape();
    if n == 0 || m < degree + 1 || argvals.len() != m || degree == 0 {
        return TrendResult::empty(
            data,
            n,
            m,
            Cow::Owned(format!("polynomial({degree})")),
            degree + 1,
        );
    }
    if degree == 1 {
        let mut result = super::linear::detrend_linear(data, argvals);
        result.method = Cow::Borrowed("polynomial(1)");
        return result;
    }
    let n_coef = degree + 1;
    let t_min = argvals.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = argvals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t_range = if (t_max - t_min).abs() > 1e-15 {
        t_max - t_min
    } else {
        1.0
    };
    let t_norm: Vec<f64> = argvals.iter().map(|&t| (t - t_min) / t_range).collect();
    let design = build_vandermonde_matrix(&t_norm, m, n_coef);
    let svd = design.clone().svd(true, true);
    let results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            fit_polynomial_single_curve(&curve, &svd, &design, n_coef, m)
        })
        .collect();
    let (trend, detrended, coefficients, rss) =
        reassemble_polynomial_results(results, n, m, n_coef);
    TrendResult {
        trend,
        detrended,
        method: Cow::Owned(format!("polynomial({degree})")),
        coefficients: Some(coefficients),
        rss,
        n_params: n_coef,
    }
}
