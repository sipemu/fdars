//! Shared helper functions for basis computations.

use nalgebra::{DMatrix, SVD};

/// Compute pseudoinverse of a symmetric matrix via SVD.
///
/// Uses singular value decomposition with threshold-based truncation
/// for numerical stability with near-singular matrices.
pub(super) fn svd_pseudoinverse(mat: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    let n = mat.nrows();
    let svd = SVD::new(mat.clone(), true, true);
    let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let eps = 1e-10 * max_sv;

    let u = svd.u.as_ref()?;
    let v_t = svd.v_t.as_ref()?;

    let s_inv: Vec<f64> = svd
        .singular_values
        .iter()
        .map(|&s| if s > eps { 1.0 / s } else { 0.0 })
        .collect();

    let mut result = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n.min(s_inv.len()) {
                sum += v_t[(k, i)] * s_inv[k] * u[(j, k)];
            }
            result[(i, j)] = sum;
        }
    }

    Some(result)
}

/// Compute model selection criterion (GCV, AIC, or BIC).
///
/// # Arguments
/// * `rss` - Residual sum of squares
/// * `n_points` - Number of data points
/// * `edf` - Effective degrees of freedom
/// * `criterion` - 0=GCV, 1=AIC, 2=BIC
pub(super) fn compute_model_criterion(rss: f64, n_points: f64, edf: f64, criterion: i32) -> f64 {
    match criterion {
        0 => {
            let gcv_denom = 1.0 - edf / n_points;
            if gcv_denom.abs() > 1e-10 {
                (rss / n_points) / (gcv_denom * gcv_denom)
            } else {
                f64::INFINITY
            }
        }
        1 => {
            let mse = rss / n_points;
            n_points * mse.ln() + 2.0 * edf
        }
        _ => {
            let mse = rss / n_points;
            n_points * mse.ln() + n_points.ln() * edf
        }
    }
}

/// Compute GCV, AIC, and BIC model selection criteria.
pub(super) fn compute_fit_criteria(
    total_rss: f64,
    total_points: f64,
    edf: f64,
    m: usize,
) -> (f64, f64, f64) {
    let gcv_denom = 1.0 - edf / m as f64;
    let gcv = if gcv_denom.abs() > 1e-10 {
        (total_rss / total_points) / (gcv_denom * gcv_denom)
    } else {
        f64::INFINITY
    };
    let mse = total_rss / total_points;
    let aic = total_points * mse.ln() + 2.0 * edf;
    let bic = total_points * mse.ln() + total_points.ln() * edf;
    (gcv, aic, bic)
}
