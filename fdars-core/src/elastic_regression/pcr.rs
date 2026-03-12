//! Elastic principal component regression.

use crate::alignment::{karcher_mean, KarcherMeanResult};
use crate::elastic_fpca::{
    horiz_fpca, joint_fpca, vert_fpca, HorizFpcaResult, JointFpcaResult, VertFpcaResult,
};
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};

use super::{ElasticPcrConfig, PcaMethod};

/// Result of elastic principal component regression.
#[derive(Debug, Clone)]
pub struct ElasticPcrResult {
    /// Intercept.
    pub alpha: f64,
    /// Regression coefficients on PC scores, length ncomp.
    pub coefficients: Vec<f64>,
    /// Fitted values, length n.
    pub fitted_values: Vec<f64>,
    /// Residual sum of squares.
    pub sse: f64,
    /// Coefficient of determination.
    pub r_squared: f64,
    /// PCA method used.
    pub pca_method: PcaMethod,
    /// Karcher mean result.
    pub karcher: KarcherMeanResult,
    /// Vertical FPCA result (stored when method is Vertical or Joint).
    pub vert_fpca: Option<VertFpcaResult>,
    /// Horizontal FPCA result (stored when method is Horizontal or Joint).
    pub horiz_fpca: Option<HorizFpcaResult>,
    /// Joint FPCA result (stored when method is Joint).
    pub joint_fpca: Option<JointFpcaResult>,
}

/// Elastic principal component regression.
///
/// Performs Karcher mean alignment, then FPCA (vert/horiz/joint), then OLS
/// on the PC scores.
///
/// # Arguments
/// * `data` — Functional data (n × m)
/// * `y` — Scalar responses (length n)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of PCs to use
/// * `pca_method` — Which FPCA variant to use
/// * `lambda` — Alignment penalty (passed to karcher_mean)
/// * `max_iter` — Maximum iterations for karcher_mean
/// * `tol` — Convergence tolerance for karcher_mean
///
/// # Errors
///
/// Returns [`crate::FdarError::InvalidDimension`] if `n < 2`, `y.len() != n`, or
/// `ncomp < 1`.
/// Returns [`crate::FdarError::ComputationFailed`] if the elastic FPCA step or OLS
/// on PC scores fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pcr(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    ncomp: usize,
    pca_method: PcaMethod,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ElasticPcrResult, crate::FdarError> {
    let (n, _m) = data.shape();
    if n < 2 || y.len() != n || ncomp < 1 {
        return Err(crate::FdarError::InvalidDimension {
            parameter: "data/y",
            expected: "n >= 2, y.len() == n, ncomp >= 1".to_string(),
            actual: format!("n={}, y.len()={}, ncomp={}", n, y.len(), ncomp),
        });
    }

    // Karcher mean alignment
    let km = karcher_mean(data, argvals, max_iter, tol, lambda);

    // FPCA
    let mut stored_vert: Option<VertFpcaResult> = None;
    let mut stored_horiz: Option<HorizFpcaResult> = None;
    let mut stored_joint: Option<JointFpcaResult> = None;

    let scores_mat = match pca_method {
        PcaMethod::Vertical => {
            let fpca = vert_fpca(&km, argvals, ncomp)?;
            let scores = fpca.scores.clone();
            stored_vert = Some(fpca);
            scores
        }
        PcaMethod::Horizontal => {
            let fpca = horiz_fpca(&km, argvals, ncomp)?;
            let scores = fpca.scores.clone();
            stored_horiz = Some(fpca);
            scores
        }
        PcaMethod::Joint => {
            let fpca = joint_fpca(&km, argvals, ncomp, None)?;
            let scores = fpca.scores.clone();
            stored_joint = Some(fpca);
            scores
        }
    };

    let actual_ncomp = scores_mat.ncols();
    let (coefs, alpha, fitted_values, sse, r_squared) =
        ols_on_scores(&scores_mat, y, n, actual_ncomp).ok_or_else(|| {
            crate::FdarError::ComputationFailed {
                operation: "OLS",
                detail: "OLS on PC scores failed".to_string(),
            }
        })?;

    Ok(ElasticPcrResult {
        alpha,
        coefficients: coefs,
        fitted_values,
        sse,
        r_squared,
        pca_method,
        karcher: km,
        vert_fpca: stored_vert,
        horiz_fpca: stored_horiz,
        joint_fpca: stored_joint,
    })
}

/// Elastic PCR using a configuration struct.
///
/// Equivalent to [`elastic_pcr`] but bundles method parameters in [`ElasticPcrConfig`].
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_pcr_with_config(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &ElasticPcrConfig,
) -> Result<ElasticPcrResult, crate::FdarError> {
    elastic_pcr(
        data,
        y,
        argvals,
        config.ncomp,
        config.pca_method,
        config.lambda,
        config.max_iter,
        config.tol,
    )
}

// ─── Internal helpers ───────────────────────────────────────────────────────

/// OLS regression on PC scores: returns (coefs, alpha, fitted, sse, r²).
fn ols_on_scores(
    scores_mat: &FdMatrix,
    y: &[f64],
    n: usize,
    ncomp: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, f64, f64)> {
    let y_mean: f64 = y.iter().sum::<f64>() / n as f64;
    let mut score_means = vec![0.0; ncomp];
    for k in 0..ncomp {
        for i in 0..n {
            score_means[k] += scores_mat[(i, k)];
        }
        score_means[k] /= n as f64;
    }

    let mut x_cen = DMatrix::zeros(n, ncomp);
    for i in 0..n {
        for k in 0..ncomp {
            x_cen[(i, k)] = scores_mat[(i, k)] - score_means[k];
        }
    }
    let y_cen: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let y_vec = DVector::from_vec(y_cen);

    let xtx = x_cen.transpose() * &x_cen;
    let xty = x_cen.transpose() * &y_vec;
    let coefficients = if let Some(chol) = xtx.clone().cholesky() {
        chol.solve(&xty)
    } else {
        let svd = nalgebra::SVD::new(xtx, true, true);
        svd.solve(&xty, 1e-10).ok()?
    };
    let coefs: Vec<f64> = coefficients.iter().copied().collect();

    let alpha = y_mean
        - coefs
            .iter()
            .zip(score_means.iter())
            .map(|(&c, &sm)| c * sm)
            .sum::<f64>();

    let mut fitted_values = vec![0.0; n];
    for i in 0..n {
        fitted_values[i] = alpha;
        for k in 0..ncomp {
            fitted_values[i] += coefs[k] * scores_mat[(i, k)];
        }
    }

    let sse: f64 = y
        .iter()
        .zip(fitted_values.iter())
        .map(|(&yi, &yh)| (yi - yh).powi(2))
        .sum();
    let ss_tot: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = if ss_tot > 0.0 {
        1.0 - sse / ss_tot
    } else {
        0.0
    };

    Some((coefs, alpha, fitted_values, sse, r_squared))
}
