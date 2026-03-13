//! Elastic regression models (alignment-integrated regression).
//!
//! These models from fdasrvf align curves during the regression fitting process,
//! jointly optimizing alignment and regression coefficients.
//!
//! Key capabilities:
//! - [`elastic_regression`] — Scalar-on-function regression with elastic alignment
//! - [`elastic_logistic`] — Binary classification with elastic alignment
//! - [`elastic_pcr`] — Principal component regression after elastic alignment

pub mod logistic;
pub mod pcr;
pub mod regression;

#[cfg(test)]
mod tests;

// Re-export all public items
pub use logistic::{
    elastic_logistic, elastic_logistic_with_config, predict_elastic_logistic, ElasticLogisticResult,
};
pub use pcr::{elastic_pcr, elastic_pcr_with_config, ElasticPcrResult};
pub use regression::{
    elastic_regression, elastic_regression_with_config, predict_elastic_regression,
    ElasticRegressionResult,
};

use crate::alignment::reparameterize_curve;
use crate::matrix::FdMatrix;

// ─── Config Structs ─────────────────────────────────────────────────────────

/// Configuration for [`elastic_regression`] and [`elastic_logistic`].
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticConfig {
    /// Number of basis functions for the beta coefficient (for elastic_regression).
    pub ncomp_beta: usize,
    /// Roughness penalty weight.
    pub lambda: f64,
    /// Maximum iterations for iterative alignment.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            ncomp_beta: 10,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
        }
    }
}

/// Configuration for [`elastic_pcr`].
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticPcrConfig {
    /// Number of principal components to retain.
    pub ncomp: usize,
    /// PCA method (vertical, horizontal, or joint).
    pub pca_method: PcaMethod,
    /// Roughness penalty weight.
    pub lambda: f64,
    /// Maximum iterations for Karcher mean.
    pub max_iter: usize,
    /// Convergence tolerance for Karcher mean.
    pub tol: f64,
}

impl Default for ElasticPcrConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            pca_method: PcaMethod::Vertical,
            lambda: 0.0,
            max_iter: 20,
            tol: 1e-4,
        }
    }
}

// ─── Types ──────────────────────────────────────────────────────────────────

/// PCA method for elastic PCR.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PcaMethod {
    Vertical,
    Horizontal,
    Joint,
}

// ─── Shared Helpers ────────────────────────────────────────────────────────

/// Apply warping functions to SRSFs, producing aligned SRSFs with sqrt(γ') factor.
pub(super) fn apply_warps_to_srsfs(
    q_all: &FdMatrix,
    gammas: &FdMatrix,
    argvals: &[f64],
) -> FdMatrix {
    let (n, m) = q_all.shape();
    let h = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let mut q_aligned = FdMatrix::zeros(n, m);
    for i in 0..n {
        let qi: Vec<f64> = (0..m).map(|j| q_all[(i, j)]).collect();
        let gam: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let q_warped = reparameterize_curve(&qi, argvals, &gam);
        let gam_deriv = crate::helpers::gradient_uniform(&gam, h);
        for j in 0..m {
            q_aligned[(i, j)] = q_warped[j] * gam_deriv[j].max(0.0).sqrt();
        }
    }
    q_aligned
}

/// Initialize warping functions to identity (γ_i(t) = t).
pub(super) fn init_identity_warps(n: usize, argvals: &[f64]) -> FdMatrix {
    let m = argvals.len();
    let mut gammas = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            gammas[(i, j)] = argvals[j];
        }
    }
    gammas
}

/// Compute fitted values: ŷ_i = α + ∫ q_aligned_i · β · w dt.
pub(super) fn srsf_fitted_values(
    q_aligned: &FdMatrix,
    beta: &[f64],
    weights: &[f64],
    alpha: f64,
) -> Vec<f64> {
    let (n, m) = q_aligned.shape();
    let mut fitted = vec![0.0; n];
    for i in 0..n {
        fitted[i] = alpha;
        for j in 0..m {
            fitted[i] += q_aligned[(i, j)] * beta[j] * weights[j];
        }
    }
    fitted
}

/// Check relative convergence of β.
pub(super) fn beta_converged(beta_new: &[f64], beta_old: &[f64], tol: f64) -> bool {
    let diff: f64 = beta_new
        .iter()
        .zip(beta_old.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = beta_old
        .iter()
        .map(|&b| b * b)
        .sum::<f64>()
        .sqrt()
        .max(1e-10);
    diff / norm < tol
}
