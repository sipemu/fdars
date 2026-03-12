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

use crate::matrix::FdMatrix;
use std::borrow::Cow;

pub mod auto;
pub mod decompose;
pub mod diff;
pub mod linear;
pub mod loess;
pub mod polynomial;
pub mod stl;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Result of detrending operation.
#[derive(Debug, Clone)]
pub struct TrendResult {
    /// Estimated trend values (n x m)
    pub trend: FdMatrix,
    /// Detrended data (n x m)
    pub detrended: FdMatrix,
    /// Method used for detrending
    pub method: Cow<'static, str>,
    /// Polynomial coefficients (for polynomial methods, per sample)
    /// For n samples with polynomial degree d: n x (d+1)
    pub coefficients: Option<FdMatrix>,
    /// Residual sum of squares for each sample
    pub rss: Vec<f64>,
    /// Number of parameters (for AIC calculation)
    pub n_params: usize,
}

impl TrendResult {
    /// Construct a no-op TrendResult (zero trend, data copied to detrended).
    pub(super) fn empty(
        data: &FdMatrix,
        n: usize,
        m: usize,
        method: Cow<'static, str>,
        n_params: usize,
    ) -> Self {
        TrendResult {
            trend: FdMatrix::zeros(n, m),
            detrended: FdMatrix::from_slice(data.as_slice(), n, m)
                .unwrap_or_else(|_| FdMatrix::zeros(n, m)),
            method,
            coefficients: None,
            rss: vec![0.0; n],
            n_params,
        }
    }
}

/// Result of seasonal decomposition.
#[derive(Debug, Clone)]
pub struct DecomposeResult {
    /// Trend component (n x m)
    pub trend: FdMatrix,
    /// Seasonal component (n x m)
    pub seasonal: FdMatrix,
    /// Remainder/residual component (n x m)
    pub remainder: FdMatrix,
    /// Period used for decomposition
    pub period: f64,
    /// Decomposition method ("additive" or "multiplicative")
    pub method: Cow<'static, str>,
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Reassemble per-curve (trend, detrended, coefficients, rss) results into FdMatrix outputs.
pub(super) fn reassemble_polynomial_results(
    results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>, f64)>,
    n: usize,
    m: usize,
    n_coef: usize,
) -> (FdMatrix, FdMatrix, FdMatrix, Vec<f64>) {
    let mut trend = FdMatrix::zeros(n, m);
    let mut detrended = FdMatrix::zeros(n, m);
    let mut coefficients = FdMatrix::zeros(n, n_coef);
    let mut rss = vec![0.0; n];
    for (i, (t, d, coefs, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[(i, j)] = t[j];
            detrended[(i, j)] = d[j];
        }
        for k in 0..n_coef {
            coefficients[(i, k)] = coefs[k];
        }
        rss[i] = r;
    }
    (trend, detrended, coefficients, rss)
}

/// Reassemble per-curve (trend, detrended, rss) results into FdMatrix outputs.
pub(super) fn reassemble_trend_results(
    results: Vec<(Vec<f64>, Vec<f64>, f64)>,
    n: usize,
    m: usize,
) -> (FdMatrix, FdMatrix, Vec<f64>) {
    let mut trend = FdMatrix::zeros(n, m);
    let mut detrended = FdMatrix::zeros(n, m);
    let mut rss = vec![0.0; n];
    for (i, (t, d, r)) in results.into_iter().enumerate() {
        for j in 0..m {
            trend[(i, j)] = t[j];
            detrended[(i, j)] = d[j];
        }
        rss[i] = r;
    }
    (trend, detrended, rss)
}

// ---------------------------------------------------------------------------
// Re-exports -- preserves the external API
// ---------------------------------------------------------------------------

pub use auto::auto_detrend;
pub use decompose::{decompose_additive, decompose_multiplicative};
pub use diff::detrend_diff;
pub use linear::detrend_linear;
pub use loess::detrend_loess;
pub use polynomial::detrend_polynomial;
pub use stl::{stl_decompose, stl_fdata, StlResult};
