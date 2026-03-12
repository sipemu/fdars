//! Model-based functional clustering via Gaussian mixture models.
//!
//! Implements the fdaMocca approach (Arnqvist & Sjöstedt de Luna, 2023):
//! project curves onto a basis, concatenate with scalar covariates, and fit
//! a Gaussian mixture using EM.
//!
//! Key functions:
//! - [`gmm_cluster`] — Main clustering entry point with automatic K selection
//! - [`gmm_em`] — Single-K EM algorithm
//! - [`predict_gmm`] — Assign new observations to clusters

use crate::matrix::FdMatrix;

pub mod cluster;
pub mod covariance;
pub mod em;
pub mod init;
pub mod linalg;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

/// Covariance structure for GMM components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CovType {
    /// Full covariance matrix (d² parameters per component)
    Full,
    /// Diagonal covariance (d parameters per component)
    Diagonal,
}

/// Result from a single GMM fit with fixed K.
pub struct GmmResult {
    /// Hard cluster assignments (length n)
    pub cluster: Vec<usize>,
    /// Posterior membership probabilities (n x K)
    pub membership: FdMatrix,
    /// Component means (K x d)
    pub means: Vec<Vec<f64>>,
    /// Component covariances: for Full, each is d×d flattened; for Diagonal, each is length d
    pub covariances: Vec<Vec<f64>>,
    /// Mixing proportions (length K)
    pub weights: Vec<f64>,
    /// Log-likelihood at convergence
    pub log_likelihood: f64,
    /// BIC value
    pub bic: f64,
    /// ICL value (BIC penalized by entropy)
    pub icl: f64,
    /// Number of EM iterations
    pub iterations: usize,
    /// Whether EM converged
    pub converged: bool,
    /// Number of clusters
    pub k: usize,
    /// Feature dimension (basis coefficients + covariates)
    pub d: usize,
}

/// Result from automatic K selection.
pub struct GmmClusterResult {
    /// Best GMM result (by BIC or ICL)
    pub best: GmmResult,
    /// BIC values for each K tried
    pub bic_values: Vec<(usize, f64)>,
    /// ICL values for each K tried
    pub icl_values: Vec<(usize, f64)>,
}

// Re-export all public items
pub use cluster::{gmm_cluster, predict_gmm};
pub use em::gmm_em;
