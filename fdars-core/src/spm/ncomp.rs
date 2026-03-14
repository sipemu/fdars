//! Automatic selection of the number of principal components (ncomp).
//!
//! Provides methods for choosing how many FPC scores to retain in
//! FPCA-based monitoring: cumulative variance, elbow detection, or
//! a fixed count.

use crate::error::FdarError;

/// Method for selecting the number of principal components.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NcompMethod {
    /// Retain components until cumulative variance >= threshold.
    CumulativeVariance(f64),
    /// Detect the elbow (max second finite difference) in the scree plot.
    Elbow,
    /// Use a fixed number of components.
    Fixed(usize),
}

/// Select the number of principal components from an eigenvalue spectrum.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues in decreasing order (e.g. from FPCA)
/// * `method` - Selection method
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `eigenvalues` is empty.
/// Returns [`FdarError::InvalidParameter`] if `CumulativeVariance` threshold
/// is not in (0, 1].
#[must_use = "selected ncomp should not be discarded"]
pub fn select_ncomp(eigenvalues: &[f64], method: &NcompMethod) -> Result<usize, FdarError> {
    if eigenvalues.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: "at least 1 eigenvalue".to_string(),
            actual: "0 eigenvalues".to_string(),
        });
    }

    match method {
        NcompMethod::CumulativeVariance(threshold) => {
            if *threshold <= 0.0 || *threshold > 1.0 {
                return Err(FdarError::InvalidParameter {
                    parameter: "threshold",
                    message: format!("threshold must be in (0, 1], got {threshold}"),
                });
            }
            cumulative_variance_ncomp(eigenvalues, *threshold)
        }
        NcompMethod::Elbow => {
            if eigenvalues.len() < 3 {
                // Fallback to CV(0.95) when too few values for elbow
                return cumulative_variance_ncomp(eigenvalues, 0.95);
            }
            Ok(elbow_ncomp(eigenvalues))
        }
        NcompMethod::Fixed(k) => Ok((*k).max(1).min(eigenvalues.len())),
    }
}

/// Cumulative variance method: first k where cumsum/total >= threshold.
fn cumulative_variance_ncomp(eigenvalues: &[f64], threshold: f64) -> Result<usize, FdarError> {
    let total: f64 = eigenvalues.iter().sum();
    if total <= 0.0 {
        return Err(FdarError::ComputationFailed {
            operation: "select_ncomp",
            detail: "total variance is non-positive".to_string(),
        });
    }

    let mut cumsum = 0.0;
    for (k, &ev) in eigenvalues.iter().enumerate() {
        cumsum += ev;
        if cumsum / total >= threshold {
            return Ok(k + 1);
        }
    }
    // If threshold is exactly 1.0, we need all components
    Ok(eigenvalues.len())
}

/// Elbow method: argmax of second finite difference d2[k] = λ[k-1] - 2λ[k] + λ[k+1].
fn elbow_ncomp(eigenvalues: &[f64]) -> usize {
    let n = eigenvalues.len();
    // d2[k] for k = 1..n-2 (using 0-indexed: k from 1 to n-2)
    let mut best_k = 1;
    let mut best_d2 = f64::NEG_INFINITY;

    for k in 1..n - 1 {
        let d2 = eigenvalues[k - 1] - 2.0 * eigenvalues[k] + eigenvalues[k + 1];
        if d2 > best_d2 {
            best_d2 = d2;
            best_k = k;
        }
    }
    // Return k+1 since best_k is 0-indexed and represents the elbow point
    // The number of components to retain is the index up to the elbow
    (best_k + 1).max(1).min(eigenvalues.len())
}
