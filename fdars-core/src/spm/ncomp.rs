//! Automatic selection of the number of principal components (ncomp).
//!
//! Provides methods for choosing how many FPC scores to retain in
//! FPCA-based monitoring: cumulative variance, elbow detection, or
//! a fixed count.
//!
//! # References
//!
//! - Cattell, R.B. (1966). The scree test for the number of factors.
//!   *Multivariate Behavioral Research*, 1(2), 245-276.
//! - Hall, P. & Horowitz, J.L. (2007). Methodology and convergence rates
//!   for functional linear regression. *Annals of Statistics*, 35(1), 70-91.
//! - Kaiser, H.F. (1960). The application of electronic computers to factor
//!   analysis. *Educational and Psychological Measurement*, 20, 141-151.

use crate::error::FdarError;

/// Method for selecting the number of principal components.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum NcompMethod {
    /// Retain components until cumulative variance >= threshold.
    ///
    /// Threshold 0.95 retains 95% of variance, which is a widely-used default
    /// (Jackson, 1991). For monitoring applications, 0.90 may suffice as the
    /// remaining variance contributes to SPE rather than T-squared.
    CumulativeVariance(f64),
    /// Detect the elbow (max second finite difference) in the scree plot.
    ///
    /// The elbow method is sensitive to noise in the eigenvalue spectrum. The
    /// internal 3-point moving average smoothing reduces this sensitivity.
    /// Falls back to `CumulativeVariance(0.95)` when fewer than 3 eigenvalues
    /// are available.
    Elbow,
    /// Use a fixed number of components.
    Fixed(usize),
    /// Retain eigenvalues above the mean (Kaiser criterion variant).
    Kaiser,
}

/// Select the number of principal components from an eigenvalue spectrum.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues in decreasing order (e.g. from FPCA)
/// * `method` - Selection method
///
/// # Example
///
/// ```
/// use fdars_core::spm::ncomp::{select_ncomp, NcompMethod};
/// let eigenvalues = vec![10.0, 5.0, 1.0, 0.1, 0.01];
/// let k = select_ncomp(&eigenvalues, &NcompMethod::CumulativeVariance(0.95)).unwrap();
/// assert!(k >= 2 && k <= 4);
/// let k_fixed = select_ncomp(&eigenvalues, &NcompMethod::Fixed(3)).unwrap();
/// assert_eq!(k_fixed, 3);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `eigenvalues` is empty.
/// Returns [`FdarError::InvalidParameter`] if any eigenvalue is NaN or
/// negative, or if `CumulativeVariance` threshold is not in (0, 1].
#[must_use = "selected ncomp should not be discarded"]
pub fn select_ncomp(eigenvalues: &[f64], method: &NcompMethod) -> Result<usize, FdarError> {
    if eigenvalues.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: "at least 1 eigenvalue".to_string(),
            actual: "0 eigenvalues".to_string(),
        });
    }

    // Validate eigenvalues: NaN or negative values indicate upstream issues
    // (e.g., numerical instability in FPCA or corrupted data).
    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev.is_nan() {
            return Err(FdarError::InvalidParameter {
                parameter: "eigenvalues",
                message: format!("eigenvalue[{i}] is NaN"),
            });
        }
        if ev < 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "eigenvalues",
                message: format!("eigenvalue[{i}] = {ev} is negative"),
            });
        }
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
        NcompMethod::Kaiser => {
            if eigenvalues.len() < 2 {
                return Ok(1); // With 1 eigenvalue, always keep it
            }
            Ok(kaiser_ncomp(eigenvalues))
        }
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

/// Kaiser criterion: retain components with eigenvalue > mean(eigenvalues).
fn kaiser_ncomp(eigenvalues: &[f64]) -> usize {
    let mean = eigenvalues.iter().sum::<f64>() / eigenvalues.len() as f64;
    let k = eigenvalues.iter().filter(|&&ev| ev > mean).count();
    k.max(1) // Always retain at least 1
}

/// Elbow method: argmax of second finite difference d2[k] = λ[k-1] - 2λ[k] + λ[k+1].
///
/// Applies a 3-point moving average to reduce noise sensitivity before
/// computing the second differences (Cattell, 1966).
fn elbow_ncomp(eigenvalues: &[f64]) -> usize {
    let n = eigenvalues.len();
    // 3-point moving average smoothing (edges use 2-point)
    let smoothed: Vec<f64> = (0..n)
        .map(|i| {
            if i == 0 {
                (eigenvalues[0] + eigenvalues[1]) / 2.0
            } else if i == n - 1 {
                (eigenvalues[n - 2] + eigenvalues[n - 1]) / 2.0
            } else {
                (eigenvalues[i - 1] + eigenvalues[i] + eigenvalues[i + 1]) / 3.0
            }
        })
        .collect();

    // d2[k] for k = 1..n-2
    let mut best_k = 1;
    let mut best_d2 = f64::NEG_INFINITY;

    for k in 1..n - 1 {
        let d2 = smoothed[k - 1] - 2.0 * smoothed[k] + smoothed[k + 1];
        if d2 > best_d2 {
            best_d2 = d2;
            best_k = k;
        }
    }
    (best_k + 1).max(1).min(eigenvalues.len())
}
