//! Hotelling T-squared and SPE (Squared Prediction Error) statistics.
//!
//! These are the two fundamental monitoring statistics for functional
//! statistical process control: T-squared captures systematic variation
//! in the principal component subspace, while SPE captures residual
//! variation outside it.
//!
//! # References
//!
//! - Hotelling, H. (1947). Multivariate quality control. *Techniques of
//!   Statistical Analysis*, 111-184.
//! - Bersimis, S., Psarakis, S. & Panaretos, J. (2007). Multivariate
//!   statistical process control charts: an overview. *Quality and
//!   Reliability Engineering International*, 23(5), 517-543.
//!
//! # Assumptions
//!
//! The statistics in this module assume the FPCA scores are approximately
//! uncorrelated (diagonal covariance). This holds by construction when
//! eigenvalues come from PCA/SVD. For non-PCA score vectors, use the full
//! covariance Mahalanobis distance instead.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

/// Compute Hotelling T-squared statistic for each observation.
///
/// T-squared_i = sum_{l=1}^{L} scores_{i,l}^2 / eigenvalues_l
///
/// Under in-control conditions with Gaussian scores, T² follows
/// approximately a chi²(ncomp) distribution. Use
/// [`t2_control_limit`](super::control::t2_control_limit) to obtain the
/// corresponding upper control limit.
///
/// # Arguments
/// * `scores` - FPC score matrix (n x ncomp)
/// * `eigenvalues` - Eigenvalues (length ncomp): sv_l^2 / (n-1)
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::stats::hotelling_t2;
/// let scores = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 0.5, 1.0, 1.5], 3, 2).unwrap();
/// let eigenvalues = vec![2.0, 1.0];
/// let t2 = hotelling_t2(&scores, &eigenvalues).unwrap();
/// assert_eq!(t2.len(), 3);
/// assert!((t2[0] - 0.75).abs() < 1e-10); // 1²/2 + 0.5²/1
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if scores columns != eigenvalues length.
/// Returns [`FdarError::InvalidParameter`] if any eigenvalue is non-positive.
pub fn hotelling_t2(scores: &FdMatrix, eigenvalues: &[f64]) -> Result<Vec<f64>, FdarError> {
    let (n, ncomp) = scores.shape();
    if ncomp != eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: format!("{ncomp} (matching scores columns)"),
            actual: format!("{}", eigenvalues.len()),
        });
    }
    for (l, &ev) in eigenvalues.iter().enumerate() {
        if ev <= 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "eigenvalues",
                message: format!("eigenvalue[{l}] = {ev} must be positive"),
            });
        }
    }

    let t2: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for l in 0..ncomp {
                let s = scores[(i, l)];
                sum += s * s / eigenvalues[l];
            }
            sum
        })
        .collect();

    Ok(t2)
}

/// Compute Hotelling T-squared with eigenvalue regularization.
///
/// Like [`hotelling_t2`], but applies a floor to eigenvalues to prevent
/// numerical instability from near-zero values. Useful when some
/// eigenvalues are very small (< 1e-8) which would cause numerical
/// instability in standard T². The epsilon floor prevents division by
/// near-zero eigenvalues.
///
/// Choosing epsilon: set epsilon approximately 1e-2 times the minimum
/// eigenvalue to regularize without substantially altering the statistic.
/// The default regularization in `spm_monitor` uses this approach. For
/// manual use, epsilon = 1e-6 is a safe general-purpose default that
/// handles eigenvalue ratios up to 1e6.
///
/// # Arguments
/// * `scores` - Score matrix (n × ncomp)
/// * `eigenvalues` - Eigenvalues (length ncomp)
/// * `epsilon` - Minimum eigenvalue floor (values below this are set to epsilon)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shapes are inconsistent.
/// Returns [`FdarError::InvalidParameter`] if epsilon is non-positive.
pub fn hotelling_t2_regularized(
    scores: &FdMatrix,
    eigenvalues: &[f64],
    epsilon: f64,
) -> Result<Vec<f64>, FdarError> {
    if epsilon <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "epsilon",
            message: format!("epsilon must be positive, got {epsilon}"),
        });
    }
    let (n, ncomp) = scores.shape();
    if ncomp != eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: format!("{ncomp}"),
            actual: format!("{}", eigenvalues.len()),
        });
    }

    let mut t2 = vec![0.0; n];
    for i in 0..n {
        for l in 0..ncomp {
            let ev = eigenvalues[l].max(epsilon);
            t2[i] += scores[(i, l)] * scores[(i, l)] / ev;
        }
    }
    Ok(t2)
}

/// Compute SPE (Squared Prediction Error) for univariate functional data.
///
/// SPE_i = integral (x_centered_i(t) - x_reconstructed_i(t))^2 w(t) dt
///
/// Requires `argvals` to be sorted in ascending order with at least 3
/// points for Simpson's rule integration. Non-uniform spacing is handled
/// correctly by the quadrature weights.
///
/// # Arguments
/// * `centered` - Centered functional data (n x m)
/// * `reconstructed` - Reconstructed data from FPCA (n x m), already centered (mean subtracted)
/// * `argvals` - Grid points (length m)
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::stats::spe_univariate;
/// let centered = FdMatrix::from_column_major(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
/// let reconstructed = FdMatrix::from_column_major(vec![0.0; 4], 2, 2).unwrap();
/// let argvals = vec![0.0, 1.0];
/// let spe = spe_univariate(&centered, &reconstructed, &argvals).unwrap();
/// assert_eq!(spe.len(), 2);
/// assert!(spe[0] > 0.0);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shapes do not match.
pub fn spe_univariate(
    centered: &FdMatrix,
    reconstructed: &FdMatrix,
    argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let (n, m) = centered.shape();
    if reconstructed.shape() != (n, m) {
        return Err(FdarError::InvalidDimension {
            parameter: "reconstructed",
            expected: format!("{n}x{m}"),
            actual: format!("{}x{}", reconstructed.nrows(), reconstructed.ncols()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    let weights = simpsons_weights(argvals);

    let spe: Vec<f64> = (0..n)
        .map(|i| {
            let mut sum = 0.0;
            for j in 0..m {
                let diff = centered[(i, j)] - reconstructed[(i, j)];
                sum += diff * diff * weights[j];
            }
            sum
        })
        .collect();

    Ok(spe)
}

/// Compute SPE for multivariate functional data.
///
/// Sum of per-variable integrated squared differences, each variable using
/// its own grid-specific integration weights. Each entry in `argvals_list`
/// must be sorted in ascending order with at least 3 points for Simpson's
/// rule integration. Non-uniform spacing is handled correctly by the
/// quadrature weights.
///
/// # Arguments
/// * `standardized_vars` - Per-variable standardized centered data (each n x m_p)
/// * `reconstructed_vars` - Per-variable reconstructed data (each n x m_p)
/// * `argvals_list` - Per-variable grid points
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if the number of variables is inconsistent
/// or any shapes do not match.
pub fn spe_multivariate(
    standardized_vars: &[&FdMatrix],
    reconstructed_vars: &[&FdMatrix],
    argvals_list: &[&[f64]],
) -> Result<Vec<f64>, FdarError> {
    let p = standardized_vars.len();
    if reconstructed_vars.len() != p || argvals_list.len() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: format!("{p} variables for all arguments"),
            actual: format!(
                "std={}, recon={}, argvals={}",
                standardized_vars.len(),
                reconstructed_vars.len(),
                argvals_list.len()
            ),
        });
    }

    if p == 0 {
        return Ok(Vec::new());
    }

    let n = standardized_vars[0].nrows();

    let mut spe = vec![0.0; n];

    for v in 0..p {
        let std_v = standardized_vars[v];
        let rec_v = reconstructed_vars[v];
        let argvals = argvals_list[v];

        if std_v.nrows() != n || rec_v.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "variables",
                expected: format!("{n} rows for variable {v}"),
                actual: format!("std={}, recon={}", std_v.nrows(), rec_v.nrows()),
            });
        }

        let m_v = std_v.ncols();
        if rec_v.ncols() != m_v || argvals.len() != m_v {
            return Err(FdarError::InvalidDimension {
                parameter: "argvals_list",
                expected: format!("{m_v} for variable {v}"),
                actual: format!(
                    "recon_cols={}, argvals_len={}",
                    rec_v.ncols(),
                    argvals.len()
                ),
            });
        }

        let weights = simpsons_weights(argvals);

        for i in 0..n {
            let mut var_spe = 0.0;
            for j in 0..m_v {
                let diff = std_v[(i, j)] - rec_v[(i, j)];
                var_spe += diff * diff * weights[j];
            }
            spe[i] += var_spe;
        }
    }

    Ok(spe)
}
