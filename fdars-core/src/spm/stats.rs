//! Hotelling T-squared and SPE (Squared Prediction Error) statistics.
//!
//! These are the two fundamental monitoring statistics for functional
//! statistical process control: T-squared captures systematic variation
//! in the principal component subspace, while SPE captures residual
//! variation outside it.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

/// Compute Hotelling T-squared statistic for each observation.
///
/// T-squared_i = sum_{l=1}^{L} scores_{i,l}^2 / eigenvalues_l
///
/// # Arguments
/// * `scores` - FPC score matrix (n x ncomp)
/// * `eigenvalues` - Eigenvalues (length ncomp): sv_l^2 / (n-1)
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

/// Compute SPE (Squared Prediction Error) for univariate functional data.
///
/// SPE_i = integral (x_centered_i(t) - x_reconstructed_i(t))^2 w(t) dt
///
/// # Arguments
/// * `centered` - Centered functional data (n x m)
/// * `reconstructed` - Reconstructed data from FPCA (n x m), already centered (mean subtracted)
/// * `argvals` - Grid points (length m)
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
/// its own grid-specific integration weights.
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
