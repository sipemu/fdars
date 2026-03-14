//! Contribution diagnostics for SPM.
//!
//! When an alarm is triggered, these functions identify which functional
//! variables contribute most to the elevated T-squared or SPE statistic.
//! This aids in root-cause analysis of process faults.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

/// Compute per-variable T-squared contributions.
///
/// For a multivariate SPM setting, splits the overall T-squared into
/// per-variable contributions by grouping score components by variable.
///
/// Returns an n x p matrix where element (i, v) is the T-squared
/// contribution of variable v for observation i.
///
/// # Arguments
/// * `scores` - Score matrix (n x ncomp)
/// * `eigenvalues` - Eigenvalues (length ncomp)
/// * `grid_sizes` - Number of grid points per variable (length p), used to
///   define blocks of consecutive score components belonging to each variable.
///   The total sum of components assigned must equal ncomp.
///
/// # Note
///
/// For univariate SPM, the contribution is simply the per-component T-squared.
/// For multivariate SPM, components are assigned to variables proportionally
/// to their grid sizes.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shapes are inconsistent.
pub fn t2_contributions(
    scores: &FdMatrix,
    eigenvalues: &[f64],
    grid_sizes: &[usize],
) -> Result<FdMatrix, FdarError> {
    let (n, ncomp) = scores.shape();
    if ncomp != eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: format!("{ncomp}"),
            actual: format!("{}", eigenvalues.len()),
        });
    }

    let p = grid_sizes.len();
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "grid_sizes",
            expected: "at least 1 variable".to_string(),
            actual: "0 variables".to_string(),
        });
    }

    // Assign score components to variables proportionally
    // For a simple approach: each component contributes to all variables
    // weighted by their grid size proportion. A more rigorous approach
    // would use the MFPCA eigenfunction structure, but this is a standard
    // approximation.

    // Alternative simpler approach: divide ncomp components among variables
    // proportionally to grid sizes (round-robin if not exact).
    let total_grid: usize = grid_sizes.iter().sum();

    let mut contrib = FdMatrix::zeros(n, p);

    for i in 0..n {
        for l in 0..ncomp {
            if eigenvalues[l] <= 0.0 {
                continue;
            }
            let comp_contrib = scores[(i, l)] * scores[(i, l)] / eigenvalues[l];

            // Distribute this component's contribution across variables
            // proportionally to grid sizes
            for v in 0..p {
                let weight = grid_sizes[v] as f64 / total_grid as f64;
                contrib[(i, v)] += comp_contrib * weight;
            }
        }
    }

    Ok(contrib)
}

/// Compute per-variable SPE contributions.
///
/// Returns an n x p matrix where element (i, v) is the integrated squared
/// reconstruction error for variable v for observation i.
///
/// # Arguments
/// * `standardized_vars` - Per-variable standardized centered data
/// * `reconstructed_vars` - Per-variable reconstructed data
/// * `argvals_list` - Per-variable grid points
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if array lengths are inconsistent.
pub fn spe_contributions(
    standardized_vars: &[&FdMatrix],
    reconstructed_vars: &[&FdMatrix],
    argvals_list: &[&[f64]],
) -> Result<FdMatrix, FdarError> {
    let p = standardized_vars.len();
    if reconstructed_vars.len() != p || argvals_list.len() != p {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: format!("{p} for all arguments"),
            actual: format!(
                "std={}, recon={}, argvals={}",
                standardized_vars.len(),
                reconstructed_vars.len(),
                argvals_list.len()
            ),
        });
    }

    if p == 0 {
        return Ok(FdMatrix::zeros(0, 0));
    }

    let n = standardized_vars[0].nrows();

    let mut contrib = FdMatrix::zeros(n, p);

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
            contrib[(i, v)] = var_spe;
        }
    }

    Ok(contrib)
}

/// Compute per-PC T-squared contributions.
///
/// Returns an n × ncomp matrix where element (i, l) is the contribution
/// of principal component l to the T-squared statistic for observation i.
/// Rows sum to the Hotelling T-squared value for each observation.
///
/// # Arguments
/// * `scores` - Score matrix (n × ncomp)
/// * `eigenvalues` - Eigenvalues (length ncomp)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shapes are inconsistent.
/// Returns [`FdarError::InvalidParameter`] if any eigenvalue is non-positive.
pub fn t2_pc_contributions(scores: &FdMatrix, eigenvalues: &[f64]) -> Result<FdMatrix, FdarError> {
    let (n, ncomp) = scores.shape();
    if ncomp != eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenvalues",
            expected: format!("{ncomp}"),
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

    let mut contrib = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for l in 0..ncomp {
            contrib[(i, l)] = scores[(i, l)] * scores[(i, l)] / eigenvalues[l];
        }
    }

    Ok(contrib)
}
