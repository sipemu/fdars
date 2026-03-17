//! Contribution diagnostics for SPM.
//!
//! When an alarm is triggered, these functions identify which functional
//! variables contribute most to the elevated T-squared or SPE statistic.
//! This aids in root-cause analysis of process faults.
//!
//! # References
//!
//! - Kourti, T. & MacGregor, J.F. (1996). Multivariate SPC methods for
//!   process and product monitoring. *Journal of Quality Technology*, 28(4),
//!   409-428.
//! - Westerhuis, J.A., Gurden, S.P. & Smilde, A.K. (2000). Generalized
//!   contribution plots in multivariate statistical process monitoring.
//!   *Chemometrics and Intelligent Laboratory Systems*, 51(1), 95-114.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

use super::chi_squared::chi2_quantile;
use super::mfpca::MfpcaResult;

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
/// to their grid sizes. For more accurate decomposition when variables have
/// heterogeneous eigenfunction energy, use [`t2_contributions_mfpca()`] which
/// computes the actual per-variable eigenfunction norms.
///
/// Non-positive eigenvalues are skipped (their T-squared contribution is
/// undefined). If this occurs, the row sums will be less than the full
/// Hotelling T-squared.
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

    // Proportional allocation distributes each PC's T² contribution to variables
    // in proportion to grid sizes. This approximates the true decomposition
    // T²_v ≈ Σ_l (score_l² / eigenvalue_l) · ||ψ_l^(v)||² / ||ψ_l||² under
    // the assumption that per-variable eigenfunction norms ||ψ_l^(v)||² are
    // proportional to grid sizes. This holds exactly when eigenfunctions have
    // uniform energy density across the domain. For a more accurate
    // decomposition, use t2_contributions_mfpca() which computes the actual
    // per-variable eigenfunction norms (Westerhuis et al., 2000).
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
/// Row sums give the total SPE per observation, which should match
/// `spe_univariate` or `spe_multivariate` (up to numerical precision).
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
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::contrib::t2_pc_contributions;
/// let scores = FdMatrix::from_column_major(vec![3.0, 1.0, 2.0, 0.5], 2, 2).unwrap();
/// let eigenvalues = vec![2.0, 1.0];
/// let contrib = t2_pc_contributions(&scores, &eigenvalues).unwrap();
/// // Row sums equal Hotelling T²
/// let t2_obs0 = contrib[(0, 0)] + contrib[(0, 1)]; // 3²/2 + 2²/1 = 8.5
/// assert!((t2_obs0 - 8.5).abs() < 1e-10);
/// ```
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

/// Compute per-variable T-squared contributions using MFPCA eigenfunctions.
///
/// This is the recommended method for multivariate SPM. Uses the squared
/// loading norms from each variable's block of the MFPCA eigenfunctions to
/// weight each PC's contribution. This is more accurate than grid-size
/// proportional allocation (in [`t2_contributions`]) because it accounts for
/// how much each variable actually contributes to each principal component.
/// The proportional allocation in `t2_contributions` is a fast approximation
/// suitable for variables with similar eigenfunction structure.
///
/// Returns an n × p matrix where element (i, v) is the T² contribution
/// of variable v for observation i.
///
/// # Arguments
/// * `scores` - Score matrix (n × ncomp)
/// * `mfpca` - MFPCA result (provides eigenvalues and eigenfunctions)
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if shapes are inconsistent.
pub fn t2_contributions_mfpca(
    scores: &FdMatrix,
    mfpca: &MfpcaResult,
) -> Result<FdMatrix, FdarError> {
    let (n, ncomp) = scores.shape();
    let eigenvalues = &mfpca.eigenvalues;

    if ncomp > eigenvalues.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "scores",
            expected: format!("at most {} columns", eigenvalues.len()),
            actual: format!("{ncomp} columns"),
        });
    }

    let p = mfpca.eigenfunctions.len();
    if p == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "eigenfunctions",
            expected: "at least 1 variable".to_string(),
            actual: "0 variables".to_string(),
        });
    }

    // Compute loading weights: for each PC l and variable v,
    // weight[v][l] = sum_j (eigenfunction_v[j, l])^2 / sum_v' sum_j (eigenfunction_v'[j, l])^2
    let mut weights = vec![vec![0.0_f64; ncomp]; p];
    for l in 0..ncomp {
        let mut total_norm = 0.0;
        let mut var_norms = vec![0.0_f64; p];
        for (v, ef) in mfpca.eigenfunctions.iter().enumerate() {
            let m_v = ef.nrows();
            let ef_ncomp = ef.ncols().min(ncomp);
            if l < ef_ncomp {
                let norm_sq: f64 = (0..m_v).map(|j| ef[(j, l)] * ef[(j, l)]).sum();
                var_norms[v] = norm_sq;
                total_norm += norm_sq;
            }
        }
        if total_norm > 0.0 {
            for v in 0..p {
                weights[v][l] = var_norms[v] / total_norm;
            }
        }
    }

    // Compute contributions
    let mut contrib = FdMatrix::zeros(n, p);
    for i in 0..n {
        for l in 0..ncomp {
            if eigenvalues[l] <= 0.0 {
                continue;
            }
            let comp_contrib = scores[(i, l)] * scores[(i, l)] / eigenvalues[l];
            for v in 0..p {
                contrib[(i, v)] += comp_contrib * weights[v][l];
            }
        }
    }

    Ok(contrib)
}

/// Test per-PC T² contributions for Bonferroni-adjusted significance.
///
/// Returns an n × ncomp matrix of booleans (as 0.0/1.0) indicating whether
/// each PC's contribution exceeds the Bonferroni-corrected threshold
/// `χ²(1, 1 - α/ncomp)`. This helps identify which principal components
/// drive an overall T² alarm without inflating the family-wise error rate.
///
/// This is conservative: with many PCs, the per-component threshold increases
/// and individual effects may be missed. For ncomp > 10, consider using
/// Holm-Bonferroni (step-down) or Benjamini-Hochberg (FDR) procedures
/// externally on the per-PC p-values. For exploratory analysis where
/// family-wise error control is less critical, consider using alpha directly
/// (without Bonferroni correction) and interpreting results as suggestive.
///
/// # Arguments
/// * `contributions` - Per-PC T² contributions (n × ncomp), from [`t2_pc_contributions`]
/// * `alpha` - Family-wise significance level (e.g. 0.05)
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::contrib::t2_pc_significance;
/// let contributions = FdMatrix::from_column_major(vec![50.0, 0.1], 1, 2).unwrap();
/// let sig = t2_pc_significance(&contributions, 0.05).unwrap();
/// assert_eq!(sig[(0, 0)], 1.0); // large contribution -> significant
/// assert_eq!(sig[(0, 1)], 0.0); // small contribution -> not significant
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if alpha is not in (0, 1).
pub fn t2_pc_significance(contributions: &FdMatrix, alpha: f64) -> Result<FdMatrix, FdarError> {
    if alpha <= 0.0 || alpha >= 1.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("alpha must be in (0, 1), got {alpha}"),
        });
    }

    let (n, ncomp) = contributions.shape();
    // Bonferroni correction: test each PC at α/ncomp
    let threshold = chi2_quantile(1.0 - alpha / ncomp as f64, 1);

    let mut significant = FdMatrix::zeros(n, ncomp);
    for i in 0..n {
        for l in 0..ncomp {
            if contributions[(i, l)] > threshold {
                significant[(i, l)] = 1.0;
            }
        }
    }

    Ok(significant)
}
