//! Multivariate Functional Principal Component Analysis (MFPCA).
//!
//! Extends univariate FPCA to handle multiple functional variables
//! observed on potentially different grids. Variables are optionally
//! weighted by their inverse standard deviation before joint SVD.
//!
//! # Algorithm
//!
//! 1. Center each variable by its column means.
//! 2. Standardize each variable by its scale (sqrt of mean column variance)
//!    when `weighted = true`. This ensures variables on different scales
//!    contribute equally to the joint decomposition.
//! 3. Horizontally concatenate the centered/scaled variables into an
//!    n × (sum m_p) matrix and perform truncated SVD.
//! 4. Extract scores as U * S and eigenfunctions from V. Eigenfunctions
//!    are back-scaled by the per-variable scale factor when reconstructing,
//!    so that reconstruction returns values on the original measurement scale.
//!
//! # Projection accuracy
//!
//! The projection error ||X - X_hat||² is bounded by sum_{l > ncomp} lambda_l
//! (the sum of discarded eigenvalues). This provides an a priori error bound
//! for choosing ncomp: retain enough components so that the discarded
//! eigenvalue tail is acceptably small relative to the total variance.
//!
//! # Scale threshold
//!
//! Variables with scale < 1e-12 (relative to the maximum scale across all
//! variables) are treated as constant and receive unit weight (scale = 1.0).
//! This prevents numerical issues from near-zero denominators in the
//! standardization step.
//!
//! # References
//!
//! - Happ, C. & Greven, S. (2018). Multivariate functional principal component
//!   analysis for data observed on different (dimensional) domains. *Journal of
//!   the American Statistical Association*, 113(522), 649-659. The weighting
//!   scheme follows Section 2.2 (standardization by variable-specific scale)
//!   and the eigendecomposition is described in Section 2.3, Eq. 3-5.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use nalgebra::SVD;

/// Configuration for multivariate FPCA.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MfpcaConfig {
    /// Number of principal components to extract (default 5).
    pub ncomp: usize,
    /// Whether to weight each variable by 1/std_dev before SVD (default true).
    pub weighted: bool,
}

impl Default for MfpcaConfig {
    fn default() -> Self {
        Self {
            ncomp: 5,
            weighted: true,
        }
    }
}

/// Result of multivariate FPCA.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MfpcaResult {
    /// Score matrix (n x ncomp).
    pub scores: FdMatrix,
    /// Eigenfunctions split by variable (one FdMatrix per variable, each m_p x ncomp).
    pub eigenfunctions: Vec<FdMatrix>,
    /// Eigenvalues (length ncomp): squared singular values divided by (n-1).
    pub eigenvalues: Vec<f64>,
    /// Per-variable mean functions.
    pub means: Vec<Vec<f64>>,
    /// Per-variable standard deviations (sqrt of mean column variance).
    pub scales: Vec<f64>,
    /// Grid sizes per variable.
    pub grid_sizes: Vec<usize>,
    /// Combined rotation matrix (sum(m_p) x ncomp) — internal use for projection.
    pub(super) combined_rotation: FdMatrix,
    /// Threshold below which a variable's scale is treated as 1.0 (avoids division by near-zero).
    pub(super) scale_threshold: f64,
}

impl MfpcaResult {
    /// Project new multivariate functional data onto the MFPCA score space.
    ///
    /// Each element of `new_data` is an n_new x m_p matrix for variable p.
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] if the number of variables or
    /// their grid sizes do not match the training data.
    pub fn project(&self, new_data: &[&FdMatrix]) -> Result<FdMatrix, FdarError> {
        if new_data.len() != self.means.len() {
            return Err(FdarError::InvalidDimension {
                parameter: "new_data",
                expected: format!("{} variables", self.means.len()),
                actual: format!("{} variables", new_data.len()),
            });
        }

        // Early check: total columns across all variables must match the
        // combined rotation matrix rows (i.e., the sum of training grid sizes).
        let total_input_cols: usize = new_data.iter().map(|v| v.ncols()).sum();
        let expected_total: usize = self.grid_sizes.iter().sum();
        if total_input_cols != expected_total {
            return Err(FdarError::InvalidDimension {
                parameter: "new_data",
                expected: format!("{expected_total} total columns across all variables"),
                actual: format!("{total_input_cols} total columns"),
            });
        }

        let n_new = new_data[0].nrows();
        let ncomp = self.scores.ncols();
        let total_cols: usize = self.grid_sizes.iter().sum();

        // Center, scale, and stack
        let mut stacked = FdMatrix::zeros(n_new, total_cols);
        let mut col_offset = 0;
        for (p, &var) in new_data.iter().enumerate() {
            let m_p = self.grid_sizes[p];
            if var.ncols() != m_p {
                return Err(FdarError::InvalidDimension {
                    parameter: "new_data",
                    expected: format!("{m_p} columns for variable {p}"),
                    actual: format!("{} columns", var.ncols()),
                });
            }
            if var.nrows() != n_new {
                return Err(FdarError::InvalidDimension {
                    parameter: "new_data",
                    expected: format!("{n_new} rows for all variables"),
                    actual: format!("{} rows for variable {p}", var.nrows()),
                });
            }
            let scale = if self.scales[p] >= self.scale_threshold {
                self.scales[p]
            } else {
                1.0
            };
            for i in 0..n_new {
                for j in 0..m_p {
                    let centered = var[(i, j)] - self.means[p][j];
                    stacked[(i, col_offset + j)] = centered / scale;
                }
            }
            col_offset += m_p;
        }

        // Multiply by combined rotation: scores = stacked * rotation
        let mut scores = FdMatrix::zeros(n_new, ncomp);
        for i in 0..n_new {
            for k in 0..ncomp {
                let mut sum = 0.0;
                for j in 0..total_cols {
                    sum += stacked[(i, j)] * self.combined_rotation[(j, k)];
                }
                scores[(i, k)] = sum;
            }
        }
        Ok(scores)
    }

    /// Reconstruct multivariate functional data from scores.
    ///
    /// Returns one FdMatrix per variable (n x m_p).
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidParameter`] if `ncomp` exceeds available components.
    pub fn reconstruct(&self, scores: &FdMatrix, ncomp: usize) -> Result<Vec<FdMatrix>, FdarError> {
        let max_comp = self.combined_rotation.ncols().min(scores.ncols());
        if ncomp == 0 || ncomp > max_comp {
            return Err(FdarError::InvalidParameter {
                parameter: "ncomp",
                message: format!("ncomp={ncomp} must be in 1..={max_comp}"),
            });
        }

        let n = scores.nrows();
        let total_cols: usize = self.grid_sizes.iter().sum();

        // Reconstruct in combined space: stacked_recon = scores * rotation^T
        let mut stacked = FdMatrix::zeros(n, total_cols);
        for i in 0..n {
            for j in 0..total_cols {
                let mut val = 0.0;
                for k in 0..ncomp {
                    val += scores[(i, k)] * self.combined_rotation[(j, k)];
                }
                stacked[(i, j)] = val;
            }
        }

        // Split by variable, un-scale, and add means
        let mut result = Vec::with_capacity(self.means.len());
        let mut col_offset = 0;
        for (p, m_p) in self.grid_sizes.iter().enumerate() {
            let scale = if self.scales[p] >= self.scale_threshold {
                self.scales[p]
            } else {
                1.0
            };
            let mut var_mat = FdMatrix::zeros(n, *m_p);
            for i in 0..n {
                for j in 0..*m_p {
                    var_mat[(i, j)] = stacked[(i, col_offset + j)] * scale + self.means[p][j];
                }
            }
            col_offset += m_p;
            result.push(var_mat);
        }
        Ok(result)
    }
}

/// Perform multivariate FPCA on multiple functional variables.
///
/// # Arguments
/// * `variables` - Slice of n x m_p matrices, one per functional variable
/// * `config` - MFPCA configuration
///
/// # Example
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::spm::mfpca::{mfpca, MfpcaConfig};
/// let var1 = FdMatrix::from_column_major(vec![1.0,2.0,3.0,4.0,5.0,6.0], 3, 2).unwrap();
/// let var2 = FdMatrix::from_column_major(vec![0.5,1.5,2.5,3.5,4.5,5.5], 3, 2).unwrap();
/// let config = MfpcaConfig { ncomp: 2, weighted: true };
/// let result = mfpca(&[&var1, &var2], &config).unwrap();
/// assert_eq!(result.eigenvalues.len(), 2);
/// assert!(result.eigenvalues[0] >= result.eigenvalues[1]);
/// ```
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if no variables are provided or
/// variables have inconsistent row counts. Returns [`FdarError::ComputationFailed`]
/// if the SVD fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn mfpca(variables: &[&FdMatrix], config: &MfpcaConfig) -> Result<MfpcaResult, FdarError> {
    if variables.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: "at least 1 variable".to_string(),
            actual: "0 variables".to_string(),
        });
    }

    let n = variables[0].nrows();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: "at least 2 observations".to_string(),
            actual: format!("{n} observations"),
        });
    }

    for (p, var) in variables.iter().enumerate() {
        if var.nrows() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "variables",
                expected: format!("{n} rows for all variables"),
                actual: format!("{} rows for variable {p}", var.nrows()),
            });
        }
    }

    let grid_sizes: Vec<usize> = variables.iter().map(|v| v.ncols()).collect();
    let total_cols: usize = grid_sizes.iter().sum();
    let ncomp = config.ncomp.min(n).min(total_cols);

    // Step 1: Center each variable and compute scale
    let mut means: Vec<Vec<f64>> = Vec::with_capacity(variables.len());
    let mut scales: Vec<f64> = Vec::with_capacity(variables.len());

    for var in variables.iter() {
        let (_, m_p) = var.shape();
        let mut mean = vec![0.0; m_p];
        for j in 0..m_p {
            let col = var.column(j);
            mean[j] = col.iter().sum::<f64>() / n as f64;
        }

        // Scale = sqrt(mean of column variances)
        let mut mean_var = 0.0;
        for j in 0..m_p {
            let col = var.column(j);
            let var_j: f64 =
                col.iter().map(|&v| (v - mean[j]).powi(2)).sum::<f64>() / (n as f64 - 1.0);
            mean_var += var_j;
        }
        mean_var /= m_p as f64;
        let scale = mean_var.sqrt();

        means.push(mean);
        scales.push(scale);
    }

    // Use relative threshold for scale: 1e-12 * max(scales).
    // Variables with scale below this threshold contribute negligible variance
    // and are effectively constant. Treating them as unscaled avoids division
    // by near-zero values.
    let max_scale = scales.iter().cloned().fold(0.0_f64, f64::max);
    let scale_threshold = 1e-12 * max_scale.max(1e-15); // floor to avoid 0

    // Step 2: Build stacked matrix (n x total_cols)
    let mut stacked = FdMatrix::zeros(n, total_cols);
    let mut col_offset = 0;
    for (p, var) in variables.iter().enumerate() {
        let m_p = grid_sizes[p];
        let scale = if config.weighted && scales[p] > scale_threshold {
            scales[p]
        } else {
            1.0
        };
        // Update scales to reflect actual scaling applied
        if !config.weighted {
            scales[p] = 1.0;
        }
        for i in 0..n {
            for j in 0..m_p {
                let centered = var[(i, j)] - means[p][j];
                stacked[(i, col_offset + j)] = centered / scale;
            }
        }
        col_offset += m_p;
    }

    // Step 3: SVD on stacked matrix
    let svd = SVD::new(stacked.to_dmatrix(), true, true);

    let v_t = svd
        .v_t
        .as_ref()
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "MFPCA SVD",
            detail: "SVD failed to produce V_t matrix".to_string(),
        })?;

    let u = svd.u.as_ref().ok_or_else(|| FdarError::ComputationFailed {
        operation: "MFPCA SVD",
        detail: "SVD failed to produce U matrix".to_string(),
    })?;

    // Step 4: Extract components
    let singular_values: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();

    // Eigenvalues = sv^2 / (n - 1)
    let eigenvalues: Vec<f64> = singular_values
        .iter()
        .map(|&sv| sv * sv / (n as f64 - 1.0))
        .collect();

    // Combined rotation: V[:, :ncomp] (total_cols x ncomp)
    let mut combined_rotation = FdMatrix::zeros(total_cols, ncomp);
    for k in 0..ncomp {
        for j in 0..total_cols {
            combined_rotation[(j, k)] = v_t[(k, j)];
        }
    }

    // Scores: U * S (n x ncomp)
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        let sv_k = singular_values[k];
        for i in 0..n {
            scores[(i, k)] = u[(i, k)] * sv_k;
        }
    }

    // Step 5: Split eigenfunctions by variable
    let mut eigenfunctions = Vec::with_capacity(variables.len());
    let mut col_off = 0;
    for m_p in &grid_sizes {
        let mut ef = FdMatrix::zeros(*m_p, ncomp);
        for k in 0..ncomp {
            for j in 0..*m_p {
                ef[(j, k)] = combined_rotation[(col_off + j, k)];
            }
        }
        col_off += m_p;
        eigenfunctions.push(ef);
    }

    Ok(MfpcaResult {
        scores,
        eigenfunctions,
        eigenvalues,
        means,
        scales,
        grid_sizes,
        combined_rotation,
        scale_threshold,
    })
}
