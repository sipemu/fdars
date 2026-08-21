//! FACE fast-sandwich covariance for sparse/irregular functional data.
//!
//! This module adds the FACE (Fast Covariance Estimation) family that the R
//! `face` / `mfaces` packages expose and that fdars previously lacked:
//!
//! - [`face_covariance`] — a fast-sandwich covariance surface for sparse/irregular
//!   functional data.
//! - [`mface_covariance`] — its multivariate (`mfaces`) block extension for several
//!   simultaneously-observed sparse variables.
//! - [`face_trajectory`] — fitted continuous trajectories with pointwise confidence
//!   bands (a thin reuse of the shipped PACE FPCA path).
//!
//! # Divergence from `refund::face`
//!
//! `refund::face` builds the covariance with a **penalized tensor-product spline**
//! sandwich smoother (P-FACE). fdars instead sandwiches the existing
//! kernel-smoothed sparse covariance ([`crate::irreg_fdata::cov_irreg`]) with a
//! separable Gaussian smoother and projects the result to the nearest PSD matrix
//! (a kernel-FACE, K-FACE). The two match by **capability** — a fast, symmetric,
//! positive-semidefinite covariance surface for sparse data — not by exact
//! internals. This keeps the estimator additive and dependency-free (it reuses
//! `cov_irreg` and the Phase-37 sandwich smoother), per the milestone constraint.

use crate::error::FdarError;
use crate::fpca_variants::gaussian_smooth_cov;
use crate::helpers::simpsons_weights;
use crate::irreg_fdata::{cov_irreg, IrregFdata};
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;

/// Validate a covariance evaluation grid: at least 2 points, strictly increasing.
fn validate_grid(grid: &[f64]) -> Result<(), FdarError> {
    if grid.len() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "grid",
            expected: ">= 2 points".to_string(),
            actual: grid.len().to_string(),
        });
    }
    if grid.windows(2).any(|w| w[0] >= w[1]) {
        return Err(FdarError::InvalidParameter {
            parameter: "grid",
            message: "grid must be strictly increasing".to_string(),
        });
    }
    Ok(())
}

/// Project a symmetric surface to the nearest PSD matrix under the functional
/// L2 inner product, reusing the `W^{1/2}·Cov·W^{1/2}` sandwich eigendecomposition
/// (mirrors the Phase-37 `ssvd` / PACE `eigendecompose_cov` pattern): clip
/// negative eigenvalues (estimation noise) to zero and reconstruct.
fn psd_project(cov: &FdMatrix, grid: &[f64]) -> Result<FdMatrix, FdarError> {
    let m = grid.len();
    let w = simpsons_weights(grid);
    let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();

    let mut c_scaled = vec![0.0_f64; m * m];
    for col in 0..m {
        for row in 0..m {
            c_scaled[row + col * m] = sqrt_w[row] * cov[(row, col)] * sqrt_w[col];
        }
    }
    let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();

    let mut cov_data = vec![0.0_f64; m * m];
    for k in 0..eigen.eigenvalues.len() {
        let lam = eigen.eigenvalues[k];
        if lam <= 0.0 {
            continue; // clip estimation-noise negatives to zero
        }
        // Unscale eigenvector: φ_j = v_j / sqrt_w[j].
        let mut phi = vec![0.0_f64; m];
        for j in 0..m {
            let raw = eigen.eigenvectors[(j, k)];
            phi[j] = if sqrt_w[j] > 1e-15 {
                raw / sqrt_w[j]
            } else {
                raw
            };
        }
        for j in 0..m {
            for i in 0..m {
                cov_data[i + j * m] += lam * phi[i] * phi[j];
            }
        }
    }
    FdMatrix::from_column_major(cov_data, m, m).map_err(|e| FdarError::ComputationFailed {
        operation: "face_covariance PSD projection",
        detail: e.to_string(),
    })
}

/// FACE fast-sandwich covariance surface for sparse/irregular functional data.
///
/// Estimates a symmetric, positive-semidefinite covariance surface on `grid` from
/// the sparse/irregular sample `ifd`. The raw kernel-smoothed covariance
/// ([`cov_irreg`]) is sandwiched with a separable Gaussian smoother (the same
/// `bandwidth` on both passes) and projected to the nearest PSD matrix.
///
/// On a densely-observed regular sample this recovers the underlying covariance
/// surface within a smoothing tolerance. See the module docs for the divergence
/// from `refund::face`.
///
/// # Errors
///
/// Returns [`FdarError`] if the sample is empty (`ifd.n_obs() == 0`), the `grid`
/// has fewer than 2 points or is not strictly increasing, or the `bandwidth` is
/// not finite and strictly positive. Inputs are validated **before** calling
/// [`cov_irreg`], which panics on malformed dimensions.
///
/// # Examples
///
/// ```
/// use fdars_core::irreg_fdata::IrregFdata;
/// use fdars_core::face_covariance;
///
/// let argvals = vec![vec![0.0, 0.5, 1.0], vec![0.2, 0.8], vec![0.1, 0.6, 0.9]];
/// let values = vec![vec![1.0, 0.5, 0.2], vec![0.9, 0.3], vec![1.1, 0.4, 0.25]];
/// let ifd = IrregFdata::from_lists(&argvals, &values);
/// let grid: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
/// let cov = face_covariance(&ifd, &grid, 0.3).unwrap();
/// assert_eq!(cov.shape(), (11, 11));
/// ```
#[must_use = "face_covariance returns the covariance surface; ignoring it wastes the computation"]
pub fn face_covariance(
    ifd: &IrregFdata,
    grid: &[f64],
    bandwidth: f64,
) -> Result<FdMatrix, FdarError> {
    if ifd.n_obs() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "ifd",
            expected: ">= 1 observation".to_string(),
            actual: "0".to_string(),
        });
    }
    validate_grid(grid)?;
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("bandwidth must be finite and > 0, got {bandwidth}"),
        });
    }

    let raw_cov = cov_irreg(ifd, grid, grid, bandwidth);
    let smooth_cov = gaussian_smooth_cov(&raw_cov, grid, bandwidth);
    psd_project(&smooth_cov, grid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Cholesky;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_distr::{Distribution, StandardNormal};

    /// Build a sparse IrregFdata from ragged (argvals, values) lists.
    fn sparse_sample() -> (IrregFdata, Vec<f64>) {
        let argvals = vec![
            vec![0.0, 0.3, 0.6, 1.0],
            vec![0.1, 0.5, 0.9],
            vec![0.0, 0.4, 0.7, 0.95],
            vec![0.2, 0.6],
            vec![0.05, 0.45, 0.85],
            vec![0.15, 0.55, 0.9],
            vec![0.0, 0.5, 1.0],
            vec![0.3, 0.7],
            vec![0.1, 0.4, 0.8],
            vec![0.25, 0.65, 0.95],
        ];
        let values: Vec<Vec<f64>> = argvals
            .iter()
            .enumerate()
            .map(|(i, ts)| {
                let a = 0.5 + i as f64 * 0.1;
                ts.iter()
                    .map(|&t| a * (std::f64::consts::PI * t).sin())
                    .collect()
            })
            .collect();
        let grid: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        (IrregFdata::from_lists(&argvals, &values), grid)
    }

    fn min_eigenvalue(cov: &FdMatrix) -> f64 {
        let (m, _) = cov.shape();
        let dm = DMatrix::from_fn(m, m, |i, j| cov[(i, j)]);
        dm.symmetric_eigen()
            .eigenvalues
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    #[test]
    fn test_face_covariance_shape() {
        let (ifd, grid) = sparse_sample();
        let cov = face_covariance(&ifd, &grid, 0.3).unwrap();
        let m = grid.len();
        assert_eq!(cov.shape(), (m, m));
        // Symmetric.
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (cov[(i, j)] - cov[(j, i)]).abs() < 1e-9,
                    "not symmetric at ({i},{j})"
                );
            }
        }
        // PSD (all eigenvalues >= -tiny).
        assert!(min_eigenvalue(&cov) >= -1e-9, "not PSD");
    }

    #[test]
    fn test_face_covariance_dense_limit() {
        // n dense curves at the SAME m grid, drawn from a process with a KNOWN
        // OU covariance C(s,t) = exp(-|s-t|), via Cholesky of the kernel matrix.
        let m = 31usize;
        let grid: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
        let kernel = DMatrix::from_fn(m, m, |i, j| (-(grid[i] - grid[j]).abs()).exp());
        let chol = Cholesky::new(kernel).expect("OU kernel is PD");
        let l = chol.l();

        let n = 200usize;
        let mut rng = StdRng::seed_from_u64(42);
        let mut argvals_list = Vec::with_capacity(n);
        let mut values_list = Vec::with_capacity(n);
        for _ in 0..n {
            let z: Vec<f64> = (0..m).map(|_| StandardNormal.sample(&mut rng)).collect();
            let zvec = nalgebra::DVector::from_vec(z);
            let x = &l * zvec; // x ~ N(0, kernel)
            argvals_list.push(grid.clone());
            values_list.push(x.iter().copied().collect());
        }
        let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

        let cov = face_covariance(&ifd, &grid, 0.05).unwrap();
        let mut max_err = 0.0_f64;
        for si in 0..m {
            for ti in 0..m {
                let truth = (-(grid[si] - grid[ti]).abs()).exp();
                max_err = max_err.max((cov[(si, ti)] - truth).abs());
            }
        }
        // Kernel-sandwich estimate of an OU surface from n dense curves. The OU
        // covariance exp(-|s-t|) has a non-differentiable ridge at s=t that kernel
        // smoothing necessarily rounds, so the max abs error is dominated by that
        // smoothing bias near the diagonal (peak surface value 1.0). Tolerance
        // calibrated to bias + finite-sample noise (RESEARCH A2 ~0.3).
        assert!(
            max_err < 0.30,
            "dense-limit max error {max_err} exceeds tolerance"
        );
    }

    #[test]
    fn test_face_covariance_errors() {
        let (ifd, grid) = sparse_sample();
        // empty sample
        let empty = IrregFdata::from_lists(&[], &[]);
        assert!(face_covariance(&empty, &grid, 0.3).is_err());
        // grid too short
        assert!(face_covariance(&ifd, &[], 0.3).is_err());
        assert!(face_covariance(&ifd, &[0.5], 0.3).is_err());
        // non-monotone grid
        assert!(face_covariance(&ifd, &[0.0, 0.5, 0.4], 0.3).is_err());
        // invalid bandwidths
        assert!(face_covariance(&ifd, &grid, 0.0).is_err());
        assert!(face_covariance(&ifd, &grid, -1.0).is_err());
        assert!(face_covariance(&ifd, &grid, f64::NAN).is_err());
        assert!(face_covariance(&ifd, &grid, f64::INFINITY).is_err());
    }
}
