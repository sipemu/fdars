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
use crate::irreg_fdata::kernels::kernel_gaussian;
use crate::irreg_fdata::{cov_irreg, mean_irreg, IrregFdata, KernelType};
use crate::matrix::FdMatrix;
use crate::pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult};
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

/// Multivariate FACE block covariance across several sparse variables.
///
/// Returned by [`mface_covariance`]. Carries the full `(G_total × G_total)` block
/// covariance where `G_total = Σ_p grids[p].len()`. The block for variable pair
/// `(p, q)` occupies rows `offsets[p] .. offsets[p] + grids[p].len()` and columns
/// `offsets[q] .. offsets[q] + grids[q].len()` of [`block_cov`](Self::block_cov).
/// Diagonal blocks are the per-variable [`face_covariance`]; off-diagonal blocks
/// are the cross-variable covariance (symmetric: block `(q, p)` is block `(p, q)`
/// transposed). Use [`block`](Self::block) to extract a sub-block.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MfaceCovResult {
    /// The full `(G_total × G_total)` block covariance matrix (column-major).
    pub block_cov: FdMatrix,
    /// The per-variable evaluation grids (one per variable, in order).
    pub grids: Vec<Vec<f64>>,
    /// Row/column start offset of each variable's block: `offsets[p] = Σ_{k<p} grids[k].len()`.
    pub offsets: Vec<usize>,
}

impl MfaceCovResult {
    /// Extract the `grids[p].len() × grids[q].len()` covariance sub-block between
    /// variable `p` and variable `q`.
    ///
    /// # Panics
    ///
    /// Panics if `p` or `q` is out of range (`>= grids.len()`).
    #[must_use]
    pub fn block(&self, p: usize, q: usize) -> FdMatrix {
        let gp = self.grids[p].len();
        let gq = self.grids[q].len();
        let op = self.offsets[p];
        let oq = self.offsets[q];
        let mut out = FdMatrix::zeros(gp, gq);
        for col in 0..gq {
            for row in 0..gp {
                out[(row, col)] = self.block_cov[(op + row, oq + col)];
            }
        }
        out
    }
}

/// Kernel-smoothed cross-covariance between two sparse variables observed on the
/// same `n` subjects, evaluated on `(s_grid × t_grid)`.
///
/// Mirrors `accumulate_cov_at_point` but pairs each subject's variable-p points
/// with the SAME subject's variable-q points (an outer loop over subjects).
fn cross_cov_surface(
    off_p: &[usize],
    t_p: &[f64],
    c_p: &[f64],
    off_q: &[usize],
    t_q: &[f64],
    c_q: &[f64],
    n: usize,
    s_grid: &[f64],
    t_grid: &[f64],
    bandwidth: f64,
) -> FdMatrix {
    let ns = s_grid.len();
    let nt = t_grid.len();
    let mut data = vec![0.0_f64; ns * nt];
    for (si, &s) in s_grid.iter().enumerate() {
        for (ti, &t) in t_grid.iter().enumerate() {
            let mut sum_w = 0.0;
            let mut sum_p = 0.0;
            for i in 0..n {
                let (ps, pe) = (off_p[i], off_p[i + 1]);
                let (qs, qe) = (off_q[i], off_q[i + 1]);
                for j1 in ps..pe {
                    let w1 = kernel_gaussian((t_p[j1] - s) / bandwidth);
                    for j2 in qs..qe {
                        let w2 = kernel_gaussian((t_q[j2] - t) / bandwidth);
                        let w = w1 * w2;
                        sum_w += w;
                        sum_p += w * c_p[j1] * c_q[j2];
                    }
                }
            }
            data[si + ti * ns] = if sum_w > 0.0 { sum_p / sum_w } else { 0.0 };
        }
    }
    FdMatrix::from_column_major(data, ns, nt).expect("dimension invariant: ns*nt")
}

/// Multivariate FACE (`mfaces`) block covariance across `P >= 2` sparse variables.
///
/// Given `P` sparse variables observed on the **same** `n` subjects (one
/// `IrregFdata` and one evaluation grid per variable), returns the
/// `(G_total × G_total)` block covariance: diagonal blocks are each variable's
/// [`face_covariance`], off-diagonal blocks are the kernel-smoothed cross-variable
/// covariance (symmetric by construction). See [`MfaceCovResult`] for the layout.
///
/// # Divergence from `mfaces`
///
/// Like [`face_covariance`], this is a kernel-sandwich approximation of the R
/// `mfaces` penalized-spline multivariate FACE — matched by capability (a joint
/// block covariance across simultaneously-observed sparse variables), not by exact
/// internals.
///
/// # Errors
///
/// Returns [`FdarError`] if fewer than 2 variables are supplied, `variables` and
/// `grids` have different lengths, the variables have differing observation counts,
/// any grid has fewer than 2 points, or the `bandwidth` is not finite and strictly
/// positive. Returns [`FdarError::ComputationFailed`] if the `G_total²` block
/// allocation would overflow `usize`.
#[must_use = "mface_covariance returns the block covariance; ignoring it wastes the computation"]
pub fn mface_covariance(
    variables: &[IrregFdata],
    grids: &[Vec<f64>],
    bandwidth: f64,
) -> Result<MfaceCovResult, FdarError> {
    let p_count = variables.len();
    if p_count < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "variables",
            expected: ">= 2 variables".to_string(),
            actual: p_count.to_string(),
        });
    }
    if grids.len() != p_count {
        return Err(FdarError::InvalidDimension {
            parameter: "grids",
            expected: format!("{p_count} grids (one per variable)"),
            actual: grids.len().to_string(),
        });
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "bandwidth",
            message: format!("bandwidth must be finite and > 0, got {bandwidth}"),
        });
    }
    let n = variables[0].n_obs();
    for (p, var) in variables.iter().enumerate() {
        if var.n_obs() != n {
            return Err(FdarError::InvalidDimension {
                parameter: "variables",
                expected: format!("all variables observed on {n} subjects"),
                actual: format!("variable {p} has {} observations", var.n_obs()),
            });
        }
        if grids[p].len() < 2 {
            return Err(FdarError::InvalidDimension {
                parameter: "grids",
                expected: ">= 2 points per grid".to_string(),
                actual: format!("grid {p} has {} points", grids[p].len()),
            });
        }
    }

    // Per-variable offsets and total size.
    let mut offsets = Vec::with_capacity(p_count);
    let mut g_total = 0usize;
    for g in grids {
        offsets.push(g_total);
        g_total += g.len();
    }
    g_total
        .checked_mul(g_total)
        .ok_or_else(|| FdarError::ComputationFailed {
            operation: "mface_covariance block allocation",
            detail: format!("G_total={g_total}: G_total² overflows usize"),
        })?;

    // Per-variable smoothed-mean-centered values (for the cross blocks).
    let centered: Vec<Vec<f64>> = variables
        .iter()
        .map(|var| {
            let mean = mean_irreg(var, &var.argvals, bandwidth, KernelType::Gaussian);
            var.values
                .iter()
                .zip(mean.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        })
        .collect();

    let mut block_data = vec![0.0_f64; g_total * g_total];

    // Diagonal blocks: per-variable FACE covariance.
    for p in 0..p_count {
        let diag = face_covariance(&variables[p], &grids[p], bandwidth)?;
        let op = offsets[p];
        let gp = grids[p].len();
        for col in 0..gp {
            for row in 0..gp {
                block_data[(op + row) + (op + col) * g_total] = diag[(row, col)];
            }
        }
    }

    // Off-diagonal upper triangle p < q: cross-covariance + symmetric transpose.
    for p in 0..p_count {
        for q in (p + 1)..p_count {
            let cross = cross_cov_surface(
                &variables[p].offsets,
                &variables[p].argvals,
                &centered[p],
                &variables[q].offsets,
                &variables[q].argvals,
                &centered[q],
                n,
                &grids[p],
                &grids[q],
                bandwidth,
            );
            let op = offsets[p];
            let oq = offsets[q];
            let gp = grids[p].len();
            let gq = grids[q].len();
            for col in 0..gq {
                for row in 0..gp {
                    let val = cross[(row, col)];
                    block_data[(op + row) + (oq + col) * g_total] = val;
                    block_data[(oq + col) + (op + row) * g_total] = val; // symmetric
                }
            }
        }
    }

    let block_cov = FdMatrix::from_column_major(block_data, g_total, g_total).map_err(|e| {
        FdarError::ComputationFailed {
            operation: "mface_covariance block assembly",
            detail: e.to_string(),
        }
    })?;
    Ok(MfaceCovResult {
        block_cov,
        grids: grids.to_vec(),
        offsets,
    })
}

/// Fitted continuous trajectories with pointwise confidence bands for sparse
/// functional data.
///
/// A thin, semantically-named entry point over the shipped PACE FPCA BLUP engine
/// ([`pace_fpca`]): it returns per-curve fitted trajectories together with
/// pointwise (Gaussian) confidence bands on the config's work grid. The returned
/// [`PaceFpcaResult`] carries `fitted`, `fitted_lower`, `fitted_upper`, and
/// `argvals`. `sigma2` (measurement-error variance) and `alpha` (band level) in
/// the [`PaceFpcaConfig`] are the primary tuning knobs.
///
/// For the FACE covariance **surface** itself, use [`face_covariance`] separately.
///
/// # Errors
///
/// Propagates every [`pace_fpca`] validation error unchanged (empty sample, too
/// few points, out-of-range `ncomp`/`bandwidth`/`sigma2`/`alpha`, etc.).
#[must_use = "face_trajectory returns fitted trajectories + bands; ignoring it wastes the computation"]
pub fn face_trajectory(
    data: &IrregFdata,
    config: &PaceFpcaConfig,
) -> Result<PaceFpcaResult, FdarError> {
    pace_fpca(data, config)
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

    // ---- mface_covariance ------------------------------------------------

    /// Build P=2 dense variables on n subjects with X_i=a_i·sin(πt), Y_i=a_i·cos(πt).
    /// Returns (variables, grids, lambda_pop) where lambda_pop = mean((a_i-ā)²).
    fn two_var_sample(n: usize, m: usize, seed: u64) -> (Vec<IrregFdata>, Vec<Vec<f64>>, f64) {
        let grid: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let amps: Vec<f64> = (0..n)
            .map(|_| {
                let z: f64 = StandardNormal.sample(&mut rng);
                1.0 + z
            })
            .collect();
        let abar = amps.iter().sum::<f64>() / n as f64;
        let lambda_pop = amps.iter().map(|&a| (a - abar).powi(2)).sum::<f64>() / n as f64;

        let mut ax = Vec::with_capacity(n);
        let mut vx = Vec::with_capacity(n);
        let mut ay = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        for &a in &amps {
            ax.push(grid.clone());
            vx.push(
                grid.iter()
                    .map(|&t| a * (std::f64::consts::PI * t).sin())
                    .collect(),
            );
            ay.push(grid.clone());
            vy.push(
                grid.iter()
                    .map(|&t| a * (std::f64::consts::PI * t).cos())
                    .collect(),
            );
        }
        let vars = vec![
            IrregFdata::from_lists(&ax, &vx),
            IrregFdata::from_lists(&ay, &vy),
        ];
        (vars, vec![grid.clone(), grid], lambda_pop)
    }

    #[test]
    fn test_mface_shape() {
        let (vars, grids, _) = two_var_sample(20, 9, 7);
        let res = mface_covariance(&vars, &grids, 0.15).unwrap();
        let g0 = grids[0].len();
        let g1 = grids[1].len();
        let gt = g0 + g1;
        assert_eq!(res.block_cov.shape(), (gt, gt));
        assert_eq!(res.offsets, vec![0, g0]);
        // Full block matrix symmetric.
        for i in 0..gt {
            for j in 0..gt {
                assert!(
                    (res.block_cov[(i, j)] - res.block_cov[(j, i)]).abs() < 1e-9,
                    "block matrix not symmetric at ({i},{j})"
                );
            }
        }
        // Diagonal block equals standalone face_covariance.
        let f0 = face_covariance(&vars[0], &grids[0], 0.15).unwrap();
        let b00 = res.block(0, 0);
        for i in 0..g0 {
            for j in 0..g0 {
                assert!((b00[(i, j)] - f0[(i, j)]).abs() < 1e-9);
            }
        }
        // block(1,0) == block(0,1)ᵀ.
        let b01 = res.block(0, 1);
        let b10 = res.block(1, 0);
        assert_eq!(b01.shape(), (g0, g1));
        assert_eq!(b10.shape(), (g1, g0));
        for i in 0..g0 {
            for j in 0..g1 {
                assert!((b01[(i, j)] - b10[(j, i)]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_mface_known_structure() {
        // Off-diagonal block should recover λ·sin(πs)·cos(πt), λ = pop var of a.
        let m = 21usize;
        let (vars, grids, lambda) = two_var_sample(200, m, 11);
        let res = mface_covariance(&vars, &grids, 0.08).unwrap();
        let b01 = res.block(0, 1);
        let grid = &grids[0];
        let mut max_err = 0.0_f64;
        for si in 0..m {
            for ti in 0..m {
                let truth = lambda
                    * (std::f64::consts::PI * grid[si]).sin()
                    * (std::f64::consts::PI * grid[ti]).cos();
                max_err = max_err.max((b01[(si, ti)] - truth).abs());
            }
        }
        // Kernel-smoothed cross-covariance of a rank-1 structure (RESEARCH A3 ~0.4).
        assert!(
            max_err < 0.4,
            "mface cross-block max error {max_err} exceeds tolerance"
        );
    }

    #[test]
    fn test_mface_errors() {
        let (vars, grids, _) = two_var_sample(10, 6, 3);
        // fewer than 2 variables
        assert!(mface_covariance(&[], &[], 0.1).is_err());
        assert!(mface_covariance(&vars[..1], &grids[..1], 0.1).is_err());
        // variables/grids length mismatch
        assert!(mface_covariance(&vars, &grids[..1], 0.1).is_err());
        // mismatched n_obs across variables
        let (vars_a, grids_a, _) = two_var_sample(10, 6, 5);
        let (vars_b, _, _) = two_var_sample(8, 6, 6);
        let mixed = vec![vars_a[0].clone(), vars_b[0].clone()];
        assert!(mface_covariance(&mixed, &grids_a, 0.1).is_err());
        // grid < 2 points
        let short_grids = vec![vec![0.5], grids[1].clone()];
        assert!(mface_covariance(&vars, &short_grids, 0.1).is_err());
        // invalid bandwidth
        assert!(mface_covariance(&vars, &grids, 0.0).is_err());
        assert!(mface_covariance(&vars, &grids, f64::NAN).is_err());
    }

    // ---- face_trajectory -------------------------------------------------

    fn dense_sample(n: usize, m: usize, seed: u64) -> (IrregFdata, Vec<f64>, Vec<Vec<f64>>) {
        let grid: Vec<f64> = (0..m).map(|i| i as f64 / (m as f64 - 1.0)).collect();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut argvals = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);
        let mut truth = Vec::with_capacity(n);
        for _ in 0..n {
            let z: f64 = StandardNormal.sample(&mut rng);
            let a = 1.0 + 0.5 * z;
            let curve: Vec<f64> = grid
                .iter()
                .map(|&t| a * (std::f64::consts::PI * t).sin())
                .collect();
            argvals.push(grid.clone());
            values.push(curve.clone());
            truth.push(curve);
        }
        (IrregFdata::from_lists(&argvals, &values), grid, truth)
    }

    #[test]
    fn test_face_trajectory_delegation() {
        let (data, grid, _) = dense_sample(15, 21, 1);
        let config = PaceFpcaConfig {
            ncomp: 2,
            bandwidth: 0.1,
            sigma2: 0.01,
            work_grid: grid,
            alpha: 0.05,
        };
        let a = face_trajectory(&data, &config).unwrap();
        let b = pace_fpca(&data, &config).unwrap();
        assert!(a == b, "face_trajectory must delegate to pace_fpca exactly");
    }

    #[test]
    fn test_face_trajectory_bands() {
        let m = 41usize;
        let (data, grid, truth) = dense_sample(25, m, 2);
        let config = PaceFpcaConfig {
            ncomp: 2,
            bandwidth: 0.1,
            sigma2: 0.01,
            work_grid: grid,
            alpha: 0.05,
        };
        let res = face_trajectory(&data, &config).unwrap();
        let n = truth.len();
        let mut inside = 0usize;
        let mut total = 0usize;
        for i in 0..n {
            for j in 0..m {
                let lo = res.fitted_lower[(i, j)];
                let hi = res.fitted_upper[(i, j)];
                if truth[i][j] >= lo && truth[i][j] <= hi {
                    inside += 1;
                }
                total += 1;
            }
        }
        let frac = inside as f64 / total as f64;
        // Pointwise 95% BLUP bands on a smooth process: empirical coverage is
        // typically ~85-92% in finite samples (kernel-smoothing bias shifts the
        // fitted mean, so nominal 95% pointwise bands undercover slightly).
        assert!(frac >= 0.85, "only {frac} of true points inside bands");
    }

    // ---- crate-root re-exports -------------------------------------------

    #[test]
    fn test_reexports() {
        // Reference each symbol through the crate root so a broken re-export
        // chain fails to compile.
        use crate::{face_covariance, face_trajectory, mface_covariance, MfaceCovResult};
        let (vars, grids, _) = two_var_sample(12, 7, 99);
        let _diag = face_covariance(&vars[0], &grids[0], 0.15).unwrap();
        let res: MfaceCovResult = mface_covariance(&vars, &grids, 0.15).unwrap();
        let _ = res.block(0, 1);
        let config = PaceFpcaConfig {
            ncomp: 1,
            bandwidth: 0.15,
            sigma2: 0.01,
            work_grid: grids[0].clone(),
            alpha: 0.05,
        };
        let _traj = face_trajectory(&vars[0], &config).unwrap();
    }
}
