//! Multidimensional (R^d) SRSF transforms and elastic alignment.

use super::srsf::reparameterize_curve;
use super::{
    dp_alignment_core, dp_edge_weight, dp_grid_solve, dp_lambda_penalty, dp_path_to_gamma,
};
use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, l2_distance, simpsons_weights};
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of aligning multidimensional (R^d) curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AlignmentResultNd {
    /// Optimal warping function (length m), same for all dimensions.
    pub gamma: Vec<f64>,
    /// Aligned curve: d vectors, each length m.
    pub f_aligned: Vec<Vec<f64>>,
    /// Elastic distance after alignment.
    pub distance: f64,
}

/// Scale derivative vector at one point by 1/√‖f'‖, writing into result_dims.
#[inline]
fn srsf_scale_point(derivs: &[FdMatrix], result_dims: &mut [FdMatrix], i: usize, j: usize) {
    let d = derivs.len();
    let norm_sq: f64 = derivs.iter().map(|dd| dd[(i, j)].powi(2)).sum();
    let norm = norm_sq.sqrt();
    if norm < 1e-15 {
        for k in 0..d {
            result_dims[k][(i, j)] = 0.0;
        }
    } else {
        let scale = 1.0 / norm.sqrt();
        for k in 0..d {
            result_dims[k][(i, j)] = derivs[k][(i, j)] * scale;
        }
    }
}

/// Compute the SRSF transform for multidimensional (R^d) curves.
///
/// For f: \[0,1\] → R^d, the SRSF is q(t) = f'(t) / √‖f'(t)‖ where ‖·‖ is the
/// Euclidean norm in R^d. For d=1 this reduces to `sign(f') · √|f'|`.
///
/// # Arguments
/// * `data` — Set of n curves in R^d, each with m evaluation points
/// * `argvals` — Evaluation points (length m)
///
/// # Returns
/// `FdCurveSet` of SRSF values with the same shape as input.
pub fn srsf_transform_nd(data: &FdCurveSet, argvals: &[f64]) -> FdCurveSet {
    let d = data.ndim();
    let n = data.ncurves();
    let m = data.npoints();

    if d == 0 || n == 0 || m == 0 || argvals.len() != m {
        return FdCurveSet {
            dims: (0..d).map(|_| FdMatrix::zeros(n, m)).collect(),
        };
    }

    let derivs: Vec<FdMatrix> = data
        .dims
        .iter()
        .map(|dim_mat| crate::fdata::deriv_1d(dim_mat, argvals, 1))
        .collect();

    let mut result_dims: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(n, m)).collect();
    for i in 0..n {
        for j in 0..m {
            srsf_scale_point(&derivs, &mut result_dims, i, j);
        }
    }

    FdCurveSet { dims: result_dims }
}

/// Reconstruct an R^d curve from its SRSF.
///
/// Given d-dimensional SRSF vectors and initial point f0, reconstructs:
/// `f_k(t) = f0_k + ∫₀ᵗ q_k(s) · ‖q(s)‖ ds` for each dimension k.
///
/// # Arguments
/// * `q` — SRSF: d vectors, each length m
/// * `argvals` — Evaluation points (length m)
/// * `f0` — Initial values in R^d (length d)
///
/// # Returns
/// Reconstructed curve: d vectors, each length m.
pub fn srsf_inverse_nd(q: &[Vec<f64>], argvals: &[f64], f0: &[f64]) -> Vec<Vec<f64>> {
    let d = q.len();
    if d == 0 {
        return Vec::new();
    }
    let m = q[0].len();
    if m == 0 {
        return vec![Vec::new(); d];
    }

    // Compute ||q(t)|| at each time point
    let norms: Vec<f64> = (0..m)
        .map(|j| {
            let norm_sq: f64 = q.iter().map(|qk| qk[j].powi(2)).sum();
            norm_sq.sqrt()
        })
        .collect();

    // For each dimension, integrand = q_k(t) * ||q(t)||
    let mut result = Vec::with_capacity(d);
    for k in 0..d {
        let integrand: Vec<f64> = (0..m).map(|j| q[k][j] * norms[j]).collect();
        let integral = cumulative_trapz(&integrand, argvals);
        let curve: Vec<f64> = integral.iter().map(|&v| f0[k] + v).collect();
        result.push(curve);
    }

    result
}

/// Core DP alignment for R^d SRSFs.
///
/// Same DP grid and coprime neighborhood as `dp_alignment_core`, but edge weight
/// is the sum of `dp_edge_weight` over d dimensions.
fn dp_alignment_core_nd(
    q1: &[Vec<f64>],
    q2: &[Vec<f64>],
    argvals: &[f64],
    lambda: f64,
) -> Vec<f64> {
    let d = q1.len();
    let m = argvals.len();
    if m < 2 || d == 0 {
        return argvals.to_vec();
    }

    // For d=1, delegate to existing implementation for exact backward compat
    if d == 1 {
        return dp_alignment_core(&q1[0], &q2[0], argvals, lambda);
    }

    // Normalize each dimension's SRSF to unit L2 norm
    let q1n: Vec<Vec<f64>> = q1
        .iter()
        .map(|qk| {
            let norm = qk.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
            qk.iter().map(|&v| v / norm).collect()
        })
        .collect();
    let q2n: Vec<Vec<f64>> = q2
        .iter()
        .map(|qk| {
            let norm = qk.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-10);
            qk.iter().map(|&v| v / norm).collect()
        })
        .collect();

    let path = dp_grid_solve(m, m, |sr, sc, tr, tc| {
        let w: f64 = (0..d)
            .map(|k| dp_edge_weight(&q1n[k], &q2n[k], argvals, sc, tc, sr, tr))
            .sum();
        w + dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
    });

    dp_path_to_gamma(&path, argvals)
}

/// Align an R^d curve f2 to f1 using the elastic framework.
///
/// Finds the optimal warping γ (shared across all dimensions) such that
/// f2∘γ is as close as possible to f1 in the elastic metric.
///
/// # Arguments
/// * `f1` — Target curves (d dimensions)
/// * `f2` — Curves to align (d dimensions)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Penalty weight (0.0 = no penalty)
pub fn elastic_align_pair_nd(
    f1: &FdCurveSet,
    f2: &FdCurveSet,
    argvals: &[f64],
    lambda: f64,
) -> AlignmentResultNd {
    let d = f1.ndim();
    let m = f1.npoints();

    // Compute SRSFs
    let q1_set = srsf_transform_nd(f1, argvals);
    let q2_set = srsf_transform_nd(f2, argvals);

    // Extract first curve from each dimension
    let q1: Vec<Vec<f64>> = q1_set.dims.iter().map(|dm| dm.row(0)).collect();
    let q2: Vec<Vec<f64>> = q2_set.dims.iter().map(|dm| dm.row(0)).collect();

    // DP alignment using summed cost over dimensions
    let gamma = dp_alignment_core_nd(&q1, &q2, argvals, lambda);

    // Apply warping to f2 in each dimension
    let f_aligned: Vec<Vec<f64>> = f2
        .dims
        .iter()
        .map(|dm| {
            let row = dm.row(0);
            reparameterize_curve(&row, argvals, &gamma)
        })
        .collect();

    // Compute elastic distance: sum of squared L2 distances between aligned SRSFs
    let f_aligned_set = {
        let dims: Vec<FdMatrix> = f_aligned
            .iter()
            .map(|fa| {
                FdMatrix::from_slice(fa, 1, m).expect("dimension invariant: data.len() == n * m")
            })
            .collect();
        FdCurveSet { dims }
    };
    let q_aligned = srsf_transform_nd(&f_aligned_set, argvals);
    let weights = simpsons_weights(argvals);

    let mut dist_sq = 0.0;
    for k in 0..d {
        let q1k = q1_set.dims[k].row(0);
        let qak = q_aligned.dims[k].row(0);
        let d_k = l2_distance(&q1k, &qak, &weights);
        dist_sq += d_k * d_k;
    }

    AlignmentResultNd {
        gamma,
        f_aligned,
        distance: dist_sq.sqrt(),
    }
}

/// Elastic distance between two R^d curves.
///
/// Aligns f2 to f1 and returns the post-alignment SRSF distance.
pub fn elastic_distance_nd(f1: &FdCurveSet, f2: &FdCurveSet, argvals: &[f64], lambda: f64) -> f64 {
    elastic_align_pair_nd(f1, f2, argvals, lambda).distance
}

// ─── Karcher Mean for N-d Curves ─────────────────────────────────────────

/// Result of the Karcher mean computation for multidimensional (R^d) curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KarcherMeanResultNd {
    /// Karcher mean curve: d vectors of length m.
    pub mean: Vec<Vec<f64>>,
    /// SRSF of the Karcher mean: d vectors of length m.
    pub mean_srsf: Vec<Vec<f64>>,
    /// Final warping functions (n x m).
    pub gammas: FdMatrix,
    /// Curves aligned to the mean: d matrices, each n x m.
    pub aligned_data: Vec<FdMatrix>,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Result of PCA on aligned multidimensional (R^d) curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PcaNdResult {
    /// PC scores (n x ncomp).
    pub scores: FdMatrix,
    /// Principal components per dimension: d matrices, each ncomp x m.
    pub components: Vec<FdMatrix>,
    /// Explained variance for each component.
    pub explained_variance: Vec<f64>,
    /// Cumulative proportion of variance explained.
    pub cumulative_variance: Vec<f64>,
    /// Covariance eigenvalues (same as explained_variance for convenience).
    pub covariance_eigenvalues: Vec<f64>,
}

/// Compute SRSF for a single R^d curve (d vectors of length m).
fn srsf_single_nd(curve: &[Vec<f64>], argvals: &[f64]) -> Vec<Vec<f64>> {
    let m = argvals.len();
    let dims: Vec<FdMatrix> = curve
        .iter()
        .map(|c| FdMatrix::from_slice(c, 1, m).expect("dimension invariant: data.len() == n * m"))
        .collect();
    let cs = FdCurveSet { dims };
    let q_set = srsf_transform_nd(&cs, argvals);
    q_set.dims.iter().map(|dm| dm.row(0)).collect()
}

/// Compute the relative change between two N-d mean SRSFs.
fn relative_change_nd(old: &[Vec<f64>], new: &[Vec<f64>]) -> f64 {
    let mut diff_sq = 0.0;
    let mut old_sq = 0.0;
    for (qo, qn) in old.iter().zip(new.iter()) {
        for (&a, &b) in qo.iter().zip(qn.iter()) {
            diff_sq += (a - b).powi(2);
            old_sq += a * a;
        }
    }
    diff_sq.sqrt() / old_sq.sqrt().max(1e-10)
}

/// Select the curve whose SRSF is closest to the pointwise mean SRSF.
///
/// Returns the index of the template curve.
fn select_template_nd(data: &[FdCurveSet], srsfs: &[Vec<Vec<f64>>]) -> usize {
    let n = data.len();
    let d = srsfs[0].len();
    let m = srsfs[0][0].len();

    // Compute pointwise mean SRSF
    let mut mean_q: Vec<Vec<f64>> = vec![vec![0.0; m]; d];
    for q in srsfs {
        for k in 0..d {
            for j in 0..m {
                mean_q[k][j] += q[k][j];
            }
        }
    }
    for k in 0..d {
        for j in 0..m {
            mean_q[k][j] /= n as f64;
        }
    }

    // Find curve closest to mean
    let mut min_dist = f64::INFINITY;
    let mut min_idx = 0;
    for (i, q) in srsfs.iter().enumerate() {
        let mut dist_sq = 0.0;
        for k in 0..d {
            for j in 0..m {
                dist_sq += (q[k][j] - mean_q[k][j]).powi(2);
            }
        }
        if dist_sq < min_dist {
            min_dist = dist_sq;
            min_idx = i;
        }
    }
    min_idx
}

/// Compute the Karcher (Frechet) mean for multidimensional (R^d) curves.
///
/// Iteratively aligns all N-d curves to the current mean estimate in SRSF space,
/// computes the pointwise mean of aligned SRSFs per dimension, and reconstructs
/// the mean curve.
///
/// # Arguments
/// * `data` — Slice of n `FdCurveSet`s, each with d dimensions and m evaluation points
/// * `argvals` — Evaluation points (length m)
/// * `max_iter` — Maximum number of iterations
/// * `tol` — Convergence tolerance (relative SRSF change)
/// * `lambda` — Roughness penalty weight (0.0 = no penalty)
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if inputs are inconsistent.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_mean_nd(
    data: &[FdCurveSet],
    argvals: &[f64],
    max_iter: usize,
    tol: f64,
    lambda: f64,
) -> Result<KarcherMeanResultNd, FdarError> {
    let n = data.len();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves".to_string(),
            actual: format!("{n}"),
        });
    }

    let d = data[0].ndim();
    let m = data[0].npoints();
    if d == 0 || m < 2 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data/argvals",
            expected: format!("d > 0, m >= 2, argvals.len() == m (m={m})"),
            actual: format!("d={d}, m={m}, argvals.len()={}", argvals.len()),
        });
    }

    // Verify all curves have the same dimensions
    for (i, cs) in data.iter().enumerate() {
        if cs.ndim() != d || cs.npoints() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "data",
                expected: format!("all curves d={d}, m={m}"),
                actual: format!("curve {i}: d={}, m={}", cs.ndim(), cs.npoints()),
            });
        }
    }

    // Extract curves as Vec<Vec<f64>> (d vectors) per observation
    let curves: Vec<Vec<Vec<f64>>> = (0..n)
        .map(|i| data[i].dims.iter().map(|dm| dm.row(0)).collect())
        .collect();

    // Compute SRSFs for all curves
    let srsfs: Vec<Vec<Vec<f64>>> = curves.iter().map(|c| srsf_single_nd(c, argvals)).collect();

    // Select template (closest to mean SRSF)
    let template_idx = select_template_nd(data, &srsfs);
    let mut mu_q = srsfs[template_idx].clone();
    let mut mu_f = curves[template_idx].clone();

    // Iterative alignment loop
    let mut converged = false;
    let mut n_iter = 0;
    let mut gammas = FdMatrix::zeros(n, m);

    for iter in 0..max_iter {
        n_iter = iter + 1;

        // Align all curves to current mean (parallel)
        let align_results: Vec<(Vec<f64>, Vec<Vec<f64>>)> = iter_maybe_parallel!(0..n)
            .map(|i| {
                // Build single-curve FdCurveSet for mean and curve i
                let mean_cs = {
                    let dims: Vec<FdMatrix> = mu_f
                        .iter()
                        .map(|v| {
                            FdMatrix::from_slice(v, 1, m)
                                .expect("dimension invariant: data.len() == n * m")
                        })
                        .collect();
                    FdCurveSet { dims }
                };
                let curve_cs = {
                    let dims: Vec<FdMatrix> = curves[i]
                        .iter()
                        .map(|v| {
                            FdMatrix::from_slice(v, 1, m)
                                .expect("dimension invariant: data.len() == n * m")
                        })
                        .collect();
                    FdCurveSet { dims }
                };

                let result = elastic_align_pair_nd(&mean_cs, &curve_cs, argvals, lambda);
                (result.gamma, result.f_aligned)
            })
            .collect();

        // Store gammas and compute aligned SRSFs
        let mut new_mu_q: Vec<Vec<f64>> = vec![vec![0.0; m]; d];
        for (i, (gamma, f_aligned)) in align_results.iter().enumerate() {
            for j in 0..m {
                gammas[(i, j)] = gamma[j];
            }

            // Compute SRSF of aligned curve
            let q_aligned = srsf_single_nd(f_aligned, argvals);
            for k in 0..d {
                for j in 0..m {
                    new_mu_q[k][j] += q_aligned[k][j];
                }
            }
        }
        for k in 0..d {
            for j in 0..m {
                new_mu_q[k][j] /= n as f64;
            }
        }

        // Check convergence
        let rel = relative_change_nd(&mu_q, &new_mu_q);
        mu_q = new_mu_q;

        // Reconstruct mean curve from mean SRSF
        let f0: Vec<f64> = mu_f.iter().map(|v| v[0]).collect();
        mu_f = srsf_inverse_nd(&mu_q, argvals, &f0);

        if rel < tol {
            converged = true;
            break;
        }
    }

    // Post-centering: center the warps via sqrt_mean_inverse
    let gam_inv = super::sqrt_mean_inverse(&gammas, argvals);
    for i in 0..n {
        let gam_i: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        let gam_centered = reparameterize_curve(&gam_i, argvals, &gam_inv);
        for j in 0..m {
            gammas[(i, j)] = gam_centered[j];
        }
    }

    // Recompute aligned data using final centered warps
    let mut aligned_data: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(n, m)).collect();
    for i in 0..n {
        let gamma_i: Vec<f64> = (0..m).map(|j| gammas[(i, j)]).collect();
        for k in 0..d {
            let f_aligned = reparameterize_curve(&curves[i][k], argvals, &gamma_i);
            for j in 0..m {
                aligned_data[k][(i, j)] = f_aligned[j];
            }
        }
    }

    // Recompute mean from final aligned data
    let mut mean: Vec<Vec<f64>> = vec![vec![0.0; m]; d];
    for k in 0..d {
        for j in 0..m {
            for i in 0..n {
                mean[k][j] += aligned_data[k][(i, j)];
            }
            mean[k][j] /= n as f64;
        }
    }

    // Recompute mean SRSF
    let mean_srsf = srsf_single_nd(&mean, argvals);

    Ok(KarcherMeanResultNd {
        mean,
        mean_srsf,
        gammas,
        aligned_data,
        n_iter,
        converged,
    })
}

/// Compute the cross-dimensional covariance matrix of aligned N-d curves.
///
/// Stacks all d dimensions of aligned curves into a single (n x d*m) matrix,
/// centers columns, and returns X^T X / (n-1) as a (d*m x d*m) covariance matrix.
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if d*m exceeds 10000 (to prevent
/// excessive memory usage) or if input dimensions are inconsistent.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn karcher_covariance_nd(
    result: &KarcherMeanResultNd,
    argvals: &[f64],
) -> Result<FdMatrix, FdarError> {
    let d = result.aligned_data.len();
    if d == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_data",
            expected: "d > 0".to_string(),
            actual: "0".to_string(),
        });
    }
    let (n, m) = result.aligned_data[0].shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    let dm = d * m;
    if dm > 10_000 {
        return Err(FdarError::InvalidParameter {
            parameter: "d*m",
            message: format!(
                "d*m = {dm} exceeds limit of 10000; covariance matrix would be too large"
            ),
        });
    }

    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_data",
            expected: "n >= 2".to_string(),
            actual: format!("{n}"),
        });
    }

    // Build (n x dm) stacked matrix
    let mut stacked = FdMatrix::zeros(n, dm);
    for k in 0..d {
        for i in 0..n {
            for j in 0..m {
                stacked[(i, k * m + j)] = result.aligned_data[k][(i, j)];
            }
        }
    }

    // Center columns
    let mut col_mean = vec![0.0; dm];
    for j in 0..dm {
        for i in 0..n {
            col_mean[j] += stacked[(i, j)];
        }
        col_mean[j] /= n as f64;
    }
    for i in 0..n {
        for j in 0..dm {
            stacked[(i, j)] -= col_mean[j];
        }
    }

    // Compute covariance: X^T X / (n-1)
    let nf = (n - 1) as f64;
    let mut cov = FdMatrix::zeros(dm, dm);
    for p in 0..dm {
        for q in p..dm {
            let mut s = 0.0;
            for i in 0..n {
                s += stacked[(i, p)] * stacked[(i, q)];
            }
            s /= nf;
            cov[(p, q)] = s;
            cov[(q, p)] = s;
        }
    }

    Ok(cov)
}

/// Perform PCA on aligned multidimensional (R^d) curves.
///
/// Stacks aligned data from all d dimensions into an (n x d*m) matrix, centers,
/// computes the SVD, and extracts principal components and scores.
///
/// # Arguments
/// * `result` — Pre-computed Karcher mean result for N-d curves
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal components to extract
///
/// # Errors
/// Returns `FdarError` if inputs are invalid or SVD fails.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn pca_nd(
    result: &KarcherMeanResultNd,
    argvals: &[f64],
    ncomp: usize,
) -> Result<PcaNdResult, FdarError> {
    let d = result.aligned_data.len();
    if d == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_data",
            expected: "d > 0".to_string(),
            actual: "0".to_string(),
        });
    }
    let (n, m) = result.aligned_data[0].shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_data/argvals/ncomp",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={n}, m={m}, ncomp={ncomp}, argvals.len()={}",
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1);
    let dm = d * m;

    // Build (n x dm) stacked matrix and center
    let mut stacked = FdMatrix::zeros(n, dm);
    for k in 0..d {
        for i in 0..n {
            for j in 0..m {
                stacked[(i, k * m + j)] = result.aligned_data[k][(i, j)];
            }
        }
    }

    // Center columns
    let mut col_mean = vec![0.0; dm];
    for j in 0..dm {
        for i in 0..n {
            col_mean[j] += stacked[(i, j)];
        }
        col_mean[j] /= n as f64;
    }
    for i in 0..n {
        for j in 0..dm {
            stacked[(i, j)] -= col_mean[j];
        }
    }

    // Economy SVD: compute Gram matrix G = X X^T / (n-1), size n x n
    // (much smaller than dm x dm when dm >> n)
    let nf = (n - 1) as f64;
    let mut gram = FdMatrix::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for p in 0..dm {
                s += stacked[(i, p)] * stacked[(j, p)];
            }
            s /= nf;
            gram[(i, j)] = s;
            gram[(j, i)] = s;
        }
    }

    // Eigen-decompose Gram matrix via nalgebra SVD (symmetric, so SVD = eigendecomposition)
    use nalgebra::SVD;
    let svd = SVD::new(gram.to_dmatrix(), true, true);
    let u = svd.u.as_ref().ok_or_else(|| FdarError::ComputationFailed {
        operation: "SVD",
        detail: "SVD failed to compute U matrix for Gram matrix".to_string(),
    })?;

    // Eigenvalues of Gram = singular values of Gram = eigenvalues of X X^T / (n-1)
    // These are also the eigenvalues of the covariance matrix (for the top n components)
    let eigenvalues: Vec<f64> = svd.singular_values.iter().take(ncomp).copied().collect();

    // Scores: score_ik = u_ik * sqrt(lambda_k * (n-1))
    // Since Gram = X X^T / (n-1), and SVD(Gram) = U S U^T,
    // the scores of the data are: X V = U * sqrt(S * (n-1))
    let mut scores = FdMatrix::zeros(n, ncomp);
    for k in 0..ncomp {
        let scale = (eigenvalues[k] * nf).sqrt();
        for i in 0..n {
            scores[(i, k)] = u[(i, k)] * scale;
        }
    }

    // Loadings: V_k = X^T U_k / sqrt(lambda_k * (n-1))
    // Reshape back into d matrices of (ncomp x m)
    let mut components: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(ncomp, m)).collect();
    for k in 0..ncomp {
        let scale = (eigenvalues[k] * nf).sqrt().max(1e-15);
        let mut loading = vec![0.0; dm];
        for p in 0..dm {
            let mut s = 0.0;
            for i in 0..n {
                s += stacked[(i, p)] * u[(i, k)];
            }
            loading[p] = s / scale;
        }

        // Distribute into per-dimension matrices
        for dim in 0..d {
            for j in 0..m {
                components[dim][(k, j)] = loading[dim * m + j];
            }
        }
    }

    // Cumulative variance
    let total_var: f64 = svd.singular_values.iter().sum();
    let mut cumulative_variance = Vec::with_capacity(ncomp);
    let mut running = 0.0;
    for ev in &eigenvalues {
        running += ev;
        cumulative_variance.push(if total_var > 0.0 {
            running / total_var
        } else {
            0.0
        });
    }

    // Explained variance = eigenvalues
    let explained_variance = eigenvalues.clone();
    let covariance_eigenvalues = eigenvalues;

    Ok(PcaNdResult {
        scores,
        components,
        explained_variance,
        cumulative_variance,
        covariance_eigenvalues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build n identical R^2 curves (circle-like) as FdCurveSets.
    fn make_identical_curves(n: usize, m: usize) -> (Vec<FdCurveSet>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let dim0: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let dim1: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).cos()).collect();

        let data: Vec<FdCurveSet> = (0..n)
            .map(|_| {
                let m0 = FdMatrix::from_slice(&dim0, 1, m)
                    .expect("dimension invariant: data.len() == n * m");
                let m1 = FdMatrix::from_slice(&dim1, 1, m)
                    .expect("dimension invariant: data.len() == n * m");
                FdCurveSet { dims: vec![m0, m1] }
            })
            .collect();
        (data, t)
    }

    /// Build n shifted R^2 sine curves.
    fn make_shifted_curves(n: usize, m: usize) -> (Vec<FdCurveSet>, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let data: Vec<FdCurveSet> = (0..n)
            .map(|i| {
                let shift = 0.05 * (i as f64 - n as f64 / 2.0);
                let dim0: Vec<f64> = t
                    .iter()
                    .map(|&ti| (2.0 * PI * (ti + shift)).sin())
                    .collect();
                let dim1: Vec<f64> = t
                    .iter()
                    .map(|&ti| (2.0 * PI * (ti + shift)).cos())
                    .collect();
                let m0 = FdMatrix::from_slice(&dim0, 1, m)
                    .expect("dimension invariant: data.len() == n * m");
                let m1 = FdMatrix::from_slice(&dim1, 1, m)
                    .expect("dimension invariant: data.len() == n * m");
                FdCurveSet { dims: vec![m0, m1] }
            })
            .collect();
        (data, t)
    }

    #[test]
    fn karcher_mean_nd_identical_curves() {
        let (data, t) = make_identical_curves(5, 31);
        let result = karcher_mean_nd(&data, &t, 10, 1e-4, 0.0).expect("should succeed");

        let d = 2;
        let m = 31;

        // Mean should be close to the input curves
        let input_dim0: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
        let input_dim1: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).cos()).collect();

        let max_diff_0: f64 = result.mean[0]
            .iter()
            .zip(input_dim0.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let max_diff_1: f64 = result.mean[1]
            .iter()
            .zip(input_dim1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        assert!(
            max_diff_0 < 0.3,
            "Mean dim 0 should be close to input, max diff = {max_diff_0}"
        );
        assert!(
            max_diff_1 < 0.3,
            "Mean dim 1 should be close to input, max diff = {max_diff_1}"
        );

        // Gammas should be near-identity
        let n = 5;
        for i in 0..n {
            for j in 0..m {
                let diff = (result.gammas[(i, j)] - t[j]).abs();
                assert!(
                    diff < 0.15,
                    "Warp for identical curves should be near identity: gamma[{i},{j}] diff = {diff}"
                );
            }
        }

        // Correct number of dimensions
        assert_eq!(result.mean.len(), d);
        assert_eq!(result.mean_srsf.len(), d);
        assert_eq!(result.aligned_data.len(), d);
    }

    #[test]
    fn karcher_mean_nd_output_dimensions() {
        let (data, t) = make_shifted_curves(8, 25);
        let result = karcher_mean_nd(&data, &t, 5, 1e-3, 0.0).expect("should succeed");

        let n = 8;
        let m = 25;
        let d = 2;

        assert_eq!(result.mean.len(), d);
        assert_eq!(result.mean_srsf.len(), d);
        for k in 0..d {
            assert_eq!(result.mean[k].len(), m);
            assert_eq!(result.mean_srsf[k].len(), m);
        }
        assert_eq!(result.gammas.shape(), (n, m));
        assert_eq!(result.aligned_data.len(), d);
        for k in 0..d {
            assert_eq!(result.aligned_data[k].shape(), (n, m));
        }
        assert!(result.n_iter <= 5);
    }

    #[test]
    fn karcher_mean_nd_convergence() {
        let (data, t) = make_shifted_curves(10, 31);
        let result = karcher_mean_nd(&data, &t, 20, 1e-3, 0.0).expect("should succeed");

        // With well-behaved shifted sine curves, algorithm should converge
        assert!(
            result.converged,
            "Algorithm should converge for shifted sine curves, n_iter={}",
            result.n_iter
        );
    }

    #[test]
    fn pca_nd_basic_properties() {
        let (data, t) = make_shifted_curves(10, 31);
        let km = karcher_mean_nd(&data, &t, 10, 1e-3, 0.0).expect("karcher_mean should succeed");
        let pca = pca_nd(&km, &t, 3).expect("pca_nd should succeed");

        let n = 10;
        let ncomp = 3;
        let m = 31;

        // Scores shape
        assert_eq!(pca.scores.shape(), (n, ncomp));

        // Components shape: d=2, each ncomp x m
        assert_eq!(pca.components.len(), 2);
        for comp in &pca.components {
            assert_eq!(comp.shape(), (ncomp, m));
        }

        // Explained variance: non-negative
        for ev in &pca.explained_variance {
            assert!(
                *ev >= -1e-10,
                "Explained variance should be non-negative: {ev}"
            );
        }

        // Explained variance should be approximately decreasing
        for i in 1..pca.explained_variance.len() {
            assert!(
                pca.explained_variance[i] <= pca.explained_variance[i - 1] + 1e-8,
                "Explained variance should be decreasing: {} > {}",
                pca.explained_variance[i],
                pca.explained_variance[i - 1]
            );
        }

        // Cumulative variance should be increasing
        for i in 1..pca.cumulative_variance.len() {
            assert!(
                pca.cumulative_variance[i] >= pca.cumulative_variance[i - 1] - 1e-10,
                "Cumulative variance should be increasing"
            );
        }
    }

    #[test]
    fn karcher_covariance_nd_symmetric() {
        let (data, t) = make_shifted_curves(8, 21);
        let km = karcher_mean_nd(&data, &t, 5, 1e-3, 0.0).expect("karcher_mean should succeed");
        let cov = karcher_covariance_nd(&km, &t).expect("covariance should succeed");

        let dm = 2 * 21;
        assert_eq!(cov.shape(), (dm, dm));

        // Verify symmetry
        for p in 0..dm {
            for q in p..dm {
                let diff = (cov[(p, q)] - cov[(q, p)]).abs();
                assert!(
                    diff < 1e-12,
                    "Covariance should be symmetric at ({p},{q}): diff = {diff}"
                );
            }
        }
    }
}
