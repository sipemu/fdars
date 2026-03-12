//! Multidimensional (R^d) SRSF transforms and elastic alignment.

use super::srsf::reparameterize_curve;
use super::{
    dp_alignment_core, dp_edge_weight, dp_grid_solve, dp_lambda_penalty, dp_path_to_gamma,
};
use crate::helpers::{cumulative_trapz, l2_distance, simpsons_weights};
use crate::matrix::{FdCurveSet, FdMatrix};

/// Result of aligning multidimensional (R^d) curves.
#[derive(Debug, Clone)]
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
