//! Distance metrics and semimetrics for functional data.
//!
//! This module provides various distance measures including:
//! - Lp distances (L1, L2, L∞)
//! - Hausdorff distance
//! - Dynamic Time Warping (DTW), warping paths, and Sakoe-Chiba / Itakura windows
//! - DTW lower bounds (LB_Kim, LB_Keogh, LB_Webb), early-abandoning pruned DTW
//!   (EAPrunedDTW) and exact 1-NN search, see [`dtw_lb`]
//! - DTW Barycenter Averaging (DBA, Petitjean et al. 2011)
//! - Fourier-based semimetric
//! - Horizontal shift semimetric
//! - Soft-DTW (differentiable DTW relaxation)
//! - PCA-based semimetric
//! - Derivative-based semimetric
//! - Basis coefficient semimetric
//! - Kullback-Leibler divergence
//! - The aeon elastic-distance family (DDTW, WDTW, WDDTW, ADTW, ERP, EDR, TWE, MSM,
//!   LCSS, shape DTW), see [`elastic_family`]
//! - Multivariate (multichannel, `d × m` per series) DTW, DBA, elastic distances
//!   and DTW k-means (`*_multivariate`), following aeon / tslearn's dependent
//!   multichannel semantics

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

pub mod basis_coef;
pub mod dba;
pub mod deriv;
pub mod dtw;
pub mod dtw_lb;
pub mod elastic_family;
pub mod fourier;
pub mod gak;
pub mod hausdorff;
pub mod hshift;
pub mod kl;
pub mod lp;
pub mod pca;
pub mod sbd;
pub mod soft_dtw;
pub mod window;

#[cfg(test)]
mod multivariate_goldens;
#[cfg(test)]
mod multivariate_tests;
#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------
// Shared helpers (used by multiple submodules)
// ---------------------------------------------------------------------------

/// Compute weighted Lp distance between two rows of an FdMatrix, branching on p.
///
/// Avoids `powf(p)` in the inner loop for p=1 and p=2, which are ~50-100× faster.
#[inline(always)]
pub(super) fn lp_weighted_distance(
    data1: &FdMatrix,
    i: usize,
    data2: &FdMatrix,
    j: usize,
    weights: &[f64],
    n_points: usize,
    p: f64,
) -> f64 {
    if (p - 2.0).abs() < 1e-14 {
        let mut sum = 0.0;
        for k in 0..n_points {
            let diff = data1[(i, k)] - data2[(j, k)];
            sum += diff * diff * weights[k];
        }
        sum.sqrt()
    } else if (p - 1.0).abs() < 1e-14 {
        let mut sum = 0.0;
        for k in 0..n_points {
            sum += (data1[(i, k)] - data2[(j, k)]).abs() * weights[k];
        }
        sum
    } else {
        let mut sum = 0.0;
        for k in 0..n_points {
            sum += (data1[(i, k)] - data2[(j, k)]).abs().powf(p) * weights[k];
        }
        sum.powf(1.0 / p)
    }
}

/// Merge base integration weights with optional user weights.
pub(super) fn merge_weights(base: Vec<f64>, user_weights: &[f64]) -> Vec<f64> {
    if user_weights.len() == base.len() {
        base.iter()
            .zip(user_weights.iter())
            .map(|(b, u)| b * u)
            .collect()
    } else {
        base
    }
}

/// Build a symmetric distance matrix by computing upper-triangle entries in parallel.
#[inline]
pub(super) fn self_distance_matrix(
    n: usize,
    compute: impl Fn(usize, usize) -> f64 + Sync,
) -> FdMatrix {
    // Pre-allocate flat buffer for upper triangle: n*(n-1)/2 entries
    let tri_len = n * (n - 1) / 2;
    let upper_vals: Vec<f64> = iter_maybe_parallel!(0..n)
        .flat_map(|i| ((i + 1)..n).map(|j| compute(i, j)).collect::<Vec<_>>())
        .collect();
    debug_assert_eq!(upper_vals.len(), tri_len);
    // Scatter into symmetric matrix
    let mut dist = FdMatrix::zeros(n, n);
    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let d = upper_vals[idx];
            dist[(i, j)] = d;
            dist[(j, i)] = d;
            idx += 1;
        }
    }
    dist
}

/// Build an n1×n2 distance matrix by computing all pairs in parallel.
#[inline]
pub(super) fn cross_distance_matrix(
    n1: usize,
    n2: usize,
    compute: impl Fn(usize, usize) -> f64 + Sync,
) -> FdMatrix {
    // Pre-allocate flat buffer for all n1*n2 entries (row-major order)
    let vals: Vec<f64> = iter_maybe_parallel!(0..n1)
        .flat_map(|i| (0..n2).map(|j| compute(i, j)).collect::<Vec<_>>())
        .collect();
    // Scatter row-major vals into column-major FdMatrix
    let mut dist = FdMatrix::zeros(n1, n2);
    for i in 0..n1 {
        for j in 0..n2 {
            dist[(i, j)] = vals[i * n2 + j];
        }
    }
    dist
}

/// Distance matrix from a fallible pair function, computed in parallel.
///
/// `n2 = None` builds the symmetric `n1 × n1` self-distance matrix (zero
/// diagonal, only `i < j` evaluated), otherwise the full `n1 × n2` matrix.
pub(super) fn pairwise_result(
    n1: usize,
    n2: Option<usize>,
    compute: impl Fn(usize, usize) -> Result<f64, crate::error::FdarError> + Sync,
) -> Result<FdMatrix, crate::error::FdarError> {
    let cols = n2.unwrap_or(n1);
    let rows: Vec<Vec<f64>> = iter_maybe_parallel!(0..n1)
        .map(|i| {
            let start = if n2.is_none() { i + 1 } else { 0 };
            (start..cols).map(|j| compute(i, j)).collect()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut dist = FdMatrix::zeros(n1, cols);
    for (i, row) in rows.iter().enumerate() {
        let start = if n2.is_none() { i + 1 } else { 0 };
        for (off, &d) in row.iter().enumerate() {
            let j = start + off;
            dist[(i, j)] = d;
            if n2.is_none() {
                dist[(j, i)] = d;
            }
        }
    }
    Ok(dist)
}

// ---------------------------------------------------------------------------
// Re-exports — preserves the external API
// ---------------------------------------------------------------------------

pub use basis_coef::{basis_coef_cross, basis_coef_self};
pub use dba::{
    dtw_barycenter_averaging, dtw_barycenter_averaging_multivariate, DbaConfig, DbaInit,
    DbaMultivariateResult, DbaResult,
};
pub use deriv::{deriv_cross, deriv_self};
pub use dtw::{
    dtw_cross, dtw_distance, dtw_distance_multivariate, dtw_distance_windowed,
    dtw_pairwise_distance_multivariate, dtw_path, dtw_path_multivariate, dtw_self, DtwPath,
};
pub use dtw_lb::{
    dtw_distance_early_abandon, dtw_k_nearest_neighbors, dtw_nearest_neighbor, lb_enhanced,
    lb_envelope, lb_envelope_windowed, lb_improved, lb_keogh, lb_kim, lb_webb, DtwEnvelope,
    DtwKNearestNeighbors, DtwNearestNeighbors, DtwSearchStats,
};
pub use elastic_family::{
    adtw_alignment_path, adtw_alignment_path_multivariate, adtw_cost_matrix,
    adtw_cost_matrix_multivariate, adtw_distance, adtw_distance_multivariate,
    adtw_pairwise_distance, adtw_pairwise_distance_multivariate, ddtw_alignment_path,
    ddtw_alignment_path_multivariate, ddtw_cost_matrix, ddtw_cost_matrix_multivariate,
    ddtw_distance, ddtw_distance_multivariate, ddtw_pairwise_distance,
    ddtw_pairwise_distance_multivariate, edr_alignment_path, edr_alignment_path_multivariate,
    edr_cost_matrix, edr_cost_matrix_multivariate, edr_distance, edr_distance_multivariate,
    edr_pairwise_distance, edr_pairwise_distance_multivariate, erp_alignment_path,
    erp_alignment_path_multivariate, erp_cost_matrix, erp_cost_matrix_multivariate, erp_distance,
    erp_distance_multivariate, erp_pairwise_distance, erp_pairwise_distance_multivariate,
    lcss_alignment_path, lcss_alignment_path_multivariate, lcss_cost_matrix,
    lcss_cost_matrix_multivariate, lcss_distance, lcss_distance_multivariate,
    lcss_pairwise_distance, lcss_pairwise_distance_multivariate, msm_alignment_path,
    msm_alignment_path_multivariate, msm_cost_matrix, msm_cost_matrix_multivariate, msm_distance,
    msm_distance_multivariate, msm_pairwise_distance, msm_pairwise_distance_multivariate,
    shape_dtw_alignment_path, shape_dtw_alignment_path_multivariate, shape_dtw_cost_matrix,
    shape_dtw_cost_matrix_multivariate, shape_dtw_distance, shape_dtw_distance_multivariate,
    shape_dtw_pairwise_distance, shape_dtw_pairwise_distance_multivariate, twe_alignment_path,
    twe_alignment_path_multivariate, twe_cost_matrix, twe_cost_matrix_multivariate, twe_distance,
    twe_distance_multivariate, twe_pairwise_distance, twe_pairwise_distance_multivariate,
    wddtw_alignment_path, wddtw_alignment_path_multivariate, wddtw_cost_matrix,
    wddtw_cost_matrix_multivariate, wddtw_distance, wddtw_distance_multivariate,
    wddtw_pairwise_distance, wddtw_pairwise_distance_multivariate, wdtw_alignment_path,
    wdtw_alignment_path_multivariate, wdtw_cost_matrix, wdtw_cost_matrix_multivariate,
    wdtw_distance, wdtw_distance_multivariate, wdtw_pairwise_distance,
    wdtw_pairwise_distance_multivariate, ElasticDistancePath, ShapeDtwDescriptor,
};
pub use fourier::{fourier_cross, fourier_self};
pub use gak::{
    gak, gak_gram_matrix, gak_gram_predict, gak_gram_train, gak_triangular, sigma_gak, GakConfig,
    GakGramTrain,
};
pub use hausdorff::{hausdorff_3d, hausdorff_cross, hausdorff_self};
pub use hshift::{hshift_cross, hshift_self};
pub use kl::{kl_cross, kl_self};
pub use lp::{lp_cross, lp_self, LpDomain};
pub use pca::{pca_cross, pca_self};
pub use sbd::{sbd, sbd_distance_matrix, SbdResult};
pub use soft_dtw::{
    soft_dtw_alignment, soft_dtw_barycenter, soft_dtw_barycenter_with_config, soft_dtw_cross,
    soft_dtw_distance, soft_dtw_distance_generic, soft_dtw_distance_windowed, soft_dtw_div_cross,
    soft_dtw_div_self, soft_dtw_divergence, soft_dtw_self, SoftDtwAlignment,
    SoftDtwBarycenterConfig, SoftDtwBarycenterResult,
};
pub use window::DtwWindow;

// Test-only crate-internal re-exports of the underlying `_1d` primitives, so the
// inline `#[cfg(test)]` modules that pin bit-identity against the dispatchers keep
// resolving them via `super::*` after the public surface consolidated onto the
// `Dim`-dispatched entry points. Not needed by non-test code.
#[cfg(test)]
pub(crate) use basis_coef::{basis_coef_cross_1d, basis_coef_self_1d};
#[cfg(test)]
pub(crate) use deriv::{deriv_cross_1d, deriv_self_1d};
#[cfg(test)]
pub(crate) use dtw::{dtw_cross_1d, dtw_self_1d};
#[cfg(test)]
pub(crate) use fourier::{fourier_cross_1d, fourier_self_1d};
#[cfg(test)]
pub(crate) use hshift::{hshift_cross_1d, hshift_self_1d};
#[cfg(test)]
pub(crate) use kl::{kl_cross_1d, kl_self_1d};
#[cfg(test)]
pub(crate) use pca::{pca_cross_1d, pca_self_1d};
#[cfg(test)]
pub(crate) use soft_dtw::{soft_dtw_cross_1d, soft_dtw_div_self_1d, soft_dtw_self_1d};
