//! Distance metrics and semimetrics for functional data.
//!
//! This module provides various distance measures including:
//! - Lp distances (L1, L2, L∞)
//! - Hausdorff distance
//! - Dynamic Time Warping (DTW)
//! - Fourier-based semimetric
//! - Horizontal shift semimetric
//! - Soft-DTW (differentiable DTW relaxation)
//! - Kullback-Leibler divergence

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

pub mod dtw;
pub mod fourier;
pub mod hausdorff;
pub mod hshift;
pub mod lp;
pub mod soft_dtw;

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

// ---------------------------------------------------------------------------
// Re-exports — preserves the external API
// ---------------------------------------------------------------------------

pub use dtw::{dtw_cross_1d, dtw_distance, dtw_self_1d};
pub use fourier::{fourier_cross_1d, fourier_self_1d};
pub use hausdorff::{
    hausdorff_3d, hausdorff_cross_1d, hausdorff_cross_2d, hausdorff_self_1d, hausdorff_self_2d,
};
pub use hshift::{hshift_cross_1d, hshift_self_1d};
pub use lp::{lp_cross_1d, lp_cross_2d, lp_self_1d, lp_self_2d};
pub use soft_dtw::{
    soft_dtw_barycenter, soft_dtw_cross_1d, soft_dtw_distance, soft_dtw_div_cross_1d,
    soft_dtw_div_self_1d, soft_dtw_divergence, soft_dtw_self_1d, SoftDtwBarycenterResult,
};
