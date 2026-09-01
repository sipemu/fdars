//! Random projection depth measures.

use crate::dim::Dim;
use crate::matrix::FdMatrix;

use super::random_depth_core;

/// Compute random projection depth for 1D functional data.
///
/// Projects curves to scalars using random projections and computes
/// average univariate depth.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nproj` - Number of random projections
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::depth::random_projection_1d;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let depths = random_projection_1d(&data, &data, 50);
/// assert_eq!(depths.len(), 5);
/// assert!(depths.iter().all(|&d| d >= 0.0));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_projection_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_projection_1d_seeded(data_obj, data_ori, nproj, None)
}

/// Compute random projection depth with optional seed for reproducibility.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_projection_1d_seeded(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    nproj: usize,
    seed: Option<u64>,
) -> Vec<f64> {
    random_depth_core(
        data_obj,
        data_ori,
        nproj,
        seed,
        0.0,
        |acc, d| acc + d,
        |acc, n| acc / n as f64,
    )
}

/// Compute random projection depth for 1D or 2D functional data via a unified [`Dim`] dispatch.
///
/// The 2D path never diverged from the 1D one, so both [`Dim`] arms forward to
/// [`random_projection_1d`]. Because `random_projection_1d` draws fresh entropy
/// from `thread_rng()` (no public seed), the forwarding is a compile-time
/// guarantee (single-arm `match`), not a runtime-equality one; use
/// [`random_projection_1d_seeded`] for reproducible results.
///
/// # Arguments
/// * `data_obj` - Data to compute depth for
/// * `data_ori` - Reference data
/// * `nproj` - Number of random projections
/// * `dim` - Dimensionality selector ([`Dim::One`] or [`Dim::Two`])
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_projection(
    data_obj: &FdMatrix,
    data_ori: &FdMatrix,
    nproj: usize,
    dim: Dim,
) -> Vec<f64> {
    match dim {
        Dim::One | Dim::Two => random_projection_1d(data_obj, data_ori, nproj),
    }
}

/// Compute random projection depth for 2D functional data.
#[deprecated(
    since = "0.30.0",
    note = "redundant with `random_projection(…, Dim::Two)`; body just forwards to `random_projection_1d`"
)]
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_projection_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_projection_1d(data_obj, data_ori, nproj)
}
