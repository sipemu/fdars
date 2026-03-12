//! Random projection depth measures.

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

/// Compute random projection depth for 2D functional data.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn random_projection_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, nproj: usize) -> Vec<f64> {
    random_projection_1d(data_obj, data_ori, nproj)
}
