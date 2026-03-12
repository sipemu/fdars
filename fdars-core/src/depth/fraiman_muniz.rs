//! Fraiman-Muniz depth measures.

use crate::matrix::FdMatrix;
use crate::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz};

/// Compute Fraiman-Muniz depth for 1D functional data.
///
/// Uses the FM1 formula: d = 1 - |0.5 - Fn(x)|
/// With scale=true: d = 2 * min(Fn(x), 1-Fn(x))
///
/// # Arguments
/// * `data_obj` - Data to compute depth for (nobj x n_points)
/// * `data_ori` - Reference data (nori x n_points)
/// * `scale` - Whether to scale the depth values
pub fn fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() == 0 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = SortedReferenceState::from_reference(data_ori);
    let streaming = StreamingFraimanMuniz::new(state, scale);
    streaming.depth_batch(data_obj)
}

/// Compute Fraiman-Muniz depth for 2D functional data (surfaces).
pub fn fraiman_muniz_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // Same implementation as 1D - iterate over all grid points
    fraiman_muniz_1d(data_obj, data_ori, scale)
}
