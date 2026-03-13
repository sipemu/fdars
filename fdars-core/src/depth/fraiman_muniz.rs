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
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::depth::fraiman_muniz_1d;
///
/// let data = FdMatrix::from_column_major(
///     (0..50).map(|i| (i as f64 * 0.1).sin()).collect(),
///     5, 10,
/// ).unwrap();
/// let depths = fraiman_muniz_1d(&data, &data, true);
/// assert_eq!(depths.len(), 5);
/// // Depths should be in [0, 1]
/// assert!(depths.iter().all(|&d| d >= 0.0 && d <= 1.0 + 1e-10));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fraiman_muniz_1d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    if data_obj.nrows() == 0 || data_ori.nrows() == 0 || data_obj.ncols() == 0 {
        return Vec::new();
    }
    let state = SortedReferenceState::from_reference(data_ori);
    let streaming = StreamingFraimanMuniz::new(state, scale);
    streaming.depth_batch(data_obj)
}

/// Compute Fraiman-Muniz depth for 2D functional data (surfaces).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fraiman_muniz_2d(data_obj: &FdMatrix, data_ori: &FdMatrix, scale: bool) -> Vec<f64> {
    // Same implementation as 1D - iterate over all grid points
    fraiman_muniz_1d(data_obj, data_ori, scale)
}
