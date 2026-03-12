//! Streaming / online depth computation for functional data.
//!
//! This module decouples reference-set construction from query evaluation,
//! enabling efficient depth computation in streaming scenarios:
//!
//! - [`SortedReferenceState`] pre-sorts reference values per time point for O(log N) rank queries.
//! - [`StreamingMbd`] uses a rank-based combinatorial identity to compute Modified Band Depth
//!   in O(T log N) per query instead of O(N^2 T).
//! - [`StreamingFraimanMuniz`] computes Fraiman-Muniz depth via binary search on sorted columns.
//! - [`StreamingBd`] computes Band Depth with decoupled reference and early-exit optimisation.
//! - [`RollingReference`] maintains a sliding window of reference curves with incremental
//!   sorted-column updates.

use crate::matrix::FdMatrix;

pub mod bd;
pub mod fraiman_muniz;
pub mod mbd;
pub mod rolling;
pub mod sorted_ref;

#[cfg(test)]
mod tests;

// Re-export all public types and traits
pub use bd::{FullReferenceState, StreamingBd};
pub use fraiman_muniz::StreamingFraimanMuniz;
pub use mbd::StreamingMbd;
pub use rolling::RollingReference;
pub use sorted_ref::SortedReferenceState;

// ---------------------------------------------------------------------------
// Shared trait and helper
// ---------------------------------------------------------------------------

/// Trait for streaming depth estimators backed by a pre-built reference state.
pub trait StreamingDepth {
    /// Depth of a single curve given as a contiguous `&[f64]` of length `n_points`.
    fn depth_one(&self, curve: &[f64]) -> f64;

    /// Batch depth for a matrix of query curves (`nobj x n_points`).
    fn depth_batch(&self, data_obj: &FdMatrix) -> Vec<f64>;

    /// Number of evaluation points.
    fn n_points(&self) -> usize;

    /// Number of reference observations backing this estimator.
    fn n_reference(&self) -> usize;
}

/// Helper: choose-2 combinator
#[inline]
pub(super) fn c2(k: usize) -> usize {
    k * k.wrapping_sub(1) / 2
}
