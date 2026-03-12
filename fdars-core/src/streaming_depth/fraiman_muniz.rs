//! Streaming Fraiman-Muniz depth estimator.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::sorted_ref::SortedReferenceState;
use super::StreamingDepth;

/// Streaming Fraiman-Muniz depth estimator.
///
/// Uses binary search on sorted columns to compute the empirical CDF at each
/// time point: Fn(x) = #{ref <= x} / N.
///
/// Per-query complexity: **O(T x log N)** instead of O(T x N).
#[derive(Debug, Clone)]
pub struct StreamingFraimanMuniz {
    state: SortedReferenceState,
    scale: bool,
}

impl StreamingFraimanMuniz {
    pub fn new(state: SortedReferenceState, scale: bool) -> Self {
        Self { state, scale }
    }

    #[inline]
    fn fm_one_inner(&self, curve: &[f64]) -> f64 {
        let n = self.state.nori;
        if n == 0 {
            return 0.0;
        }
        let t_len = self.state.n_points;
        if t_len == 0 {
            return 0.0;
        }
        let scale_factor = if self.scale { 2.0 } else { 1.0 };
        let mut depth_sum = 0.0;
        for t in 0..t_len {
            let col = &self.state.sorted_columns[t];
            let at_or_below = col.partition_point(|&v| v <= curve[t]);
            let fn_x = at_or_below as f64 / n as f64;
            depth_sum += fn_x.min(1.0 - fn_x) * scale_factor;
        }
        depth_sum / t_len as f64
    }

    /// Compute FM depth for row `row` of `data` without allocating a temporary Vec.
    #[inline]
    fn fm_one_from_row(&self, data: &FdMatrix, row: usize) -> f64 {
        let n = self.state.nori;
        if n == 0 {
            return 0.0;
        }
        let t_len = self.state.n_points;
        if t_len == 0 {
            return 0.0;
        }
        let scale_factor = if self.scale { 2.0 } else { 1.0 };
        let mut depth_sum = 0.0;
        for t in 0..t_len {
            let col = &self.state.sorted_columns[t];
            let at_or_below = col.partition_point(|&v| v <= data[(row, t)]);
            let fn_x = at_or_below as f64 / n as f64;
            depth_sum += fn_x.min(1.0 - fn_x) * scale_factor;
        }
        depth_sum / t_len as f64
    }
}

impl StreamingDepth for StreamingFraimanMuniz {
    fn depth_one(&self, curve: &[f64]) -> f64 {
        self.fm_one_inner(curve)
    }

    fn depth_batch(&self, data_obj: &FdMatrix) -> Vec<f64> {
        let nobj = data_obj.nrows();
        if nobj == 0 || self.state.n_points == 0 || self.state.nori == 0 {
            return vec![0.0; nobj];
        }
        iter_maybe_parallel!(0..nobj)
            .map(|i| self.fm_one_from_row(data_obj, i))
            .collect()
    }

    fn n_points(&self) -> usize {
        self.state.n_points
    }

    fn n_reference(&self) -> usize {
        self.state.nori
    }
}
