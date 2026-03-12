//! Streaming Modified Band Depth (MBD) estimator.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::sorted_ref::SortedReferenceState;
use super::{c2, StreamingDepth};

/// Rank-based Modified Band Depth estimator.
///
/// Uses the combinatorial identity: at time t with `b` values strictly below
/// x(t) and `a` strictly above,
///
/// > pairs containing x(t) = C(N,2) - C(b,2) - C(a,2)
///
/// MBD(x) = (1 / (C(N,2) x T)) x sum_t [C(N,2) - C(b_t,2) - C(a_t,2)]
///
/// Per-query complexity: **O(T x log N)** instead of O(N^2 x T).
pub struct StreamingMbd {
    state: SortedReferenceState,
}

impl StreamingMbd {
    pub fn new(state: SortedReferenceState) -> Self {
        Self { state }
    }

    /// Compute MBD for a single row-layout curve using rank formula.
    #[inline]
    fn mbd_one_inner(&self, curve: &[f64]) -> f64 {
        let n = self.state.nori;
        if n < 2 {
            return 0.0;
        }
        let cn2 = c2(n);
        let t_len = self.state.n_points;
        let mut total = 0usize;
        for t in 0..t_len {
            let (below, above) = self.state.rank_at(t, curve[t]);
            total += cn2 - c2(below) - c2(above);
        }
        total as f64 / (cn2 as f64 * t_len as f64)
    }

    /// Compute MBD for row `row` of `data` without allocating a temporary Vec.
    #[inline]
    fn mbd_one_from_row(&self, data: &FdMatrix, row: usize) -> f64 {
        let n = self.state.nori;
        if n < 2 {
            return 0.0;
        }
        let cn2 = c2(n);
        let t_len = self.state.n_points;
        let mut total = 0usize;
        for t in 0..t_len {
            let (below, above) = self.state.rank_at(t, data[(row, t)]);
            total += cn2 - c2(below) - c2(above);
        }
        total as f64 / (cn2 as f64 * t_len as f64)
    }
}

impl StreamingDepth for StreamingMbd {
    fn depth_one(&self, curve: &[f64]) -> f64 {
        self.mbd_one_inner(curve)
    }

    fn depth_batch(&self, data_obj: &FdMatrix) -> Vec<f64> {
        let nobj = data_obj.nrows();
        if nobj == 0 || self.state.n_points == 0 || self.state.nori < 2 {
            return vec![0.0; nobj];
        }
        iter_maybe_parallel!(0..nobj)
            .map(|i| self.mbd_one_from_row(data_obj, i))
            .collect()
    }

    fn n_points(&self) -> usize {
        self.state.n_points
    }

    fn n_reference(&self) -> usize {
        self.state.nori
    }
}
