//! Streaming Band Depth (BD) estimator.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::sorted_ref::SortedReferenceState;
use super::{c2, StreamingDepth};

/// Full reference state that keeps per-curve values alongside sorted columns.
///
/// Required by Band Depth (BD), which checks all-or-nothing containment across
/// ALL time points and therefore cannot decompose into per-point rank queries.
#[derive(Debug, Clone)]
pub struct FullReferenceState {
    /// Sorted columns for rank queries (shared with MBD/FM estimators if desired).
    pub sorted: SortedReferenceState,
    /// `values_by_curve[j][t]` = reference curve j at time point t (row layout).
    values_by_curve: Vec<Vec<f64>>,
}

impl FullReferenceState {
    /// Build from a column-major reference matrix.
#[must_use = "expensive computation whose result should not be discarded"]
    pub fn from_reference(data_ori: &FdMatrix) -> Self {
        let nori = data_ori.nrows();
        let n_points = data_ori.ncols();
        let sorted = SortedReferenceState::from_reference(data_ori);
        let values_by_curve: Vec<Vec<f64>> = (0..nori)
            .map(|j| (0..n_points).map(|t| data_ori[(j, t)]).collect())
            .collect();
        Self {
            sorted,
            values_by_curve,
        }
    }
}

/// Streaming Band Depth estimator.
///
/// BD requires all-or-nothing containment across ALL time points -- it does not
/// decompose per-point like MBD. The streaming advantage here is **reference
/// decoupling** (no re-parsing the matrix) and **early-exit per pair** (break
/// on first time point where x is outside the band), not an asymptotic
/// improvement.
#[derive(Debug, Clone)]
pub struct StreamingBd {
    state: FullReferenceState,
}

impl StreamingBd {
    pub fn new(state: FullReferenceState) -> Self {
        Self { state }
    }

    #[inline]
    fn bd_one_inner(&self, curve: &[f64]) -> f64 {
        let n = self.state.sorted.nori;
        if n < 2 {
            return 0.0;
        }
        let n_pairs = c2(n);
        let n_points = self.state.sorted.n_points;

        let mut count_in_band = 0usize;
        for j in 0..n {
            for k in (j + 1)..n {
                let mut inside = true;
                for t in 0..n_points {
                    let x_t = curve[t];
                    let y_j_t = self.state.values_by_curve[j][t];
                    let y_k_t = self.state.values_by_curve[k][t];
                    let band_min = y_j_t.min(y_k_t);
                    let band_max = y_j_t.max(y_k_t);
                    if x_t < band_min || x_t > band_max {
                        inside = false;
                        break;
                    }
                }
                if inside {
                    count_in_band += 1;
                }
            }
        }
        count_in_band as f64 / n_pairs as f64
    }

    /// Compute BD for row `row` of `data` without allocating a temporary Vec.
    #[inline]
    fn bd_one_from_row(&self, data: &FdMatrix, row: usize) -> f64 {
        let n = self.state.sorted.nori;
        if n < 2 {
            return 0.0;
        }
        let n_pairs = c2(n);
        let n_points = self.state.sorted.n_points;

        let mut count_in_band = 0usize;
        for j in 0..n {
            for k in (j + 1)..n {
                let mut inside = true;
                for t in 0..n_points {
                    let x_t = data[(row, t)];
                    let y_j_t = self.state.values_by_curve[j][t];
                    let y_k_t = self.state.values_by_curve[k][t];
                    let band_min = y_j_t.min(y_k_t);
                    let band_max = y_j_t.max(y_k_t);
                    if x_t < band_min || x_t > band_max {
                        inside = false;
                        break;
                    }
                }
                if inside {
                    count_in_band += 1;
                }
            }
        }
        count_in_band as f64 / n_pairs as f64
    }
}

impl StreamingDepth for StreamingBd {
    fn depth_one(&self, curve: &[f64]) -> f64 {
        self.bd_one_inner(curve)
    }

    fn depth_batch(&self, data_obj: &FdMatrix) -> Vec<f64> {
        let nobj = data_obj.nrows();
        let n = self.state.sorted.nori;
        if nobj == 0 || self.state.sorted.n_points == 0 || n < 2 {
            return vec![0.0; nobj];
        }
        iter_maybe_parallel!(0..nobj)
            .map(|i| self.bd_one_from_row(data_obj, i))
            .collect()
    }

    fn n_points(&self) -> usize {
        self.state.sorted.n_points
    }

    fn n_reference(&self) -> usize {
        self.state.sorted.nori
    }
}
