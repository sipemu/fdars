//! Pre-sorted reference state for O(log N) rank queries.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Pre-sorted reference values at each time point for O(log N) rank queries.
///
/// Constructed once from a column-major reference matrix and then shared
/// (immutably) by any number of streaming depth estimators.
#[derive(Debug, Clone, PartialEq)]
pub struct SortedReferenceState {
    /// `sorted_columns[t]` contains the reference values at time point `t`, sorted ascending.
    pub(crate) sorted_columns: Vec<Vec<f64>>,
    pub(crate) nori: usize,
    pub(crate) n_points: usize,
}

impl SortedReferenceState {
    /// Build from a column-major reference matrix.
    ///
    /// * `data_ori` -- reference matrix of shape `nori x n_points`
    ///
    /// Complexity: O(T x N log N)  (parallelised over time points).
    #[must_use = "expensive computation whose result should not be discarded"]
    pub fn from_reference(data_ori: &FdMatrix) -> Self {
        let nori = data_ori.nrows();
        let n_points = data_ori.ncols();
        if nori == 0 || n_points == 0 {
            return Self {
                sorted_columns: Vec::new(),
                nori,
                n_points,
            };
        }
        let sorted_columns: Vec<Vec<f64>> = iter_maybe_parallel!(0..n_points)
            .map(|t| {
                let mut col = data_ori.column(t).to_vec();
                col.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                col
            })
            .collect();
        Self {
            sorted_columns,
            nori,
            n_points,
        }
    }

    /// Returns `(below, above)` -- the count of reference values strictly below
    /// and strictly above `x` at time point `t`.
    ///
    /// Complexity: O(log N) via two binary searches.
    #[inline]
    pub fn rank_at(&self, t: usize, x: f64) -> (usize, usize) {
        let col = &self.sorted_columns[t];
        let below = col.partition_point(|&v| v < x);
        let at_or_below = col.partition_point(|&v| v <= x);
        let above = self.nori - at_or_below;
        (below, above)
    }

    /// Number of reference observations.
    #[inline]
    pub fn nori(&self) -> usize {
        self.nori
    }

    /// Number of evaluation points.
    #[inline]
    pub fn n_points(&self) -> usize {
        self.n_points
    }
}
