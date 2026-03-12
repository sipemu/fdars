//! Irregular functional data operations.
//!
//! This module provides data structures and algorithms for functional data
//! where observations have different evaluation points (irregular/sparse sampling).
//!
//! ## Storage Format
//!
//! Uses a CSR-like (Compressed Sparse Row) format for efficient storage:
//! - `offsets[i]..offsets[i+1]` gives the slice indices for observation i
//! - `argvals` and `values` store all data contiguously
//!
//! This format is memory-efficient and enables parallel processing of observations.
//!
//! ## Example
//!
//! For 3 curves with varying numbers of observation points:
//! - Curve 0: 5 points
//! - Curve 1: 3 points
//! - Curve 2: 7 points
//!
//! The offsets would be: [0, 5, 8, 15]

pub mod kernels;
pub mod smoothing;

#[cfg(test)]
mod tests;

// Re-export all public items
pub use kernels::{mean_irreg, KernelType};
pub use smoothing::{
    cov_irreg, integrate_irreg, metric_lp_irreg, norm_lp_irreg, to_regular_grid,
};

/// Compressed storage for irregular functional data.
///
/// Uses CSR-style layout where each observation can have a different
/// number of evaluation points.
#[derive(Clone, Debug)]
pub struct IrregFdata {
    /// Start indices for each observation (length n+1)
    /// `offsets[i]..offsets[i+1]` gives the range for observation i
    pub offsets: Vec<usize>,
    /// All observation points concatenated
    pub argvals: Vec<f64>,
    /// All values concatenated
    pub values: Vec<f64>,
    /// Domain range `[min, max]`
    pub rangeval: [f64; 2],
}

impl IrregFdata {
    /// Create from lists of argvals and values (one per observation).
    ///
    /// # Arguments
    /// * `argvals_list` - List of observation point vectors
    /// * `values_list` - List of value vectors (same lengths as argvals_list)
    ///
    /// # Panics
    /// Panics if the lists have different lengths or if any pair has mismatched lengths.
    pub fn from_lists(argvals_list: &[Vec<f64>], values_list: &[Vec<f64>]) -> Self {
        let n = argvals_list.len();
        assert_eq!(
            n,
            values_list.len(),
            "argvals_list and values_list must have same length"
        );

        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);

        let total_points: usize = argvals_list.iter().map(std::vec::Vec::len).sum();
        let mut argvals = Vec::with_capacity(total_points);
        let mut values = Vec::with_capacity(total_points);

        let mut range_min = f64::INFINITY;
        let mut range_max = f64::NEG_INFINITY;

        for i in 0..n {
            assert_eq!(
                argvals_list[i].len(),
                values_list[i].len(),
                "Observation {i} has mismatched argvals/values lengths"
            );

            argvals.extend_from_slice(&argvals_list[i]);
            values.extend_from_slice(&values_list[i]);
            offsets.push(argvals.len());

            if let (Some(&min), Some(&max)) = (argvals_list[i].first(), argvals_list[i].last()) {
                range_min = range_min.min(min);
                range_max = range_max.max(max);
            }
        }

        IrregFdata {
            offsets,
            argvals,
            values,
            rangeval: [range_min, range_max],
        }
    }

    /// Create from flattened representation (for R interop).
    ///
    /// Returns `None` if offsets are empty, argvals/values lengths differ,
    /// the last offset doesn't match argvals length, or offsets are non-monotonic.
    ///
    /// # Arguments
    /// * `offsets` - Start indices (length n+1)
    /// * `argvals` - All observation points concatenated
    /// * `values` - All values concatenated
    /// * `rangeval` - Domain range `[min, max]`
    pub fn from_flat(
        offsets: Vec<usize>,
        argvals: Vec<f64>,
        values: Vec<f64>,
        rangeval: [f64; 2],
    ) -> Result<Self, crate::FdarError> {
        let last = offsets.last().copied().unwrap_or(0);
        if offsets.is_empty()
            || argvals.len() != values.len()
            || last != argvals.len()
            || offsets.windows(2).any(|w| w[0] > w[1])
        {
            return Err(crate::FdarError::InvalidDimension {
                parameter: "offsets/argvals/values",
                expected: "non-empty offsets, argvals.len() == values.len(), monotone offsets"
                    .to_string(),
                actual: format!(
                    "offsets.len()={}, argvals.len()={}, values.len()={}",
                    offsets.len(),
                    argvals.len(),
                    values.len()
                ),
            });
        }
        Ok(IrregFdata {
            offsets,
            argvals,
            values,
            rangeval,
        })
    }

    /// Number of observations stored in this irregular functional data object.
    #[inline]
    pub fn n_obs(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Number of points for observation i.
    #[inline]
    pub fn n_points(&self, i: usize) -> usize {
        self.offsets[i + 1] - self.offsets[i]
    }

    /// Get observation i as a pair of slices (argvals, values).
    #[inline]
    pub fn get_obs(&self, i: usize) -> (&[f64], &[f64]) {
        let start = self.offsets[i];
        let end = self.offsets[i + 1];
        (&self.argvals[start..end], &self.values[start..end])
    }

    /// Total number of observation points across all curves.
    #[inline]
    pub fn total_points(&self) -> usize {
        self.argvals.len()
    }

    /// Get observation counts for all curves.
    pub fn obs_counts(&self) -> Vec<usize> {
        (0..self.n_obs()).map(|i| self.n_points(i)).collect()
    }

    /// Get minimum number of observations per curve.
    pub fn min_obs(&self) -> usize {
        (0..self.n_obs())
            .map(|i| self.n_points(i))
            .min()
            .unwrap_or(0)
    }

    /// Get maximum number of observations per curve.
    pub fn max_obs(&self) -> usize {
        (0..self.n_obs())
            .map(|i| self.n_points(i))
            .max()
            .unwrap_or(0)
    }
}

/// Linear interpolation at point t.
pub(super) fn linear_interp(argvals: &[f64], values: &[f64], t: f64) -> f64 {
    if t <= argvals[0] {
        return values[0];
    }
    if t >= argvals[argvals.len() - 1] {
        return values[values.len() - 1];
    }

    // Find the interval
    let idx = argvals
        .iter()
        .position(|&x| x > t)
        .expect("element must exist in collection");
    let t0 = argvals[idx - 1];
    let t1 = argvals[idx];
    let x0 = values[idx - 1];
    let x1 = values[idx];

    // Linear interpolation
    x0 + (x1 - x0) * (t - t0) / (t1 - t0)
}
