//! Rolling reference window with incremental sorted-column updates.

use std::collections::VecDeque;

use super::c2;
use super::sorted_ref::SortedReferenceState;

/// Sliding window of reference curves with incrementally maintained sorted columns.
///
/// When a new curve is pushed and the window is at capacity, the oldest curve
/// is evicted. For each time point the old value is removed (binary-search +
/// `Vec::remove`) and the new value is inserted (binary-search + `Vec::insert`).
///
/// Complexity per push: O(T x N) due to element shifting in the sorted vectors.
#[derive(Debug, Clone)]
pub struct RollingReference {
    curves: VecDeque<Vec<f64>>,
    capacity: usize,
    n_points: usize,
    sorted_columns: Vec<Vec<f64>>,
}

impl RollingReference {
    /// Create an empty rolling window.
    ///
    /// * `capacity` -- maximum number of curves in the window (must be >= 1).
    /// * `n_points` -- number of evaluation points per curve.
    pub fn new(capacity: usize, n_points: usize) -> Self {
        assert!(capacity >= 1, "capacity must be at least 1");
        Self {
            curves: VecDeque::with_capacity(capacity),
            capacity,
            n_points,
            sorted_columns: (0..n_points)
                .map(|_| Vec::with_capacity(capacity))
                .collect(),
        }
    }

    /// Push a new curve into the window.
    ///
    /// If the window is at capacity, the oldest curve is evicted and returned.
    /// For each time point, the sorted column is updated incrementally.
    pub fn push(&mut self, curve: &[f64]) -> Option<Vec<f64>> {
        assert_eq!(
            curve.len(),
            self.n_points,
            "curve length {} does not match n_points {}",
            curve.len(),
            self.n_points
        );

        let evicted = if self.curves.len() == self.capacity {
            let old = self
                .curves
                .pop_front()
                .expect("capacity invariant: deque is non-empty");
            // Remove old values from sorted columns
            for t in 0..self.n_points {
                let col = &mut self.sorted_columns[t];
                let old_val = old[t];
                let pos = col.partition_point(|&v| v < old_val);
                // Find exact match (handles duplicates by scanning nearby)
                let mut found = false;
                for idx in pos..col.len() {
                    if col[idx] == old_val {
                        col.remove(idx);
                        found = true;
                        break;
                    }
                    if col[idx] > old_val {
                        break;
                    }
                }
                if !found {
                    // Fallback: scan from pos backwards for floating-point edge cases
                    for idx in (0..pos).rev() {
                        if col[idx] == old_val {
                            col.remove(idx);
                            break;
                        }
                        if col[idx] < old_val {
                            break;
                        }
                    }
                }
            }
            Some(old)
        } else {
            None
        };

        // Insert new values into sorted columns
        let new_curve: Vec<f64> = curve.to_vec();
        for t in 0..self.n_points {
            let col = &mut self.sorted_columns[t];
            let val = new_curve[t];
            let pos = col.partition_point(|&v| v < val);
            col.insert(pos, val);
        }
        self.curves.push_back(new_curve);

        evicted
    }

    /// Take a snapshot of the current sorted reference state.
    ///
    /// This clones the sorted columns. For repeated queries, prefer
    /// [`mbd_one`](Self::mbd_one) which queries the window directly.
    pub fn snapshot(&self) -> SortedReferenceState {
        SortedReferenceState {
            sorted_columns: self.sorted_columns.clone(),
            nori: self.curves.len(),
            n_points: self.n_points,
        }
    }

    /// Compute rank-based MBD for a single curve directly against the current window.
    ///
    /// Avoids the overhead of cloning sorted columns into a snapshot.
    pub fn mbd_one(&self, curve: &[f64]) -> f64 {
        let n = self.curves.len();
        if n < 2 || self.n_points == 0 {
            return 0.0;
        }
        assert_eq!(
            curve.len(),
            self.n_points,
            "curve length {} does not match n_points {}",
            curve.len(),
            self.n_points
        );
        let cn2 = c2(n);
        let mut total = 0usize;
        for t in 0..self.n_points {
            let col = &self.sorted_columns[t];
            let below = col.partition_point(|&v| v < curve[t]);
            let at_or_below = col.partition_point(|&v| v <= curve[t]);
            let above = n - at_or_below;
            total += cn2 - c2(below) - c2(above);
        }
        total as f64 / (cn2 as f64 * self.n_points as f64)
    }

    /// Number of curves currently in the window.
    #[inline]
    pub fn len(&self) -> usize {
        self.curves.len()
    }

    /// Whether the window is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// Maximum capacity of the window.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
