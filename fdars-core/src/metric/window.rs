//! Global warping-window constraints for DTW-family dynamic programs.
//!
//! A [`DtwWindow`] restricts which cells `(i, j)` of the `n × m` alignment
//! lattice an alignment path may visit. The admissible cells mirror tslearn's
//! `sakoe_chiba_mask` and `itakura_mask` exactly (tslearn marks admissible
//! cells with `0` and forbidden cells with `inf`; [`DtwWindow::mask`] returns
//! `true` for admissible cells). In every windowed DP of this crate a masked
//! cell behaves as `+inf` accumulated cost: no path may pass through it.
//!
//! Note that [`DtwWindow::SakoeChiba`] is tslearn's band, **not** the `w`
//! argument of [`dtw_distance`](super::dtw::dtw_distance), whose legacy band is
//! asymmetric (`j − i ∈ [1 − w, w]` in 1-based indices).

use crate::error::FdarError;

/// Global constraint on the DTW alignment lattice.
///
/// # Examples
/// ```
/// use fdars_core::DtwWindow;
/// let mask = DtwWindow::SakoeChiba { radius: 2 }.mask(4, 4).unwrap();
/// assert_eq!(mask[0], vec![true, true, true, false]);
/// assert_eq!(mask[3], vec![false, true, true, true]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DtwWindow {
    /// No constraint: every cell of the lattice is admissible.
    #[default]
    None,
    /// Sakoe-Chiba band with the given `radius`, using tslearn's
    /// `sakoe_chiba_mask` semantics. For unequal lengths the band is widened by
    /// `|n − m|` along the longer axis so the end cell stays reachable.
    ///
    /// This is **not** the `w` argument of
    /// [`dtw_distance`](super::dtw::dtw_distance).
    SakoeChiba {
        /// Band radius (in samples). `radius ≥ max(n, m)` admits every cell.
        radius: usize,
    },
    /// Itakura parallelogram with the given maximum slope, using tslearn's
    /// `itakura_mask` semantics (including its 2-decimal rounding of the
    /// parallelogram edges). Requires a finite `max_slope ≥ 1`.
    Itakura {
        /// Maximum slope of the parallelogram (finite, `≥ 1`).
        max_slope: f64,
    },
}

impl DtwWindow {
    /// Per-row half-open admissible column ranges `[lo, hi)` (0-based).
    ///
    /// This is the single source of window admissibility for every windowed DP.
    ///
    /// # Errors
    /// - [`FdarError::InvalidDimension`] if `n == 0` or `m == 0`.
    /// - [`FdarError::InvalidParameter`] (parameter `"window"`) for an invalid
    ///   Itakura slope, or when some row or column admits no cell.
    pub(crate) fn row_bounds(&self, n: usize, m: usize) -> Result<Vec<(usize, usize)>, FdarError> {
        if n == 0 || m == 0 {
            return Err(FdarError::InvalidDimension {
                parameter: "window",
                expected: "non-empty series lengths (n > 0, m > 0)".to_string(),
                actual: format!("{n}x{m}"),
            });
        }
        let bounds = match *self {
            DtwWindow::None => vec![(0, m); n],
            DtwWindow::SakoeChiba { radius } => sakoe_chiba_bounds(n, m, radius),
            DtwWindow::Itakura { max_slope } => itakura_bounds(n, m, max_slope)?,
        };
        check_feasible(&bounds, n, m)?;
        Ok(bounds)
    }

    /// Boolean admissibility grid (`n` rows × `m` columns, row-major).
    ///
    /// `mask[i][j] == true` means cell `(i, j)` is admissible, matching the
    /// zero entries of tslearn's `sakoe_chiba_mask` / `itakura_mask`.
    ///
    /// # Errors
    /// Same as the window's admissibility check: [`FdarError::InvalidDimension`]
    /// for an empty size, [`FdarError::InvalidParameter`] for an invalid Itakura
    /// slope or an unfeasible constraint.
    ///
    /// # Examples
    /// ```
    /// use fdars_core::DtwWindow;
    /// let mask = DtwWindow::SakoeChiba { radius: 2 }.mask(4, 4).unwrap();
    /// let t = true;
    /// let f = false;
    /// assert_eq!(
    ///     mask,
    ///     vec![
    ///         vec![t, t, t, f],
    ///         vec![t, t, t, t],
    ///         vec![t, t, t, t],
    ///         vec![f, t, t, t],
    ///     ]
    /// );
    /// ```
    #[must_use = "expensive computation whose result should not be discarded"]
    pub fn mask(&self, n: usize, m: usize) -> Result<Vec<Vec<bool>>, FdarError> {
        let bounds = self.row_bounds(n, m)?;
        Ok(bounds
            .iter()
            .map(|&(lo, hi)| (0..m).map(|j| j >= lo && j < hi).collect())
            .collect())
    }
}

/// Closed form of tslearn's `_njit_sakoe_chiba_mask`, expressed per row.
///
/// Saturating arithmetic keeps `radius = usize::MAX` overflow-free.
fn sakoe_chiba_bounds(n: usize, m: usize, radius: usize) -> Vec<(usize, usize)> {
    (0..n)
        .map(|i| {
            if n <= m {
                let lo = i.saturating_sub(radius);
                let hi = i
                    .saturating_add(m - n)
                    .saturating_add(radius)
                    .saturating_add(1)
                    .min(m);
                (lo, hi)
            } else {
                let lo = i.saturating_sub((n - m).saturating_add(radius));
                let hi = i.saturating_add(radius).saturating_add(1).min(m);
                (lo, hi)
            }
        })
        .collect()
}

/// numba's 2-decimal rounding (`round(v, 2)`, ties to even).
#[inline]
fn round2(v: f64) -> f64 {
    (v * 100.0).round_ties_even() / 100.0
}

/// Column-wise `[lower_j, upper_j)` row range of tslearn's `_njit_itakura_mask`.
#[inline]
fn itakura_column(j: usize, n: usize, m: usize, ms: f64, mn: f64) -> (usize, usize) {
    let jf = j as f64;
    let n1 = (n - 1) as f64;
    let m1 = (m - 1) as f64;
    let lower = round2(mn * jf).max(round2(n1 - ms * m1 + ms * jf)).ceil();
    let upper = (round2(ms * jf).min(round2(n1 - mn * m1 + mn * jf)) + 1.0).floor();
    let nf = n as f64;
    (lower.clamp(0.0, nf) as usize, upper.clamp(0.0, nf) as usize)
}

/// Port of tslearn's `_njit_itakura_mask`, converted to per-row bounds.
fn itakura_bounds(n: usize, m: usize, max_slope: f64) -> Result<Vec<(usize, usize)>, FdarError> {
    if !(max_slope.is_finite() && max_slope >= 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "window",
            message: format!("Itakura max_slope must be finite and >= 1, got {max_slope}"),
        });
    }
    let ratio = n as f64 / m as f64;
    let ms = max_slope * ratio;
    let mn = (1.0 / max_slope) * ratio;
    let cols: Vec<(usize, usize)> = (0..m).map(|j| itakura_column(j, n, m, ms, mn)).collect();

    let mut bounds = vec![(0usize, 0usize); n];
    for (i, b) in bounds.iter_mut().enumerate() {
        let mut admitted = (0..m).filter(|&j| cols[j].0 <= i && i < cols[j].1);
        if let Some(first) = admitted.next() {
            let last = admitted.next_back().unwrap_or(first);
            debug_assert!(
                (first..=last).all(|j| cols[j].0 <= i && i < cols[j].1),
                "Itakura row admissible set must be contiguous"
            );
            *b = (first, last + 1);
        }
    }
    Ok(bounds)
}

/// Reject constraints where some row or column admits no cell.
fn check_feasible(bounds: &[(usize, usize)], n: usize, m: usize) -> Result<(), FdarError> {
    let unfeasible = || FdarError::InvalidParameter {
        parameter: "window",
        message: format!(
            "the global constraint is unfeasible (no admissible warping path) for sizes {n} x {m}"
        ),
    };
    let mut covered = vec![false; m];
    for &(lo, hi) in bounds {
        if lo >= hi {
            return Err(unfeasible());
        }
        covered[lo..hi].iter_mut().for_each(|c| *c = true);
    }
    if covered.iter().all(|&c| c) {
        Ok(())
    } else {
        Err(unfeasible())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(rows: &[&str]) -> Vec<Vec<bool>> {
        rows.iter()
            .map(|r| r.chars().map(|c| c == 'T').collect())
            .collect()
    }

    #[test]
    fn sakoe_chiba_square_golden() {
        let mask = DtwWindow::SakoeChiba { radius: 2 }.mask(4, 4).unwrap();
        assert_eq!(mask, grid(&["TTTF", "TTTT", "TTTT", "FTTT"]));
    }

    #[test]
    fn sakoe_chiba_rect_golden() {
        let mask = DtwWindow::SakoeChiba { radius: 1 }.mask(7, 3).unwrap();
        assert_eq!(
            mask,
            grid(&["TTF", "TTT", "TTT", "TTT", "TTT", "TTT", "FTT"])
        );
    }

    #[test]
    fn itakura_slope3_golden() {
        let mask = DtwWindow::Itakura { max_slope: 3.0 }.mask(6, 6).unwrap();
        assert_eq!(
            mask,
            grid(&["TFFFFF", "FTTTFF", "FTTTTF", "FTTTTF", "FFTTTF", "FFFFFT"])
        );
    }

    #[test]
    fn itakura_slope2_golden() {
        let mask = DtwWindow::Itakura { max_slope: 2.0 }.mask(6, 6).unwrap();
        assert_eq!(
            mask,
            grid(&["TFFFFF", "FTTFFF", "FTTTFF", "FFTTTF", "FFFTTF", "FFFFFT"])
        );
    }

    #[test]
    fn none_mask_all_true() {
        let mask = DtwWindow::None.mask(3, 5).unwrap();
        assert_eq!(mask, vec![vec![true; 5]; 3]);
    }

    #[test]
    fn sakoe_chiba_huge_radius_no_overflow() {
        let full = DtwWindow::None.row_bounds(5, 9).unwrap();
        for (n, m) in [(5, 9), (9, 5)] {
            let none = DtwWindow::None.row_bounds(n, m).unwrap();
            let big = DtwWindow::SakoeChiba { radius: usize::MAX }
                .row_bounds(n, m)
                .unwrap();
            assert_eq!(none, big);
        }
        assert_eq!(full.len(), 5);
    }

    #[test]
    fn window_errors() {
        assert!(matches!(
            DtwWindow::None.mask(0, 3),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            DtwWindow::None.mask(3, 0),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            DtwWindow::Itakura { max_slope: 0.5 }.mask(4, 4),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            DtwWindow::Itakura {
                max_slope: f64::NAN
            }
            .mask(4, 4),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            DtwWindow::Itakura { max_slope: 1.0 }.mask(3, 7),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
