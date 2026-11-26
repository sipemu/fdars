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
    /// aeon's Sakoe-Chiba band, given as a **fraction** of the shorter series
    /// (aeon's `window` argument of every elastic distance, e.g.
    /// `aeon.distances.wdtw_distance(x, y, window=0.2)`).
    ///
    /// The band follows aeon's `create_bounding_matrix(n, m, window=…)` exactly:
    /// half-width `thickness = ⌊window · min(n, m)⌋` cells around the
    /// *interpolated* diagonal joining `(0, 0)` and `(n − 1, m − 1)`. For equal
    /// lengths this equals [`DtwWindow::SakoeChiba`] with `radius = thickness`;
    /// for unequal lengths the two differ (tslearn widens the band by `|n − m|`,
    /// aeon slants it). `window = 1.0` admits every cell (aeon's special case);
    /// `window = 0.0` keeps only the interpolated diagonal staircase.
    SakoeChibaFraction {
        /// Band half-width as a fraction of `min(n, m)`, in `[0, 1]`.
        window: f64,
    },
    /// aeon's Itakura parallelogram, given as aeon's `itakura_max_slope`
    /// fraction (`create_bounding_matrix(n, m, itakura_max_slope=…)`, ported
    /// exactly). Requires `max_slope ∈ (0, 1]` with `⌊max_slope · min(n, m)⌋ ≥ 1`
    /// and `n ≥ 3` (aeon divides by `n − 2`).
    ItakuraFraction {
        /// Slope fraction in `(0, 1]` (aeon `itakura_max_slope`).
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
            DtwWindow::SakoeChibaFraction { window } => aeon_sakoe_chiba_bounds(n, m, window)?,
            DtwWindow::ItakuraFraction { max_slope } => aeon_itakura_bounds(n, m, max_slope)?,
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

/// Validate an aeon fraction parameter (`window` / `itakura_max_slope`).
fn check_fraction(v: f64, what: &str) -> Result<(), FdarError> {
    if v.is_finite() && (0.0..=1.0).contains(&v) {
        Ok(())
    } else {
        Err(FdarError::InvalidParameter {
            parameter: "window",
            message: format!("aeon {what} must be finite and in [0, 1], got {v}"),
        })
    }
}

/// `⌊(step / max_size) · len⌋`, in aeon's operation order.
#[inline]
fn interp_index(step: usize, max_size: usize, len: usize) -> isize {
    ((step as f64 / max_size as f64) * len as f64).floor() as isize
}

/// aeon `_sakoe_chiba_bounds`, branch `n <= m`: raw `(j_start, j_end)` arrays.
fn sakoe_rows_short(n: usize, m: usize, t: isize) -> (Vec<isize>, Vec<isize>) {
    let max_size = n.max(m) + 1;
    let ni = n as isize;
    let mut start = vec![0isize; n];
    let mut end = vec![m as isize; n];
    let mut next_row = 0isize;
    for step in 0..max_size {
        let (xi, yi) = (
            interp_index(step, max_size, n),
            interp_index(step, max_size, m),
        );
        while next_row < ni.min(xi + t + 1) {
            start[next_row as usize] = yi;
            next_row += 1;
        }
    }
    let mut prev_row = ni - 1;
    for step in (0..max_size).rev() {
        let (xi, yi) = (
            interp_index(step, max_size, n),
            interp_index(step, max_size, m),
        );
        while prev_row >= (xi - t).max(0) {
            end[prev_row as usize] = yi + 1;
            prev_row -= 1;
        }
    }
    (start, end)
}

/// aeon `_sakoe_chiba_bounds`, branch `n > m`: raw `(j_start, j_end)` arrays.
fn sakoe_rows_long(n: usize, m: usize, t: isize) -> (Vec<isize>, Vec<isize>) {
    let max_size = n.max(m) + 1;
    let mi = m as isize;
    let mut start = vec![0isize; n];
    let mut end = vec![mi; n];
    let mut next_row = 0isize;
    for step in 0..max_size {
        let (a, b) = (
            interp_index(step, max_size, m),
            interp_index(step, max_size, n),
        );
        while next_row <= b {
            start[next_row as usize] = (a - t).max(0);
            next_row += 1;
        }
    }
    let mut prev_row = n as isize - 1;
    for step in (0..max_size).rev() {
        let (a, b) = (
            interp_index(step, max_size, m),
            interp_index(step, max_size, n),
        );
        while prev_row >= b {
            end[prev_row as usize] = mi.min(a + t + 1);
            prev_row -= 1;
        }
    }
    (start, end)
}

/// Port of aeon's `_sakoe_chiba_bounds` (row bounds of `_sakoe_chiba_bounding`).
fn aeon_sakoe_chiba_bounds(
    n: usize,
    m: usize,
    window: f64,
) -> Result<Vec<(usize, usize)>, FdarError> {
    check_fraction(window, "window")?;
    if window == 1.0 {
        return Ok(vec![(0, m); n]);
    }
    let t = (window * n.min(m) as f64) as isize;
    let (start, end) = if n <= m {
        sakoe_rows_short(n, m, t)
    } else {
        sakoe_rows_long(n, m, t)
    };
    Ok(start
        .into_iter()
        .zip(end)
        .map(|(lo, hi)| (lo as usize, hi.max(lo) as usize))
        .collect())
}

/// Port of aeon's `_itakura_bounds` (row bounds of `_itakura_parallelogram`).
fn aeon_itakura_bounds(
    n: usize,
    m: usize,
    max_slope_percent: f64,
) -> Result<Vec<(usize, usize)>, FdarError> {
    check_fraction(max_slope_percent, "itakura_max_slope")?;
    let one_percent = n.min(m) as f64 / 100.0;
    let mut max_slope = ((max_slope_percent * one_percent) * 100.0).floor();
    if max_slope < 1.0 || n < 3 {
        return Err(FdarError::InvalidParameter {
            parameter: "window",
            message: format!(
                "aeon Itakura needs floor(itakura_max_slope * min(n, m)) >= 1 and n >= 3, \
                 got itakura_max_slope = {max_slope_percent} for sizes {n} x {m}"
            ),
        });
    }
    let mut min_slope = 1.0 / max_slope;
    max_slope *= (m as f64 - 1.0) / (n as f64 - 2.0);
    max_slope = max_slope.max(1.0);
    min_slope *= (m as f64 - 2.0) / (n as f64 - 1.0);
    min_slope = min_slope.min(1.0);
    let (nf, mf) = (n as f64, m as f64);
    let mut lower = vec![0.0; n];
    let mut upper = vec![0.0; n];
    for i in 0..n {
        let fi = i as f64;
        let cs = fi - nf + 1.0;
        lower[i] = round2(min_slope * fi)
            .max(round2(max_slope * cs + mf - 1.0))
            .ceil();
        upper[i] = round2(max_slope * fi + 1.0)
            .min(round2(min_slope * cs + mf))
            .floor();
    }
    if max_slope == 1.0 {
        if m > n {
            upper[..n - 1].copy_from_slice(&lower[1..]);
        } else {
            for i in 0..n {
                upper[i] = lower[i] + 1.0;
            }
        }
    }
    Ok(lower
        .iter()
        .zip(&upper)
        .map(|(&lo, &hi)| {
            let lo = lo.clamp(0.0, mf) as usize;
            let hi = hi.clamp(0.0, mf) as usize;
            (lo, hi.max(lo))
        })
        .collect())
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
