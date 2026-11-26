//! The aeon elastic-distance family: DDTW, WDTW, WDDTW, ADTW, ERP, EDR, TWE,
//! MSM and LCSS.
//!
//! Every distance comes with aeon's four-function pattern
//! (`aeon.distances.<name>_distance`, `_pairwise_distance`, `_cost_matrix`,
//! `_alignment_path`) and reproduces aeon 1.6.0 numerically (golden-tested to
//! `1e-10`), including aeon's boundary conventions and normalisations:
//!
//! | distance | local cost | parameters (aeon defaults) | returned distance |
//! |---|---|---|---|
//! | [`ddtw_distance`] | squared, on slope derivatives | — | accumulated cost |
//! | [`wdtw_distance`] | squared × logistic weight of `\|i − j\|` | `g = 0.05` | accumulated cost |
//! | [`wddtw_distance`] | WDTW on slope derivatives | `g = 0.05` | accumulated cost |
//! | [`adtw_distance`] | squared, `warp_penalty` per non-diagonal step | `warp_penalty = 1.0` | accumulated cost |
//! | [`erp_distance`] | absolute; gaps cost `\|x − g\|` | `g = 0.0` | accumulated cost |
//! | [`edr_distance`] | 0/1 match within `epsilon` | `epsilon = None` → `max(sd x, sd y)/4` | edits / `max(n, m)` |
//! | [`twe_distance`] | absolute + stiffness `nu`, edit penalty `lmbda` | `nu = 0.001`, `lmbda = 1.0` | accumulated cost |
//! | [`msm_distance`] | absolute (or squared if dependent), move/split/merge cost `c` | `c = 1.0`, `independent = true` | accumulated cost |
//! | [`lcss_distance`] | match if `\|x − y\| ≤ epsilon` | `epsilon = 1.0` | `1 − LCSS / min(n, m)` |
//!
//! All functions act on **univariate** series (`&[f64]`, or rows of an
//! [`FdMatrix`]), so aeon's multivariate-only arguments collapse: ERP's
//! per-channel `g_arr` is the scalar `g`, and MSM's `independent` only switches
//! between aeon's absolute (independent) and squared (dependent) costs.
//!
//! # Windows
//! The `window` argument is a [`DtwWindow`]. aeon's `window=<fraction>` is
//! [`DtwWindow::SakoeChibaFraction`] and aeon's `itakura_max_slope=<fraction>`
//! is [`DtwWindow::ItakuraFraction`]; both are exact ports of aeon's
//! `create_bounding_matrix`. aeon's `window=None` is [`DtwWindow::None`]. The
//! tslearn-style [`DtwWindow::SakoeChiba`] / [`DtwWindow::Itakura`] are
//! accepted too (same in-band semantics as every other DP in this crate). For
//! DDTW / WDDTW the window applies to the derivative series (length `n − 2`),
//! as in aeon.
//!
//! # Deviations from aeon (documented)
//! - When no admissible alignment exists under the window (aeon returns
//!   `inf`), these functions return [`FdarError::ComputationFailed`].
//! - `lcss_alignment_path` follows aeon's traceback, but where aeon's loop
//!   would spin forever on an out-of-band cell it takes the non-match step.
//! - `ddtw_pairwise_distance` / `wddtw_pairwise_distance` under a window agree
//!   with the pointwise `ddtw_distance` / `wddtw_distance` (window sized for the
//!   length-`n − 2` derivatives). aeon's equal-length pairwise path sizes the
//!   window for the raw length `n` and so disagrees with its own pointwise
//!   distance; without a window both agree.
//! - `erp_alignment_path` honours an Itakura window (aeon silently ignores
//!   `itakura_max_slope` there).

mod adtw;
mod ddtw;
mod edr;
mod erp;
mod lcss;
mod msm;
mod twe;
mod wdtw;

#[cfg(test)]
mod goldens;
#[cfg(test)]
mod tests;

use std::borrow::Cow;

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::window::DtwWindow;
use super::{cross_distance_matrix, self_distance_matrix};

pub use adtw::{adtw_alignment_path, adtw_cost_matrix, adtw_distance, adtw_pairwise_distance};
pub use ddtw::{ddtw_alignment_path, ddtw_cost_matrix, ddtw_distance, ddtw_pairwise_distance};
pub use edr::{edr_alignment_path, edr_cost_matrix, edr_distance, edr_pairwise_distance};
pub use erp::{erp_alignment_path, erp_cost_matrix, erp_distance, erp_pairwise_distance};
pub use lcss::{lcss_alignment_path, lcss_cost_matrix, lcss_distance, lcss_pairwise_distance};
pub use msm::{msm_alignment_path, msm_cost_matrix, msm_distance, msm_pairwise_distance};
pub use twe::{twe_alignment_path, twe_cost_matrix, twe_distance, twe_pairwise_distance};
pub use wdtw::{
    wddtw_alignment_path, wddtw_cost_matrix, wddtw_distance, wddtw_pairwise_distance,
    wdtw_alignment_path, wdtw_cost_matrix, wdtw_distance, wdtw_pairwise_distance,
};

/// Optimal alignment returned by the `<name>_alignment_path` functions.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct ElasticDistancePath {
    /// Aligned index pairs `(i, j)` (0-based), as returned by aeon's
    /// `<name>_alignment_path`. For DDTW / WDDTW the indices refer to the
    /// derivative series (length `n − 2`); for LCSS only the matched pairs are
    /// listed.
    pub path: Vec<(usize, usize)>,
    /// The distance aeon reports alongside the path (for LCSS without the
    /// clamp at zero that `lcss_distance` applies).
    pub distance: f64,
}

// ---------------------------------------------------------------------------
// Shared padded-lattice DP core
// ---------------------------------------------------------------------------

/// Padded `(n + 1) × (m + 1)` DP table, row-major. Cell `(a, b)` of the
/// alignment (0-based) lives at padded position `(a + 1, b + 1)`.
pub(super) struct Lattice {
    cols: usize,
    v: Vec<f64>,
}

impl Lattice {
    /// Table for sizes `n × m` with every entry set to `fill`.
    fn new(n: usize, m: usize, fill: f64) -> Self {
        Lattice {
            cols: m + 1,
            v: vec![fill; (n + 1) * (m + 1)],
        }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> f64 {
        self.v[i * self.cols + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, val: f64) {
        self.v[i * self.cols + j] = val;
    }

    /// Fill every admissible cell in row-major order:
    /// `T[a+1][b+1] = cell(a, b, diag = T[a][b], up = T[a][b+1], left = T[a+1][b])`.
    fn fill(
        &mut self,
        bounds: &[(usize, usize)],
        mut cell: impl FnMut(usize, usize, f64, f64, f64) -> f64,
    ) {
        for (a, &(lo, hi)) in bounds.iter().enumerate() {
            for b in lo..hi {
                let val = cell(a, b, self.get(a, b), self.get(a, b + 1), self.get(a + 1, b));
                self.set(a + 1, b + 1, val);
            }
        }
    }

    /// Unpadded alignment cost `(a, b)` (aeon's returned cost matrix entry).
    #[inline]
    fn cost(&self, a: usize, b: usize) -> f64 {
        self.get(a + 1, b + 1)
    }
}

/// How the alignment path is traced back.
#[derive(Clone, Copy)]
pub(super) enum Traceback {
    /// aeon `compute_min_return_path` on the unpadded cost matrix.
    MinReturn,
    /// aeon `compute_lcss_return_path` with the given matching threshold.
    Lcss(f64),
}

/// One elastic distance: lattice builder + read-outs.
pub(super) struct Recipe<F> {
    /// Name used in error messages.
    pub(super) name: &'static str,
    /// Apply aeon's slope derivative to both series first (DDTW / WDDTW).
    pub(super) derivative: bool,
    /// Return the padded `(n + 1) × (m + 1)` table as cost matrix (LCSS).
    pub(super) padded_output: bool,
    /// Build the filled lattice for prepared series and row bounds.
    pub(super) lattice: F,
    /// Distance read-out from the filled lattice.
    pub(super) score: fn(&Lattice, usize, usize) -> f64,
    /// Distance read-out reported with the alignment path.
    pub(super) path_score: fn(&Lattice, usize, usize) -> f64,
    /// Traceback rule.
    pub(super) traceback: Traceback,
}

/// The default read-out: last unpadded cell.
pub(super) fn last_cell(l: &Lattice, n: usize, m: usize) -> f64 {
    l.cost(n - 1, m - 1)
}

/// Validate a scalar parameter as finite (and optionally `≥ 0`).
pub(super) fn check_param(
    parameter: &'static str,
    v: f64,
    nonnegative: bool,
) -> Result<(), FdarError> {
    if v.is_finite() && (!nonnegative || v >= 0.0) {
        Ok(())
    } else {
        let need = if nonnegative {
            "finite and >= 0"
        } else {
            "finite"
        };
        Err(FdarError::InvalidParameter {
            parameter,
            message: format!("must be {need}, got {v}"),
        })
    }
}

/// aeon's univariate Euclidean distance `sqrt((a − b)²)`.
#[inline]
pub(super) fn euc(a: f64, b: f64) -> f64 {
    let d = a - b;
    (d * d).sqrt()
}

/// Whether 0-based cell `(i, j)` is inside the row bounds.
#[inline]
pub(super) fn in_band(bounds: &[(usize, usize)], i: usize, j: usize) -> bool {
    let (lo, hi) = bounds[i];
    lo <= j && j < hi
}

/// aeon's `slope_derivative` (Keogh & Pazzani 2001 average of slopes):
/// `q'_i = ((q_i − q_{i−1}) + (q_{i+1} − q_{i−1}) / 2) / 2` for interior points
/// only, so the result has length `n − 2` (no boundary padding).
pub(super) fn slope_derivative(x: &[f64]) -> Vec<f64> {
    x.windows(3)
        .map(|w| ((w[1] - w[0]) + (w[2] - w[0]) / 2.0) / 2.0)
        .collect()
}

impl<F> Recipe<F>
where
    F: Fn(&[f64], &[f64], &[(usize, usize)]) -> Lattice + Sync,
{
    fn min_len(&self) -> usize {
        if self.derivative {
            3
        } else {
            1
        }
    }

    fn check_len(&self, parameter: &'static str, len: usize) -> Result<(), FdarError> {
        if len < self.min_len() {
            return Err(FdarError::InvalidDimension {
                parameter,
                expected: format!("series of length >= {}", self.min_len()),
                actual: format!("length {len}"),
            });
        }
        Ok(())
    }

    fn prepare<'x>(
        &self,
        parameter: &'static str,
        x: &'x [f64],
    ) -> Result<Cow<'x, [f64]>, FdarError> {
        self.check_len(parameter, x.len())?;
        Ok(if self.derivative {
            Cow::Owned(slope_derivative(x))
        } else {
            Cow::Borrowed(x)
        })
    }

    fn no_path(&self) -> FdarError {
        FdarError::ComputationFailed {
            operation: self.name,
            detail: "no admissible alignment under the window (aeon returns inf)".to_string(),
        }
    }

    fn finite(&self, d: f64) -> Result<f64, FdarError> {
        if d.is_finite() {
            Ok(d)
        } else {
            Err(self.no_path())
        }
    }

    /// Prepared series, their lengths, row bounds and the filled lattice.
    fn solve(
        &self,
        x: &[f64],
        y: &[f64],
        window: DtwWindow,
    ) -> Result<(Vec<f64>, Vec<f64>, Lattice), FdarError> {
        let xs = self.prepare("x", x)?.into_owned();
        let ys = self.prepare("y", y)?.into_owned();
        let bounds = window.row_bounds(xs.len(), ys.len())?;
        let lat = (self.lattice)(&xs, &ys, &bounds);
        Ok((xs, ys, lat))
    }

    pub(super) fn distance(
        &self,
        x: &[f64],
        y: &[f64],
        window: DtwWindow,
    ) -> Result<f64, FdarError> {
        let (xs, ys, lat) = self.solve(x, y, window)?;
        self.finite((self.score)(&lat, xs.len(), ys.len()))
    }

    pub(super) fn cost_matrix(
        &self,
        x: &[f64],
        y: &[f64],
        window: DtwWindow,
    ) -> Result<FdMatrix, FdarError> {
        let (xs, ys, lat) = self.solve(x, y, window)?;
        let (n, m) = (xs.len(), ys.len());
        Ok(if self.padded_output {
            let mut out = FdMatrix::zeros(n + 1, m + 1);
            for i in 0..=n {
                for j in 0..=m {
                    out[(i, j)] = lat.get(i, j);
                }
            }
            out
        } else {
            let mut out = FdMatrix::zeros(n, m);
            for i in 0..n {
                for j in 0..m {
                    out[(i, j)] = lat.cost(i, j);
                }
            }
            out
        })
    }

    pub(super) fn alignment_path(
        &self,
        x: &[f64],
        y: &[f64],
        window: DtwWindow,
    ) -> Result<ElasticDistancePath, FdarError> {
        let (xs, ys, lat) = self.solve(x, y, window)?;
        let (n, m) = (xs.len(), ys.len());
        let distance = self.finite((self.path_score)(&lat, n, m))?;
        let path = match self.traceback {
            Traceback::MinReturn => min_return_path(&lat, n, m),
            Traceback::Lcss(epsilon) => {
                let bounds = window.row_bounds(n, m)?;
                lcss::lcss_return_path(&xs, &ys, epsilon, &bounds, &lat)
            }
        };
        Ok(ElasticDistancePath { path, distance })
    }

    pub(super) fn pairwise(
        &self,
        data: &FdMatrix,
        other: Option<&FdMatrix>,
        window: DtwWindow,
    ) -> Result<FdMatrix, FdarError> {
        self.check_len("data", data.ncols())?;
        if let Some(o) = other {
            self.check_len("other", o.ncols())?;
        }
        let rows1 = self.prepared_rows(data);
        let n_other = other.map_or(rows1.len(), FdMatrix::nrows);
        if rows1.is_empty() || n_other == 0 {
            return Ok(FdMatrix::zeros(rows1.len(), n_other));
        }
        let result = match other {
            None => {
                let n = rows1[0].len();
                let bounds = window.row_bounds(n, n)?;
                self_distance_matrix(rows1.len(), |i, j| self.pair(&rows1[i], &rows1[j], &bounds))
            }
            Some(other) => {
                let rows2 = self.prepared_rows(other);
                let bounds = window.row_bounds(rows1[0].len(), rows2[0].len())?;
                cross_distance_matrix(rows1.len(), rows2.len(), |i, j| {
                    self.pair(&rows1[i], &rows2[j], &bounds)
                })
            }
        };
        if result.as_slice().iter().all(|d| d.is_finite()) {
            Ok(result)
        } else {
            Err(self.no_path())
        }
    }

    fn prepared_rows(&self, data: &FdMatrix) -> Vec<Vec<f64>> {
        (0..data.nrows())
            .map(|i| {
                let row = data.row(i);
                if self.derivative {
                    slope_derivative(&row)
                } else {
                    row
                }
            })
            .collect()
    }

    fn pair(&self, x: &[f64], y: &[f64], bounds: &[(usize, usize)]) -> f64 {
        (self.score)(&(self.lattice)(x, y, bounds), x.len(), y.len())
    }
}

/// Port of aeon's `compute_min_return_path` on the unpadded cost matrix
/// (`np.argmin` first-minimum tie rule over diagonal, up, left).
fn min_return_path(lat: &Lattice, n: usize, m: usize) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (n - 1, m - 1);
    let mut path = Vec::with_capacity(n + m);
    while i > 0 || j > 0 {
        path.push((i, j));
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            let diag = lat.cost(i - 1, j - 1);
            let up = lat.cost(i - 1, j);
            let left = lat.cost(i, j - 1);
            if diag <= up && diag <= left {
                i -= 1;
                j -= 1;
            } else if up <= left {
                i -= 1;
            } else {
                j -= 1;
            }
        }
    }
    path.push((0, 0));
    path.reverse();
    path
}

/// Generate aeon's four public functions for one distance.
///
/// `$recipe` is an expression of the parameter identifiers returning
/// `Result<Recipe<_>, FdarError>`; `$ex` are example argument literals.
macro_rules! elastic_family_fns {
    (
        label: $label:literal,
        distance: $dist:ident,
        pairwise: $pair:ident,
        cost: $cost:ident,
        path: $path:ident,
        params: ($($p:ident : $t:ty),*),
        params_doc: $pdoc:literal,
        example: $ex:literal,
        recipe: $recipe:expr $(,)?
    ) => {
        #[doc = concat!($label, " distance between two univariate series (aeon `", stringify!($dist), "`).")]
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// - [`FdarError::InvalidDimension`] if a series is too short.
        /// - [`FdarError::InvalidParameter`] for an invalid parameter or window.
        /// - [`FdarError::ComputationFailed`] if no admissible alignment exists.
        ///
        /// # Examples
        /// ```
        #[doc = concat!("use fdars_core::metric::{", stringify!($dist), ", DtwWindow};")]
        /// let x = [0.0, 1.0, 2.0, 1.5, 0.5, 0.0];
        /// let y = [0.0, 0.5, 1.5, 2.0, 1.0, 0.0];
        #[doc = concat!("let d = ", stringify!($dist), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        #[doc = concat!("let s = ", stringify!($dist), "(&y, &x, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(d >= 0.0 && (d - s).abs() < 1e-12);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $dist(x: &[f64], y: &[f64], window: DtwWindow $(, $p: $t)*) -> Result<f64, FdarError> {
            ($recipe)?.distance(x, y, window)
        }

        #[doc = concat!($label, " distance matrix of a collection (aeon `", stringify!($pair), "`).")]
        ///
        /// Rows of `data` (and `other`) are the series. With `other = None` the
        /// symmetric `n × n` self-distance matrix (zero diagonal) is returned,
        /// otherwise the `n1 × n2` cross-distance matrix. Pairs are computed in
        /// parallel with the `parallel` feature.
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the pointwise distance; additionally the matrices must have
        /// enough columns.
        ///
        /// # Examples
        /// ```
        /// use fdars_core::matrix::FdMatrix;
        #[doc = concat!("use fdars_core::metric::{", stringify!($pair), ", DtwWindow};")]
        /// let data = FdMatrix::from_column_major(
        ///     (0..24).map(|k| ((k * 7 % 11) as f64 * 0.3).sin()).collect(), 4, 6,
        /// ).unwrap();
        #[doc = concat!("let d = ", stringify!($pair), "(&data, None, DtwWindow::None", $ex, ").unwrap();")]
        /// assert_eq!(d.shape(), (4, 4));
        /// assert_eq!(d[(0, 0)], 0.0);
        /// assert!((d[(1, 2)] - d[(2, 1)]).abs() < 1e-12);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $pair(
            data: &FdMatrix,
            other: Option<&FdMatrix>,
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<FdMatrix, FdarError> {
            ($recipe)?.pairwise(data, other, window)
        }

        #[doc = concat!($label, " accumulated cost matrix (aeon `", stringify!($cost), "`).")]
        ///
        /// Out-of-window cells hold aeon's fill value (`inf`, or `0` for LCSS).
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the pointwise distance (except that an unreachable end cell is
        /// returned as is rather than reported).
        ///
        /// # Examples
        /// ```
        #[doc = concat!("use fdars_core::metric::{", stringify!($cost), ", DtwWindow};")]
        /// let x = [0.0, 1.0, 2.0, 1.5, 0.5, 0.0];
        /// let y = [0.0, 0.5, 1.5, 2.0, 1.0];
        #[doc = concat!("let cm = ", stringify!($cost), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(cm.nrows() >= 4 && cm.ncols() >= 3);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $cost(x: &[f64], y: &[f64], window: DtwWindow $(, $p: $t)*) -> Result<FdMatrix, FdarError> {
            ($recipe)?.cost_matrix(x, y, window)
        }

        #[doc = concat!($label, " optimal alignment path and distance (aeon `", stringify!($path), "`).")]
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the pointwise distance.
        ///
        /// # Examples
        /// ```
        #[doc = concat!("use fdars_core::metric::{", stringify!($path), ", DtwWindow};")]
        /// let x = [0.0, 1.0, 2.0, 1.5, 0.5, 0.0];
        /// let y = [0.0, 0.5, 1.5, 2.0, 1.0, 0.0];
        #[doc = concat!("let res = ", stringify!($path), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(res.distance >= 0.0);
        /// assert!(res.path.windows(2).all(|w| w[0].0 <= w[1].0 && w[0].1 <= w[1].1));
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $path(
            x: &[f64],
            y: &[f64],
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<ElasticDistancePath, FdarError> {
            ($recipe)?.alignment_path(x, y, window)
        }
    };
}
pub(super) use elastic_family_fns;
