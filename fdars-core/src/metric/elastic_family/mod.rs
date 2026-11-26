//! The aeon elastic-distance family: DDTW, WDTW, WDDTW, ADTW, ERP, EDR, TWE,
//! MSM, LCSS and shape DTW.
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
//! | [`shape_dtw_distance`] | DTW on subsequence shape descriptors | `descriptor = Identity`, `reach = 15` | raw squared cost along the descriptor alignment |
//!
//! The plain functions act on **univariate** series (`&[f64]`, or rows of an
//! [`FdMatrix`]); there MSM's `independent` only switches between aeon's
//! absolute (independent) and squared (dependent) costs.
//!
//! # Multivariate series
//! Every distance also has a `_multivariate` variant of each of the four
//! functions (e.g. [`msm_distance_multivariate`]) taking one `d × m`
//! [`FdMatrix`] per series — channels as rows, time points as columns, aeon's
//! `(n_channels, n_timepoints)` layout — and collections as `&[FdMatrix]`
//! (series may differ in length, not in channel count). aeon's multichannel
//! semantics are reproduced: pointwise costs are **dependent**, i.e. the
//! squared Euclidean distance `Σ_c (x_{c,i} − y_{c,j})²` (or its root where
//! aeon uses the Euclidean distance: ERP, EDR, TWE, LCSS) across channels;
//! DDTW / WDDTW differentiate every channel; ERP's scalar `g` is broadcast to
//! all channels (aeon's `g` without `g_arr`); EDR's default `epsilon` uses the
//! standard deviation over all entries of each series (aeon's `np.std(x)`);
//! MSM with `independent = true` sums the per-channel univariate MSM cost
//! matrices (aeon `_msm_independent_cost_matrix`), with `independent = false`
//! it uses aeon's vector split/merge cost. With `d = 1` the multivariate
//! functions perform exactly the floating-point operations of the univariate
//! ones, so the results are bit-identical.
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
mod shape_dtw;
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

pub use adtw::{
    adtw_alignment_path, adtw_alignment_path_multivariate, adtw_cost_matrix,
    adtw_cost_matrix_multivariate, adtw_distance, adtw_distance_multivariate,
    adtw_pairwise_distance, adtw_pairwise_distance_multivariate,
};
pub use ddtw::{
    ddtw_alignment_path, ddtw_alignment_path_multivariate, ddtw_cost_matrix,
    ddtw_cost_matrix_multivariate, ddtw_distance, ddtw_distance_multivariate,
    ddtw_pairwise_distance, ddtw_pairwise_distance_multivariate,
};
pub use edr::{
    edr_alignment_path, edr_alignment_path_multivariate, edr_cost_matrix,
    edr_cost_matrix_multivariate, edr_distance, edr_distance_multivariate, edr_pairwise_distance,
    edr_pairwise_distance_multivariate,
};
pub use erp::{
    erp_alignment_path, erp_alignment_path_multivariate, erp_cost_matrix,
    erp_cost_matrix_multivariate, erp_distance, erp_distance_multivariate, erp_pairwise_distance,
    erp_pairwise_distance_multivariate,
};
pub use lcss::{
    lcss_alignment_path, lcss_alignment_path_multivariate, lcss_cost_matrix,
    lcss_cost_matrix_multivariate, lcss_distance, lcss_distance_multivariate,
    lcss_pairwise_distance, lcss_pairwise_distance_multivariate,
};
pub use msm::{
    msm_alignment_path, msm_alignment_path_multivariate, msm_cost_matrix,
    msm_cost_matrix_multivariate, msm_distance, msm_distance_multivariate, msm_pairwise_distance,
    msm_pairwise_distance_multivariate,
};
pub use shape_dtw::{
    shape_dtw_alignment_path, shape_dtw_alignment_path_multivariate, shape_dtw_cost_matrix,
    shape_dtw_cost_matrix_multivariate, shape_dtw_distance, shape_dtw_distance_multivariate,
    shape_dtw_pairwise_distance, shape_dtw_pairwise_distance_multivariate, ShapeDtwDescriptor,
};
pub use twe::{
    twe_alignment_path, twe_alignment_path_multivariate, twe_cost_matrix,
    twe_cost_matrix_multivariate, twe_distance, twe_distance_multivariate, twe_pairwise_distance,
    twe_pairwise_distance_multivariate,
};
pub use wdtw::{
    wddtw_alignment_path, wddtw_alignment_path_multivariate, wddtw_cost_matrix,
    wddtw_cost_matrix_multivariate, wddtw_distance, wddtw_distance_multivariate,
    wddtw_pairwise_distance, wddtw_pairwise_distance_multivariate, wdtw_alignment_path,
    wdtw_alignment_path_multivariate, wdtw_cost_matrix, wdtw_cost_matrix_multivariate,
    wdtw_distance, wdtw_distance_multivariate, wdtw_pairwise_distance,
    wdtw_pairwise_distance_multivariate,
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

    /// Element-wise `self += other` (same shape).
    fn add_assign(&mut self, other: &Lattice) {
        for (u, &w) in self.v.iter_mut().zip(&other.v) {
            *u += w;
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

/// A series as a channel-interleaved view: time point `i` occupies
/// `v[i·d .. (i + 1)·d]` (the flat buffer of a column-major `d × m`
/// [`FdMatrix`]; `d = 1` is a plain univariate slice).
#[derive(Clone, Copy)]
pub(super) struct Ser<'a> {
    pub(super) v: &'a [f64],
    pub(super) d: usize,
}

impl<'a> Ser<'a> {
    /// Univariate series.
    #[inline]
    pub(super) fn uni(v: &'a [f64]) -> Self {
        Ser { v, d: 1 }
    }

    /// A `d × m` matrix (channels × time points) as a series.
    #[inline]
    pub(super) fn of(x: &'a FdMatrix) -> Self {
        Ser {
            v: x.as_slice(),
            d: x.nrows().max(1),
        }
    }

    /// Number of time points.
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.v.len() / self.d
    }

    /// All channels at time point `i`.
    #[inline]
    pub(super) fn at(&self, i: usize) -> &'a [f64] {
        &self.v[i * self.d..(i + 1) * self.d]
    }

    /// Channel `c` as a univariate series.
    pub(super) fn channel(&self, c: usize) -> Vec<f64> {
        self.v.iter().skip(c).step_by(self.d).copied().collect()
    }
}

/// aeon's `_univariate_squared_distance`: `Σ_c (a_c − b_c)²`, accumulated from
/// `0.0` in channel order (for one channel exactly `(a − b)²`).
#[inline]
pub(super) fn sqe(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (u, w) in a.iter().zip(b) {
        let t = u - w;
        s += t * t;
    }
    s
}

/// aeon's `_univariate_euclidean_distance` `sqrt(Σ_c (a_c − b_c)²)`.
#[inline]
pub(super) fn euc(a: &[f64], b: &[f64]) -> f64 {
    sqe(a, b).sqrt()
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

/// [`slope_derivative`] of every channel of a channel-interleaved series
/// (aeon `slope_derivative_2d`); `d = 1` is [`slope_derivative`].
pub(super) fn slope_derivative_ser(x: Ser) -> Vec<f64> {
    let d = x.d;
    if d == 1 {
        return slope_derivative(x.v);
    }
    let n = x.len();
    let mut out = Vec::with_capacity(n.saturating_sub(2) * d);
    for t in 0..n.saturating_sub(2) {
        for c in 0..d {
            let (a, b, e) = (x.v[t * d + c], x.v[(t + 1) * d + c], x.v[(t + 2) * d + c]);
            out.push(((b - a) + (e - a) / 2.0) / 2.0);
        }
    }
    out
}

/// Channel count of a multivariate series, validated against an expected one.
pub(super) fn channels_of(
    parameter: &'static str,
    x: &FdMatrix,
    expected: Option<usize>,
) -> Result<usize, FdarError> {
    let d = x.nrows();
    if d == 0 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "a d x m series with d >= 1 channels".to_string(),
            actual: format!("{}x{}", x.nrows(), x.ncols()),
        });
    }
    match expected {
        Some(e) if e != d => Err(FdarError::InvalidDimension {
            parameter,
            expected: format!("{e} channels (rows)"),
            actual: format!("{d} channels"),
        }),
        _ => Ok(d),
    }
}

/// Validate two `d × m` series (same channel count) and view them as [`Ser`].
pub(super) fn mv_pair<'a>(
    x: &'a FdMatrix,
    y: &'a FdMatrix,
) -> Result<(Ser<'a>, Ser<'a>), FdarError> {
    let d = channels_of("x", x, None)?;
    channels_of("y", y, Some(d))?;
    Ok((Ser::of(x), Ser::of(y)))
}

/// Common channel count of the series of one or two collections.
pub(super) fn collection_channels(
    data: &[FdMatrix],
    other: Option<&[FdMatrix]>,
) -> Result<usize, FdarError> {
    let mut d = None;
    for x in data {
        d = Some(channels_of("data", x, d)?);
    }
    for x in other.unwrap_or(&[]) {
        d = Some(channels_of("other", x, d)?);
    }
    Ok(d.unwrap_or(1))
}

impl<F> Recipe<F>
where
    F: Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync,
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
        x: Ser<'x>,
    ) -> Result<Cow<'x, [f64]>, FdarError> {
        self.check_len(parameter, x.len())?;
        Ok(if self.derivative {
            Cow::Owned(slope_derivative_ser(x))
        } else {
            Cow::Borrowed(x.v)
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

    /// Prepared series (flat, channel-interleaved), their lengths, row bounds
    /// and the filled lattice.
    fn solve(
        &self,
        x: Ser,
        y: Ser,
        window: DtwWindow,
    ) -> Result<(Vec<f64>, Vec<f64>, Lattice), FdarError> {
        let d = x.d;
        let xs = self.prepare("x", x)?.into_owned();
        let ys = self.prepare("y", y)?.into_owned();
        let (n, m) = (xs.len() / d, ys.len() / d);
        let bounds = window.row_bounds(n, m)?;
        let lat = (self.lattice)(Ser { v: &xs, d }, Ser { v: &ys, d }, &bounds);
        Ok((xs, ys, lat))
    }

    pub(super) fn distance(&self, x: Ser, y: Ser, window: DtwWindow) -> Result<f64, FdarError> {
        let d = x.d;
        let (xs, ys, lat) = self.solve(x, y, window)?;
        self.finite((self.score)(&lat, xs.len() / d, ys.len() / d))
    }

    pub(super) fn cost_matrix(
        &self,
        x: Ser,
        y: Ser,
        window: DtwWindow,
    ) -> Result<FdMatrix, FdarError> {
        let d = x.d;
        let (xs, ys, lat) = self.solve(x, y, window)?;
        let (n, m) = (xs.len() / d, ys.len() / d);
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
        x: Ser,
        y: Ser,
        window: DtwWindow,
    ) -> Result<ElasticDistancePath, FdarError> {
        let d = x.d;
        let (xs, ys, lat) = self.solve(x, y, window)?;
        let (n, m) = (xs.len() / d, ys.len() / d);
        let distance = self.finite((self.path_score)(&lat, n, m))?;
        let path = match self.traceback {
            Traceback::MinReturn => min_return_path(&lat, n, m),
            Traceback::Lcss(epsilon) => {
                let bounds = window.row_bounds(n, m)?;
                lcss::lcss_return_path(Ser { v: &xs, d }, Ser { v: &ys, d }, epsilon, &bounds, &lat)
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
                self_distance_matrix(rows1.len(), |i, j| {
                    self.pair(Ser::uni(&rows1[i]), Ser::uni(&rows1[j]), &bounds)
                })
            }
            Some(other) => {
                let rows2 = self.prepared_rows(other);
                let bounds = window.row_bounds(rows1[0].len(), rows2[0].len())?;
                cross_distance_matrix(rows1.len(), rows2.len(), |i, j| {
                    self.pair(Ser::uni(&rows1[i]), Ser::uni(&rows2[j]), &bounds)
                })
            }
        };
        if result.as_slice().iter().all(|d| d.is_finite()) {
            Ok(result)
        } else {
            Err(self.no_path())
        }
    }

    /// Pairwise distances of collections of `d × m` series (lengths may vary;
    /// the window is sized per pair, as in aeon's unequal-length path).
    pub(super) fn pairwise_multivariate(
        &self,
        data: &[FdMatrix],
        other: Option<&[FdMatrix]>,
        window: DtwWindow,
    ) -> Result<FdMatrix, FdarError> {
        let d = collection_channels(data, other)?;
        let prep = |parameter, xs: &[FdMatrix]| -> Result<Vec<Vec<f64>>, FdarError> {
            xs.iter()
                .map(|x| Ok(self.prepare(parameter, Ser::of(x))?.into_owned()))
                .collect()
        };
        let rows1 = prep("data", data)?;
        let rows2 = match other {
            Some(o) => Some(prep("other", o)?),
            None => None,
        };
        let one = |a: &[f64], b: &[f64]| -> Result<f64, FdarError> {
            let bounds = window.row_bounds(a.len() / d, b.len() / d)?;
            self.finite(self.pair(Ser { v: a, d }, Ser { v: b, d }, &bounds))
        };
        crate::metric::pairwise_result(rows1.len(), rows2.as_ref().map(Vec::len), |i, j| {
            one(&rows1[i], &rows2.as_ref().unwrap_or(&rows1)[j])
        })
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

    fn pair(&self, x: Ser, y: Ser, bounds: &[(usize, usize)]) -> f64 {
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
        distance_mv: $dist_mv:ident,
        pairwise_mv: $pair_mv:ident,
        cost_mv: $cost_mv:ident,
        path_mv: $path_mv:ident,
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
            ($recipe)?.distance(super::Ser::uni(x), Ser::uni(y), window)
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
            ($recipe)?.cost_matrix(super::Ser::uni(x), Ser::uni(y), window)
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
            ($recipe)?.alignment_path(super::Ser::uni(x), Ser::uni(y), window)
        }

        #[doc = concat!($label, " distance between two multivariate series (aeon `", stringify!($dist), "` on `(n_channels, n_timepoints)` arrays).")]
        ///
        /// `x` (`d × n`) and `y` (`d × m`) hold one channel per row and one time
        /// point per column; see the [module docs](crate::metric::elastic_family) for aeon's multichannel
        /// semantics. With `d = 1` the result is bit-identical to the univariate
        #[doc = concat!("[`", stringify!($dist), "`].")]
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the univariate distance; additionally
        /// [`FdarError::InvalidDimension`] if a series has no channel or the
        /// channel counts differ.
        ///
        /// # Examples
        /// ```
        /// use fdars_core::matrix::FdMatrix;
        #[doc = concat!("use fdars_core::metric::{", stringify!($dist_mv), ", DtwWindow};")]
        /// // Two channels, six time points (column-major: one column per time point).
        /// let x = FdMatrix::from_column_major(
        ///     vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0, 1.5, -0.5, 0.5, 0.0, 0.0, 0.5], 2, 6,
        /// ).unwrap();
        /// let y = FdMatrix::from_column_major(
        ///     vec![0.0, 0.5, 0.5, 1.0, 1.5, 0.0, 2.0, -0.5, 1.0, 0.0, 0.0, 0.5], 2, 6,
        /// ).unwrap();
        #[doc = concat!("let d = ", stringify!($dist_mv), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        #[doc = concat!("let s = ", stringify!($dist_mv), "(&y, &x, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(d >= 0.0 && (d - s).abs() < 1e-12);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $dist_mv(
            x: &FdMatrix,
            y: &FdMatrix,
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<f64, FdarError> {
            let (xs, ys) = super::mv_pair(x, y)?;
            ($recipe)?.distance(xs, ys, window)
        }

        #[doc = concat!($label, " distance matrix of a collection of multivariate series (aeon `", stringify!($pair), "`).")]
        ///
        /// Each element of `data` (and `other`) is one `d × m` series (channels
        /// as rows); all series share `d`, lengths may differ (the window is
        /// sized per pair). With `other = None` the symmetric self-distance
        /// matrix (zero diagonal) is returned, otherwise the cross-distance
        /// matrix. Pairs are computed in parallel with the `parallel` feature.
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the pointwise distance; additionally
        /// [`FdarError::InvalidDimension`] if channel counts differ.
        ///
        /// # Examples
        /// ```
        /// use fdars_core::matrix::FdMatrix;
        #[doc = concat!("use fdars_core::metric::{", stringify!($pair_mv), ", DtwWindow};")]
        /// let series: Vec<FdMatrix> = (0..3)
        ///     .map(|s| FdMatrix::from_column_major(
        ///         (0..12).map(|k| ((k * 5 + s * 3) % 7) as f64 * 0.4).collect(), 2, 6,
        ///     ).unwrap())
        ///     .collect();
        #[doc = concat!("let d = ", stringify!($pair_mv), "(&series, None, DtwWindow::None", $ex, ").unwrap();")]
        /// assert_eq!(d.shape(), (3, 3));
        /// assert_eq!(d[(1, 1)], 0.0);
        /// assert!((d[(0, 2)] - d[(2, 0)]).abs() < 1e-12);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $pair_mv(
            data: &[FdMatrix],
            other: Option<&[FdMatrix]>,
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<FdMatrix, FdarError> {
            ($recipe)?.pairwise_multivariate(data, other, window)
        }

        #[doc = concat!($label, " accumulated cost matrix of two multivariate series (aeon `", stringify!($cost), "`).")]
        ///
        /// Series are `d × m` (channels as rows). Out-of-window cells hold aeon's
        /// fill value (`inf`, or `0` for LCSS).
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the univariate cost matrix; additionally
        /// [`FdarError::InvalidDimension`] if channel counts differ.
        ///
        /// # Examples
        /// ```
        /// use fdars_core::matrix::FdMatrix;
        #[doc = concat!("use fdars_core::metric::{", stringify!($cost_mv), ", DtwWindow};")]
        /// let x = FdMatrix::from_column_major(
        ///     vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0, 1.5, -0.5, 0.5, 0.0, 0.0, 0.5], 2, 6,
        /// ).unwrap();
        /// let y = FdMatrix::from_column_major(
        ///     vec![0.0, 0.5, 0.5, 1.0, 1.5, 0.0, 2.0, -0.5, 1.0, 0.0], 2, 5,
        /// ).unwrap();
        #[doc = concat!("let cm = ", stringify!($cost_mv), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(cm.nrows() >= 4 && cm.ncols() >= 3);
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $cost_mv(
            x: &FdMatrix,
            y: &FdMatrix,
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<FdMatrix, FdarError> {
            let (xs, ys) = super::mv_pair(x, y)?;
            ($recipe)?.cost_matrix(xs, ys, window)
        }

        #[doc = concat!($label, " optimal alignment path and distance of two multivariate series (aeon `", stringify!($path), "`).")]
        ///
        /// Series are `d × m` (channels as rows).
        ///
        #[doc = $pdoc]
        ///
        /// # Errors
        /// As for the univariate path; additionally
        /// [`FdarError::InvalidDimension`] if channel counts differ.
        ///
        /// # Examples
        /// ```
        /// use fdars_core::matrix::FdMatrix;
        #[doc = concat!("use fdars_core::metric::{", stringify!($path_mv), ", DtwWindow};")]
        /// let x = FdMatrix::from_column_major(
        ///     vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0, 1.5, -0.5, 0.5, 0.0, 0.0, 0.5], 2, 6,
        /// ).unwrap();
        /// let y = FdMatrix::from_column_major(
        ///     vec![0.0, 0.5, 0.5, 1.0, 1.5, 0.0, 2.0, -0.5, 1.0, 0.0, 0.0, 0.5], 2, 6,
        /// ).unwrap();
        #[doc = concat!("let res = ", stringify!($path_mv), "(&x, &y, DtwWindow::None", $ex, ").unwrap();")]
        /// assert!(res.distance >= 0.0);
        /// assert!(res.path.windows(2).all(|w| w[0].0 <= w[1].0 && w[0].1 <= w[1].1));
        /// ```
        #[must_use = "expensive computation whose result should not be discarded"]
        pub fn $path_mv(
            x: &FdMatrix,
            y: &FdMatrix,
            window: DtwWindow
            $(, $p: $t)*
        ) -> Result<ElasticDistancePath, FdarError> {
            let (xs, ys) = super::mv_pair(x, y)?;
            ($recipe)?.alignment_path(xs, ys, window)
        }
    };
}
pub(super) use elastic_family_fns;
