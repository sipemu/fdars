//! DTW lower bounds, early-abandoning pruned DTW and exact 1-NN search.
//!
//! All quantities follow the convention of
//! [`dtw_distance_windowed`](super::dtw::dtw_distance_windowed): the DTW cost is
//! the accumulated `Σ |x_i − y_j|^p` along the optimal admissible path, **with no
//! root taken**. tslearn's `dtw` and `lb_keogh` (and dtaidistance's) report the
//! square root of the `p = 2` value; square their output to compare.
//!
//! The query `x` indexes the rows `i` of the alignment lattice and the candidate
//! `y` its columns `j`; a [`DtwWindow`] restricts the admissible cells. Every
//! bound here is a lower bound of the windowed DTW under the **same** window:
//!
//! - [`lb_kim`] — first/last/min/max bound (Kim, Park & Chu 2001), O(1) after an
//!   O(n) scan, window independent;
//! - [`lb_keogh`] with [`lb_envelope`] / [`lb_envelope_windowed`] — LB_Keogh
//!   (Keogh 2002) over envelopes computed with Lemire's (2009) streaming min/max;
//! - [`lb_webb`] — LB_Webb (Webb & Petitjean 2021, *Tight lower bounds for
//!   Dynamic Time Warping*, Pattern Recognition 115), always `≥ LB_Keogh`;
//! - [`dtw_distance_early_abandon`] — EAPrunedDTW (Herrmann & Webb 2021):
//!   exact windowed DTW that abandons as soon as the cost must exceed a cutoff;
//! - [`dtw_nearest_neighbor`] — exact 1-NN search cascading the bounds above.
//!
//! # Generalised windows
//! The papers state LB_Keogh / LB_Webb for equal lengths and a Sakoe-Chiba
//! radius `w`, where row `i` may align with columns `[i − w, i + w]`. Here the
//! envelopes are taken over the admissible ranges of an arbitrary [`DtwWindow`]
//! (rows for the candidate's envelope, columns for the query's), so the bounds
//! hold for Itakura and aeon windows and for unequal lengths too. For equal
//! lengths and [`DtwWindow::SakoeChiba`] they reduce exactly to the published
//! definitions.

use std::sync::OnceLock;

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::dtw::{validate_dtw_inputs, LocalCost};
use super::window::DtwWindow;

// ---------------------------------------------------------------------------
// Envelopes
// ---------------------------------------------------------------------------

/// Lower and upper envelope of a series (LB_Keogh envelope).
///
/// `lower[i]` / `upper[i]` are the minimum / maximum of the series over the
/// window of position `i`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwEnvelope {
    /// Running minimum over each window (tslearn `envelope_down`).
    pub lower: Vec<f64>,
    /// Running maximum over each window (tslearn `envelope_up`).
    pub upper: Vec<f64>,
}

/// `true` when both ends of the half-open ranges are non-decreasing.
fn is_monotone(bounds: &[(usize, usize)]) -> bool {
    bounds
        .windows(2)
        .all(|w| w[0].0 <= w[1].0 && w[0].1 <= w[1].1)
}

/// Running extremum of `values` over half-open, non-empty ranges `[lo, hi)`.
///
/// `dominated(old, new)` says when an older value can never again be the
/// extremum once `new` has entered the window (`old <= new` for a maximum).
/// Uses Lemire's streaming min/max — a monotone deque of indices, here a
/// `Vec` with a moving head, O(len + output) — when the ranges slide
/// monotonically, and a direct scan otherwise.
fn sliding_extremum(
    values: &[f64],
    bounds: &[(usize, usize)],
    dominated: impl Fn(f64, f64) -> bool,
) -> Vec<f64> {
    if !is_monotone(bounds) {
        return bounds
            .iter()
            .map(|&(lo, hi)| {
                values[lo + 1..hi].iter().fold(
                    values[lo],
                    |acc, &v| {
                        if dominated(acc, v) {
                            v
                        } else {
                            acc
                        }
                    },
                )
            })
            .collect();
    }
    let mut out = Vec::with_capacity(bounds.len());
    let mut deque: Vec<usize> = Vec::with_capacity(values.len());
    let mut head = 0usize;
    let mut next = 0usize;
    for &(lo, hi) in bounds {
        while next < hi {
            let v = values[next];
            while deque.len() > head && dominated(values[deque[deque.len() - 1]], v) {
                deque.pop();
            }
            deque.push(next);
            next += 1;
        }
        while deque[head] < lo {
            head += 1;
        }
        out.push(values[deque[head]]);
    }
    out
}

/// Running minimum over each range.
fn sliding_min(values: &[f64], bounds: &[(usize, usize)]) -> Vec<f64> {
    sliding_extremum(values, bounds, |old, new| old >= new)
}

/// Running maximum over each range.
fn sliding_max(values: &[f64], bounds: &[(usize, usize)]) -> Vec<f64> {
    sliding_extremum(values, bounds, |old, new| old <= new)
}

/// Running (min, max) over each range.
fn sliding_extrema(values: &[f64], bounds: &[(usize, usize)]) -> (Vec<f64>, Vec<f64>) {
    (sliding_min(values, bounds), sliding_max(values, bounds))
}

fn empty_series_error(parameter: &'static str) -> FdarError {
    FdarError::InvalidDimension {
        parameter,
        expected: "non-empty series".to_string(),
        actual: "length 0".to_string(),
    }
}

/// LB_Keogh envelope of `x` with a symmetric radius (tslearn `lb_envelope`).
///
/// `lower[i] = min x[max(0, i − radius) ..= min(n − 1, i + radius)]` and
/// `upper[i]` the corresponding maximum, computed in O(n) with Lemire's
/// streaming min/max. For two series of equal length `n`, the envelope of the
/// candidate with radius `r` is the one that bounds the DTW under
/// [`DtwWindow::SakoeChiba`]` { radius: r }`; for other windows or unequal
/// lengths use [`lb_envelope_windowed`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `x` is empty.
///
/// # Examples
/// ```
/// use fdars_core::lb_envelope;
/// // tslearn docstring: lb_envelope([1, 2, 3, 2, 1], radius=1)
/// let env = lb_envelope(&[1.0, 2.0, 3.0, 2.0, 1.0], 1).unwrap();
/// assert_eq!(env.lower, vec![1.0, 1.0, 2.0, 1.0, 1.0]);
/// assert_eq!(env.upper, vec![2.0, 3.0, 3.0, 3.0, 2.0]);
/// ```
pub fn lb_envelope(x: &[f64], radius: usize) -> Result<DtwEnvelope, FdarError> {
    if x.is_empty() {
        return Err(empty_series_error("x"));
    }
    let n = x.len();
    let bounds: Vec<(usize, usize)> = (0..n)
        .map(|i| {
            (
                i.saturating_sub(radius),
                i.saturating_add(radius).saturating_add(1).min(n),
            )
        })
        .collect();
    let (lower, upper) = sliding_extrema(x, &bounds);
    Ok(DtwEnvelope { lower, upper })
}

/// Envelope of the candidate `y` as seen by a query of length `query_len`
/// under an arbitrary [`DtwWindow`].
///
/// Position `i` (a query index, `0 ≤ i < query_len`) holds the min / max of
/// `y` over the columns that row `i` may align with under `window`. With this
/// envelope, [`lb_keogh`] lower-bounds
/// [`dtw_distance_windowed`](super::dtw::dtw_distance_windowed)`(query, y, p, window)`
/// for any window and any lengths. For equal lengths and
/// [`DtwWindow::SakoeChiba`]` { radius }` it equals [`lb_envelope`]`(y, radius)`;
/// for unequal lengths the Sakoe-Chiba envelope matches dtaidistance's
/// `lb_keogh` window (`window = radius + 1`).
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `y` is empty or `query_len == 0`.
/// - [`FdarError::InvalidParameter`] if the window is invalid or unfeasible.
///
/// # Examples
/// ```
/// use fdars_core::{lb_envelope, lb_envelope_windowed, DtwWindow};
/// let y = [0.0, 2.0, -1.0, 3.0, 1.0];
/// let a = lb_envelope_windowed(&y, 5, DtwWindow::SakoeChiba { radius: 1 }).unwrap();
/// assert_eq!(a, lb_envelope(&y, 1).unwrap());
/// ```
pub fn lb_envelope_windowed(
    y: &[f64],
    query_len: usize,
    window: DtwWindow,
) -> Result<DtwEnvelope, FdarError> {
    if y.is_empty() {
        return Err(empty_series_error("y"));
    }
    let rows = window.row_bounds(query_len, y.len())?;
    let (lower, upper) = sliding_extrema(y, &rows);
    Ok(DtwEnvelope { lower, upper })
}

// ---------------------------------------------------------------------------
// LB_Keogh and LB_Kim
// ---------------------------------------------------------------------------

/// Distance of `q` to the interval `[lo, up]` under the local cost.
#[inline]
fn keogh_term(q: f64, lo: f64, up: f64, cost: LocalCost) -> f64 {
    if q > up {
        cost.eval(q, up)
    } else if q < lo {
        cost.eval(q, lo)
    } else {
        0.0
    }
}

fn keogh_sum(query: &[f64], lower: &[f64], upper: &[f64], cost: LocalCost) -> f64 {
    query
        .iter()
        .zip(lower.iter().zip(upper))
        .map(|(&q, (&lo, &up))| keogh_term(q, lo, up, cost))
        .sum()
}

fn validate_p(p: f64) -> Result<(), FdarError> {
    if p.is_finite() && p > 0.0 {
        Ok(())
    } else {
        Err(FdarError::InvalidParameter {
            parameter: "p",
            message: format!("must be finite and > 0, got {p}"),
        })
    }
}

/// LB_Keogh: cost of the query's excursions outside a candidate envelope.
///
/// Returns `Σ_i |query_i − upper_i|^p` over `query_i > upper_i` plus
/// `Σ_i |lower_i − query_i|^p` over `query_i < lower_i` — no root, the same
/// convention as [`dtw_distance_windowed`](super::dtw::dtw_distance_windowed).
/// tslearn's `lb_keogh` returns the square root of the `p = 2` value.
///
/// It lower-bounds the windowed DTW when `candidate_envelope` is the
/// candidate's envelope for that window ([`lb_envelope`] for an equal-length
/// Sakoe-Chiba band, [`lb_envelope_windowed`] in general). Valid for every
/// `p > 0`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `query` is empty or the envelope length
///   differs from `query.len()`.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`.
///
/// # Examples
/// ```
/// use fdars_core::{lb_envelope, lb_keogh};
/// // tslearn docstring: lb_keogh([0]*5, [1, 2, 3, 2, 1], radius=1) = 2.8284… = sqrt(8)
/// let env = lb_envelope(&[1.0, 2.0, 3.0, 2.0, 1.0], 1).unwrap();
/// let lb = lb_keogh(&[0.0; 5], &env, 2.0).unwrap();
/// assert_eq!(lb, 8.0);
/// ```
pub fn lb_keogh(query: &[f64], candidate_envelope: &DtwEnvelope, p: f64) -> Result<f64, FdarError> {
    if query.is_empty() {
        return Err(empty_series_error("query"));
    }
    let env = candidate_envelope;
    if env.lower.len() != query.len() || env.upper.len() != query.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "candidate_envelope",
            expected: format!("lower/upper of length {}", query.len()),
            actual: format!("lengths {} / {}", env.lower.len(), env.upper.len()),
        });
    }
    validate_p(p)?;
    Ok(keogh_sum(query, &env.lower, &env.upper, LocalCost::new(p)))
}

/// First, last, minimum and maximum of a series (LB_Kim features).
#[derive(Clone, Copy)]
struct KimFeatures {
    first: f64,
    last: f64,
    min: f64,
    max: f64,
    len: usize,
}

impl KimFeatures {
    fn new(x: &[f64]) -> Self {
        KimFeatures {
            first: x[0],
            last: x[x.len() - 1],
            min: x.iter().copied().fold(f64::INFINITY, f64::min),
            max: x.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            len: x.len(),
        }
    }

    fn bound(&self, other: &KimFeatures, cost: LocalCost) -> f64 {
        let first = cost.eval(self.first, other.first);
        let last = if self.len == 1 && other.len == 1 {
            0.0
        } else {
            cost.eval(self.last, other.last)
        };
        (first + last)
            .max(cost.eval(self.max, other.max))
            .max(cost.eval(self.min, other.min))
    }
}

/// LB_Kim lower bound (first / last / minimum / maximum).
///
/// Adapted to the sum-of-costs DTW (no root): every warping path contains the
/// two corner cells `(0, 0)` and `(n − 1, m − 1)`, and aligns the maximum
/// (minimum) of each series with some element of the other, so
/// `max(c(x₀, y₀) + c(x_{n−1}, y_{m−1}), c(max x, max y), c(min x, min y))`
/// with `c(a, b) = |a − b|^p` lower-bounds the DTW under **every** window
/// (corner cells are counted once when `n = m = 1`). Valid for every `p > 0`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `x` or `y` is empty.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`.
///
/// # Examples
/// ```
/// use fdars_core::{dtw_distance_windowed, lb_kim, DtwWindow};
/// let x = [0.0, 3.0, 1.0, 2.0];
/// let y = [0.5, 0.0, 0.25, 0.0];
/// let lb = lb_kim(&x, &y, 2.0).unwrap();
/// assert_eq!(lb, 6.25); // (max x − max y)² = (3 − 0.5)²
/// assert!(lb <= dtw_distance_windowed(&x, &y, 2.0, DtwWindow::None).unwrap());
/// ```
pub fn lb_kim(x: &[f64], y: &[f64], p: f64) -> Result<f64, FdarError> {
    validate_dtw_inputs(x, y, p)?;
    Ok(KimFeatures::new(x).bound(&KimFeatures::new(y), LocalCost::new(p)))
}

// ---------------------------------------------------------------------------
// LB_Webb
// ---------------------------------------------------------------------------

/// Admissible row ranges and (hulls of) column ranges of a windowed lattice.
struct Lattice {
    /// `rows[i] = [lo, hi)`: columns row `i` may align with.
    rows: Vec<(usize, usize)>,
    /// `cols[j] = [lo, hi)`: rows column `j` may align with (contiguous hull).
    cols: Vec<(usize, usize)>,
}

impl Lattice {
    fn new(n: usize, m: usize, window: DtwWindow) -> Result<Self, FdarError> {
        let rows = window.row_bounds(n, m)?;
        let cols = column_ranges(&rows, m);
        Ok(Lattice { rows, cols })
    }

    #[inline]
    fn admissible(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = self.rows[i];
        lo <= j && j < hi
    }
}

/// Column ranges from row ranges. Exact (two pointers, O(n + m)) for monotone
/// staircase windows; otherwise the contiguous hull, which is a superset and
/// keeps every bound valid (only looser).
fn column_ranges(rows: &[(usize, usize)], m: usize) -> Vec<(usize, usize)> {
    let n = rows.len();
    if is_monotone(rows) {
        let (mut a, mut b) = (0usize, 0usize);
        return (0..m)
            .map(|j| {
                while a < n && rows[a].1 <= j {
                    a += 1;
                }
                while b < n && rows[b].0 <= j {
                    b += 1;
                }
                (a, b)
            })
            .collect();
    }
    let mut cols = vec![(usize::MAX, 0usize); m];
    for (i, &(lo, hi)) in rows.iter().enumerate() {
        for c in &mut cols[lo..hi] {
            c.0 = c.0.min(i);
            c.1 = c.1.max(i + 1);
        }
    }
    cols
}

/// Query-side envelopes for LB_Webb (`A` = query, indexed by rows).
struct WebbQuery<'a> {
    a: &'a [f64],
    /// Envelope of `A` over each column's rows (length `m`).
    la: Vec<f64>,
    ua: Vec<f64>,
    /// `min` of `ua` over each row's columns (length `n`): `L^{U^A}`.
    lua: Vec<f64>,
    /// `max` of `la` over each row's columns (length `n`): `U^{L^A}`.
    ula: Vec<f64>,
}

impl<'a> WebbQuery<'a> {
    fn new(a: &'a [f64], lat: &Lattice) -> Self {
        let (la, ua) = sliding_extrema(a, &lat.cols);
        let lua = sliding_min(&ua, &lat.rows);
        let ula = sliding_max(&la, &lat.rows);
        WebbQuery {
            a,
            la,
            ua,
            lua,
            ula,
        }
    }
}

/// Candidate-side envelopes (`B` = candidate, indexed by columns).
struct WebbCandidate {
    /// Envelope of `B` over each row's columns (length `n`): the LB_Keogh envelope.
    lb: Vec<f64>,
    ub: Vec<f64>,
    /// Envelopes of the envelope, built on first use (most 1-NN candidates are
    /// discarded by LB_Keogh first): `max` of `lb` and `min` of `ub` over each
    /// column's rows (length `m`), i.e. `(U^{L^B}, L^{U^B})`.
    second: OnceLock<(Vec<f64>, Vec<f64>)>,
}

impl WebbCandidate {
    fn new(b: &[f64], lat: &Lattice) -> Self {
        let (lb, ub) = sliding_extrema(b, &lat.rows);
        WebbCandidate {
            lb,
            ub,
            second: OnceLock::new(),
        }
    }

    /// `(U^{L^B}, L^{U^B})`.
    fn second(&self, lat: &Lattice) -> &(Vec<f64>, Vec<f64>) {
        self.second.get_or_init(|| {
            (
                sliding_max(&self.lb, &lat.cols),
                sliding_min(&self.ub, &lat.cols),
            )
        })
    }
}

/// The seven possible first-three-cell path prefixes (Webb & Petitjean, Fig. 11).
const END_PATHS: [[(usize, usize); 3]; 7] = [
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (0, 1), (1, 2)],
    [(0, 0), (1, 1), (1, 2)],
    [(0, 0), (1, 1), (2, 2)],
    [(0, 0), (1, 1), (2, 1)],
    [(0, 0), (1, 0), (2, 1)],
    [(0, 0), (1, 0), (2, 0)],
];

/// Minimum cost over the admissible three-cell prefixes (or, mirrored, suffixes).
fn min_end_path(a: &[f64], b: &[f64], lat: &Lattice, cost: LocalCost, mirror: bool) -> Option<f64> {
    let (n, m) = (a.len(), b.len());
    let mut best: Option<f64> = None;
    for path in &END_PATHS {
        let mut s = 0.0;
        let mut ok = true;
        for &(di, dj) in path {
            let (i, j) = if mirror {
                (n - 1 - di, m - 1 - dj)
            } else {
                (di, dj)
            };
            if !lat.admissible(i, j) {
                ok = false;
                break;
            }
            s += cost.eval(a[i], b[j]);
        }
        if ok {
            best = Some(best.map_or(s, |v: f64| v.min(s)));
        }
    }
    best
}

/// `MinLRPaths(A, B)`: requires `n, m ≥ 6` so the two end paths and the bridge
/// rows / columns `[3, len − 3)` are disjoint.
fn min_lr_paths(a: &[f64], b: &[f64], lat: &Lattice, cost: LocalCost) -> Option<f64> {
    if a.len() < 6 || b.len() < 6 {
        return None;
    }
    Some(min_end_path(a, b, lat, cost, false)? + min_end_path(a, b, lat, cost, true)?)
}

/// LB_Keogh bridge over rows `[r0, r1)`, recording prefix counts of rows that
/// break "free above U^A" / "free below L^A".
fn webb_keogh_bridge(
    q: &WebbQuery,
    c: &WebbCandidate,
    (r0, r1): (usize, usize),
    cost: LocalCost,
    bad_up: &mut [u32],
    bad_dn: &mut [u32],
) -> f64 {
    let mut lb = 0.0;
    bad_up[0] = 0;
    bad_dn[0] = 0;
    for (i, &ai) in q.a.iter().enumerate() {
        let (mut up, mut dn) = (false, false);
        if i >= r0 && i < r1 {
            if ai > c.ub[i] {
                lb += cost.eval(ai, c.ub[i]);
                up = true;
                dn = c.ub[i] < q.ula[i];
            } else if ai < c.lb[i] {
                lb += cost.eval(ai, c.lb[i]);
                dn = true;
                up = c.lb[i] > q.lua[i];
            }
        }
        bad_up[i + 1] = bad_up[i] + u32::from(up);
        bad_dn[i + 1] = bad_dn[i] + u32::from(dn);
    }
    lb
}

/// One side of the LB_Webb allowance for `b_j` beyond the query envelope value
/// `edge` (`U^A_j` above, `L^A_j` below). `inner` is `U^{L^B}_j` (above) or
/// `L^{U^B}_j` (below) and `beyond(u, v)` tests that `u` lies strictly beyond `v`.
#[inline]
fn webb_side_term(
    bj: f64,
    edge: f64,
    inner: f64,
    free: bool,
    cost: LocalCost,
    beyond: impl Fn(f64, f64) -> bool,
) -> f64 {
    if free {
        cost.eval(bj, edge)
    } else if beyond(bj, inner) && !beyond(edge, inner) {
        cost.eval(bj, edge) - cost.eval(inner, edge)
    } else {
        0.0
    }
}

/// Allowance for candidate element `b_j` that the Keogh bridge cannot reach.
#[inline]
fn webb_column_term(
    j: usize,
    bj: f64,
    q: &WebbQuery,
    (ulb, lub): (&[f64], &[f64]),
    free: (bool, bool),
    cost: LocalCost,
) -> f64 {
    let (ua, la) = (q.ua[j], q.la[j]);
    if bj > ua {
        webb_side_term(bj, ua, ulb[j], free.0, cost, |u, v| u > v)
    } else if bj < la {
        webb_side_term(bj, la, lub[j], free.1, cost, |u, v| u < v)
    } else {
        0.0
    }
}

/// One LB_Webb pass: `base` (MinLRPaths or 0) + Keogh bridge over rows
/// `[edge, n − edge)` + candidate allowances over columns `[edge, m − edge)`.
fn webb_pass(
    q: &WebbQuery,
    b: &[f64],
    c: &WebbCandidate,
    lat: &Lattice,
    cost: LocalCost,
    (edge, base): (usize, f64),
    cutoff: f64,
) -> f64 {
    let n = q.a.len();
    let mut bad_up = vec![0u32; n + 1];
    let mut bad_dn = vec![0u32; n + 1];
    let mut lb = base + webb_keogh_bridge(q, c, (edge, n - edge), cost, &mut bad_up, &mut bad_dn);
    if lb > cutoff {
        return lb;
    }
    let (ulb, lub) = c.second(lat);
    for (j, &bj) in b.iter().enumerate().take(b.len() - edge).skip(edge) {
        let (lo, hi) = lat.cols[j];
        let free = (bad_up[hi] == bad_up[lo], bad_dn[hi] == bad_dn[lo]);
        lb += webb_column_term(j, bj, q, (ulb, lub), free, cost);
    }
    lb
}

/// LB_Webb as the larger of the variant with left/right paths and the
/// parameter-free `LB_Webb_NoLR` (both valid; the latter is always `≥ LB_Keogh`).
fn webb_bound(
    q: &WebbQuery,
    b: &[f64],
    c: &WebbCandidate,
    lat: &Lattice,
    cost: LocalCost,
    cutoff: f64,
) -> f64 {
    let nolr = webb_pass(q, b, c, lat, cost, (0, 0.0), cutoff);
    if nolr > cutoff {
        return nolr;
    }
    match min_lr_paths(q.a, b, lat, cost) {
        Some(base) => nolr.max(webb_pass(q, b, c, lat, cost, (3, base), cutoff)),
        None => nolr,
    }
}

fn validate_webb_p(p: f64) -> Result<(), FdarError> {
    if p >= 1.0 {
        Ok(())
    } else {
        Err(FdarError::InvalidParameter {
            parameter: "p",
            message: format!("LB_Webb needs a convex local cost |a − b|^p, i.e. p >= 1, got {p}"),
        })
    }
}

/// LB_Webb lower bound (Webb & Petitjean 2021).
///
/// Adds to an LB_Keogh bridge an allowance for elements of the candidate `y`
/// that the bridge cannot reach: `y_j` above the query's envelope `U^A_j`
/// contributes `c(y_j, U^A_j)` when it is *free above* (no query element in
/// its window charged an LB_Keogh distance that could reach above `U^A`),
/// and `c(y_j, U^A_j) − c(U^{L^B}_j, U^A_j)` otherwise (symmetrically below).
/// The variant with the paper's left/right paths (`MinLRPaths`, the minimum
/// over the seven admissible first / last three-cell paths, for `n, m ≥ 6`)
/// and the parameter-free `LB_Webb_NoLR` are both computed and the larger is
/// returned, so `lb_keogh ≤ lb_webb ≤ DTW` holds unconditionally.
///
/// Envelopes follow the admissible ranges of `window` (see the module docs),
/// so this bounds `dtw_distance_windowed(x, y, p, window)` for every window
/// and for unequal lengths. The bound needs the convexity of `|a − b|^p`
/// (Webb & Petitjean, Theorem 2), so `p ≥ 1` is required.
///
/// This convenience entry point builds all envelopes (O(n + m) with Lemire's
/// streaming min/max); [`dtw_nearest_neighbor`] precomputes them once.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `x` or `y` is empty.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `≥ 1`, or the
///   window is invalid / unfeasible for these lengths.
///
/// # Examples
/// ```
/// use fdars_core::{dtw_distance_windowed, lb_envelope, lb_keogh, lb_webb, DtwWindow};
/// let x = [0.0, 1.0, 3.0, 2.0, 0.0, -1.0, -2.0, 0.0];
/// let y = [0.0, -1.0, 0.5, 4.0, 3.0, 1.0, 0.0, 1.0];
/// let w = DtwWindow::SakoeChiba { radius: 1 };
/// let keogh = lb_keogh(&x, &lb_envelope(&y, 1).unwrap(), 2.0).unwrap();
/// let webb = lb_webb(&x, &y, 2.0, w).unwrap();
/// let dtw = dtw_distance_windowed(&x, &y, 2.0, w).unwrap();
/// assert!(keogh <= webb && webb <= dtw);
/// ```
pub fn lb_webb(x: &[f64], y: &[f64], p: f64, window: DtwWindow) -> Result<f64, FdarError> {
    validate_dtw_inputs(x, y, p)?;
    validate_webb_p(p)?;
    let lat = Lattice::new(x.len(), y.len(), window)?;
    let q = WebbQuery::new(x, &lat);
    let c = WebbCandidate::new(y, &lat);
    Ok(webb_bound(
        &q,
        y,
        &c,
        &lat,
        LocalCost::new(p),
        f64::INFINITY,
    ))
}

// ---------------------------------------------------------------------------
// EAPrunedDTW
// ---------------------------------------------------------------------------

/// Early-abandoning pruned DTW core (Herrmann & Webb 2021).
///
/// Cells whose value exceeds `limit(i)` are pruned (stored as `+inf`); each row
/// starts at the first kept cell of the previous row and stops once it is past
/// the previous row's last kept cell and the current cell is pruned. Every cell
/// whose true value lies within the limit is computed with the exact recurrence
/// and operand order of `dtw_distance_windowed`, hence bit-identically.
///
/// `rest[i]` (optional) is a lower bound on the cost still to come after row
/// `i`; `limit(i) = ub_relaxed − rest[i]`. Returns `Some(d)` iff the DTW `d` is
/// `≤ ub`.
fn ea_pruned_dtw(
    x: &[f64],
    y: &[f64],
    rows: &[(usize, usize)],
    cost: LocalCost,
    ub: f64,
    rest: Option<(&[f64], f64)>,
) -> Option<f64> {
    let m = y.len();
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    let mut prev_span = (0usize, 1usize);
    let (mut prev_first, mut prev_last) = (0usize, 0usize);
    for (i, &xi) in x.iter().enumerate() {
        let limit = match rest {
            Some((r, ub_relaxed)) => ub_relaxed - r[i],
            None => ub,
        };
        let (lo, hi) = rows[i];
        let start = (lo + 1).max(prev_first).max(1);
        let (first, last, end) =
            ea_fill_row(&prev, &mut curr, xi, y, (start, hi), prev_last, limit, cost);
        prev[prev_span.0..prev_span.1].fill(f64::INFINITY);
        std::mem::swap(&mut prev, &mut curr);
        prev_span = (start, end);
        prev_first = first?;
        prev_last = last;
    }
    (prev_last == m && prev[m] <= ub).then_some(prev[m])
}

/// Fill one pruned row over 1-based columns `[start, hi]`. Returns the first
/// and last kept column and the end of the written span.
#[allow(clippy::too_many_arguments)]
#[inline]
fn ea_fill_row(
    prev: &[f64],
    curr: &mut [f64],
    xi: f64,
    y: &[f64],
    (start, hi): (usize, usize),
    prev_last: usize,
    limit: f64,
    cost: LocalCost,
) -> (Option<usize>, usize, usize) {
    let mut first = None;
    let mut last = 0usize;
    let mut j = start;
    while j <= hi {
        let c = cost.eval(xi, y[j - 1]);
        let v = c + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        if v <= limit {
            curr[j] = v;
            first.get_or_insert(j);
            last = j;
        } else {
            curr[j] = f64::INFINITY;
            if j > prev_last {
                j += 1;
                break;
            }
        }
        j += 1;
    }
    (first, last, j.min(curr.len()))
}

/// Exact windowed DTW with pruning and early abandoning (EAPrunedDTW).
///
/// Returns `Some(d)` with `d` bit-identical to
/// [`dtw_distance_windowed`](super::dtw::dtw_distance_windowed)`(x, y, p, window)`
/// whenever `d ≤ cutoff`, and `None` as soon as the cost is certain to exceed
/// `cutoff` (Herrmann & Webb 2021, *Early abandoning and pruning for elastic
/// distances including dynamic time warping*, DMKD 35). Cells above the cutoff
/// are pruned, each row skips the pruned prefix of the previous row and stops
/// at the previous row's pruning point, and the computation abandons when a
/// whole row is pruned. `cutoff = +inf` computes the plain DTW (O(m) memory).
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `x` or `y` is empty.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`, `cutoff`
///   is NaN, or the window is invalid / unfeasible for these lengths.
///
/// # Examples
/// ```
/// use fdars_core::{dtw_distance_early_abandon, dtw_distance_windowed, DtwWindow};
/// let x = [0.0, 1.0, 2.0, 1.0, 0.0];
/// let y = [0.0, 0.0, 1.0, 2.0, 1.0];
/// let w = DtwWindow::SakoeChiba { radius: 1 };
/// let d = dtw_distance_windowed(&x, &y, 2.0, w).unwrap();
/// assert_eq!(dtw_distance_early_abandon(&x, &y, 2.0, w, d).unwrap(), Some(d));
/// assert_eq!(dtw_distance_early_abandon(&x, &y, 2.0, w, d * 0.5).unwrap(), None);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_distance_early_abandon(
    x: &[f64],
    y: &[f64],
    p: f64,
    window: DtwWindow,
    cutoff: f64,
) -> Result<Option<f64>, FdarError> {
    validate_dtw_inputs(x, y, p)?;
    if cutoff.is_nan() {
        return Err(FdarError::InvalidParameter {
            parameter: "cutoff",
            message: "must not be NaN".to_string(),
        });
    }
    let rows = window.row_bounds(x.len(), y.len())?;
    Ok(ea_pruned_dtw(x, y, &rows, LocalCost::new(p), cutoff, None))
}

// ---------------------------------------------------------------------------
// Exact 1-NN search
// ---------------------------------------------------------------------------

/// Pruning statistics of [`dtw_nearest_neighbor`], summed over all queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwSearchStats {
    /// Candidates discarded by LB_Keogh (visited in increasing LB_Keogh order).
    pub pruned_lb_keogh: usize,
    /// Candidates discarded by LB_Kim.
    pub pruned_lb_kim: usize,
    /// Candidates discarded by LB_Webb (only when `p ≥ 1`).
    pub pruned_lb_webb: usize,
    /// Candidates whose EAPrunedDTW abandoned before completion.
    pub early_abandoned: usize,
    /// Candidates whose DTW was computed to completion.
    pub dtw_completed: usize,
}

impl DtwSearchStats {
    fn add(&mut self, o: &DtwSearchStats) {
        self.pruned_lb_keogh += o.pruned_lb_keogh;
        self.pruned_lb_kim += o.pruned_lb_kim;
        self.pruned_lb_webb += o.pruned_lb_webb;
        self.early_abandoned += o.early_abandoned;
        self.dtw_completed += o.dtw_completed;
    }
}

/// Result of [`dtw_nearest_neighbor`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwNearestNeighbors {
    /// `indices[q]`: row of `references` nearest to query row `q` (lowest index
    /// on ties).
    pub indices: Vec<usize>,
    /// `distances[q]`: its windowed DTW cost (no root), bit-identical to
    /// `dtw_distance_windowed`.
    pub distances: Vec<f64>,
    /// Pruning statistics summed over all queries.
    pub stats: DtwSearchStats,
}

/// Precomputed per-reference data.
struct Reference {
    row: Vec<f64>,
    kim: KimFeatures,
    env: WebbCandidate,
}

/// Shared search context.
struct SearchCtx<'a> {
    refs: &'a [Reference],
    lat: Lattice,
    cost: LocalCost,
    use_webb: bool,
    slack: f64,
}

/// Relax a cutoff by a relative `slack` so that floating-point rounding in the
/// bounds can never prune the true nearest neighbour.
#[inline]
fn relax(v: f64, slack: f64) -> f64 {
    if v.is_finite() {
        v + v.abs() * slack
    } else {
        v
    }
}

/// Suffix sums of the per-row LB_Keogh terms: `rest[i]` bounds the cost of
/// rows `i + 1 ..`.
fn keogh_suffix(a: &[f64], env: &WebbCandidate, cost: LocalCost) -> Vec<f64> {
    let n = a.len();
    let mut rest = vec![0.0; n];
    for i in (0..n.saturating_sub(1)).rev() {
        rest[i] = rest[i + 1] + keogh_term(a[i + 1], env.lb[i + 1], env.ub[i + 1], cost);
    }
    rest
}

/// Best-so-far state of one query.
struct Best {
    dist: f64,
    idx: usize,
}

impl Best {
    fn offer(&mut self, d: f64, k: usize) {
        if d < self.dist || (d == self.dist && k < self.idx) {
            self.dist = d;
            self.idx = k;
        }
    }
}

/// Run the cheap bounds on candidate `k`; `true` when it is pruned.
fn pruned_by_bounds(
    ctx: &SearchCtx,
    q: &(KimFeatures, Option<WebbQuery>),
    k: usize,
    thr: f64,
    stats: &mut DtwSearchStats,
) -> bool {
    let r = &ctx.refs[k];
    if q.0.bound(&r.kim, ctx.cost) > thr {
        stats.pruned_lb_kim += 1;
        return true;
    }
    if let Some(wq) = &q.1 {
        if webb_bound(wq, &r.row, &r.env, &ctx.lat, ctx.cost, thr) > thr {
            stats.pruned_lb_webb += 1;
            return true;
        }
    }
    false
}

/// Exact 1-NN of one query.
fn search_one(ctx: &SearchCtx, a: &[f64]) -> (usize, f64, DtwSearchStats) {
    let mut stats = DtwSearchStats::default();
    let q = (
        KimFeatures::new(a),
        ctx.use_webb.then(|| WebbQuery::new(a, &ctx.lat)),
    );
    let mut order: Vec<(f64, usize)> = ctx
        .refs
        .iter()
        .enumerate()
        .map(|(k, r)| (keogh_sum(a, &r.env.lb, &r.env.ub, ctx.cost), k))
        .collect();
    order.sort_by(|u, v| u.0.total_cmp(&v.0).then(u.1.cmp(&v.1)));

    let mut best = Best {
        dist: f64::INFINITY,
        idx: usize::MAX,
    };
    for (pos, &(keogh, k)) in order.iter().enumerate() {
        let thr = relax(best.dist, ctx.slack);
        if keogh > thr {
            stats.pruned_lb_keogh += order.len() - pos;
            break;
        }
        if pruned_by_bounds(ctx, &q, k, thr, &mut stats) {
            continue;
        }
        let r = &ctx.refs[k];
        let rest = keogh_suffix(a, &r.env, ctx.cost);
        match ea_pruned_dtw(
            a,
            &r.row,
            &ctx.lat.rows,
            ctx.cost,
            best.dist,
            Some((&rest, thr)),
        ) {
            Some(d) => {
                stats.dtw_completed += 1;
                best.offer(d, k);
            }
            None => stats.early_abandoned += 1,
        }
    }
    (best.idx, best.dist, stats)
}

fn validate_search_matrix(data: &FdMatrix, parameter: &'static str) -> Result<(), FdarError> {
    if data.nrows() == 0 || data.ncols() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "at least one non-empty series".to_string(),
            actual: format!("{}x{}", data.nrows(), data.ncols()),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "all values must be finite".to_string(),
        });
    }
    Ok(())
}

/// Exact DTW 1-nearest-neighbour search with cascading lower bounds.
///
/// For every row of `queries`, finds the row of `references` minimising
/// [`dtw_distance_windowed`](super::dtw::dtw_distance_windowed)`(query, reference, p, window)`
/// and returns exactly the brute-force answer — the same index (lowest index
/// on ties) and a bit-identical distance (cost `Σ |x_i − y_j|^p`, no root).
///
/// Per query, candidates are visited in increasing LB_Keogh order (envelopes
/// of the references precomputed once with Lemire's streaming min/max) and
/// discarded as soon as a bound exceeds the best distance so far: LB_Keogh
/// (which also ends the scan, as later candidates have larger LB_Keogh), then
/// LB_Kim, then LB_Webb (when `p ≥ 1`), and finally EAPrunedDTW with the
/// best-so-far as cutoff, tightened by the LB_Keogh cost of the remaining rows
/// (UCR-suite cumulative bound). Bound comparisons carry a tiny relative slack
/// (`max(1e-10, 16 (n + m) ε)`) so floating-point rounding can never discard
/// the true neighbour. Queries run in parallel with the `parallel` feature.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if either matrix has no rows or columns.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`, a value is
///   not finite, or the window is invalid / unfeasible for the lengths
///   `(queries.ncols(), references.ncols())`.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_distance_windowed, dtw_nearest_neighbor, DtwWindow};
/// let (n_ref, m) = (5, 20);
/// let curve = |shift: f64| -> Vec<f64> {
///     (0..m).map(|k| (k as f64 * 0.3 + shift).sin()).collect()
/// };
/// let mut data = vec![0.0; n_ref * m];
/// for s in 0..n_ref {
///     for (k, v) in curve(s as f64).into_iter().enumerate() {
///         data[s + k * n_ref] = v; // column-major
///     }
/// }
/// let references = FdMatrix::from_column_major(data, n_ref, m).unwrap();
/// let q = curve(2.1);
/// let queries = FdMatrix::from_column_major(q.clone(), 1, m).unwrap();
/// let w = DtwWindow::SakoeChiba { radius: 2 };
/// let nn = dtw_nearest_neighbor(&queries, &references, 2.0, w).unwrap();
/// assert_eq!(nn.indices, vec![2]);
/// assert_eq!(nn.distances[0], dtw_distance_windowed(&q, &curve(2.0), 2.0, w).unwrap());
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_nearest_neighbor(
    queries: &FdMatrix,
    references: &FdMatrix,
    p: f64,
    window: DtwWindow,
) -> Result<DtwNearestNeighbors, FdarError> {
    validate_search_matrix(queries, "queries")?;
    validate_search_matrix(references, "references")?;
    validate_p(p)?;
    let (n, m) = (queries.ncols(), references.ncols());
    let lat = Lattice::new(n, m, window)?;
    let refs: Vec<Reference> = iter_maybe_parallel!(0..references.nrows())
        .map(|k| {
            let row = references.row(k);
            Reference {
                kim: KimFeatures::new(&row),
                env: WebbCandidate::new(&row, &lat),
                row,
            }
        })
        .collect();
    let ctx = SearchCtx {
        refs: &refs,
        lat,
        cost: LocalCost::new(p),
        use_webb: p >= 1.0,
        slack: (16.0 * (n + m) as f64 * f64::EPSILON).max(1e-10),
    };
    let results: Vec<(usize, f64, DtwSearchStats)> = iter_maybe_parallel!(0..queries.nrows())
        .map(|qi| search_one(&ctx, &queries.row(qi)))
        .collect();
    let mut stats = DtwSearchStats::default();
    let mut indices = Vec::with_capacity(results.len());
    let mut distances = Vec::with_capacity(results.len());
    for (idx, d, s) in &results {
        indices.push(*idx);
        distances.push(*d);
        stats.add(s);
    }
    Ok(DtwNearestNeighbors {
        indices,
        distances,
        stats,
    })
}

#[cfg(test)]
mod tests;
