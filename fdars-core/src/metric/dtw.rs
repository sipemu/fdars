//! Dynamic Time Warping (DTW) distance.

use crate::dim::Dim;
use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::window::DtwWindow;
use super::{cross_distance_matrix, self_distance_matrix};

/// Run the two-row DTW dynamic programming loop with a given cost function.
#[inline]
fn dtw_dp_loop(x: &[f64], y: &[f64], w: usize, cost_fn: impl Fn(f64, f64) -> f64) -> f64 {
    let n = x.len();
    let m = y.len();
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;

    // Precompute band bounds as usize to avoid isize casts in inner loop
    let bounds: Vec<(usize, usize)> = (1..=n)
        .map(|i| {
            let r_i = i + 1;
            let j_start = (r_i as isize - w as isize).max(1) as usize;
            let j_end = (r_i + w - 1).min(m);
            (j_start, j_end)
        })
        .collect();

    for (i, &(j_start, j_end)) in bounds.iter().enumerate() {
        curr.fill(f64::INFINITY);
        for j in j_start..=j_end {
            let cost = cost_fn(x[i], y[j - 1]);
            curr[j] = cost + prev[j].min(curr[j - 1]).min(prev[j - 1]);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

/// Compute DTW distance between two time series using two-row DP.
///
/// Uses O(m) memory instead of O(nm) by keeping only two rows of the DP table.
pub fn dtw_distance(x: &[f64], y: &[f64], p: f64, w: usize) -> f64 {
    if (p - 2.0).abs() < 1e-14 {
        dtw_dp_loop(x, y, w, |a, b| {
            let d = a - b;
            d * d
        })
    } else if (p - 1.0).abs() < 1e-14 {
        dtw_dp_loop(x, y, w, |a, b| (a - b).abs())
    } else {
        dtw_dp_loop(x, y, w, |a, b| (a - b).abs().powf(p))
    }
}

/// Compute DTW distance matrix for self-distances (symmetric).
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::dtw_self;
/// use fdars_core::dim::Dim;
///
/// let data = FdMatrix::from_column_major(
///     (0..40).map(|i| (i as f64 * 0.1).sin()).collect(),
///     4, 10,
/// ).unwrap();
/// let dist = dtw_self(&data, 2.0, 10, Dim::One);
/// assert_eq!(dist.shape(), (4, 4));
/// assert!((dist[(0, 0)]).abs() < 1e-10);
/// assert!((dist[(0, 1)] - dist[(1, 0)]).abs() < 1e-10);
/// ```
pub(crate) fn dtw_self_1d(data: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n = data.nrows();
    if n == 0 || data.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    self_distance_matrix(n, |i, j| dtw_distance(&rows[i], &rows[j], p, w))
}

/// Compute DTW cross-distance matrix.
pub(crate) fn dtw_cross_1d(data1: &FdMatrix, data2: &FdMatrix, p: f64, w: usize) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    if n1 == 0 || n2 == 0 || data1.ncols() == 0 || data2.ncols() == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let rows1: Vec<Vec<f64>> = (0..n1).map(|i| data1.row(i)).collect();
    let rows2: Vec<Vec<f64>> = (0..n2).map(|i| data2.row(i)).collect();
    cross_distance_matrix(n1, n2, |i, j| dtw_distance(&rows1[i], &rows2[j], p, w))
}

/// DTW self-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to `dtw_self_1d`; `dim` makes intent explicit.
pub fn dtw_self(data: &FdMatrix, p: f64, w: usize, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => dtw_self_1d(data, p, w),
    }
}

/// DTW cross-distance matrix via a unified [`Dim`] dispatch.
///
/// Both [`Dim`] arms forward to `dtw_cross_1d`; `dim` makes intent explicit.
pub fn dtw_cross(data1: &FdMatrix, data2: &FdMatrix, p: f64, w: usize, dim: Dim) -> FdMatrix {
    match dim {
        Dim::One | Dim::Two => dtw_cross_1d(data1, data2, p, w),
    }
}

// ---------------------------------------------------------------------------
// Warping path + windowed DTW (tslearn-compatible windows)
// ---------------------------------------------------------------------------

/// Optimal hard-DTW alignment returned by [`dtw_path`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DtwPath {
    /// Aligned index pairs `(i, j)` (0-based), monotone and contiguous, from
    /// `(0, 0)` to `(n − 1, m − 1)`.
    pub path: Vec<(usize, usize)>,
    /// Accumulated cost `Σ |x_i − y_j|^p` over the path — the same convention as
    /// [`dtw_distance`] (no root is taken). tslearn's `dtw_path` reports
    /// `sqrt(cost)` for `p = 2`.
    pub cost: f64,
}

/// Local cost `|a − b|^p`, with the same three-way dispatch as [`dtw_distance`]
/// so per-cell costs are bit-identical.
#[derive(Clone, Copy)]
pub(super) enum LocalCost {
    Squared,
    Absolute,
    Power(f64),
}

impl LocalCost {
    pub(super) fn new(p: f64) -> Self {
        if (p - 2.0).abs() < 1e-14 {
            LocalCost::Squared
        } else if (p - 1.0).abs() < 1e-14 {
            LocalCost::Absolute
        } else {
            LocalCost::Power(p)
        }
    }

    #[inline]
    pub(super) fn eval(self, a: f64, b: f64) -> f64 {
        match self {
            LocalCost::Squared => {
                let d = a - b;
                d * d
            }
            LocalCost::Absolute => (a - b).abs(),
            LocalCost::Power(p) => (a - b).abs().powf(p),
        }
    }
}

/// Validate the shared inputs of the windowed DTW entry points.
pub(super) fn validate_dtw_inputs(x: &[f64], y: &[f64], p: f64) -> Result<(), FdarError> {
    if x.is_empty() || y.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "x/y",
            expected: "non-empty series".to_string(),
            actual: format!("lengths {} and {}", x.len(), y.len()),
        });
    }
    if !(p.is_finite() && p > 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "p",
            message: format!("must be finite and > 0, got {p}"),
        });
    }
    Ok(())
}

/// Fill one DP row over the admissible 0-based columns `[lo, hi)`.
///
/// `prev` / `curr` are 1-based rows of length `m + 1`; cells outside the range
/// must already hold `+inf`. Same recurrence and operand order as `dtw_dp_loop`.
#[inline]
fn dtw_fill_row(
    prev: &[f64],
    curr: &mut [f64],
    xi: f64,
    y: &[f64],
    bounds: (usize, usize),
    cost: LocalCost,
) {
    fill_row_with(prev, curr, bounds, |j| cost.eval(xi, y[j]));
}

/// [`dtw_fill_row`] with an arbitrary per-column cost `cell(j)` (0-based `j`).
#[inline]
fn fill_row_with(
    prev: &[f64],
    curr: &mut [f64],
    (lo, hi): (usize, usize),
    cell: impl Fn(usize) -> f64,
) {
    for j in (lo + 1)..=hi {
        let c = cell(j - 1);
        curr[j] = c + prev[j].min(curr[j - 1]).min(prev[j - 1]);
    }
}

fn no_path_error(operation: &'static str) -> FdarError {
    FdarError::ComputationFailed {
        operation,
        detail: "no admissible warping path under the window".to_string(),
    }
}

/// Trace the optimal path back through the full accumulated table (tslearn rule:
/// first strict minimum over diagonal, vertical, horizontal).
fn dtw_traceback(acc: &[Vec<f64>], n: usize, m: usize) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (n - 1, m - 1);
    let mut path = vec![(i, j)];
    while i > 0 || j > 0 {
        if i == 0 {
            j -= 1;
        } else if j == 0 {
            i -= 1;
        } else {
            // 0-based cell (a, b) lives at acc[a + 1][b + 1].
            let diag = acc[i][j];
            let vert = acc[i][j + 1];
            let horiz = acc[i + 1][j];
            if diag <= vert && diag <= horiz {
                i -= 1;
                j -= 1;
            } else if vert <= horiz {
                i -= 1;
            } else {
                j -= 1;
            }
        }
        path.push((i, j));
    }
    path.reverse();
    path
}

/// Optimal hard-DTW warping path between two series, with an optional window.
///
/// Fills the full accumulated-cost table (O(n·m) memory, required for the
/// traceback) with the same recurrence as [`dtw_distance`], then traces back
/// from `(n − 1, m − 1)` with tslearn's `dtw_path` tie-breaking (diagonal, then
/// vertical, then horizontal). With [`DtwWindow::None`] the returned `cost`
/// equals `dtw_distance(x, y, p, max(n, m))` bit-for-bit.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `x` or `y` is empty.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`, or the
///   window is invalid / unfeasible for these lengths.
/// - [`FdarError::ComputationFailed`] if no admissible path reaches the end cell.
///
/// # Examples
/// ```
/// use fdars_core::{dtw_path, DtwWindow};
/// let res = dtw_path(&[1.0, 2.0, 3.0], &[1.0, 2.0, 2.0, 3.0], 2.0, DtwWindow::None).unwrap();
/// assert_eq!(res.path, vec![(0, 0), (1, 1), (1, 2), (2, 3)]);
/// assert_eq!(res.cost, 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_path(x: &[f64], y: &[f64], p: f64, window: DtwWindow) -> Result<DtwPath, FdarError> {
    validate_dtw_inputs(x, y, p)?;
    let n = x.len();
    let m = y.len();
    let bounds = window.row_bounds(n, m)?;
    let cost = LocalCost::new(p);

    let mut acc = vec![vec![f64::INFINITY; m + 1]; n + 1];
    acc[0][0] = 0.0;
    for i in 1..=n {
        let (before, after) = acc.split_at_mut(i);
        dtw_fill_row(
            &before[i - 1],
            &mut after[0],
            x[i - 1],
            y,
            bounds[i - 1],
            cost,
        );
    }
    let total = acc[n][m];
    if !total.is_finite() {
        return Err(no_path_error("dtw_path"));
    }
    Ok(DtwPath {
        path: dtw_traceback(&acc, n, m),
        cost: total,
    })
}

/// DTW distance under a tslearn-compatible global window, O(m) memory.
///
/// Returns the accumulated cost `Σ |x_i − y_j|^p` of the optimal admissible
/// path (no root), like [`dtw_distance`]. [`DtwWindow::None`] delegates to
/// `dtw_distance(x, y, p, max(n, m))`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `x` or `y` is empty.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`, or the
///   window is invalid / unfeasible for these lengths.
/// - [`FdarError::ComputationFailed`] if no admissible path reaches the end cell.
///
/// # Examples
/// ```
/// use fdars_core::{dtw_distance_windowed, DtwWindow};
/// let x = [0.0, 1.0, 2.0, 1.0];
/// let y = [0.0, 0.0, 1.0, 2.0];
/// let free = dtw_distance_windowed(&x, &y, 2.0, DtwWindow::None).unwrap();
/// let banded = dtw_distance_windowed(&x, &y, 2.0, DtwWindow::SakoeChiba { radius: 0 }).unwrap();
/// assert!(banded >= free);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_distance_windowed(
    x: &[f64],
    y: &[f64],
    p: f64,
    window: DtwWindow,
) -> Result<f64, FdarError> {
    validate_dtw_inputs(x, y, p)?;
    let n = x.len();
    let m = y.len();
    if window == DtwWindow::None {
        return Ok(dtw_distance(x, y, p, n.max(m)));
    }
    let bounds = window.row_bounds(n, m)?;
    let cost = LocalCost::new(p);

    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for (i, &b) in bounds.iter().enumerate() {
        curr.fill(f64::INFINITY);
        dtw_fill_row(&prev, &mut curr, x[i], y, b, cost);
        std::mem::swap(&mut prev, &mut curr);
    }
    let total = prev[m];
    if total.is_finite() {
        Ok(total)
    } else {
        Err(no_path_error("dtw_distance_windowed"))
    }
}

// ---------------------------------------------------------------------------
// Multivariate (multichannel) DTW
// ---------------------------------------------------------------------------

/// Channel-aware local cost `‖a − b‖₂^p` between two time points of a
/// channel-interleaved series. A single channel uses [`LocalCost`] directly,
/// so `d = 1` is bit-identical to the univariate DTW; for `p = 2` and `d > 1`
/// it is the squared Euclidean distance accumulated from `0.0` in channel order
/// (aeon / tslearn's dependent multichannel cost).
#[derive(Clone, Copy)]
pub(crate) struct ChannelCost(LocalCost);

impl ChannelCost {
    pub(crate) fn new(p: f64) -> Self {
        ChannelCost(LocalCost::new(p))
    }

    #[inline]
    pub(crate) fn eval(self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() == 1 {
            return self.0.eval(a[0], b[0]);
        }
        let mut s = 0.0;
        for (u, w) in a.iter().zip(b) {
            let t = u - w;
            s += t * t;
        }
        match self.0 {
            LocalCost::Squared => s,
            LocalCost::Absolute => s.sqrt(),
            LocalCost::Power(p) => s.powf(p / 2.0),
        }
    }
}

/// Validate a `d × m` multivariate series pair and return `d`.
pub(crate) fn validate_mv_pair(x: &FdMatrix, y: &FdMatrix, p: f64) -> Result<usize, FdarError> {
    let d = x.nrows();
    if d == 0 || y.nrows() != d || x.ncols() == 0 || y.ncols() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "x/y",
            expected: "two non-empty d x m series with the same d >= 1 channels (rows)".to_string(),
            actual: format!(
                "{}x{} and {}x{}",
                x.nrows(),
                x.ncols(),
                y.nrows(),
                y.ncols()
            ),
        });
    }
    validate_dtw_inputs(&[0.0], &[0.0], p)?;
    Ok(d)
}

/// Common channel count of the (non-empty) series of a collection pair.
pub(crate) fn validate_mv_collection(
    parameter: &'static str,
    data: &[FdMatrix],
    d: Option<usize>,
) -> Result<Option<usize>, FdarError> {
    let mut d = d;
    for x in data {
        if x.nrows() == 0 || x.ncols() == 0 || d.is_some_and(|e| e != x.nrows()) {
            return Err(FdarError::InvalidDimension {
                parameter,
                expected: format!(
                    "non-empty d x m series with {} channels (rows)",
                    d.map_or("the same d >= 1".to_string(), |e| e.to_string())
                ),
                actual: format!("{}x{}", x.nrows(), x.ncols()),
            });
        }
        d = Some(x.nrows());
    }
    Ok(d)
}

/// Windowed DTW cost of two channel-interleaved series with `d` channels
/// (`x.len() = n·d`). `d = 1` delegates to [`dtw_distance_windowed`].
pub(crate) fn dtw_distance_windowed_flat(
    x: &[f64],
    y: &[f64],
    d: usize,
    p: f64,
    window: DtwWindow,
) -> Result<f64, FdarError> {
    if d == 1 {
        return dtw_distance_windowed(x, y, p, window);
    }
    validate_dtw_inputs(x, y, p)?;
    let (n, m) = (x.len() / d, y.len() / d);
    let bounds = window.row_bounds(n, m)?;
    let cost = ChannelCost::new(p);
    let mut prev = vec![f64::INFINITY; m + 1];
    let mut curr = vec![f64::INFINITY; m + 1];
    prev[0] = 0.0;
    for (i, &b) in bounds.iter().enumerate() {
        curr.fill(f64::INFINITY);
        let xi = &x[i * d..(i + 1) * d];
        fill_row_with(&prev, &mut curr, b, |j| {
            cost.eval(xi, &y[j * d..(j + 1) * d])
        });
        std::mem::swap(&mut prev, &mut curr);
    }
    let total = prev[m];
    if total.is_finite() {
        Ok(total)
    } else {
        Err(no_path_error("dtw_distance_multivariate"))
    }
}

/// Optimal warping path of two channel-interleaved series with `d` channels.
/// `d = 1` delegates to [`dtw_path`].
pub(crate) fn dtw_path_flat(
    x: &[f64],
    y: &[f64],
    d: usize,
    p: f64,
    window: DtwWindow,
) -> Result<DtwPath, FdarError> {
    if d == 1 {
        return dtw_path(x, y, p, window);
    }
    validate_dtw_inputs(x, y, p)?;
    let (n, m) = (x.len() / d, y.len() / d);
    let bounds = window.row_bounds(n, m)?;
    let cost = ChannelCost::new(p);
    let mut acc = vec![vec![f64::INFINITY; m + 1]; n + 1];
    acc[0][0] = 0.0;
    for i in 1..=n {
        let (before, after) = acc.split_at_mut(i);
        let xi = &x[(i - 1) * d..i * d];
        fill_row_with(&before[i - 1], &mut after[0], bounds[i - 1], |j| {
            cost.eval(xi, &y[j * d..(j + 1) * d])
        });
    }
    let total = acc[n][m];
    if !total.is_finite() {
        return Err(no_path_error("dtw_path_multivariate"));
    }
    Ok(DtwPath {
        path: dtw_traceback(&acc, n, m),
        cost: total,
    })
}

/// DTW between two multivariate series (dependent DTW, aeon / tslearn).
///
/// `x` is `d × n` and `y` is `d × m`: one channel per row, one time point per
/// column (aeon's `(n_channels, n_timepoints)` layout; tslearn's
/// `(sz, d)` transposed). The local cost of aligning time points `i` and `j`
/// is `‖x_{·,i} − y_{·,j}‖₂^p`; for `p = 2` this is the squared Euclidean
/// distance across channels, so the result equals aeon's
/// `dtw_distance(x, y)` and the square of tslearn's `dtw(x.T, y.T)`. The
/// returned cost is accumulated without a root, like [`dtw_distance_windowed`].
/// With `d = 1` the result is bit-identical to
/// `dtw_distance_windowed(x.row(0), y.row(0), p, window)`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if a series is empty or the channel
///   counts differ.
/// - [`FdarError::InvalidParameter`] if `p` is not finite and `> 0`, or the
///   window is invalid / unfeasible for these lengths.
/// - [`FdarError::ComputationFailed`] if no admissible path reaches the end cell.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_distance_multivariate, DtwWindow};
/// // Two channels: x = [[0, 1, 2], [0, 0, 1]], y = [[0, 2], [0, 1]].
/// let x = FdMatrix::from_column_major(vec![0.0, 0.0, 1.0, 0.0, 2.0, 1.0], 2, 3).unwrap();
/// let y = FdMatrix::from_column_major(vec![0.0, 0.0, 2.0, 1.0], 2, 2).unwrap();
/// let d = dtw_distance_multivariate(&x, &y, 2.0, DtwWindow::None).unwrap();
/// // Optimal path (0,0), (1,0) or (1,1), (2,1): best cost 1 + 0 = 1.
/// assert_eq!(d, 1.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_distance_multivariate(
    x: &FdMatrix,
    y: &FdMatrix,
    p: f64,
    window: DtwWindow,
) -> Result<f64, FdarError> {
    let d = validate_mv_pair(x, y, p)?;
    dtw_distance_windowed_flat(x.as_slice(), y.as_slice(), d, p, window)
}

/// Optimal DTW warping path between two multivariate series.
///
/// Same local cost and conventions as [`dtw_distance_multivariate`] and the
/// same traceback tie-breaking as [`dtw_path`] (tslearn's `dtw_path`); the
/// `cost` equals [`dtw_distance_multivariate`] bit-for-bit. With `d = 1` the
/// result is identical to [`dtw_path`] on the single channel.
///
/// # Errors
/// As for [`dtw_distance_multivariate`].
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_path_multivariate, DtwWindow};
/// let x = FdMatrix::from_column_major(vec![1.0, 0.0, 2.0, 1.0, 3.0, 1.0], 2, 3).unwrap();
/// let y = FdMatrix::from_column_major(vec![1.0, 0.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0], 2, 4)
///     .unwrap();
/// let res = dtw_path_multivariate(&x, &y, 2.0, DtwWindow::None).unwrap();
/// assert_eq!(res.path, vec![(0, 0), (1, 1), (1, 2), (2, 3)]);
/// assert_eq!(res.cost, 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_path_multivariate(
    x: &FdMatrix,
    y: &FdMatrix,
    p: f64,
    window: DtwWindow,
) -> Result<DtwPath, FdarError> {
    let d = validate_mv_pair(x, y, p)?;
    dtw_path_flat(x.as_slice(), y.as_slice(), d, p, window)
}

/// DTW distance matrix of collections of multivariate series.
///
/// Each element of `data` (and `other`) is one `d × m` series (channels as
/// rows); all series share `d`, lengths may differ (the window is sized per
/// pair). `other = None` returns the symmetric self-distance matrix (zero
/// diagonal), otherwise the `data.len() × other.len()` cross-distance matrix,
/// with entries [`dtw_distance_multivariate`] (aeon `dtw_pairwise_distance`).
/// Pairs run in parallel with the `parallel` feature.
///
/// # Errors
/// As for [`dtw_distance_multivariate`], for any pair.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_distance_multivariate, dtw_pairwise_distance_multivariate, DtwWindow};
/// let s: Vec<FdMatrix> = (0..3)
///     .map(|k| FdMatrix::from_column_major(
///         (0..10).map(|t| ((t * 3 + k * 5) % 7) as f64).collect(), 2, 5,
///     ).unwrap())
///     .collect();
/// let d = dtw_pairwise_distance_multivariate(&s, None, 2.0, DtwWindow::None).unwrap();
/// assert_eq!(d[(0, 0)], 0.0);
/// assert_eq!(d[(0, 2)], dtw_distance_multivariate(&s[0], &s[2], 2.0, DtwWindow::None).unwrap());
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_pairwise_distance_multivariate(
    data: &[FdMatrix],
    other: Option<&[FdMatrix]>,
    p: f64,
    window: DtwWindow,
) -> Result<FdMatrix, FdarError> {
    let d = validate_mv_collection("data", data, None)?;
    let d = match other {
        Some(o) => validate_mv_collection("other", o, d)?,
        None => d,
    }
    .unwrap_or(1);
    validate_dtw_inputs(&[0.0], &[0.0], p)?;
    let second = other.unwrap_or(data);
    super::pairwise_result(data.len(), other.map(<[FdMatrix]>::len), |i, j| {
        dtw_distance_windowed_flat(data[i].as_slice(), second[j].as_slice(), d, p, window)
    })
}

#[cfg(test)]
mod path_tests {
    use super::*;

    fn series(n: usize, phase: f64, freq: f64) -> Vec<f64> {
        (0..n)
            .map(|k| {
                let t = k as f64 / n as f64;
                (freq * t * 6.0 + phase).sin() + 0.3 * (2.0 * freq * t + phase).cos()
            })
            .collect()
    }

    fn check_path(x: &[f64], y: &[f64], p: f64, window: DtwWindow) -> DtwPath {
        let (n, m) = (x.len(), y.len());
        let res = dtw_path(x, y, p, window).unwrap();
        let mask = window.mask(n, m).unwrap();
        assert_eq!(res.path.first(), Some(&(0, 0)));
        assert_eq!(res.path.last(), Some(&(n - 1, m - 1)));
        for w in res.path.windows(2) {
            let step = (w[1].0 - w[0].0, w[1].1 - w[0].1);
            assert!(
                matches!(step, (1, 0) | (0, 1) | (1, 1)),
                "bad step {step:?}"
            );
        }
        let cost = LocalCost::new(p);
        let mut s = 0.0;
        for &(i, j) in &res.path {
            assert!(mask[i][j], "inadmissible cell ({i},{j})");
            s += cost.eval(x[i], y[j]);
        }
        assert_eq!(s.to_bits(), res.cost.to_bits());
        let wd = dtw_distance_windowed(x, y, p, window).unwrap();
        assert_eq!(wd.to_bits(), res.cost.to_bits());
        let free = dtw_distance(x, y, p, n.max(m));
        assert!(res.cost >= free);
        if window == DtwWindow::None {
            assert_eq!(res.cost.to_bits(), free.to_bits());
        }
        res
    }

    #[test]
    fn tslearn_docstring_path() {
        let res = dtw_path(&[1., 2., 3.], &[1., 2., 2., 3.], 2.0, DtwWindow::None).unwrap();
        assert_eq!(res.path, vec![(0, 0), (1, 1), (1, 2), (2, 3)]);
        assert_eq!(res.cost, 0.0);
    }

    #[test]
    fn diagonal_first_tie_break() {
        let res = dtw_path(&[0., 1., 2.], &[0., 2.], 2.0, DtwWindow::None).unwrap();
        assert_eq!(res.path, vec![(0, 0), (1, 0), (2, 1)]);
        assert_eq!(res.cost, 1.0);
    }

    #[test]
    fn unique_optimum_path() {
        let res = dtw_path(&[1., 2., 2., 3.], &[1., 2., 3., 4.], 2.0, DtwWindow::None).unwrap();
        assert_eq!(res.path, vec![(0, 0), (1, 1), (2, 1), (3, 2), (3, 3)]);
        assert_eq!(res.cost, 1.0);
    }

    #[test]
    fn property_sweep() {
        for &(n, m) in &[(8usize, 8usize), (7, 11), (12, 5)] {
            let x = series(n, 0.3, 1.0);
            let y = series(m, 1.1, 1.7);
            for &p in &[1.0, 2.0, 1.5] {
                let none = check_path(&x, &y, p, DtwWindow::None);
                let mut windows = vec![
                    DtwWindow::SakoeChiba { radius: 1 },
                    DtwWindow::SakoeChiba { radius: 3 },
                ];
                if n == m {
                    windows.push(DtwWindow::SakoeChiba { radius: 0 });
                }
                if (DtwWindow::Itakura { max_slope: 2.0 }).mask(n, m).is_ok() {
                    windows.push(DtwWindow::Itakura { max_slope: 2.0 });
                }
                for w in windows {
                    let res = check_path(&x, &y, p, w);
                    if w == (DtwWindow::SakoeChiba { radius: 0 }) {
                        let diag: Vec<(usize, usize)> = (0..n).map(|k| (k, k)).collect();
                        assert_eq!(res.path, diag);
                    }
                }
                for radius in [n.max(m), usize::MAX] {
                    let full = check_path(&x, &y, p, DtwWindow::SakoeChiba { radius });
                    assert_eq!(full.path, none.path);
                    assert_eq!(full.cost.to_bits(), none.cost.to_bits());
                }
            }
        }
    }

    #[test]
    fn path_errors() {
        let w = DtwWindow::None;
        assert!(matches!(
            dtw_path(&[], &[1.0], 2.0, w),
            Err(FdarError::InvalidDimension { .. })
        ));
        assert!(matches!(
            dtw_path(&[1.0], &[], 2.0, w),
            Err(FdarError::InvalidDimension { .. })
        ));
        for p in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                dtw_path(&[1.0], &[1.0], p, w),
                Err(FdarError::InvalidParameter { .. })
            ));
            assert!(matches!(
                dtw_distance_windowed(&[1.0], &[1.0], p, w),
                Err(FdarError::InvalidParameter { .. })
            ));
        }
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        for bad in [
            DtwWindow::Itakura { max_slope: 0.5 },
            DtwWindow::Itakura {
                max_slope: f64::NAN,
            },
            DtwWindow::Itakura { max_slope: 1.0 },
        ] {
            assert!(matches!(
                dtw_path(&x, &y, 2.0, bad),
                Err(FdarError::InvalidParameter { .. })
            ));
            assert!(matches!(
                dtw_distance_windowed(&x, &y, 2.0, bad),
                Err(FdarError::InvalidParameter { .. })
            ));
        }
    }
}
