//! Shape DTW (Zhao & Itti 2018), aeon `shape_dtw_*`.
//!
//! Every time point is described by the *shape* of its neighbourhood: the
//! series is edge-padded by `reach` samples on both sides, the length-
//! `2·reach + 1` subsequence centred on each point is mapped through a shape
//! descriptor ([`ShapeDtwDescriptor`]) per channel, and DTW (squared Euclidean
//! cost) aligns the descriptor sequences. The reported distance is then the
//! accumulated squared cost of the **raw** samples along aeon's traceback of
//! that alignment, as in aeon's `_get_shape_dtw_distance_from_cost_mat`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::metric::window::DtwWindow;

use super::ddtw::dtw_lattice;
use super::{
    collection_channels, min_return_path, mv_pair, sqe, ElasticDistancePath, Lattice, Ser,
};
use crate::metric::{cross_distance_matrix, self_distance_matrix};

/// Shape descriptor applied to every length-`2·reach + 1` subsequence (per
/// channel) by the `shape_dtw_*` functions.
///
/// [`Identity`](Self::Identity) is aeon's only (and default) descriptor; the
/// others are the descriptors of sktime's `ShapeDTW` classifier (Zhao & Itti
/// 2018), ported from sktime 1.3.0 and applied to aeon's subsequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum ShapeDtwDescriptor {
    /// The raw subsequence (aeon `descriptor="identity"`, sktime `"raw"`).
    #[default]
    Identity,
    /// Piecewise aggregate approximation with `num_intervals` (fractional)
    /// frames (sktime `PAAlegacy`; sktime default `8`). Needs
    /// `1 ≤ num_intervals ≤ 2·reach + 1`.
    Paa {
        /// Number of PAA frames.
        num_intervals: usize,
    },
    /// Haar discrete wavelet transform with `num_levels` levels (sktime
    /// `DWTTransformer`; sktime default `3`); `0` levels is the identity.
    Dwt {
        /// Number of decomposition levels.
        num_levels: usize,
    },
    /// Total-least-squares slope of each of `num_intervals` segments (sktime
    /// `SlopeTransformer`; sktime default `8`). Needs
    /// `1 ≤ num_intervals ≤ 2·reach + 1`.
    Slope {
        /// Number of segments.
        num_intervals: usize,
    },
    /// First derivative by central differences, one-sided at the ends
    /// (`numpy.gradient`, sktime `DerivativeSlopeTransformer`). Needs
    /// `reach ≥ 1`.
    Derivative,
}

impl ShapeDtwDescriptor {
    fn validate(self, reach: usize) -> Result<(), FdarError> {
        let len = reach.checked_mul(2).and_then(|v| v.checked_add(1)).ok_or(
            FdarError::InvalidParameter {
                parameter: "reach",
                message: "too large".to_string(),
            },
        )?;
        let bad = |message: String| {
            Err(FdarError::InvalidParameter {
                parameter: "descriptor",
                message,
            })
        };
        match self {
            Self::Paa { num_intervals: k } | Self::Slope { num_intervals: k }
                if k == 0 || k > len =>
            {
                bad(format!(
                    "num_intervals must be in 1..={len} (the subsequence length 2·reach + 1), got {k}"
                ))
            }
            Self::Derivative if reach == 0 => {
                bad("the derivative descriptor needs reach >= 1".to_string())
            }
            _ => Ok(()),
        }
    }

    /// Append the descriptor of subsequence `s` to `out`.
    fn describe(self, s: &[f64], out: &mut Vec<f64>) {
        match self {
            Self::Identity => out.extend_from_slice(s),
            Self::Paa { num_intervals } => paa(s, num_intervals, out),
            Self::Dwt { num_levels } => dwt(s, num_levels, out),
            Self::Slope { num_intervals } => slopes(s, num_intervals, out),
            Self::Derivative => gradient(s, out),
        }
    }
}

/// sktime `PAAlegacy._perform_paa_along_dim` for one series (fractional frames).
fn paa(s: &[f64], k: usize, out: &mut Vec<f64>) {
    let frame_length = s.len() as f64 / k as f64;
    let (mut current_frame, mut size, mut sum) = (0usize, 0.0f64, 0.0f64);
    for &v in s {
        let remaining = frame_length - size;
        if remaining > 1.0 {
            sum += v;
            size += 1.0;
        } else {
            sum += remaining * v;
            size += remaining;
        }
        if size == frame_length {
            out.push(sum / frame_length);
            current_frame += 1;
            sum = (1.0 - remaining) * v;
            size = 1.0 - remaining;
        }
    }
    if current_frame + 1 == k {
        out.push(sum / frame_length);
    }
}

/// sktime `DWTTransformer._extract_wavelet_coefficients` (Haar) for one series.
fn dwt(s: &[f64], levels: usize, out: &mut Vec<f64>) {
    if levels == 0 {
        out.extend_from_slice(s);
        return;
    }
    let half = |a: &[f64], sign: f64| -> Vec<f64> {
        if a.len() == 1 {
            return vec![a[0]];
        }
        (0..a.len() / 2)
            .map(|x| (a[2 * x] + sign * a[2 * x + 1]) / std::f64::consts::SQRT_2)
            .collect()
    };
    let mut coeffs: Vec<f64> = Vec::new();
    let mut current = s.to_vec();
    for _ in 0..levels {
        let approx = half(&current, 1.0);
        let mut wav = half(&current, -1.0);
        wav.reverse();
        coeffs.extend(wav);
        current = approx;
    }
    current.reverse();
    coeffs.extend(current);
    coeffs.reverse();
    out.extend(coeffs);
}

/// sktime `SlopeTransformer._get_gradients_of_lines` for one series.
fn slopes(s: &[f64], k: usize, out: &mut Vec<f64>) {
    let len = s.len();
    let avg = len as f64 / k as f64;
    let mut beginning = 0.0f64;
    while beginning < len as f64 {
        let (a, b) = (beginning as usize, ((beginning + avg) as usize).min(len));
        out.push(tls_gradient(&s[a..b]));
        beginning += avg;
    }
}

/// sktime `SlopeTransformer._get_gradient`: total-least-squares slope of `y`
/// against `x = 1, …, len`.
fn tls_gradient(y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let mean_x = (n + 1.0) / 2.0;
    let mean_y = y.iter().sum::<f64>() / n;
    let (mut syy, mut sxx, mut sxy) = (0.0, 0.0, 0.0);
    for (i, &v) in y.iter().enumerate() {
        let dx = (i + 1) as f64 - mean_x;
        let dy = v - mean_y;
        syy += dy * dy;
        sxx += dx * dx;
        sxy += dx * dy;
    }
    let w = syy - sxx;
    let r = 2.0 * sxy;
    if r == 0.0 {
        0.0
    } else {
        (w + (w * w + r * r).sqrt()) / r
    }
}

/// `numpy.gradient` with unit spacing (second-order interior, first-order ends).
fn gradient(s: &[f64], out: &mut Vec<f64>) {
    let n = s.len();
    out.push(s[1] - s[0]);
    for i in 1..n - 1 {
        out.push((s[i + 1] - s[i - 1]) / 2.0);
    }
    out.push(s[n - 1] - s[n - 2]);
}

/// A series prepared for shape DTW: raw channel-interleaved samples and the
/// descriptor sequence (`n` time points × `dim` features, features of channel
/// 0 first as in aeon's `_transform_subsequences`).
struct Shaped {
    raw: Vec<f64>,
    d: usize,
    feat: Vec<f64>,
    dim: usize,
}

impl Shaped {
    fn new(x: Ser, descriptor: ShapeDtwDescriptor, reach: usize) -> Result<Self, FdarError> {
        let (n, d) = (x.len(), x.d);
        if n == 0 {
            return Err(FdarError::InvalidDimension {
                parameter: "x/y",
                expected: "non-empty series".to_string(),
                actual: "length 0".to_string(),
            });
        }
        // aeon `_pad_ts_edges`: repeat the first / last sample `reach` times.
        let len = 2 * reach + 1;
        let mut feat = Vec::new();
        let mut sub = vec![0.0; len];
        for i in 0..n {
            for c in 0..d {
                for (k, v) in sub.iter_mut().enumerate() {
                    let t = (i + k).saturating_sub(reach).min(n - 1);
                    *v = x.v[t * d + c];
                }
                descriptor.describe(&sub, &mut feat);
            }
        }
        let dim = feat.len() / n;
        Ok(Shaped {
            raw: x.v.to_vec(),
            d,
            feat,
            dim,
        })
    }

    fn len(&self) -> usize {
        self.raw.len() / self.d
    }

    fn features(&self) -> Ser<'_> {
        Ser {
            v: &self.feat,
            d: self.dim,
        }
    }

    fn raw(&self) -> Ser<'_> {
        Ser {
            v: &self.raw,
            d: self.d,
        }
    }
}

/// aeon `_get_shape_dtw_distance_from_cost_mat`: walk back from the end cell
/// with strict comparisons (diagonal, then left, else up), summing the raw
/// squared costs. Indices `−1` wrap to the last row / column exactly like
/// numpy's negative indexing in aeon.
fn raw_traceback_cost(x: Ser, y: Ser, lat: &Lattice) -> f64 {
    let (n, m) = (x.len() as isize, y.len() as isize);
    let at = |i: isize, j: isize| lat.cost(i.rem_euclid(n) as usize, j.rem_euclid(m) as usize);
    let (mut i, mut j) = (n - 1, m - 1);
    let mut dist = 0.0;
    while i >= 0 && j >= 0 {
        dist += sqe(x.at(i as usize), y.at(j as usize));
        let a = at(i - 1, j - 1);
        let b = at(i, j - 1);
        let c = at(i - 1, j);
        if a < b && a < c {
            i -= 1;
            j -= 1;
        } else if b < c {
            j -= 1;
        } else {
            i -= 1;
        }
    }
    dist
}

fn solve(a: &Shaped, b: &Shaped, window: DtwWindow) -> Result<Lattice, FdarError> {
    let bounds = window.row_bounds(a.len(), b.len())?;
    Ok(dtw_lattice(a.features(), b.features(), &bounds))
}

fn shaped_pair(
    x: Ser,
    y: Ser,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<(Shaped, Shaped), FdarError> {
    descriptor.validate(reach)?;
    Ok((
        Shaped::new(x, descriptor, reach)?,
        Shaped::new(y, descriptor, reach)?,
    ))
}

fn distance_impl(
    x: Ser,
    y: Ser,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<f64, FdarError> {
    let (a, b) = shaped_pair(x, y, descriptor, reach)?;
    let lat = solve(&a, &b, window)?;
    Ok(raw_traceback_cost(a.raw(), b.raw(), &lat))
}

fn cost_matrix_impl(
    x: Ser,
    y: Ser,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<FdMatrix, FdarError> {
    let (a, b) = shaped_pair(x, y, descriptor, reach)?;
    let lat = solve(&a, &b, window)?;
    let (n, m) = (a.len(), b.len());
    let mut out = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            out[(i, j)] = lat.cost(i, j);
        }
    }
    Ok(out)
}

fn path_impl(
    x: Ser,
    y: Ser,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<ElasticDistancePath, FdarError> {
    let (a, b) = shaped_pair(x, y, descriptor, reach)?;
    let lat = solve(&a, &b, window)?;
    Ok(ElasticDistancePath {
        path: min_return_path(&lat, a.len(), b.len()),
        distance: raw_traceback_cost(a.raw(), b.raw(), &lat),
    })
}

fn pair_cost(a: &Shaped, b: &Shaped, window: DtwWindow) -> Result<f64, FdarError> {
    Ok(raw_traceback_cost(a.raw(), b.raw(), &solve(a, b, window)?))
}

/// Shape-DTW distance between two univariate series (aeon `shape_dtw_distance`).
///
/// Each point is replaced by the `descriptor` of its edge-padded neighbourhood
/// of length `2·reach + 1` (aeon defaults: [`ShapeDtwDescriptor::Identity`],
/// `reach = 15`), the descriptor sequences are aligned by DTW (squared
/// Euclidean cost, optional `window`), and the distance is the accumulated
/// squared cost `Σ (x_i − y_j)²` of the **raw** series along aeon's traceback
/// of that alignment (no root). `reach = 0` with the identity descriptor is
/// plain DTW's alignment. The `window` is sized for the original lengths, as
/// in aeon.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if a series is empty.
/// - [`FdarError::InvalidParameter`] for an invalid descriptor parameter (see
///   [`ShapeDtwDescriptor`]) or an invalid / unfeasible window.
///
/// # Examples
/// ```
/// use fdars_core::metric::{shape_dtw_distance, DtwWindow, ShapeDtwDescriptor};
/// let x = [0.0, 1.0, 2.0, 1.5, 0.5, 0.0, -0.5, 0.0];
/// let y = [0.0, 0.5, 1.5, 2.0, 1.0, 0.0, -0.5, -0.2];
/// let d = shape_dtw_distance(&x, &y, DtwWindow::None, ShapeDtwDescriptor::Identity, 2)
///     .unwrap();
/// assert!(d >= 0.0);
/// assert_eq!(
///     shape_dtw_distance(&x, &x, DtwWindow::None, ShapeDtwDescriptor::Derivative, 3).unwrap(),
///     0.0
/// );
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_distance(
    x: &[f64],
    y: &[f64],
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<f64, FdarError> {
    distance_impl(Ser::uni(x), Ser::uni(y), window, descriptor, reach)
}

/// Shape-DTW accumulated cost matrix of the descriptor alignment (aeon
/// `shape_dtw_cost_matrix`).
///
/// The `n × m` DTW cost matrix of the descriptor sequences (see
/// [`shape_dtw_distance`]); out-of-window cells hold `inf`.
///
/// # Errors
/// As for [`shape_dtw_distance`].
///
/// # Examples
/// ```
/// use fdars_core::metric::{shape_dtw_cost_matrix, DtwWindow, ShapeDtwDescriptor};
/// let x = [0.0, 1.0, 2.0, 1.5, 0.5];
/// let y = [0.0, 0.5, 1.5, 2.0];
/// let cm = shape_dtw_cost_matrix(&x, &y, DtwWindow::None, ShapeDtwDescriptor::Identity, 1)
///     .unwrap();
/// assert_eq!(cm.shape(), (5, 4));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_cost_matrix(
    x: &[f64],
    y: &[f64],
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<FdMatrix, FdarError> {
    cost_matrix_impl(Ser::uni(x), Ser::uni(y), window, descriptor, reach)
}

/// Shape-DTW alignment path and distance (aeon `shape_dtw_alignment_path`).
///
/// The path is aeon's `compute_min_return_path` of the descriptor cost matrix;
/// the distance is [`shape_dtw_distance`].
///
/// # Errors
/// As for [`shape_dtw_distance`].
///
/// # Examples
/// ```
/// use fdars_core::metric::{
///     shape_dtw_alignment_path, shape_dtw_distance, DtwWindow, ShapeDtwDescriptor,
/// };
/// let x = [0.0, 1.0, 2.0, 1.5, 0.5, 0.0];
/// let y = [0.0, 0.5, 1.5, 2.0, 1.0, 0.0];
/// let desc = ShapeDtwDescriptor::Paa { num_intervals: 2 };
/// let res = shape_dtw_alignment_path(&x, &y, DtwWindow::None, desc, 2).unwrap();
/// assert_eq!(res.path.first(), Some(&(0, 0)));
/// assert_eq!(res.distance, shape_dtw_distance(&x, &y, DtwWindow::None, desc, 2).unwrap());
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_alignment_path(
    x: &[f64],
    y: &[f64],
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<ElasticDistancePath, FdarError> {
    path_impl(Ser::uni(x), Ser::uni(y), window, descriptor, reach)
}

/// Shape-DTW distance matrix of a collection (aeon `shape_dtw_pairwise_distance`).
///
/// Rows of `data` (and `other`) are the series; `other = None` gives the
/// symmetric self-distance matrix (zero diagonal), otherwise the
/// cross-distance matrix. Descriptors are computed once per series; pairs run
/// in parallel with the `parallel` feature.
///
/// # Errors
/// As for [`shape_dtw_distance`].
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::{shape_dtw_pairwise_distance, DtwWindow, ShapeDtwDescriptor};
/// let data = FdMatrix::from_column_major(
///     (0..24).map(|k| ((k * 7 % 11) as f64 * 0.3).sin()).collect(), 4, 6,
/// ).unwrap();
/// let d = shape_dtw_pairwise_distance(&data, None, DtwWindow::None, ShapeDtwDescriptor::Identity, 2)
///     .unwrap();
/// assert_eq!(d.shape(), (4, 4));
/// assert_eq!(d[(2, 2)], 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_pairwise_distance(
    data: &FdMatrix,
    other: Option<&FdMatrix>,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<FdMatrix, FdarError> {
    descriptor.validate(reach)?;
    let prep = |m: &FdMatrix| -> Result<Vec<Shaped>, FdarError> {
        (0..m.nrows())
            .map(|i| Shaped::new(Ser::uni(&m.row(i)), descriptor, reach))
            .collect()
    };
    let rows1 = prep(data)?;
    let n2 = other.map_or(rows1.len(), FdMatrix::nrows);
    if rows1.is_empty() || n2 == 0 {
        return Ok(FdMatrix::zeros(rows1.len(), n2));
    }
    match other {
        None => {
            window.row_bounds(rows1[0].len(), rows1[0].len())?;
            Ok(self_distance_matrix(rows1.len(), |i, j| {
                pair_cost(&rows1[i], &rows1[j], window).unwrap_or(f64::NAN)
            }))
        }
        Some(o) => {
            let rows2 = prep(o)?;
            window.row_bounds(rows1[0].len(), rows2[0].len())?;
            Ok(cross_distance_matrix(rows1.len(), rows2.len(), |i, j| {
                pair_cost(&rows1[i], &rows2[j], window).unwrap_or(f64::NAN)
            }))
        }
    }
}

/// Shape-DTW distance between two multivariate series (aeon
/// `shape_dtw_distance` on `(n_channels, n_timepoints)` arrays).
///
/// `x` is `d × n`, `y` is `d × m` (channels as rows). Each channel's
/// subsequence is described separately and the descriptors of all channels
/// are concatenated per time point (aeon `_transform_subsequences`); the DTW
/// and the raw traceback cost use squared Euclidean distances across all
/// features / channels. With `d = 1` the result is bit-identical to
/// [`shape_dtw_distance`].
///
/// # Errors
/// As for [`shape_dtw_distance`]; additionally
/// [`FdarError::InvalidDimension`] if the channel counts differ.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::{shape_dtw_distance_multivariate, DtwWindow, ShapeDtwDescriptor};
/// let x = FdMatrix::from_column_major(
///     vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0, 1.5, -0.5, 0.5, 0.0, 0.0, 0.5], 2, 6,
/// ).unwrap();
/// let d = shape_dtw_distance_multivariate(&x, &x, DtwWindow::None, ShapeDtwDescriptor::Identity, 2)
///     .unwrap();
/// assert_eq!(d, 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_distance_multivariate(
    x: &FdMatrix,
    y: &FdMatrix,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<f64, FdarError> {
    let (xs, ys) = mv_pair(x, y)?;
    distance_impl(xs, ys, window, descriptor, reach)
}

/// Shape-DTW descriptor cost matrix of two multivariate series (aeon
/// `shape_dtw_cost_matrix`); see [`shape_dtw_distance_multivariate`].
///
/// # Errors
/// As for [`shape_dtw_distance_multivariate`].
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::{shape_dtw_cost_matrix_multivariate, DtwWindow, ShapeDtwDescriptor};
/// let x = FdMatrix::from_column_major(vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0], 2, 3).unwrap();
/// let y = FdMatrix::from_column_major(vec![0.0, 0.5, 1.5, 0.0], 2, 2).unwrap();
/// let cm = shape_dtw_cost_matrix_multivariate(&x, &y, DtwWindow::None, ShapeDtwDescriptor::Identity, 1)
///     .unwrap();
/// assert_eq!(cm.shape(), (3, 2));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_cost_matrix_multivariate(
    x: &FdMatrix,
    y: &FdMatrix,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<FdMatrix, FdarError> {
    let (xs, ys) = mv_pair(x, y)?;
    cost_matrix_impl(xs, ys, window, descriptor, reach)
}

/// Shape-DTW alignment path and distance of two multivariate series (aeon
/// `shape_dtw_alignment_path`); see [`shape_dtw_distance_multivariate`].
///
/// # Errors
/// As for [`shape_dtw_distance_multivariate`].
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::{shape_dtw_alignment_path_multivariate, DtwWindow, ShapeDtwDescriptor};
/// let x = FdMatrix::from_column_major(vec![0.0, 1.0, 1.0, 0.5, 2.0, 0.0], 2, 3).unwrap();
/// let res = shape_dtw_alignment_path_multivariate(&x, &x, DtwWindow::None, ShapeDtwDescriptor::Identity, 1)
///     .unwrap();
/// assert_eq!(res.path, vec![(0, 0), (1, 1), (2, 2)]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_alignment_path_multivariate(
    x: &FdMatrix,
    y: &FdMatrix,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<ElasticDistancePath, FdarError> {
    let (xs, ys) = mv_pair(x, y)?;
    path_impl(xs, ys, window, descriptor, reach)
}

/// Shape-DTW distance matrix of collections of multivariate series (aeon
/// `shape_dtw_pairwise_distance`).
///
/// Each element is one `d × m` series (shared `d`, lengths may differ; the
/// window is sized per pair). `other = None` gives the symmetric
/// self-distance matrix, otherwise the cross-distance matrix.
///
/// # Errors
/// As for [`shape_dtw_distance_multivariate`].
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::{shape_dtw_pairwise_distance_multivariate, DtwWindow, ShapeDtwDescriptor};
/// let s: Vec<FdMatrix> = (0..3)
///     .map(|k| FdMatrix::from_column_major(
///         (0..12).map(|t| ((t * 3 + k * 5) % 7) as f64).collect(), 2, 6,
///     ).unwrap())
///     .collect();
/// let d = shape_dtw_pairwise_distance_multivariate(&s, None, DtwWindow::None, ShapeDtwDescriptor::Identity, 2)
///     .unwrap();
/// assert_eq!(d.shape(), (3, 3));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_dtw_pairwise_distance_multivariate(
    data: &[FdMatrix],
    other: Option<&[FdMatrix]>,
    window: DtwWindow,
    descriptor: ShapeDtwDescriptor,
    reach: usize,
) -> Result<FdMatrix, FdarError> {
    collection_channels(data, other)?;
    descriptor.validate(reach)?;
    let prep = |xs: &[FdMatrix]| -> Result<Vec<Shaped>, FdarError> {
        xs.iter()
            .map(|x| Shaped::new(Ser::of(x), descriptor, reach))
            .collect()
    };
    let rows1 = prep(data)?;
    let rows2 = match other {
        Some(o) => Some(prep(o)?),
        None => None,
    };
    let second = rows2.as_ref().unwrap_or(&rows1);
    crate::metric::pairwise_result(rows1.len(), rows2.as_ref().map(Vec::len), |i, j| {
        pair_cost(&rows1[i], &second[j], window)
    })
}

#[cfg(test)]
#[path = "shape_dtw_goldens.rs"]
mod goldens;

#[cfg(test)]
mod tests {
    use super::goldens::*;
    use super::*;

    pub(super) struct ShapeGolden {
        pub(super) kind: &'static str,
        pub(super) k: usize,
        pub(super) reach: usize,
        pub(super) window: Option<f64>,
        pub(super) mv: bool,
        pub(super) distance: f64,
        pub(super) shape: (usize, usize),
        pub(super) cost: &'static [f64],
        pub(super) path: &'static [(usize, usize)],
        pub(super) path_distance: f64,
    }

    fn close(a: f64, b: f64, tol: f64) -> bool {
        if a.is_infinite() || b.is_infinite() {
            return a == b;
        }
        (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
    }

    fn descriptor(kind: &str, k: usize) -> ShapeDtwDescriptor {
        match kind {
            "identity" => ShapeDtwDescriptor::Identity,
            "paa" => ShapeDtwDescriptor::Paa { num_intervals: k },
            "dwt" => ShapeDtwDescriptor::Dwt { num_levels: k },
            "slope" => ShapeDtwDescriptor::Slope { num_intervals: k },
            "derivative" => ShapeDtwDescriptor::Derivative,
            _ => unreachable!(),
        }
    }

    fn mat(flat: &[f64], d: usize) -> FdMatrix {
        FdMatrix::from_column_major(flat.to_vec(), d, flat.len() / d).unwrap()
    }

    #[test]
    fn sktime_descriptor_goldens() {
        let s = [0.3, -1.2, 0.8, 2.5, 1.1, -0.4, 0.0, 1.7, -2.2];
        let cases: [(&str, usize, &[f64]); 10] = [
            ("paa", 3, &DESC_PAA_3),
            ("paa", 4, &DESC_PAA_4),
            ("paa", 2, &DESC_PAA_2),
            ("dwt", 1, &DESC_DWT_1),
            ("dwt", 2, &DESC_DWT_2),
            ("dwt", 5, &DESC_DWT_5),
            ("slope", 3, &DESC_SLOPE_3),
            ("slope", 2, &DESC_SLOPE_2),
            ("slope", 4, &DESC_SLOPE_4),
            ("derivative", 0, &DESC_DERIVATIVE_0),
        ];
        for (kind, k, want) in cases {
            let mut out = Vec::new();
            descriptor(kind, k).describe(&s, &mut out);
            assert_eq!(out.len(), want.len(), "{kind} {k}");
            for (a, b) in out.iter().zip(want) {
                assert!(close(*a, *b, 1e-12), "{kind} {k}: {a} vs {b}");
            }
        }
        let mut out = Vec::new();
        ShapeDtwDescriptor::Dwt { num_levels: 0 }.describe(&s, &mut out);
        assert_eq!(out, s);
    }

    #[test]
    fn aeon_shape_dtw_goldens() {
        for g in SHAPE_GOLDENS {
            let what = format!(
                "{} k={} reach={} w={:?} mv={}",
                g.kind, g.k, g.reach, g.window, g.mv
            );
            let desc = descriptor(g.kind, g.k);
            let w = g
                .window
                .map_or(DtwWindow::None, |window| DtwWindow::SakoeChibaFraction {
                    window,
                });
            let (d, cm, p) = if g.mv {
                let (x, y) = (mat(&SMX, 2), mat(&SMY, 2));
                (
                    shape_dtw_distance_multivariate(&x, &y, w, desc, g.reach).unwrap(),
                    shape_dtw_cost_matrix_multivariate(&x, &y, w, desc, g.reach).unwrap(),
                    shape_dtw_alignment_path_multivariate(&x, &y, w, desc, g.reach).unwrap(),
                )
            } else {
                (
                    shape_dtw_distance(&SX, &SY, w, desc, g.reach).unwrap(),
                    shape_dtw_cost_matrix(&SX, &SY, w, desc, g.reach).unwrap(),
                    shape_dtw_alignment_path(&SX, &SY, w, desc, g.reach).unwrap(),
                )
            };
            assert!(close(d, g.distance, 1e-10), "{what}: {d} vs {}", g.distance);
            assert_eq!(cm.shape(), g.shape, "{what}");
            let c = g.shape.1;
            for i in 0..g.shape.0 {
                for j in 0..c {
                    let (a, b) = (cm[(i, j)], g.cost[i * c + j]);
                    assert!(close(a, b, 1e-10), "{what} cost[{i},{j}]: {a} vs {b}");
                }
            }
            assert_eq!(p.path, g.path, "{what} path");
            assert!(close(p.distance, g.path_distance, 1e-10), "{what}");
        }
    }

    #[test]
    fn aeon_shape_dtw_pairwise_goldens() {
        let rows = [&SC0[..], &SC1, &SC2];
        let m = SC0.len();
        let mut flat = vec![0.0; 3 * m];
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                flat[i + j * 3] = v;
            }
        }
        let data = FdMatrix::from_column_major(flat, 3, m).unwrap();
        let mut other = FdMatrix::zeros(2, m);
        for j in 0..m {
            other[(0, j)] = SC0[j] + 0.3;
            other[(1, j)] = SC2[j] * 0.5;
        }
        let id = ShapeDtwDescriptor::Identity;
        let pw = shape_dtw_pairwise_distance(&data, None, DtwWindow::None, id, 4).unwrap();
        let pc = shape_dtw_pairwise_distance(&data, Some(&other), DtwWindow::None, id, 4).unwrap();
        for (got, want, c) in [(&pw, &SPW[..], 3), (&pc, &SPC[..], 2)] {
            for i in 0..3 {
                for j in 0..c {
                    assert!(close(got[(i, j)], want[i * c + j], 1e-10), "[{i},{j}]");
                }
            }
        }
        // Multivariate collection of 1 × m series agrees bit-for-bit.
        let coll: Vec<FdMatrix> = rows.iter().map(|r| mat(r, 1)).collect();
        let pm =
            shape_dtw_pairwise_distance_multivariate(&coll, None, DtwWindow::None, id, 4).unwrap();
        assert_eq!(pm, pw);
    }

    #[test]
    fn single_channel_bit_identity_and_invariants() {
        let descs = [
            ShapeDtwDescriptor::Identity,
            ShapeDtwDescriptor::Paa { num_intervals: 3 },
            ShapeDtwDescriptor::Dwt { num_levels: 2 },
            ShapeDtwDescriptor::Slope { num_intervals: 2 },
            ShapeDtwDescriptor::Derivative,
        ];
        let (xm, ym) = (mat(&SX, 1), mat(&SY, 1));
        for desc in descs {
            for w in [DtwWindow::None, DtwWindow::SakoeChiba { radius: 4 }] {
                let u = shape_dtw_distance(&SX, &SY, w, desc, 3).unwrap();
                let v = shape_dtw_distance_multivariate(&xm, &ym, w, desc, 3).unwrap();
                assert_eq!(u.to_bits(), v.to_bits());
                let pu = shape_dtw_alignment_path(&SX, &SY, w, desc, 3).unwrap();
                let pv = shape_dtw_alignment_path_multivariate(&xm, &ym, w, desc, 3).unwrap();
                assert_eq!(pu, pv);
                // Identity of indiscernibles.
                assert_eq!(shape_dtw_distance(&SX, &SX, w, desc, 3).unwrap(), 0.0);
            }
        }
        // reach = 0 with the identity descriptor aligns exactly like plain DTW
        // and the distance is the DTW cost of that alignment.
        let id = ShapeDtwDescriptor::Identity;
        let p = shape_dtw_alignment_path(&SX, &SY, DtwWindow::None, id, 0).unwrap();
        let dtw = crate::metric::dtw::dtw_path(&SX, &SY, 2.0, DtwWindow::None).unwrap();
        assert_eq!(p.path, dtw.path);
        assert!(close(p.distance, dtw.cost, 1e-12));
    }

    #[test]
    fn shape_dtw_errors() {
        let id = ShapeDtwDescriptor::Identity;
        assert!(matches!(
            shape_dtw_distance(&[], &[1.0], DtwWindow::None, id, 2),
            Err(FdarError::InvalidDimension { .. })
        ));
        for bad in [
            ShapeDtwDescriptor::Paa { num_intervals: 0 },
            ShapeDtwDescriptor::Paa { num_intervals: 6 },
            ShapeDtwDescriptor::Slope { num_intervals: 6 },
        ] {
            assert!(matches!(
                shape_dtw_distance(&SX, &SY, DtwWindow::None, bad, 2),
                Err(FdarError::InvalidParameter { .. })
            ));
        }
        assert!(matches!(
            shape_dtw_distance(&SX, &SY, DtwWindow::None, ShapeDtwDescriptor::Derivative, 0),
            Err(FdarError::InvalidParameter { .. })
        ));
        assert!(matches!(
            shape_dtw_distance(&SX, &SY, DtwWindow::Itakura { max_slope: 0.5 }, id, 2),
            Err(FdarError::InvalidParameter { .. })
        ));
        let (x, y) = (mat(&SMX, 2), mat(&SX, 1));
        assert!(matches!(
            shape_dtw_distance_multivariate(&x, &y, DtwWindow::None, id, 2),
            Err(FdarError::InvalidDimension { .. })
        ));
    }
}
