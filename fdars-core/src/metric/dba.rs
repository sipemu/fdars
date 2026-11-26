//! DTW Barycenter Averaging (DBA).
//!
//! DBA (Petitjean, Ketterlin & Gançarski, *A global averaging method for dynamic
//! time warping, with applications to clustering*, Pattern Recognition 44(3),
//! 2011) averages a set of series under hard DTW. Each iteration
//!
//! 1. aligns the current barycenter `b` to every series `x_k` with an optimal
//!    DTW warping path ([`dtw_path`], squared local cost, optional
//!    [`DtwWindow`]), and
//! 2. replaces every barycenter sample by the weighted mean of all series
//!    samples aligned to it:
//!
//! ```text
//! b_i ← Σ_k w_k Σ_{j : (i, j) ∈ path_k} x_k[j]  /  Σ_k w_k · |{j : (i, j) ∈ path_k}|
//! ```
//!
//! This is a majorize–minimize step on the weighted DTW inertia
//! `Σ_k w_k DTW²(b, x_k) / Σ_k w_k` (Schultz & Jain 2018), so the cost never
//! increases. The loop follows tslearn's `dtw_barycenter_averaging` exactly
//! (same stopping rule, same path tie-breaking), so results agree with tslearn
//! to rounding.
//!
//! Unlike [`soft_dtw_barycenter`](super::soft_dtw::soft_dtw_barycenter), which
//! minimises a smoothed objective by gradient descent, DBA works on hard DTW and
//! is the standard averaging baseline in tslearn, dtaidistance and
//! DynamicAxisWarping.jl. Use it as the centroid step of a DTW k-means (see
//! [`dtw_kmeans`](crate::dtw_kmeans::dtw_kmeans)).

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::dtw::{dtw_distance_windowed_flat, dtw_path_flat, validate_mv_collection};
use super::window::DtwWindow;

/// DBA uses the squared local cost `(a − b)²`.
pub(crate) const DBA_P: f64 = 2.0;

/// Initial barycenter for [`dtw_barycenter_averaging`].
#[derive(Debug, Clone, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum DbaInit {
    /// Pointwise (unweighted) mean of the series, linearly interpolated to the
    /// barycenter length — tslearn's default. Sample weights are **not** used
    /// here, matching tslearn's `_init_avg`.
    #[default]
    Mean,
    /// The medoid: the input series minimising `Σ_k w_k · DTW²(x_i, x_k)` under
    /// the configured window (lowest index on ties), linearly interpolated to
    /// the barycenter length. This is the initialisation Petitjean et al.
    /// recommend. It costs `n(n − 1)/2` extra DTW evaluations.
    Medoid,
    /// A user-supplied starting barycenter (finite, non-empty). Its length sets
    /// the barycenter length (tslearn's `init_barycenter`). In a k-means loop
    /// this is the previous centroid. Univariate only.
    Custom(Vec<f64>),
    /// A user-supplied `d × size` starting barycenter for
    /// [`dtw_barycenter_averaging_multivariate`] (channels as rows; finite,
    /// non-empty). [`dtw_barycenter_averaging`] accepts it only with one row.
    CustomMultivariate(FdMatrix),
}

/// Configuration for [`dtw_barycenter_averaging`].
///
/// Defaults match tslearn's `dtw_barycenter_averaging`: `max_iter = 30`,
/// `tol = 1e-5`, no window, mean initialisation, barycenter length equal to the
/// series length, uniform weights.
///
/// External construction is `default()` plus field assignment:
///
/// ```
/// use fdars_core::{DbaConfig, DbaInit, DtwWindow};
/// let mut cfg = DbaConfig::default();
/// cfg.init = DbaInit::Medoid;
/// cfg.window = DtwWindow::SakoeChiba { radius: 3 };
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbaConfig {
    /// Maximum number of DBA iterations (default `30`). `0` returns the initial
    /// barycenter unchanged.
    pub max_iter: usize,
    /// Stop when the absolute change of the DBA cost between two iterations is
    /// below `tol` (default `1e-5`). Must be finite and `≥ 0`.
    pub tol: f64,
    /// Global warping window applied to every barycenter–series alignment
    /// (default [`DtwWindow::None`]).
    pub window: DtwWindow,
    /// Initial barycenter (default [`DbaInit::Mean`]).
    pub init: DbaInit,
    /// Barycenter length; `None` (default) uses the series length, or the
    /// length of a [`DbaInit::Custom`] barycenter. Must agree with a custom
    /// initial barycenter when both are given.
    pub barycenter_size: Option<usize>,
    /// Non-negative per-series weights (length `n`, positive sum); `None`
    /// (default) means uniform weights.
    pub weights: Option<Vec<f64>>,
}

impl Default for DbaConfig {
    fn default() -> Self {
        Self {
            max_iter: 30,
            tol: 1e-5,
            window: DtwWindow::None,
            init: DbaInit::Mean,
            barycenter_size: None,
            weights: None,
        }
    }
}

/// Result of [`dtw_barycenter_averaging`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbaResult {
    /// The DBA barycenter.
    pub barycenter: Vec<f64>,
    /// Weighted DTW inertia of the returned barycenter,
    /// `Σ_k w_k · DTW²(barycenter, x_k) / Σ_k w_k`, where `DTW²` is the
    /// accumulated squared cost (the value of [`dtw_distance_windowed`] with
    /// `p = 2`, no root). tslearn's internal loss is the same quantity
    /// evaluated *before* the last update.
    pub cost: f64,
    /// Number of DBA iterations performed.
    pub n_iter: usize,
    /// Whether the cost change dropped below `tol` before `max_iter` (`false`
    /// when the iteration cap was hit or the loop stopped on a floating-point
    /// cost increase).
    pub converged: bool,
}

/// DTW Barycenter Averaging of the rows of `data` (Petitjean et al. 2011).
///
/// Returns a barycenter that (locally) minimises the weighted DTW inertia
/// `Σ_k w_k · DTW²(b, x_k) / Σ_k w_k`, using squared local costs and the
/// optional [`DbaConfig::window`]. The iteration, stopping rule and path
/// tie-breaking follow tslearn's `dtw_barycenter_averaging`.
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty, a weight vector has
///   the wrong length, or `barycenter_size` disagrees with a custom init.
/// - [`FdarError::InvalidParameter`] for non-finite data, a negative /
///   non-finite `tol`, invalid weights (negative, non-finite or zero sum), a
///   zero `barycenter_size`, an empty or non-finite custom init, or an invalid
///   or unfeasible window.
/// - [`FdarError::ComputationFailed`] if no admissible warping path exists.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_barycenter_averaging, DbaConfig};
///
/// // tslearn docstring example: [[1, 2, 3, 4], [1, 2, 4, 5]] (column-major).
/// let data = FdMatrix::from_column_major(vec![1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0], 2, 4)
///     .unwrap();
/// let mut cfg = DbaConfig::default();
/// cfg.max_iter = 5;
/// let res = dtw_barycenter_averaging(&data, &cfg).unwrap();
/// assert_eq!(res.barycenter, vec![1.0, 2.0, 3.5, 4.5]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_barycenter_averaging(
    data: &FdMatrix,
    config: &DbaConfig,
) -> Result<DbaResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least one series with at least one point".to_string(),
            actual: format!("{n}x{m}"),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(non_finite_data());
    }
    let weights = resolve_weights(config.weights.as_deref(), n)?;
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    let init = initial_barycenter(&refs, 1, &weights, config)?;
    dba_from_init(
        &refs,
        1,
        &weights,
        init,
        config.max_iter,
        config.tol,
        config.window,
    )
}

/// Result of [`dtw_barycenter_averaging_multivariate`].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DbaMultivariateResult {
    /// The DBA barycenter as a `d × size` matrix (channels as rows).
    pub barycenter: FdMatrix,
    /// Weighted DTW inertia of the returned barycenter (as
    /// [`DbaResult::cost`], with the multichannel squared Euclidean cost of
    /// [`dtw_distance_multivariate`](super::dtw::dtw_distance_multivariate)).
    pub cost: f64,
    /// Number of DBA iterations performed.
    pub n_iter: usize,
    /// Whether the cost change dropped below `tol` before `max_iter`.
    pub converged: bool,
}

/// DTW Barycenter Averaging of multivariate series (tslearn
/// `dtw_barycenter_averaging` on `(n_ts, sz, d)` data).
///
/// Each element of `data` is one `d × m_k` series (channels as rows, time
/// points as columns); all series share `d`, their lengths may differ. The
/// alignments use the dependent multichannel DTW of
/// [`dtw_distance_multivariate`](super::dtw::dtw_distance_multivariate)
/// (squared Euclidean cost across channels) and each barycenter time point is
/// the weighted mean of the aligned series *vectors*, exactly as in tslearn.
/// Configuration and stopping rule are those of [`dtw_barycenter_averaging`]:
/// [`DbaInit::Mean`] averages the series pointwise over the series that reach
/// each time point (tslearn's `nanmean` of the NaN-padded dataset, length
/// `max m_k`) and resamples every channel linearly to the barycenter size;
/// [`DbaInit::CustomMultivariate`] supplies a `d × size` start (a
/// [`DbaInit::Custom`] vector is accepted when `d = 1`). With `d = 1` the
/// result is bit-identical to [`dtw_barycenter_averaging`].
///
/// # Errors
/// - [`FdarError::InvalidDimension`] if `data` is empty, a series is empty,
///   channel counts differ, a weight vector has the wrong length, or a custom
///   init has the wrong shape.
/// - [`FdarError::InvalidParameter`] as for [`dtw_barycenter_averaging`].
/// - [`FdarError::ComputationFailed`] if no admissible warping path exists.
///
/// # Examples
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::{dtw_barycenter_averaging_multivariate, DbaConfig};
/// // Two-channel series (column-major 2 × 6): a bump up in channel 0 and down in
/// // channel 1, the second series shifted by one sample.
/// let x0 = vec![0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0];
/// let x1 = vec![0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 1.0, -1.0, 0.0, 0.0];
/// let data = [
///     FdMatrix::from_column_major(x0.clone(), 2, 6).unwrap(),
///     FdMatrix::from_column_major(x1, 2, 6).unwrap(),
/// ];
/// let res = dtw_barycenter_averaging_multivariate(&data, &DbaConfig::default()).unwrap();
/// assert_eq!(res.barycenter.shape(), (2, 6));
/// // Averaging a series with itself returns it exactly.
/// let same = [data[0].clone(), data[0].clone()];
/// let fixed = dtw_barycenter_averaging_multivariate(&same, &DbaConfig::default()).unwrap();
/// assert_eq!(fixed.barycenter.as_slice(), &x0[..]);
/// assert_eq!(fixed.cost, 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn dtw_barycenter_averaging_multivariate(
    data: &[FdMatrix],
    config: &DbaConfig,
) -> Result<DbaMultivariateResult, FdarError> {
    if data.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least one series".to_string(),
            actual: "0 series".to_string(),
        });
    }
    let d = validate_mv_collection("data", data, None)?.unwrap_or(1);
    if data
        .iter()
        .any(|x| x.as_slice().iter().any(|v| !v.is_finite()))
    {
        return Err(non_finite_data());
    }
    let weights = resolve_weights(config.weights.as_deref(), data.len())?;
    let refs: Vec<&[f64]> = data.iter().map(FdMatrix::as_slice).collect();
    let init = initial_barycenter(&refs, d, &weights, config)?;
    let res = dba_from_init(
        &refs,
        d,
        &weights,
        init,
        config.max_iter,
        config.tol,
        config.window,
    )?;
    let size = res.barycenter.len() / d;
    Ok(DbaMultivariateResult {
        barycenter: FdMatrix::from_column_major(res.barycenter, d, size)?,
        cost: res.cost,
        n_iter: res.n_iter,
        converged: res.converged,
    })
}

fn non_finite_data() -> FdarError {
    FdarError::InvalidParameter {
        parameter: "data",
        message: "all values must be finite".to_string(),
    }
}

/// Validate (or default) the per-series weights.
fn resolve_weights(weights: Option<&[f64]>, n: usize) -> Result<Vec<f64>, FdarError> {
    let Some(w) = weights else {
        return Ok(vec![1.0; n]);
    };
    if w.len() != n {
        return Err(FdarError::InvalidDimension {
            parameter: "weights",
            expected: format!("{n} weights (one per series)"),
            actual: format!("{} weights", w.len()),
        });
    }
    let sum: f64 = w.iter().sum();
    if w.iter().any(|v| !(v.is_finite() && *v >= 0.0)) || !(sum > 0.0 && sum.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "weights",
            message: "weights must be finite, non-negative and have a positive sum".to_string(),
        });
    }
    Ok(w.to_vec())
}

/// Pointwise mean of channel-interleaved series over the series that reach
/// each flat position (tslearn's `nanmean` of the NaN-padded dataset). For
/// equal lengths this is the plain mean `Σ_k x_k[j] / n`.
fn nan_mean(rows: &[&[f64]]) -> Vec<f64> {
    let len = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    (0..len)
        .map(|j| {
            let reach = || rows.iter().filter(move |r| j < r.len());
            reach().map(|r| r[j]).sum::<f64>() / reach().count() as f64
        })
        .collect()
}

/// Build the starting barycenter requested by `config` (channel-interleaved,
/// `d` channels).
fn initial_barycenter(
    rows: &[&[f64]],
    d: usize,
    weights: &[f64],
    config: &DbaConfig,
) -> Result<Vec<f64>, FdarError> {
    let m = rows[0].len() / d;
    if config.barycenter_size == Some(0) {
        return Err(FdarError::InvalidParameter {
            parameter: "barycenter_size",
            message: "must be at least 1".to_string(),
        });
    }
    match &config.init {
        DbaInit::Mean => {
            let mean = nan_mean(rows);
            let size = config.barycenter_size.unwrap_or(mean.len() / d);
            Ok(interpolate_channels(&mean, d, size))
        }
        DbaInit::Medoid => {
            let idx = dtw_medoid_index_flat(rows, d, weights, config.window)?;
            let size = config.barycenter_size.unwrap_or(m);
            Ok(interpolate_channels(rows[idx], d, size))
        }
        DbaInit::Custom(b) => {
            if d != 1 {
                return Err(FdarError::InvalidParameter {
                    parameter: "init",
                    message: format!(
                        "DbaInit::Custom is univariate; use DbaInit::CustomMultivariate for {d} channels"
                    ),
                });
            }
            validate_custom_init(b, config.barycenter_size)?;
            Ok(b.clone())
        }
        DbaInit::CustomMultivariate(b) => {
            if b.nrows() != d {
                return Err(FdarError::InvalidDimension {
                    parameter: "init",
                    expected: format!("a custom barycenter with {d} channels (rows)"),
                    actual: format!("{}x{}", b.nrows(), b.ncols()),
                });
            }
            validate_custom_init(b.as_slice(), config.barycenter_size.map(|s| s * d))?;
            Ok(b.as_slice().to_vec())
        }
    }
}

fn validate_custom_init(b: &[f64], size: Option<usize>) -> Result<(), FdarError> {
    if b.is_empty() || b.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "init",
            message: "a custom initial barycenter must be non-empty and finite".to_string(),
        });
    }
    match size {
        Some(s) if s != b.len() => Err(FdarError::InvalidDimension {
            parameter: "barycenter_size",
            expected: format!("{} (the custom init length)", b.len()),
            actual: s.to_string(),
        }),
        _ => Ok(()),
    }
}

/// Linear interpolation of `a` (on an equispaced `[0, 1]` grid) onto `size`
/// equispaced points — tslearn's `_init_avg` resampling. Identity when
/// `size == a.len()`.
#[cfg(test)]
fn interpolate_linear(a: &[f64], size: usize) -> Vec<f64> {
    interpolate_channels(a, 1, size)
}

/// [`interpolate_linear`] of every channel of a channel-interleaved series.
fn interpolate_channels(a: &[f64], d: usize, size: usize) -> Vec<f64> {
    let len = a.len() / d;
    if size == len {
        return a.to_vec();
    }
    if len == 1 {
        return (0..size).flat_map(|_| a.iter().copied()).collect();
    }
    let span = (len - 1) as f64;
    let mut out = Vec::with_capacity(size * d);
    for k in 0..size {
        let q = if size == 1 {
            0.0
        } else {
            k as f64 / (size - 1) as f64
        };
        let pos = q * span;
        let lo = (pos.floor() as usize).min(len - 2);
        let frac = pos - lo as f64;
        for c in 0..d {
            let (u, w) = (a[lo * d + c], a[(lo + 1) * d + c]);
            out.push(u + (w - u) * frac);
        }
    }
    out
}

/// Index of the DTW medoid: `argmin_i Σ_k w_k · DTW²(x_i, x_k)`.
#[cfg(test)]
pub(crate) fn dtw_medoid_index(
    rows: &[&[f64]],
    weights: &[f64],
    window: DtwWindow,
) -> Result<usize, FdarError> {
    dtw_medoid_index_flat(rows, 1, weights, window)
}

/// DTW medoid of channel-interleaved series with `d` channels.
pub(crate) fn dtw_medoid_index_flat(
    rows: &[&[f64]],
    d: usize,
    weights: &[f64],
    window: DtwWindow,
) -> Result<usize, FdarError> {
    let n = rows.len();
    let upper: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| {
            ((i + 1)..n)
                .map(|k| dtw_distance_windowed_flat(rows[i], rows[k], d, DBA_P, window))
                .collect::<Result<Vec<f64>, FdarError>>()
        })
        .collect::<Result<Vec<_>, FdarError>>()?;
    let mut totals = vec![0.0; n];
    for (i, row) in upper.iter().enumerate() {
        for (off, &dist) in row.iter().enumerate() {
            let k = i + 1 + off;
            totals[i] += weights[k] * dist;
            totals[k] += weights[i] * dist;
        }
    }
    let mut best = 0;
    for (i, &t) in totals.iter().enumerate() {
        if t < totals[best] {
            best = i;
        }
    }
    Ok(best)
}

/// Run the DBA iterations from a given starting barycenter.
///
/// `rows` are the series (any lengths), channel-interleaved with `d`
/// channels, `weights` the validated per-series weights. Shared with the DTW
/// k-means centroid step.
pub(crate) fn dba_from_init(
    rows: &[&[f64]],
    d: usize,
    weights: &[f64],
    init: Vec<f64>,
    max_iter: usize,
    tol: f64,
    window: DtwWindow,
) -> Result<DbaResult, FdarError> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "tol",
            message: format!("must be finite and >= 0, got {tol}"),
        });
    }
    let mut barycenter = init;
    let mut cost_prev = f64::INFINITY;
    let mut n_iter = 0;
    let mut converged = false;
    for _ in 0..max_iter {
        let (next, cost) = dba_step(rows, d, weights, &barycenter, window)?;
        barycenter = next;
        n_iter += 1;
        if (cost_prev - cost).abs() < tol {
            converged = true;
            break;
        }
        if cost_prev < cost {
            break;
        }
        cost_prev = cost;
    }
    let cost = dba_cost(rows, d, weights, &barycenter, window)?;
    Ok(DbaResult {
        barycenter,
        cost,
        n_iter,
        converged,
    })
}

/// One DBA iteration: align, then average. Returns the updated barycenter and
/// the cost of the barycenter *before* the update (tslearn's loss).
fn dba_step(
    rows: &[&[f64]],
    d: usize,
    weights: &[f64],
    barycenter: &[f64],
    window: DtwWindow,
) -> Result<(Vec<f64>, f64), FdarError> {
    let paths = iter_maybe_parallel!(0..rows.len())
        .map(|k| dtw_path_flat(barycenter, rows[k], d, DBA_P, window))
        .collect::<Result<Vec<_>, FdarError>>()?;
    let size = barycenter.len() / d;
    let mut num = vec![0.0; size * d];
    let mut den = vec![0.0; size];
    let mut cost = 0.0;
    for (k, p) in paths.iter().enumerate() {
        let w = weights[k];
        cost += p.cost * w;
        for &(i, j) in &p.path {
            for c in 0..d {
                num[i * d + c] += w * rows[k][j * d + c];
            }
            den[i] += w;
        }
    }
    let wsum: f64 = weights.iter().sum();
    let next = num
        .iter()
        .enumerate()
        .map(|(q, s)| s / den[q / d])
        .collect();
    Ok((next, cost / wsum))
}

/// Weighted DTW inertia `Σ_k w_k DTW²(b, x_k) / Σ_k w_k` (O(m) memory).
pub(crate) fn dba_cost(
    rows: &[&[f64]],
    d: usize,
    weights: &[f64],
    barycenter: &[f64],
    window: DtwWindow,
) -> Result<f64, FdarError> {
    let costs = iter_maybe_parallel!(0..rows.len())
        .map(|k| dtw_distance_windowed_flat(barycenter, rows[k], d, DBA_P, window))
        .collect::<Result<Vec<f64>, FdarError>>()?;
    let wsum: f64 = weights.iter().sum();
    Ok(costs.iter().zip(weights).map(|(c, w)| c * w).sum::<f64>() / wsum)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values: tslearn 0.10.0 `tslearn.barycenters.dtw_barycenter_averaging`
    // (max_iter=30, tol=1e-5) and dtaidistance 2.5.1 `dtw_barycenter.dba` /
    // `dba_loop(thr=None)`, on the `d1()` dataset below (same formula in Python).
    // Costs are `sum_k w_k cdist_dtw(b, X)[k]^2 / sum w` of the returned
    // barycenter. Windowed cases pass `metric_params={"global_constraint": 2,
    // "sakoe_chiba_radius": 1}` / `{"itakura_max_slope": 2.0}` (DBA forwards
    // metric_params raw to numba, which needs the integer constraint code).
    const MEAN_NONE: [f64; 20] = [
        -0.3567551009342645,
        0.10693852719591773,
        0.5872902198854368,
        0.852280813393002,
        1.039484014061174,
        0.704001007621002,
        0.5380599609602225,
        0.06997524571833683,
        -0.4093216702934144,
        -0.7180723921557179,
        -1.0003787008521896,
        -0.8154346944372209,
        -0.47202156758511904,
        -0.097789622986396,
        0.3946127988555588,
        0.6638696328396367,
        0.9332992729716441,
        1.0571707064618427,
        0.8295010922421178,
        0.42054805977994536,
    ];
    const MEAN_NONE_COST: f64 = 0.6040364949140383;
    const MEAN_SC1: [f64; 20] = [
        -0.3567551009342645,
        0.10693852719591773,
        0.569647325115843,
        0.8910142680898133,
        1.0402541668595497,
        0.8418524922738437,
        0.5548970412402993,
        0.06997524571833683,
        -0.4093216702934144,
        -0.7180723921557179,
        -1.0003787008521896,
        -0.8154346944372209,
        -0.47202156758511904,
        -0.097789622986396,
        0.3946127988555588,
        0.7300639993665154,
        0.9688022298808261,
        1.0424393275012183,
        0.8295010922421178,
        0.42054805977994536,
    ];
    const MEAN_SC1_COST: f64 = 0.6406653429771824;
    const MEAN_IT2: [f64; 20] = [
        -0.33311454617828407,
        0.09287368253864994,
        0.5872902198854368,
        0.9065698130497253,
        1.0351507112667384,
        0.704001007621002,
        0.5380599609602225,
        0.06997524571833683,
        -0.4093216702934144,
        -0.7180723921557179,
        -1.0003787008521896,
        -0.8154346944372209,
        -0.47202156758511904,
        -0.097789622986396,
        0.3946127988555588,
        0.6638696328396367,
        0.9332992729716441,
        1.0379487180724285,
        0.7691350524219913,
        0.3780687239515906,
    ];
    const MEAN_IT2_COST: f64 = 0.793936155175988;
    const MEAN_SIZE12: [f64; 12] = [
        -0.29872633128429577,
        0.35323151181969537,
        0.9680812796980065,
        0.6828113343172177,
        0.14216174324306302,
        -0.4608312754110423,
        -0.9084227035135216,
        -0.4419017775334373,
        0.15665906917363392,
        0.6793476684630441,
        1.0030245575285484,
        0.42054805977994536,
    ];
    const MEAN_SIZE12_COST: f64 = 0.6984747415925758;
    const WEIGHTED: [f64; 20] = [
        -0.22473816162839436,
        0.24340301061169115,
        0.6401876370143221,
        0.853303190311712,
        0.9838028419119313,
        0.7258483820571799,
        0.5071092476973754,
        0.05598313889754249,
        -0.41425486927434535,
        -0.6711393702877491,
        -0.9865230358061151,
        -0.765024881236911,
        -0.4050982882641282,
        -0.06898821756723648,
        0.4434708975959505,
        0.6695151800079697,
        0.968187592826522,
        1.0918587807043993,
        0.785675954116658,
        0.3081783020152948,
    ];
    const WEIGHTED_COST: f64 = 0.4897523180509612;
    const CUSTOM_INIT_ROW3: [f64; 20] = [
        -0.29872633128429577,
        0.27330854509098057,
        0.6267703739611339,
        1.0112656750425544,
        0.704001007621002,
        0.2944180213561777,
        -0.00756361373314412,
        -0.4702773357163444,
        -0.7480762354038768,
        -0.9877828401963,
        -0.7068632030748178,
        -0.40504947951421305,
        0.07435119111056739,
        0.5446559068921866,
        0.8320709908321136,
        1.0642322309555614,
        0.9092481681515425,
        0.8874660175637865,
        0.6592635465534897,
        0.3780687239515906,
    ];
    const CUSTOM_INIT_ROW3_COST: f64 = 0.5870084440179962;
    const MEDOID_INIT: [f64; 20] = [
        -0.3567551009342645,
        0.10693852719591773,
        0.5872902198854368,
        0.852280813393002,
        1.044535764372652,
        0.9327610093614795,
        0.704001007621002,
        0.2944180213561777,
        -0.18999893768454212,
        -0.4981805173939152,
        -0.9523248962989853,
        -0.7724000961675609,
        -0.40504947951421305,
        0.07435119111056739,
        0.4627334686103437,
        0.6950700425351416,
        1.0470176396399136,
        0.8874660175637865,
        0.6592635465534897,
        0.3780687239515906,
    ];
    const MEDOID_INIT_COST: f64 = 0.6058610898700282;
    const DTAI_STEP1_FROM_ROW2: [f64; 20] = [
        -0.33311454617828407,
        -0.2774909766481863,
        -0.03665083172634446,
        0.36498638923757815,
        0.7039119935414667,
        1.0183315535533426,
        0.704001007621002,
        0.2944180213561777,
        -0.10164897759420218,
        -0.4702773357163444,
        -0.7480762354038769,
        -0.9924631462936452,
        -0.7724000961675609,
        -0.40504947951421305,
        0.07435119111056739,
        0.5716590067971153,
        0.9127281719909859,
        1.0239130084209909,
        1.007846349028321,
        0.5398734790656645,
    ];
    const DTAI_LOOP10_FROM_ROW2: [f64; 20] = [
        -0.35675510093426455,
        -0.03665083172634446,
        0.032160211274650186,
        0.43044094485623363,
        0.7479663697639015,
        1.0342892789042362,
        0.704001007621002,
        0.2944180213561777,
        -0.10164897759420218,
        -0.4702773357163444,
        -0.7480762354038769,
        -0.9924631462936452,
        -0.7724000961675609,
        -0.40504947951421305,
        0.07435119111056739,
        0.5446559068921867,
        0.8320709908321136,
        1.0642322309555614,
        0.8819909861883338,
        0.42054805977994536,
    ];
    fn d1() -> FdMatrix {
        let (n, m) = (6usize, 20usize);
        let mut flat = vec![0.0; n * m];
        for i in 0..n {
            for t in 0..m {
                let u = t as f64 / (m - 1) as f64;
                let shift = ((i * 37) % 11) as f64 / 11.0 * 0.15;
                let noise = ((i * 7919 + t * 104729) % 1009) as f64 / 1009.0;
                flat[i + t * n] =
                    (2.0 * std::f64::consts::PI * (u - shift) * 1.5 + 0.08 * i as f64).sin()
                        + 0.25 * (5.0 * u + i as f64).cos()
                        + 0.05 * noise;
            }
        }
        FdMatrix::from_column_major(flat, n, m).unwrap()
    }

    fn from_rows(rows: &[&[f64]]) -> FdMatrix {
        let (n, m) = (rows.len(), rows[0].len());
        let mut flat = vec![0.0; n * m];
        for (i, r) in rows.iter().enumerate() {
            for (j, &v) in r.iter().enumerate() {
                flat[i + j * n] = v;
            }
        }
        FdMatrix::from_column_major(flat, n, m).unwrap()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length");
        for (k, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "{what}[{k}]: got {g}, want {w}");
        }
    }

    fn run(cfg: &DbaConfig) -> DbaResult {
        dtw_barycenter_averaging(&d1(), cfg).unwrap()
    }

    #[test]
    fn tslearn_docstring_examples() {
        let data = from_rows(&[&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 4.0, 5.0]]);
        let cfg = DbaConfig {
            max_iter: 5,
            ..DbaConfig::default()
        };
        let res = dtw_barycenter_averaging(&data, &cfg).unwrap();
        assert_eq!(res.barycenter, vec![1.0, 2.0, 3.5, 4.5]);

        let data = from_rows(&[&[0.0, 0.0, 0.0], &[10.0, 10.0, 10.0]]);
        let cfg = DbaConfig {
            max_iter: 1,
            weights: Some(vec![0.75, 0.25]),
            ..DbaConfig::default()
        };
        let res = dtw_barycenter_averaging(&data, &cfg).unwrap();
        assert_eq!(res.barycenter, vec![2.5, 2.5, 2.5]);
        assert_eq!(res.n_iter, 1);
    }

    #[test]
    fn tslearn_golden_mean_init() {
        let res = run(&DbaConfig::default());
        assert_close(&res.barycenter, &MEAN_NONE, 1e-10, "mean");
        assert!((res.cost - MEAN_NONE_COST).abs() < 1e-10);
        assert!(res.converged);
    }

    #[test]
    fn tslearn_golden_windows() {
        let mut cfg = DbaConfig {
            window: DtwWindow::SakoeChiba { radius: 1 },
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &MEAN_SC1, 1e-10, "sakoe-chiba r=1");
        assert!((res.cost - MEAN_SC1_COST).abs() < 1e-10);

        cfg.window = DtwWindow::Itakura { max_slope: 2.0 };
        let res = run(&cfg);
        assert_close(&res.barycenter, &MEAN_IT2, 1e-10, "itakura 2");
        assert!((res.cost - MEAN_IT2_COST).abs() < 1e-10);
    }

    #[test]
    fn tslearn_golden_size_weights_custom() {
        let cfg = DbaConfig {
            barycenter_size: Some(12),
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &MEAN_SIZE12, 1e-10, "size 12");
        assert!((res.cost - MEAN_SIZE12_COST).abs() < 1e-10);

        let cfg = DbaConfig {
            weights: Some(vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.25]),
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &WEIGHTED, 1e-10, "weighted");
        assert!((res.cost - WEIGHTED_COST).abs() < 1e-10);

        let cfg = DbaConfig {
            init: DbaInit::Custom(d1().row(3)),
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &CUSTOM_INIT_ROW3, 1e-10, "custom init");
        assert!((res.cost - CUSTOM_INIT_ROW3_COST).abs() < 1e-10);
    }

    #[test]
    fn medoid_init_matches_brute_force_and_tslearn() {
        let data = d1();
        let rows: Vec<Vec<f64>> = (0..6).map(|i| data.row(i)).collect();
        let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
        let idx = dtw_medoid_index(&refs, &[1.0; 6], DtwWindow::None).unwrap();
        // numpy.argmin((cdist_dtw(X, X) ** 2).sum(axis=1)) == 4
        assert_eq!(idx, 4);
        let totals: Vec<f64> = rows
            .iter()
            .map(|a| {
                rows.iter()
                    .map(|b| super::super::dtw::dtw_distance(a, b, 2.0, 20))
                    .sum()
            })
            .collect();
        assert!(totals.iter().all(|&t| t >= totals[4]));

        let cfg = DbaConfig {
            init: DbaInit::Medoid,
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &MEDOID_INIT, 1e-10, "medoid init");
        assert!((res.cost - MEDOID_INIT_COST).abs() < 1e-10);
    }

    #[test]
    fn dtaidistance_dba_steps() {
        let mut cfg = DbaConfig {
            init: DbaInit::Custom(d1().row(2)),
            max_iter: 1,
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_close(&res.barycenter, &DTAI_STEP1_FROM_ROW2, 1e-12, "dtai step 1");

        cfg.max_iter = 10;
        cfg.tol = 0.0;
        let res = run(&cfg);
        assert_close(
            &res.barycenter,
            &DTAI_LOOP10_FROM_ROW2,
            1e-12,
            "dtai loop 10",
        );
    }

    #[test]
    fn cost_is_monotone_non_increasing() {
        let mut prev = f64::INFINITY;
        for it in 0..=12 {
            let cfg = DbaConfig {
                max_iter: it,
                tol: 0.0,
                window: DtwWindow::SakoeChiba { radius: 3 },
                ..DbaConfig::default()
            };
            let res = run(&cfg);
            assert!(
                res.cost <= prev + 1e-12,
                "iteration {it}: {} > {prev}",
                res.cost
            );
            prev = res.cost;
        }
    }

    #[test]
    fn zero_radius_band_gives_weighted_arithmetic_mean() {
        let data = d1();
        let w = [1.0, 2.0, 0.5, 3.0, 1.5, 0.25];
        let cfg = DbaConfig {
            window: DtwWindow::SakoeChiba { radius: 0 },
            weights: Some(w.to_vec()),
            ..DbaConfig::default()
        };
        let res = dtw_barycenter_averaging(&data, &cfg).unwrap();
        let wsum: f64 = w.iter().sum();
        let want: Vec<f64> = (0..20)
            .map(|j| {
                w.iter()
                    .enumerate()
                    .map(|(i, wi)| wi * data[(i, j)])
                    .sum::<f64>()
                    / wsum
            })
            .collect();
        assert_close(&res.barycenter, &want, 1e-12, "radius-0 mean");
        let euclid: f64 = w
            .iter()
            .enumerate()
            .map(|(i, wi)| {
                wi * want
                    .iter()
                    .enumerate()
                    .map(|(j, b)| (data[(i, j)] - b).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / wsum;
        assert!((res.cost - euclid).abs() < 1e-12);
    }

    #[test]
    fn identical_series_are_a_fixed_point() {
        let x = [0.0, 1.0, 3.0, 2.0, -1.0];
        let data = from_rows(&[&x, &x, &x]);
        for init in [DbaInit::Mean, DbaInit::Medoid] {
            let cfg = DbaConfig {
                init,
                ..DbaConfig::default()
            };
            let res = dtw_barycenter_averaging(&data, &cfg).unwrap();
            assert_eq!(res.barycenter, x.to_vec());
            assert_eq!(res.cost, 0.0);
            assert!(res.converged);
        }
    }

    #[test]
    fn weight_scale_invariance() {
        let a = DbaConfig {
            weights: Some(vec![1.0, 2.0, 0.5, 3.0, 1.5, 0.25]),
            ..DbaConfig::default()
        };
        let mut b = a.clone();
        b.weights = Some(vec![4.0, 8.0, 2.0, 12.0, 6.0, 1.0]);
        let (ra, rb) = (run(&a), run(&b));
        assert_close(&ra.barycenter, &rb.barycenter, 1e-12, "scaled weights");
        assert!((ra.cost - rb.cost).abs() < 1e-12);
    }

    #[test]
    fn zero_iterations_return_init() {
        let cfg = DbaConfig {
            max_iter: 0,
            init: DbaInit::Custom(d1().row(1)),
            ..DbaConfig::default()
        };
        let res = run(&cfg);
        assert_eq!(res.barycenter, d1().row(1));
        assert_eq!(res.n_iter, 0);
        assert!(!res.converged);
    }

    #[test]
    fn interpolation_matches_linear_resampling() {
        assert_eq!(
            interpolate_linear(&[0.0, 2.0, 4.0], 5),
            vec![0.0, 1.0, 2.0, 3.0, 4.0]
        );
        assert_eq!(interpolate_linear(&[7.0], 3), vec![7.0; 3]);
        assert_eq!(interpolate_linear(&[1.0, 3.0], 1), vec![1.0]);
    }

    #[test]
    fn errors() {
        let data = d1();
        let bad = |cfg: DbaConfig| dtw_barycenter_averaging(&data, &cfg).unwrap_err();
        let empty = FdMatrix::zeros(0, 5);
        assert!(matches!(
            dtw_barycenter_averaging(&empty, &DbaConfig::default()),
            Err(FdarError::InvalidDimension { .. })
        ));
        let c = DbaConfig {
            weights: Some(vec![1.0; 5]),
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidDimension { .. }));
        for w in [
            vec![-1.0, 1., 1., 1., 1., 1.],
            vec![0.0; 6],
            vec![f64::NAN; 6],
        ] {
            let c = DbaConfig {
                weights: Some(w),
                ..DbaConfig::default()
            };
            assert!(matches!(bad(c), FdarError::InvalidParameter { .. }));
        }
        let c = DbaConfig {
            tol: -1.0,
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidParameter { .. }));
        let c = DbaConfig {
            barycenter_size: Some(0),
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidParameter { .. }));
        let c = DbaConfig {
            init: DbaInit::Custom(vec![]),
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidParameter { .. }));
        let c = DbaConfig {
            init: DbaInit::Custom(vec![0.0; 7]),
            barycenter_size: Some(8),
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidDimension { .. }));
        let c = DbaConfig {
            window: DtwWindow::Itakura { max_slope: 0.5 },
            ..DbaConfig::default()
        };
        assert!(matches!(bad(c), FdarError::InvalidParameter { .. }));
        let mut nan = d1();
        nan[(0, 0)] = f64::NAN;
        assert!(matches!(
            dtw_barycenter_averaging(&nan, &DbaConfig::default()),
            Err(FdarError::InvalidParameter { .. })
        ));
    }
}
