//! Downstream analyses of R^d curve shapes: TSRVF tangent-space coordinates,
//! elastic curve depth and outlier detection.

use super::geometry::{self, Grid};
use super::karcher::{
    align_all, curve_shape_karcher_mean, curve_shape_karcher_median, exp_map, log_map,
};
use super::{
    align_to_srvf, curve_from_set, prepare, total_distance, validate_pair, CurveShapeConfig,
    CurveShapeKarcherResult, Prepared,
};
use crate::alignment::outlier::{quantile_threshold, tukey_fence};
use crate::alignment::OutlierThreshold;
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
use crate::warping::phase_distance;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Shared input handling ──────────────────────────────────────────────────

/// Validate a curve set against `config` and preprocess every curve.
pub(super) fn prepare_set(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
    min_curves: usize,
) -> Result<(Vec<Prepared>, Grid), FdarError> {
    let n = curves.ncurves();
    if n < min_curves || curves.ndim() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: format!("at least {min_curves} curves with d >= 1"),
            actual: format!("{n} curves, d = {}", curves.ndim()),
        });
    }
    let list: Vec<FdMatrix> = (0..n)
        .map(|i| curve_from_set(curves, i))
        .collect::<Result<_, _>>()?;
    // A set has a common shape, so validating each curve against itself checks
    // the configuration, sample count, finiteness and the closed-curve d >= 2.
    for c in &list {
        validate_pair(c, c, config)?;
    }
    let grid = Grid::new(curves.npoints());
    let prepared = list
        .iter()
        .map(|c| prepare(c, config, &grid))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((prepared, grid))
}

/// Flatten a `d × m` field into the `k·m + j` layout of shooting vectors.
fn flatten_into(out: &mut FdMatrix, i: usize, v: &FdMatrix) {
    let (d, m) = v.shape();
    for k in 0..d {
        for j in 0..m {
            out[(i, k * m + j)] = v[(k, j)];
        }
    }
}

// ─── TSRVF ──────────────────────────────────────────────────────────────────

/// Tangent-space (TSRVF) representation of curves at a Karcher shape mean.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeTsrvfResult {
    /// Tangent vectors `vᵢ = log_μ(qᵢ*)` (`n × (d·m)`, column `k·m + j` =
    /// coordinate `k` at sample `j`, the layout of
    /// [`CurveShapeKarcherResult::shooting_vectors`]). They lie in the tangent
    /// space at μ: orthogonal to μ when scale is removed and to the closure
    /// normals for closed contours.
    pub tangent_vectors: FdMatrix,
    /// Shape distance of each aligned curve to the mean (`= ‖vᵢ‖` on the sphere).
    pub distances: Vec<f64>,
    /// SRVFs of the curves aligned to the mean.
    pub aligned_srvfs: FdCurveSet,
    /// Per-curve optimal rotation (`d × d`).
    pub rotations: Vec<FdMatrix>,
    /// Per-curve warps (`n × m`).
    pub gammas: FdMatrix,
    /// Per-curve starting-point shifts (0 for open curves).
    pub seeds: Vec<usize>,
}

/// TSRVF of R^d curves: align each curve to the Karcher mean `μ` of `mean`
/// (rotation, scale, warp and seed under `mean.config`) and map its SRVF into the
/// tangent space at `μ` with the inverse exponential map — the sphere log map
/// when scale is removed, the flat difference otherwise, projected onto the
/// tangent space of the closed-curve manifold for closed contours (fdasrsf
/// `fdacurve.karcher_mean` shooting vectors / fdasrvf `multivariate_karcher_cov`
/// input). Tangent vectors live in a linear space, so they can feed ordinary
/// multivariate methods (PCA, regression, classification); map them back with
/// [`curve_shape_tangent_to_curve`].
///
/// Applied to the curves `mean` was computed from, this reproduces
/// [`CurveShapeKarcherResult::shooting_vectors`] exactly. `curves` may also be
/// new curves sampled on the same grid.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `curves` is empty or its `d`, `m` differ
/// from the mean; [`FdarError::InvalidParameter`] / [`FdarError::ComputationFailed`]
/// for invalid or zero-length curves.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_karcher_mean, curve_shape_tsrvf, curves_to_set, CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curve = |a: f64| {
///     let mut b = FdMatrix::zeros(2, m);
///     for j in 0..m {
///         let t = j as f64 / (m - 1) as f64;
///         b[(0, j)] = t;
///         b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///     }
///     b
/// };
/// let set = curves_to_set(&[curve(0.2), curve(0.3), curve(0.4)]).unwrap();
/// let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 10, 1e-3).unwrap();
/// let new = curves_to_set(&[curve(0.5)]).unwrap();
/// let ts = curve_shape_tsrvf(&km, &new).unwrap();
/// assert_eq!(ts.tangent_vectors.shape(), (1, 2 * m));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_tsrvf(
    mean: &CurveShapeKarcherResult,
    curves: &FdCurveSet,
) -> Result<CurveShapeTsrvfResult, FdarError> {
    let (d, m) = mean.mean_srvf.shape();
    if curves.ndim() != d || curves.npoints() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: format!("d = {d}, m = {m} (as the mean)"),
            actual: format!("d = {}, m = {}", curves.ndim(), curves.npoints()),
        });
    }
    let cfg = &mean.config;
    let (prepared, grid) = prepare_set(curves, cfg, 1)?;
    let mu = &mean.mean_srvf;
    let cands = align_all(mu, &prepared, cfg, &grid);
    let n = cands.len();
    let mut tangent_vectors = FdMatrix::zeros(n, d * m);
    let mut gammas = FdMatrix::zeros(n, m);
    for (i, c) in cands.iter().enumerate() {
        flatten_into(&mut tangent_vectors, i, &log_map(mu, &c.q, cfg, &grid));
        for (j, &g) in c.gamma.iter().enumerate() {
            gammas[(i, j)] = g;
        }
    }
    let aligned: Vec<FdMatrix> = cands.iter().map(|c| c.q.clone()).collect();
    Ok(CurveShapeTsrvfResult {
        tangent_vectors,
        distances: cands.iter().map(|c| c.distance).collect(),
        aligned_srvfs: super::curves_to_set(&aligned)?,
        rotations: cands.iter().map(|c| c.rotation.clone()).collect(),
        gammas,
        seeds: cands.iter().map(|c| c.seed).collect(),
    })
}

/// Map a tangent vector at the Karcher mean back to a curve:
/// `β = srvf⁻¹(exp_μ(v))` (closure-projected for closed contours, centred; unit
/// length when scale is removed). `v` has length `d·m` in the layout of
/// [`CurveShapeTsrvfResult::tangent_vectors`] rows; it is first projected onto
/// the tangent space at `μ`, so any vector (e.g. a regression prediction) is
/// accepted. `v = 0` gives the mean shape.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `v.len() != d·m`;
/// [`FdarError::InvalidParameter`] if `v` is not finite.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_karcher_mean, curve_shape_tangent_to_curve, curve_shape_tsrvf,
///     curves_to_set, CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curve = |a: f64| {
///     let mut b = FdMatrix::zeros(2, m);
///     for j in 0..m {
///         let t = j as f64 / (m - 1) as f64;
///         b[(0, j)] = t;
///         b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///     }
///     b
/// };
/// let set = curves_to_set(&[curve(0.2), curve(0.3), curve(0.4)]).unwrap();
/// let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 10, 1e-3).unwrap();
/// let ts = curve_shape_tsrvf(&km, &set).unwrap();
/// let beta = curve_shape_tangent_to_curve(&km, &ts.tangent_vectors.row(2)).unwrap();
/// assert_eq!(beta.shape(), (2, m)); // ≈ curve 2 aligned to the mean
/// ```
pub fn curve_shape_tangent_to_curve(
    mean: &CurveShapeKarcherResult,
    v: &[f64],
) -> Result<FdMatrix, FdarError> {
    let (d, m) = mean.mean_srvf.shape();
    if v.len() != d * m {
        return Err(FdarError::InvalidDimension {
            parameter: "v",
            expected: format!("length d*m = {}", d * m),
            actual: format!("length {}", v.len()),
        });
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "v",
            message: "contains non-finite values".to_string(),
        });
    }
    let mut field = FdMatrix::zeros(d, m);
    for k in 0..d {
        for j in 0..m {
            field[(k, j)] = v[k * m + j];
        }
    }
    let grid = Grid::new(m);
    let cfg = &mean.config;
    let field = geometry::project_tangent(&field, &mean.mean_srvf, &grid, cfg.scale, cfg.closed);
    let q = exp_map(&mean.mean_srvf, &field, cfg, &grid);
    Ok(geometry::center(&geometry::q_to_curve(&q, &grid), &grid))
}

// ─── Depth ──────────────────────────────────────────────────────────────────

/// Elastic amplitude / phase depth of R^d curves (see [`curve_shape_depth`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeDepthResult {
    /// Amplitude (shape) depth `1/(1 + medianⱼ dᵢⱼ)` in `(0, 1]`.
    pub amplitude_depth: Vec<f64>,
    /// Phase depth `((2+π)/π)·(1/(1 + medianⱼ dxᵢⱼ) − 2/(2+π))` in `[0, 1]`.
    pub phase_depth: Vec<f64>,
    /// Pairwise shape distances `dᵢⱼ` (`n × n`, symmetric, zero diagonal).
    pub amplitude_distances: FdMatrix,
    /// Pairwise phase distances `dxᵢⱼ` (geodesic distance of the optimal warp to
    /// the identity on the Hilbert sphere, in `[0, π/2]`).
    pub phase_distances: FdMatrix,
}

/// Pairwise shape and phase distance matrices (curve `j` aligned to `i`, `i < j`).
fn pairwise_amp_phase(
    prepared: &[Prepared],
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> (FdMatrix, FdMatrix) {
    let n = prepared.len();
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let vals: Vec<(f64, f64)> = iter_maybe_parallel!(pairs.clone())
        .map(|(i, j)| {
            let c = align_to_srvf(&prepared[i].q, &prepared[j].beta, cfg, grid);
            let amp = total_distance(c.distance, prepared[i].length, prepared[j].length, cfg);
            (amp, phase_distance(&c.gamma, &grid.t))
        })
        .collect();
    let mut amp = FdMatrix::zeros(n, n);
    let mut phs = FdMatrix::zeros(n, n);
    for (&(i, j), &(a, p)) in pairs.iter().zip(&vals) {
        amp[(i, j)] = a;
        amp[(j, i)] = a;
        phs[(i, j)] = p;
        phs[(j, i)] = p;
    }
    (amp, phs)
}

/// R-style median (mean of the two middle order statistics for even length).
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        0.5 * (values[n / 2 - 1] + values[n / 2])
    }
}

/// Elastic depth of R^d curves in shape space (fdasrvf `curve_depth`).
///
/// Every pair of curves is aligned under `config` (rotation, scale, warp,
/// seed); `dᵢⱼ` is the resulting shape distance and `dxᵢⱼ` the phase distance
/// of the optimal warp. With the row medians over all `n` entries (the zero
/// diagonal included, as in fdasrvf) the depths are
///
/// * amplitude: `1 / (1 + medianⱼ dᵢⱼ)`,
/// * phase: `((2+π)/π) · (1/(1 + medianⱼ dxᵢⱼ) − 2/(2+π))`, rescaled to `[0, 1]`
///   because phase distances are at most `π/2`.
///
/// The amplitude depth is invariant to rotating, rescaling, translating and
/// reparametrising any curve (up to the DP discretisation). Pairs are computed
/// in parallel with the `parallel` feature.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 2 curves or invalid shapes;
/// [`FdarError::InvalidParameter`] for an invalid `config`;
/// [`FdarError::ComputationFailed`] for a zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_depth, curves_to_set, CurveShapeConfig};
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curves: Vec<FdMatrix> = (0..5)
///     .map(|i| {
///         let a = 0.1 + 0.1 * i as f64;
///         let mut b = FdMatrix::zeros(2, m);
///         for j in 0..m {
///             let t = j as f64 / (m - 1) as f64;
///             b[(0, j)] = t;
///             b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///         }
///         b
///     })
///     .collect();
/// let depth = curve_shape_depth(&curves_to_set(&curves).unwrap(), &CurveShapeConfig::default())
///     .unwrap();
/// // the ends of the one-parameter family are less deep than its middle
/// let amp = &depth.amplitude_depth;
/// assert!(amp[2] > amp[0] && amp[2] > amp[4]);
/// assert!(depth.phase_depth.iter().all(|&p| (0.0..=1.0).contains(&p)));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_depth(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
) -> Result<CurveShapeDepthResult, FdarError> {
    let (prepared, grid) = prepare_set(curves, config, 2)?;
    let n = prepared.len();
    let (amplitude_distances, phase_distances) = pairwise_amp_phase(&prepared, config, &grid);
    let row_median = |mat: &FdMatrix, i: usize| median(&mut mat.row(i));
    let pi = std::f64::consts::PI;
    let amplitude_depth = (0..n)
        .map(|i| 1.0 / (1.0 + row_median(&amplitude_distances, i)))
        .collect();
    let phase_depth = (0..n)
        .map(|i| {
            let p = 1.0 / (1.0 + row_median(&phase_distances, i));
            ((2.0 + pi) / pi) * (p - 2.0 / (2.0 + pi))
        })
        .collect();
    Ok(CurveShapeDepthResult {
        amplitude_depth,
        phase_depth,
        amplitude_distances,
        phase_distances,
    })
}

// ─── Outlier detection ──────────────────────────────────────────────────────

/// Centre of a set of curve shapes.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CurveShapeCentre {
    /// Karcher mean ([`curve_shape_karcher_mean`]).
    Mean,
    /// Karcher median ([`curve_shape_karcher_median`]) — robust to the outliers
    /// being detected (default).
    #[default]
    Median,
}

/// Configuration of [`curve_shape_outlier_detection`].
///
/// Construct with `CurveShapeOutlierConfig::default()` and assign fields (the
/// struct is `#[non_exhaustive]`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CurveShapeOutlierConfig {
    /// Shape quotient (rotation, scale, closure, …). Default
    /// `CurveShapeConfig::default()`.
    pub shape: CurveShapeConfig,
    /// Reference shape (default: Karcher median).
    pub centre: CurveShapeCentre,
    /// Cutoff rule on the distances to the centre (default: Tukey fence
    /// `Q3 + 1.5·IQR`, as fdasrvf `outlier.detection`).
    pub threshold_method: OutlierThreshold,
    /// Tail fraction for [`OutlierThreshold::Quantile`] (default 0.05; no effect
    /// with the Tukey fence).
    pub alpha: f64,
    /// Karcher iterations for the centre (default 20).
    pub max_iter: usize,
    /// Karcher tolerance on the shooting-vector norm (default 1e-3).
    pub tol: f64,
}

impl Default for CurveShapeOutlierConfig {
    fn default() -> Self {
        Self {
            shape: CurveShapeConfig::default(),
            centre: CurveShapeCentre::default(),
            threshold_method: OutlierThreshold::default(),
            alpha: 0.05,
            max_iter: 20,
            tol: 1e-3,
        }
    }
}

/// Result of [`curve_shape_outlier_detection`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeOutlierResult {
    /// Indices of the flagged curves (increasing).
    pub outlier_indices: Vec<usize>,
    /// Shape distance of each curve to the centre.
    pub distances: Vec<f64>,
    /// Phase distance of each curve's optimal warp to the identity.
    pub phase_distances: Vec<f64>,
    /// Cutoff applied to `distances`.
    pub threshold: f64,
    /// The Karcher mean / median the distances refer to (with the aligned curves).
    pub centre: CurveShapeKarcherResult,
}

/// Validate the threshold rule of a curve outlier configuration.
fn validate_outlier_config(config: &CurveShapeOutlierConfig) -> Result<(), FdarError> {
    match config.threshold_method {
        OutlierThreshold::TukeyFence { k } if !(k.is_finite() && k >= 0.0) => {
            Err(FdarError::InvalidParameter {
                parameter: "threshold_method",
                message: format!("Tukey fence k must be finite and >= 0, got {k}"),
            })
        }
        OutlierThreshold::Quantile if !(config.alpha > 0.0 && config.alpha < 1.0) => {
            Err(FdarError::InvalidParameter {
                parameter: "alpha",
                message: format!(
                    "must be in (0, 1) for the quantile rule, got {}",
                    config.alpha
                ),
            })
        }
        _ => Ok(()),
    }
}

/// Shape outlier detection for R^d curves: compute the Karcher median (or
/// mean) shape, take each curve's shape distance to it and flag the curves
/// beyond the cutoff of `config.threshold_method` — the Tukey fence
/// `Q3 + k·IQR` (default `k = 1.5`, the rule of fdasrvf `outlier.detection`) or
/// the empirical `(1 − alpha)` quantile; the same rules as the 1-D
/// [`elastic_outlier_detection`](crate::alignment::elastic_outlier_detection).
/// Distances are invariant to translation, scale, rotation and
/// reparametrisation of the curves (per `config.shape`).
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 2 curves or invalid shapes;
/// [`FdarError::InvalidParameter`] for invalid configuration;
/// [`FdarError::ComputationFailed`] for a zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_outlier_detection, curves_to_set, CurveShapeOutlierConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 40;
/// let mut curves: Vec<FdMatrix> = (0..8)
///     .map(|i| {
///         let a = 0.3 + 0.01 * i as f64;
///         let mut b = FdMatrix::zeros(2, m);
///         for j in 0..m {
///             let t = j as f64 / (m - 1) as f64;
///             b[(0, j)] = t;
///             b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///         }
///         b
///     })
///     .collect();
/// // a zig-zag curve has a very different shape
/// for j in 0..m {
///     curves[5][(1, j)] = 0.3 * (8.0 * std::f64::consts::PI * j as f64 / (m - 1) as f64).sin();
/// }
/// let set = curves_to_set(&curves).unwrap();
/// let res = curve_shape_outlier_detection(&set, &CurveShapeOutlierConfig::default()).unwrap();
/// assert_eq!(res.outlier_indices, vec![5]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_outlier_detection(
    curves: &FdCurveSet,
    config: &CurveShapeOutlierConfig,
) -> Result<CurveShapeOutlierResult, FdarError> {
    validate_outlier_config(config)?;
    let centre = match config.centre {
        CurveShapeCentre::Median => {
            curve_shape_karcher_median(curves, &config.shape, config.max_iter, config.tol)?
        }
        _ => curve_shape_karcher_mean(curves, &config.shape, config.max_iter, config.tol)?,
    };
    let distances = centre.distances.clone();
    let m = centre.gammas.ncols();
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let phase_distances = (0..centre.gammas.nrows())
        .map(|i| phase_distance(&centre.gammas.row(i), &t))
        .collect();
    let threshold = match config.threshold_method {
        OutlierThreshold::TukeyFence { k } => tukey_fence(&distances, k),
        _ => quantile_threshold(&distances, config.alpha),
    };
    let outlier_indices = (0..distances.len())
        .filter(|&i| distances[i] > threshold)
        .collect();
    Ok(CurveShapeOutlierResult {
        outlier_indices,
        distances,
        phase_distances,
        threshold,
        centre,
    })
}
