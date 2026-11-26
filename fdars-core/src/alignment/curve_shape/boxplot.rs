//! Shape box plot and bootstrap confidence region for R^d curves.

use rand::Rng;

use super::geometry::{self, Grid};
use super::karcher::curve_shape_karcher_mean;
use super::{
    curve_from_set, curve_shape_align_pair, curves_to_set, transform_to_curve, validate_curve,
    CurveShapeConfig, CurveShapeKarcherResult,
};
use crate::error::FdarError;
use crate::helpers::{quantile_sorted, seed_for_thread};
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Box plot ───────────────────────────────────────────────────────────────

/// Configuration of the curve shape box plot (fdasrvf `curve_boxplot`).
///
/// Construct with `CurveShapeBoxplotConfig::default()` and assign fields (the
/// struct is `#[non_exhaustive]`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CurveShapeBoxplotConfig {
    /// The outer central region holds `round(n(1 − alpha))` curves (default 0.05).
    pub alpha: f64,
    /// Whisker length in units of the IQR (fdasrvf `range`, default 1.0).
    pub k_a: f64,
}

impl Default for CurveShapeBoxplotConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            k_a: 1.0,
        }
    }
}

/// Elastic shape box plot of R^d curves (see [`curve_shape_boxplot_from_aligned`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeBoxplot {
    /// Median shape as a centred curve (`d × m`).
    pub median: FdMatrix,
    /// SRVF of the median.
    pub median_srvf: FdMatrix,
    /// Shape distance of each curve to the median.
    pub distances: Vec<f64>,
    /// Largest distance inside the 50% central region.
    pub r50: f64,
    /// Largest distance inside the `(1 − alpha)` central region.
    pub ralpha: f64,
    /// Index of the first quartile curve.
    pub q1_index: usize,
    /// Index of the third quartile curve.
    pub q3_index: usize,
    /// Index of the lower `(1 − alpha)` quantile curve.
    pub q1a_index: usize,
    /// Index of the upper `(1 − alpha)` quantile curve.
    pub q3a_index: usize,
    /// First quartile curve (aligned, `d × m`).
    pub q1: FdMatrix,
    /// Third quartile curve.
    pub q3: FdMatrix,
    /// Lower `(1 − alpha)` quantile curve.
    pub q1a: FdMatrix,
    /// Upper `(1 − alpha)` quantile curve.
    pub q3a: FdMatrix,
    /// SRVF of the upper whisker `Q3 + k_a·IQR·(q_{Q3} − q_med)/‖·‖`.
    pub upper_whisker_srvf: FdMatrix,
    /// SRVF of the lower whisker.
    pub lower_whisker_srvf: FdMatrix,
    /// Upper whisker curve (centred).
    pub upper_whisker: FdMatrix,
    /// Lower whisker curve (centred).
    pub lower_whisker: FdMatrix,
    /// Whisker distance `max(d(upper, median), d(lower, median))`.
    pub whisker_distance: f64,
    /// Curves beyond the whisker distance (decreasing distance).
    pub outlier_indices: Vec<usize>,
    /// Index of the curve outside the 50% region closest to the lower whisker
    /// (the Q1 curve if no such curve exists).
    pub min_index: usize,
    /// Index of the curve outside the 50% region closest to the upper whisker
    /// (the Q3 curve if no such curve exists).
    pub max_index: usize,
    /// Minimum extreme curve (aligned).
    pub minimum: FdMatrix,
    /// Maximum extreme curve (aligned).
    pub maximum: FdMatrix,
}

/// Shape distance used by the box plot (fdasrvf `curvebox_data`): arc length
/// `arccos⟨a, b⟩` (clamped) on the sphere, `‖a − b‖` in the flat case.
fn box_dist(a: &FdMatrix, b: &FdMatrix, scale: bool, grid: &Grid) -> f64 {
    if scale {
        geometry::inner(a, b, grid).clamp(-1.0, 1.0).acos()
    } else {
        geometry::norm(&geometry::axpby(1.0, a, -1.0, b), grid)
    }
}

/// `x / ‖x‖` (zero for a zero field).
fn unit(x: &FdMatrix, grid: &Grid) -> FdMatrix {
    let n = geometry::norm(x, grid);
    geometry::scaled(x, if n > 0.0 { 1.0 / n } else { 0.0 })
}

/// Quartile pair inside a central region by the fdasrvf energy criterion
/// `(1−λ)(dᵢ + dⱼ)/max d − λ(⟨uᵢ, uⱼ⟩ + 1)`, `λ = 0.5`, `uᵢ` the unit direction
/// of `qᵢ − q_med`. Returns positions in `region`, reproducing R's
/// `which(energy == max(energy), arr.ind = TRUE)[1, ]` (first maximum in
/// column-major order of the zero-initialised matrix).
fn quartile_pair(region: &[usize], dirs: &[FdMatrix], dist: &[f64], grid: &Grid) -> (usize, usize) {
    const LAM: f64 = 0.5;
    let rmax = region
        .iter()
        .map(|&i| dist[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let units: Vec<FdMatrix> = region.iter().map(|&i| unit(&dirs[i], grid)).collect();
    let mut best = (0, 0);
    let mut best_e = 0.0;
    for j in 0..region.len() {
        for i in 0..j {
            let angle = geometry::inner(&units[i], &units[j], grid);
            let e = (1.0 - LAM) * (dist[region[i]] / rmax + dist[region[j]] / rmax)
                - LAM * (angle + 1.0);
            if e > best_e {
                best_e = e;
                best = (i, j);
            }
        }
    }
    best
}

/// Number of curves in a central region of nominal fraction `p` (R `round`, half
/// to even), at least 2 so that a quartile pair exists.
fn region_size(n: usize, p: f64) -> usize {
    ((n as f64 * p).round_ties_even() as usize).clamp(2, n)
}

fn validate_box_inputs(
    median_srvf: &FdMatrix,
    aligned_srvfs: &FdCurveSet,
    aligned_curves: &FdCurveSet,
    config: &CurveShapeBoxplotConfig,
) -> Result<(), FdarError> {
    validate_curve(median_srvf, "median_srvf")?;
    let (d, m) = median_srvf.shape();
    let n = aligned_srvfs.ncurves();
    for (name, set) in [
        ("aligned_srvfs", aligned_srvfs),
        ("aligned_curves", aligned_curves),
    ] {
        if set.ndim() != d || set.npoints() != m || set.ncurves() != n {
            return Err(FdarError::InvalidDimension {
                parameter: name,
                expected: format!("{n} curves, d = {d}, m = {m}"),
                actual: format!(
                    "{} curves, d = {}, m = {}",
                    set.ncurves(),
                    set.ndim(),
                    set.npoints()
                ),
            });
        }
    }
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_srvfs",
            expected: "at least 3 curves".to_string(),
            actual: format!("{n} curves"),
        });
    }
    if !(config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {}", config.alpha),
        });
    }
    if !(config.k_a > 0.0 && config.k_a.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "k_a",
            message: format!("must be positive and finite, got {}", config.k_a),
        });
    }
    Ok(())
}

/// Whisker SRVF `base + k·IQR·dir/‖dir‖`.
fn whisker(base: &FdMatrix, dir: &FdMatrix, len: f64, grid: &Grid) -> FdMatrix {
    geometry::axpby(1.0, base, len, &unit(dir, grid))
}

/// Outliers (decreasing distance, fdasrvf loop) and the extreme indices.
fn outliers_and_extremes(
    dist: &[f64],
    order: &[usize],
    cr50: &[usize],
    whisker_distance: f64,
    qs: &[FdMatrix],
    whiskers: (&FdMatrix, &FdMatrix),
    scale: bool,
    grid: &Grid,
) -> (Vec<usize>, Option<(usize, usize)>) {
    let outliers: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .take_while(|&i| dist[i] > whisker_distance)
        .collect();
    let (upper, lower) = whiskers;
    let mut best: Option<(usize, f64, usize, f64)> = None;
    for j in (0..dist.len()).filter(|j| !cr50.contains(j) && !outliers.contains(j)) {
        let (du, dl) = (
            box_dist(upper, &qs[j], scale, grid),
            box_dist(lower, &qs[j], scale, grid),
        );
        best = Some(match best {
            None => (j, du, j, dl),
            Some((mx, bu, mn, bl)) => {
                let (mx, bu) = if du < bu { (j, du) } else { (mx, bu) };
                let (mn, bl) = if dl < bl { (j, dl) } else { (mn, bl) };
                (mx, bu, mn, bl)
            }
        });
    }
    (outliers, best.map(|(mx, _, mn, _)| (mn, mx)))
}

/// Elastic shape box plot from curves already aligned to a median shape — a
/// port of fdasrvf 2.5.0 `curve_boxplot` (`curvebox_data`).
///
/// With `dᵢ` the shape distance of aligned SRVF `qᵢ` to the median SRVF
/// (`arccos⟨q_med, qᵢ⟩` if `scale`, else `‖q_med − qᵢ‖`): the 50% and
/// `(1 − alpha)` central regions are the `round(n/2)` and `round(n(1 − alpha))`
/// closest curves; inside each, the quartile pair maximises the fdasrvf energy
/// (far from the median, in opposite directions). With `IQR = d_{Q1} + d_{Q3}`
/// the whiskers are `q_{Q3} + k_a·IQR·u₃` and `q_{Q1} + k_a·IQR·u₁` (`uₖ` the
/// unit direction from the median), the whisker distance is the larger of
/// their distances to the median, curves beyond it are outliers, and the
/// extremes are the remaining curves outside the 50% region closest to each
/// whisker. Inner products are trapezoidal on the grid `t_j = j/(m−1)`.
///
/// `aligned_curves` only supplies the returned quartile / extreme curves.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 3 curves or mismatched shapes;
/// [`FdarError::InvalidParameter`] for `alpha ∉ (0, 1)`, `k_a ≤ 0` or non-finite
/// input.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_boxplot, curve_shape_karcher_median, curves_to_set, CurveShapeBoxplotConfig,
///     CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curves: Vec<FdMatrix> = (0..9)
///     .map(|i| {
///         let a = 0.1 + 0.05 * i as f64;
///         let mut b = FdMatrix::zeros(2, m);
///         for j in 0..m {
///             let t = j as f64 / (m - 1) as f64;
///             b[(0, j)] = t;
///             b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///         }
///         b
///     })
///     .collect();
/// let set = curves_to_set(&curves).unwrap();
/// let med = curve_shape_karcher_median(&set, &CurveShapeConfig::default(), 10, 1e-3).unwrap();
/// let bp = curve_shape_boxplot(&med, &CurveShapeBoxplotConfig::default()).unwrap();
/// assert!(bp.r50 <= bp.ralpha && bp.q1_index != bp.q3_index);
/// ```
#[must_use = "box plot result should not be discarded"]
pub fn curve_shape_boxplot_from_aligned(
    median_srvf: &FdMatrix,
    aligned_srvfs: &FdCurveSet,
    aligned_curves: &FdCurveSet,
    scale: bool,
    config: &CurveShapeBoxplotConfig,
) -> Result<CurveShapeBoxplot, FdarError> {
    let shape = CurveShapeConfig {
        scale,
        ..CurveShapeConfig::default()
    };
    boxplot_impl(median_srvf, aligned_srvfs, aligned_curves, &shape, config)
}

/// Box plot body; `shape` supplies `scale` and the metric (for reconstructing
/// curves from the median / whisker transforms).
fn boxplot_impl(
    median_srvf: &FdMatrix,
    aligned_srvfs: &FdCurveSet,
    aligned_curves: &FdCurveSet,
    shape: &CurveShapeConfig,
    config: &CurveShapeBoxplotConfig,
) -> Result<CurveShapeBoxplot, FdarError> {
    validate_box_inputs(median_srvf, aligned_srvfs, aligned_curves, config)?;
    let scale = shape.scale;
    let n = aligned_srvfs.ncurves();
    let grid = Grid::new(median_srvf.ncols());
    let qs: Vec<FdMatrix> = (0..n)
        .map(|i| curve_from_set(aligned_srvfs, i))
        .collect::<Result<_, _>>()?;
    let curve = |i: usize| curve_from_set(aligned_curves, i);
    let dist: Vec<f64> = qs
        .iter()
        .map(|q| box_dist(median_srvf, q, scale, &grid))
        .collect();
    let dirs: Vec<FdMatrix> = qs
        .iter()
        .map(|q| geometry::axpby(1.0, q, -1.0, median_srvf))
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]));
    let cr50 = &order[..region_size(n, 0.5)];
    let cra = &order[..region_size(n, 1.0 - config.alpha)];
    let region_max = |r: &[usize]| r.iter().map(|&i| dist[i]).fold(0.0, f64::max);
    let (a, b) = quartile_pair(cr50, &dirs, &dist, &grid);
    let (q1, q3) = (cr50[a], cr50[b]);
    let (a, b) = quartile_pair(cra, &dirs, &dist, &grid);
    let (q1a, q3a) = (cra[a], cra[b]);

    let iqr = dist[q1] + dist[q3];
    let upper_q = whisker(&qs[q3], &dirs[q3], config.k_a * iqr, &grid);
    let lower_q = whisker(&qs[q1], &dirs[q1], config.k_a * iqr, &grid);
    let whisker_distance = box_dist(&upper_q, median_srvf, scale, &grid).max(box_dist(
        &lower_q,
        median_srvf,
        scale,
        &grid,
    ));
    let (outlier_indices, extremes) = outliers_and_extremes(
        &dist,
        &order,
        cr50,
        whisker_distance,
        &qs,
        (&upper_q, &lower_q),
        scale,
        &grid,
    );
    let (min_index, max_index) = extremes.unwrap_or((q1, q3));
    let to_curve = |q: &FdMatrix| geometry::center(&transform_to_curve(q, shape, &grid), &grid);
    Ok(CurveShapeBoxplot {
        median: to_curve(median_srvf),
        median_srvf: median_srvf.clone(),
        r50: region_max(cr50),
        ralpha: region_max(cra),
        q1_index: q1,
        q3_index: q3,
        q1a_index: q1a,
        q3a_index: q3a,
        q1: curve(q1)?,
        q3: curve(q3)?,
        q1a: curve(q1a)?,
        q3a: curve(q3a)?,
        upper_whisker: to_curve(&upper_q),
        lower_whisker: to_curve(&lower_q),
        upper_whisker_srvf: upper_q,
        lower_whisker_srvf: lower_q,
        whisker_distance,
        outlier_indices,
        min_index,
        max_index,
        minimum: curve(min_index)?,
        maximum: curve(max_index)?,
        distances: dist,
    })
}

/// Elastic shape box plot of the curves summarised by a Karcher **median**
/// result ([`curve_shape_karcher_median`](crate::alignment::curve_shape_karcher_median),
/// fdasrvf `curve_boxplot`, which requires a "Karcher Median" input). A mean
/// result is accepted but gives the box plot around the mean.
///
/// # Errors
/// As [`curve_shape_boxplot_from_aligned`].
#[must_use = "box plot result should not be discarded"]
pub fn curve_shape_boxplot(
    median: &CurveShapeKarcherResult,
    config: &CurveShapeBoxplotConfig,
) -> Result<CurveShapeBoxplot, FdarError> {
    boxplot_impl(
        &median.mean_srvf,
        &median.aligned_srvfs,
        &median.aligned_curves,
        &median.config,
        config,
    )
}

// ─── Bootstrap confidence region ────────────────────────────────────────────

/// Configuration of [`curve_shape_confidence_interval`].
///
/// Construct with `CurveShapeCiConfig::default()` and assign fields (the struct
/// is `#[non_exhaustive]`).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CurveShapeCiConfig {
    /// Number of bootstrap resamples (default 200).
    pub n_bootstrap: usize,
    /// Confidence level in `(0, 1)` (default 0.95).
    pub confidence_level: f64,
    /// Karcher iterations per mean (default 15).
    pub max_iter: usize,
    /// Karcher tolerance (default 1e-3).
    pub tol: f64,
    /// Random seed; resample `b` uses `seed + b` (default 42).
    pub seed: u64,
}

impl Default for CurveShapeCiConfig {
    fn default() -> Self {
        Self {
            n_bootstrap: 200,
            confidence_level: 0.95,
            max_iter: 15,
            tol: 1e-3,
            seed: 42,
        }
    }
}

/// Bootstrap confidence region of the Karcher mean shape.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeCiResult {
    /// Karcher mean of the full sample (with its alignment).
    pub mean: CurveShapeKarcherResult,
    /// Bootstrap Karcher means aligned to the full-sample mean (`d × m` each).
    pub bootstrap_means: Vec<FdMatrix>,
    /// Shape distance of each bootstrap mean to the full-sample mean.
    pub bootstrap_distances: Vec<f64>,
    /// Geodesic radius of the confidence ball around the mean: the
    /// `confidence_level` quantile of `bootstrap_distances`.
    pub radius: f64,
    /// Pointwise lower band of the aligned bootstrap means (`d × m`).
    pub lower_band: FdMatrix,
    /// Pointwise upper band (`d × m`).
    pub upper_band: FdMatrix,
}

fn validate_ci(curves: &FdCurveSet, config: &CurveShapeCiConfig) -> Result<(), FdarError> {
    if curves.ncurves() < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: "at least 3 curves".to_string(),
            actual: format!("{} curves", curves.ncurves()),
        });
    }
    if !(config.confidence_level > 0.0 && config.confidence_level < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "confidence_level",
            message: format!("must be in (0, 1), got {}", config.confidence_level),
        });
    }
    if config.n_bootstrap == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_bootstrap",
            message: "must be >= 1".to_string(),
        });
    }
    Ok(())
}

/// Pointwise `(alpha/2, 1 − alpha/2)` quantile bands of `d × m` fields.
fn pointwise_bands(fields: &[FdMatrix], level: f64) -> (FdMatrix, FdMatrix) {
    let (d, m) = fields[0].shape();
    let alpha = 1.0 - level;
    let mut lo = FdMatrix::zeros(d, m);
    let mut hi = FdMatrix::zeros(d, m);
    for k in 0..d {
        for j in 0..m {
            let mut vals: Vec<f64> = fields.iter().map(|f| f[(k, j)]).collect();
            vals.sort_by(f64::total_cmp);
            lo[(k, j)] = quantile_sorted(&vals, alpha / 2.0);
            hi[(k, j)] = quantile_sorted(&vals, 1.0 - alpha / 2.0);
        }
    }
    (lo, hi)
}

/// Bootstrap confidence region for the Karcher mean shape of R^d curves.
///
/// Resamples the curves with replacement (`n_bootstrap` times, resample `b`
/// seeded with `seed + b`), computes the Karcher mean of each resample under
/// `shape`, aligns it to the full-sample mean and reports (a) the confidence
/// ball of geodesic radius `radius` (the `confidence_level` quantile of the
/// bootstrap distances) around the mean shape and (b) pointwise coordinate
/// bands of the aligned bootstrap means. Resamples run in parallel with the
/// `parallel` feature; results do not depend on the thread count.
///
/// This is the bootstrap counterpart of the 1-D
/// [`shape_confidence_interval`](crate::alignment::shape_confidence_interval).
/// fdasrvf `shape_CI` instead samples shapes from the tangent Gaussian model at
/// the Karcher median and reports the `(1 − a)` box plot quantile curves; use
/// [`curve_shape_boxplot`] for that kind of summary.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 3 curves or invalid shapes;
/// [`FdarError::InvalidParameter`] for `confidence_level ∉ (0, 1)`,
/// `n_bootstrap = 0` or an invalid shape configuration;
/// [`FdarError::ComputationFailed`] for a zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_confidence_interval, curves_to_set, CurveShapeCiConfig, CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 25;
/// let curves: Vec<FdMatrix> = (0..6)
///     .map(|i| {
///         let a = 0.25 + 0.03 * i as f64;
///         let mut b = FdMatrix::zeros(2, m);
///         for j in 0..m {
///             let t = j as f64 / (m - 1) as f64;
///             b[(0, j)] = t;
///             b[(1, j)] = a * (std::f64::consts::PI * t).sin();
///         }
///         b
///     })
///     .collect();
/// let mut ci = CurveShapeCiConfig::default();
/// ci.n_bootstrap = 10;
/// let res = curve_shape_confidence_interval(
///     &curves_to_set(&curves).unwrap(),
///     &CurveShapeConfig::default(),
///     &ci,
/// )
/// .unwrap();
/// assert_eq!(res.bootstrap_means.len(), 10);
/// assert!(res.radius >= 0.0);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_confidence_interval(
    curves: &FdCurveSet,
    shape: &CurveShapeConfig,
    config: &CurveShapeCiConfig,
) -> Result<CurveShapeCiResult, FdarError> {
    validate_ci(curves, config)?;
    let full = curve_shape_karcher_mean(curves, shape, config.max_iter, config.tol)?;
    let n = curves.ncurves();
    let list: Vec<FdMatrix> = (0..n)
        .map(|i| curve_from_set(curves, i))
        .collect::<Result<_, _>>()?;
    let boots: Vec<(FdMatrix, f64)> = iter_maybe_parallel!(0..config.n_bootstrap)
        .map(|b| {
            let mut rng = seed_for_thread(config.seed, b);
            let sample: Vec<FdMatrix> = (0..n).map(|_| list[rng.gen_range(0..n)].clone()).collect();
            let km = curve_shape_karcher_mean(
                &curves_to_set(&sample)?,
                shape,
                config.max_iter,
                config.tol,
            )?;
            let al = curve_shape_align_pair(&full.mean, &km.mean, shape)?;
            Ok((al.beta2_aligned, al.distance))
        })
        .collect::<Result<_, FdarError>>()?;
    let (bootstrap_means, bootstrap_distances): (Vec<FdMatrix>, Vec<f64>) =
        boots.into_iter().unzip();
    let mut sorted = bootstrap_distances.clone();
    sorted.sort_by(f64::total_cmp);
    let radius = quantile_sorted(&sorted, config.confidence_level);
    let (lower_band, upper_band) = pointwise_bands(&bootstrap_means, config.confidence_level);
    Ok(CurveShapeCiResult {
        mean: full,
        bootstrap_means,
        bootstrap_distances,
        radius,
        lower_band,
        upper_band,
    })
}
