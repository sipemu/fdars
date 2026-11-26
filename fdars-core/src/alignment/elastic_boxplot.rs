//! Elastic amplitude and phase box plots with new-function classification.
//!
//! Ports of fdasrsf's `ampbox` / `phbox` (fdasrsf 2.7.5 `boxplots.py`,
//! fdasrvf `AmplitudeBoxplot` / `PhaseBoxplot`), following Xie, Kurtek,
//! Bharath & Sun (2017).
//!
//! * The **amplitude box plot** ranks aligned SRSFs by their L² distance to
//!   the median SRSF. The 50% and `(1-alpha)` central regions are the
//!   `ceil(n/2)` and `round(n(1-alpha))` closest curves. Inside each region
//!   the quartile pair (Q1, Q3) maximises an energy that rewards large
//!   distances and opposite directions from the median. The whisker distance
//!   extends Q1 and Q3 by `k_a · IQR` along their directions, and curves
//!   farther than the whisker distance are outliers.
//! * The **phase box plot** does the same for warping functions on the
//!   Hilbert sphere `ψ = √γ̇`. The median is the Karcher median warp, found
//!   by a Weiszfeld iteration (fdasrsf `SqrtMedian`), and distances are
//!   lengths of shooting vectors.
//!
//! Both box plots can **classify new functions** (fdasrsf `ampbox.classify` /
//! `phbox.classify`): a new function's distance to the median puts it in the
//! 50% region, the `(1-alpha)` region, within the whiskers, or among the
//! outliers ([`BoxplotRegion`]), on the Q1 or Q3 side ([`BoxplotSide`]).
//! [`elastic_boxplots`] fits both box plots from raw data (Karcher median
//! alignment) and [`ElasticBoxplots::classify`] aligns new functions to the
//! fitted template before classifying them.
//!
//! # References
//!
//! - Xie, W., Kurtek, S., Bharath, K. & Sun, Y. (2017). A geometric approach
//!   to visualization of variability in functional data. *Journal of the
//!   American Statistical Association*, 112(519), 979-993.

use super::optim::{optimal_warp, WarpOptimConfig};
use super::robust_karcher::{karcher_median, RobustKarcherConfig};
use super::srsf::{reparameterize_curve, srsf_transform};
use crate::error::FdarError;
use crate::helpers::trapz;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Configuration for the elastic amplitude / phase box plots.
///
/// Construct via `ElasticBoxplotConfig::default()`, then assign the fields
/// you need. This struct is `#[non_exhaustive]`, so external crates cannot
/// build it with a struct literal.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ElasticBoxplotConfig {
    /// Quantile level: the outer central region holds `round(n(1-alpha))`
    /// curves (fdasrsf default 0.05, i.e. a 95% region). Must be in (0, 1).
    pub alpha: f64,
    /// Whisker scale `k_a`: whiskers extend Q1/Q3 by `k_a · IQR`
    /// (fdasrsf default 1.0). Must be positive.
    pub k_a: f64,
}

impl Default for ElasticBoxplotConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            k_a: 1.0,
        }
    }
}

/// Box plot region a function falls into (fdasrsf `boxplot_region` levels 1–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoxplotRegion {
    /// Inside the 50% central region (level 1).
    Central50,
    /// Inside the `(1-alpha)` central region but not the 50% region (level 2).
    CentralAlpha,
    /// Outside the central regions but within the whiskers (level 3).
    WithinWhiskers,
    /// Beyond the whiskers (level 4).
    Outlier,
}

impl BoxplotRegion {
    /// fdasrsf numeric level: 1 = 50% region, 2 = `(1-alpha)` region,
    /// 3 = within whiskers, 4 = outlier.
    #[must_use]
    pub fn level(self) -> u8 {
        match self {
            Self::Central50 => 1,
            Self::CentralAlpha => 2,
            Self::WithinWhiskers => 3,
            Self::Outlier => 4,
        }
    }
}

/// Direction from the median that a function lies in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BoxplotSide {
    /// Inner product with the Q3 direction is negative.
    Q1,
    /// Inner product with the Q3 direction is non-negative.
    Q3,
}

/// Classification of functions against a fitted elastic box plot.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct BoxplotClassification {
    /// Distance of each function to the median (amplitude: SRSF L² distance;
    /// phase: shooting-vector length on the Hilbert sphere).
    pub distances: Vec<f64>,
    /// Region of each function.
    pub regions: Vec<BoxplotRegion>,
    /// Quartile side of each function (sign of the inner product with Q3).
    pub sides: Vec<BoxplotSide>,
}

/// Elastic amplitude box plot (fdasrsf `ampbox`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct AmplitudeBoxplot {
    /// Median function.
    pub median: Vec<f64>,
    /// Median SRSF.
    pub median_srsf: Vec<f64>,
    /// First amplitude quartile function.
    pub q1: Vec<f64>,
    /// Third amplitude quartile function.
    pub q3: Vec<f64>,
    /// Lower `alpha` quantile function.
    pub q1a: Vec<f64>,
    /// Upper `alpha` quantile function.
    pub q3a: Vec<f64>,
    /// Minimum extreme (non-outlier function closest to the lower whisker).
    pub min_extreme: Vec<f64>,
    /// Maximum extreme (non-outlier function closest to the upper whisker).
    pub max_extreme: Vec<f64>,
    /// SRSF of the Q1 function.
    pub q1_srsf: Vec<f64>,
    /// SRSF of the Q3 function.
    pub q3_srsf: Vec<f64>,
    /// Lower whisker SRSF.
    pub lower_whisker_srsf: Vec<f64>,
    /// Upper whisker SRSF.
    pub upper_whisker_srsf: Vec<f64>,
    /// Row index of Q1.
    pub q1_index: usize,
    /// Row index of Q3.
    pub q3_index: usize,
    /// Row index of the lower `alpha` quantile.
    pub q1a_index: usize,
    /// Row index of the upper `alpha` quantile.
    pub q3a_index: usize,
    /// Row index of the minimum extreme.
    pub min_index: usize,
    /// Row index of the maximum extreme.
    pub max_index: usize,
    /// Amplitude distance of each training function to the median SRSF.
    pub distances: Vec<f64>,
    /// Distance bounding the 50% central region.
    pub r50: f64,
    /// Distance bounding the `(1-alpha)` central region.
    pub ralpha: f64,
    /// Whisker (outlier cut-off) distance.
    pub whisker_distance: f64,
    /// Outlier row indices, ordered by decreasing distance.
    pub outlier_indices: Vec<usize>,
    /// Quantile level used.
    pub alpha: f64,
    /// Whisker scale used.
    pub k_a: f64,
}

/// Elastic phase box plot on the Hilbert sphere (fdasrsf `phbox`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticPhaseBoxplot {
    /// Karcher median warping function (on the `argvals` domain).
    pub median: Vec<f64>,
    /// Median warp on the sphere, `ψ = √γ̇` on the unit-interval grid.
    pub median_psi: Vec<f64>,
    /// First phase quartile warp.
    pub q1: Vec<f64>,
    /// Third phase quartile warp.
    pub q3: Vec<f64>,
    /// Lower `alpha` quantile warp.
    pub q1a: Vec<f64>,
    /// Upper `alpha` quantile warp.
    pub q3a: Vec<f64>,
    /// Minimum extreme warp.
    pub min_extreme: Vec<f64>,
    /// Maximum extreme warp.
    pub max_extreme: Vec<f64>,
    /// Shooting vectors from the median to each warp (n × m).
    pub shooting_vectors: FdMatrix,
    /// Row index of Q1.
    pub q1_index: usize,
    /// Row index of Q3.
    pub q3_index: usize,
    /// Row index of the lower `alpha` quantile.
    pub q1a_index: usize,
    /// Row index of the upper `alpha` quantile.
    pub q3a_index: usize,
    /// Row index of the minimum extreme.
    pub min_index: usize,
    /// Row index of the maximum extreme.
    pub max_index: usize,
    /// Phase distance of each training warp to the median.
    pub distances: Vec<f64>,
    /// Distance bounding the 50% central region.
    pub r50: f64,
    /// Distance bounding the `(1-alpha)` central region.
    pub ralpha: f64,
    /// Whisker (outlier cut-off) distance.
    pub whisker_distance: f64,
    /// Outlier row indices, ordered by decreasing distance.
    pub outlier_indices: Vec<usize>,
    /// Quantile level used.
    pub alpha: f64,
    /// Whisker scale used.
    pub k_a: f64,
    /// Number of Weiszfeld iterations of the median.
    pub median_iter: usize,
}

/// Amplitude and phase box plots fitted from raw data, with the alignment
/// template needed to classify new functions.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticBoxplots {
    /// Amplitude box plot.
    pub amplitude: AmplitudeBoxplot,
    /// Phase box plot of the alignment warps.
    pub phase: ElasticPhaseBoxplot,
    /// Template SRSF (Karcher median SRSF) new functions are aligned to.
    pub template_srsf: Vec<f64>,
    /// Aligned training functions (n × m).
    pub aligned_data: FdMatrix,
    /// Aligned training SRSFs (n × m).
    pub aligned_srsfs: FdMatrix,
    /// Training warps (n × m).
    pub gammas: FdMatrix,
    /// Alignment roughness penalty used.
    pub lambda: f64,
}

/// Classification of new functions against [`ElasticBoxplots`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticBoxplotClassification {
    /// Amplitude box plot classification.
    pub amplitude: BoxplotClassification,
    /// Phase box plot classification.
    pub phase: BoxplotClassification,
    /// New functions aligned to the template (k × m).
    pub aligned_data: FdMatrix,
    /// SRSFs of the aligned new functions (k × m).
    pub aligned_srsfs: FdMatrix,
    /// Warps aligning the new functions to the template (k × m).
    pub gammas: FdMatrix,
}

// ─── Shared helpers ─────────────────────────────────────────────────────────

/// Region from a distance and the three box plot edges (fdasrsf
/// `boxplot_region`; the whisker test runs last, as there).
fn region_of(dist: f64, r50: f64, ralpha: f64, whisker: f64) -> BoxplotRegion {
    if dist > whisker {
        BoxplotRegion::Outlier
    } else if dist <= r50 {
        BoxplotRegion::Central50
    } else if dist <= ralpha {
        BoxplotRegion::CentralAlpha
    } else {
        BoxplotRegion::WithinWhiskers
    }
}

fn side_of(ip: f64) -> BoxplotSide {
    if ip >= 0.0 {
        BoxplotSide::Q3
    } else {
        BoxplotSide::Q1
    }
}

fn validate_config(
    n: usize,
    m: usize,
    argvals: &[f64],
    cfg: &ElasticBoxplotConfig,
) -> Result<(), FdarError> {
    if n < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }
    if m < 3 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m} (m >= 3)"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if !(cfg.alpha > 0.0 && cfg.alpha < 1.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "alpha",
            message: format!("must be in (0, 1), got {}", cfg.alpha),
        });
    }
    if !(cfg.k_a > 0.0 && cfg.k_a.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "k_a",
            message: format!("must be positive and finite, got {}", cfg.k_a),
        });
    }
    Ok(())
}

fn l2_norm(v: &[f64], t: &[f64]) -> f64 {
    let sq: Vec<f64> = v.iter().map(|x| x * x).collect();
    trapz(&sq, t).max(0.0).sqrt()
}

fn l2_inner(a: &[f64], b: &[f64], t: &[f64]) -> f64 {
    let p: Vec<f64> = a.iter().zip(b).map(|(x, y)| x * y).collect();
    trapz(&p, t)
}

/// `numpy.argsort` (stable) of a distance vector.
fn argsort(d: &[f64]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..d.len()).collect();
    idx.sort_by(|&a, &b| d[a].partial_cmp(&d[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx
}

/// Quartile-pair search inside a central region (fdasrsf energy criterion,
/// `lam = 0.5`). `dirs[i]` is the direction of curve `i` from the median and
/// `dist[i]` its distance. Returns `(row index, col index)` into `region`,
/// reproducing `numpy.argmax` over the full (zero-initialised) energy matrix.
fn quartile_pair(region: &[usize], dirs: &[Vec<f64>], dist: &[f64], t: &[f64]) -> (usize, usize) {
    const LAM: f64 = 0.5;
    let r = region.len();
    let rmax = region
        .iter()
        .map(|&i| dist[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let unit: Vec<Vec<f64>> = region
        .iter()
        .map(|&i| {
            let nrm = l2_norm(&dirs[i], t);
            let s = if nrm > 0.0 { 1.0 / nrm } else { 0.0 };
            dirs[i].iter().map(|v| v * s).collect()
        })
        .collect();
    let mut best = (0usize, 0usize);
    let mut best_e = 0.0; // energy[0, 0] == 0 (the matrix is zero-initialised)
    for i in 0..r {
        for j in 0..r {
            let e = if j > i {
                let angle = l2_inner(&unit[i], &unit[j], t);
                (1.0 - LAM) * (dist[region[i]] / rmax + dist[region[j]] / rmax)
                    - LAM * (angle + 1.0)
            } else {
                0.0
            };
            if e > best_e {
                best_e = e;
                best = (i, j);
            }
        }
    }
    best
}

/// Indices and radii of the 50% and `(1-alpha)` central regions plus their
/// quartile pairs, shared by the amplitude and phase box plots.
struct CentralRegions {
    order: Vec<usize>,
    r50: f64,
    ralpha: f64,
    q1: usize,
    q3: usize,
    q1a: usize,
    q3a: usize,
}

fn central_regions(dist: &[f64], dirs: &[Vec<f64>], t: &[f64], alpha: f64) -> CentralRegions {
    let n = dist.len();
    let order = argsort(dist);
    let n50 = n.div_ceil(2);
    let na = ((n as f64 * (1.0 - alpha)).round_ties_even() as usize).clamp(1, n);
    let cr50 = &order[..n50];
    let cra = &order[..na];
    let r50 = cr50
        .iter()
        .map(|&i| dist[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let ralpha = cra
        .iter()
        .map(|&i| dist[i])
        .fold(f64::NEG_INFINITY, f64::max);
    let (a, b) = quartile_pair(cr50, dirs, dist, t);
    let (q1, q3) = (cr50[a], cr50[b]);
    let (a, b) = quartile_pair(cra, dirs, dist, t);
    let (mut q1a, mut q3a) = (cra[a], cra[b]);
    // keep the quantile pair oriented like the quartile pair
    if l2_inner(&dirs[q1a], &dirs[q1], t) < 0.0 {
        std::mem::swap(&mut q1a, &mut q3a);
    }
    CentralRegions {
        order,
        r50,
        ralpha,
        q1,
        q3,
        q1a,
        q3a,
    }
}

/// Whisker point `base + k_a · IQR · dir/‖dir‖`.
fn whisker_point(base: &[f64], dir: &[f64], scale: f64, t: &[f64]) -> Vec<f64> {
    let nrm = l2_norm(dir, t);
    let s = if nrm > 0.0 { scale / nrm } else { 0.0 };
    base.iter().zip(dir).map(|(b, d)| b + s * d).collect()
}

/// Outliers (decreasing distance) and the extreme indices: the non-outlier
/// curves closest to the upper / lower whisker points.
fn outliers_and_extremes(
    dist: &[f64],
    order: &[usize],
    whisker: f64,
    pts: &[Vec<f64>],
    upper: &[f64],
    lower: &[f64],
    t: &[f64],
) -> (Vec<usize>, usize, usize) {
    let outliers: Vec<usize> = order
        .iter()
        .rev()
        .copied()
        .filter(|&i| dist[i] > whisker)
        .collect();
    let mut du = vec![f64::INFINITY; dist.len()];
    let mut dl = vec![f64::INFINITY; dist.len()];
    for i in (0..dist.len()).filter(|i| !outliers.contains(i)) {
        let a: Vec<f64> = upper.iter().zip(&pts[i]).map(|(u, p)| u - p).collect();
        let b: Vec<f64> = lower.iter().zip(&pts[i]).map(|(l, p)| l - p).collect();
        du[i] = l2_norm(&a, t);
        dl[i] = l2_norm(&b, t);
    }
    (outliers, first_argmin(&du), first_argmin(&dl))
}

fn first_argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[best] {
            best = i;
        }
    }
    best
}

fn rows(mat: &FdMatrix) -> Vec<Vec<f64>> {
    (0..mat.nrows()).map(|i| mat.row(i)).collect()
}

// ─── Amplitude box plot ─────────────────────────────────────────────────────

/// Build the elastic amplitude box plot from already-aligned data.
///
/// Port of fdasrsf 2.7.5 `ampbox.construct_boxplot`. Distances are
/// `‖q_median − q_i‖` (trapezoidal L² on `argvals`).
///
/// # Arguments
/// * `aligned_data` — Aligned functions (n × m), typically from a Karcher
///   median alignment.
/// * `aligned_srsfs` — SRSFs of the aligned functions (n × m).
/// * `median` — Median function (length m).
/// * `median_srsf` — Median SRSF (length m).
/// * `argvals` — Evaluation grid (length m).
/// * `config` — `alpha` and `k_a`.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if n < 3, m < 3 or any shape disagrees;
/// [`FdarError::InvalidParameter`] for `alpha ∉ (0, 1)` or `k_a ≤ 0`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{amplitude_boxplot_from_aligned, srsf_transform, ElasticBoxplotConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 8;
/// let mut f = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     for j in 0..m {
///         f[(i, j)] = (0.8 + 0.05 * i as f64) * (6.0 * t[j]).sin();
///     }
/// }
/// let q = srsf_transform(&f, &t);
/// let med: Vec<f64> = t.iter().map(|&x| 0.97 * (6.0 * x).sin()).collect();
/// let med_q = srsf_transform(&FdMatrix::from_slice(&med, 1, m).unwrap(), &t).row(0);
/// let bp = amplitude_boxplot_from_aligned(&f, &q, &med, &med_q, &t, &ElasticBoxplotConfig::default()).unwrap();
/// assert!(bp.r50 <= bp.ralpha);
/// let cls = bp.classify_srsfs(&q, &t).unwrap();
/// assert_eq!(cls.distances, bp.distances);
/// ```
#[must_use = "box plot result should not be discarded"]
pub fn amplitude_boxplot_from_aligned(
    aligned_data: &FdMatrix,
    aligned_srsfs: &FdMatrix,
    median: &[f64],
    median_srsf: &[f64],
    argvals: &[f64],
    config: &ElasticBoxplotConfig,
) -> Result<AmplitudeBoxplot, FdarError> {
    let (n, m) = aligned_data.shape();
    validate_config(n, m, argvals, config)?;
    if aligned_srsfs.shape() != (n, m) || median.len() != m || median_srsf.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "aligned_srsfs/median/median_srsf",
            expected: format!("{n} x {m} matrix and length-{m} vectors"),
            actual: format!(
                "{:?}, {}, {}",
                aligned_srsfs.shape(),
                median.len(),
                median_srsf.len()
            ),
        });
    }
    let t = argvals;
    let qs = rows(aligned_srsfs);
    let dirs: Vec<Vec<f64>> = qs
        .iter()
        .map(|q| q.iter().zip(median_srsf).map(|(a, b)| a - b).collect())
        .collect();
    let dist: Vec<f64> = dirs.iter().map(|v| l2_norm(v, t)).collect();
    let cr = central_regions(&dist, &dirs, t, config.alpha);

    let iqr = dist[cr.q1] + dist[cr.q3];
    let upper = whisker_point(&qs[cr.q3], &dirs[cr.q3], config.k_a * iqr, t);
    let lower = whisker_point(&qs[cr.q1], &dirs[cr.q1], config.k_a * iqr, t);
    let dev = |w: &[f64]| -> f64 {
        let d: Vec<f64> = w.iter().zip(median_srsf).map(|(a, b)| a - b).collect();
        l2_norm(&d, t)
    };
    let whisker = dev(&upper).max(dev(&lower));
    let (outliers, max_index, min_index) =
        outliers_and_extremes(&dist, &cr.order, whisker, &qs, &upper, &lower, t);

    Ok(AmplitudeBoxplot {
        median: median.to_vec(),
        median_srsf: median_srsf.to_vec(),
        q1: aligned_data.row(cr.q1),
        q3: aligned_data.row(cr.q3),
        q1a: aligned_data.row(cr.q1a),
        q3a: aligned_data.row(cr.q3a),
        min_extreme: aligned_data.row(min_index),
        max_extreme: aligned_data.row(max_index),
        q1_srsf: qs[cr.q1].clone(),
        q3_srsf: qs[cr.q3].clone(),
        lower_whisker_srsf: lower,
        upper_whisker_srsf: upper,
        q1_index: cr.q1,
        q3_index: cr.q3,
        q1a_index: cr.q1a,
        q3a_index: cr.q3a,
        min_index,
        max_index,
        distances: dist,
        r50: cr.r50,
        ralpha: cr.ralpha,
        whisker_distance: whisker,
        outlier_indices: outliers,
        alpha: config.alpha,
        k_a: config.k_a,
    })
}

impl AmplitudeBoxplot {
    /// Classify already-aligned SRSFs (fdasrsf `ampbox.classify` after its
    /// `align_new` step).
    ///
    /// Each SRSF's distance to the median SRSF is compared with `r50`,
    /// `ralpha` and the whisker distance; the side is the sign of its inner
    /// product with the Q3 direction.
    ///
    /// # Errors
    /// [`FdarError::InvalidDimension`] if the column count or `argvals`
    /// length differs from the fitted grid.
    pub fn classify_srsfs(
        &self,
        aligned_srsfs: &FdMatrix,
        argvals: &[f64],
    ) -> Result<BoxplotClassification, FdarError> {
        let m = self.median_srsf.len();
        check_new_shape(aligned_srsfs.ncols(), argvals.len(), m)?;
        let v3: Vec<f64> = self
            .q3_srsf
            .iter()
            .zip(&self.median_srsf)
            .map(|(a, b)| a - b)
            .collect();
        let tangents: Vec<Vec<f64>> = rows(aligned_srsfs)
            .into_iter()
            .map(|q| {
                q.iter()
                    .zip(&self.median_srsf)
                    .map(|(a, b)| a - b)
                    .collect()
            })
            .collect();
        Ok(classify_tangents(
            &tangents,
            &v3,
            argvals,
            self.r50,
            self.ralpha,
            self.whisker_distance,
        ))
    }
}

fn check_new_shape(ncols: usize, nargs: usize, m: usize) -> Result<(), FdarError> {
    if ncols != m || nargs != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new data / argvals",
            expected: format!("{m} columns and argvals of length {m}"),
            actual: format!("{ncols} columns, argvals of length {nargs}"),
        });
    }
    Ok(())
}

fn classify_tangents(
    tangents: &[Vec<f64>],
    v3: &[f64],
    t: &[f64],
    r50: f64,
    ralpha: f64,
    whisker: f64,
) -> BoxplotClassification {
    let distances: Vec<f64> = tangents.iter().map(|v| l2_norm(v, t)).collect();
    let regions = distances
        .iter()
        .map(|&d| region_of(d, r50, ralpha, whisker))
        .collect();
    let sides = tangents
        .iter()
        .map(|v| side_of(l2_inner(v, v3, t)))
        .collect();
    BoxplotClassification {
        distances,
        regions,
        sides,
    }
}

// ─── Phase box plot ─────────────────────────────────────────────────────────

/// `numpy.gradient` with uniform spacing (second-order interior, first-order edges).
fn np_gradient(y: &[f64], h: f64) -> Vec<f64> {
    let m = y.len();
    let mut g = vec![0.0; m];
    g[0] = (y[1] - y[0]) / h;
    g[m - 1] = (y[m - 1] - y[m - 2]) / h;
    for j in 1..m - 1 {
        g[j] = (y[j + 1] - y[j - 1]) / (2.0 * h);
    }
    g
}

/// `scipy.integrate.cumulative_trapezoid(y, t, initial=0)`.
fn cumtrapz(y: &[f64], t: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; y.len()];
    for k in 1..y.len() {
        out[k] = out[k - 1] + 0.5 * (t[k] - t[k - 1]) * (y[k] + y[k - 1]);
    }
    out
}

/// Log map on the Hilbert sphere with the angle from `atan2` (fdasrsf
/// `geometry.inv_exp_map`). Returns `(v, theta)`.
fn sphere_log(mu: &[f64], psi: &[f64], t: &[f64]) -> (Vec<f64>, f64) {
    let ip = l2_inner(mu, psi, t);
    let w: Vec<f64> = psi.iter().zip(mu).map(|(p, u)| p - ip * u).collect();
    let wn = l2_norm(&w, t);
    let theta = wn.atan2(ip);
    if theta < 1e-10 {
        (vec![0.0; mu.len()], theta)
    } else {
        (w.iter().map(|x| theta / wn * x).collect(), theta)
    }
}

fn sphere_exp(mu: &[f64], v: &[f64], t: &[f64]) -> Vec<f64> {
    let vn = l2_norm(v, t);
    if vn == 0.0 {
        return mu.to_vec();
    }
    mu.iter()
        .zip(v)
        .map(|(p, x)| vn.cos() * p + vn.sin() * x / vn)
        .collect()
}

/// Weiszfeld step direction: `Σ v_k/d_k / Σ 1/d_k`, skipping warps at the
/// current median (`d_k ≈ 0`) where fdasrsf would divide by zero.
fn weiszfeld_direction(vs: &[Vec<f64>], ds: &[f64], m: usize) -> Vec<f64> {
    let mut vbar = vec![0.0; m];
    let mut wsum = 0.0;
    for (v, &d) in vs.iter().zip(ds) {
        if d <= 1e-12 {
            continue;
        }
        wsum += 1.0 / d;
        for j in 0..m {
            vbar[j] += v[j] / d;
        }
    }
    if wsum > 0.0 {
        vbar.iter_mut().for_each(|x| *x /= wsum);
    }
    vbar
}

/// Port of fdasrsf `SqrtMedian`: Karcher median of `psis` on the sphere.
/// Returns `(median_psi, iterations)`.
fn sqrt_median(psis: &[Vec<f64>], t: &[f64]) -> (Vec<f64>, usize) {
    const STEP: f64 = 0.3;
    const MAX_ITER: usize = 501;
    let m = t.len();
    let mut mu = vec![1.0; m];
    let logs = |mu: &[f64]| -> (Vec<Vec<f64>>, Vec<f64>) {
        psis.iter().map(|p| sphere_log(mu, p, t)).unzip()
    };
    let (vs, ds) = logs(&mu);
    let mut vbar = weiszfeld_direction(&vs, &ds, m);
    let mut r = 1;
    while l2_norm(&vbar, t) > 1e-8 && r < MAX_ITER {
        let step: Vec<f64> = vbar.iter().map(|x| STEP * x).collect();
        mu = sphere_exp(&mu, &step, t);
        r += 1;
        let (vs, _) = logs(&mu);
        let ds: Vec<f64> = psis
            .iter()
            .map(|p| l2_inner(&mu, p, t).clamp(-1.0, 1.0).acos())
            .collect();
        vbar = weiszfeld_direction(&vs, &ds, m);
    }
    (mu, r)
}

/// Warps rescaled to the unit interval and their sphere representation.
fn warps_to_psi(gammas: &FdMatrix, argvals: &[f64]) -> Vec<Vec<f64>> {
    let m = gammas.ncols();
    let (t0, t1) = (argvals[0], argvals[m - 1]);
    let h = 1.0 / (m - 1) as f64;
    (0..gammas.nrows())
        .map(|i| {
            let g: Vec<f64> = (0..m).map(|j| (gammas[(i, j)] - t0) / (t1 - t0)).collect();
            np_gradient(&g, h)
                .iter()
                .map(|x| x.max(0.0).sqrt())
                .collect()
        })
        .collect()
}

fn unit_grid(m: usize) -> Vec<f64> {
    (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
}

/// Build the elastic phase box plot of a set of warping functions.
///
/// Port of fdasrsf 2.7.5 `phbox.construct_boxplot`. Warps are rescaled to
/// `[0, 1]`, mapped to the Hilbert sphere `ψ = √γ̇`, and the Karcher median
/// is found by the Weiszfeld iteration of fdasrsf `SqrtMedian` (step 0.3, at
/// most 500 steps). Distances are shooting-vector lengths from the median.
///
/// Deviation: a warp that coincides with the current median estimate
/// (distance 0) is skipped in that Weiszfeld step instead of producing a
/// division by zero.
///
/// # Arguments
/// * `gammas` — Warping functions (n × m) on the `argvals` domain.
/// * `argvals` — Evaluation grid (length m).
/// * `config` — `alpha` and `k_a`.
///
/// # Errors
/// [`FdarError::InvalidDimension`] if n < 3, m < 3 or `argvals` length
/// differs; [`FdarError::InvalidParameter`] for invalid `alpha` / `k_a`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_phase_boxplot, ElasticBoxplotConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 9;
/// let mut g = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let c = 0.1 * (i as f64 - 4.0) + 0.01;
///     for j in 0..m {
///         g[(i, j)] = t[j] + c * t[j] * (1.0 - t[j]);
///     }
/// }
/// let bp = elastic_phase_boxplot(&g, &t, &ElasticBoxplotConfig::default()).unwrap();
/// assert!(bp.r50 <= bp.ralpha);
/// let cls = bp.classify_warps(&g, &t).unwrap();
/// assert_eq!(cls.regions.len(), n);
/// ```
#[must_use = "box plot result should not be discarded"]
pub fn elastic_phase_boxplot(
    gammas: &FdMatrix,
    argvals: &[f64],
    config: &ElasticBoxplotConfig,
) -> Result<ElasticPhaseBoxplot, FdarError> {
    let (n, m) = gammas.shape();
    validate_config(n, m, argvals, config)?;
    let t = unit_grid(m);
    let psis = warps_to_psi(gammas, argvals);
    let (median_psi, median_iter) = sqrt_median(&psis, &t);
    let vs: Vec<Vec<f64>> = psis
        .iter()
        .map(|p| sphere_log(&median_psi, p, &t).0)
        .collect();
    let dist: Vec<f64> = vs.iter().map(|v| l2_norm(v, &t)).collect();
    let cr = central_regions(&dist, &vs, &t, config.alpha);

    let iqr = dist[cr.q1] + dist[cr.q3];
    let upper = whisker_point(&vs[cr.q3], &vs[cr.q3], config.k_a * iqr, &t);
    let lower = whisker_point(&vs[cr.q1], &vs[cr.q1], config.k_a * iqr, &t);
    let whisker = l2_norm(&upper, &t).max(l2_norm(&lower, &t));
    let (outliers, max_index, min_index) =
        outliers_and_extremes(&dist, &cr.order, whisker, &vs, &upper, &lower, &t);

    let sq: Vec<f64> = median_psi.iter().map(|p| p * p).collect();
    let g = cumtrapz(&sq, &t);
    let (gmin, gmax) = g
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
            (a.min(x), b.max(x))
        });
    let (t0, t1) = (argvals[0], argvals[m - 1]);
    let median: Vec<f64> = g
        .iter()
        .map(|x| t0 + (x - gmin) / (gmax - gmin) * (t1 - t0))
        .collect();

    let mut shooting_vectors = FdMatrix::zeros(n, m);
    for (i, v) in vs.iter().enumerate() {
        for j in 0..m {
            shooting_vectors[(i, j)] = v[j];
        }
    }
    Ok(ElasticPhaseBoxplot {
        median,
        median_psi,
        q1: gammas.row(cr.q1),
        q3: gammas.row(cr.q3),
        q1a: gammas.row(cr.q1a),
        q3a: gammas.row(cr.q3a),
        min_extreme: gammas.row(min_index),
        max_extreme: gammas.row(max_index),
        shooting_vectors,
        q1_index: cr.q1,
        q3_index: cr.q3,
        q1a_index: cr.q1a,
        q3a_index: cr.q3a,
        min_index,
        max_index,
        distances: dist,
        r50: cr.r50,
        ralpha: cr.ralpha,
        whisker_distance: whisker,
        outlier_indices: outliers,
        alpha: config.alpha,
        k_a: config.k_a,
        median_iter,
    })
}

impl ElasticPhaseBoxplot {
    /// Classify new warping functions (fdasrsf `phbox.classify` after its
    /// `align_new` step).
    ///
    /// # Errors
    /// [`FdarError::InvalidDimension`] if the column count or `argvals`
    /// length differs from the fitted grid.
    pub fn classify_warps(
        &self,
        gammas: &FdMatrix,
        argvals: &[f64],
    ) -> Result<BoxplotClassification, FdarError> {
        let m = self.median_psi.len();
        check_new_shape(gammas.ncols(), argvals.len(), m)?;
        let t = unit_grid(m);
        let v3 = self.shooting_vectors.row(self.q3_index);
        let tangents: Vec<Vec<f64>> = warps_to_psi(gammas, argvals)
            .iter()
            .map(|p| sphere_log(&self.median_psi, p, &t).0)
            .collect();
        Ok(classify_tangents(
            &tangents,
            &v3,
            &t,
            self.r50,
            self.ralpha,
            self.whisker_distance,
        ))
    }
}

// ─── Combined fit from raw data ─────────────────────────────────────────────

/// Fit elastic amplitude and phase box plots from raw functional data.
///
/// Aligns the curves to their Karcher median ([`karcher_median`], as
/// fdasrsf requires `srsf_align(method="median")` for `ampbox`/`phbox`),
/// then builds [`amplitude_boxplot_from_aligned`] on the aligned functions
/// and [`elastic_phase_boxplot`] on the warps.
///
/// # Arguments
/// * `data` — Functional data (n × m).
/// * `argvals` — Evaluation grid (length m).
/// * `lambda` — Alignment roughness penalty (0 = none).
/// * `config` — Box plot configuration.
///
/// # Errors
/// As [`amplitude_boxplot_from_aligned`] / [`karcher_median`].
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{elastic_boxplots, ElasticBoxplotConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let n = 8;
/// let mut f = FdMatrix::zeros(n, m);
/// for i in 0..n {
///     let s = 0.03 * (i as f64 - 3.5);
///     for j in 0..m {
///         f[(i, j)] = (1.0 + 0.05 * i as f64) * (std::f64::consts::PI * 2.0 * (t[j] + s)).sin();
///     }
/// }
/// let bp = elastic_boxplots(&f, &t, 0.0, &ElasticBoxplotConfig::default()).unwrap();
/// let cls = bp.classify(&f, &t).unwrap();
/// assert_eq!(cls.amplitude.regions.len(), n);
/// assert_eq!(cls.phase.regions.len(), n);
/// assert!(bp.amplitude.r50 <= bp.amplitude.ralpha);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_boxplots(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
    config: &ElasticBoxplotConfig,
) -> Result<ElasticBoxplots, FdarError> {
    let (n, m) = data.shape();
    validate_config(n, m, argvals, config)?;
    let kcfg = RobustKarcherConfig {
        lambda,
        ..RobustKarcherConfig::default()
    };
    let med = karcher_median(data, argvals, &kcfg)?;
    let aligned_data = med.aligned_data.clone();
    let gammas = med.gammas.clone();
    let aligned_srsfs = srsf_transform(&aligned_data, argvals);
    let amplitude = amplitude_boxplot_from_aligned(
        &aligned_data,
        &aligned_srsfs,
        &med.mean,
        &med.mean_srsf,
        argvals,
        config,
    )?;
    let phase = elastic_phase_boxplot(&gammas, argvals, config)?;
    Ok(ElasticBoxplots {
        amplitude,
        phase,
        template_srsf: med.mean_srsf,
        aligned_data,
        aligned_srsfs,
        gammas,
        lambda,
    })
}

impl ElasticBoxplots {
    /// Classify new functions (fdasrsf `ampbox.classify` + `phbox.classify`).
    ///
    /// Each new function is aligned to the fitted template SRSF with the
    /// fitted `lambda`; the aligned SRSF is placed in the amplitude box plot
    /// and the alignment warp in the phase box plot.
    ///
    /// # Errors
    /// [`FdarError::InvalidDimension`] if `new_data` / `argvals` do not match
    /// the fitted grid.
    #[must_use = "expensive computation whose result should not be discarded"]
    pub fn classify(
        &self,
        new_data: &FdMatrix,
        argvals: &[f64],
    ) -> Result<ElasticBoxplotClassification, FdarError> {
        let m = self.template_srsf.len();
        check_new_shape(new_data.ncols(), argvals.len(), m)?;
        let k = new_data.nrows();
        let q_new = srsf_transform(new_data, argvals);
        let cfg = WarpOptimConfig::classic_dp(self.lambda);
        let warps: Vec<Vec<f64>> = iter_maybe_parallel!(0..k)
            .map(|i| optimal_warp(&self.template_srsf, &q_new.row(i), argvals, &cfg, None))
            .collect();
        let mut gammas = FdMatrix::zeros(k, m);
        let mut aligned_data = FdMatrix::zeros(k, m);
        for (i, g) in warps.iter().enumerate() {
            let fw = reparameterize_curve(&new_data.row(i), argvals, g);
            for j in 0..m {
                gammas[(i, j)] = g[j];
                aligned_data[(i, j)] = fw[j];
            }
        }
        let aligned_srsfs = srsf_transform(&aligned_data, argvals);
        Ok(ElasticBoxplotClassification {
            amplitude: self.amplitude.classify_srsfs(&aligned_srsfs, argvals)?,
            phase: self.phase.classify_warps(&gammas, argvals)?,
            aligned_data,
            aligned_srsfs,
            gammas,
        })
    }
}

#[cfg(test)]
#[path = "elastic_boxplot_tests.rs"]
mod tests;
