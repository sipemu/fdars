//! fdasrvf-compatible model-based shape confidence region (`shape_CI`,
//! `sample_shapes`) for R^d curves (GH #71).

use rand::Rng;
use rand_distr::StandardNormal;

use super::boxplot::{boxplot_impl, pointwise_bands};
use super::geometry::{self, Grid};
use super::karcher::{curve_shape_karcher_median, curve_shape_pca, exp_map};
use super::{
    curves_to_set, prepare, srvf_distance, transform_to_curve, CurveShapeBoxplot,
    CurveShapeBoxplotConfig, CurveShapeCiConfig, CurveShapeCiResult, CurveShapeConfig,
    CurveShapeKarcherResult, CurveShapePcaResult,
};
use crate::elastic_fpca::horiz_fpca_from_alignment;
use crate::error::FdarError;
use crate::helpers::{quantile_sorted, seed_for_thread};
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
use crate::warping::{exp_map_sphere, invert_gamma, psi_to_gam};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Geodesic steps used by fdasrvf `sample_shapes` for closed curves (`N − 1`
/// with `N = 5`); open curves use one exact exponential-map step.
const CLOSED_STEPS: usize = 4;

/// Outputs of the model-based (fdasrvf `shape_CI`) confidence region, see
/// [`CurveShapeCiMethod::ModelBased`](super::CurveShapeCiMethod::ModelBased).
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_confidence_interval, curves_to_set, CurveShapeCiConfig, CurveShapeCiMethod,
///     CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 25;
/// let curves: Vec<FdMatrix> = (0..6)
///     .map(|i| {
///         let a = 0.2 + 0.04 * i as f64;
///         let c = 0.03 * (i as f64 - 2.5);
///         let mut b = FdMatrix::zeros(2, m);
///         for j in 0..m {
///             let t = j as f64 / (m - 1) as f64;
///             b[(0, j)] = t;
///             b[(1, j)] = a * (std::f64::consts::PI * t).sin()
///                 + c * (2.0 * std::f64::consts::PI * t).sin();
///         }
///         b
///     })
///     .collect();
/// let mut shape = CurveShapeConfig::default();
/// shape.rotation = false; // fdasrvf shape_CI(rotated = FALSE)
/// let mut ci = CurveShapeCiConfig::default();
/// ci.method = CurveShapeCiMethod::ModelBased;
/// ci.n_samples = 20;
/// let res = curve_shape_confidence_interval(&curves_to_set(&curves).unwrap(), &shape, &ci)
///     .unwrap();
/// let model = res.model.unwrap();
/// assert_eq!(model.samples.ncurves(), 20);
/// assert_eq!(model.q1a.shape(), (2, m)); // fdasrvf Q1a
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeModelCi {
    /// Lower `(1 − alpha)` box-plot quantile curve of the posed samples
    /// (fdasrvf `Q1a`, `d × m`).
    pub q1a: FdMatrix,
    /// Upper `(1 − alpha)` box-plot quantile curve (fdasrvf `Q3a`).
    pub q3a: FdMatrix,
    /// Box plot of the posed samples around the median (fdasrvf `box`).
    pub boxplot: CurveShapeBoxplot,
    /// Shape PCA at the median whose leading components define the model.
    pub pca: CurveShapePcaResult,
    /// Sampled principal scores `z_k √λ_k` (`n_samples × ncomp`).
    pub scores: FdMatrix,
    /// Sampled shapes in the median's frame, centred, unit length (fdasrvf
    /// `betans`).
    pub shape_samples: FdCurveSet,
    /// SRVFs of `shape_samples` (fdasrvf `qns`).
    pub shape_samples_srvf: FdCurveSet,
    /// Posed samples (random rotation, then random warp; fdasrvf `betas`), the
    /// input of the box plot.
    pub samples: FdCurveSet,
    /// SRVFs of the posed samples (fdasrvf `qs`).
    pub samples_srvf: FdCurveSet,
    /// Applied warps `γ⁻¹` (`n_samples × m`; identity if `reparam = false`),
    /// posed sample = `(R·β)∘γ⁻¹` (fdasrvf `gams`).
    pub warps: FdMatrix,
    /// Applied rotations (`d × d`; identity if `rotation = false`).
    pub rotations: Vec<FdMatrix>,
}

/// One model sample.
struct Draw {
    scores: Vec<f64>,
    shape: FdMatrix,
    shape_q: FdMatrix,
    posed: FdMatrix,
    posed_q: FdMatrix,
    warp: Vec<f64>,
    rotation: FdMatrix,
    distance: f64,
}

/// Phase model from the median's warps (fdasrvf `randomGamma`).
struct WarpModel {
    mu_psi: Vec<f64>,
    dirs: Vec<Vec<f64>>,
    sds: Vec<f64>,
}

/// Rotation-angle model (fdasrvf: mean and sd of the alignment angles).
struct RotationModel {
    mean: f64,
    sd: f64,
}

/// Everything shared by the per-sample draws.
struct Model<'a> {
    median: &'a CurveShapeKarcherResult,
    pca: &'a CurveShapePcaResult,
    warp: Option<WarpModel>,
    rotation: Option<RotationModel>,
    grid: Grid,
}

fn validate_model(
    curves: &FdCurveSet,
    shape: &CurveShapeConfig,
    config: &CurveShapeCiConfig,
) -> Result<(), FdarError> {
    let bad = |parameter: &'static str, message: String| {
        Err(FdarError::InvalidParameter { parameter, message })
    };
    if config.n_samples < 3 {
        return bad(
            "n_samples",
            format!("must be >= 3, got {}", config.n_samples),
        );
    }
    if config.n_components == 0 {
        return bad("n_components", "must be >= 1".to_string());
    }
    if config.n_warp_components == 0 {
        return bad("n_warp_components", "must be >= 1".to_string());
    }
    if !shape.scale {
        return bad(
            "scale",
            "the model-based CI requires scale = true (fdasrvf sample_shapes)".to_string(),
        );
    }
    if shape.rotation && curves.ndim() != 2 {
        return bad(
            "rotation",
            format!(
                "random rotations are only sampled for planar curves (d = 2), got d = {}; \
                 set rotation = false",
                curves.ndim()
            ),
        );
    }
    Ok(())
}

fn warp_model(
    median: &CurveShapeKarcherResult,
    ncomp: usize,
    grid: &Grid,
) -> Result<Option<WarpModel>, FdarError> {
    if !median.config.reparam {
        return Ok(None);
    }
    let h = horiz_fpca_from_alignment(&median.gammas, &grid.t, ncomp)?;
    let m = grid.len();
    let dirs = (0..h.eigenvalues.len())
        .map(|k| (0..m).map(|j| h.eigenfunctions_psi[(k, j)]).collect())
        .collect();
    let sds = h.eigenvalues.iter().map(|&l| l.max(0.0).sqrt()).collect();
    Ok(Some(WarpModel {
        mu_psi: h.mean_psi,
        dirs,
        sds,
    }))
}

fn rotation_model(median: &CurveShapeKarcherResult) -> Option<RotationModel> {
    if !median.config.rotation {
        return None;
    }
    let theta: Vec<f64> = median
        .rotations
        .iter()
        .map(|r| r[(1, 0)].atan2(r[(0, 0)]))
        .collect();
    let n = theta.len() as f64;
    let mean = theta.iter().sum::<f64>() / n;
    let var = theta.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some(RotationModel {
        mean,
        sd: var.sqrt(),
    })
}

/// fdasrvf `sample_vonmises`: rejection sampler on `[0, 2π)` with density
/// `∝ exp((cos(φ − μ) − 1)/σ²)`. A zero spread returns `μ` (fdasrvf would
/// divide by zero).
pub(super) fn sample_von_mises<R: Rng>(mu: f64, sigma: f64, rng: &mut R) -> f64 {
    if sigma.is_nan() || sigma <= 1e-12 {
        return mu;
    }
    loop {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let phi = 2.0 * std::f64::consts::PI * u1;
        if u2 < (((phi - mu).cos() - 1.0) / (sigma * sigma)).exp() {
            return phi;
        }
    }
}

/// fdasrvf `parallel_translate` (mode "C"): transport `w` from `q1` to `q2`,
/// project onto the tangent space at `q2` and restore the transported length.
fn parallel_translate(
    w: &FdMatrix,
    q1: &FdMatrix,
    q2: &FdMatrix,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> FdMatrix {
    let s = geometry::axpby(1.0, q1, 1.0, q2);
    let denom = geometry::inner(&s, &s, grid);
    let wt = if denom > 1e-8 {
        geometry::axpby(1.0, w, -2.0 * geometry::inner(w, q2, grid) / denom, &s)
    } else {
        let c = geometry::inner(w, q2, grid) / geometry::inner(q2, q2, grid);
        geometry::axpby(1.0, w, -c, q2)
    };
    let l = geometry::norm(&wt, grid);
    let wb = geometry::project_tangent(&wt, q2, grid, true, true, cfg.metric.ratio());
    let nb = geometry::norm(&wb, grid);
    if nb > 1e-4 {
        geometry::scaled(&wb, l / nb)
    } else {
        wb
    }
}

/// Shoot the geodesic from `mu` with initial velocity `v` (fdasrvf
/// `sample_shapes`): one exact step for open curves; for closed curves
/// [`CLOSED_STEPS`] steps of length `‖v‖/4`, each projected onto the closed
/// manifold and followed by parallel translation of the velocity.
fn shoot(mu: &FdMatrix, v: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    if !cfg.closed {
        return exp_map(mu, v, cfg, grid);
    }
    let eps = 1.0 / CLOSED_STEPS as f64;
    let mut q1 = mu.clone();
    let mut w = v.clone();
    for _ in 0..CLOSED_STEPS {
        if geometry::norm(&w, grid) < 1e-4 {
            break;
        }
        let q2 = exp_map(&q1, &geometry::scaled(&w, eps), cfg, grid);
        w = parallel_translate(&w, &q1, &q2, cfg, grid);
        q1 = q2;
    }
    q1
}

/// Random warp `γ⁻¹` of the phase model (identity without one).
fn draw_warp<R: Rng>(model: &Option<WarpModel>, grid: &Grid, rng: &mut R) -> Vec<f64> {
    let Some(wm) = model else {
        return grid.t.clone();
    };
    let mut v = vec![0.0; grid.len()];
    for (dir, &sd) in wm.dirs.iter().zip(&wm.sds) {
        let z: f64 = rng.sample(StandardNormal);
        for (vj, &dj) in v.iter_mut().zip(dir) {
            *vj += z * sd * dj;
        }
    }
    let psi = exp_map_sphere(&wm.mu_psi, &v, &grid.t);
    invert_gamma(&psi_to_gam(&psi, &grid.t), &grid.t)
}

fn draw_rotation<R: Rng>(model: &Option<RotationModel>, d: usize, rng: &mut R) -> FdMatrix {
    let Some(rm) = model else {
        return geometry::identity(d);
    };
    let theta = sample_von_mises(rm.mean, rm.sd, rng);
    let (s, c) = theta.sin_cos();
    let mut r = FdMatrix::zeros(2, 2);
    r[(0, 0)] = c;
    r[(1, 0)] = s;
    r[(0, 1)] = -s;
    r[(1, 1)] = c;
    r
}

/// Draw model sample `i` (rng `seed + i`): shape scores, then warp scores, then
/// the rotation angle.
fn draw(model: &Model<'_>, seed: u64, i: usize) -> Result<Draw, FdarError> {
    let mut rng = seed_for_thread(seed, i);
    let (mu, cfg, grid) = (&model.median.mean_srvf, &model.median.config, &model.grid);
    let mut v = FdMatrix::zeros(mu.nrows(), mu.ncols());
    let mut scores = Vec::with_capacity(model.pca.components.len());
    for (phi, &lam) in model.pca.components.iter().zip(&model.pca.eigenvalues) {
        let z: f64 = rng.sample(StandardNormal);
        let s = z * lam.max(0.0).sqrt();
        scores.push(s);
        v = geometry::axpby(1.0, &v, s, phi);
    }
    let q = shoot(mu, &v, cfg, grid);
    let shape = prepare(&transform_to_curve(&q, cfg, grid), cfg, grid)?;
    let warp = draw_warp(&model.warp, grid, &mut rng);
    let rotation = draw_rotation(&model.rotation, mu.nrows(), &mut rng);
    let rotated = geometry::rotate(&rotation, &shape.beta);
    let posed = prepare(&geometry::warp_curve(&rotated, grid, &warp), cfg, grid)?;
    Ok(Draw {
        scores,
        distance: srvf_distance(mu, &shape.q, cfg, grid),
        shape: shape.beta,
        shape_q: shape.q,
        posed: posed.beta,
        posed_q: posed.q,
        warp,
        rotation,
    })
}

/// Collect the draws into the model output.
fn assemble(
    draws: Vec<Draw>,
    pca: CurveShapePcaResult,
    median: &CurveShapeKarcherResult,
    level: f64,
) -> Result<(CurveShapeModelCi, Vec<FdMatrix>, Vec<f64>), FdarError> {
    let n = draws.len();
    let m = median.mean_srvf.ncols();
    let ncomp = pca.components.len();
    let mut scores = FdMatrix::zeros(n, ncomp);
    let mut warps = FdMatrix::zeros(n, m);
    for (i, dr) in draws.iter().enumerate() {
        for (k, &s) in dr.scores.iter().enumerate() {
            scores[(i, k)] = s;
        }
        for (j, &g) in dr.warp.iter().enumerate() {
            warps[(i, j)] = g;
        }
    }
    let collect = |f: fn(&Draw) -> &FdMatrix| -> Result<FdCurveSet, FdarError> {
        curves_to_set(&draws.iter().map(|d| f(d).clone()).collect::<Vec<_>>())
    };
    let samples = collect(|d| &d.posed)?;
    let samples_srvf = collect(|d| &d.posed_q)?;
    let box_cfg = CurveShapeBoxplotConfig {
        alpha: 1.0 - level,
        k_a: 1.0,
    };
    let boxplot = boxplot_impl(
        &median.mean_srvf,
        &samples_srvf,
        &samples,
        &median.config,
        &box_cfg,
    )?;
    let model = CurveShapeModelCi {
        q1a: boxplot.q1a.clone(),
        q3a: boxplot.q3a.clone(),
        boxplot,
        pca,
        scores,
        shape_samples: collect(|d| &d.shape)?,
        shape_samples_srvf: collect(|d| &d.shape_q)?,
        samples,
        samples_srvf,
        warps,
        rotations: draws.iter().map(|d| d.rotation.clone()).collect(),
    };
    let shapes = draws.iter().map(|d| d.shape.clone()).collect();
    let distances = draws.iter().map(|d| d.distance).collect();
    Ok((model, shapes, distances))
}

/// Model-based (fdasrvf `shape_CI`) body of
/// [`curve_shape_confidence_interval`](super::curve_shape_confidence_interval).
pub(super) fn model_based_ci(
    curves: &FdCurveSet,
    shape: &CurveShapeConfig,
    config: &CurveShapeCiConfig,
) -> Result<CurveShapeCiResult, FdarError> {
    validate_model(curves, shape, config)?;
    let median = curve_shape_karcher_median(curves, shape, config.max_iter, config.tol)?;
    let pca = curve_shape_pca(&median, config.n_components)?;
    let grid = Grid::new(curves.npoints());
    let model = Model {
        median: &median,
        pca: &pca,
        warp: warp_model(&median, config.n_warp_components, &grid)?,
        rotation: rotation_model(&median),
        grid,
    };
    let draws: Vec<Draw> = iter_maybe_parallel!(0..config.n_samples)
        .map(|i| draw(&model, config.seed, i))
        .collect::<Result<_, FdarError>>()?;
    let (model_out, shapes, distances) =
        assemble(draws, pca.clone(), &median, config.confidence_level)?;
    let mut sorted = distances.clone();
    sorted.sort_by(f64::total_cmp);
    let radius = quantile_sorted(&sorted, config.confidence_level);
    let (lower_band, upper_band) = pointwise_bands(&shapes, config.confidence_level);
    Ok(CurveShapeCiResult {
        mean: median,
        bootstrap_means: shapes,
        bootstrap_distances: distances,
        radius,
        lower_band,
        upper_band,
        model: Some(model_out),
    })
}
