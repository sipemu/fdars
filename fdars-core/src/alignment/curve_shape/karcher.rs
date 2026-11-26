//! Karcher mean, covariance and shape PCA of R^d curves under the shape
//! quotients of [`CurveShapeConfig`].

use super::geometry::{self, Grid};
use super::{
    align_to_srvf, curve_from_set, prepare, srvf_distance, validate_curve, Candidate,
    CurveShapeConfig, Prepared,
};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Karcher step size on the pre-shape sphere (fdasrsf `delta = 0.5`).
const SPHERE_STEP: f64 = 0.5;

/// Result of [`curve_shape_karcher_mean`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeKarcherResult {
    /// Karcher mean shape as a centred curve, `d × m` (unit length if `scale`).
    pub mean: FdMatrix,
    /// SRVF of the mean, `d × m` (unit norm if `scale`, closed if `closed`).
    pub mean_srvf: FdMatrix,
    /// Curves aligned to the mean (rotation, seed, warp; centred; unit length if
    /// `scale`).
    pub aligned_curves: FdCurveSet,
    /// SRVFs of the aligned curves.
    pub aligned_srvfs: FdCurveSet,
    /// Per-curve optimal rotation (`d × d`), applied as `R · βᵢ`.
    pub rotations: Vec<FdMatrix>,
    /// Per-curve warps (`n × m`).
    pub gammas: FdMatrix,
    /// Per-curve starting-point shifts (0 for open curves).
    pub seeds: Vec<usize>,
    /// Shooting vectors `vᵢ = log_μ(qᵢ*)` in the tangent space at the mean,
    /// `n × (d·m)`, column `k·m + j` = coordinate `k` at sample `j`.
    pub shooting_vectors: FdMatrix,
    /// Shape distance of each aligned curve to the mean.
    pub distances: Vec<f64>,
    /// Lengths `∫‖βᵢ'‖` of the input curves.
    pub lengths: Vec<f64>,
    /// Geometric mean of the lengths (fdasrsf `mean_scale`).
    pub mean_length: f64,
    /// Energy `Σ dᵢ²` at each iteration.
    pub energy: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the mean shooting vector norm fell below `tol`.
    pub converged: bool,
    /// The configuration used (determines the geometry).
    pub config: CurveShapeConfig,
}

/// Result of [`curve_shape_pca`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapePcaResult {
    /// PC scores `⟨vᵢ − v̄, φ_k⟩` (`n × ncomp`).
    pub scores: FdMatrix,
    /// Principal directions `φ_k` in the tangent space at the mean, each `d × m`,
    /// orthonormal in L² (trapezoidal).
    pub components: Vec<FdMatrix>,
    /// Eigenvalues (variances of the scores), decreasing.
    pub eigenvalues: Vec<f64>,
    /// Cumulative proportion of total tangent variance.
    pub cumulative_variance: Vec<f64>,
    /// Mean shooting vector `v̄` (`d × m`, ≈ 0 at a converged mean).
    pub mean_tangent: FdMatrix,
}

/// Log map at `mu` (sphere if scale removed, flat otherwise), projected onto the
/// tangent space of the closed-curve manifold when `closed`.
pub(super) fn log_map(
    mu: &FdMatrix,
    q: &FdMatrix,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> FdMatrix {
    let v = if cfg.scale {
        let theta = srvf_distance(mu, q, cfg, grid);
        let c = theta.cos();
        let u = geometry::axpby(1.0, q, -c, mu);
        let nu = geometry::norm(&u, grid);
        if nu > 1e-12 {
            geometry::scaled(&u, theta / nu)
        } else {
            FdMatrix::zeros(mu.nrows(), mu.ncols())
        }
    } else {
        geometry::axpby(1.0, q, -1.0, mu)
    };
    if cfg.closed || cfg.scale {
        geometry::project_tangent(&v, mu, grid, cfg.scale, cfg.closed)
    } else {
        v
    }
}

/// Exp map at `mu` (sphere or flat), projected back onto the closed manifold
/// when `closed`.
pub(super) fn exp_map(
    mu: &FdMatrix,
    v: &FdMatrix,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> FdMatrix {
    let q = if cfg.scale {
        let nv = geometry::norm(v, grid);
        if nv < 1e-300 {
            mu.clone()
        } else {
            geometry::axpby(nv.cos(), mu, nv.sin() / nv, v)
        }
    } else {
        geometry::axpby(1.0, mu, 1.0, v)
    };
    let q = if cfg.scale {
        let nq = geometry::norm(&q, grid);
        geometry::scaled(&q, 1.0 / nq)
    } else {
        q
    };
    if cfg.closed {
        geometry::project_closed(&q, grid).0
    } else {
        q
    }
}

/// One Karcher sweep: align every curve to `mu`.
struct Sweep {
    cands: Vec<Candidate>,
    shoots: Vec<FdMatrix>,
    energy: f64,
    vbar: FdMatrix,
}

fn sweep(
    mu: &FdMatrix,
    prepared: &[Prepared],
    cfg: &CurveShapeConfig,
    grid: &Grid,
    median: bool,
) -> Sweep {
    let n = prepared.len();
    let cands = align_all(mu, prepared, cfg, grid);
    let shoots: Vec<FdMatrix> = cands.iter().map(|c| log_map(mu, &c.q, cfg, grid)).collect();
    let (energy, vbar) = if median {
        weiszfeld_direction(&cands, &shoots, grid)
    } else {
        let energy = cands.iter().map(|c| c.distance * c.distance).sum();
        let mut vbar = FdMatrix::zeros(mu.nrows(), mu.ncols());
        for v in &shoots {
            vbar = geometry::axpby(1.0, &vbar, 1.0 / n as f64, v);
        }
        (energy, vbar)
    };
    Sweep {
        cands,
        shoots,
        energy,
        vbar,
    }
}

/// Align every prepared curve to the SRVF `mu` (in parallel).
pub(super) fn align_all(
    mu: &FdMatrix,
    prepared: &[Prepared],
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> Vec<Candidate> {
    iter_maybe_parallel!(0..prepared.len())
        .map(|i| align_to_srvf(mu, &prepared[i].beta, cfg, grid))
        .collect()
}

/// Karcher-median energy `Σ dᵢ` and Weiszfeld direction
/// `v̄ = Σ vᵢ/‖vᵢ‖ / Σ_{‖vᵢ‖>0} 1/‖vᵢ‖` (fdasrvf `pointwise_karcher_mean(ms =
/// "median")`; curves at distance 0 are skipped).
fn weiszfeld_direction(cands: &[Candidate], shoots: &[FdMatrix], grid: &Grid) -> (f64, FdMatrix) {
    let energy = cands.iter().map(|c| c.distance).sum();
    let (d, m) = shoots[0].shape();
    let mut sum = FdMatrix::zeros(d, m);
    let mut inv_sum = 0.0;
    for v in shoots {
        let nv = geometry::norm(v, grid);
        if nv > 0.0 {
            sum = geometry::axpby(1.0, &sum, 1.0 / nv, v);
            inv_sum += 1.0 / nv;
        }
    }
    let vbar = if inv_sum > 0.0 {
        geometry::scaled(&sum, 1.0 / inv_sum)
    } else {
        sum
    };
    (energy, vbar)
}

/// Bring an arbitrary SRVF onto the configured manifold (unit norm if `scale`,
/// closure-projected if `closed`).
fn to_manifold(q: FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    let q = if cfg.scale {
        let nq = geometry::norm(&q, grid);
        if nq > 1e-300 {
            geometry::scaled(&q, 1.0 / nq)
        } else {
            q
        }
    } else {
        q
    };
    if cfg.closed {
        geometry::project_closed(&q, grid).0
    } else {
        q
    }
}

/// Medoid initialisation of the Karcher median (fdasrvf): align every curve to
/// the first one, average the aligned SRVFs and start at the curve closest to
/// that average.
fn median_init(prepared: &[Prepared], cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    let cands = align_all(&prepared[0].q, prepared, cfg, grid);
    let n = cands.len() as f64;
    let (d, m) = prepared[0].q.shape();
    let mut avg = FdMatrix::zeros(d, m);
    for c in &cands {
        avg = geometry::axpby(1.0, &avg, 1.0 / n, &c.q);
    }
    let avg = to_manifold(avg, cfg, grid);
    let mut best = 0;
    let mut best_d = f64::INFINITY;
    for (i, c) in cands.iter().enumerate() {
        let di = srvf_distance(&avg, &c.q, cfg, grid);
        if di < best_d {
            best_d = di;
            best = i;
        }
    }
    prepared[best].q.clone()
}

/// Validate the inputs of [`curve_shape_karcher_mean`] and extract the curves.
fn karcher_inputs(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
    tol: f64,
) -> Result<Vec<FdMatrix>, FdarError> {
    config.validate()?;
    let n = curves.ncurves();
    if n < 2 || curves.ndim() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: "at least 2 curves with d >= 1".to_string(),
            actual: format!("{n} curves, d = {}", curves.ndim()),
        });
    }
    if config.closed && curves.ndim() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: "d >= 2 for closed curves".to_string(),
            actual: format!("d = {}", curves.ndim()),
        });
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "tol",
            message: format!("must be finite and >= 0, got {tol}"),
        });
    }
    let list: Vec<FdMatrix> = (0..n)
        .map(|i| curve_from_set(curves, i))
        .collect::<Result<_, _>>()?;
    for c in &list {
        validate_curve(c, "curves")?;
    }
    Ok(list)
}

/// Iterate the Karcher gradient descent; returns `(mu, final sweep, energy
/// history, n_iter, converged)`.
fn karcher_iterate(
    prepared: &[Prepared],
    cfg: &CurveShapeConfig,
    grid: &Grid,
    max_iter: usize,
    tol: f64,
    median: bool,
) -> (FdMatrix, Sweep, Vec<f64>, usize, bool) {
    // The median always takes the damped Weiszfeld step of fdasrvf (0.5).
    let step = if cfg.scale || median {
        SPHERE_STEP
    } else {
        1.0
    };
    let mut mu = if median {
        median_init(prepared, cfg, grid)
    } else {
        prepared[0].q.clone()
    };
    let mut cur = sweep(&mu, prepared, cfg, grid, median);
    let mut energy = vec![cur.energy];
    let mut n_iter = 0;
    let mut converged = geometry::norm(&cur.vbar, grid) < tol;
    while !converged && n_iter < max_iter {
        n_iter += 1;
        let mu_new = exp_map(&mu, &geometry::scaled(&cur.vbar, step), cfg, grid);
        let next = sweep(&mu_new, prepared, cfg, grid, median);
        if next.energy > cur.energy {
            // Energy increased (step overshoot or DP discretisation floor): keep
            // the previous mean, as fdasrsf does by stopping.
            break;
        }
        mu = mu_new;
        cur = next;
        energy.push(cur.energy);
        converged = geometry::norm(&cur.vbar, grid) < tol;
    }
    (mu, cur, energy, n_iter, converged)
}

/// Karcher (Fréchet) mean of a set of curves in elastic shape space, under the
/// rotation / scale / reparametrisation / closure / seed quotients of `config`
/// (fdasrsf `fdacurve.karcher_mean`, fdasrvf `multivariate_karcher_mean`).
///
/// Gradient descent on the shape manifold: each sweep aligns every curve to the
/// current mean SRVF μ (in parallel), maps it to the tangent space at μ
/// (sphere log map when `scale`, flat otherwise; closure-normal projection when
/// `closed`) and moves μ along the mean shooting vector `v̄` (step 0.5 on the
/// sphere as in fdasrsf, 1 in the flat case — the exact average of aligned
/// SRVFs), re-projecting onto the closed manifold. Initialised at the first
/// curve. Stops when `‖v̄‖ < tol` (converged), when the energy `Σ dᵢ²` would
/// increase, or after `max_iter` updates; all outputs refer to the final mean.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for fewer than 2 curves or invalid shapes,
/// [`FdarError::InvalidParameter`] for an invalid `config` / `tol`,
/// [`FdarError::ComputationFailed`] for a zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_karcher_mean, curves_to_set, CurveShapeConfig};
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curves: Vec<FdMatrix> = (0..4)
///     .map(|i| {
///         let a = 0.2 + 0.05 * i as f64;
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
/// let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 10, 1e-3).unwrap();
/// assert_eq!(km.mean.shape(), (2, m));
/// assert!(km.distances.iter().all(|&d| d < 0.2));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_karcher_mean(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
    max_iter: usize,
    tol: f64,
) -> Result<CurveShapeKarcherResult, FdarError> {
    karcher_impl(curves, config, max_iter, tol, false)
}

/// Karcher (geometric) **median** of a set of curves in elastic shape space —
/// the minimiser of `Σ d(μ, [βᵢ])` instead of `Σ d²` (fdasrvf
/// `multivariate_karcher_mean(ms = "median")`, the centre used by
/// `curve_boxplot` and `shape_CI`).
///
/// Same alignment sweeps as [`curve_shape_karcher_mean`], but the update uses
/// the Weiszfeld direction `v̄ = Σ vᵢ/‖vᵢ‖ / Σ 1/‖vᵢ‖` of the shooting vectors
/// with fdasrvf's damped step 0.5, and the iteration starts at the medoid (the
/// curve closest to the average of the SRVFs aligned to the first curve). The
/// returned `energy` history is `Σ dᵢ` (not squared); `converged` refers to
/// `‖v̄‖ < tol`. Robust to outlying curves, unlike the mean.
///
/// # Errors
/// As [`curve_shape_karcher_mean`].
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_karcher_median, curves_to_set, CurveShapeConfig};
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curves: Vec<FdMatrix> = (0..5)
///     .map(|i| {
///         let a = 0.2 + 0.05 * i as f64;
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
/// assert_eq!(med.mean.shape(), (2, m)); // the median shape (field name shared with the mean)
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_karcher_median(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
    max_iter: usize,
    tol: f64,
) -> Result<CurveShapeKarcherResult, FdarError> {
    karcher_impl(curves, config, max_iter, tol, true)
}

/// Shared body of the Karcher mean (`median = false`) and median.
fn karcher_impl(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
    max_iter: usize,
    tol: f64,
    median: bool,
) -> Result<CurveShapeKarcherResult, FdarError> {
    let list = karcher_inputs(curves, config, tol)?;
    let (n, d, m) = (curves.ncurves(), curves.ndim(), curves.npoints());
    let grid = Grid::new(m);
    let prepared: Vec<Prepared> = list
        .iter()
        .map(|c| prepare(c, config, &grid))
        .collect::<Result<_, _>>()?;
    let (mu, fin, energy, n_iter, converged) =
        karcher_iterate(&prepared, config, &grid, max_iter, tol, median);

    let mut aligned = Vec::with_capacity(n);
    let mut aligned_q = Vec::with_capacity(n);
    let mut gammas = FdMatrix::zeros(n, m);
    let mut shooting_vectors = FdMatrix::zeros(n, d * m);
    for (i, (c, v)) in fin.cands.iter().zip(&fin.shoots).enumerate() {
        aligned.push(geometry::center(&c.beta, &grid));
        aligned_q.push(c.q.clone());
        for j in 0..m {
            gammas[(i, j)] = c.gamma[j];
            for k in 0..d {
                shooting_vectors[(i, k * m + j)] = v[(k, j)];
            }
        }
    }
    let lengths: Vec<f64> = prepared.iter().map(|p| p.length).collect();
    let mean_length = (lengths.iter().map(|l| l.ln()).sum::<f64>() / n as f64).exp();
    Ok(CurveShapeKarcherResult {
        mean: geometry::center(&geometry::q_to_curve(&mu, &grid), &grid),
        mean_srvf: mu,
        aligned_curves: super::curves_to_set(&aligned)?,
        aligned_srvfs: super::curves_to_set(&aligned_q)?,
        rotations: fin.cands.iter().map(|c| c.rotation.clone()).collect(),
        gammas,
        seeds: fin.cands.iter().map(|c| c.seed).collect(),
        shooting_vectors,
        distances: fin.cands.iter().map(|c| c.distance).collect(),
        lengths,
        mean_length,
        energy,
        n_iter,
        converged,
        config: config.clone(),
    })
}

/// Check a Karcher result's tangent data and return `(n, d, m)`.
fn tangent_dims(result: &CurveShapeKarcherResult) -> Result<(usize, usize, usize), FdarError> {
    let (d, m) = result.mean_srvf.shape();
    let (n, dm) = result.shooting_vectors.shape();
    if n < 2 || dm != d * m || d == 0 || m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "result",
            expected: "n >= 2 shooting vectors of length d*m".to_string(),
            actual: format!("{n} x {dm} for d = {d}, m = {m}"),
        });
    }
    Ok((n, d, m))
}

/// Column means of the shooting vectors.
fn shooting_mean(v: &FdMatrix) -> Vec<f64> {
    let (n, p) = v.shape();
    (0..p)
        .map(|c| v.column(c).iter().sum::<f64>() / n as f64)
        .collect()
}

/// Sample covariance (`(n−1)` denominator) of the shooting vectors of a Karcher
/// shape mean — the Karcher covariance in the tangent space at the mean
/// (fdasrsf `fdacurve.karcher_cov`, fdasrvf `multivariate_karcher_cov`).
///
/// Returns a `(d·m) × (d·m)` matrix indexed like
/// [`CurveShapeKarcherResult::shooting_vectors`] (plain Euclidean coordinates of
/// the sampled fields, as in fdasrsf).
///
/// # Errors
/// [`FdarError::InvalidDimension`] for inconsistent results;
/// [`FdarError::InvalidParameter`] if `d·m > 10000`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_karcher_covariance(
    result: &CurveShapeKarcherResult,
) -> Result<FdMatrix, FdarError> {
    let (n, d, m) = tangent_dims(result)?;
    let p = d * m;
    if p > 10_000 {
        return Err(FdarError::InvalidParameter {
            parameter: "d*m",
            message: format!("d*m = {p} exceeds limit of 10000; use curve_shape_pca instead"),
        });
    }
    let v = &result.shooting_vectors;
    let mean = shooting_mean(v);
    let mut cov = FdMatrix::zeros(p, p);
    for a in 0..p {
        for b in a..p {
            let s: f64 = (0..n)
                .map(|i| (v[(i, a)] - mean[a]) * (v[(i, b)] - mean[b]))
                .sum::<f64>()
                / (n - 1) as f64;
            cov[(a, b)] = s;
            cov[(b, a)] = s;
        }
    }
    Ok(cov)
}

/// L²-weighted, centred tangent data matrix `X_{i,(k,j)} = (vᵢ − v̄)_{kj} √w_j`.
fn weighted_centered(result: &CurveShapeKarcherResult, grid: &Grid) -> (FdMatrix, Vec<f64>) {
    let v = &result.shooting_vectors;
    let (n, p) = v.shape();
    let m = grid.len();
    let mean = shooting_mean(v);
    let mut x = FdMatrix::zeros(n, p);
    for c in 0..p {
        let sw = grid.w[c % m].sqrt();
        for i in 0..n {
            x[(i, c)] = (v[(i, c)] - mean[c]) * sw;
        }
    }
    (x, mean)
}

/// Gram matrix `X Xᵀ/(n−1)`.
fn gram(x: &FdMatrix) -> nalgebra::DMatrix<f64> {
    let (n, p) = x.shape();
    let mut g = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let s: f64 = (0..p).map(|c| x[(i, c)] * x[(j, c)]).sum::<f64>() / (n - 1) as f64;
            g[(i, j)] = s;
            g[(j, i)] = s;
        }
    }
    g
}

/// Reshape a length-`d·m` vector into a `d × m` matrix.
fn to_field(v: &[f64], d: usize, m: usize) -> FdMatrix {
    let mut f = FdMatrix::zeros(d, m);
    for k in 0..d {
        for j in 0..m {
            f[(k, j)] = v[k * m + j];
        }
    }
    f
}

/// Shape PCA (principal geodesic analysis in the tangent space at the Karcher
/// mean; fdasrsf `fdacurve.shape_pca`, fdasrvf `multivariate_pca`).
///
/// PCA of the shooting vectors in the L² metric of the tangent space
/// (trapezoidal weights), computed through the `n × n` Gram matrix so there is
/// no `d·m` cap. Components are L²-orthonormal; use
/// [`CurveShapePcaResult::mode_shape`] to map a principal direction back to a
/// curve.
///
/// # Errors
/// [`FdarError::InvalidDimension`] for inconsistent results;
/// [`FdarError::InvalidParameter`] if `ncomp == 0`;
/// [`FdarError::ComputationFailed`] if the eigen-decomposition fails.
///
/// # Example
/// ```
/// use fdars_core::alignment::{
///     curve_shape_karcher_mean, curve_shape_pca, curves_to_set, CurveShapeConfig,
/// };
/// use fdars_core::matrix::FdMatrix;
/// let m = 30;
/// let curves: Vec<FdMatrix> = (0..5)
///     .map(|i| {
///         let a = 0.1 + 0.08 * i as f64;
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
/// let km = curve_shape_karcher_mean(&set, &CurveShapeConfig::default(), 10, 1e-3).unwrap();
/// let pca = curve_shape_pca(&km, 2).unwrap();
/// assert!(pca.cumulative_variance[0] > 0.9); // one-parameter family
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_pca(
    result: &CurveShapeKarcherResult,
    ncomp: usize,
) -> Result<CurveShapePcaResult, FdarError> {
    let (n, d, m) = tangent_dims(result)?;
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "must be >= 1".to_string(),
        });
    }
    let ncomp = ncomp.min(n - 1);
    let grid = Grid::new(m);
    let (x, mean) = weighted_centered(result, &grid);
    let eig = nalgebra::SymmetricEigen::try_new(gram(&x), 1e-14, 10_000).ok_or_else(|| {
        FdarError::ComputationFailed {
            operation: "curve_shape_pca",
            detail: "eigen-decomposition of the Gram matrix failed".to_string(),
        }
    })?;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let total: f64 = eig.eigenvalues.iter().map(|&l| l.max(0.0)).sum();

    let mut scores = FdMatrix::zeros(n, ncomp);
    let mut components = Vec::with_capacity(ncomp);
    let mut eigenvalues = Vec::with_capacity(ncomp);
    let mut cumulative_variance = Vec::with_capacity(ncomp);
    let mut running = 0.0;
    let nf = (n - 1) as f64;
    for (k, &idx) in order.iter().take(ncomp).enumerate() {
        let lambda = eig.eigenvalues[idx].max(0.0);
        let u = eig.eigenvectors.column(idx);
        let s = (lambda * nf).sqrt().max(1e-300);
        // φ_k = Xᵀu/s (unit Euclidean in weighted coords) → divide by √w.
        let phi: Vec<f64> = (0..d * m)
            .map(|c| (0..n).map(|i| x[(i, c)] * u[i]).sum::<f64>() / s / grid.w[c % m].sqrt())
            .collect();
        for i in 0..n {
            scores[(i, k)] = u[i] * s;
        }
        components.push(to_field(&phi, d, m));
        eigenvalues.push(lambda);
        running += lambda;
        cumulative_variance.push(if total > 0.0 { running / total } else { 0.0 });
    }
    Ok(CurveShapePcaResult {
        scores,
        components,
        eigenvalues,
        cumulative_variance,
        mean_tangent: to_field(&mean, d, m),
    })
}

impl CurveShapePcaResult {
    /// Curve on the principal geodesic of component `component` at `coef`
    /// standard deviations from the mean: `β = srvf⁻¹(exp_μ(v̄ + coef·√λ_k·φ_k))`
    /// (closure-projected for closed contours; centred). `coef = 0` returns
    /// (approximately) the mean shape.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `component` is out of range or `coef`
    /// is not finite; [`FdarError::InvalidDimension`] if `mean` does not match.
    pub fn mode_shape(
        &self,
        mean: &CurveShapeKarcherResult,
        component: usize,
        coef: f64,
    ) -> Result<FdMatrix, FdarError> {
        if component >= self.components.len() || !coef.is_finite() {
            return Err(FdarError::InvalidParameter {
                parameter: "component/coef",
                message: format!(
                    "component {component} of {} with coef {coef}",
                    self.components.len()
                ),
            });
        }
        if mean.mean_srvf.shape() != self.mean_tangent.shape() {
            return Err(FdarError::InvalidDimension {
                parameter: "mean",
                expected: format!("{:?}", self.mean_tangent.shape()),
                actual: format!("{:?}", mean.mean_srvf.shape()),
            });
        }
        let grid = Grid::new(self.mean_tangent.ncols());
        let cfg = &mean.config;
        let sd = self.eigenvalues[component].sqrt();
        let v = geometry::axpby(
            1.0,
            &self.mean_tangent,
            coef * sd,
            &self.components[component],
        );
        let v = geometry::project_tangent(&v, &mean.mean_srvf, &grid, cfg.scale, cfg.closed);
        let q = exp_map(&mean.mean_srvf, &v, cfg, &grid);
        Ok(geometry::center(&geometry::q_to_curve(&q, &grid), &grid))
    }
}
