//! Elastic means of sparsely observed open curves (elasdics
//! `compute_elastic_mean` / `fit_mean` / `get_model_data` / `make_design`).

use super::discrete::{compute_distance, find_optimal_t_discrete, Reference};
use super::polygon::{
    eval_polygon, points_from_srv, prepare_curve, srv_from_points_unchecked, srv_from_polygon,
    SrvPoly,
};
use super::smooth::{find_optimal_t_smooth, integrate_curve, smooth_distance, LinSpline};
use super::{to_points_matrix, validate_eps, SparseCurve};
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Spline type of the SRV of a [`sparse_elastic_mean`] (elasdics `type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SparseMeanType {
    /// Linear SRV spline (hat basis on the knots): a differentiable mean curve
    /// (elasdics `type = "smooth"`, the default).
    #[default]
    Smooth,
    /// Constant SRV spline on the knot intervals: a polygonal mean with corners
    /// at the knots (elasdics `type = "polygon"`).
    Polygon,
}

/// Configuration of [`sparse_elastic_mean`].
///
/// # Examples
/// ```
/// use fdars_core::alignment::{SparseElasticMeanConfig, SparseMeanType};
/// let mut cfg = SparseElasticMeanConfig::default();
/// cfg.knots = (0..=10).map(|k| k as f64 / 10.0).collect();
/// cfg.mean_type = SparseMeanType::Polygon;
/// assert_eq!(cfg.max_iter, 50);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseElasticMeanConfig {
    /// Spline knots of the mean SRV: strictly increasing, from 0 to 1, at least
    /// 2 values. Default `[0, 0.25, 0.5, 0.75, 1]` (elasdics `seq(0, 1, len = 5)`).
    pub knots: Vec<f64>,
    /// Spline type. Default [`SparseMeanType::Smooth`].
    pub mean_type: SparseMeanType,
    /// Stop when the relative squared change of the spline coefficients
    /// `‖c − c_old‖²/‖c‖²` is below `eps`; iteration `i` aligns with tolerance
    /// `100·eps/i` (elasdics `eps`). Default `0.01`.
    pub eps: f64,
    /// Maximum number of fit/align iterations (≥ 1). Default 50.
    pub max_iter: usize,
}

impl Default for SparseElasticMeanConfig {
    fn default() -> Self {
        Self {
            knots: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            mean_type: SparseMeanType::Smooth,
            eps: 0.01,
            max_iter: 50,
        }
    }
}

/// Result of [`sparse_elastic_mean`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseElasticMeanResult {
    /// Spline type of the mean SRV.
    pub mean_type: SparseMeanType,
    /// Knots of the mean SRV spline (length `K + 1`).
    pub knots: Vec<f64>,
    /// SRV spline coefficients, `d × nbasis` (column `j` = coefficient vector `j`):
    /// `nbasis = K + 1` (values at the knots) for `Smooth`, `K` (value on knot
    /// interval `j`) for `Polygon`. elasdics `coefs` transposed.
    pub coefs: FdMatrix,
    /// Observed points of every curve after duplicate removal, `d × m_i`.
    pub curves: Vec<FdMatrix>,
    /// Initial parametrisation of every curve (length `m_i`).
    pub params: Vec<Vec<f64>>,
    /// Optimal parametrisation of every curve when aligned to the mean.
    pub params_optim: Vec<Vec<f64>>,
    /// Elastic distance of every curve (at `params_optim`) to the returned mean.
    pub distances: Vec<f64>,
    /// Number of fit/align iterations performed.
    pub n_iter: usize,
    /// Whether the coefficient criterion fell below `eps` within `max_iter`.
    pub converged: bool,
}

impl SparseElasticMeanResult {
    fn d(&self) -> usize {
        self.coefs.nrows()
    }

    /// Evaluate the mean SRV at parameter values `t` (in `[0, 1]`); `d × len`.
    ///
    /// # Examples
    /// See [`sparse_elastic_mean`].
    pub fn srv_eval(&self, t: &[f64]) -> FdMatrix {
        let d = self.d();
        let mut out = FdMatrix::zeros(d, t.len());
        let c = self.coefs.as_slice();
        let k = self.knots.len();
        for (n, &x) in t.iter().enumerate() {
            match self.mean_type {
                SparseMeanType::Smooth => {
                    let sp = LinSpline {
                        d,
                        knots: &self.knots,
                        c,
                    };
                    sp.eval(x, out.column_mut(n));
                }
                SparseMeanType::Polygon => {
                    let j = self
                        .knots
                        .partition_point(|&v| v <= x)
                        .saturating_sub(1)
                        .min(k - 2);
                    out.column_mut(n).copy_from_slice(&c[j * d..(j + 1) * d]);
                }
            }
        }
        out
    }

    /// Evaluate the mean curve at parameter values `t_grid` (non-decreasing, in
    /// `[0, 1]`); `d × len`. The curve starts at the origin; with `centering`
    /// the mean of the evaluated points is subtracted (elasdics `get_evals`).
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `t_grid` is empty, not non-decreasing
    /// or leaves `[0, 1]`.
    ///
    /// # Examples
    /// See [`sparse_elastic_mean`].
    pub fn mean_curve(&self, t_grid: &[f64], centering: bool) -> Result<FdMatrix, FdarError> {
        let ok = !t_grid.is_empty()
            && t_grid.iter().all(|v| (0.0..=1.0).contains(v))
            && t_grid.windows(2).all(|w| w[1] >= w[0]);
        if !ok {
            return Err(FdarError::InvalidParameter {
                parameter: "t_grid",
                message: "must be non-empty, non-decreasing and within [0, 1]".to_string(),
            });
        }
        let d = self.d();
        let c = self.coefs.as_slice();
        let mut pts = match self.mean_type {
            SparseMeanType::Smooth => integrate_curve(
                &LinSpline {
                    d,
                    knots: &self.knots,
                    c,
                },
                t_grid,
            ),
            SparseMeanType::Polygon => {
                let srv = SrvPoly {
                    d,
                    t: self.knots.clone(),
                    q: c.to_vec(),
                };
                let corners = points_from_srv(&srv);
                let mut out = vec![0.0; t_grid.len() * d];
                for (n, &x) in t_grid.iter().enumerate() {
                    eval_polygon(d, &corners, &self.knots, x, &mut out[n * d..(n + 1) * d]);
                }
                out
            }
        };
        if centering {
            let len = t_grid.len() as f64;
            for c in 0..d {
                let mean = pts.iter().skip(c).step_by(d).sum::<f64>() / len;
                pts.iter_mut().skip(c).step_by(d).for_each(|v| *v -= mean);
            }
        }
        to_points_matrix(d, &pts)
    }
}

fn validate_mean_config(cfg: &SparseElasticMeanConfig) -> Result<(), FdarError> {
    validate_eps(cfg.eps)?;
    let k = &cfg.knots;
    if k.len() < 2
        || k[0] != 0.0
        || k[k.len() - 1] != 1.0
        || k.windows(2)
            .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
    {
        return Err(FdarError::InvalidParameter {
            parameter: "knots",
            message: "must be strictly increasing from 0 to 1 with at least 2 values".to_string(),
        });
    }
    if cfg.max_iter == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "max_iter",
            message: "must be >= 1".to_string(),
        });
    }
    Ok(())
}

/// Spline design row at `x` (elasdics `make_design`, `closed = FALSE`):
/// indicator of the knot interval (polygon) or hat functions (smooth).
fn design_row(knots: &[f64], mean_type: SparseMeanType, x: f64, row: &mut [f64]) {
    row.fill(0.0);
    let j = knots
        .partition_point(|&v| v <= x)
        .saturating_sub(1)
        .min(knots.len() - 2);
    match mean_type {
        SparseMeanType::Polygon => row[j] = 1.0,
        SparseMeanType::Smooth => {
            let w1 = (x - knots[j]) / (knots[j + 1] - knots[j]);
            row[j] = 1.0 - w1;
            row[j + 1] = w1;
        }
    }
}

/// Long-format model data: covariate `m` and warped SRV rows (elasdics
/// `get_model_data`).
fn model_data(
    srvs: &[SrvPoly],
    t_optims: &[Vec<f64>],
    knots: &[f64],
    mean_type: SparseMeanType,
) -> (Vec<f64>, Vec<f64>) {
    let d = srvs[0].d;
    let mut m = Vec::new();
    let mut y = Vec::new();
    for (srv, topt) in srvs.iter().zip(t_optims) {
        match mean_type {
            SparseMeanType::Polygon => {
                let pts = points_from_srv(srv);
                let mut at_knots = vec![0.0; knots.len() * d];
                for (n, &x) in knots.iter().enumerate() {
                    eval_polygon(d, &pts, topt, x, &mut at_knots[n * d..(n + 1) * d]);
                }
                let s = srv_from_points_unchecked(d, &at_knots, knots);
                m.extend(knots.windows(2).map(|w| w[1] - 0.5 * (w[1] - w[0])));
                y.extend_from_slice(&s.q);
            }
            SparseMeanType::Smooth => {
                for k in 0..srv.nseg() {
                    let old = srv.t[k + 1] - srv.t[k];
                    let new = topt[k + 1] - topt[k];
                    let f = (old / new).sqrt();
                    y.extend(srv.qk(k).iter().map(|v| v * f));
                    m.push(topt[k + 1] - 0.5 * new);
                }
            }
        }
    }
    (m, y)
}

/// Normal equations `XᵀX`, `Xᵀy` of coordinate `c`, skipping non-finite rows.
fn normal_equations(
    m: &[f64],
    y: &[f64],
    d: usize,
    c: usize,
    knots: &[f64],
    mean_type: SparseMeanType,
    nbasis: usize,
) -> (DMatrix<f64>, DVector<f64>) {
    let mut xtx = DMatrix::<f64>::zeros(nbasis, nbasis);
    let mut xty = DVector::<f64>::zeros(nbasis);
    let mut row = vec![0.0; nbasis];
    for (n, &mn) in m.iter().enumerate() {
        let yv = y[n * d + c];
        if !yv.is_finite() || !mn.is_finite() {
            continue;
        }
        design_row(knots, mean_type, mn, &mut row);
        let nz: Vec<usize> = (0..nbasis).filter(|&a| row[a] != 0.0).collect();
        for &a in &nz {
            xty[a] += row[a] * yv;
            for &b in &nz {
                xtx[(a, b)] += row[a] * row[b];
            }
        }
    }
    (xtx, xty)
}

/// Least-squares spline coefficients per coordinate, dropping non-finite
/// responses (elasdics `lm(q ~ -1 + design)`); point-major `nbasis × d`.
fn fit_coefs(m: &[f64], y: &[f64], d: usize, knots: &[f64], mean_type: SparseMeanType) -> Vec<f64> {
    let nbasis = match mean_type {
        SparseMeanType::Smooth => knots.len(),
        SparseMeanType::Polygon => knots.len() - 1,
    };
    let mut coefs = vec![0.0; nbasis * d];
    for c in 0..d {
        let (xtx, xty) = normal_equations(m, y, d, c, knots, mean_type, nbasis);
        let sol = xtx
            .svd(true, true)
            .solve(&xty, 1e-12)
            .unwrap_or_else(|_| DVector::zeros(nbasis));
        for a in 0..nbasis {
            coefs[a * d + c] = sol[a];
        }
    }
    coefs
}

/// Align every curve to the current mean.
fn align_all(
    srvs: &[SrvPoly],
    t_optims: &[Vec<f64>],
    knots: &[f64],
    coefs: &[f64],
    mean_type: SparseMeanType,
    eps: f64,
) -> Vec<Vec<f64>> {
    let d = srvs[0].d;
    iter_maybe_parallel!(0..srvs.len())
        .map(|j| match mean_type {
            SparseMeanType::Polygon => {
                let reference = Reference {
                    d,
                    r: knots,
                    p: coefs,
                };
                find_optimal_t_discrete(&reference, &srvs[j], &t_optims[j], eps)
            }
            SparseMeanType::Smooth => {
                let sp = LinSpline { d, knots, c: coefs };
                find_optimal_t_smooth(&sp, &srvs[j], &t_optims[j], eps)
            }
        })
        .collect()
}

/// Distance of every curve at its parametrisation to the mean.
fn distances_to_mean(
    srvs: &[SrvPoly],
    t_optims: &[Vec<f64>],
    knots: &[f64],
    coefs: &[f64],
    mean_type: SparseMeanType,
) -> Vec<f64> {
    let d = srvs[0].d;
    srvs.iter()
        .zip(t_optims)
        .map(|(srv, t)| match mean_type {
            SparseMeanType::Polygon => {
                let mean = SrvPoly {
                    d,
                    t: knots.to_vec(),
                    q: coefs.to_vec(),
                };
                compute_distance(&mean, srv, t, false)
            }
            SparseMeanType::Smooth => smooth_distance(&LinSpline { d, knots, c: coefs }, srv, t),
        })
        .collect()
}

/// Elastic (Fréchet) mean of sparsely observed open curves (elasdics
/// `compute_elastic_mean`, `closed = FALSE`).
///
/// Each curve is a polygon through its own observed points (any number per
/// curve, any dimension `d ≥ 1`); the mean is a curve whose SRV is a spline on
/// `config.knots` — linear ([`SparseMeanType::Smooth`]) or constant
/// ([`SparseMeanType::Polygon`]). Starting from the curves' own
/// parametrisations, the algorithm alternates
/// 1. a least-squares fit of the spline coefficients to the warped SRVs of all
///    curves (long format, as elasdics), and
/// 2. the elastic alignment of every curve to the current mean (optimising its
///    parametrisation at its observation points),
///
/// until the relative change of the coefficients is below `config.eps`.
///
/// The mean is identified up to reparametrisation (and translation); evaluate
/// it with [`SparseElasticMeanResult::mean_curve`] /
/// [`SparseElasticMeanResult::srv_eval`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] for an empty curve list or mixed
/// dimensions; [`FdarError::InvalidParameter`] for invalid curves / knots /
/// `eps` / `max_iter`.
///
/// # Examples
/// ```
/// use fdars_core::alignment::{sparse_elastic_mean, SparseCurve, SparseElasticMeanConfig};
/// use fdars_core::matrix::FdMatrix;
/// // three samples of a half circle, each observed at its own points
/// let curves: Vec<SparseCurve> = [25usize, 32, 40]
///     .iter()
///     .map(|&m| {
///         let mut v = Vec::new();
///         for j in 0..m {
///             let s = std::f64::consts::PI * (j as f64 / (m - 1) as f64).powf(1.2);
///             v.extend([s.cos(), s.sin()]);
///         }
///         SparseCurve::new(FdMatrix::from_column_major(v, 2, m).unwrap())
///     })
///     .collect();
/// let mut cfg = SparseElasticMeanConfig::default();
/// cfg.knots = (0..=8).map(|k| k as f64 / 8.0).collect();
/// let res = sparse_elastic_mean(&curves, &cfg).unwrap();
/// let mean = res.mean_curve(&[0.0, 0.5, 1.0], false).unwrap();
/// assert_eq!(mean.shape(), (2, 3));
/// // the half circle from (1, 0) to (-1, 0): the mean ends ≈ 2 to the left
/// assert!((mean[(0, 2)] + 2.0).abs() < 0.05 && mean[(1, 2)].abs() < 0.05);
/// assert!(res.distances.iter().all(|&dist| dist < 0.15));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn sparse_elastic_mean(
    curves: &[SparseCurve],
    config: &SparseElasticMeanConfig,
) -> Result<SparseElasticMeanResult, FdarError> {
    validate_mean_config(config)?;
    if curves.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: "at least one curve".to_string(),
            actual: "0".to_string(),
        });
    }
    let polys = curves
        .iter()
        .map(|c| prepare_curve(c, false, "curves"))
        .collect::<Result<Vec<_>, _>>()?;
    let d = polys[0].d;
    if let Some(bad) = polys.iter().position(|p| p.d != d) {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: format!("{d} coordinates for every curve"),
            actual: format!("curve {bad} has {}", polys[bad].d),
        });
    }
    let srvs = polys
        .iter()
        .map(|p| srv_from_polygon(p, "curves"))
        .collect::<Result<Vec<_>, _>>()?;
    let (knots, mean_type) = (&config.knots[..], config.mean_type);

    let mut t_optims: Vec<Vec<f64>> = srvs.iter().map(|s| s.t.clone()).collect();
    let mut coefs_old: Option<Vec<f64>> = None;
    let mut coefs = Vec::new();
    let mut converged = false;
    let mut n_iter = 0;
    for it in 1..=config.max_iter {
        n_iter = it;
        let (m, y) = model_data(&srvs, &t_optims, knots, mean_type);
        coefs = fit_coefs(&m, &y, d, knots, mean_type);
        let crit = coefs_old.as_ref().map_or(1.0, |old| {
            let num: f64 = coefs.iter().zip(old).map(|(a, b)| (a - b).powi(2)).sum();
            num / coefs.iter().map(|a| a * a).sum::<f64>()
        });
        if crit < config.eps {
            converged = true;
            break;
        }
        let eps_align = config.eps * 100.0 / it as f64;
        t_optims = align_all(&srvs, &t_optims, knots, &coefs, mean_type, eps_align);
        coefs_old = Some(coefs.clone());
    }
    let distances = distances_to_mean(&srvs, &t_optims, knots, &coefs, mean_type);
    let nbasis = coefs.len() / d;
    Ok(SparseElasticMeanResult {
        mean_type,
        knots: knots.to_vec(),
        coefs: FdMatrix::from_column_major(coefs, d, nbasis)?,
        curves: polys
            .iter()
            .map(|p| to_points_matrix(d, &p.pts))
            .collect::<Result<Vec<_>, _>>()?,
        params: polys.iter().map(|p| p.t.clone()).collect(),
        params_optim: t_optims,
        distances,
        n_iter,
        converged,
    })
}
