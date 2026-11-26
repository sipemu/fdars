//! Elastic alignment and elastic means for curves observed on sparse,
//! irregular or differing point sets (Steyer, Stöcker & Greven 2023; R
//! `elasdics` 1.1.3).
//!
//! The dense elastic functions of this crate need every curve on one common
//! `argvals` grid. Here each observation is used *as observed*: a curve
//! `β : [0, 1] → R^d` (any `d ≥ 1`, so 1-D functions are included) sampled at
//! `m_i` points — different for every curve — is treated as the **polygon**
//! through its points, whose square-root-velocity (SRV) transform
//! `q = β'/√‖β'‖` is piecewise constant:
//! `q_k = Δβ_k / √(Δt_k ‖Δβ_k‖)` on `[t_k, t_{k+1})`. The parametrisation `t` is
//! user supplied ([`SparseCurve::with_param`]) or arc length
//! ([`SparseCurve::new`]); consecutive duplicate points are dropped.
//!
//! * [`sparse_elastic_align_pair`] (elasdics `align_curves`) — optimal
//!   reparametrisation of one polygon to another, for open curves and closed
//!   curves (`closed = true`: periodic extension, starting-point pre-alignment).
//!   The warp is optimised at the observation points of the second curve by the
//!   elasdics coordinate-wise **analytic** maximisation of the warped inner
//!   product `Σ_k √(Δs_k ∫_{t_k}^{t_{k+1}} ⟨q_k, p(u)⟩_+² du)` (warps may be
//!   non-strictly increasing, i.e. the closure of the warping group).
//! * [`sparse_elastic_mean`] (elasdics `compute_elastic_mean`, open and closed
//!   curves) — an elastic (Fréchet) mean whose SRV is a spline on given `knots`:
//!   [`SparseMeanType::Smooth`] (linear SRV spline, differentiable mean curve)
//!   or [`SparseMeanType::Polygon`] (constant SRV spline, polygonal mean).
//!   Alternates a least-squares spline fit of the warped SRVs with the
//!   alignment of every observation to the current mean. Closed means
//!   (`closed = true`, elasdics `fit_mean_closed`) pre-align the starting
//!   points, use a periodic spline and enforce closedness by a penalty.
//! * [`sparse_curves_from_irreg`] converts 1-D irregular functional data
//!   ([`IrregFdata`]) into sparse curves.
//!
//! The elastic distance is invariant to reparametrisation and translation (not
//! to rotation or scale).
//!
//! **Deviations from elasdics 1.1.3** (also stated on the items):
//! 1. Closed `align_curves` reports the distance *after* wrapping the optimal
//!    parametrisation into `[0, 1]`, which breaks its monotonicity and drops the
//!    wrapped segment's contribution (overestimating the distance; e.g. 2.339
//!    instead of 1.369 on its own example). This crate reports the distance of
//!    the optimal (unwrapped) warp — the value elasdics computes internally in
//!    `find_optimal_t_discrete_closed`.
//! 2. The smooth mean integrates the linear SRV spline exactly and maximises
//!    each warp coordinate with a bracketed 1-D search, where elasdics uses
//!    `integrate(rel.tol = 0.1)` inside `constrOptim`; results agree to the
//!    optimiser tolerance, not bit-for-bit.
//! 3. [`SparseElasticMeanResult::distances`] are the distances of each curve,
//!    at its returned parametrisation, to the *returned* mean (elasdics'
//!    `dist_to_mean` refers to the mean of the previous iteration).
//! 4. The duplicate-point threshold uses `max|x|·ε` instead of `max(x)·ε`.
//! 5. Closed elastic means work in any dimension `d ≥ 1` (elasdics
//!    `fit_mean_closed` is 2-D only); the gradient of the closure penalty is
//!    analytic (elasdics: `numDeriv::grad` on numerical integrals), and the
//!    optional extra closure steps (`closure_steps > 1`) are an addition.

mod closed_mean;
mod discrete;
#[cfg(test)]
mod golden;
#[cfg(test)]
mod golden_closed;
mod mean;
mod polygon;
mod smooth;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_closed;

pub use mean::{
    sparse_elastic_mean, SparseElasticMeanConfig, SparseElasticMeanResult, SparseMeanType,
};

use crate::error::FdarError;
use crate::irreg_fdata::IrregFdata;
use crate::matrix::FdMatrix;
use discrete::{
    compute_distance, find_optimal_t_discrete, find_optimal_t_discrete_closed, Reference,
};
use polygon::{arc_length_param, prepare_curve, srv_from_polygon};

/// A curve in R^d observed at its own (sparse, irregular) set of points.
///
/// `points` is `d × m` (row = coordinate, column = observed point, in order
/// along the curve), as for [`crate::alignment::curve_shape_align_pair`]
/// curves; `m` may differ between curves. `param` is an optional
/// parametrisation of the points (length `m`, non-decreasing in `[0, 1]`,
/// starting at 0 and — for open curves — ending at 1; strictly increasing once
/// duplicate points are removed). Without it the arc-length parametrisation is
/// used.
///
/// # Examples
/// ```
/// use fdars_core::alignment::SparseCurve;
/// use fdars_core::matrix::FdMatrix;
/// // the 2-D polygon (1,1) -> (0.5,-0.5) -> (-1,-1) -> (-1,1)
/// let pts = FdMatrix::from_column_major(vec![1.0, 1.0, 0.5, -0.5, -1.0, -1.0, -1.0, 1.0], 2, 4)
///     .unwrap();
/// let c = SparseCurve::new(pts.clone());
/// assert!(c.param.is_none());
/// let c2 = SparseCurve::with_param(pts, vec![0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]);
/// assert_eq!(c2.param.as_ref().unwrap().len(), 4);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseCurve {
    /// Observed points, `d × m` (row = coordinate, column = point).
    pub points: FdMatrix,
    /// Optional parametrisation of the points (length `m`); arc length if `None`.
    pub param: Option<Vec<f64>>,
}

impl SparseCurve {
    /// Curve parametrised by arc length.
    pub fn new(points: FdMatrix) -> Self {
        Self {
            points,
            param: None,
        }
    }

    /// Curve with a given parametrisation `param` of its points.
    pub fn with_param(points: FdMatrix, param: Vec<f64>) -> Self {
        Self {
            points,
            param: Some(param),
        }
    }
}

/// Convert 1-D irregular functional data into sparse curves for
/// [`sparse_elastic_mean`] / [`sparse_elastic_align_pair`].
///
/// Observation `i` (points `(a_ij, x_ij)`) becomes the `1 × m_i` curve of its
/// values with parametrisation `(a_ij − a_i0)/(a_i,last − a_i0)`, i.e. its own
/// observed interval mapped to `[0, 1]`.
///
/// # Errors
/// [`FdarError::InvalidParameter`] if an observation has fewer than 2 points or
/// non-increasing argvals.
///
/// # Examples
/// ```
/// use fdars_core::alignment::sparse_curves_from_irreg;
/// use fdars_core::irreg_fdata::IrregFdata;
/// let data = IrregFdata::from_lists(
///     &[vec![0.0, 0.5, 1.0], vec![0.0, 0.2, 0.7, 1.0]],
///     &[vec![0.0, 1.0, 0.0], vec![0.0, 0.5, 0.8, 0.0]],
/// );
/// let curves = sparse_curves_from_irreg(&data).unwrap();
/// assert_eq!(curves.len(), 2);
/// assert_eq!(curves[1].points.shape(), (1, 4));
/// ```
pub fn sparse_curves_from_irreg(data: &IrregFdata) -> Result<Vec<SparseCurve>, FdarError> {
    (0..data.n_obs())
        .map(|i| {
            let (a, x) = data.get_obs(i);
            let m = a.len();
            if m < 2 || a.windows(2).any(|w| w[1] <= w[0]) {
                return Err(FdarError::InvalidParameter {
                    parameter: "data",
                    message: format!(
                        "observation {i} needs at least 2 points with strictly increasing argvals"
                    ),
                });
            }
            let span = a[m - 1] - a[0];
            let mut param: Vec<f64> = a.iter().map(|v| (v - a[0]) / span).collect();
            param[0] = 0.0;
            param[m - 1] = 1.0;
            let points = FdMatrix::from_column_major(x.to_vec(), 1, m)?;
            Ok(SparseCurve::with_param(points, param))
        })
        .collect()
}

/// Configuration of [`sparse_elastic_align_pair`].
///
/// # Examples
/// ```
/// use fdars_core::alignment::SparseAlignConfig;
/// let mut cfg = SparseAlignConfig::default();
/// cfg.closed = true;
/// assert_eq!(cfg.eps, 0.01);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseAlignConfig {
    /// Treat both curves as closed (the first point is appended when the curve
    /// does not end where it starts). Default `false`.
    pub closed: bool,
    /// Convergence tolerance: sweeps stop when no warp point moves by more than
    /// `eps / (number of points)` (elasdics `eps`). Default `0.01`.
    pub eps: f64,
}

impl Default for SparseAlignConfig {
    fn default() -> Self {
        Self {
            closed: false,
            eps: 0.01,
        }
    }
}

/// Result of [`sparse_elastic_align_pair`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseAlignmentResult {
    /// Template points after duplicate removal / closure, `d × m1`.
    pub template_points: FdMatrix,
    /// Parametrisation of the template points (length `m1`).
    pub template_param: Vec<f64>,
    /// Points of the aligned curve after duplicate removal / closure, `d × m2`.
    pub points: FdMatrix,
    /// Initial parametrisation of the aligned curve (length `m2`).
    pub param: Vec<f64>,
    /// Optimal parametrisation of the aligned curve's points (length `m2`;
    /// non-decreasing for open curves; for closed curves wrapped into `[0, 1]`
    /// as in elasdics, i.e. non-decreasing up to one wrap-around).
    pub param_optim: Vec<f64>,
    /// Elastic distance between the curves at the optimal parametrisation.
    pub distance: f64,
    /// Whether the curves were treated as closed.
    pub closed: bool,
    /// Whether duplicated consecutive points were removed from either curve.
    pub removed_duplicates: bool,
}

fn validate_eps(eps: f64) -> Result<(), FdarError> {
    if !(eps.is_finite() && eps > 0.0) {
        return Err(FdarError::InvalidParameter {
            parameter: "eps",
            message: format!("must be finite and > 0, got {eps}"),
        });
    }
    Ok(())
}

pub(super) fn to_points_matrix(d: usize, pts: &[f64]) -> Result<FdMatrix, FdarError> {
    FdMatrix::from_column_major(pts.to_vec(), d, pts.len() / d)
}

/// Elastic alignment of two sparsely observed curves (elasdics `align_curves`).
///
/// Finds the optimal reparametrisation of `curve2` to `curve1` with respect to
/// the elastic (SRV) distance, treating both as polygons through their points;
/// the warp is optimised at `curve2`'s points by coordinate-wise analytic
/// maximisation, starting from `curve2`'s arc-length parametrisation (open) or
/// from the best circular shift of it (closed). The curves may have different
/// numbers of points but must have the same dimension `d ≥ 1`.
///
/// For closed curves the reported distance is that of the optimal warp before
/// wrapping (see the module docs, deviation 1).
///
/// # Errors
/// [`FdarError::InvalidDimension`] for empty curves, fewer than 2 points or a
/// dimension mismatch; [`FdarError::InvalidParameter`] for non-finite points,
/// an invalid parametrisation or `eps <= 0`.
///
/// # Examples
/// ```
/// use fdars_core::alignment::{sparse_elastic_align_pair, SparseAlignConfig, SparseCurve};
/// use fdars_core::matrix::FdMatrix;
/// // elasdics `align_curves` example: a 4-point and a 6-point open curve in R^2
/// let c1 = FdMatrix::from_column_major(vec![1.0, 1.0, 0.5, -0.5, -1.0, -1.0, -1.0, 1.0], 2, 4)
///     .unwrap();
/// let mut v = Vec::new();
/// for i in 1..=6 {
///     let a = if i % 2 == 1 { 0.1 } else { 0.7 };
///     v.extend([a * (i as f64).sin(), (i as f64).cos()]);
/// }
/// let c2 = FdMatrix::from_column_major(v, 2, 6).unwrap();
/// let res = sparse_elastic_align_pair(
///     &SparseCurve::new(c1),
///     &SparseCurve::new(c2),
///     &SparseAlignConfig::default(),
/// )
/// .unwrap();
/// assert!((res.distance - 1.312_178_275_8).abs() < 1e-8); // elasdics 1.1.3
/// assert!(res.param_optim.windows(2).all(|w| w[1] >= w[0]));
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn sparse_elastic_align_pair(
    curve1: &SparseCurve,
    curve2: &SparseCurve,
    config: &SparseAlignConfig,
) -> Result<SparseAlignmentResult, FdarError> {
    validate_eps(config.eps)?;
    let closed = config.closed;
    let p1 = prepare_curve(curve1, closed, "curve1")?;
    let p2 = prepare_curve(curve2, closed, "curve2")?;
    if p1.d != p2.d {
        return Err(FdarError::InvalidDimension {
            parameter: "curve2",
            expected: format!("{} coordinates (as curve1)", p1.d),
            actual: p2.d.to_string(),
        });
    }
    let srv1 = srv_from_polygon(&p1, "curve1")?;
    let srv2 = srv_from_polygon(&p2, "curve2")?;
    let (param_optim, distance) = if closed {
        // pre-alignment over starting points of the arc-length parametrisation
        let t = arc_length_param(p2.d, &p2.pts);
        let mut best = (0usize, f64::INFINITY);
        for i in 0..srv2.nseg() {
            let shifted: Vec<f64> = t.iter().map(|v| v - t[i]).collect();
            let dist = compute_distance(&srv1, &srv2, &shifted, true);
            if dist < best.1 {
                best = (i, dist);
            }
        }
        let initial: Vec<f64> = t.iter().map(|v| v - t[best.0]).collect();
        let topt = find_optimal_t_discrete_closed(&srv1, &srv2, &initial, config.eps);
        let dist = compute_distance(&srv1, &srv2, &topt, true);
        let wrapped = topt
            .iter()
            .map(|&v| {
                let v = if v < 0.0 { v + 1.0 } else { v };
                if v > 1.0 {
                    v - 1.0
                } else {
                    v
                }
            })
            .collect();
        (wrapped, dist)
    } else {
        let reference = Reference {
            d: srv1.d,
            r: &srv1.t,
            p: &srv1.q,
        };
        let initial = arc_length_param(p2.d, &p2.pts);
        let topt = find_optimal_t_discrete(&reference, &srv2, &initial, config.eps);
        let dist = compute_distance(&srv1, &srv2, &topt, false);
        (topt, dist)
    };
    Ok(SparseAlignmentResult {
        template_points: to_points_matrix(p1.d, &p1.pts)?,
        template_param: p1.t.clone(),
        points: to_points_matrix(p2.d, &p2.pts)?,
        param: p2.t.clone(),
        param_optim,
        distance,
        closed,
        removed_duplicates: p1.removed || p2.removed,
    })
}
