//! Elastic shape analysis of curves in R^d (open curves and closed contours).
//!
//! A curve `β : [0, 1] → R^d` is represented by its square-root velocity
//! function (SRVF) `q = β'/√‖β'‖`. Shapes are equivalence classes of curves
//! modulo
//!
//! * **translation** — removed by the SRVF itself (curves are also centred on
//!   their arc-length centroid for output);
//! * **scale** — `scale = true` rescales every curve to unit length, so SRVFs
//!   live on the unit L² sphere (the *pre-shape sphere*) and the shape distance
//!   is the great-circle distance `arccos⟨q1, q2*⟩`; with `scale = false` the
//!   size is kept and the distance is the flat L² distance `‖q1 − q2*‖`;
//! * **rotation** — optimal `R ∈ SO(d)` by SVD Procrustes (Kabsch), optionally
//!   over O(d) (`allow_reflection`);
//! * **reparametrisation** — optimal warp `γ` by the shared fdasrvf
//!   coprime-neighbourhood dynamic program (summed over coordinates), or the
//!   Riemannian L-BFGS / finer-DP options of [`CurveShapeConfig::warp_optim`]
//!   that remove the DP's grid-independent slope-quantisation floor;
//! * **starting point** (closed contours) — seed search over circular shifts;
//!   closed SRVFs are projected onto the closed-curve manifold
//!   `{q : ∫ q‖q‖ = 0}` (Srivastava–Klassen).
//!
//! Rotation and reparametrisation are optimised by alternation (`align_iter`
//! rounds of Procrustes ↔ DP), which never does worse than the single
//! rotation-then-DP pass of fdasrsf `find_rotation_and_seed_coord`.
//!
//! Group statistics ([`curve_shape_karcher_mean`],
//! [`curve_shape_karcher_covariance`], [`curve_shape_pca`]) work in the tangent
//! space at the Karcher mean (sphere log/exp maps when scale is removed, flat
//! maps otherwise; closure-normal projection for closed contours), following
//! fdasrsf `fdacurve.karcher_mean` / `karcher_cov` / `shape_pca`; the Karcher
//! median ([`curve_shape_karcher_median`]) uses the Weiszfeld step of fdasrvf
//! `multivariate_karcher_mean(ms = "median")`.
//!
//! Downstream analyses: tangent-space (TSRVF) coordinates at the mean
//! ([`curve_shape_tsrvf`], [`curve_shape_tangent_to_curve`]), amplitude / phase
//! depth ([`curve_shape_depth`], fdasrvf `curve_depth`), shape outlier detection
//! ([`curve_shape_outlier_detection`]), the shape box plot
//! ([`curve_shape_boxplot`], fdasrvf `curve_boxplot`) and a bootstrap confidence
//! region of the mean shape ([`curve_shape_confidence_interval`]).
//!
//! **Data layout.** A single curve is an `FdMatrix` of shape `d × m` (row =
//! coordinate, column = sample) on the implicit uniform grid `t_j = j/(m−1)` of
//! `[0, 1]`; a set of `n` curves is an [`FdCurveSet`] (`d` matrices of shape
//! `n × m`) — convert with [`curve_from_set`] / [`curves_to_set`]. Closed
//! contours are sampled with the last point repeating the first
//! (`β(:, m−1) = β(:, 0)`), as in fdasrsf.
//!
//! **General (a,b) elastic metrics (planar curves).** With
//! [`CurveShapeConfig::metric`] set to [`CurveShapeMetric::Elastic`] every
//! function of this module works with the first-order elastic metric
//! `g_c(h, k) = ∫ a²⟨D_s h, N⟩⟨D_s k, N⟩ + b²⟨D_s h, T⟩⟨D_s k, T⟩ ds`
//! (`a` bending, `b` stretching) instead of the SRVF metric, through the
//! Kurtek–Needham transform [`ab_transform`]
//! `F_{a,b}(c) = 2b |c'|^{1/2} (c'/|c'|)^{a/(2b)}` that pulls the L² metric back
//! to `g^{a,b}`. The SRVF is the case `a = 1, b = 1/2` (geomstats
//! `ElasticMetric(a=1, b=0.5)`); see [`CurveShapeMetric`] for the conventions.
//! Closed contours are supported through the F-space closure condition
//! [`ab_closure_residual`] / projection [`ab_project_closed`]. For polygons,
//! [`ab_elastic_distance_pl`] computes the exact `(a, b)` distance modulo
//! reparametrisation (Bauer et al., J. Nonlinear Sci. 2024, §4.1) without any
//! discretisation of the matching.
//!
//! Reference: Srivastava & Klassen, *Functional and Shape Data Analysis*
//! (Springer 2016), ch. 5–6; fdasrsf 2.7.5 `curve_functions` / `curve_stats`;
//! Kurtek & Needham, "Simplifying transforms for general elastic metrics on the
//! space of plane curves", SIAM J. Imaging Sci. 13 (2020); geomstats 2.8.0
//! `discrete_curves.FTransform` / `ElasticMetric`.

#[cfg(test)]
mod ab_closed_tests;
#[cfg(test)]
mod ab_tests;
mod boxplot;
mod downstream;
#[cfg(test)]
mod downstream_tests;
mod geometry;
mod karcher;
mod model_ci;
#[cfg(test)]
mod model_ci_tests;
#[cfg(test)]
mod optim_tests;
mod pl_exact;
#[cfg(test)]
mod pl_exact_tests;
#[cfg(test)]
mod tests;

pub use boxplot::{
    curve_shape_boxplot, curve_shape_boxplot_from_aligned, curve_shape_confidence_interval,
    CurveShapeBoxplot, CurveShapeBoxplotConfig, CurveShapeCiConfig, CurveShapeCiMethod,
    CurveShapeCiResult,
};
pub use downstream::{
    curve_shape_depth, curve_shape_outlier_detection, curve_shape_tangent_to_curve,
    curve_shape_tsrvf, CurveShapeCentre, CurveShapeDepthResult, CurveShapeOutlierConfig,
    CurveShapeOutlierResult, CurveShapeTsrvfResult,
};
pub use karcher::{
    curve_shape_karcher_covariance, curve_shape_karcher_mean, curve_shape_karcher_median,
    curve_shape_pca, CurveShapeKarcherResult, CurveShapePcaResult,
};
pub use model_ci::CurveShapeModelCi;
pub use pl_exact::{ab_elastic_distance_pl, ab_elastic_match_pl, AbPlMatching};

use geometry::Grid;

use super::optim::WarpOptimConfig;
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::{FdCurveSet, FdMatrix};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ─── Configuration ──────────────────────────────────────────────────────────

/// Riemannian metric of the elastic shape space.
///
/// * [`Srvf`](Self::Srvf) (default) — the square-root velocity (Fisher–Rao)
///   metric, for curves in any R^d.
/// * [`Elastic { a, b }`](Self::Elastic) — the general first-order elastic
///   metric `g^{a,b}` for **planar** curves (`d = 2`), open or closed,
///   computed in the coordinates of the Kurtek–Needham transform
///   `F(c) = 2b |c'|^{1/2} e^{iλθ}`, `λ = a/(2b)`, `θ` the continuous lift of the
///   tangent angle (see [`ab_transform`]).
///
/// # Conventions
/// * Parameters follow Kurtek–Needham and geomstats 2.8.0 `ElasticMetric(a, b)`:
///   `a` weights bending (normal part of `D_s h`), `b` stretching (tangential
///   part). The SRVF metric is `a = 1, b = 1/2` (`λ = 1`, `2b = 1`); the metric
///   depends on the pair only up to an overall scale `2b` once `λ` is fixed.
///   `Elastic { a: 1.0, b: 0.5 }` reproduces [`Srvf`](Self::Srvf) up to
///   rounding.
/// * Reparametrisation acts on `F` exactly as on SRVFs,
///   `F(c∘γ) = (F(c)∘γ)·√γ'` (the direction factor is invariant under
///   orientation-preserving `γ`), so the shared DP engine is used unchanged.
/// * A rotation of the curve by `φ` multiplies `F` by `e^{iλφ}`; for non-integer
///   `λ` the transform is multivalued (branches differ by `e^{2πiλk}`), and the
///   code uses the `k = 0` branch `θ(0) = atan2 ∈ (−π, π]` like geomstats. With
///   `rotation = true` the optimal F-space rotation `ψ` over all of SO(2) is used,
///   which is the exact quotient over rotations **and** branches; the reported
///   curve rotation is `ψ/λ`. With `rotation = false` the distance is the plain
///   L² distance of the `k = 0` branches (geomstats `ElasticMetric.dist`), or,
///   with [`CurveShapeConfig::branch_search`], the minimum over the relative
///   branches `k ∈ {−1, 0, 1}`.
/// * Closed contours (`closed = true`): the curve rebuilt from `F` has
///   displacement `c(1) − c(0) = (4b²)⁻¹ ∫ |F|² e^{iφ/λ}` (`φ` the lifted `arg F`),
///   so the closed-curve manifold is `{F : ∫ |F|² e^{iφ/λ} = 0}` (`∫ q‖q‖ = 0`
///   for `λ = 1`); transforms are projected onto it ([`ab_project_closed`]) and
///   Karcher statistics use its tangent space. Starting points are searched as
///   for the SRVF.
/// * Scale: `‖F(c)‖² = 4b²·length(c)`. With `scale = false` the distance is
///   `‖F1 − F2*‖` (the geomstats pull-back distance). With `scale = true` the
///   transforms are normalised to the unit sphere and the distance is the
///   great-circle **angle** `θ ∈ [0, π]` (as for the SRVF); the `g^{a,b}`
///   geodesic length on unit-length curves is `2b·θ`.
/// * The trapezoidal rule on the full grid `t_j = j/(m−1)` with central
///   differences is used (as for the SRVF), whereas geomstats uses forward
///   differences and a left Riemann sum over `m − 1` segments; both are
///   second-order consistent and agree to ~1e-6 at `m = 2001`.
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_distance, CurveShapeConfig, CurveShapeMetric};
/// use fdars_core::matrix::FdMatrix;
/// let m = 60;
/// let mut b1 = FdMatrix::zeros(2, m);
/// let mut b2 = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let t = j as f64 / (m - 1) as f64;
///     b1[(0, j)] = t;
///     b1[(1, j)] = 0.3 * (3.0 * t).sin();
///     b2[(0, j)] = t;
///     b2[(1, j)] = 0.2 * (5.0 * t).sin();
/// }
/// let mut cfg = CurveShapeConfig::default();
/// let d_srvf = curve_shape_distance(&b1, &b2, &cfg).unwrap();
/// cfg.metric = CurveShapeMetric::Elastic { a: 1.0, b: 0.5 }; // = SRVF
/// let d_ab = curve_shape_distance(&b1, &b2, &cfg).unwrap();
/// assert!((d_srvf - d_ab).abs() < 1e-9);
/// cfg.metric = CurveShapeMetric::Elastic { a: 1.0, b: 1.0 }; // more stretch weight
/// assert!(curve_shape_distance(&b1, &b2, &cfg).unwrap() > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurveShapeMetric {
    /// Square-root velocity (Fisher–Rao) metric, any dimension.
    #[default]
    Srvf,
    /// General `(a, b)` elastic metric for planar curves (`a, b > 0`).
    Elastic {
        /// Bending weight `a > 0`.
        a: f64,
        /// Stretching weight `b > 0`.
        b: f64,
    },
}

impl CurveShapeMetric {
    /// The exponent `λ = a/(2b)` of the F-transform (1 for the SRVF).
    #[must_use]
    pub fn ratio(&self) -> f64 {
        match *self {
            Self::Srvf => 1.0,
            Self::Elastic { a, b } => a / (2.0 * b),
        }
    }

    /// Validate the metric parameters.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `a` or `b` is not finite and positive.
    pub fn validate(&self) -> Result<(), FdarError> {
        if let Self::Elastic { a, b } = *self {
            check_ab(a, b)?;
        }
        Ok(())
    }
}

/// Check the `(a, b)` parameters of the elastic metric.
fn check_ab(a: f64, b: f64) -> Result<(), FdarError> {
    for (parameter, v) in [("a", a), ("b", b)] {
        if !(v.is_finite() && v > 0.0) {
            return Err(FdarError::InvalidParameter {
                parameter,
                message: format!("must be finite and > 0, got {v}"),
            });
        }
    }
    Ok(())
}

/// Configuration of the shape quotient for R^d curve shape analysis.
///
/// Construct with `CurveShapeConfig::default()` and assign fields (the struct is
/// `#[non_exhaustive]`).
///
/// # Example
/// ```
/// use fdars_core::alignment::CurveShapeConfig;
/// let mut cfg = CurveShapeConfig::default();
/// cfg.closed = true; // closed contours: closure projection + seed search
/// assert!(cfg.rotation && cfg.scale && cfg.validate().is_ok());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CurveShapeConfig {
    /// Quotient out rotations (SVD Procrustes). Default `true`.
    pub rotation: bool,
    /// Allow reflections (optimise over O(d) instead of SO(d)). Default `false`.
    pub allow_reflection: bool,
    /// Quotient out scale (unit-length curves, pre-shape sphere geometry).
    /// Default `true`.
    pub scale: bool,
    /// With `scale = true`: add the log SRVF-norm ratio to the distance,
    /// `d = √(θ² + ln²(‖q1‖/‖q2‖))` (fdasrsf `elastic_distance_curve(scale=True)`).
    /// Default `false`.
    pub include_length: bool,
    /// Treat the curves as closed contours (closure projection and seed search).
    /// Default `false`.
    pub closed: bool,
    /// Search over starting points (closed contours only). Default `true`.
    pub seed_search: bool,
    /// Coarse seed stride in samples; the best coarse seed is refined at every
    /// sample within `±(seed_step − 1)`. Default 4 (fdasrsf's stride). Must be ≥ 1.
    pub seed_step: usize,
    /// Quotient out reparametrisations (elastic DP). Default `true`; `false`
    /// gives rigid (rotation/scale/seed only) Procrustes shape analysis.
    pub reparam: bool,
    /// Warp roughness penalty `λ ∫(γ' − 1)²` in the DP. Default 0.
    pub lambda: f64,
    /// Rounds of the Procrustes ↔ DP alternation per seed candidate (≥ 1).
    /// Default 2.
    pub align_iter: usize,
    /// Shape-space metric. Default [`CurveShapeMetric::Srvf`];
    /// [`CurveShapeMetric::Elastic`] requires planar curves (open or closed).
    pub metric: CurveShapeMetric,
    /// With [`CurveShapeMetric::Elastic`] and `rotation = false`: minimise the
    /// distance over the F-angle ambiguity. `F` is multivalued for non-integer
    /// `λ = a/(2b)` (branches differ by `e^{2πiλk}`) and the default compares the
    /// `k = 0` branches `θ(0) ∈ (−π, π]` of both curves (geomstats
    /// `ElasticMetric.dist`), so two curves whose initial tangents straddle the
    /// branch cut (angles near `±π`) are compared through a lifted angle
    /// difference near `2π`. With `branch_search = true` the transform of `β2`
    /// is also tried on the relative branches `k = ±1` (`F2 ↦ e^{±2πiλ}F2`,
    /// which cover every relative initial-angle lift in `(−π, π]`) and the
    /// best is kept; the curve itself is never rotated. Ignored with
    /// `rotation = true` (the SO(2) Procrustes in F-space already quotients the
    /// branches) and for the SRVF. Default `false`.
    ///
    /// ```
    /// use fdars_core::alignment::{curve_shape_distance, CurveShapeConfig, CurveShapeMetric};
    /// use fdars_core::matrix::FdMatrix;
    /// let m = 50;
    /// // two nearly identical leftward segments, initial angles just above/below π
    /// let mut b1 = FdMatrix::zeros(2, m);
    /// let mut b2 = FdMatrix::zeros(2, m);
    /// for j in 0..m {
    ///     let t = j as f64 / (m - 1) as f64;
    ///     b1[(0, j)] = -t;
    ///     b1[(1, j)] = 0.02 * t;
    ///     b2[(0, j)] = -t;
    ///     b2[(1, j)] = -0.02 * t;
    /// }
    /// let mut cfg = CurveShapeConfig::default();
    /// cfg.metric = CurveShapeMetric::Elastic { a: 1.0, b: 1.0 }; // λ = 1/2
    /// cfg.rotation = false;
    /// cfg.reparam = false;
    /// let d0 = curve_shape_distance(&b1, &b2, &cfg).unwrap();
    /// cfg.branch_search = true;
    /// let d1 = curve_shape_distance(&b1, &b2, &cfg).unwrap();
    /// assert!(d1 < 0.1 && d0 > 1.0);
    /// ```
    #[cfg_attr(feature = "serde", serde(default))]
    pub branch_search: bool,
    /// Warp optimiser of the reparametrisation step (default: the classic
    /// 35-move DP, bit-identical to the behaviour before this option).
    ///
    /// The DP restricts `γ'` to the slopes `{dr/dc : dr, dc ≤ nbhd_dim}` of its
    /// moves, so it cannot undo a smooth warp exactly: the self-distance
    /// `d(β, β∘γ)` stalls at a **grid-independent floor** (≈ 0.02 for the SRVF
    /// with the default 7-move neighbourhood, see GH #65) instead of shrinking
    /// with `m`. Recommended settings when that floor matters:
    ///
    /// * [`WarpOptimMethod::Rbfgs`](super::WarpOptimMethod::Rbfgs) — Riemannian L-BFGS refinement of a coarse
    ///   DP warp over smooth warps (fdasrsf `method="RBFGS"`, here summed over
    ///   the coordinates); with [`align_iter`](Self::align_iter) `≥ 4` (so the
    ///   rotation ↔ warp alternation settles) the residual drops to the
    ///   resampling floor of the exact warp, an order of magnitude below the DP
    ///   floor, and keeps shrinking with `m` (≈ 0.0019 at m = 81, 0.0005 at
    ///   m = 321);
    /// * `nbhd_dim = 10` + [`DpEdgeCost::PiecewiseLinear`](super::DpEdgeCost::PiecewiseLinear) (fdasrsf `DP2`) —
    ///   stays a global DP, lowers the floor moderately at ≈ 5× the DP cost.
    ///
    /// The warp penalty weight is always [`lambda`](Self::lambda) with the
    /// penalty shape `warp_optim.penalty` (`L2Gam` = the classic `∫(γ' − 1)²`);
    /// `warp_optim.lambda` must stay `0`. With RBFGS `lambda` has fdasrsf's
    /// RBFGS scale (see [`WarpObjectivePenalty`](super::WarpObjectivePenalty)),
    /// which for curves differs from the DP scale by `φ = scl/(e1·e2)`; set
    /// `warp_optim.lambda_scale` to convert (see
    /// [`WarpLambdaScale`](super::WarpLambdaScale)). The RBFGS coarse-DP seed
    /// honours `warp_optim.nbhd_dim` / `dp_scl` / `dp_cost`.
    ///
    /// ```
    /// use fdars_core::alignment::{curve_shape_distance, CurveShapeConfig, WarpOptimMethod};
    /// use fdars_core::matrix::FdMatrix;
    /// let m = 60;
    /// let curve = |g: &dyn Fn(f64) -> f64| {
    ///     let mut b = FdMatrix::zeros(2, m);
    ///     for j in 0..m {
    ///         let t = g(j as f64 / (m - 1) as f64);
    ///         b[(0, j)] = t;
    ///         b[(1, j)] = 0.3 * (6.0 * t).sin();
    ///     }
    ///     b
    /// };
    /// let b1 = curve(&|t| t);
    /// let b2 = curve(&|t| t + 0.25 * t * (1.0 - t)); // same shape, warped
    /// let dp = curve_shape_distance(&b1, &b2, &CurveShapeConfig::default()).unwrap();
    /// let mut cfg = CurveShapeConfig::default();
    /// cfg.warp_optim.method = WarpOptimMethod::Rbfgs;
    /// cfg.align_iter = 4;
    /// let rbfgs = curve_shape_distance(&b1, &b2, &cfg).unwrap();
    /// assert!(rbfgs < dp);
    /// ```
    #[cfg_attr(feature = "serde", serde(default))]
    pub warp_optim: WarpOptimConfig,
}

impl Default for CurveShapeConfig {
    fn default() -> Self {
        Self {
            rotation: true,
            allow_reflection: false,
            scale: true,
            include_length: false,
            closed: false,
            seed_search: true,
            seed_step: 4,
            reparam: true,
            lambda: 0.0,
            align_iter: 2,
            metric: CurveShapeMetric::Srvf,
            branch_search: false,
            warp_optim: WarpOptimConfig::default(),
        }
    }
}

impl CurveShapeConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `seed_step == 0`, `align_iter == 0`,
    /// `lambda` is negative / non-finite, the metric parameters are invalid,
    /// `warp_optim.lambda ≠ 0`, `warp_optim.band_frac ≠ 0` (no band for curve
    /// shapes), or `warp_optim` (with weight `lambda`) is invalid
    /// ([`WarpOptimConfig::validate`]).
    pub fn validate(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        if self.seed_step == 0 {
            return bad("seed_step", "must be >= 1".to_string());
        }
        if self.align_iter == 0 {
            return bad("align_iter", "must be >= 1".to_string());
        }
        if !(self.lambda.is_finite() && self.lambda >= 0.0) {
            return bad(
                "lambda",
                format!("must be finite and >= 0, got {}", self.lambda),
            );
        }
        self.metric.validate()?;
        if self.warp_optim.lambda != 0.0 {
            return bad(
                "warp_optim.lambda",
                format!(
                    "the warp penalty weight is CurveShapeConfig::lambda; leave \
                     warp_optim.lambda at 0 (got {})",
                    self.warp_optim.lambda
                ),
            );
        }
        if self.warp_optim.band_frac != 0.0 {
            return bad(
                "warp_optim.band_frac",
                format!(
                    "the Sakoe–Chiba band is not supported for curve shapes (got {})",
                    self.warp_optim.band_frac
                ),
            );
        }
        let mut warp = self.warp_optim.clone();
        warp.lambda = self.lambda;
        warp.validate()?;
        Ok(())
    }

    /// Check that curves of dimension `d` are supported by the metric.
    fn check_dim(&self, d: usize) -> Result<(), FdarError> {
        if self.metric != CurveShapeMetric::Srvf && d != 2 {
            return Err(FdarError::InvalidDimension {
                parameter: "curves",
                expected: "planar curves (d = 2) for the (a,b) elastic metric".to_string(),
                actual: format!("d = {d}"),
            });
        }
        Ok(())
    }
}

// ─── Results ────────────────────────────────────────────────────────────────

/// Result of aligning curve `beta2` to curve `beta1` in shape space.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct CurveShapeAlignment {
    /// Preprocessed target `β1` (centred; unit length if `scale`), `d × m`.
    pub beta1: FdMatrix,
    /// `β2` optimally aligned to `β1`: `R · β2(shifted)∘γ`, centred (unit
    /// length if `scale`), `d × m`.
    pub beta2_aligned: FdMatrix,
    /// SRVF of the preprocessed target (closure-projected if `closed`), `d × m`
    /// — the F-transform under [`CurveShapeMetric::Elastic`] (unit norm if
    /// `scale`).
    pub q1: FdMatrix,
    /// SRVF (F-transform) of the aligned `β2` (closure-projected if `closed`),
    /// `d × m`. Under the elastic metric it is `R_F·F(β2∘γ)`, a branch of the
    /// transform of `beta2_aligned`.
    pub q2_aligned: FdMatrix,
    /// Optimal rotation `R` (`d × d`), applied as `R · β2` (under the elastic
    /// metric the curve rotation `Rot(ψ/λ)` of the F-space rotation `Rot(ψ)`).
    pub rotation: FdMatrix,
    /// Optimal warp `γ` on the grid `t_j = j/(m−1)` (identity if `reparam = false`).
    pub gamma: Vec<f64>,
    /// Optimal starting-point shift (samples) applied to `β2` before warping
    /// (0 for open curves).
    pub seed: usize,
    /// Length `∫‖β1'‖` of the input `β1`.
    pub length1: f64,
    /// Length `∫‖β2'‖` of the input `β2`.
    pub length2: f64,
    /// Shape distance after alignment.
    pub distance: f64,
}

// ─── Public geometry helpers ────────────────────────────────────────────────

/// Extract curve `i` of a set as a `d × m` matrix.
///
/// # Errors
/// [`FdarError::InvalidParameter`] if `i` is out of range.
pub fn curve_from_set(curves: &FdCurveSet, i: usize) -> Result<FdMatrix, FdarError> {
    if i >= curves.ncurves() {
        return Err(FdarError::InvalidParameter {
            parameter: "i",
            message: format!("index {i} out of range ({} curves)", curves.ncurves()),
        });
    }
    let (d, m) = (curves.ndim(), curves.npoints());
    let mut out = FdMatrix::zeros(d, m);
    for (k, dim) in curves.dims.iter().enumerate() {
        for j in 0..m {
            out[(k, j)] = dim[(i, j)];
        }
    }
    Ok(out)
}

/// Stack `d × m` curves into an [`FdCurveSet`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `curves` is empty or shapes differ.
pub fn curves_to_set(curves: &[FdMatrix]) -> Result<FdCurveSet, FdarError> {
    let first = curves.first().ok_or_else(|| FdarError::InvalidDimension {
        parameter: "curves",
        expected: "at least one curve".to_string(),
        actual: "0".to_string(),
    })?;
    let (d, m) = first.shape();
    let n = curves.len();
    let mut dims: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(n, m)).collect();
    for (i, c) in curves.iter().enumerate() {
        if c.shape() != (d, m) {
            return Err(FdarError::InvalidDimension {
                parameter: "curves",
                expected: format!("all curves {d} x {m}"),
                actual: format!("curve {i}: {} x {}", c.nrows(), c.ncols()),
            });
        }
        for (k, dim) in dims.iter_mut().enumerate() {
            for j in 0..m {
                dim[(i, j)] = c[(k, j)];
            }
        }
    }
    Ok(FdCurveSet { dims })
}

/// Validate a single `d × m` curve.
fn validate_curve(beta: &FdMatrix, parameter: &'static str) -> Result<(), FdarError> {
    let (d, m) = beta.shape();
    if d == 0 || m < 4 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "d >= 1 coordinates and m >= 4 samples (d x m matrix)".to_string(),
            actual: format!("{d} x {m}"),
        });
    }
    if beta.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter,
            message: "contains non-finite values".to_string(),
        });
    }
    Ok(())
}

/// Square-root velocity function `q = β'/√‖β'‖` of a `d × m` curve on the
/// uniform grid of `[0, 1]`; with `scale = true` it is normalised to unit L²
/// norm (equivalently, the curve is rescaled to unit length).
///
/// # Errors
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for an
/// invalid curve; [`FdarError::ComputationFailed`] if `scale` is requested for a
/// zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::curve_srvf;
/// use fdars_core::matrix::FdMatrix;
/// // straight segment of length 2 along the x axis
/// let m = 11;
/// let mut beta = FdMatrix::zeros(2, m);
/// for j in 0..m { beta[(0, j)] = 2.0 * j as f64 / (m - 1) as f64; }
/// let q = curve_srvf(&beta, false).unwrap();
/// assert!((q[(0, 5)] - 2f64.sqrt()).abs() < 1e-12 && q[(1, 5)].abs() < 1e-12);
/// ```
pub fn curve_srvf(beta: &FdMatrix, scale: bool) -> Result<FdMatrix, FdarError> {
    validate_curve(beta, "beta")?;
    let grid = Grid::new(beta.ncols());
    let q = geometry::srvf_raw(beta, &grid);
    if !scale {
        return Ok(q);
    }
    let nq = geometry::norm(&q, &grid);
    if nq < 1e-300 {
        return Err(FdarError::ComputationFailed {
            operation: "curve_srvf",
            detail: "zero-length curve cannot be scaled".to_string(),
        });
    }
    Ok(geometry::scaled(&q, 1.0 / nq))
}

/// Reconstruct a curve from its SRVF, `β(t) = ∫₀ᵗ q‖q‖`, starting at the origin.
///
/// # Errors
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for an
/// invalid `d × m` input.
pub fn srvf_to_curve(q: &FdMatrix) -> Result<FdMatrix, FdarError> {
    validate_curve(q, "q")?;
    Ok(geometry::q_to_curve(q, &Grid::new(q.ncols())))
}

/// Validate a planar curve / transform and the `(a, b)` parameters.
fn validate_planar(x: &FdMatrix, parameter: &'static str, a: f64, b: f64) -> Result<(), FdarError> {
    validate_curve(x, parameter)?;
    if x.nrows() != 2 {
        return Err(FdarError::InvalidDimension {
            parameter,
            expected: "2 x m (planar curve)".to_string(),
            actual: format!("{} x {}", x.nrows(), x.ncols()),
        });
    }
    check_ab(a, b)
}

/// Kurtek–Needham F-transform of a planar `2 × m` curve,
/// `F_{a,b}(c) = 2b |c'|^{1/2} (c'/|c'|)^{a/(2b)}` (complex power), whose L²
/// metric pulls back to the `(a, b)` elastic metric
/// `∫ a²⟨D_s h, N⟩⟨D_s k, N⟩ + b²⟨D_s h, T⟩⟨D_s k, T⟩ ds`.
///
/// The power is taken on the continuous lift `θ(t)` of the tangent angle with
/// `θ(0) = atan2(c'(0)) ∈ (−π, π]` — the `k = 0` branch of geomstats 2.8.0
/// `FTransform`. `c'` is computed by central differences on the uniform grid
/// of `[0, 1]` (as [`curve_srvf`]); `a = 1, b = 1/2` gives the SRVF. See
/// [`CurveShapeMetric`] for the conventions.
///
/// # Errors
/// [`FdarError::InvalidDimension`] unless `beta` is `2 × m` with `m ≥ 4`;
/// [`FdarError::InvalidParameter`] for non-finite values or `a, b ≤ 0`.
///
/// # Example
/// ```
/// use fdars_core::alignment::{ab_transform, ab_transform_inverse, curve_srvf};
/// use fdars_core::matrix::FdMatrix;
/// let m = 50;
/// let mut beta = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let t = j as f64 / (m - 1) as f64;
///     beta[(0, j)] = t;
///     beta[(1, j)] = 0.4 * (4.0 * t).sin();
/// }
/// // a = 1, b = 1/2 is the square-root velocity function
/// let f = ab_transform(&beta, 1.0, 0.5).unwrap();
/// let q = curve_srvf(&beta, false).unwrap();
/// assert!(f.as_slice().iter().zip(q.as_slice()).all(|(x, y)| (x - y).abs() < 1e-12));
/// // the inverse recovers the curve (up to translation and quadrature error)
/// let f = ab_transform(&beta, 1.0, 1.0).unwrap();
/// let back = ab_transform_inverse(&f, 1.0, 1.0).unwrap();
/// assert!((back[(1, m - 1)] - (beta[(1, m - 1)] - beta[(1, 0)])).abs() < 1e-2);
/// ```
pub fn ab_transform(beta: &FdMatrix, a: f64, b: f64) -> Result<FdMatrix, FdarError> {
    validate_planar(beta, "beta", a, b)?;
    Ok(geometry::f_transform(beta, &Grid::new(beta.ncols()), a, b))
}

/// Inverse of [`ab_transform`]: the planar curve starting at the origin with
/// speed `|F|²/(4b²)` and tangent angle `φ(t)/λ`, `λ = a/(2b)`, where `φ` is the
/// continuous lift of `arg F` with `φ(0) = atan2 ∈ (−π, π]` (geomstats 2.8.0
/// `FTransform.inverse`; integrated by the cumulative trapezoid rule).
///
/// The F-transform is a bijection only for `λ = 1`. For `λ ≤ 1` this is a left
/// inverse (`ab_transform_inverse(ab_transform(c)) = c − c(0)` up to
/// quadrature error). For `λ ≥ 1` it is a right inverse: it returns one curve
/// among the several with transform `f` (the one whose initial tangent angle
/// lies in `(−π/λ, π/λ]`).
///
/// # Errors
/// As [`ab_transform`].
pub fn ab_transform_inverse(f: &FdMatrix, a: f64, b: f64) -> Result<FdMatrix, FdarError> {
    validate_planar(f, "f", a, b)?;
    Ok(geometry::f_to_curve(f, &Grid::new(f.ncols()), a, b))
}

/// Closure residual `∫ q(t)‖q(t)‖ dt ∈ R^d` of an SRVF — the displacement
/// `β(1) − β(0)` of the curve it represents (zero for a closed curve).
///
/// # Errors
/// As [`srvf_to_curve`].
pub fn closure_residual(q: &FdMatrix) -> Result<Vec<f64>, FdarError> {
    validate_curve(q, "q")?;
    Ok(geometry::closure_vec(q, &Grid::new(q.ncols())))
}

/// Project an SRVF onto the closed-curve manifold `{q : ∫ q‖q‖ = 0}`
/// (Srivastava–Klassen; fdasrsf `project_curve`, generalised to R^d with the
/// exact Jacobian). The L² norm is preserved; on return the closure residual is
/// below `1e-10 · ‖q‖²`.
///
/// # Errors
/// As [`srvf_to_curve`], plus [`FdarError::ComputationFailed`] if `d < 2` or the
/// Newton iteration does not reach the tolerance.
///
/// # Example
/// ```
/// use fdars_core::alignment::{closure_residual, curve_srvf, project_closed_srvf};
/// use fdars_core::matrix::FdMatrix;
/// // an open arc (three quarters of a circle)
/// let m = 101;
/// let mut beta = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let th = 1.5 * std::f64::consts::PI * j as f64 / (m - 1) as f64;
///     beta[(0, j)] = th.cos();
///     beta[(1, j)] = th.sin();
/// }
/// let q = curve_srvf(&beta, true).unwrap();
/// let qc = project_closed_srvf(&q).unwrap();
/// let r = closure_residual(&qc).unwrap();
/// assert!(r.iter().all(|v| v.abs() < 1e-9));
/// ```
pub fn project_closed_srvf(q: &FdMatrix) -> Result<FdMatrix, FdarError> {
    validate_curve(q, "q")?;
    if q.nrows() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "q",
            expected: "d >= 2 for closed curves".to_string(),
            actual: format!("d = {}", q.nrows()),
        });
    }
    let (qc, ok) = geometry::project_closed(q, &Grid::new(q.ncols()));
    if !ok {
        return Err(FdarError::ComputationFailed {
            operation: "project_closed_srvf",
            detail: "closure projection did not converge".to_string(),
        });
    }
    Ok(qc)
}

/// Closure residual of a planar `(a, b)` F-transform: the displacement
/// `c(1) − c(0) = (4b²)⁻¹ ∫ |F|² e^{iφ/λ} dt` of the curve
/// [`ab_transform_inverse`] rebuilds from `f` (`λ = a/(2b)`, `φ` the continuous
/// lift of `arg F`). Zero iff that curve is closed; for `a = 1, b = 1/2` it is
/// [`closure_residual`].
///
/// **Derivation.** The inverse transform is `c' = |F|²/(4b²) · e^{iφ/λ}`
/// (speed `|F|²/(4b²)`, tangent angle `φ/λ`), so `c(1) − c(0) = ∫ c'`. A
/// different starting branch rotates the rebuilt curve rigidly and leaves
/// closure unchanged; the residual is homogeneous of degree 2 in `F`.
///
/// # Errors
/// As [`ab_transform_inverse`].
///
/// # Example
/// ```
/// use fdars_core::alignment::{ab_closure_residual, ab_transform};
/// use fdars_core::matrix::FdMatrix;
/// let m = 201;
/// let mut circle = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let th = 2.0 * std::f64::consts::PI * j as f64 / (m - 1) as f64;
///     circle[(0, j)] = th.cos();
///     circle[(1, j)] = th.sin();
/// }
/// let f = ab_transform(&circle, 1.0, 1.0).unwrap(); // λ = 1/2
/// let r = ab_closure_residual(&f, 1.0, 1.0).unwrap();
/// assert!(r.iter().all(|v| v.abs() < 1e-3)); // closed up to quadrature error
/// ```
pub fn ab_closure_residual(f: &FdMatrix, a: f64, b: f64) -> Result<Vec<f64>, FdarError> {
    validate_planar(f, "f", a, b)?;
    let g = geometry::closure_vec_ab(f, &Grid::new(f.ncols()), a / (2.0 * b));
    Ok(g.iter().map(|v| v / (4.0 * b * b)).collect())
}

/// Project a planar `(a, b)` F-transform onto the closed-curve manifold
/// `{F : ∫ |F|² e^{iφ/λ} = 0}` (see [`ab_closure_residual`]): Gauss–Newton
/// iterations along the closure normals
/// `b_x = 2cos(φ/λ)·F − λ⁻¹ sin(φ/λ)·iF`, `b_y = 2sin(φ/λ)·F + λ⁻¹ cos(φ/λ)·iF`
/// (the L² gradients of the residual) with the exact Gram-matrix Jacobian and a
/// backtracking line search. The L² norm is preserved; on return the residual
/// of the unit-norm copy is below `1e-10`. For `λ = a/(2b) = 1` this is
/// [`project_closed_srvf`] (bit-identical).
///
/// # Errors
/// As [`ab_transform_inverse`], plus [`FdarError::ComputationFailed`] if the
/// iteration does not reach the tolerance.
///
/// # Example
/// ```
/// use fdars_core::alignment::{ab_closure_residual, ab_project_closed, ab_transform};
/// use fdars_core::alignment::ab_transform_inverse;
/// use fdars_core::matrix::FdMatrix;
/// // an open arc (three quarters of a circle)
/// let m = 101;
/// let mut beta = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let th = 1.5 * std::f64::consts::PI * j as f64 / (m - 1) as f64;
///     beta[(0, j)] = th.cos();
///     beta[(1, j)] = th.sin();
/// }
/// let (a, b) = (1.0, 2.0); // λ = 1/4
/// let f = ab_project_closed(&ab_transform(&beta, a, b).unwrap(), a, b).unwrap();
/// assert!(ab_closure_residual(&f, a, b).unwrap().iter().all(|v| v.abs() < 1e-9));
/// let c = ab_transform_inverse(&f, a, b).unwrap();
/// assert!((c[(0, m - 1)] - c[(0, 0)]).abs() < 1e-9 && (c[(1, m - 1)] - c[(1, 0)]).abs() < 1e-9);
/// ```
pub fn ab_project_closed(f: &FdMatrix, a: f64, b: f64) -> Result<FdMatrix, FdarError> {
    validate_planar(f, "f", a, b)?;
    let (fc, ok) = geometry::project_closed_lam(f, &Grid::new(f.ncols()), a / (2.0 * b));
    if !ok {
        return Err(FdarError::ComputationFailed {
            operation: "ab_project_closed",
            detail: "F-space closure projection did not converge".to_string(),
        });
    }
    Ok(fc)
}

/// Optimal rotation `R = argmax ⟨q1, R q2⟩` over SO(d) (or O(d) with
/// `allow_reflection`) by SVD Procrustes (Kabsch). Exact for `q1 = R₀ q2`.
///
/// Inputs are `d × m` SRVFs (or any `d × m` vector fields; for curves of equal
/// parametrisation, passing SRVFs gives the shape-space optimum).
///
/// # Errors
/// [`FdarError::InvalidDimension`] if shapes differ or are invalid;
/// [`FdarError::ComputationFailed`] if the SVD fails.
///
/// # Example
/// ```
/// use fdars_core::alignment::optimal_rotation;
/// use fdars_core::matrix::FdMatrix;
/// let m = 20;
/// let (c, s) = (0.6f64.cos(), 0.6f64.sin());
/// let mut q2 = FdMatrix::zeros(2, m);
/// let mut q1 = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let t = j as f64 / (m - 1) as f64;
///     q2[(0, j)] = 1.0 + t;
///     q2[(1, j)] = (3.0 * t).sin();
///     q1[(0, j)] = c * q2[(0, j)] - s * q2[(1, j)];
///     q1[(1, j)] = s * q2[(0, j)] + c * q2[(1, j)];
/// }
/// let r = optimal_rotation(&q1, &q2, false).unwrap();
/// assert!((r[(0, 0)] - c).abs() < 1e-12 && (r[(1, 0)] - s).abs() < 1e-12);
/// ```
pub fn optimal_rotation(
    q1: &FdMatrix,
    q2: &FdMatrix,
    allow_reflection: bool,
) -> Result<FdMatrix, FdarError> {
    validate_curve(q1, "q1")?;
    validate_curve(q2, "q2")?;
    check_same_shape(q1, q2)?;
    geometry::procrustes(q1, q2, &Grid::new(q1.ncols()), allow_reflection)
}

fn check_same_shape(a: &FdMatrix, b: &FdMatrix) -> Result<(), FdarError> {
    if a.shape() != b.shape() {
        return Err(FdarError::InvalidDimension {
            parameter: "beta2",
            expected: format!("{} x {}", a.nrows(), a.ncols()),
            actual: format!("{} x {}", b.nrows(), b.ncols()),
        });
    }
    Ok(())
}

// ─── Internal: preprocessing and the alignment core ────────────────────────

/// A curve brought to the representation used by the alignment core.
#[derive(Debug, Clone)]
struct Prepared {
    /// Centred curve (unit length if `scale`).
    pub beta: FdMatrix,
    /// Its SRVF (unit norm if `scale`, closure-projected if `closed`).
    pub q: FdMatrix,
    /// Length of the original curve.
    pub length: f64,
}

/// SRVF of an (already centred / scaled) curve under the configuration:
/// renormalised to unit norm if `scale`, projected if `closed`.
fn srvf_cfg(beta: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    let mut q = match cfg.metric {
        CurveShapeMetric::Srvf => geometry::srvf_raw(beta, grid),
        CurveShapeMetric::Elastic { a, b } => geometry::f_transform(beta, grid, a, b),
    };
    if cfg.scale {
        let nq = geometry::norm(&q, grid);
        if nq > 1e-300 {
            q = geometry::scaled(&q, 1.0 / nq);
        }
    }
    if cfg.closed {
        q = geometry::project_closed_lam(&q, grid, cfg.metric.ratio()).0;
    }
    q
}

/// Reconstruct a (non-centred) curve from a configured transform: the SRVF
/// inverse, or the F-transform inverse rescaled so that a unit-norm transform
/// gives a unit-length curve when `scale` (`‖F‖² = 4b²·length`).
fn transform_to_curve(q: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    match cfg.metric {
        CurveShapeMetric::Srvf => geometry::q_to_curve(q, grid),
        CurveShapeMetric::Elastic { a, b } => {
            let f = if cfg.scale {
                geometry::scaled(q, 2.0 * b)
            } else {
                q.clone()
            };
            geometry::f_to_curve(&f, grid, a, b)
        }
    }
}

/// Map an optimal transform-space rotation `R_F` to the rotation of the curve:
/// identity map for the SRVF; for the `(a, b)` metric `R_F = Rot(ψ)·S`
/// (`S` = reflection or identity) corresponds to the curve map `Rot(ψ/λ)·S`.
fn curve_rotation(r: FdMatrix, cfg: &CurveShapeConfig) -> FdMatrix {
    match cfg.metric {
        CurveShapeMetric::Srvf => r,
        CurveShapeMetric::Elastic { .. } => {
            let psi = r[(1, 0)].atan2(r[(0, 0)]);
            let (s, c) = (psi / cfg.metric.ratio()).sin_cos();
            let refl = if r[(0, 0)] * r[(1, 1)] - r[(0, 1)] * r[(1, 0)] < 0.0 {
                -1.0
            } else {
                1.0
            };
            let mut out = FdMatrix::zeros(2, 2);
            out[(0, 0)] = c;
            out[(1, 0)] = s;
            out[(0, 1)] = -s * refl;
            out[(1, 1)] = c * refl;
            out
        }
    }
}

/// Centre, scale and transform one curve.
fn prepare(beta: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> Result<Prepared, FdarError> {
    let length = geometry::curve_length(beta, grid);
    if length.is_nan() || length <= 1e-300 {
        return Err(FdarError::ComputationFailed {
            operation: "curve shape preprocessing",
            detail: "zero-length (constant) curve has no shape".to_string(),
        });
    }
    let mut b = geometry::center(beta, grid);
    if cfg.scale {
        b = geometry::scaled(&b, 1.0 / length);
    }
    let q = srvf_cfg(&b, cfg, grid);
    Ok(Prepared { beta: b, q, length })
}

/// Shape distance between two configured SRVFs (without the length term).
fn srvf_distance(q1: &FdMatrix, q2: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> f64 {
    if cfg.scale {
        // Great-circle angle via the chord, θ = 2·asin(‖u1 − u2‖/2) for unit
        // u1, u2 — numerically stable near 0, unlike arccos⟨u1, u2⟩.
        let (n1, n2) = (geometry::norm(q1, grid), geometry::norm(q2, grid));
        if n1 <= 0.0 || n2 <= 0.0 {
            return 0.0;
        }
        let chord = geometry::norm(&geometry::axpby(1.0 / n1, q1, -1.0 / n2, q2), grid);
        2.0 * (0.5 * chord).min(1.0).asin()
    } else {
        geometry::norm(&geometry::axpby(1.0, q1, -1.0, q2), grid)
    }
}

/// One candidate alignment of β2 to the target.
#[derive(Debug, Clone)]
struct Candidate {
    pub rotation: FdMatrix,
    pub gamma: Vec<f64>,
    pub seed: usize,
    pub beta: FdMatrix,
    pub q: FdMatrix,
    pub distance: f64,
}

/// Build a candidate from a (rotation, warp) of the seed-shifted curve.
fn make_candidate(
    q1: &FdMatrix,
    r: FdMatrix,
    gamma: Vec<f64>,
    seed: usize,
    beta_w: &FdMatrix,
    q_w: &FdMatrix,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> Candidate {
    let q = geometry::rotate(&r, q_w);
    // Without rotation the curve is never rotated (a branch change of the
    // F-transform under `branch_search` is not a curve rotation).
    let rotation = if cfg.rotation {
        curve_rotation(r, cfg)
    } else {
        geometry::identity(q_w.nrows())
    };
    let beta = geometry::rotate(&rotation, beta_w);
    let distance = srvf_distance(q1, &q, cfg, grid);
    Candidate {
        rotation,
        gamma,
        seed,
        beta,
        q,
        distance,
    }
}

/// Procrustes rotation if enabled; otherwise the identity, or (elastic metric
/// with `branch_search`) the best relative F-branch rotation.
fn best_rotation(q1: &FdMatrix, q2: &FdMatrix, cfg: &CurveShapeConfig, grid: &Grid) -> FdMatrix {
    if cfg.rotation {
        geometry::procrustes(q1, q2, grid, cfg.allow_reflection)
            .unwrap_or_else(|_| geometry::identity(q1.nrows()))
    } else if cfg.branch_search && matches!(cfg.metric, CurveShapeMetric::Elastic { .. }) {
        best_branch(q1, q2, cfg.metric.ratio(), grid)
    } else {
        geometry::identity(q1.nrows())
    }
}

/// F-space rotation `Rot(2πλk)`, `k ∈ {−1, 0, 1}`, maximising `⟨q1, R q2⟩`
/// (ties keep `k = 0`).
fn best_branch(q1: &FdMatrix, q2: &FdMatrix, lam: f64, grid: &Grid) -> FdMatrix {
    let rot = |k: f64| {
        let (s, c) = (2.0 * std::f64::consts::PI * lam * k).sin_cos();
        let mut r = FdMatrix::zeros(2, 2);
        r[(0, 0)] = c;
        r[(1, 0)] = s;
        r[(0, 1)] = -s;
        r[(1, 1)] = c;
        r
    };
    let mut best = geometry::identity(2);
    let mut best_ip = geometry::inner(q1, q2, grid);
    for k in [-1.0, 1.0] {
        let r = rot(k);
        let ip = geometry::inner(q1, &geometry::rotate(&r, q2), grid);
        if ip > best_ip {
            best_ip = ip;
            best = r;
        }
    }
    best
}

/// Align a prepared curve with a fixed starting point: alternate Procrustes and
/// DP for `align_iter` rounds and keep the best candidate.
fn align_fixed_seed(
    q1: &FdMatrix,
    beta2: &FdMatrix,
    seed: usize,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> Candidate {
    let q2 = srvf_cfg(beta2, cfg, grid);
    let mut r = best_rotation(q1, &q2, cfg, grid);
    let mut best = make_candidate(q1, r.clone(), grid.t.clone(), seed, beta2, &q2, cfg, grid);
    if !cfg.reparam {
        return best;
    }
    for _ in 0..cfg.align_iter {
        let q2r = geometry::rotate(&r, &q2);
        let gamma = geometry::optimal_warp(q1, &q2r, grid, cfg.lambda, &cfg.warp_optim);
        let beta_w = geometry::warp_curve(beta2, grid, &gamma);
        let q_w = srvf_cfg(&beta_w, cfg, grid);
        r = best_rotation(q1, &q_w, cfg, grid);
        let cand = make_candidate(q1, r.clone(), gamma, seed, &beta_w, &q_w, cfg, grid);
        if cand.distance < best.distance {
            best = cand;
        }
    }
    best
}

/// Starting-point candidates: coarse stride over the `m − 1` distinct samples.
fn coarse_seeds(m: usize, cfg: &CurveShapeConfig) -> Vec<usize> {
    if !(cfg.closed && cfg.seed_search) {
        return vec![0];
    }
    (0..m - 1).step_by(cfg.seed_step).collect()
}

/// Align a prepared `β2` to the target SRVF `q1` over rotation, warp and (for
/// closed curves) starting point.
fn align_to_srvf(
    q1: &FdMatrix,
    beta2: &FdMatrix,
    cfg: &CurveShapeConfig,
    grid: &Grid,
) -> Candidate {
    let m = grid.len();
    let eval = |s: usize| {
        let b = if s == 0 {
            beta2.clone()
        } else {
            geometry::shift_closed(beta2, s)
        };
        align_fixed_seed(q1, &b, s, cfg, grid)
    };
    let pick = |a: Candidate, b: Candidate| if b.distance < a.distance { b } else { a };
    let seeds = coarse_seeds(m, cfg);
    let mut best = seeds
        .iter()
        .map(|&s| eval(s))
        .reduce(pick)
        .expect("at least one seed candidate");
    if cfg.closed && cfg.seed_search && cfg.seed_step > 1 {
        let p = m - 1;
        let s0 = best.seed;
        for off in 1..cfg.seed_step {
            for s in [(s0 + off) % p, (s0 + p - off % p) % p] {
                best = pick(best, eval(s));
            }
        }
    }
    best
}

/// Total distance including the optional length term.
fn total_distance(theta: f64, len1: f64, len2: f64, cfg: &CurveShapeConfig) -> f64 {
    if cfg.scale && cfg.include_length {
        let l = (len1.sqrt() / len2.sqrt()).ln();
        (theta * theta + l * l).sqrt()
    } else {
        theta
    }
}

/// Validate a pair of curves against the configuration.
fn validate_pair(
    beta1: &FdMatrix,
    beta2: &FdMatrix,
    cfg: &CurveShapeConfig,
) -> Result<(), FdarError> {
    cfg.validate()?;
    validate_curve(beta1, "beta1")?;
    validate_curve(beta2, "beta2")?;
    check_same_shape(beta1, beta2)?;
    cfg.check_dim(beta1.nrows())?;
    if cfg.closed && beta1.nrows() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "beta1",
            expected: "d >= 2 for closed curves".to_string(),
            actual: format!("d = {}", beta1.nrows()),
        });
    }
    Ok(())
}

// ─── Public pairwise API ────────────────────────────────────────────────────

/// Align curve `beta2` to curve `beta1` in elastic shape space: optimal
/// rotation (SVD Procrustes), scale (unit length), reparametrisation (DP) and,
/// for closed contours, starting point — see [`CurveShapeConfig`].
///
/// `beta1`, `beta2` are `d × m` matrices (row = coordinate) sampled on a common
/// uniform parameter grid; closed contours repeat their first point at the end.
///
/// # Errors
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for
/// mismatched or invalid inputs or configuration; [`FdarError::ComputationFailed`]
/// for a zero-length curve.
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_align_pair, CurveShapeConfig};
/// use fdars_core::matrix::FdMatrix;
/// let m = 40;
/// let (c, s) = (1.0f64.cos(), 1.0f64.sin());
/// let mut b1 = FdMatrix::zeros(2, m);
/// let mut b2 = FdMatrix::zeros(2, m);
/// for j in 0..m {
///     let t = j as f64 / (m - 1) as f64;
///     let (x, y) = (t, 0.3 * (6.0 * t).sin());
///     b1[(0, j)] = x;
///     b1[(1, j)] = y;
///     // rotated by 1 rad, scaled by 3 and translated
///     b2[(0, j)] = 3.0 * (c * x - s * y) + 5.0;
///     b2[(1, j)] = 3.0 * (s * x + c * y) - 2.0;
/// }
/// let res = curve_shape_align_pair(&b1, &b2, &CurveShapeConfig::default()).unwrap();
/// assert!(res.distance < 1e-6);
/// assert!((res.rotation[(0, 0)] - c).abs() < 1e-6); // undoes the rotation
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_align_pair(
    beta1: &FdMatrix,
    beta2: &FdMatrix,
    config: &CurveShapeConfig,
) -> Result<CurveShapeAlignment, FdarError> {
    validate_pair(beta1, beta2, config)?;
    let grid = Grid::new(beta1.ncols());
    let p1 = prepare(beta1, config, &grid)?;
    let p2 = prepare(beta2, config, &grid)?;
    let best = align_to_srvf(&p1.q, &p2.beta, config, &grid);
    let distance = total_distance(best.distance, p1.length, p2.length, config);
    Ok(CurveShapeAlignment {
        beta1: p1.beta,
        beta2_aligned: geometry::center(&best.beta, &grid),
        q1: p1.q,
        q2_aligned: best.q,
        rotation: best.rotation,
        gamma: best.gamma,
        seed: best.seed,
        length1: p1.length,
        length2: p2.length,
        distance,
    })
}

/// Elastic shape distance between two `d × m` curves under the quotient set by
/// `config` (see [`curve_shape_align_pair`]).
///
/// With `scale = true` this is the geodesic distance `arccos⟨q1, q2*⟩` on the
/// pre-shape sphere (fdasrsf `elastic_distance_curve`, fdasrvf
/// `calc_shape_dist`); with `scale = false` the L² distance `‖q1 − q2*‖`.
///
/// # Errors
/// As [`curve_shape_align_pair`].
///
/// # Example
/// ```
/// use fdars_core::alignment::{curve_shape_distance, CurveShapeConfig};
/// use fdars_core::matrix::FdMatrix;
/// let m = 41;
/// let ellipse = |a: f64, b: f64, phase: f64| {
///     let mut e = FdMatrix::zeros(2, m);
///     for j in 0..m {
///         let th = 2.0 * std::f64::consts::PI * j as f64 / (m - 1) as f64 + phase;
///         e[(0, j)] = a * th.cos();
///         e[(1, j)] = b * th.sin();
///     }
///     e
/// };
/// let mut cfg = CurveShapeConfig::default();
/// cfg.closed = true;
/// cfg.reparam = false; // rigid: rotation + scale + seed only
/// // same ellipse, twice as large, starting 7 samples later
/// let shift = 2.0 * std::f64::consts::PI * 7.0 / (m - 1) as f64;
/// let d = curve_shape_distance(&ellipse(1.5, 0.8, 0.0), &ellipse(3.0, 1.6, shift), &cfg).unwrap();
/// assert!(d < 1e-8);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_distance(
    beta1: &FdMatrix,
    beta2: &FdMatrix,
    config: &CurveShapeConfig,
) -> Result<f64, FdarError> {
    Ok(curve_shape_align_pair(beta1, beta2, config)?.distance)
}

/// Symmetric `n × n` matrix of pairwise shape distances between the curves of a
/// set (curve `j` aligned to curve `i` for `i < j`, mirrored; zero diagonal).
/// Pairs are computed in parallel with the `parallel` feature.
///
/// # Errors
/// As [`curve_shape_align_pair`], plus [`FdarError::InvalidDimension`] for an
/// empty set.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_shape_self_distance_matrix(
    curves: &FdCurveSet,
    config: &CurveShapeConfig,
) -> Result<FdMatrix, FdarError> {
    let n = curves.ncurves();
    if n == 0 || curves.ndim() == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "curves",
            expected: "at least one curve with d >= 1".to_string(),
            actual: format!("{n} curves, d = {}", curves.ndim()),
        });
    }
    let list: Vec<FdMatrix> = (0..n)
        .map(|i| curve_from_set(curves, i))
        .collect::<Result<_, _>>()?;
    // An FdCurveSet has a common shape, so validating each curve against itself
    // checks the configuration, the sample count and finiteness.
    for c in &list {
        validate_pair(c, c, config)?;
    }
    let grid = Grid::new(curves.npoints());
    let prepared: Vec<Prepared> = list
        .iter()
        .map(|c| prepare(c, config, &grid))
        .collect::<Result<_, _>>()?;
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();
    let dists: Vec<f64> = iter_maybe_parallel!(pairs.clone())
        .map(|(i, j)| {
            let c = align_to_srvf(&prepared[i].q, &prepared[j].beta, config, &grid);
            total_distance(c.distance, prepared[i].length, prepared[j].length, config)
        })
        .collect();
    let mut out = FdMatrix::zeros(n, n);
    for (&(i, j), &dv) in pairs.iter().zip(&dists) {
        out[(i, j)] = dv;
        out[(j, i)] = dv;
    }
    Ok(out)
}
