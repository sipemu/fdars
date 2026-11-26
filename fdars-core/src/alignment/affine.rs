//! Shift, dilation and affine warp registration `h(t) = a·t + b`.
//!
//! [`affine_registration`] aligns curves to a template (re-estimated sample
//! mean, a fixed curve or a closure — [`ShiftTemplate`]) using a *linear* warp
//! class ([`LinearWarpClass`]): a pure shift `h(t) = t + b`, a pure dilation
//! `h(t) = a·t`, or the full affine map `h(t) = a·t + b`. Each iteration
//! re-estimates the template, fits every curve's warp parameters by bounded
//! minimisation of a squared dissimilarity ([`AffineMetric`]) and optionally
//! centres the warps (mean dilation 1, mean shift 0). This is the registration
//! step of R `fdacluster::fdakmeans(warping_class = "shift" | "dilation" |
//! "affine", metric = "l2" | "normalized_l2" | "pearson")` (fdacluster 0.4.2);
//! the same machinery is the linear warp option of
//! [`crate::clustering_advanced::align_cluster_fd`].
//!
//! # Warp convention
//!
//! As in fdacluster, the warp acts on the *grid*: the observed value `fᵢ(tⱼ)`
//! is placed at `hᵢ(tⱼ) = aᵢ·tⱼ + bᵢ`, so the registered curve is
//! `gᵢ = fᵢ ∘ hᵢ⁻¹`, i.e. `gᵢ(s) = fᵢ((s − bᵢ)/aᵢ)`. A pure shift therefore
//! uses the sign of [`crate::alignment::iterative_shift_registration`]
//! (`gᵢ(t) = fᵢ(t − δᵢ)` with `δᵢ = bᵢ`). Dilation is about `t = 0` (not about the
//! domain start), exactly as fdacluster. If `fᵢ(t) = μ(aᵢ·t + bᵢ)` then the
//! registration recovers `(aᵢ, bᵢ)` (up to the common frame fixed by centring).
//!
//! # Comparison domain
//!
//! * [`AffineDomain::Full`] (default): `gᵢ` is evaluated on the full `argvals`
//!   grid (points whose pre-image leaves the domain follow
//!   [`ShiftExtrapolation`]) and compared with the template there — a
//!   fixed-domain least-squares criterion valid for every warp class.
//! * [`AffineDomain::Overlap`] (fdacluster): both curves are linearly
//!   interpolated onto `m` equispaced points of the intersection of
//!   `hᵢ([t₀, t_{m−1}])` and the template domain and compared there. Because
//!   the plain L2 distance can then be reduced by shrinking the overlap, L2 is
//!   rejected for dilation / affine warps on the overlap (fdacluster raises the
//!   same error); use [`AffineMetric::NormalizedL2`] or [`AffineMetric::Pearson`].
//!
//! # Deviations from fdacluster 0.4.2
//!
//! * The template lives on the fixed `argvals` grid. fdacluster lets the
//!   template grid move (union grid of the warped curves, re-centred by a
//!   separate optimisation); the identification of the common frame therefore
//!   differs. Frame-invariant quantities — the relative warps `hᵢ ∘ h₁⁻¹` and
//!   the cluster memberships — are what the golden tests compare.
//! * Bounds `[1 − max_dilation, 1 + max_dilation]` and
//!   `[−max_shift·L, max_shift·L]` (`L` = domain length; defaults 0.15 / 0.15 as
//!   fdacluster's `warping_options`) apply to the *total* warp; fdacluster
//!   bounds each iteration's increment. After centring a reported parameter may
//!   lie slightly outside its bounds.
//! * The optimiser is a deterministic coarse grid search followed by
//!   golden-section (1-D) or box-projected Nelder–Mead (2-D) refinement instead
//!   of NLopt BOBYQA.
//! * The mean template on the overlap domain averages, at each grid point, the
//!   curves that observe it (falling back to the extrapolated mean); fdacluster
//!   fills the gaps by local linear extension.

use super::shift_iterative::{eval_extrapolated, validate_data_grid};
use super::{ShiftExtrapolation, ShiftTemplate};
use crate::error::FdarError;
use crate::helpers::linear_interp;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// ---------------------------------------------------------------------------
// Public enums and config
// ---------------------------------------------------------------------------

/// Linear warp class `h(t) = a·t + b` (fdacluster `warping_class`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinearWarpClass {
    /// No warping: `h(t) = t` (fdacluster `"none"`).
    None,
    /// Pure shift `h(t) = t + b` (fdacluster `"shift"`).
    Shift,
    /// Pure dilation about 0, `h(t) = a·t` (fdacluster `"dilation"`).
    Dilation,
    /// Affine `h(t) = a·t + b` (fdacluster `"affine"`, default).
    #[default]
    Affine,
}

impl LinearWarpClass {
    fn has_dilation(self) -> bool {
        matches!(self, Self::Dilation | Self::Affine)
    }

    fn has_shift(self) -> bool {
        matches!(self, Self::Shift | Self::Affine)
    }
}

/// Dissimilarity between a registered curve and the template (fdacluster
/// `metric`). All integrals use the trapezoidal rule on the comparison grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AffineMetric {
    /// `d = √∫(g − μ)²` (fdacluster `"l2"`, default).
    #[default]
    L2,
    /// `d = √∫(g − μ)² / (√∫g² + √∫μ²)` (fdacluster `"normalized_l2"`);
    /// `0` when both norms are below `√ε`.
    NormalizedL2,
    /// fdacluster `"pearson"`: with `‖u‖² = Σ_{k≥1} u_k² / (m − 1)` over the
    /// comparison points except the first, `d = ‖g/‖g‖ − μ/‖μ‖‖ / 2`. It is
    /// *uncentred* (invariant to amplitude scaling, not to vertical offsets),
    /// `0` when both norms are below `√ε`, and `1/2` when exactly one is zero.
    Pearson,
}

/// Where registered curves are compared with the template (see the
/// module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AffineDomain {
    /// Full `argvals` grid with extrapolation (default).
    #[default]
    Full,
    /// Intersection of the warped domain and the template domain (fdacluster).
    Overlap,
}

/// Configuration of [`affine_registration`] and of the linear warp classes of
/// [`crate::clustering_advanced::align_cluster_fd`].
///
/// Construct with `AffineRegistrationConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineRegistrationConfig {
    /// Dissimilarity (default [`AffineMetric::L2`]).
    pub metric: AffineMetric,
    /// Comparison domain (default [`AffineDomain::Full`]).
    pub domain: AffineDomain,
    /// Out-of-domain evaluation of registered curves (default
    /// [`ShiftExtrapolation::Boundary`]).
    pub extrapolation: ShiftExtrapolation,
    /// Dilation bound `dl ∈ [0, 1)`: `a ∈ [1 − dl, 1 + dl]` (default 0.15).
    pub max_dilation: f64,
    /// Shift bound `sl ≥ 0` as a fraction of the domain length `L`:
    /// `b ∈ [−sl·L, sl·L]` (default 0.15).
    pub max_shift: f64,
    /// Points per parameter of the coarse grid search (default 21, minimum 3).
    pub grid_size: usize,
    /// Maximum number of template/warp iterations (default 20; ignored by
    /// `align_cluster_fd`, which uses its own `max_iter`).
    pub max_iter: usize,
    /// Stop when `maxᵢ max(|Δaᵢ|, |Δbᵢ|/L) ≤ tol` (default 1e-6).
    pub tol: f64,
    /// Centre the warps after every iteration so that the mean dilation is 1
    /// and the mean shift 0 (composition with the inverse mean warp). Ignored
    /// for a fixed template, which fixes the frame itself (default `true`).
    pub center_warps: bool,
    /// Relative parameter tolerance of the local refinement (default 1e-10).
    pub optim_tol: f64,
}

impl Default for AffineRegistrationConfig {
    fn default() -> Self {
        Self {
            metric: AffineMetric::L2,
            domain: AffineDomain::Full,
            extrapolation: ShiftExtrapolation::Boundary,
            max_dilation: 0.15,
            max_shift: 0.15,
            grid_size: 21,
            max_iter: 20,
            tol: 1e-6,
            center_warps: true,
            optim_tol: 1e-10,
        }
    }
}

impl AffineRegistrationConfig {
    /// Validate the parameters for the given warp class.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `max_dilation ∉ [0, 1)`,
    /// `max_shift` is negative or not finite, `grid_size < 3`, `tol` or
    /// `optim_tol` is negative / not finite, or the L2 metric is combined with
    /// the overlap domain and a dilation or affine warp.
    pub fn validate(&self, warp_class: LinearWarpClass) -> Result<(), FdarError> {
        let checks: [(&'static str, f64, bool, &str); 4] = [
            (
                "max_dilation",
                self.max_dilation,
                (0.0..1.0).contains(&self.max_dilation),
                "must be in [0, 1)",
            ),
            (
                "max_shift",
                self.max_shift,
                self.max_shift >= 0.0,
                "must be finite and >= 0",
            ),
            ("tol", self.tol, self.tol >= 0.0, "must be finite and >= 0"),
            (
                "optim_tol",
                self.optim_tol,
                self.optim_tol >= 0.0,
                "must be finite and >= 0",
            ),
        ];
        for (parameter, value, ok, rule) in checks {
            if !(value.is_finite() && ok) {
                return Err(FdarError::InvalidParameter {
                    parameter,
                    message: format!("{rule}, got {value}"),
                });
            }
        }
        if self.grid_size < 3 {
            return Err(FdarError::InvalidParameter {
                parameter: "grid_size",
                message: format!("must be >= 3, got {}", self.grid_size),
            });
        }
        if self.metric == AffineMetric::L2
            && self.domain == AffineDomain::Overlap
            && warp_class.has_dilation()
        {
            return Err(FdarError::InvalidParameter {
                parameter: "metric",
                message: "the L2 metric on the overlap domain is not dilation-invariant; use \
                          NormalizedL2/Pearson or AffineDomain::Full"
                    .to_string(),
            });
        }
        Ok(())
    }
}

/// Per-curve linear warp parameters `hᵢ(t) = dilations[i]·t + shifts[i]`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineWarps {
    /// Dilations `aᵢ` (1 for shift-only warps).
    pub dilations: Vec<f64>,
    /// Shifts `bᵢ` (0 for dilation-only warps).
    pub shifts: Vec<f64>,
}

/// Result of [`affine_registration`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AffineRegistrationResult {
    /// Registered curves `gᵢ(tⱼ) = fᵢ((tⱼ − bᵢ)/aᵢ)` on `argvals` (n × m), with
    /// out-of-domain pre-images following the configured extrapolation.
    pub registered_data: FdMatrix,
    /// Dilations `aᵢ`.
    pub dilations: Vec<f64>,
    /// Shifts `bᵢ` (fdars shift sign: positive moves the curve right).
    pub shifts: Vec<f64>,
    /// Template at the final warps, on `argvals`.
    pub template: Vec<f64>,
    /// Dissimilarity of each registered curve to `template`.
    pub distances: Vec<f64>,
    /// Number of iterations performed.
    pub n_iter: usize,
    /// Whether the last parameter update satisfied the tolerance.
    pub converged: bool,
}

// ---------------------------------------------------------------------------
// Warp parameters and dissimilarity
// ---------------------------------------------------------------------------

/// Linear warp `t ↦ a·t + b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Warp {
    pub(crate) a: f64,
    pub(crate) b: f64,
}

impl Warp {
    pub(crate) const IDENTITY: Warp = Warp { a: 1.0, b: 0.0 };

    /// Pre-image `h⁻¹(s)`.
    fn inverse_at(self, s: f64) -> f64 {
        (s - self.b) / self.a
    }
}

fn trapz(x: &[f64], y: &[f64]) -> f64 {
    x.windows(2)
        .zip(y.windows(2))
        .map(|(xw, yw)| 0.5 * (xw[1] - xw[0]) * (yw[0] + yw[1]))
        .sum()
}

fn pearson_distance(v1: &[f64], v2: &[f64]) -> f64 {
    let m = v1.len();
    let s1: f64 = v1[1..].iter().map(|v| v * v).sum();
    let s2: f64 = v2[1..].iter().map(|v| v * v).sum();
    let eps = f64::EPSILON.sqrt();
    if s1 < eps && s2 < eps {
        return 0.0;
    }
    if s1 == 0.0 || s2 == 0.0 {
        return 0.5;
    }
    let denom = (m - 1) as f64;
    let (r1, r2) = ((s1 / denom).sqrt(), (s2 / denom).sqrt());
    let sq: f64 = v1[1..]
        .iter()
        .zip(&v2[1..])
        .map(|(a, b)| {
            let d = a / r1 - b / r2;
            d * d
        })
        .sum();
    (sq / denom).sqrt() / 2.0
}

/// Dissimilarity of `v1` and `v2` sampled on `grid`.
fn metric_value(metric: AffineMetric, grid: &[f64], v1: &[f64], v2: &[f64]) -> f64 {
    let diff_sq: Vec<f64> = v1.iter().zip(v2).map(|(a, b)| (a - b) * (a - b)).collect();
    match metric {
        AffineMetric::L2 => trapz(grid, &diff_sq).max(0.0).sqrt(),
        AffineMetric::NormalizedL2 => {
            let sq = |v: &[f64]| trapz(grid, &v.iter().map(|x| x * x).collect::<Vec<_>>());
            let (n1, n2) = (sq(v1), sq(v2));
            let eps = f64::EPSILON.sqrt();
            if n1 < eps && n2 < eps {
                return 0.0;
            }
            trapz(grid, &diff_sq).max(0.0).sqrt() / (n1.max(0.0).sqrt() + n2.max(0.0).sqrt())
        }
        AffineMetric::Pearson => pearson_distance(v1, v2),
    }
}

/// Dissimilarity of curve `(argvals, curve)` warped by `w` to `template`.
pub(crate) fn warp_dissimilarity(
    curve: &[f64],
    argvals: &[f64],
    template: &[f64],
    w: Warp,
    config: &AffineRegistrationConfig,
) -> f64 {
    let m = argvals.len();
    match config.domain {
        AffineDomain::Full => {
            let g: Vec<f64> = argvals
                .iter()
                .map(|&t| eval_extrapolated(argvals, curve, w.inverse_at(t), config.extrapolation))
                .collect();
            metric_value(config.metric, argvals, &g, template)
        }
        AffineDomain::Overlap => {
            let (t0, t1) = (argvals[0], argvals[m - 1]);
            let lo = (w.a * t0 + w.b).max(t0);
            let hi = (w.a * t1 + w.b).min(t1);
            if lo >= hi {
                return f64::MAX;
            }
            let step = (hi - lo) / (m - 1) as f64;
            let grid: Vec<f64> = (0..m)
                .map(|k| if k + 1 == m { hi } else { lo + k as f64 * step })
                .collect();
            let g: Vec<f64> = grid
                .iter()
                .map(|&x| linear_interp(argvals, curve, w.inverse_at(x)))
                .collect();
            let mu: Vec<f64> = grid
                .iter()
                .map(|&x| linear_interp(argvals, template, x))
                .collect();
            metric_value(config.metric, &grid, &g, &mu)
        }
    }
}

// ---------------------------------------------------------------------------
// Bounded optimisation
// ---------------------------------------------------------------------------

/// Box `[a_lo, a_hi] × [b_lo, b_hi]` of admissible warp parameters.
#[derive(Debug, Clone, Copy)]
struct Bounds {
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Bounds {
    fn new(class: LinearWarpClass, config: &AffineRegistrationConfig, argvals: &[f64]) -> Self {
        let len = argvals[argvals.len() - 1] - argvals[0];
        let (dl, sl) = (
            if class.has_dilation() {
                config.max_dilation
            } else {
                0.0
            },
            if class.has_shift() {
                config.max_shift * len
            } else {
                0.0
            },
        );
        Self {
            lo: [1.0 - dl, -sl],
            hi: [1.0 + dl, sl],
        }
    }

    fn clamp(&self, x: [f64; 2]) -> [f64; 2] {
        [
            x[0].clamp(self.lo[0], self.hi[0]),
            x[1].clamp(self.lo[1], self.hi[1]),
        ]
    }

    fn width(&self, k: usize) -> f64 {
        self.hi[k] - self.lo[k]
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|k| {
            if k + 1 == n {
                hi
            } else {
                lo + (hi - lo) * k as f64 / (n - 1) as f64
            }
        })
        .collect()
}

/// Golden-section minimisation of `f` on `[lo, hi]`.
fn golden_section(f: &dyn Fn(f64) -> f64, mut lo: f64, mut hi: f64, tol: f64) -> (f64, f64) {
    let inv_phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut c = hi - inv_phi * (hi - lo);
    let mut d = lo + inv_phi * (hi - lo);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..200 {
        if hi - lo <= tol {
            break;
        }
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - inv_phi * (hi - lo);
            fc = f(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + inv_phi * (hi - lo);
            fd = f(d);
        }
    }
    if fc < fd {
        (c, fc)
    } else {
        (d, fd)
    }
}

/// Grid search + golden-section refinement of a 1-D objective on `[lo, hi]`,
/// also considering the warm start `start`.
fn minimise_1d(
    f: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    start: f64,
    grid_size: usize,
    tol: f64,
) -> (f64, f64) {
    if hi <= lo {
        return (lo, f(lo));
    }
    let xs = linspace(lo, hi, grid_size);
    let mut best = (xs[0], f(xs[0]));
    for &x in &xs[1..] {
        let fx = f(x);
        if fx < best.1 {
            best = (x, fx);
        }
    }
    let s = start.clamp(lo, hi);
    let fs = f(s);
    if fs < best.1 {
        best = (s, fs);
    }
    let h = (hi - lo) / (grid_size - 1) as f64;
    let (l, r) = ((best.0 - h).max(lo), (best.0 + h).min(hi));
    let refined = golden_section(f, l, r, tol * (hi - lo));
    if refined.1 < best.1 {
        refined
    } else {
        best
    }
}

/// Coarse 2-D grid search over `bounds` (plus the warm start).
fn grid_search_2d(
    f: &dyn Fn([f64; 2]) -> f64,
    bounds: &Bounds,
    start: [f64; 2],
    grid_size: usize,
) -> ([f64; 2], f64) {
    let xa = linspace(bounds.lo[0], bounds.hi[0], grid_size);
    let xb = linspace(bounds.lo[1], bounds.hi[1], grid_size);
    let s = bounds.clamp(start);
    let mut best = (s, f(s));
    for &a in &xa {
        for &b in &xb {
            let fx = f([a, b]);
            if fx < best.1 {
                best = ([a, b], fx);
            }
        }
    }
    best
}

/// One Nelder–Mead step on `simplex` (sorted ascending by `fv`); returns the
/// number of function evaluations used.
fn nelder_mead_step(
    f: &dyn Fn([f64; 2]) -> f64,
    bounds: &Bounds,
    simplex: &mut [[f64; 2]; 3],
    fv: &mut [f64; 3],
) -> usize {
    let lerp = |p: [f64; 2], q: [f64; 2], t: f64| {
        bounds.clamp([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])])
    };
    let c = [
        0.5 * (simplex[0][0] + simplex[1][0]),
        0.5 * (simplex[0][1] + simplex[1][1]),
    ];
    let worst = simplex[2];
    let xr = lerp(c, worst, -1.0);
    let fr = f(xr);
    if fr < fv[0] {
        let xe = lerp(c, worst, -2.0);
        let fe = f(xe);
        (simplex[2], fv[2]) = if fe < fr { (xe, fe) } else { (xr, fr) };
        return 2;
    }
    if fr < fv[1] {
        (simplex[2], fv[2]) = (xr, fr);
        return 1;
    }
    let xc = if fr < fv[2] {
        lerp(c, xr, 0.5)
    } else {
        lerp(c, worst, 0.5)
    };
    let fc = f(xc);
    if fc < fr.min(fv[2]) {
        (simplex[2], fv[2]) = (xc, fc);
        return 2;
    }
    for k in 1..3 {
        simplex[k] = lerp(simplex[0], simplex[k], 0.5);
        fv[k] = f(simplex[k]);
    }
    4
}

fn sort_simplex(simplex: &mut [[f64; 2]; 3], fv: &mut [f64; 3]) {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| {
        fv[i]
            .partial_cmp(&fv[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (s, v) = (*simplex, *fv);
    for (k, &i) in idx.iter().enumerate() {
        simplex[k] = s[i];
        fv[k] = v[i];
    }
}

/// Box-projected Nelder–Mead started at `x0` with initial steps `step`.
fn nelder_mead_box(
    f: &dyn Fn([f64; 2]) -> f64,
    bounds: &Bounds,
    x0: [f64; 2],
    step: [f64; 2],
    tol: f64,
) -> ([f64; 2], f64) {
    let vertex = |k: usize| {
        let mut v = x0;
        v[k] = if x0[k] + step[k] <= bounds.hi[k] {
            x0[k] + step[k]
        } else {
            x0[k] - step[k]
        };
        bounds.clamp(v)
    };
    let mut simplex = [x0, vertex(0), vertex(1)];
    let mut fv = [f(simplex[0]), f(simplex[1]), f(simplex[2])];
    let scale = [bounds.width(0).max(1e-300), bounds.width(1).max(1e-300)];
    let mut evals = 3;
    while evals < 1000 {
        sort_simplex(&mut simplex, &mut fv);
        let size = simplex[1..]
            .iter()
            .flat_map(|v| (0..2).map(move |k| (v[k] - simplex[0][k]).abs() / scale[k]))
            .fold(0.0_f64, f64::max);
        if size <= tol {
            break;
        }
        evals += nelder_mead_step(f, bounds, &mut simplex, &mut fv);
    }
    sort_simplex(&mut simplex, &mut fv);
    (simplex[0], fv[0])
}

/// Minimise a 2-D objective over `bounds` (grid search + Nelder–Mead).
fn minimise_2d(
    f: &dyn Fn([f64; 2]) -> f64,
    bounds: &Bounds,
    start: [f64; 2],
    grid_size: usize,
    tol: f64,
) -> ([f64; 2], f64) {
    let (x0, f0) = grid_search_2d(f, bounds, start, grid_size);
    let h = (grid_size - 1) as f64;
    let step = [bounds.width(0) / h, bounds.width(1) / h];
    let (x1, f1) = nelder_mead_box(f, bounds, x0, step, tol);
    if f1 < f0 {
        (x1, f1)
    } else {
        (x0, f0)
    }
}

/// Fit the warp of one curve to `template`; returns the warp and the
/// (unsquared) dissimilarity at it.
pub(crate) fn align_curve_to_template(
    curve: &[f64],
    argvals: &[f64],
    template: &[f64],
    class: LinearWarpClass,
    config: &AffineRegistrationConfig,
    start: Warp,
) -> (Warp, f64) {
    let bounds = Bounds::new(class, config, argvals);
    let obj = |w: Warp| {
        let d = warp_dissimilarity(curve, argvals, template, w, config);
        d * d
    };
    let (g, tol) = (config.grid_size, config.optim_tol);
    let w = match class {
        LinearWarpClass::None => Warp::IDENTITY,
        LinearWarpClass::Shift => {
            let f = |b: f64| obj(Warp { a: 1.0, b });
            let (b, _) = minimise_1d(&f, bounds.lo[1], bounds.hi[1], start.b, g, tol);
            Warp { a: 1.0, b }
        }
        LinearWarpClass::Dilation => {
            let f = |a: f64| obj(Warp { a, b: 0.0 });
            let (a, _) = minimise_1d(&f, bounds.lo[0], bounds.hi[0], start.a, g, tol);
            Warp { a, b: 0.0 }
        }
        LinearWarpClass::Affine => {
            let f = |x: [f64; 2]| obj(Warp { a: x[0], b: x[1] });
            let (x, _) = minimise_2d(&f, &bounds, [start.a, start.b], g, tol);
            Warp { a: x[0], b: x[1] }
        }
    };
    (w, warp_dissimilarity(curve, argvals, template, w, config))
}

// ---------------------------------------------------------------------------
// Templates, centring, warping
// ---------------------------------------------------------------------------

/// Registered curve `g(tⱼ) = f(h⁻¹(tⱼ))` on `argvals`.
fn registered_row(curve: &[f64], argvals: &[f64], w: Warp, ext: ShiftExtrapolation) -> Vec<f64> {
    argvals
        .iter()
        .map(|&t| eval_extrapolated(argvals, curve, w.inverse_at(t), ext))
        .collect()
}

fn rows_to_matrix(rows: &[Vec<f64>], m: usize) -> FdMatrix {
    let mut out = FdMatrix::zeros(rows.len(), m);
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    out
}

/// Mean template of the curves `members` (indices into `curves`) at `warps`.
///
/// [`AffineDomain::Full`]: pointwise mean of the extrapolated registered
/// curves (the least-squares optimal template). [`AffineDomain::Overlap`]:
/// pointwise mean over the curves that observe the point, falling back to the
/// extrapolated mean where none does.
pub(crate) fn mean_template(
    curves: &[Vec<f64>],
    members: &[usize],
    argvals: &[f64],
    warps: &[Warp],
    config: &AffineRegistrationConfig,
) -> Vec<f64> {
    let m = argvals.len();
    let (t0, t1) = (argvals[0], argvals[m - 1]);
    let tol = 1e-12 * (t1 - t0);
    let mut out = vec![0.0; m];
    for (j, &t) in argvals.iter().enumerate() {
        let (mut obs_sum, mut obs_n, mut all_sum) = (0.0, 0usize, 0.0);
        for &i in members {
            let u = warps[i].inverse_at(t);
            let v = eval_extrapolated(argvals, &curves[i], u, config.extrapolation);
            all_sum += v;
            if u >= t0 - tol && u <= t1 + tol {
                obs_sum += v;
                obs_n += 1;
            }
        }
        out[j] = if config.domain == AffineDomain::Overlap && obs_n > 0 {
            obs_sum / obs_n as f64
        } else {
            all_sum / members.len() as f64
        };
    }
    out
}

/// Centre `warps[members]` so that the mean dilation is 1 and the mean shift 0:
/// `hᵢ ← H⁻¹ ∘ hᵢ` with `H(t) = ā·t + b̄`.
pub(crate) fn center_warps(warps: &mut [Warp], members: &[usize]) {
    if members.is_empty() {
        return;
    }
    let k = members.len() as f64;
    let a_bar = members.iter().map(|&i| warps[i].a).sum::<f64>() / k;
    let b_bar = members.iter().map(|&i| warps[i].b).sum::<f64>() / k;
    for &i in members {
        warps[i] = Warp {
            a: warps[i].a / a_bar,
            b: (warps[i].b - b_bar) / a_bar,
        };
    }
}

/// Largest parameter change `max(|Δa|, |Δb|/L)`.
pub(crate) fn max_warp_change(old: &[Warp], new: &[Warp], len: f64) -> f64 {
    old.iter()
        .zip(new)
        .map(|(o, n)| (o.a - n.a).abs().max((o.b - n.b).abs() / len))
        .fold(0.0, f64::max)
}

fn validate_warp_params(dilations: &[f64], shifts: &[f64], n: usize) -> Result<(), FdarError> {
    for (parameter, v) in [("dilations", dilations), ("shifts", shifts)] {
        if v.len() != n {
            return Err(FdarError::InvalidDimension {
                parameter,
                expected: n.to_string(),
                actual: v.len().to_string(),
            });
        }
    }
    if dilations.iter().any(|a| !(a.is_finite() && *a > 0.0)) {
        return Err(FdarError::InvalidParameter {
            parameter: "dilations",
            message: "all dilations must be finite and > 0".to_string(),
        });
    }
    if shifts.iter().any(|b| !b.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "shifts",
            message: "all shifts must be finite".to_string(),
        });
    }
    Ok(())
}

/// Apply linear warps: `out[(i, j)] = fᵢ((tⱼ − shifts[i]) / dilations[i])`.
///
/// The registered curves `gᵢ = fᵢ ∘ hᵢ⁻¹` of `hᵢ(t) = aᵢ·t + bᵢ` (see the
/// module documentation), by linear interpolation on `argvals`;
/// pre-images outside the domain follow `extrapolation`.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] for empty `data` or length mismatches;
/// [`FdarError::InvalidParameter`] for fewer than 2 / non-increasing
/// `argvals`, a non-positive or non-finite dilation, or a non-finite shift.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{affine_warp_curves, ShiftExtrapolation};
/// use fdars_core::matrix::FdMatrix;
///
/// let t: Vec<f64> = (0..11).map(|j| j as f64 / 10.0).collect();
/// let data = FdMatrix::from_column_major(t.iter().map(|x| x * x).collect(), 1, 11).unwrap();
/// // h(t) = 2t + 0.1: g(s) = f((s - 0.1) / 2)
/// let g = affine_warp_curves(&data, &t, &[2.0], &[0.1], ShiftExtrapolation::Boundary).unwrap();
/// assert!((g[(0, 5)] - 0.04).abs() < 1e-12); // f(0.2) = 0.04 (grid point)
/// ```
#[must_use = "affine_warp_curves returns the warped curves without modifying the input"]
pub fn affine_warp_curves(
    data: &FdMatrix,
    argvals: &[f64],
    dilations: &[f64],
    shifts: &[f64],
    extrapolation: ShiftExtrapolation,
) -> Result<FdMatrix, FdarError> {
    validate_data_grid(data, argvals)?;
    validate_warp_params(dilations, shifts, data.nrows())?;
    let rows: Vec<Vec<f64>> = (0..data.nrows())
        .map(|i| {
            let w = Warp {
                a: dilations[i],
                b: shifts[i],
            };
            registered_row(&data.row(i), argvals, w, extrapolation)
        })
        .collect();
    Ok(rows_to_matrix(&rows, argvals.len()))
}

/// Validate data, grid and config for a linear warp class (shared with
/// `align_cluster_fd`).
pub(crate) fn validate_linear_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    class: LinearWarpClass,
    config: &AffineRegistrationConfig,
) -> Result<(), FdarError> {
    validate_data_grid(data, argvals)?;
    config.validate(class)
}

// ---------------------------------------------------------------------------
// Registration driver
// ---------------------------------------------------------------------------

fn check_template(tmpl: Vec<f64>, m: usize) -> Result<Vec<f64>, FdarError> {
    if tmpl.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "template",
            expected: m.to_string(),
            actual: tmpl.len().to_string(),
        });
    }
    if tmpl.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed {
            operation: "affine_registration",
            detail: "template contains non-finite values".to_string(),
        });
    }
    Ok(tmpl)
}

fn build_template(
    curves: &[Vec<f64>],
    argvals: &[f64],
    warps: &[Warp],
    template: &ShiftTemplate<'_>,
    config: &AffineRegistrationConfig,
) -> Result<Vec<f64>, FdarError> {
    let m = argvals.len();
    let tmpl = match template {
        ShiftTemplate::Fixed(t) => t.to_vec(),
        ShiftTemplate::Mean => {
            let members: Vec<usize> = (0..curves.len()).collect();
            mean_template(curves, &members, argvals, warps, config)
        }
        ShiftTemplate::Custom(f) => {
            let rows: Vec<Vec<f64>> = curves
                .iter()
                .zip(warps)
                .map(|(c, &w)| registered_row(c, argvals, w, config.extrapolation))
                .collect();
            f(&rows_to_matrix(&rows, m))
        }
    };
    check_template(tmpl, m)
}

/// Fit all warps to `tmpl` (warm-started at `warps`).
fn fit_all(
    curves: &[Vec<f64>],
    argvals: &[f64],
    tmpl: &[f64],
    class: LinearWarpClass,
    config: &AffineRegistrationConfig,
    warps: &[Warp],
) -> Vec<Warp> {
    iter_maybe_parallel!(0..curves.len())
        .map(|i| align_curve_to_template(&curves[i], argvals, tmpl, class, config, warps[i]).0)
        .collect()
}

/// Shift, dilation or affine warp registration `hᵢ(t) = aᵢ·t + bᵢ` to a template.
///
/// Iterates (at most `config.max_iter` times):
///
/// 1. form the template from the currently registered curves ([`ShiftTemplate`]:
///    re-estimated mean, fixed curve, or closure of the registered n × m matrix);
/// 2. for every curve, minimise the squared dissimilarity
///    `d(fᵢ ∘ hᵢ⁻¹, μ)²` ([`AffineMetric`], [`AffineDomain`]) over the class's
///    parameters inside the bounds (grid search, then golden-section /
///    Nelder–Mead refinement, warm-started at the previous warp);
/// 3. unless the template is fixed, centre the warps (`config.center_warps`:
///    mean dilation 1, mean shift 0);
/// 4. stop when `maxᵢ max(|Δaᵢ|, |Δbᵢ|/L) ≤ tol`.
///
/// This is the k = 1 case of R `fdacluster::fdakmeans(warping_class = ...,
/// centroid_type = "mean", metric = ...)`; see the
/// module documentation for the warp convention, the comparison
/// domains and the documented deviations.
///
/// # Errors
///
/// * [`FdarError::InvalidDimension`] for empty `data`, `argvals.len() != m`,
///   or a fixed / custom template of length ≠ m.
/// * [`FdarError::InvalidParameter`] for fewer than 2 / non-increasing
///   `argvals` or an invalid config (see [`AffineRegistrationConfig::validate`]).
/// * [`FdarError::ComputationFailed`] if the template is non-finite.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{affine_registration, AffineRegistrationConfig,
///     LinearWarpClass, ShiftTemplate};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 101;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let mu = |s: f64| (-((s - 0.5) / 0.1).powi(2)).exp();
/// let (a, b) = ([1.06, 0.95, 1.0], [-0.04, 0.03, 0.01]);
/// let mut data = FdMatrix::zeros(3, m);
/// for i in 0..3 {
///     for (j, &tj) in t.iter().enumerate() {
///         data[(i, j)] = mu(a[i] * tj + b[i]); // f_i = mu ∘ h_i
///     }
/// }
/// let template: Vec<f64> = t.iter().map(|&s| mu(s)).collect();
/// let cfg = AffineRegistrationConfig::default();
/// let res = affine_registration(&data, &t, LinearWarpClass::Affine,
///     ShiftTemplate::Fixed(&template), &cfg).unwrap();
/// for i in 0..3 {
///     assert!((res.dilations[i] - a[i]).abs() < 1e-3);
///     assert!((res.shifts[i] - b[i]).abs() < 1e-3);
/// }
/// ```
#[must_use = "affine_registration is expensive; use the returned registration"]
pub fn affine_registration(
    data: &FdMatrix,
    argvals: &[f64],
    warp_class: LinearWarpClass,
    template: ShiftTemplate<'_>,
    config: &AffineRegistrationConfig,
) -> Result<AffineRegistrationResult, FdarError> {
    validate_data_grid(data, argvals)?;
    config.validate(warp_class)?;
    if let ShiftTemplate::Fixed(t) = template {
        check_template(t.to_vec(), argvals.len())?;
    }
    let n = data.nrows();
    let len = argvals[argvals.len() - 1] - argvals[0];
    let curves: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let center = config.center_warps && !matches!(template, ShiftTemplate::Fixed(_));
    let all: Vec<usize> = (0..n).collect();

    let mut warps = vec![Warp::IDENTITY; n];
    let (mut n_iter, mut converged) = (0, false);
    while n_iter < config.max_iter {
        let tmpl = build_template(&curves, argvals, &warps, &template, config)?;
        let mut new = fit_all(&curves, argvals, &tmpl, warp_class, config, &warps);
        if center {
            center_warps(&mut new, &all);
        }
        let diff = max_warp_change(&warps, &new, len);
        warps = new;
        n_iter += 1;
        if diff <= config.tol {
            converged = true;
            break;
        }
    }

    let tmpl = build_template(&curves, argvals, &warps, &template, config)?;
    let distances = iter_maybe_parallel!(0..n)
        .map(|i| warp_dissimilarity(&curves[i], argvals, &tmpl, warps[i], config))
        .collect();
    let rows: Vec<Vec<f64>> = curves
        .iter()
        .zip(&warps)
        .map(|(c, &w)| registered_row(c, argvals, w, config.extrapolation))
        .collect();
    Ok(AffineRegistrationResult {
        registered_data: rows_to_matrix(&rows, argvals.len()),
        dilations: warps.iter().map(|w| w.a).collect(),
        shifts: warps.iter().map(|w| w.b).collect(),
        template: tmpl,
        distances,
        n_iter,
        converged,
    })
}

#[cfg(test)]
#[path = "affine_tests.rs"]
mod tests;
