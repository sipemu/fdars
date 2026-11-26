//! Ramsay–Li continuous registration (R `fda::register.fd`).
//!
//! [`continuous_registration`] warps every curve `yᵢ` onto a target `y₀`
//! (default: the cross-sectional mean, optionally re-estimated Procrustes-style)
//! with a smooth strictly monotone warp of the Ramsay (1998) form
//!
//! ```text
//! hᵢ(t) = a + (b − a) · Fᵢ(t) / Fᵢ(b),   Fᵢ(t) = ∫ₐᵗ exp Wᵢ(s) ds,   Wᵢ = Σₖ cᵢₖ φₖ,
//! ```
//!
//! where `φₖ` is a clamped B-spline basis on `[a, b]` (the convention of R
//! `create.bspline.basis`). The registered curve is `yᵢ(hᵢ(t))`. Because the
//! warp is normalised, adding a constant to `W` has no effect; as in R the first
//! coefficient `cᵢ₁` is therefore fixed at 0 and only `c₂ … c_K` are estimated.
//!
//! # Criteria
//!
//! With `ŷ = yᵢ ∘ hᵢ` and means taken over a uniform fine grid (R's `xfine`):
//!
//! * [`ContinuousRegistrationCriterion::LeastSquares`] (`crit = 1`):
//!   `mean((y₀ − ŷ)²)`.
//! * [`ContinuousRegistrationCriterion::MinEigenvalue`] (`crit = 2`, Ramsay & Li
//!   1998): with `M = [[mean y₀², mean y₀ŷ], [mean y₀ŷ, mean ŷ²]]`, the criterion is
//!   `2·λ_min(M) = (aa + cc) − √((aa − cc)² + 4bb²)` — exactly R's `ee − dd`. It is
//!   zero whenever `ŷ` is proportional to `y₀`, so amplitude differences are not
//!   penalised.
//!
//! Both are augmented by the roughness penalty `λ · cᵀ K c` with
//! `K = ∫ Dᴸφ Dᴸφᵀ` over the estimated coefficients (R `getbasispenalty`).
//!
//! # Optimiser
//!
//! Both criteria are minimised by the same penalised Gauss–Newton iteration with
//! Levenberg–Marquardt safeguarding. The minimum-eigenvalue criterion is handled
//! by variable projection: for the unit eigenvector `v` of `λ_min(M)`,
//! `λ_min = mean((v₁y₀ + v₂ŷ)²)`, and by the envelope theorem the gradient of
//! this least-squares form is the exact gradient of `λ_min`. A step is accepted
//! only if it strictly decreases the true criterion, so the criterion sequence
//! is non-increasing.
//!
//! # Deviations from R fda 6.3.0 `register.fd`
//!
//! * Data are discrete curves on `argvals`, interpolated by a natural cubic
//!   spline (value and derivative) instead of `fd` objects; the registered curves
//!   and warps are returned on `argvals` (R projects them back onto the `yfd` and
//!   W bases, so R's `warpfd` is a basis approximation of `h`).
//! * R's gradient uses the derivative of the *registered* curve, `D(y∘h)`, where
//!   the exact chain rule needs `Dy(h(t)) = D(y∘h)(t) / h′(t)`; fdars uses the
//!   exact gradient, a Gauss–Newton (not full Newton) Hessian, and an LM step
//!   control in place of R's cubic line search. Converged criterion values are
//!   therefore lower than or equal to R's, but not bit-identical.
//! * `F(t)` is integrated by the cumulative trapezoid rule on the fine grid (R:
//!   Romberg extrapolation + trapezoid); convergence uses a *relative* criterion
//!   decrease `tol` (R: absolute `conv = 1e-4`).
//! * `periodic = TRUE` (Fourier shift parameter) is not supported.
//! * Default W basis: 6 cubic B-splines (R: 2 linear B-splines, i.e. a linear W).
//! * `max_template_iter > 1` adds Procrustes re-estimation of the mean target,
//!   which `register.fd` leaves to the caller.

use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::rbfgs::NaturalSpline;

/// Coefficient magnitude limit (R `register.fd` `climit`).
const COEF_LIMIT: f64 = 50.0;
/// Maximum number of Levenberg–Marquardt damping increases per iteration.
const LM_MAX_TRIES: usize = 40;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Fitting criterion of [`continuous_registration`] (R `register.fd` `crit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContinuousRegistrationCriterion {
    /// `crit = 1`: least squares `mean((y₀ − y∘h)²)`.
    LeastSquares,
    /// `crit = 2` (R default): Ramsay–Li minimum-eigenvalue criterion
    /// `2·λ_min` of the 2×2 cross-product matrix of `(y₀, y∘h)`.
    #[default]
    MinEigenvalue,
}

/// Configuration of [`continuous_registration`].
///
/// Construct with `ContinuousRegistrationConfig::default()` and assign fields
/// (the struct is `#[non_exhaustive]`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContinuousRegistrationConfig {
    /// Fitting criterion (default [`ContinuousRegistrationCriterion::MinEigenvalue`]).
    pub criterion: ContinuousRegistrationCriterion,
    /// Number of B-spline basis functions of `W` (default 6, ≥ 2, ≥ `norder`).
    pub nbasis: usize,
    /// B-spline order of `W` (default 4 = cubic, ≥ 2).
    pub norder: usize,
    /// Roughness penalty weight λ on `W` (default 0).
    pub lambda: f64,
    /// Derivative order `L` of the roughness penalty (default 2, `< norder`).
    pub lfd_order: usize,
    /// Maximum Gauss–Newton iterations per curve (default 50; 0 returns identity warps).
    pub max_iter: usize,
    /// Stop when the relative criterion decrease is `≤ tol` (default `1e-8`).
    pub tol: f64,
    /// Number of points of the uniform fine grid on which the criterion is
    /// evaluated (default `None` = `max(201, m)`, R uses `max(201, 10·nbasis(yfd)+1)`).
    pub n_fine: Option<usize>,
    /// Number of registration passes when the target is the mean (default 1 = R's
    /// single pass against the unregistered mean). Each further pass re-estimates
    /// the target as the mean of the registered curves and re-registers the
    /// original curves, warm-started from the previous coefficients. Ignored when
    /// an explicit target is supplied.
    pub max_template_iter: usize,
    /// Stop the template passes when `‖y₀⁽ᵏ⁾ − y₀⁽ᵏ⁻¹⁾‖ / ‖y₀⁽ᵏ⁻¹⁾‖ ≤ template_tol`
    /// (default `1e-4`).
    pub template_tol: f64,
    /// Initial W coefficients (`n × nbasis`; column 0 is ignored). Default: zeros
    /// (identity warps), as R.
    pub initial_coefficients: Option<FdMatrix>,
}

impl Default for ContinuousRegistrationConfig {
    fn default() -> Self {
        Self {
            criterion: ContinuousRegistrationCriterion::MinEigenvalue,
            nbasis: 6,
            norder: 4,
            lambda: 0.0,
            lfd_order: 2,
            max_iter: 50,
            tol: 1e-8,
            n_fine: None,
            max_template_iter: 1,
            template_tol: 1e-4,
            initial_coefficients: None,
        }
    }
}

impl ContinuousRegistrationConfig {
    /// Validate the scalar parameters.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `nbasis < 2`, `norder < 2`,
    /// `nbasis < norder`, `lfd_order ≥ norder`, `lambda` or `tol` or
    /// `template_tol` negative / not finite, `n_fine < 3`, or
    /// `max_template_iter == 0`.
    pub fn validate(&self) -> Result<(), FdarError> {
        self.validate_basis()?;
        self.validate_scalars()
    }

    fn validate_basis(&self) -> Result<(), FdarError> {
        let message = if self.norder < 2 {
            Some(("norder", format!("must be >= 2, got {}", self.norder)))
        } else if self.nbasis < 2 || self.nbasis < self.norder {
            Some((
                "nbasis",
                format!(
                    "must be >= 2 and >= norder ({}), got {}",
                    self.norder, self.nbasis
                ),
            ))
        } else if self.lfd_order >= self.norder {
            Some((
                "lfd_order",
                format!("must be < norder ({}), got {}", self.norder, self.lfd_order),
            ))
        } else {
            None
        };
        match message {
            Some((parameter, message)) => Err(FdarError::InvalidParameter { parameter, message }),
            None => Ok(()),
        }
    }

    fn validate_scalars(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        for (name, v) in [
            ("lambda", self.lambda),
            ("tol", self.tol),
            ("template_tol", self.template_tol),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(name, format!("must be finite and >= 0, got {v}"));
            }
        }
        if matches!(self.n_fine, Some(nf) if nf < 3) {
            return bad("n_fine", format!("must be >= 3, got {:?}", self.n_fine));
        }
        if self.max_template_iter == 0 {
            return bad("max_template_iter", "must be >= 1, got 0".to_string());
        }
        Ok(())
    }
}

/// Result of [`continuous_registration`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ContinuousRegistrationResult {
    /// Registered curves `yᵢ(hᵢ(t))` on `argvals` (`n × m`).
    pub registered_data: FdMatrix,
    /// Warping functions `hᵢ(t)` on `argvals` (`n × m`); strictly increasing with
    /// `hᵢ(a) = a`, `hᵢ(b) = b`.
    pub warps: FdMatrix,
    /// B-spline coefficients of `Wᵢ` (`n × nbasis`, column 0 is always 0).
    pub w_coefficients: FdMatrix,
    /// Clamped knot vector of the W basis (length `nbasis + norder`).
    pub w_knots: Vec<f64>,
    /// Final criterion value per curve, including the roughness penalty.
    pub criterion: Vec<f64>,
    /// Target of the last registration pass, on `argvals`.
    pub target: Vec<f64>,
    /// Gauss–Newton iterations per curve in the last pass.
    pub n_iter: Vec<usize>,
    /// Per-curve convergence flag of the last pass.
    pub converged: Vec<bool>,
    /// Number of registration passes performed.
    pub n_template_iter: usize,
}

// ---------------------------------------------------------------------------
// Clamped B-spline basis (R `create.bspline.basis` convention)
// ---------------------------------------------------------------------------

/// Clamped knot vector with `nbasis − norder + 2` equally spaced breaks.
pub(super) fn clamped_knots(a: f64, b: f64, nbasis: usize, norder: usize) -> Vec<f64> {
    let nbreaks = nbasis + 2 - norder;
    let mut knots = vec![a; norder - 1];
    for i in 0..nbreaks {
        knots.push(a + (b - a) * i as f64 / (nbreaks - 1) as f64);
    }
    knots.extend(std::iter::repeat(b).take(norder - 1));
    let last = knots.len() - norder;
    knots[last] = b;
    knots
}

#[inline]
fn safe_div(num: f64, den: f64) -> f64 {
    if den.abs() > 0.0 {
        num / den
    } else {
        0.0
    }
}

/// Cox–de Boor values of all orders `1..=norder` at `t` (index `k-1` holds the
/// `knots.len() − k` basis functions of order `k`).
fn bspline_table(t: f64, knots: &[f64], norder: usize) -> Vec<Vec<f64>> {
    let nk = knots.len();
    let nbasis = nk - norder;
    // Span: last non-degenerate interval containing t (right end closed).
    let span = (norder - 1..nbasis)
        .rev()
        .find(|&j| knots[j] <= t && knots[j] < knots[j + 1])
        .unwrap_or(norder - 1);
    let mut table = Vec::with_capacity(norder);
    let mut b1 = vec![0.0; nk - 1];
    b1[span] = 1.0;
    table.push(b1);
    for k in 2..=norder {
        let prev = &table[k - 2];
        let cur: Vec<f64> = (0..nk - k)
            .map(|j| {
                safe_div(t - knots[j], knots[j + k - 1] - knots[j]) * prev[j]
                    + safe_div(knots[j + k] - t, knots[j + k] - knots[j + 1]) * prev[j + 1]
            })
            .collect();
        table.push(cur);
    }
    table
}

/// `deriv`-th derivative of the order-`k` basis functions from a Cox–de Boor table.
fn bspline_deriv_from_table(table: &[Vec<f64>], knots: &[f64], k: usize, deriv: usize) -> Vec<f64> {
    if deriv == 0 {
        return table[k - 1].clone();
    }
    let lower = bspline_deriv_from_table(table, knots, k - 1, deriv - 1);
    let scale = (k - 1) as f64;
    (0..knots.len() - k)
        .map(|j| {
            scale
                * (safe_div(lower[j], knots[j + k - 1] - knots[j])
                    - safe_div(lower[j + 1], knots[j + k] - knots[j + 1]))
        })
        .collect()
}

/// Values (or `deriv`-th derivatives) of the clamped basis at `t`.
pub(super) fn clamped_bspline_eval(t: f64, knots: &[f64], norder: usize, deriv: usize) -> Vec<f64> {
    if deriv >= norder {
        return vec![0.0; knots.len() - norder];
    }
    let table = bspline_table(t, knots, norder);
    bspline_deriv_from_table(&table, knots, norder, deriv)
}

/// Roughness penalty `K = ∫ Dᴸφ Dᴸφᵀ` (row-major `nbasis × nbasis`) by
/// composite 5-point Gauss–Legendre quadrature on every knot interval.
pub(super) fn clamped_bspline_penalty(knots: &[f64], norder: usize, lfd: usize) -> Vec<f64> {
    const GL_X: [f64; 5] = [
        -0.906_179_845_938_664,
        -0.538_469_310_105_683,
        0.0,
        0.538_469_310_105_683,
        0.906_179_845_938_664,
    ];
    const GL_W: [f64; 5] = [
        0.236_926_885_056_189,
        0.478_628_670_499_366,
        0.568_888_888_888_889,
        0.478_628_670_499_366,
        0.236_926_885_056_189,
    ];
    let nb = knots.len() - norder;
    let nsub = norder.div_ceil(2);
    let mut k = vec![0.0; nb * nb];
    for w in knots.windows(2).filter(|w| w[1] > w[0]) {
        let hsub = (w[1] - w[0]) / nsub as f64;
        for s in 0..nsub {
            let lo = w[0] + s as f64 * hsub;
            for (x, wt) in GL_X.iter().zip(GL_W.iter()) {
                let t = lo + 0.5 * hsub * (x + 1.0);
                let d = clamped_bspline_eval(t, knots, norder, lfd);
                let q = 0.5 * hsub * wt;
                for p in 0..nb {
                    for r in 0..nb {
                        k[p * nb + r] += q * d[p] * d[r];
                    }
                }
            }
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Per-curve problem
// ---------------------------------------------------------------------------

/// Quantities shared by every curve: fine grid, W basis and penalty.
pub(super) struct CcSetup {
    pub(super) a: f64,
    pub(super) b: f64,
    pub(super) xfine: Vec<f64>,
    /// `nfine × nbasis`, row-major.
    pub(super) phi: Vec<f64>,
    pub(super) nb: usize,
    /// λ-scaled penalty on the active coefficients, `(nb−1)²` row-major.
    pub(super) kmat: Vec<f64>,
    pub(super) knots: Vec<f64>,
    pub(super) criterion: ContinuousRegistrationCriterion,
}

impl CcSetup {
    pub(super) fn new(a: f64, b: f64, nfine: usize, cfg: &ContinuousRegistrationConfig) -> Self {
        let knots = clamped_knots(a, b, cfg.nbasis, cfg.norder);
        let nb = cfg.nbasis;
        let xfine: Vec<f64> = (0..nfine)
            .map(|j| a + (b - a) * j as f64 / (nfine - 1) as f64)
            .collect();
        let mut phi = Vec::with_capacity(nfine * nb);
        for &x in &xfine {
            phi.extend(clamped_bspline_eval(x, &knots, cfg.norder, 0));
        }
        let na = nb - 1;
        let mut kmat = vec![0.0; na * na];
        if cfg.lambda > 0.0 {
            let full = clamped_bspline_penalty(&knots, cfg.norder, cfg.lfd_order);
            for p in 0..na {
                for r in 0..na {
                    kmat[p * na + r] = cfg.lambda * full[(p + 1) * nb + r + 1];
                }
            }
        }
        Self {
            a,
            b,
            xfine,
            phi,
            nb,
            kmat,
            knots,
            criterion: cfg.criterion,
        }
    }

    fn nfine(&self) -> usize {
        self.xfine.len()
    }

    /// `exp W` on the fine grid.
    fn exp_w(&self, c: &[f64]) -> Vec<f64> {
        let nb = self.nb;
        (0..self.nfine())
            .map(|j| {
                let row = &self.phi[j * nb..(j + 1) * nb];
                row.iter()
                    .zip(c)
                    .skip(1)
                    .map(|(p, ck)| p * ck)
                    .sum::<f64>()
                    .exp()
            })
            .collect()
    }

    fn penalty(&self, c: &[f64]) -> f64 {
        let na = self.nb - 1;
        let mut s = 0.0;
        for p in 0..na {
            for r in 0..na {
                s += c[p + 1] * self.kmat[p * na + r] * c[r + 1];
            }
        }
        s
    }
}

/// Warp `h` on the fine grid and, optionally, `∂h/∂c_k` for the active `k`
/// (`nfine × (nb−1)` row-major).
pub(super) struct FineWarp {
    pub(super) h: Vec<f64>,
    pub(super) dh: Vec<f64>,
    /// Cumulative integral `F` and integrand `exp W` (for Hermite evaluation).
    f: Vec<f64>,
    e: Vec<f64>,
}

fn fine_warp(setup: &CcSetup, c: &[f64], with_grad: bool) -> FineWarp {
    let nf = setup.nfine();
    let nb = setup.nb;
    let na = nb - 1;
    let dx = (setup.b - setup.a) / (nf - 1) as f64;
    let width = setup.b - setup.a;
    let e = setup.exp_w(c);
    let mut f = vec![0.0; nf];
    for j in 1..nf {
        f[j] = f[j - 1] + 0.5 * dx * (e[j - 1] + e[j]);
    }
    let fmax = f[nf - 1];
    let mut h: Vec<f64> = f.iter().map(|&fj| setup.a + width * fj / fmax).collect();
    h[0] = setup.a;
    h[nf - 1] = setup.b;
    let mut dh = Vec::new();
    if with_grad {
        let mut df = vec![0.0; nf * na];
        for j in 1..nf {
            for k in 0..na {
                let g0 = e[j - 1] * setup.phi[(j - 1) * nb + k + 1];
                let g1 = e[j] * setup.phi[j * nb + k + 1];
                df[j * na + k] = df[(j - 1) * na + k] + 0.5 * dx * (g0 + g1);
            }
        }
        let dfmax: Vec<f64> = df[(nf - 1) * na..].to_vec();
        dh = vec![0.0; nf * na];
        for j in 1..nf - 1 {
            for k in 0..na {
                dh[j * na + k] = width * (fmax * df[j * na + k] - f[j] * dfmax[k]) / (fmax * fmax);
            }
        }
    }
    FineWarp { h, dh, f, e }
}

/// Evaluate the warp at arbitrary points of `[a, b]` by cubic Hermite
/// interpolation of `F` (whose derivative `exp W` is known exactly).
fn warp_at(setup: &CcSetup, fw: &FineWarp, t: &[f64]) -> Vec<f64> {
    let nf = setup.nfine();
    let dx = (setup.b - setup.a) / (nf - 1) as f64;
    let fmax = fw.f[nf - 1];
    let width = setup.b - setup.a;
    t.iter()
        .map(|&ti| {
            if ti <= setup.a {
                return setup.a;
            }
            if ti >= setup.b {
                return setup.b;
            }
            let j = (((ti - setup.a) / dx).floor() as usize).min(nf - 2);
            let s = (ti - setup.xfine[j]) / dx;
            let (s2, s3) = (s * s, s * s * s);
            let fv = (2.0 * s3 - 3.0 * s2 + 1.0) * fw.f[j]
                + (s3 - 2.0 * s2 + s) * dx * fw.e[j]
                + (-2.0 * s3 + 3.0 * s2) * fw.f[j + 1]
                + (s3 - s2) * dx * fw.e[j + 1];
            (setup.a + width * fv / fmax).clamp(setup.a, setup.b)
        })
        .collect()
}

/// Criterion value plus the least-squares residual form `F = s·mean(r²) + pen`
/// with Jacobian `∂r/∂c` (`nfine × (nb−1)` row-major).
pub(super) struct CritEval {
    pub(super) f: f64,
    r: Vec<f64>,
    jr: Vec<f64>,
    s: f64,
}

/// Unit eigenvector of the smallest eigenvalue of `[[aa, bb], [bb, cc]]`.
fn min_eigvec(aa: f64, bb: f64, cc: f64) -> (f64, f64) {
    let dd = ((aa - cc).powi(2) + 4.0 * bb * bb).sqrt();
    let lam = 0.5 * (aa + cc - dd);
    let c1 = (bb, lam - aa);
    let c2 = (lam - cc, bb);
    let n1 = c1.0.hypot(c1.1);
    let n2 = c2.0.hypot(c2.1);
    if n1 >= n2 && n1 > 0.0 {
        (c1.0 / n1, c1.1 / n1)
    } else if n2 > 0.0 {
        (c2.0 / n2, c2.1 / n2)
    } else {
        (0.0, 1.0)
    }
}

/// Evaluate the criterion for coefficients `c` against `y0` (on the fine grid).
pub(super) fn crit_eval(
    setup: &CcSetup,
    spline: &NaturalSpline,
    y0: &[f64],
    c: &[f64],
    with_grad: bool,
) -> CritEval {
    let nf = setup.nfine();
    let na = setup.nb - 1;
    let fw = fine_warp(setup, c, with_grad);
    let (yv, yd): (Vec<f64>, Vec<f64>) = fw.h.iter().map(|&h| spline.eval(h)).unzip();
    let nfl = nf as f64;
    let pen = setup.penalty(c);
    let (f, r, factor, s) = match setup.criterion {
        ContinuousRegistrationCriterion::LeastSquares => {
            let r: Vec<f64> = yv.iter().zip(y0).map(|(y, t)| y - t).collect();
            let f = r.iter().map(|v| v * v).sum::<f64>() / nfl;
            (f, r, 1.0, 1.0)
        }
        ContinuousRegistrationCriterion::MinEigenvalue => {
            let aa = y0.iter().map(|v| v * v).sum::<f64>() / nfl;
            let bb = y0.iter().zip(&yv).map(|(a, b)| a * b).sum::<f64>() / nfl;
            let cc = yv.iter().map(|v| v * v).sum::<f64>() / nfl;
            let f = (aa + cc) - ((aa - cc).powi(2) + 4.0 * bb * bb).sqrt();
            let (v1, v2) = min_eigvec(aa, bb, cc);
            let r: Vec<f64> = y0.iter().zip(&yv).map(|(a, b)| v1 * a + v2 * b).collect();
            (f, r, v2, 2.0)
        }
    };
    let mut jr = Vec::new();
    if with_grad {
        jr = vec![0.0; nf * na];
        for j in 0..nf {
            for k in 0..na {
                jr[j * na + k] = factor * yd[j] * fw.dh[j * na + k];
            }
        }
    }
    CritEval {
        f: f + pen,
        r,
        jr,
        s,
    }
}

/// Gradient `g` and Gauss–Newton Hessian `H` of the active coefficients.
fn gn_system(setup: &CcSetup, ev: &CritEval, c: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let nf = setup.nfine();
    let na = setup.nb - 1;
    let w = 2.0 * ev.s / nf as f64;
    let mut g = vec![0.0; na];
    let mut hm = vec![0.0; na * na];
    for j in 0..nf {
        let row = &ev.jr[j * na..(j + 1) * na];
        for p in 0..na {
            g[p] += w * row[p] * ev.r[j];
            for q in 0..na {
                hm[p * na + q] += w * row[p] * row[q];
            }
        }
    }
    for p in 0..na {
        for q in 0..na {
            let kpq = setup.kmat[p * na + q];
            g[p] += 2.0 * kpq * c[q + 1];
            hm[p * na + q] += 2.0 * kpq;
        }
    }
    (g, hm)
}

/// Exact gradient of the criterion with respect to the active coefficients.
#[cfg(test)]
pub(super) fn crit_gradient(
    setup: &CcSetup,
    spline: &NaturalSpline,
    y0: &[f64],
    c: &[f64],
) -> Vec<f64> {
    let ev = crit_eval(setup, spline, y0, c, true);
    gn_system(setup, &ev, c).0
}

/// Outcome of one Levenberg–Marquardt-safeguarded step.
enum LmStep {
    Accepted(Vec<f64>, CritEval),
    Stalled,
}

/// Solve the damped system `(H + μ·D) δ = g` and return `c − δ` if it is admissible.
fn damped_candidate(hm: &[f64], g: &[f64], c: &[f64], mu: f64, floor: f64) -> Option<Vec<f64>> {
    let na = g.len();
    let mut a = hm.to_vec();
    for p in 0..na {
        a[p * na + p] += mu * hm[p * na + p].max(floor);
    }
    let delta = crate::linalg::cholesky_solve(&a, g, na).ok()?;
    let mut cn = c.to_vec();
    for (cp, d) in cn[1..].iter_mut().zip(&delta) {
        *cp -= d;
    }
    cn.iter()
        .all(|v| v.is_finite() && v.abs() <= COEF_LIMIT)
        .then_some(cn)
}

fn lm_step(
    setup: &CcSetup,
    spline: &NaturalSpline,
    y0: &[f64],
    c: &[f64],
    cur: &CritEval,
    mu: &mut f64,
) -> LmStep {
    let na = setup.nb - 1;
    let (g, hm) = gn_system(setup, cur, c);
    if g.iter().all(|v| *v == 0.0) {
        return LmStep::Stalled;
    }
    let maxdiag = (0..na).map(|p| hm[p * na + p]).fold(0.0_f64, f64::max);
    let floor = (1e-12 * maxdiag).max(f64::MIN_POSITIVE);
    for _ in 0..LM_MAX_TRIES {
        if let Some(cn) = damped_candidate(&hm, &g, c, *mu, floor) {
            let ev = crit_eval(setup, spline, y0, &cn, true);
            if ev.f.is_finite() && ev.f < cur.f {
                *mu = (*mu / 3.0).max(1e-12);
                return LmStep::Accepted(cn, ev);
            }
        }
        *mu *= 4.0;
    }
    LmStep::Stalled
}

/// Per-curve optimisation output.
pub(super) struct CurveFit {
    pub(super) c: Vec<f64>,
    pub(super) history: Vec<f64>,
    pub(super) n_iter: usize,
    pub(super) converged: bool,
}

/// Minimise the criterion of one curve starting from `c0`.
pub(super) fn register_curve(
    setup: &CcSetup,
    spline: &NaturalSpline,
    y0: &[f64],
    c0: &[f64],
    max_iter: usize,
    tol: f64,
) -> CurveFit {
    let mut c = c0.to_vec();
    c[0] = 0.0;
    let mut cur = crit_eval(setup, spline, y0, &c, true);
    let mut history = vec![cur.f];
    let mut mu = 1e-3;
    let mut converged = false;
    let mut n_iter = 0;
    while n_iter < max_iter {
        n_iter += 1;
        match lm_step(setup, spline, y0, &c, &cur, &mut mu) {
            LmStep::Accepted(cn, ev) => {
                let decrease = cur.f - ev.f;
                let done = decrease <= tol * cur.f.abs();
                c = cn;
                cur = ev;
                history.push(cur.f);
                if done {
                    converged = true;
                    break;
                }
            }
            LmStep::Stalled => {
                converged = true;
                break;
            }
        }
    }
    CurveFit {
        c,
        history,
        n_iter,
        converged,
    }
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

fn validate_grid(data: &FdMatrix, argvals: &[f64]) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m < 4 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 curve and 4 evaluation points".to_string(),
            actual: format!("{n} x {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: m.to_string(),
            actual: argvals.len().to_string(),
        });
    }
    if argvals.iter().any(|v| !v.is_finite()) || argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be finite and strictly increasing".to_string(),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: "contains non-finite values".to_string(),
        });
    }
    Ok(())
}

fn validate_target(target: &[f64], m: usize) -> Result<(), FdarError> {
    if target.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "target",
            expected: m.to_string(),
            actual: target.len().to_string(),
        });
    }
    if target.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "target",
            message: "contains non-finite values".to_string(),
        });
    }
    Ok(())
}

fn validate_initial(init: &FdMatrix, n: usize, nbasis: usize) -> Result<(), FdarError> {
    if init.shape() != (n, nbasis) || init.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidDimension {
            parameter: "initial_coefficients",
            expected: format!("finite {n} x {nbasis}"),
            actual: format!("{} x {}", init.nrows(), init.ncols()),
        });
    }
    Ok(())
}

fn validate_inputs(
    data: &FdMatrix,
    argvals: &[f64],
    target: Option<&[f64]>,
    cfg: &ContinuousRegistrationConfig,
) -> Result<(), FdarError> {
    validate_grid(data, argvals)?;
    if let Some(t) = target {
        validate_target(t, data.ncols())?;
    }
    if let Some(init) = &cfg.initial_coefficients {
        validate_initial(init, data.nrows(), cfg.nbasis)?;
    }
    cfg.validate()
}

/// One registration pass of all curves against `target` (on `argvals`).
fn register_pass(
    setup: &CcSetup,
    splines: &[NaturalSpline],
    argvals: &[f64],
    target: &[f64],
    coefs: &[Vec<f64>],
    cfg: &ContinuousRegistrationConfig,
) -> Vec<CurveFit> {
    let tspline = NaturalSpline::new(argvals, target);
    let y0: Vec<f64> = setup.xfine.iter().map(|&x| tspline.eval(x).0).collect();
    iter_maybe_parallel!(0..splines.len())
        .map(|i| register_curve(setup, &splines[i], &y0, &coefs[i], cfg.max_iter, cfg.tol))
        .collect()
}

fn registered_rows(
    setup: &CcSetup,
    splines: &[NaturalSpline],
    argvals: &[f64],
    fits: &[CurveFit],
) -> (FdMatrix, FdMatrix) {
    let (n, m) = (fits.len(), argvals.len());
    let mut reg = FdMatrix::zeros(n, m);
    let mut warps = FdMatrix::zeros(n, m);
    for (i, fit) in fits.iter().enumerate() {
        let fw = fine_warp(setup, &fit.c, false);
        let h = warp_at(setup, &fw, argvals);
        for (j, &hj) in h.iter().enumerate() {
            warps[(i, j)] = hj;
            reg[(i, j)] = splines[i].eval(hj).0;
        }
    }
    (reg, warps)
}

fn rel_change(new: &[f64], old: &[f64]) -> f64 {
    let num: f64 = new.iter().zip(old).map(|(a, b)| (a - b).powi(2)).sum();
    let den: f64 = old.iter().map(|v| v * v).sum();
    if den > 0.0 {
        (num / den).sqrt()
    } else {
        num.sqrt()
    }
}

/// Ramsay–Li continuous registration of curves to a target (R `fda::register.fd`).
///
/// Every curve `yᵢ` (row of `data`, observed on `argvals`) is warped by a smooth
/// strictly increasing `hᵢ(t) = a + (b−a)·∫ₐᵗ exp Wᵢ / ∫ₐᵇ exp Wᵢ` with
/// `Wᵢ` a clamped B-spline expansion, chosen to minimise the least-squares or
/// the Ramsay–Li minimum-eigenvalue criterion `2·λ_min` of the cross-product
/// matrix of `(y₀, yᵢ∘hᵢ)` (see [`ContinuousRegistrationCriterion`]) plus a
/// roughness penalty `λ ∫ (Dᴸ Wᵢ)²`. Curves are optimised independently (in
/// parallel with the `parallel` feature) by penalised Gauss–Newton with
/// Levenberg–Marquardt step control on the exact gradient.
///
/// The returned `criterion` follows R's scaling: `crit = 1` is
/// `mean((y₀ − y∘h)²)`, `crit = 2` is `(aa + cc) − √((aa − cc)² + 4bb²)` with
/// `aa = mean y₀²`, `bb = mean y₀·(y∘h)`, `cc = mean (y∘h)²` over the fine grid.
///
/// # Arguments
/// * `data` — curves to register (`n × m`).
/// * `argvals` — strictly increasing evaluation points (length `m ≥ 4`).
/// * `target` — target curve on `argvals` (`None` = cross-sectional mean, R's
///   default `mean.fd(yfd)`; then `config.max_template_iter > 1` re-estimates it
///   as the mean of the registered curves, Procrustes-style).
/// * `config` — see [`ContinuousRegistrationConfig`].
///
/// # Errors
/// [`FdarError::InvalidDimension`] for empty data, `m < 4`, or mismatched
/// `argvals` / `target` / `initial_coefficients`; [`FdarError::InvalidParameter`]
/// for non-finite data, non-increasing `argvals` or invalid configuration.
///
/// # Example
/// ```
/// use fdars_core::alignment::{continuous_registration, ContinuousRegistrationConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 51;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let g = |s: f64| (-(s - 0.5).powi(2) / 0.02).exp();
/// let mut data = FdMatrix::zeros(3, m);
/// for (i, a) in [-0.2, 0.0, 0.2].iter().enumerate() {
///     for j in 0..m {
///         data[(i, j)] = g(t[j] + a * t[j] * (1.0 - t[j]));
///     }
/// }
/// let res = continuous_registration(&data, &t, None, &ContinuousRegistrationConfig::default())
///     .unwrap();
/// // Warps are strictly increasing and fix the endpoints.
/// for i in 0..3 {
///     assert_eq!(res.warps[(i, 0)], 0.0);
///     assert_eq!(res.warps[(i, m - 1)], 1.0);
///     assert!((1..m).all(|j| res.warps[(i, j)] > res.warps[(i, j - 1)]));
/// }
/// ```
#[must_use = "expensive nonlinear registration whose result should not be discarded"]
pub fn continuous_registration(
    data: &FdMatrix,
    argvals: &[f64],
    target: Option<&[f64]>,
    config: &ContinuousRegistrationConfig,
) -> Result<ContinuousRegistrationResult, FdarError> {
    validate_inputs(data, argvals, target, config)?;
    let (n, m) = data.shape();
    let nfine = config.n_fine.unwrap_or_else(|| m.max(201));
    let setup = CcSetup::new(argvals[0], argvals[m - 1], nfine, config);
    let splines: Vec<NaturalSpline> = (0..n)
        .map(|i| NaturalSpline::new(argvals, &data.row(i)))
        .collect();
    let mut coefs: Vec<Vec<f64>> = match &config.initial_coefficients {
        Some(init) => (0..n).map(|i| init.row(i)).collect(),
        None => vec![vec![0.0; config.nbasis]; n],
    };
    let mut y0 = match target {
        Some(t) => t.to_vec(),
        None => crate::fdata::mean_1d(data),
    };
    let passes = if target.is_some() {
        1
    } else {
        config.max_template_iter
    };
    let mut n_template_iter = 0;
    loop {
        n_template_iter += 1;
        let fits = register_pass(&setup, &splines, argvals, &y0, &coefs, config);
        let (reg, warps) = registered_rows(&setup, &splines, argvals, &fits);
        coefs = fits.iter().map(|f| f.c.clone()).collect();
        let new_target = crate::fdata::mean_1d(&reg);
        let stop = n_template_iter >= passes || rel_change(&new_target, &y0) <= config.template_tol;
        if stop {
            return Ok(assemble(reg, warps, &fits, &setup, y0, n_template_iter));
        }
        y0 = new_target;
    }
}

fn assemble(
    registered_data: FdMatrix,
    warps: FdMatrix,
    fits: &[CurveFit],
    setup: &CcSetup,
    target: Vec<f64>,
    n_template_iter: usize,
) -> ContinuousRegistrationResult {
    let n = fits.len();
    let mut w_coefficients = FdMatrix::zeros(n, setup.nb);
    for (i, f) in fits.iter().enumerate() {
        for (k, &ck) in f.c.iter().enumerate() {
            w_coefficients[(i, k)] = ck;
        }
    }
    ContinuousRegistrationResult {
        registered_data,
        warps,
        w_coefficients,
        w_knots: setup.knots.clone(),
        criterion: fits
            .iter()
            .map(|f| *f.history.last().unwrap_or(&f64::NAN))
            .collect(),
        target,
        n_iter: fits.iter().map(|f| f.n_iter).collect(),
        converged: fits.iter().map(|f| f.converged).collect(),
        n_template_iter,
    }
}

#[cfg(test)]
#[path = "continuous_tests.rs"]
mod tests;
