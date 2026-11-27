//! Registration of exponential-family functional data with generalized FPCA
//! (R `registr` 2.2.1 `register_fpca`; Wrobel, Zipunnikov, Schrack & Goldsmith
//! 2019, "Registration for exponential family functional data", *Biometrics*
//! 75(1):48–57, doi:10.1111/biom.12963; Wrobel & Bauer 2021, *JOSS* 6(61):2964).
//!
//! Curves `Yᵢ(t*ᵢⱼ)` — gaussian, binary, gamma or count valued, observed on
//! curve-specific (irregular) grids — are modelled as
//!
//! ```text
//! g(E[Yᵢ(t*)]) = θᵢ(hᵢ⁻¹(t*)),   θᵢ(t) = Θ(t)ᵀ(α + ψ sᵢ),
//! ```
//!
//! with `g` the canonical link, `θᵢ` a curve-specific latent template from a
//! generalized FPCA, and `hᵢ⁻¹(t*) = Θ_h(t*)ᵀβᵢ` a monotone cubic B-spline
//! *inverse* warp (`Kh` basis functions on the curve's own observed range, inner
//! knots at quantiles of its time points). Monotonicity is enforced by
//! `βᵢ₁ ≤ … ≤ βᵢ,Kh`; for complete curves the endpoints are fixed
//! (`βᵢ₁ = t*ᵢ,min`, `βᵢ,Kh = t*ᵢ,max`).
//!
//! [`register_fpca`] alternates
//! 1. **registration** (registr `registr`): every curve's `βᵢ` maximises its
//!    likelihood given its current template (first pass: the population mean
//!    from a spline GLM, [`register_expfam`]);
//! 2. **GFPCA** of the registered data `(t̂ᵢⱼ, Yᵢⱼ)`: variational EM (registr
//!    `bfpca` / `fpca_gauss`, [`gfpca_variational`], default) or the
//!    family-specific two-step GFPCA (registr `gfpca_twoStep`,
//!    [`gfpca_two_step`], [`FpcaType::TwoStep`]),
//!
//! until the mean squared change of `t̂ / t_max` is `≤ 1e-4` or
//! `max_iterations` is reached, followed by a final GFPCA.
//!
//! # Incomplete curves
//!
//! With [`RegisterFpcaConfig::incompleteness`] the warps may start
//! ([`CurveIncompleteness::Leading`]), end ([`CurveIncompleteness::Trailing`])
//! or both ([`CurveIncompleteness::Full`]) off the diagonal, anywhere in the
//! global domain `[t_min, t_max]`, and the negative log-likelihood is penalised
//! by `nᵢ · λ_inc · pen`, with `pen = (hᵢ⁻¹(t*ᵢ,min) − t*ᵢ,min)²`,
//! `(hᵢ⁻¹(t*ᵢ,max) − t*ᵢ,max)²` or `((hᵢ⁻¹(t*ᵢ,max) − hᵢ⁻¹(t*ᵢ,min)) −
//! (t*ᵢ,max − t*ᵢ,min))²` (registr `loss_h`).
//!
//! # Deviations from registr 2.2.1
//!
//! * **Optimiser.** R minimises over the free `β` entries with `constrOptim`
//!   (adaptive log-barrier, BFGS with the analytic gradient for gaussian /
//!   binomial, Nelder–Mead for gamma / poisson). fdars uses an exact
//!   reparameterisation of the constraint set — `βₗ = A + S Σ_{k<l} p_k` with
//!   `p = softmax(0, u)` and the free endpoint `A` / span `S` through logistic
//!   transforms — and L-BFGS with analytic gradients for all four families (the
//!   gamma scale on the log scale). The optimum is therefore that of the
//!   constrained problem without a barrier term: registration losses are equal
//!   to or lower than R's (strictly monotone warps; R allows ties).
//! * **Random initialisation.** The GFPCA starts from `0.5·N(0, 1)` FPC
//!   coefficients (and ξ for binary data) drawn from Rust's `StdRng` seeded with
//!   [`GfpcaConfig::seed`], not R's Mersenne twister; pass
//!   [`GfpcaConfig::psi_init`] / [`GfpcaConfig::xi_init`] to reproduce a given
//!   start. Results agree with R up to EM convergence tolerance.
//! * **Template GLM** by IRLS to a relative deviance change of `1e-12` (R `glm`:
//!   `1e-8`; `fastglm` is used when installed).
//! * The gamma log-likelihood uses `ln Γ` instead of `log(gamma(·))` (identical
//!   where R is finite) and `ln Γ(y + 1)` for `log(y!)`.
//! * **Two-step GFPCA without mgcv / gamm4.** The P-spline mean and the
//!   tensor-product covariance smooth select their smoothing parameters by
//!   minimising the same GCV / UBRE criteria as mgcv's default `"GCV.Cp"` (grid
//!   search plus golden-section / L-BFGS refinement instead of mgcv's Newton
//!   iteration); the mixed model is fitted by penalised IRLS (Newton steps for
//!   the gamma log link) and L-BFGS on lme4's REML / Laplace criteria with
//!   finite-difference gradients (lme4: bobyqa / Nelder–Mead). Agreement with R
//!   is to the optimisers' tolerances (covariance ≈ 1e-6, mean and scores ≈ 1e-7
//!   gaussian, ≲ 5e-4 for the Laplace fits). The smooth's variance component is
//!   reported in fdars' penalty scaling, and the variance parameters of the
//!   previous iteration warm-start the next fit, as registr does.
//!   registr's latent-scale transformation of the covariance is reproduced as
//!   is (gamma / poisson: `diag(η) C diag(η)` with `η` the log mean).
//!   `index_significant_digits` rounding uses the number of integer digits of
//!   the largest `|t|` (registr's `nchar()` rule without R's scientific
//!   notation for ≥ 10⁵).
//! * **Two-step registration templates** are the spline fits of the latent
//!   two-step predictions on the grid points clamped to `[t_min, t_max]` (R's
//!   `bs` stretches its boundary to rounded grid points just outside), and the
//!   final `yhat` is not truncated at each curve's warped end (registr's
//!   cosmetic `Yhat` filter).
//! * **`periodic`** uses the periodic basis consistently for the two-step
//!   knots as well (registr passes `periodic` only to registration and the
//!   variational EM, so a two-step fit keeps `Kt − 4` knots there).
//! * **Piecewise-linear warps** are evaluated at the scaled observation times
//!   (registr: an equally spaced grid of the same length, identical for
//!   equally spaced complete curves) and optimised by L-BFGS plus a Nelder–Mead
//!   polish over an exact parameterisation of the constraint triangles; the
//!   initial registration also starts from the best point of a coarse grid of
//!   knot configurations (the loss is multi-modal for binary data), so losses
//!   are equal to or lower than R's `constrOptim` / Nelder–Mead.
//! * Not ported: the > 10⁷-row subsampling of the template GLM (registr subsamples
//!   curves only above 10⁷ observations, a memory guard irrelevant in fdars).

mod basis;
mod gfpca;
mod glmm;
mod mgcv;
mod pl2;
mod twostep;
mod warp;

#[cfg(test)]
mod golden;
#[cfg(test)]
mod golden_twostep;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_twostep;

pub use gfpca::{gfpca_variational, FpcaType, GfpcaConfig, GfpcaResult, NpcCriterion};
pub use twostep::gfpca_two_step;

use crate::error::FdarError;
use crate::irreg_fdata::IrregFdata;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use basis::{dot, glm_fit, quantile_knots, template_knots, CubicBs, TemplateBasis};
use gfpca::{check_family_values, curve_data, gfpca_core, CurveData, KT_NPC_CRITERION};
use pl2::{register_curve_pl2, Pl2Problem, PL2_START};
use twostep::two_step_core;
use warp::{register_curve, CurveFit, CurveProblem, ObsLoss, TemplateScale, WarpBounds, WarpOptim};

/// Convergence threshold of the outer loop (registr `convergence_threshold`).
const OUTER_THRESHOLD: f64 = 1e-4;

/// Exponential family of the observations (registr `family`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExpFamily {
    /// Gaussian with identity link and unit variance in the registration loss.
    #[default]
    Gaussian,
    /// Bernoulli with logit link (values in `[0, 1]`).
    Binomial,
    /// Gamma with log link (registration template) and a per-curve scale
    /// parameter (values `> 0`).
    Gamma,
    /// Poisson with log link (registration template; values `≥ 0`).
    Poisson,
}

/// Which end(s) of an incompletely observed curve may be warped off the
/// diagonal (registr `incompleteness`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CurveIncompleteness {
    /// Curves may start late: `h⁻¹(t*ᵢ,min)` is free in `[t_min, t*ᵢ,max]`.
    Leading,
    /// Curves may end early: `h⁻¹(t*ᵢ,max)` is free in `[t*ᵢ,min, t_max]`.
    Trailing,
    /// Both endpoints are free within `[t_min, t_max]`.
    Full,
}

/// Inverse-warp family (registr `warping`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarpingType {
    /// Monotone cubic B-spline inverse warps with `kh` coefficients (default).
    #[default]
    Nonparametric,
    /// Two-knot piecewise-linear inverse warps on the per-curve scaled domain
    /// `[0, 1]` (registr `"piecewise_linear2"`), see [`register_expfam`].
    PiecewiseLinear2,
}

/// Configuration of [`register_fpca`] and [`register_expfam`].
///
/// Construct with `RegisterFpcaConfig::default()` and assign fields (the struct
/// is `#[non_exhaustive]`). Defaults are registr 2.2.1's.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegisterFpcaConfig {
    /// Observation family (default gaussian).
    pub family: ExpFamily,
    /// Number of cubic B-spline basis functions of each inverse warp (registr
    /// `Kh`, default 4, `≥ 4`).
    pub kh: usize,
    /// Incomplete-curve handling (default `None`: all warps start and end on the
    /// diagonal).
    pub incompleteness: Option<CurveIncompleteness>,
    /// Endpoint penalty weight `λ_inc ≥ 0` for incomplete curves (default 0).
    pub lambda_inc: f64,
    /// Maximum number of registration/GFPCA iterations after the initial
    /// registration (default 10, `≥ 1`).
    pub max_iterations: usize,
    /// GFPCA settings; `fpca.kt` is also the template basis size `Kt` of the
    /// initial registration (default `npc = 1`, `kt = 8`).
    pub fpca: GfpcaConfig,
    /// Maximum L-BFGS iterations per curve and registration step (default 500).
    pub warp_max_iter: usize,
    /// Relative loss-decrease tolerance of the per-curve optimiser (default `1e-12`).
    pub warp_tol: f64,
    /// Initial gamma scale parameter of every registration step (default 1.5,
    /// as registr).
    pub gamma_scale_init: f64,
    /// GFPCA algorithm of [`register_fpca`] (default
    /// [`FpcaType::VariationalEm`]).
    pub fpca_type: FpcaType,
    /// Inverse-warp family (default [`WarpingType::Nonparametric`]).
    pub warping: WarpingType,
    /// Standard deviation of registr's normal priors on the piecewise-linear
    /// knots (`priors = TRUE, prior_sd`; means 0.25, 0.25, 0.75, 0.75 for
    /// `x₁, y₁, x₂, y₂`). Only with [`WarpingType::PiecewiseLinear2`] and binary
    /// data, as registr (default `None`).
    pub prior_sd: Option<f64>,
}

impl Default for RegisterFpcaConfig {
    fn default() -> Self {
        Self {
            family: ExpFamily::Gaussian,
            kh: 4,
            incompleteness: None,
            lambda_inc: 0.0,
            max_iterations: 10,
            fpca: GfpcaConfig::default(),
            warp_max_iter: 500,
            warp_tol: 1e-12,
            gamma_scale_init: 1.5,
            fpca_type: FpcaType::VariationalEm,
            warping: WarpingType::Nonparametric,
            prior_sd: None,
        }
    }
}

impl RegisterFpcaConfig {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `kh < 4`, `max_iterations == 0`,
    /// `warp_max_iter == 0`, `lambda_inc` / `warp_tol` negative or not finite,
    /// `gamma_scale_init` not finite and positive, or `fpca` is invalid.
    pub fn validate(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        if self.kh < 4 {
            return bad("kh", format!("must be >= 4, got {}", self.kh));
        }
        if self.max_iterations == 0 {
            return bad("max_iterations", "must be >= 1".into());
        }
        if self.warp_max_iter == 0 {
            return bad("warp_max_iter", "must be >= 1".into());
        }
        for (name, v) in [("lambda_inc", self.lambda_inc), ("warp_tol", self.warp_tol)] {
            if !(v.is_finite() && v >= 0.0) {
                return bad(name, format!("must be finite and >= 0, got {v}"));
            }
        }
        if !(self.gamma_scale_init.is_finite() && self.gamma_scale_init > 0.0) {
            return bad(
                "gamma_scale_init",
                format!("must be finite and > 0, got {}", self.gamma_scale_init),
            );
        }
        self.validate_warping()?;
        self.fpca.validate()
    }

    fn validate_warping(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: &str| {
            Err(FdarError::InvalidParameter {
                parameter,
                message: message.into(),
            })
        };
        let pl2 = self.warping == WarpingType::PiecewiseLinear2;
        if pl2 && self.incompleteness.is_some() {
            return bad(
                "incompleteness",
                "incomplete curves need warping = Nonparametric (as registr)",
            );
        }
        if let Some(sd) = self.prior_sd {
            if !(pl2 && self.family == ExpFamily::Binomial) {
                return bad(
                    "prior_sd",
                    "priors need warping = PiecewiseLinear2 and family = Binomial (as registr)",
                );
            }
            if !(sd.is_finite() && sd > 0.0) {
                return bad("prior_sd", "must be finite and > 0");
            }
        }
        Ok(())
    }
}

/// Result of one registration pass ([`register_expfam`], registr `registr`).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ExpFamRegistration {
    /// Registered times `t̂ᵢⱼ = hᵢ⁻¹(t*ᵢⱼ)` per curve (non-decreasing).
    pub t_hat: Vec<Vec<f64>>,
    /// Inverse-warp spline coefficients `βᵢ` (length `kh` per curve; registr
    /// `hinv_beta`); for [`WarpingType::PiecewiseLinear2`] the knot coordinates
    /// `(x₁, y₁, x₂, y₂)`.
    pub hinv_beta: Vec<Vec<f64>>,
    /// Interior knots of each curve's inverse-warp basis (registr
    /// `hinv_innerKnots`; empty for piecewise-linear warps).
    pub hinv_inner_knots: Vec<Vec<f64>>,
    /// Per-curve minimised (penalised) negative log-likelihood.
    pub curve_loss: Vec<f64>,
    /// Total loss (registr `loss`).
    pub loss: f64,
    /// Estimated gamma scale per curve (gamma family only).
    pub gamma_scales: Option<Vec<f64>>,
    /// Interior knots of the template basis.
    pub template_knots: Vec<f64>,
    /// Spline coefficients of the (population) template (length `kt`): the GLM
    /// mean on the link scale, or the GFPCA mean `α` for re-registrations.
    pub template_coefs: Vec<f64>,
    /// Per-curve optimiser convergence flags.
    pub converged: Vec<bool>,
}

impl ExpFamRegistration {
    /// Registered data: the input values on the registered times `t̂`.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not the data that was registered.
    #[must_use]
    pub fn registered_data(&self, data: &IrregFdata) -> IrregFdata {
        registered(&self.t_hat, data)
    }
}

/// Result of [`register_fpca`] (registr `registration` object).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct RegisterFpcaResult {
    /// Final registration (warps, losses, gamma scales).
    pub registration: ExpFamRegistration,
    /// Final GFPCA of the registered data.
    pub fpca: GfpcaResult,
    /// Whether the outer loop met the `1e-4` warp-change criterion.
    pub converged: bool,
    /// Number of registration/GFPCA iterations after the initial registration.
    pub iterations: usize,
    /// Mean squared change of `t̂ / t_max` per iteration (registr `delta_index`).
    pub delta_index: Vec<f64>,
    /// Total registration loss of the initial and every later registration
    /// (registr `registration_loss`).
    pub registration_loss: Vec<f64>,
}

impl RegisterFpcaResult {
    /// Registered data: the input values on the registered times `t̂`.
    ///
    /// # Panics
    ///
    /// Panics if `data` is not the data that was registered.
    #[must_use]
    pub fn registered_data(&self, data: &IrregFdata) -> IrregFdata {
        registered(&self.registration.t_hat, data)
    }
}

fn registered(t_hat: &[Vec<f64>], data: &IrregFdata) -> IrregFdata {
    let values: Vec<Vec<f64>> = (0..data.n_obs())
        .map(|i| data.get_obs(i).1.to_vec())
        .collect();
    IrregFdata::from_lists(t_hat, &values)
}

/// Template used by one registration pass.
enum Template<'a> {
    /// One template for all curves (GLM mean, link scale).
    Shared(&'a [f64]),
    /// Curve-specific GFPCA templates (rows of `subject_coefs`) and the GFPCA mean.
    PerCurve(&'a FdMatrix, &'a [f64]),
}

/// Inverse-warp basis of one curve.
struct WarpBasis {
    inner_knots: Vec<f64>,
    theta: Vec<Vec<f64>>,
    greville: Vec<f64>,
}

fn warp_basis(t: &[f64], kh: usize) -> WarpBasis {
    let inner_knots = quantile_knots(t, kh - 4);
    let b = CubicBs::new(t[0], t[t.len() - 1], &inner_knots);
    WarpBasis {
        theta: t.iter().map(|&x| b.eval(x)).collect(),
        greville: b.greville(),
        inner_knots,
    }
}

/// Shared state of a registration run.
struct Setup<'a> {
    data: &'a CurveData,
    cfg: &'a RegisterFpcaConfig,
    t_min: f64,
    t_max: f64,
    warp: Vec<WarpBasis>,
}

impl Setup<'_> {
    fn bounds(&self, i: usize) -> WarpBounds {
        let t = &self.data.t[i];
        WarpBounds {
            t_min: self.t_min,
            t_max: self.t_max,
            t_min_i: t[0],
            t_max_i: t[t.len() - 1],
        }
    }

    /// One registration pass (registr `registr`) against `template` with the
    /// given template basis knots, from `start` (or the identity).
    fn pass(
        &self,
        knots: &[f64],
        template: &Template<'_>,
        scale: TemplateScale,
        start: Option<&[Vec<f64>]>,
    ) -> ExpFamRegistration {
        let basis = TemplateBasis::new(self.t_min, self.t_max, knots, self.cfg.fpca.periodic);
        let loss = ObsLoss {
            family: self.cfg.family,
            scale,
        };
        let opt = WarpOptim {
            max_iter: self.cfg.warp_max_iter,
            tol: self.cfg.warp_tol,
            gamma_scale_init: self.cfg.gamma_scale_init,
        };
        let n = self.data.t.len();
        let fits: Vec<CurveFit> = iter_maybe_parallel!(0..n)
            .map(|i| {
                let coefs: Vec<f64> = match template {
                    Template::Shared(c) => c.to_vec(),
                    Template::PerCurve(m, _) => m.row(i),
                };
                self.fit_curve(i, &basis, &coefs, loss, start, opt)
            })
            .collect();
        let template_coefs = match template {
            Template::Shared(c) => c.to_vec(),
            Template::PerCurve(_, alpha) => alpha.to_vec(),
        };
        self.collect(fits, knots.to_vec(), template_coefs)
    }

    /// Register curve `i` against the template `coefs` in `basis`.
    fn fit_curve(
        &self,
        i: usize,
        basis: &TemplateBasis,
        coefs: &[f64],
        loss: ObsLoss,
        start: Option<&[Vec<f64>]>,
        opt: WarpOptim,
    ) -> CurveFit {
        if self.cfg.warping == WarpingType::PiecewiseLinear2 {
            let prob = Pl2Problem {
                s: &self.data.t[i],
                y: &self.data.y[i],
                template_basis: basis,
                template_coefs: coefs,
                loss,
                prior_sd: self.cfg.prior_sd,
            };
            let beta0 = start.map_or(&PL2_START[..], |s| &s[i]);
            return register_curve_pl2(&prob, beta0, opt, start.is_none());
        }
        let prob = CurveProblem {
            y: &self.data.y[i],
            theta_h: &self.warp[i].theta,
            template_basis: basis,
            template_coefs: coefs,
            loss,
            incompleteness: self.cfg.incompleteness,
            lambda_inc: self.cfg.lambda_inc,
            bounds: self.bounds(i),
        };
        let beta0 = start.map_or(&self.warp[i].greville, |s| &s[i]);
        register_curve(&prob, beta0, opt)
    }

    fn collect(
        &self,
        fits: Vec<CurveFit>,
        template_knots: Vec<f64>,
        template_coefs: Vec<f64>,
    ) -> ExpFamRegistration {
        let gamma = self.cfg.family == ExpFamily::Gamma;
        let curve_loss: Vec<f64> = fits.iter().map(|f| f.loss).collect();
        ExpFamRegistration {
            loss: curve_loss.iter().sum(),
            curve_loss,
            gamma_scales: gamma.then(|| fits.iter().map(|f| f.scale.unwrap_or(f64::NAN)).collect()),
            converged: fits.iter().map(|f| f.converged).collect(),
            hinv_inner_knots: if self.cfg.warping == WarpingType::PiecewiseLinear2 {
                vec![Vec::new(); self.warp.len()]
            } else {
                self.warp.iter().map(|w| w.inner_knots.clone()).collect()
            },
            t_hat: fits.iter().map(|f| f.t_hat.clone()).collect(),
            hinv_beta: fits.into_iter().map(|f| f.beta).collect(),
            template_knots,
            template_coefs,
        }
    }

    /// Initial registration against the GLM mean of `template` (or of the data).
    fn initial(&self, template: Option<&CurveData>) -> Result<ExpFamRegistration, FdarError> {
        let src = template.unwrap_or(self.data);
        let kt = self.cfg.fpca.kt;
        let periodic = self.cfg.fpca.periodic;
        let knots = template_knots(&src.pooled_sorted_t(), kt, periodic);
        let basis = TemplateBasis::new(self.t_min, self.t_max, &knots, periodic);
        let rows: Vec<Vec<f64>> = src.t.iter().flatten().map(|&x| basis.eval(x)).collect();
        let ys: Vec<f64> = src.y.iter().flatten().copied().collect();
        let coefs = glm_fit(&rows, &ys, self.cfg.family)?;
        Ok(self.pass(&knots, &Template::Shared(&coefs), TemplateScale::Link, None))
    }

    /// GFPCA of the registered data (`prev`: previous GFPCA, whose mixed-model
    /// variance parameters warm-start the two-step fit as registr does).
    fn fpca(
        &self,
        reg: &ExpFamRegistration,
        prev: Option<&GfpcaResult>,
    ) -> Result<GfpcaResult, FdarError> {
        let registered = CurveData {
            t: reg.t_hat.clone(),
            y: self.data.y.clone(),
        };
        match self.cfg.fpca_type {
            FpcaType::TwoStep => two_step_core(
                &registered,
                self.cfg.family,
                &self.cfg.fpca,
                Some([self.t_min, self.t_max]),
                prev.and_then(|p| p.variance_components.as_deref()),
            ),
            _ => gfpca_core(&registered, self.cfg.family, &self.cfg.fpca),
        }
    }

    fn delta(&self, new: &ExpFamRegistration, old: &ExpFamRegistration) -> f64 {
        let n_total = self.data.n_total() as f64;
        let ss: f64 = new
            .t_hat
            .iter()
            .flatten()
            .zip(old.t_hat.iter().flatten())
            .map(|(a, b)| ((a - b) / self.t_max).powi(2))
            .sum();
        ss / n_total
    }
}

fn validate_template(template: &CurveData, data: &CurveData) -> Result<(), FdarError> {
    let range = |d: &CurveData| {
        let p = d.pooled_sorted_t();
        (p[0], p[p.len() - 1])
    };
    let (a, b) = (range(template), range(data));
    let tol = 1e-12 * (b.1 - b.0).abs().max(1.0);
    if (a.0 - b.0).abs() > tol || (a.1 - b.1).abs() > tol {
        return Err(FdarError::InvalidParameter {
            parameter: "template",
            message: format!("time range {a:?} must equal the data range {b:?}"),
        });
    }
    Ok(())
}

/// Curves on their per-curve scaled domain `[0, 1]` (registr `index_scaled`,
/// used by piecewise-linear warping).
fn scaled_curves(data: &CurveData) -> CurveData {
    CurveData {
        t: data
            .t
            .iter()
            .map(|t| {
                let (lo, hi) = (t[0], t[t.len() - 1]);
                t.iter().map(|&v| (v - lo) / (hi - lo)).collect()
            })
            .collect(),
        y: data.y.clone(),
    }
}

/// Validated curve data, on the scaled domain for piecewise-linear warping.
fn prepared(
    data: &IrregFdata,
    what: &'static str,
    cfg: &RegisterFpcaConfig,
) -> Result<CurveData, FdarError> {
    let cd = curve_data(data, what)?;
    Ok(if cfg.warping == WarpingType::PiecewiseLinear2 {
        scaled_curves(&cd)
    } else {
        cd
    })
}

/// Effective configuration of [`register_fpca`]: registr sets `Kt = 20` for the
/// variational EM when `npc_criterion` is given.
fn effective_config(cfg: &RegisterFpcaConfig) -> RegisterFpcaConfig {
    let mut c = cfg.clone();
    if c.fpca_type == FpcaType::VariationalEm && c.fpca.npc_criterion.is_some() {
        c.fpca.kt = KT_NPC_CRITERION;
    }
    c
}

fn build_setup<'a>(
    data: &'a CurveData,
    cfg: &'a RegisterFpcaConfig,
) -> Result<Setup<'a>, FdarError> {
    cfg.validate()?;
    check_family_values(data, cfg.family)?;
    let pooled = data.pooled_sorted_t();
    let (t_min, t_max) = (pooled[0], pooled[pooled.len() - 1]);
    if t_max <= t_min {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: "time points must span a non-degenerate interval".into(),
        });
    }
    Ok(Setup {
        data,
        cfg,
        t_min,
        t_max,
        warp: data.t.iter().map(|t| warp_basis(t, cfg.kh)).collect(),
    })
}

fn template_data(
    template: Option<&IrregFdata>,
    data: &CurveData,
    cfg: &RegisterFpcaConfig,
) -> Result<Option<CurveData>, FdarError> {
    template
        .map(|tpl| {
            let td = prepared(tpl, "template", cfg)?;
            check_family_values(&td, cfg.family)?;
            validate_template(&td, data)?;
            Ok(td)
        })
        .transpose()
}

/// One likelihood-based registration pass of exponential-family curves to their
/// population mean (R registr 2.2.1 `registr(Y, Kt, Kh, family, incompleteness,
/// lambda_inc, Y_template)` without an FPCA object).
///
/// The template is the spline GLM `g(E[Y(t)]) = Θ(t)ᵀc` of the pooled data (or
/// of `template`, whose time range must equal the data's) with `config.fpca.kt`
/// cubic B-splines (interior knots at pooled-time quantiles); every curve's
/// monotone inverse warp is then estimated by maximising its likelihood. See the
/// module documentation for the model and the deviations from R.
///
/// With `config.fpca.periodic` the template is a periodic cubic B-spline (registr
/// `periodic = TRUE`, `pbs`). With [`WarpingType::PiecewiseLinear2`] every
/// curve is first mapped to its scaled domain `[0, 1]` (registr `index_scaled`)
/// and warped by the polygon through `(0, 0)`, `(x₁, y₁)`, `(x₂, y₂)`, `(1, 1)`
/// (`0.1 ≤ x₁`, `x₂ − x₁ ≥ 0.1`, `x₂ ≤ 0.9`, same for `y`; start
/// `(0.25, 0.3, 0.75, 0.8)`), optionally with registr's normal priors
/// ([`RegisterFpcaConfig::prior_sd`]); `t_hat` is then on the scaled domain.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for fewer
/// than 2 curves, curves with < 2 points or non-increasing argvals, values out
/// of the family range, a template with a different time range, or an invalid
/// `config`; [`FdarError::ComputationFailed`] if the template GLM is singular.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{register_expfam, ExpFamily, RegisterFpcaConfig};
/// use fdars_core::irreg_fdata::IrregFdata;
///
/// // Shifted bumps observed on a common grid.
/// let t: Vec<f64> = (0..40).map(|j| j as f64 / 39.0).collect();
/// let argvals: Vec<Vec<f64>> = (0..8).map(|_| t.clone()).collect();
/// let values: Vec<Vec<f64>> = (0..8)
///     .map(|i| {
///         let c = 0.4 + 0.03 * i as f64;
///         t.iter().map(|&s| (-(s - c).powi(2) / 0.02).exp()).collect()
///     })
///     .collect();
/// let data = IrregFdata::from_lists(&argvals, &values);
/// let reg = register_expfam(&data, None, &RegisterFpcaConfig::default()).unwrap();
/// // Warps are monotone and keep the endpoints of complete curves.
/// for th in &reg.t_hat {
///     assert!(th.windows(2).all(|w| w[1] >= w[0]));
///     assert!((th[0] - 0.0).abs() < 1e-12 && (th[39] - 1.0).abs() < 1e-12);
/// }
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn register_expfam(
    data: &IrregFdata,
    template: Option<&IrregFdata>,
    config: &RegisterFpcaConfig,
) -> Result<ExpFamRegistration, FdarError> {
    let cd = prepared(data, "data", config)?;
    let setup = build_setup(&cd, config)?;
    let td = template_data(template, &cd, config)?;
    setup.initial(td.as_ref())
}

/// Joint registration and generalized FPCA of exponential-family functional
/// data (R registr 2.2.1 `register_fpca`).
///
/// Starts with [`register_expfam`] (registration to the population mean of the
/// data or of `template`), then alternates [`gfpca_variational`] of the
/// registered data and re-registration of every curve to its own GFPCA fit
/// `Θ(t)ᵀ(α + ψ mᵢ)` (warm-started from the previous warp), for at most
/// `config.max_iterations` iterations or until the mean squared change of
/// `t̂ / t_max` is `≤ 1e-4`; a final GFPCA of the registered data is returned.
/// Gamma and poisson data use the gaussian GFPCA and a template on the mean
/// scale (registr's `*-varEM` loss), as R does for `fpca_type = "variationalEM"`.
///
/// With [`FpcaType::TwoStep`] the GFPCA step is [`gfpca_two_step`]
/// (family-specific for all four families; warm-started from the previous
/// mixed-model variance parameters, as registr) and every curve is
/// re-registered to the spline fit of its latent two-step prediction on the
/// link scale (registr `registr_oneCurve` with a two-step FPCA object). With
/// [`GfpcaConfig::npc_criterion`] the number of components is chosen in every
/// GFPCA step; for the variational EM registr then uses `Kt = 20` throughout
/// (registration templates included), which fdars reproduces.
///
/// See the module documentation for the model, incomplete curves and
/// the deviations from R.
///
/// # Errors
///
/// As [`register_expfam`], plus [`FdarError::ComputationFailed`] if a GFPCA
/// linear system is singular.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{register_fpca, ExpFamily, RegisterFpcaConfig};
/// use fdars_core::irreg_fdata::IrregFdata;
///
/// let t: Vec<f64> = (0..40).map(|j| j as f64 / 39.0).collect();
/// let argvals: Vec<Vec<f64>> = (0..10).map(|_| t.clone()).collect();
/// let values: Vec<Vec<f64>> = (0..10)
///     .map(|i| {
///         let c = 0.4 + 0.02 * i as f64;
///         let a = 1.0 + 0.05 * i as f64;
///         t.iter().map(|&s| a * (-(s - c).powi(2) / 0.02).exp()).collect()
///     })
///     .collect();
/// let data = IrregFdata::from_lists(&argvals, &values);
/// let mut cfg = RegisterFpcaConfig::default();
/// cfg.family = ExpFamily::Gaussian;
/// cfg.fpca.npc = 1;
/// let fit = register_fpca(&data, None, &cfg).unwrap();
/// assert_eq!(fit.registration.t_hat.len(), 10);
/// assert_eq!(fit.fpca.efunctions.shape(), (100, 1));
/// assert!(fit.registration_loss.len() >= 2);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn register_fpca(
    data: &IrregFdata,
    template: Option<&IrregFdata>,
    config: &RegisterFpcaConfig,
) -> Result<RegisterFpcaResult, FdarError> {
    let config = &effective_config(config);
    let cd = prepared(data, "data", config)?;
    let setup = build_setup(&cd, config)?;
    let td = template_data(template, &cd, config)?;
    let scale = match (config.fpca_type, config.family) {
        (FpcaType::TwoStep, _) | (_, ExpFamily::Gaussian | ExpFamily::Binomial) => {
            TemplateScale::Link
        }
        _ => TemplateScale::Mean,
    };
    let mut reg = setup.initial(td.as_ref())?;
    let mut registration_loss = vec![reg.loss];
    let mut delta_index = Vec::new();
    let mut prev: Option<GfpcaResult> = None;
    while delta_index.is_empty()
        || (delta_index.len() < config.max_iterations
            && delta_index.last().is_some_and(|&d| d > OUTER_THRESHOLD))
    {
        let fpca = setup.fpca(&reg, prev.as_ref())?;
        let next = setup.pass(
            &fpca.knots,
            &Template::PerCurve(&fpca.subject_coefs, &fpca.alpha_coefs),
            scale,
            Some(&reg.hinv_beta),
        );
        delta_index.push(setup.delta(&next, &reg));
        registration_loss.push(next.loss);
        reg = next;
        prev = Some(fpca);
    }
    let converged = delta_index.last().is_some_and(|&d| d <= OUTER_THRESHOLD);
    let fpca = setup.fpca(&reg, prev.as_ref())?;
    Ok(RegisterFpcaResult {
        registration: reg,
        fpca,
        converged,
        iterations: delta_index.len(),
        delta_index,
        registration_loss,
    })
}

/// [`register_fpca`] for curves on a common grid (`data` is `n × m`, row `i` is
/// curve `i` on `argvals`).
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] if `argvals.len() != m`, otherwise as
/// [`register_fpca`].
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{register_fpca_regular, RegisterFpcaConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 30;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let mut data = FdMatrix::zeros(6, m);
/// for i in 0..6 {
///     for j in 0..m {
///         data[(i, j)] = (std::f64::consts::PI * (t[j] + 0.02 * i as f64)).sin();
///     }
/// }
/// let fit = register_fpca_regular(&data, &t, &RegisterFpcaConfig::default()).unwrap();
/// assert_eq!(fit.registration.t_hat[0].len(), m);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn register_fpca_regular(
    data: &FdMatrix,
    argvals: &[f64],
    config: &RegisterFpcaConfig,
) -> Result<RegisterFpcaResult, FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    let t: Vec<Vec<f64>> = (0..n).map(|_| argvals.to_vec()).collect();
    let y: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    register_fpca(&IrregFdata::from_lists(&t, &y), None, config)
}

/// Evaluate the inverse warp `h⁻¹(t) = Θ_h(t)ᵀβ` of one curve at `t` (inside the
/// curve's observed range `[t_lo, t_hi]`) from its registr coefficients.
///
/// # Errors
///
/// [`FdarError::InvalidParameter`] if `beta.len() != inner_knots.len() + 4` or
/// `t_hi <= t_lo`.
pub fn evaluate_inverse_warp(
    beta: &[f64],
    inner_knots: &[f64],
    t_lo: f64,
    t_hi: f64,
    t: &[f64],
) -> Result<Vec<f64>, FdarError> {
    if beta.len() != inner_knots.len() + 4
        || !matches!(t_hi.partial_cmp(&t_lo), Some(std::cmp::Ordering::Greater))
    {
        return Err(FdarError::InvalidParameter {
            parameter: "beta",
            message: format!(
                "need beta.len() == inner_knots.len() + 4 ({}) and t_hi > t_lo, got {}",
                inner_knots.len() + 4,
                beta.len()
            ),
        });
    }
    let b = CubicBs::new(t_lo, t_hi, inner_knots);
    Ok(t.iter().map(|&x| dot(&b.eval(x), beta)).collect())
}
