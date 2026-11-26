//! Per-curve likelihood registration (registr `registr_oneCurve`, `loss_h`,
//! `loss_h_gradient`) with an exact monotone reparameterisation of the
//! inverse-warp coefficients and an in-crate L-BFGS optimiser.

use std::collections::VecDeque;

use crate::distributions::ln_gamma;
use crate::metric::soft_dtw::{lbfgs_dot, lbfgs_hessian_scale, two_loop_lbfgs};

use super::basis::{dot, sigmoid, CubicBs};
use super::{CurveIncompleteness, ExpFamily};

/// Lower clamp of the mean on the `*-varEM` scale (registr `loss_h`).
const MU_FLOOR: f64 = 1e-8;

/// Digamma ψ(x) for x > 0 (recurrence to x ≥ 12, then the asymptotic series).
pub(super) fn digamma(mut x: f64) -> f64 {
    let mut acc = 0.0;
    while x < 12.0 {
        acc -= 1.0 / x;
        x += 1.0;
    }
    let inv = 1.0 / x;
    let inv2 = inv * inv;
    acc + x.ln()
        - 0.5 * inv
        - inv2
            * (1.0 / 12.0
                - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0))))
}

/// Scale of the template: the linear predictor (initial GLM template, registr
/// `family`) or the mean itself (gaussian variational-EM FPCA template for
/// gamma / poisson data, registr `family = "gamma-varEM"` / `"poisson-varEM"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum TemplateScale {
    Link,
    Mean,
}

/// Negative log-likelihood of one observation given the template value `g`.
#[derive(Debug, Clone, Copy)]
pub(super) struct ObsLoss {
    pub family: ExpFamily,
    pub scale: TemplateScale,
}

impl ObsLoss {
    /// `(loss, ∂loss/∂g, ∂loss/∂s)` with `s` the gamma rate/scale parameter.
    pub(super) fn eval(self, y: f64, g: f64, s: f64) -> (f64, f64, f64) {
        match self.family {
            ExpFamily::Gaussian => (0.5 * (y - g).powi(2), g - y, 0.0),
            ExpFamily::Binomial => {
                let softplus = if g > 0.0 {
                    g + (-g).exp().ln_1p()
                } else {
                    g.exp().ln_1p()
                };
                (softplus - y * g, sigmoid(g) - y, 0.0)
            }
            ExpFamily::Poisson => {
                let (mu, dmu) = self.mean(g);
                let l = -(y * mu.ln() - ln_gamma(y + 1.0) - mu);
                (l, -(y / mu - 1.0) * dmu, 0.0)
            }
            ExpFamily::Gamma => {
                let (mu, dmu) = self.mean(g);
                let alpha = mu * s;
                let (ly, ls) = (y.ln(), s.ln());
                let psi = digamma(alpha);
                let l = -((alpha - 1.0) * ly - s * y + ls * s * mu - ln_gamma(alpha));
                let dmu_l = -(s * ly + s * ls - s * psi);
                let ds_l = -(mu * ly - y + mu * ls + mu - mu * psi);
                (l, dmu_l * dmu, ds_l)
            }
        }
    }

    /// `(μ, dμ/dg)` for the log-link or clamped-identity (varEM) template.
    fn mean(self, g: f64) -> (f64, f64) {
        match self.scale {
            TemplateScale::Link => {
                let mu = g.min(700.0).exp();
                (mu, mu)
            }
            TemplateScale::Mean if g < MU_FLOOR => (MU_FLOOR, 0.0),
            TemplateScale::Mean => (g, 1.0),
        }
    }
}

/// Domain bounds of one curve: global `[t_min, t_max]` and observed
/// `[t_min_i, t_max_i]`.
#[derive(Debug, Clone, Copy)]
pub(super) struct WarpBounds {
    pub t_min: f64,
    pub t_max: f64,
    pub t_min_i: f64,
    pub t_max_i: f64,
}

/// One curve's registration problem (registr `loss_h` arguments).
pub(super) struct CurveProblem<'a> {
    pub y: &'a [f64],
    /// Inverse-warp basis `Θ_h` at the observed times (`n_i` rows of `Kh`).
    pub theta_h: &'a [Vec<f64>],
    pub template_basis: &'a CubicBs,
    pub template_coefs: &'a [f64],
    pub loss: ObsLoss,
    pub incompleteness: Option<CurveIncompleteness>,
    pub lambda_inc: f64,
    pub bounds: WarpBounds,
}

impl CurveProblem<'_> {
    /// Penalised negative log-likelihood, its gradient with respect to the full
    /// coefficient vector `β` (length `Kh`) and its derivative in `s`.
    pub(super) fn loss_grad(&self, beta: &[f64], s: f64, want_grad: bool) -> (f64, Vec<f64>, f64) {
        let kh = beta.len();
        let mut total = 0.0;
        let mut gb = vec![0.0; kh];
        let mut gs = 0.0;
        for (row, &y) in self.theta_h.iter().zip(self.y) {
            let h = dot(row, beta);
            let g = dot(&self.template_basis.eval(h), self.template_coefs);
            let (l, dl, ds) = self.loss.eval(y, g, s);
            total += l;
            if want_grad {
                let dg = dot(&self.template_basis.deriv(h), self.template_coefs);
                let f = dl * dg;
                for (gk, &rk) in gb.iter_mut().zip(row) {
                    *gk += f * rk;
                }
                gs += ds;
            }
        }
        let (pen, dpen) = self.penalty(beta);
        total += pen;
        if want_grad {
            for (gk, dk) in gb.iter_mut().zip(&dpen) {
                *gk += dk;
            }
        }
        (total, gb, gs)
    }

    /// `n_i · λ_inc · pen` and its gradient (registr `loss_h` penalisation term).
    fn penalty(&self, beta: &[f64]) -> (f64, Vec<f64>) {
        let kh = beta.len();
        let n = self.y.len();
        let mut grad = vec![0.0; kh];
        let Some(inc) = self.incompleteness else {
            return (0.0, grad);
        };
        if self.lambda_inc == 0.0 {
            return (0.0, grad);
        }
        let first = &self.theta_h[0];
        let last = &self.theta_h[n - 1];
        let (h0, h1) = (dot(first, beta), dot(last, beta));
        let b = self.bounds;
        let w = n as f64 * self.lambda_inc;
        let r = match inc {
            CurveIncompleteness::Leading => h0 - b.t_min_i,
            CurveIncompleteness::Trailing => h1 - b.t_max_i,
            CurveIncompleteness::Full => (h1 - h0) - (b.t_max_i - b.t_min_i),
        };
        for k in 0..kh {
            let dr = match inc {
                CurveIncompleteness::Leading => first[k],
                CurveIncompleteness::Trailing => last[k],
                CurveIncompleteness::Full => last[k] - first[k],
            };
            grad[k] = w * 2.0 * r * dr;
        }
        (w * r * r, grad)
    }
}

/// Exact parameterisation of the monotone coefficient vectors allowed by
/// registr's `constraints`: `β_l = A + S·Σ_{k<l} p_k` with `p = softmax(0, u)`,
/// the start `A` and span `S` fixed (complete curves) or free through
/// logistic transforms within `[t_min, t_max]` (incomplete curves).
#[derive(Debug, Clone, Copy)]
pub(super) struct WarpParam {
    pub incompleteness: Option<CurveIncompleteness>,
    pub kh: usize,
    pub bounds: WarpBounds,
}

/// Decoded coefficients with the partial derivatives needed for the pullback.
pub(super) struct Decoded {
    pub beta: Vec<f64>,
    s: f64,
    p: Vec<f64>,
    cum: Vec<f64>,
    da: [f64; 2],
    ds: [f64; 2],
}

/// Relative distance from the constraint boundary enforced when encoding a
/// start value (cf. registr `ensure_proper_beta`). The logistic / softmax maps
/// have vanishing gradients at the boundary, so a warm start that sits on it
/// (e.g. tied coefficients from the previous iteration) is moved slightly
/// inside; the optimiser then re-converges to the boundary if that is optimal.
const FRAC_EPS: f64 = 1e-3;

fn logit(f: f64) -> f64 {
    let f = f.clamp(FRAC_EPS, 1.0 - FRAC_EPS);
    (f / (1.0 - f)).ln()
}

impl WarpParam {
    /// Number of free endpoint parameters.
    pub(super) fn n_end(&self) -> usize {
        match self.incompleteness {
            None => 0,
            Some(CurveIncompleteness::Leading | CurveIncompleteness::Trailing) => 1,
            Some(CurveIncompleteness::Full) => 2,
        }
    }

    /// Number of free parameters (equal to registr's number of free β entries).
    pub(super) fn n_params(&self) -> usize {
        self.n_end() + self.kh - 2
    }

    /// `(A, S, ∂A/∂θ_end, ∂S/∂θ_end)`.
    fn ends(&self, theta: &[f64]) -> (f64, f64, [f64; 2], [f64; 2]) {
        let b = self.bounds;
        match self.incompleteness {
            None => (b.t_min_i, b.t_max_i - b.t_min_i, [0.0; 2], [0.0; 2]),
            Some(CurveIncompleteness::Leading) => {
                let (sg, cap) = (sigmoid(theta[0]), b.t_max_i - b.t_min);
                let s = cap * sg;
                let ds0 = cap * sg * (1.0 - sg);
                (b.t_max_i - s, s, [-ds0, 0.0], [ds0, 0.0])
            }
            Some(CurveIncompleteness::Trailing) => {
                let (sg, cap) = (sigmoid(theta[0]), b.t_max - b.t_min_i);
                (b.t_min_i, cap * sg, [0.0; 2], [cap * sg * (1.0 - sg), 0.0])
            }
            Some(CurveIncompleteness::Full) => {
                let (sa, sb) = (sigmoid(theta[0]), sigmoid(theta[1]));
                let width = b.t_max - b.t_min;
                let a = b.t_min + width * sa;
                let da0 = width * sa * (1.0 - sa);
                let rest = b.t_max - a;
                (
                    a,
                    rest * sb,
                    [da0, 0.0],
                    [-da0 * sb, rest * sb * (1.0 - sb)],
                )
            }
        }
    }

    pub(super) fn decode(&self, theta: &[f64]) -> Decoded {
        let ne = self.n_end();
        let (a, s, da, ds) = self.ends(theta);
        let mut z = Vec::with_capacity(self.kh - 1);
        z.push(0.0);
        z.extend_from_slice(&theta[ne..ne + self.kh - 2]);
        let zmax = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|&v| (v - zmax).exp()).collect();
        let tot: f64 = e.iter().sum();
        let p: Vec<f64> = e.iter().map(|v| v / tot).collect();
        let mut cum = vec![0.0; self.kh];
        for l in 1..self.kh {
            cum[l] = cum[l - 1] + p[l - 1];
        }
        cum[self.kh - 1] = 1.0;
        let beta = cum.iter().map(|&c| a + s * c).collect();
        Decoded {
            beta,
            s,
            p,
            cum,
            da,
            ds,
        }
    }

    /// Gradient with respect to `θ` from the gradient with respect to `β`.
    pub(super) fn pullback(&self, dec: &Decoded, gbeta: &[f64]) -> Vec<f64> {
        let ne = self.n_end();
        let ga: f64 = gbeta.iter().sum();
        let gs: f64 = gbeta.iter().zip(&dec.cum).map(|(g, c)| g * c).sum();
        // G_k = Σ_{l>k} gβ_l
        let mut tail = vec![0.0; self.kh - 1];
        let mut acc = 0.0;
        for k in (0..self.kh - 1).rev() {
            acc += gbeta[k + 1];
            tail[k] = acc;
        }
        let gbar: f64 = dec.p.iter().zip(&tail).map(|(p, t)| p * t).sum();
        let mut out = vec![0.0; self.n_params()];
        for e in 0..ne {
            out[e] = ga * dec.da[e] + gs * dec.ds[e];
        }
        for j in 1..self.kh - 1 {
            out[ne + j - 1] = dec.s * dec.p[j] * (tail[j] - gbar);
        }
        out
    }

    /// Parameters reproducing (an interior version of) `beta`: increments are
    /// mixed with `FRAC_EPS` of the uniform increment and endpoint fractions
    /// clamped to `[FRAC_EPS, 1 − FRAC_EPS]`.
    pub(super) fn encode(&self, beta: &[f64]) -> Vec<f64> {
        let b = self.bounds;
        let kh = self.kh;
        let a = beta[0];
        let span = (beta[kh - 1] - beta[0]).max(1e-12);
        let uniform = span / (kh - 1) as f64;
        let inc: Vec<f64> = beta
            .windows(2)
            .map(|w| (1.0 - FRAC_EPS) * (w[1] - w[0]).max(0.0) + FRAC_EPS * uniform)
            .collect();
        let mut out = Vec::with_capacity(self.n_params());
        match self.incompleteness {
            None => {}
            Some(CurveIncompleteness::Leading) => out.push(logit(span / (b.t_max_i - b.t_min))),
            Some(CurveIncompleteness::Trailing) => out.push(logit(span / (b.t_max - b.t_min_i))),
            Some(CurveIncompleteness::Full) => {
                let fa = (a - b.t_min) / (b.t_max - b.t_min);
                let fa = fa.clamp(FRAC_EPS, 1.0 - FRAC_EPS);
                let a_dec = b.t_min + (b.t_max - b.t_min) * fa;
                out.push(logit(fa));
                out.push(logit(span / (b.t_max - a_dec)));
            }
        }
        let l0 = inc[0].ln();
        out.extend(inc[1..].iter().map(|v| v.ln() - l0));
        out
    }
}

/// Outcome of a minimisation.
pub(super) struct Minimum {
    pub x: Vec<f64>,
    pub f: f64,
    pub converged: bool,
}

/// L-BFGS curvature history (Nocedal & Wright, at most `M_HIST` pairs).
#[derive(Default)]
struct History {
    s: VecDeque<Vec<f64>>,
    y: VecDeque<Vec<f64>>,
}

impl History {
    const M_HIST: usize = 10;

    fn clear(&mut self) {
        self.s.clear();
        self.y.clear();
    }

    /// Store the pair if it satisfies the curvature condition `sᵀy > 0`.
    fn push(&mut self, x_new: &[f64], x: &[f64], g_new: &[f64], g: &[f64]) {
        let s_vec: Vec<f64> = x_new.iter().zip(x).map(|(a, b)| a - b).collect();
        let y_vec: Vec<f64> = g_new.iter().zip(g).map(|(a, b)| a - b).collect();
        if lbfgs_dot(&s_vec, &y_vec) > 1e-20 {
            if self.s.len() == Self::M_HIST {
                self.s.pop_front();
                self.y.pop_front();
            }
            self.s.push_back(s_vec);
            self.y.push_back(y_vec);
        }
    }

    /// Quasi-Newton descent direction (steepest descent after a reset), capped
    /// at `MAX_STEP` in max-norm: an unscaled first step along −g can jump deep
    /// into the saturated region of the softmax / logistic maps, where the
    /// gradient vanishes at a spurious stationary point.
    fn direction(&mut self, g: &[f64], q: &mut Vec<f64>, d: &mut Vec<f64>) {
        const MAX_STEP: f64 = 1.0;
        two_loop_lbfgs(
            g,
            &self.s,
            &self.y,
            lbfgs_hessian_scale(&self.s, &self.y),
            q,
            d,
        );
        if lbfgs_dot(g, d) >= 0.0 {
            self.clear();
            *d = g.iter().map(|v| -v).collect();
        }
        let dmax = d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        if dmax > MAX_STEP {
            d.iter_mut().for_each(|v| *v *= MAX_STEP / dmax);
        }
    }
}

/// Limited-memory BFGS with Armijo backtracking (history 10, steps capped at 1
/// in max-norm). Stops when the relative decrease of `f` is `≤ tol`, the
/// gradient vanishes, or no decrease is possible from a steepest-descent step.
pub(super) fn lbfgs_minimize<F>(f: F, x0: Vec<f64>, max_iter: usize, tol: f64) -> Minimum
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let mut x = x0;
    let (mut fx, mut g) = f(&x);
    let mut hist = History::default();
    let (mut q, mut d) = (Vec::new(), Vec::new());
    for _ in 0..max_iter {
        if g.iter().all(|v| v.abs() <= 1e-12) || !fx.is_finite() {
            let converged = fx.is_finite();
            return Minimum {
                x,
                f: fx,
                converged,
            };
        }
        hist.direction(&g, &mut q, &mut d);
        let Some((x_new, f_new, g_new)) = armijo(&f, &x, fx, &g, &d) else {
            if hist.s.is_empty() {
                return Minimum {
                    x,
                    f: fx,
                    converged: true,
                };
            }
            hist.clear();
            continue;
        };
        hist.push(&x_new, &x, &g_new, &g);
        let decrease = fx - f_new;
        x = x_new;
        fx = f_new;
        g = g_new;
        if decrease <= tol * (fx.abs() + tol) {
            return Minimum {
                x,
                f: fx,
                converged: true,
            };
        }
    }
    Minimum {
        x,
        f: fx,
        converged: false,
    }
}

/// Backtracking Armijo search along `d`; `None` if no decrease is found.
fn armijo<F>(f: &F, x: &[f64], fx: f64, g: &[f64], d: &[f64]) -> Option<(Vec<f64>, f64, Vec<f64>)>
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let slope = lbfgs_dot(g, d);
    let mut step = 1.0;
    for _ in 0..60 {
        let xn: Vec<f64> = x.iter().zip(d).map(|(a, b)| a + step * b).collect();
        let (fnew, gnew) = f(&xn);
        if fnew.is_finite() && fnew <= fx + 1e-4 * step * slope && fnew < fx {
            return Some((xn, fnew, gnew));
        }
        step *= 0.5;
    }
    None
}

/// Result of one curve's registration.
pub(super) struct CurveFit {
    pub beta: Vec<f64>,
    pub t_hat: Vec<f64>,
    pub loss: f64,
    pub scale: Option<f64>,
    pub converged: bool,
}

/// Optimiser settings for one curve.
#[derive(Debug, Clone, Copy)]
pub(super) struct WarpOptim {
    pub max_iter: usize,
    pub tol: f64,
    pub gamma_scale_init: f64,
}

/// Register one curve (registr `registr_oneCurve`) from the start coefficients
/// `beta0`.
pub(super) fn register_curve(prob: &CurveProblem<'_>, beta0: &[f64], opt: WarpOptim) -> CurveFit {
    let param = WarpParam {
        incompleteness: prob.incompleteness,
        kh: beta0.len(),
        bounds: prob.bounds,
    };
    let gamma = prob.loss.family == ExpFamily::Gamma;
    let np = param.n_params();
    let mut x0 = param.encode(beta0);
    if gamma {
        x0.push(opt.gamma_scale_init.ln());
    }
    let objective = |x: &[f64]| {
        let dec = param.decode(&x[..np]);
        let s = if gamma { x[np].exp() } else { 1.0 };
        let (l, gb, gs) = prob.loss_grad(&dec.beta, s, true);
        let mut gx = param.pullback(&dec, &gb);
        if gamma {
            gx.push(gs * s);
        }
        (l, gx)
    };
    let min = lbfgs_minimize(objective, x0, opt.max_iter, opt.tol);
    let beta = param.decode(&min.x[..np]).beta;
    let t_hat = prob.theta_h.iter().map(|row| dot(row, &beta)).collect();
    CurveFit {
        beta,
        t_hat,
        loss: min.f,
        scale: gamma.then(|| min.x[np].exp()),
        converged: min.converged,
    }
}
