//! Infinite-dimensional (preconditioned) HMC kernel for the warps of
//! [`super::bayesian_align_group`], plus the conjugate Gibbs template draw used with it.
//!
//! Port of fdasrsf 2.7.5 `bayesian_functions.infhmc_step` / `dual_avg_step` and the
//! sweep of `fdawarp.time_warping_bayes_infHMC` (fdasrvf_MATLAB; Beskos, Pinski,
//! Sanz-Serna & Stuart 2011, "Hybrid Monte Carlo on Hilbert spaces"):
//!
//! * target per curve `π(c) ∝ exp(−SSE_i(c) / 2σ²) · N(c; 0, C) · 1{ψ ≥ 0}`, `C` the warp
//!   prior covariance (the preconditioner), momentum `v ~ N(0, C)`;
//! * splitting integrator: half kick `v ← v − ε/2 · C∇Φ`, exact rotation of the Gaussian
//!   part `(c, v) ← (c cos ε + v sin ε, v cos ε − c sin ε)`, half kick;
//! * `H = Φ(c) + c'C⁻¹c/2 + v'C⁻¹v/2`, accept with `min(1, e^{H₀−H₁})`; a trajectory that
//!   leaves the support (`ψ < 0`) is rejected;
//! * number of leapfrog steps uniform on `1..=L`;
//! * per-curve step size adapted during burn-in by dual averaging (Hoffman & Gelman 2014,
//!   Alg. 5) towards a target acceptance, then frozen at the averaged value.

use super::{BayesianGroupConfig, ChainState, Problem, Warp};
use crate::error::FdarError;
use crate::helpers::linear_interp;
use crate::warping::l2_norm_l2;

use rand::prelude::*;
use rand_distr::StandardNormal;

// ─── Exact likelihood gradient ───────────────────────────────────────────────

/// Forward quantities of a warp needed by the reverse pass.
struct WarpForward {
    v: Vec<f64>,
    r: f64,
    p: Vec<f64>,
    cum: Vec<f64>,
    total: f64,
    qv: Vec<f64>,
    qd: Vec<f64>,
}

impl Problem {
    /// Trapezoidal quadrature weights of the sampling grid.
    fn trapz_weights(&self) -> Vec<f64> {
        let t = &self.t;
        let n = t.len();
        (0..n)
            .map(|j| {
                let left = if j > 0 { t[j] - t[j - 1] } else { 0.0 };
                let right = if j + 1 < n { t[j + 1] - t[j] } else { 0.0 };
                0.5 * (left + right)
            })
            .collect()
    }

    fn warp_forward(&self, i: usize, coef: &[f64]) -> Option<WarpForward> {
        let t = &self.t;
        let tl = t.len();
        let v = Self::expand(&self.warp_basis, coef, tl);
        let r = l2_norm_l2(&v, t);
        if self.injective && r >= std::f64::consts::PI {
            return None;
        }
        let p: Vec<f64> = if r < 1e-12 {
            vec![1.0; tl]
        } else {
            let (c, s) = (r.cos(), r.sin() / r);
            v.iter().map(|&x| c + s * x).collect()
        };
        if p.iter().any(|&x| x < 0.0 || !x.is_finite()) {
            return None;
        }
        let sq: Vec<f64> = p.iter().map(|x| x * x).collect();
        let cum = super::cumtrapz(&sq, t);
        let total = cum[tl - 1];
        if total <= 0.0 {
            return None;
        }
        let q = &self.q[i];
        let qt = self.q_abscissae(i);
        let ql = qt.len();
        let (mut qv, mut qd) = (vec![0.0; tl], vec![0.0; tl]);
        for j in 0..tl {
            let g = cum[j] / total;
            qv[j] = linear_interp(qt, q, g);
            let k = if self.q_grid.is_none() {
                ((g * (tl - 1) as f64).floor().max(0.0) as usize).min(tl - 2)
            } else {
                qt.partition_point(|&x| x <= g)
                    .saturating_sub(1)
                    .min(ql - 2)
            };
            qd[j] = (q[k + 1] - q[k]) / (qt[k + 1] - qt[k]);
        }
        Some(WarpForward {
            v,
            r,
            p,
            cum,
            total,
            qv,
            qd,
        })
    }

    /// Exact gradient of `SSE_i(c) = Σ_j (q*_j − q_i(γ_j) ψ_j)²` with respect to the
    /// warp coefficients `c` (reverse-mode through the shooting map, the trapezoidal
    /// cumulative integral, the normalisation and the linear interpolant of `q_i`).
    /// `None` if `c` is outside the support (`ψ < 0` somewhere).
    pub(super) fn sse_gradient(
        &self,
        i: usize,
        coef: &[f64],
        template: &[f64],
    ) -> Option<Vec<f64>> {
        let f = self.warp_forward(i, coef)?;
        let t = &self.t;
        let tl = t.len();
        let z = f.total;
        let a = z.sqrt().recip();
        // Adjoints of the warped SRSF w_j = q(γ_j) p_j a.
        let rho: Vec<f64> = (0..tl)
            .map(|j| -2.0 * (template[j] - f.qv[j] * f.p[j] * a))
            .collect();
        let mut pbar: Vec<f64> = (0..tl).map(|j| rho[j] * f.qv[j] * a).collect();
        let abar: f64 = (0..tl).map(|j| rho[j] * f.qv[j] * f.p[j]).sum();
        let gbar: Vec<f64> = (0..tl).map(|j| rho[j] * f.qd[j] * f.p[j] * a).collect();
        // γ_j = G_j / Z, a = Z^{-1/2}, Z = G_{T-1}.
        let zbar =
            -abar * a / (2.0 * z) - (0..tl).map(|j| gbar[j] * f.cum[j]).sum::<f64>() / (z * z);
        let mut cbar: Vec<f64> = gbar.iter().map(|g| g / z).collect();
        cbar[tl - 1] += zbar;
        // Suffix sums S_l = Σ_{j ≥ l} Ḡ_j; G_l = G_{l−1} + Δ_l (s_{l−1} + s_l) / 2.
        let mut suffix = vec![0.0; tl + 1];
        for j in (0..tl).rev() {
            suffix[j] = suffix[j + 1] + cbar[j];
        }
        for k in 0..tl {
            let mut sbar = 0.0;
            if k + 1 < tl {
                sbar += 0.5 * (t[k + 1] - t[k]) * suffix[k + 1];
            }
            if k > 0 {
                sbar += 0.5 * (t[k] - t[k - 1]) * suffix[k];
            }
            pbar[k] += 2.0 * f.p[k] * sbar;
        }
        let vbar = shooting_adjoint(&f, &pbar, &self.trapz_weights());
        Some(
            self.warp_basis
                .iter()
                .map(|phi| phi.iter().zip(&vbar).map(|(x, y)| x * y).sum())
                .collect(),
        )
    }
}

/// Adjoint of `p = cos r + (sin r / r) v`, `r = ‖v‖` (trapezoidal L² norm).
fn shooting_adjoint(f: &WarpForward, pbar: &[f64], w: &[f64]) -> Vec<f64> {
    let r = f.r;
    if r < 1e-12 {
        return pbar.to_vec();
    }
    let s = r.sin() / r;
    let ds = (r * r.cos() - r.sin()) / (r * r);
    let sum_p: f64 = pbar.iter().sum();
    let sum_pv: f64 = pbar.iter().zip(&f.v).map(|(a, b)| a * b).sum();
    let fac = -r.sin() * sum_p + ds * sum_pv;
    pbar.iter()
        .zip(&f.v)
        .zip(w)
        .map(|((&pb, &v), &wj)| s * pb + fac * wj * v / r)
        .collect()
}

// ─── Step-size adaptation ────────────────────────────────────────────────────

/// Nesterov dual averaging (Hoffman & Gelman 2014, Alg. 5; fdasrsf `dual_avg_step`).
#[derive(Clone)]
pub(super) struct DualAveraging {
    mu: f64,
    h_bar: f64,
    log_eps_bar: f64,
    n: usize,
}

impl DualAveraging {
    const T0: f64 = 10.0;
    const GAMMA: f64 = 0.05;
    const KAPPA: f64 = 0.75;

    pub(super) fn new(eps0: f64) -> Self {
        Self {
            mu: (10.0 * eps0).ln(),
            h_bar: 0.0,
            log_eps_bar: eps0.ln(),
            n: 0,
        }
    }

    /// Update with the acceptance probability of the last transition; returns the
    /// next step size (capped at π/2).
    pub(super) fn update(&mut self, alpha: f64, target: f64) -> f64 {
        self.n += 1;
        let n = self.n as f64;
        self.h_bar = (1.0 - 1.0 / (n + Self::T0)) * self.h_bar + (target - alpha) / (n + Self::T0);
        let log_eps =
            (self.mu - n.sqrt() / Self::GAMMA * self.h_bar).min(std::f64::consts::FRAC_PI_2.ln());
        let eta = n.powf(-Self::KAPPA);
        self.log_eps_bar = eta * log_eps + (1.0 - eta) * self.log_eps_bar;
        log_eps.exp()
    }

    pub(super) fn averaged(&self) -> f64 {
        self.log_eps_bar.exp()
    }
}

// ─── HMC transition ──────────────────────────────────────────────────────────

/// Outcome of one HMC transition of one curve.
pub(super) struct HmcOutcome {
    /// Accepted new warp and its SSE.
    pub(super) accepted: Option<(Warp, f64)>,
    /// `min(1, e^{H₀−H₁})` (0 if the trajectory left the support).
    pub(super) alpha: f64,
}

/// One preconditioned HMC transition of curve `i`'s warp coefficients, with the
/// template and σ² of `state` held fixed (fdasrsf `infhmc_step`).
pub(super) fn hmc_transition(
    prob: &Problem,
    state: &ChainState,
    i: usize,
    eps: f64,
    max_leapfrog: usize,
    rng: &mut StdRng,
) -> HmcOutcome {
    let var = &prob.warp_var;
    let scale = 1.0 / (2.0 * state.sigma2);
    let c0 = &state.warps[i].coef;
    let kinetic = |c: &[f64], m: &[f64]| -> f64 {
        0.5 * c
            .iter()
            .zip(m)
            .zip(var)
            .map(|((&x, &y), &s)| (x * x + y * y) / s)
            .sum::<f64>()
    };
    let mut mom: Vec<f64> = var
        .iter()
        .map(|&s| s.sqrt() * rng.sample::<f64, _>(StandardNormal))
        .collect();
    let h0 = state.sse[i] * scale + kinetic(c0, &mom);
    let n_steps = rng.gen_range(1..=max_leapfrog);
    let mut c = c0.clone();
    let mut grad = prob.sse_gradient(i, &c, &state.template);
    let (cs, sn) = (eps.cos(), eps.sin());
    for _ in 0..n_steps {
        let Some(g) = grad.as_ref() else { break };
        kick(&mut mom, g, var, 0.5 * eps * scale);
        for (x, m) in c.iter_mut().zip(mom.iter_mut()) {
            let (cx, mx) = (*x, *m);
            *x = cx * cs + mx * sn;
            *m = mx * cs - cx * sn;
        }
        grad = prob.sse_gradient(i, &c, &state.template);
        if let Some(g) = grad.as_ref() {
            kick(&mut mom, g, var, 0.5 * eps * scale);
        }
    }
    let u: f64 = rng.gen();
    let warp = grad.and_then(|_| prob.warp_from_coef(i, c.clone()));
    let Some(w) = warp else {
        return HmcOutcome {
            accepted: None,
            alpha: 0.0,
        };
    };
    let sse_new = super::sse(&state.template, &w.warped);
    let log_ratio = h0 - (sse_new * scale + kinetic(&c, &mom));
    if !log_ratio.is_finite() {
        return HmcOutcome {
            accepted: None,
            alpha: 0.0,
        };
    }
    HmcOutcome {
        alpha: log_ratio.min(0.0).exp(),
        accepted: (u.ln() < log_ratio).then_some((w, sse_new)),
    }
}

/// `m ← m − h · C ⊙ g`.
fn kick(m: &mut [f64], g: &[f64], var: &[f64], h: f64) {
    for ((mi, &gi), &s) in m.iter_mut().zip(g).zip(var) {
        *mi -= h * s * gi;
    }
}

/// Per-chain infHMC warp sampler state (step sizes and their adaptation).
pub(super) struct HmcWarpSampler {
    pub(super) eps: Vec<f64>,
    adapt: Vec<DualAveraging>,
    max_leapfrog: usize,
    do_adapt: bool,
    target: f64,
}

impl HmcWarpSampler {
    pub(super) fn new(cfg: &BayesianGroupConfig, n: usize) -> Self {
        Self {
            eps: vec![cfg.hmc_step_size; n],
            adapt: vec![DualAveraging::new(cfg.hmc_step_size); n],
            max_leapfrog: cfg.hmc_max_leapfrog,
            do_adapt: cfg.hmc_adapt,
            target: cfg.hmc_target_accept,
        }
    }

    /// One sweep over all curves at iteration `it` (1-based, as fdasrsf's `m`).
    pub(super) fn sweep(
        &mut self,
        prob: &Problem,
        state: &mut ChainState,
        it: usize,
        burn_in: usize,
        rng: &mut StdRng,
        accepts: &mut [usize],
    ) {
        for i in 0..prob.n_curves() {
            let out = hmc_transition(prob, state, i, self.eps[i], self.max_leapfrog, rng);
            if let Some((w, s)) = out.accepted {
                state.warps[i] = w;
                state.sse[i] = s;
                accepts[i] += 1;
            }
            if self.do_adapt && it < burn_in {
                self.eps[i] = self.adapt[i].update(out.alpha, self.target);
                if it + 1 == burn_in {
                    self.eps[i] = self.adapt[i].averaged();
                }
            }
        }
    }
}

// ─── Conjugate template draw ─────────────────────────────────────────────────

impl ChainState {
    /// Exact draw of the template coefficients from their Gaussian full conditional
    /// `N(P⁻¹ Bᵀ Σ_i w_i / σ², P⁻¹)`, `P = n BᵀB / σ² + diag(1 / sd²)`.
    pub(super) fn gibbs_template(
        &mut self,
        prob: &Problem,
        rng: &mut StdRng,
    ) -> Result<(), FdarError> {
        let nb = prob.tmpl_basis.len();
        let tl = prob.t_len();
        let s2 = self.sigma2;
        let n = prob.n_curves() as f64;
        let mut prec = &prob.btb * (n / s2);
        for k in 0..nb {
            prec[(k, k)] += 1.0 / (prob.tmpl_sd[k] * prob.tmpl_sd[k]);
        }
        let wsum: Vec<f64> = (0..tl)
            .map(|j| self.warps.iter().map(|w| w.warped[j]).sum::<f64>() / s2)
            .collect();
        let rhs = nalgebra::DVector::from_fn(nb, |k, _| {
            prob.tmpl_basis[k]
                .iter()
                .zip(&wsum)
                .map(|(b, w)| b * w)
                .sum::<f64>()
        });
        let chol = prec
            .cholesky()
            .ok_or_else(|| FdarError::ComputationFailed {
                operation: "bayesian_align_group",
                detail: "template full-conditional precision is not positive definite".to_string(),
            })?;
        let mean = chol.solve(&rhs);
        let z = nalgebra::DVector::from_fn(nb, |_, _| rng.sample::<f64, _>(StandardNormal));
        let noise = chol
            .l()
            .transpose()
            .solve_upper_triangular(&z)
            .ok_or_else(|| FdarError::ComputationFailed {
                operation: "bayesian_align_group",
                detail: "triangular solve failed in template draw".to_string(),
            })?;
        self.tmpl_coef = (mean + noise).iter().copied().collect();
        self.template = prob.template_from_coef(&self.tmpl_coef);
        self.sse = self
            .warps
            .iter()
            .map(|w| super::sse(&self.template, &w.warped))
            .collect();
        Ok(())
    }
}
