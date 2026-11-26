//! Riemannian L-BFGS (RBFGS) warp optimiser over `l = √γ̇` on the unit Hilbert sphere.
//!
//! Port of fdasrsf 2.7.5 `rbfgs.rlbfgs` + `utility_functions._rbfgs_reparam`
//! (the 2026 rewrite after Huang, Gallivan, Srivastava & Absil, *Riemannian
//! optimization for registration of curves in elastic shape analysis*):
//!
//! * the iterate is `l = √γ̇` on the unit sphere of `L²([0,1])` (trapezoid
//!   metric), with `γ = cumtrapz(l²)`;
//! * the cost is `‖q₁ − (q₂∘γ)·l‖²` plus `λ·penalty(l)` plus the paper's
//!   barrier `B(l) = ∫ (l² + 1/l²)·√(1 + l⁴)` with a fading weight ω;
//! * steps follow the exponential map, stored BFGS pairs are moved by parallel
//!   translation (their Lemma 1) and the cautious update is used;
//! * the solver starts from a DP warp on a coarse grid of at most 51 points and
//!   refines it on the SRSFs resampled `scl` times finer.
//!
//! The limited-memory two-loop recursion and the Barzilai–Borwein scale are the
//! crate's own L-BFGS primitives from the soft-DTW barycenter optimiser
//! ([`two_loop_lbfgs`], [`lbfgs_hessian_scale`]). They are Euclidean, so all
//! tangent vectors are stored in the coordinates `x = √w ⊙ l` (`w` = trapezoid
//! weights), in which the trapezoid `L²` metric *is* the Euclidean dot product
//! and the sphere is the ordinary unit sphere of `ℝᵀ`.
//!
//! Deviations from fdasrsf (documented, not numeric-identical): the coarse
//! start is fdars's 35-move DP (fdasrsf uses its 63-slope `DP2`), and the SRSFs
//! are resampled to the coarse grid with a natural cubic spline (scipy's
//! default `CubicSpline` is not-a-knot).

use super::optim::{DpSeed, WarpObjectivePenalty, WarpOptimConfig};
use super::{
    coprime_nbhd, dp_alignment_core_banded, dp_alignment_core_opts, dp_alignment_core_with_penalty,
    dp_l2psi_penalty, dp_lambda_penalty, dp_prune_is_exact, DpEngineOptions, DpMoves,
};
use crate::helpers::linear_interp;
use crate::metric::soft_dtw::{lbfgs_dot, lbfgs_hessian_scale, two_loop_lbfgs};
use std::collections::VecDeque;

// ─── Natural cubic spline ────────────────────────────────────────────────────

/// Interpolating natural cubic spline (zero second derivative at both ends),
/// matching scipy's `CubicSpline(..., bc_type="natural")`.
pub(super) struct NaturalSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second derivatives at the knots.
    m2: Vec<f64>,
}

impl NaturalSpline {
    /// Build the spline through `(x[i], y[i])`; `x` strictly increasing.
    pub(super) fn new(x: &[f64], y: &[f64]) -> Self {
        let n = x.len();
        let mut m2 = vec![0.0; n];
        if n >= 3 {
            let k = n - 2;
            let mut diag = vec![0.0; k];
            let mut sup = vec![0.0; k];
            let mut rhs = vec![0.0; k];
            for r in 0..k {
                let h0 = x[r + 1] - x[r];
                let h1 = x[r + 2] - x[r + 1];
                diag[r] = 2.0 * (h0 + h1);
                sup[r] = h1;
                rhs[r] = 6.0 * ((y[r + 2] - y[r + 1]) / h1 - (y[r + 1] - y[r]) / h0);
            }
            // Thomas algorithm; the sub-diagonal of row r is h_r = x[r+1] − x[r].
            for r in 1..k {
                let sub = x[r + 1] - x[r];
                let factor = sub / diag[r - 1];
                diag[r] -= factor * sup[r - 1];
                rhs[r] -= factor * rhs[r - 1];
            }
            m2[k] = rhs[k - 1] / diag[k - 1];
            for r in (0..k - 1).rev() {
                m2[r + 1] = (rhs[r] - sup[r] * m2[r + 2]) / diag[r];
            }
        }
        Self {
            x: x.to_vec(),
            y: y.to_vec(),
            m2,
        }
    }

    /// Value and first derivative at `t` (extrapolates the end polynomials).
    #[inline]
    pub(super) fn eval(&self, t: f64) -> (f64, f64) {
        let n = self.x.len();
        let i = self
            .x
            .partition_point(|&v| v <= t)
            .saturating_sub(1)
            .min(n - 2);
        let h = self.x[i + 1] - self.x[i];
        let a = (self.x[i + 1] - t) / h;
        let b = (t - self.x[i]) / h;
        let (m0, m1) = (self.m2[i], self.m2[i + 1]);
        let val = a * self.y[i]
            + b * self.y[i + 1]
            + ((a * a * a - a) * m0 + (b * b * b - b) * m1) * h * h / 6.0;
        let der = (self.y[i + 1] - self.y[i]) / h
            + (-(3.0 * a * a - 1.0) * m0 + (3.0 * b * b - 1.0) * m1) * h / 6.0;
        (val, der)
    }
}

// ─── Grid helpers ────────────────────────────────────────────────────────────

/// Trapezoidal quadrature weights: `Σ w·y = trapezoid(y, x)`.
fn trapz_weights(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut w = vec![0.0; n];
    for k in 0..n.saturating_sub(1) {
        let dx = (x[k + 1] - x[k]) / 2.0;
        w[k] += dx;
        w[k + 1] += dx;
    }
    w
}

/// Running trapezoidal integral of `y` over `x`, starting at 0.
fn cumtrapz(y: &[f64], x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; y.len()];
    for k in 1..y.len() {
        out[k] = out[k - 1] + (x[k] - x[k - 1]) * (y[k - 1] + y[k]) / 2.0;
    }
    out
}

/// Adjoint of the running trapezoidal sum (fdasrsf `cumtrapz_adjoint`).
fn cumtrapz_adjoint(x: &[f64], b: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    // acc[k] = Σ_{j > k} b[j]
    let mut acc = 0.0;
    for k in (0..n - 1).rev() {
        acc += b[k + 1];
        let c = (x[k + 1] - x[k]) / 2.0 * acc;
        out[k + 1] += c;
        out[k] += c;
    }
    out
}

/// `numpy.gradient(y, binsize)` with `edge_order = 1` on a uniform grid.
fn np_gradient_uniform(y: &[f64], binsize: f64) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    g[0] = (y[1] - y[0]) / binsize;
    g[n - 1] = (y[n - 1] - y[n - 2]) / binsize;
    for k in 1..n - 1 {
        g[k] = (y[k + 1] - y[k - 1]) / (2.0 * binsize);
    }
    g
}

/// Adjoint of [`np_gradient_uniform`] (fdasrsf `gradient_adjoint`).
fn np_gradient_adjoint(y: &[f64], binsize: f64) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    out[1] += y[0] / binsize;
    out[0] -= y[0] / binsize;
    out[n - 1] += y[n - 1] / binsize;
    out[n - 2] -= y[n - 1] / binsize;
    for k in 1..n - 1 {
        out[k + 1] += y[k] / (2.0 * binsize);
        out[k - 1] -= y[k] / (2.0 * binsize);
    }
    out
}

/// `numpy.gradient(y, x)` (edge order 1) on a possibly non-uniform grid.
fn np_gradient(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    g[0] = (y[1] - y[0]) / (x[1] - x[0]);
    g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
    for k in 1..n - 1 {
        let hl = x[k] - x[k - 1];
        let hr = x[k + 1] - x[k];
        g[k] = (hl * hl * y[k + 1] - hr * hr * y[k - 1] + (hr * hr - hl * hl) * y[k])
            / (hl * hr * (hl + hr));
    }
    g
}

fn euclid_norm(v: &[f64]) -> f64 {
    lbfgs_dot(v, v).sqrt()
}

// ─── Problem: cost and gradient ──────────────────────────────────────────────

/// Cost parts at one iterate.
struct CostGrad {
    /// Data term + λ·penalty (the reported cost).
    fixed: f64,
    /// Barrier value `B(l)`.
    barrier: f64,
    /// Partial derivatives of `fixed` w.r.t. each sample of `l`.
    fixed_grad: Vec<f64>,
    /// Partial derivatives of `barrier` w.r.t. each sample of `l`.
    barrier_grad: Vec<f64>,
}

/// Port of fdasrsf `rlbfgs` for one SRSF pair on a grid `t ⊂ [0, 1]`.
///
/// The SRSFs may have several channels (the coordinates of an R^d SRVF); the
/// data term is then summed over channels, `Σ_c ‖q1_c − (q2_c∘γ)·l‖²`, and both
/// SRSFs are normalised by their total L² norm. One channel is fdasrsf's 1-D
/// problem.
pub(super) struct RbfgsProblem {
    t: Vec<f64>,
    w: Vec<f64>,
    sqrtw: Vec<f64>,
    /// Normalised target SRSF, one vector per channel.
    q1: Vec<Vec<f64>>,
    /// Natural splines of the normalised source SRSF, one per channel.
    q2spline: Vec<NaturalSpline>,
    lam: f64,
    penalty: WarpObjectivePenalty,
    /// Slope factor `κ` of the pointwise penalties: `L2Gam` is `∫(κγ̇ − 1)²`,
    /// `L2Psi` `∫(√(κγ̇) − 1)²` (`κ = 1` is the standard penalty; landmark
    /// segments use `κ = L_r/L_c`).
    pen_slope: f64,
}

/// Solver hyper-parameters (fdasrsf 2.7.5 values).
const TOL_GRAD_NORM: f64 = 1e-10;
const MEMORY: usize = 30;
const LS_MAX_STEPS: usize = 25;
const MIN_ITER: usize = 10;
const TOL_REL_COST: f64 = 1e-3;
const REL_COST_COUNT: usize = 5;
const OMEGA0: f64 = 1.0 / 100.0;
const OMEGA_RATE: f64 = 0.8;

impl RbfgsProblem {
    /// Set up the problem. `lam` is the user's penalty weight (fdasrsf `lam`
    /// passed to `solve`); it is rescaled by `lamScale` as in fdasrsf. One-channel
    /// form of [`Self::new_multi`] (used by the tests).
    #[cfg(test)]
    pub(super) fn new(
        q1: &[f64],
        q2: &[f64],
        t: &[f64],
        lam: f64,
        penalty: WarpObjectivePenalty,
    ) -> Self {
        Self::new_multi(&[q1], &[q2], t, lam, penalty, None)
    }

    /// Multi-channel problem: `q1[c]`, `q2[c]` are channel `c` of the SRSFs
    /// (equal channel counts, each of length `t.len()`). With one channel this
    /// is the classic 1-D problem (same arithmetic, bit-identical).
    /// `euclid = Some((e1, e2))` replaces the Euclidean norms in fdasrsf's
    /// `lamScale` (landmark segments weigh their penalty by the norms of the
    /// whole curves).
    pub(super) fn new_multi(
        q1: &[&[f64]],
        q2: &[&[f64]],
        t: &[f64],
        lam: f64,
        penalty: WarpObjectivePenalty,
        euclid: Option<(f64, f64)>,
    ) -> Self {
        let w = trapz_weights(t);
        let sqrtw: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
        let norm_l2 = |chs: &[&[f64]]| -> f64 {
            chs.iter()
                .map(|f| f.iter().zip(&w).map(|(v, wi)| wi * v * v).sum::<f64>())
                .sum::<f64>()
                .sqrt()
        };
        let euclid_of =
            |chs: &[&[f64]]| -> f64 { chs.iter().map(|f| lbfgs_dot(f, f)).sum::<f64>().sqrt() };
        let n1 = norm_l2(q1).max(1e-300);
        let n2 = norm_l2(q2).max(1e-300);
        let (e1, e2) = match euclid {
            Some((a, b)) => (a.max(1e-300), b.max(1e-300)),
            None => (euclid_of(q1).max(1e-300), euclid_of(q2).max(1e-300)),
        };
        // normL2(q/‖q‖) = normL2(q)/‖q‖
        let lam_scale = 1.0 / ((n1 / e1) * (n2 / e2));
        let q1n: Vec<Vec<f64>> = q1
            .iter()
            .map(|f| f.iter().map(|v| v / n1).collect())
            .collect();
        let q2spline: Vec<NaturalSpline> = q2
            .iter()
            .map(|f| {
                let q2n: Vec<f64> = f.iter().map(|v| v / n2).collect();
                NaturalSpline::new(t, &q2n)
            })
            .collect();
        Self {
            t: t.to_vec(),
            w,
            sqrtw,
            q1: q1n,
            q2spline,
            lam: lam * lam_scale,
            penalty,
            pen_slope: 1.0,
        }
    }

    fn l_from_x(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.sqrtw).map(|(xi, s)| xi / s).collect()
    }

    /// Discretised penalty value at `h = l` (fdasrsf `penalty_cost`).
    fn penalty_cost(&self, h: &[f64]) -> f64 {
        let w = &self.w;
        match self.penalty {
            WarpObjectivePenalty::Roughness => {
                let binsize = self.binsize();
                let h2: Vec<f64> = h.iter().map(|v| v * v).collect();
                let hd = np_gradient_uniform(&h2, binsize);
                hd.iter().zip(w).map(|(d, wi)| wi * d * d).sum()
            }
            WarpObjectivePenalty::L2Gam if self.pen_slope != 1.0 => {
                let k = self.pen_slope;
                h.iter()
                    .zip(w)
                    .map(|(v, wi)| wi * (k * v * v - 1.0).powi(2))
                    .sum()
            }
            WarpObjectivePenalty::L2Psi if self.pen_slope != 1.0 => {
                let rk = self.pen_slope.sqrt();
                h.iter()
                    .zip(w)
                    .map(|(v, wi)| wi * (rk * v - 1.0).powi(2))
                    .sum()
            }
            WarpObjectivePenalty::L2Gam => h
                .iter()
                .zip(w)
                .map(|(v, wi)| wi * (v * v - 1.0).powi(2))
                .sum(),
            WarpObjectivePenalty::L2Psi => {
                h.iter().zip(w).map(|(v, wi)| wi * (v - 1.0).powi(2)).sum()
            }
            WarpObjectivePenalty::Geodesic => {
                let c = lbfgs_dot(h, w).clamp(-1.0, 1.0);
                c.acos().powi(2)
            }
        }
    }

    /// Exact partial derivatives of [`Self::penalty_cost`].
    fn penalty_grad(&self, h: &[f64]) -> Vec<f64> {
        let w = &self.w;
        match self.penalty {
            WarpObjectivePenalty::Roughness => {
                let binsize = self.binsize();
                let h2: Vec<f64> = h.iter().map(|v| v * v).collect();
                let hd = np_gradient_uniform(&h2, binsize);
                let y: Vec<f64> = hd.iter().zip(w).map(|(d, wi)| 2.0 * wi * d).collect();
                let adj = np_gradient_adjoint(&y, binsize);
                h.iter().zip(&adj).map(|(v, a)| 2.0 * v * a).collect()
            }
            WarpObjectivePenalty::L2Gam if self.pen_slope != 1.0 => {
                let k = self.pen_slope;
                h.iter()
                    .zip(w)
                    .map(|(v, wi)| 4.0 * k * v * (k * v * v - 1.0) * wi)
                    .collect()
            }
            WarpObjectivePenalty::L2Psi if self.pen_slope != 1.0 => {
                let rk = self.pen_slope.sqrt();
                h.iter()
                    .zip(w)
                    .map(|(v, wi)| 2.0 * rk * (rk * v - 1.0) * wi)
                    .collect()
            }
            WarpObjectivePenalty::L2Gam => h
                .iter()
                .zip(w)
                .map(|(v, wi)| 4.0 * v * (v * v - 1.0) * wi)
                .collect(),
            WarpObjectivePenalty::L2Psi => h
                .iter()
                .zip(w)
                .map(|(v, wi)| 2.0 * (v - 1.0) * wi)
                .collect(),
            WarpObjectivePenalty::Geodesic => {
                let c = lbfgs_dot(h, w).clamp(-1.0, 1.0);
                let scale = if 1.0 - c * c > 1e-12 {
                    c.acos() / (1.0 - c * c).sqrt()
                } else {
                    1.0
                };
                w.iter().map(|wi| -2.0 * scale * wi).collect()
            }
        }
    }

    fn binsize(&self) -> f64 {
        (self.t[self.t.len() - 1] - self.t[0]) / (self.t.len() - 1) as f64
    }

    /// Data term (+ penalty) and barrier with exact discrete gradients
    /// (fdasrsf `costgrad`), evaluated at `l`.
    fn costgrad(&self, l: &[f64]) -> CostGrad {
        let w = &self.w;
        let t = &self.t;
        let (t0, t1) = (t[0], t[t.len() - 1]);
        let l2: Vec<f64> = l.iter().map(|v| v * v).collect();
        let gam: Vec<f64> = cumtrapz(&l2, t)
            .into_iter()
            .map(|g| g.clamp(t0, t1))
            .collect();
        let n = l.len();
        let mut fixed = 0.0;
        let mut a = vec![0.0; n];
        let mut fixed_grad = vec![0.0; n];
        for k in 0..n {
            for (c, (q1c, spl)) in self.q1.iter().zip(&self.q2spline).enumerate() {
                let (qv, qd) = spl.eval(gam[k]);
                let r = q1c[k] - qv * l[k];
                fixed += w[k] * r * r;
                let (ak, gk) = (-2.0 * w[k] * r * qd * l[k], -2.0 * w[k] * r * qv);
                // The first channel assigns (as the 1-D original), so one
                // channel is bit-identical, signed zeros included.
                if c == 0 {
                    a[k] = ak;
                    fixed_grad[k] = gk;
                } else {
                    a[k] += ak;
                    fixed_grad[k] += gk;
                }
            }
        }
        let adj = cumtrapz_adjoint(t, &a);
        for k in 0..n {
            fixed_grad[k] += 2.0 * l[k] * adj[k];
        }
        if self.lam != 0.0 {
            fixed += self.lam * self.penalty_cost(l);
            let pg = self.penalty_grad(l);
            for k in 0..n {
                fixed_grad[k] += self.lam * pg[k];
            }
        }
        let (barrier, barrier_grad) = barrier_and_grad(l, w);
        CostGrad {
            fixed,
            barrier,
            fixed_grad,
            barrier_grad,
        }
    }

    /// Riemannian gradient in `x` coordinates: partials divided by `√w`,
    /// projected onto the tangent space of the unit sphere at `x`.
    fn riemannian_grad(&self, x: &[f64], c: &CostGrad, omega: f64) -> Vec<f64> {
        let mut v: Vec<f64> = (0..x.len())
            .map(|k| (c.fixed_grad[k] + omega * c.barrier_grad[k]) / self.sqrtw[k])
            .collect();
        let ip = lbfgs_dot(&v, x);
        for (vk, xk) in v.iter_mut().zip(x) {
            *vk -= ip * xk;
        }
        v
    }

    /// `l = √γ̇` of a warp sampled on `t` (central differences, floor 1e-6),
    /// returned in `x` coordinates and normalised onto the sphere.
    fn x_of_warp(&self, gam: &[f64]) -> Vec<f64> {
        let (g0, g1) = (gam[0], gam[gam.len() - 1]);
        let gn: Vec<f64> = gam.iter().map(|g| (g - g0) / (g1 - g0)).collect();
        let d = np_gradient(&gn, &self.t);
        let x: Vec<f64> = d
            .iter()
            .zip(&self.sqrtw)
            .map(|(di, s)| di.max(1e-6).sqrt() * s)
            .collect();
        normalized(x)
    }

    /// Objective (data term + λ·penalty, no barrier) of a warp sampled on `t`.
    pub(super) fn warp_cost(&self, gam: &[f64]) -> f64 {
        self.costgrad(&self.l_from_x(&self.x_of_warp(gam))).fixed
    }

    fn eval_x(&self, x: &[f64]) -> Option<CostGrad> {
        if x.iter().any(|&v| v <= 0.0) {
            return None;
        }
        Some(self.costgrad(&self.l_from_x(x)))
    }

    /// Armijo backtracking along the exponential map (fdasrsf semantics).
    /// Returns the accepted point, its cost parts and the step length.
    fn line_search(
        &self,
        st: &SolverState,
        p: &[f64],
        df0: f64,
    ) -> Option<(Vec<f64>, CostGrad, f64)> {
        let total = |c: &CostGrad| c.fixed + st.omega * c.barrier;
        let mut alpha = 1.0;
        let mut evals = 0;
        loop {
            let xnew = sphere_exp(&st.x, p, alpha);
            let cand = self.eval_x(&xnew);
            evals += 1;
            let fnew = cand.as_ref().map_or(f64::INFINITY, total);
            let armijo = fnew.is_finite() && fnew <= st.f + 1e-4 * alpha * df0;
            if armijo || evals >= LS_MAX_STEPS {
                return match cand {
                    Some(c) if fnew.is_finite() && fnew <= st.f => Some((xnew, c, alpha)),
                    _ => None,
                };
            }
            alpha *= 0.5;
        }
    }

    /// Run the solver from `gam0` (warp on `t`; `None` = identity).
    ///
    /// Returns `(γ on t normalised to [0,1], final cost without barrier)`.
    pub(super) fn solve(&self, gam0: Option<&[f64]>, max_iter: usize) -> (Vec<f64>, f64) {
        let x0 = match gam0 {
            Some(g) => self.x_of_warp(g),
            None => normalized(self.sqrtw.clone()),
        };
        let cur = self
            .eval_x(&x0)
            .unwrap_or_else(|| self.costgrad(&self.l_from_x(&x0)));
        let mut st = SolverState::new(x0, cur, OMEGA0);
        st.refresh(self);
        while st.gnorm > TOL_GRAD_NORM && st.k < max_iter {
            if !self.iterate(&mut st) {
                break;
            }
            if st.k >= MIN_ITER && st.smallchange > REL_COST_COUNT {
                break;
            }
        }
        let l = self.l_from_x(&st.x);
        let l2: Vec<f64> = l.iter().map(|v| v * v).collect();
        let gam = cumtrapz(&l2, &self.t);
        let (gmin, gmax) = (gam[0], gam[gam.len() - 1]);
        let gam = gam.iter().map(|g| (g - gmin) / (gmax - gmin)).collect();
        (gam, st.cur.fixed)
    }

    /// One RBFGS iteration. Returns `false` when the solver must stop.
    fn iterate(&self, st: &mut SolverState) -> bool {
        let mut p = Vec::new();
        let mut qbuf = Vec::new();
        let scale = lbfgs_hessian_scale(&st.s_hist, &st.y_hist);
        two_loop_lbfgs(&st.g, &st.s_hist, &st.y_hist, scale, &mut qbuf, &mut p);
        let mut df0 = lbfgs_dot(&st.g, &p);
        if df0 >= 0.0 {
            st.clear_history();
            p = st.g.iter().map(|v| -v).collect();
            df0 = -st.gnorm * st.gnorm;
        }
        let Some((xnew, nxt, alpha)) = self.line_search(st, &p, df0) else {
            st.failed += 1;
            if st.failed >= 2 || st.s_hist.is_empty() {
                return false;
            }
            st.clear_history();
            return true;
        };
        st.failed = 0;
        let gnew = self.riemannian_grad(&xnew, &nxt, st.omega);
        let step: Vec<f64> = p.iter().map(|v| alpha * v).collect();
        let sk = transp(&st.x, &xnew, &step);
        let tg = transp(&st.x, &xnew, &st.g);
        let yk: Vec<f64> = gnew.iter().zip(&tg).map(|(a, b)| a - b).collect();
        for s in st.s_hist.iter_mut() {
            *s = transp(&st.x, &xnew, s);
        }
        for y in st.y_hist.iter_mut() {
            *y = transp(&st.x, &xnew, y);
        }
        st.cautious_push(sk, yk);
        let fnew = nxt.fixed + st.omega * nxt.barrier;
        let relchange = (st.f - fnew).abs() / st.f.abs().max(1e-300);
        st.smallchange = if relchange < TOL_REL_COST {
            st.smallchange + 1
        } else {
            0
        };
        st.x = xnew;
        st.cur = nxt;
        st.k += 1;
        st.omega *= OMEGA_RATE;
        st.refresh(self);
        true
    }
}

/// Barrier `B(l) = Σ w (l² + 1/l²)·√(1 + l⁴)` and its partials (paper eq. 15).
fn barrier_and_grad(l: &[f64], w: &[f64]) -> (f64, Vec<f64>) {
    let mut b = 0.0;
    let mut g = vec![0.0; l.len()];
    for k in 0..l.len() {
        let lk = l[k];
        let l2 = lk * lk;
        let rt = (1.0 + l2 * l2).sqrt();
        let u = l2 + 1.0 / l2;
        b += w[k] * u * rt;
        let du = 2.0 * lk - 2.0 / (lk * lk * lk);
        let drt = 2.0 * lk * lk * lk / rt;
        g[k] = w[k] * (du * rt + u * drt);
    }
    (b, g)
}

/// Mutable solver state for one RBFGS run.
struct SolverState {
    x: Vec<f64>,
    cur: CostGrad,
    g: Vec<f64>,
    gnorm: f64,
    f: f64,
    omega: f64,
    k: usize,
    failed: usize,
    smallchange: usize,
    s_hist: VecDeque<Vec<f64>>,
    y_hist: VecDeque<Vec<f64>>,
}

impl SolverState {
    fn new(x: Vec<f64>, cur: CostGrad, omega: f64) -> Self {
        Self {
            x,
            cur,
            g: Vec::new(),
            gnorm: 0.0,
            f: 0.0,
            omega,
            k: 0,
            failed: 0,
            smallchange: 0,
            s_hist: VecDeque::with_capacity(MEMORY),
            y_hist: VecDeque::with_capacity(MEMORY),
        }
    }

    /// Recompute `f`, `g` and `‖g‖` at the current iterate and ω.
    fn refresh(&mut self, prob: &RbfgsProblem) {
        self.f = self.cur.fixed + self.omega * self.cur.barrier;
        self.g = prob.riemannian_grad(&self.x, &self.cur, self.omega);
        self.gnorm = euclid_norm(&self.g);
    }

    fn clear_history(&mut self) {
        self.s_hist.clear();
        self.y_hist.clear();
    }

    /// Cautious BFGS update: keep `(s, y)` only if `⟨s,y⟩/⟨s,s⟩ ≥ 1e-4·‖g‖`.
    fn cautious_push(&mut self, sk: Vec<f64>, yk: Vec<f64>) {
        let sy = lbfgs_dot(&sk, &yk);
        let ss = lbfgs_dot(&sk, &sk);
        if ss > 0.0 && sy / ss >= 1e-4 * self.gnorm {
            if self.s_hist.len() >= MEMORY {
                self.s_hist.pop_front();
                self.y_hist.pop_front();
            }
            self.s_hist.push_back(sk);
            self.y_hist.push_back(yk);
        }
    }
}

fn normalized(mut x: Vec<f64>) -> Vec<f64> {
    let n = euclid_norm(&x);
    for v in x.iter_mut() {
        *v /= n;
    }
    x
}

/// Exponential map on the unit sphere: `exp_x(δ·v)`.
fn sphere_exp(x: &[f64], v: &[f64], delta: f64) -> Vec<f64> {
    let nrm = delta * euclid_norm(v);
    if nrm > 0.0 {
        let (s, c) = nrm.sin_cos();
        x.iter()
            .zip(v)
            .map(|(xi, vi)| xi * c + delta * vi * (s / nrm))
            .collect()
    } else {
        x.to_vec()
    }
}

/// Parallel translation of `v` from `T_x` to `T_xnew` along the geodesic
/// (Huang et al., Lemma 1).
fn transp(x: &[f64], xnew: &[f64], v: &[f64]) -> Vec<f64> {
    let s: Vec<f64> = x.iter().zip(xnew).map(|(a, b)| a + b).collect();
    let n2 = lbfgs_dot(&s, &s);
    if n2 <= 0.0 {
        return v.to_vec();
    }
    let c = 2.0 * lbfgs_dot(v, xnew) / n2;
    v.iter().zip(&s).map(|(vi, si)| vi - c * si).collect()
}

// ─── Driver: coarse DP start + fine RBFGS refinement ─────────────────────────

/// Parameters of one RBFGS warp optimisation.
#[derive(Debug, Clone, Copy)]
pub(super) struct RbfgsParams {
    /// Penalty weight (fdasrsf RBFGS convention).
    pub lambda: f64,
    pub penalty: WarpObjectivePenalty,
    pub max_iter: usize,
    pub scl: usize,
    /// Sakoe–Chiba index radius on the input grid (see
    /// [`WarpOptimConfig::band_frac`]).
    pub band: Option<usize>,
    /// DP options of the coarse-DP seed.
    pub seed: DpSeed,
    /// Penalty weight of the coarse-DP seed (fdasrsf `pen_start`: the same
    /// `lambda`, unless the DP ↔ RBFGS λ conversion is active).
    pub seed_lambda: f64,
    /// Penalty slope factor `κ` (see [`RbfgsProblem`]).
    pub pen_slope: f64,
    /// Euclidean norms replacing the problem's own in `lamScale`.
    pub lam_euclid: Option<(f64, f64)>,
}

impl RbfgsParams {
    /// Parameters for `cfg` with penalty weight `lambda` (RBFGS convention)
    /// and band radius `band`.
    pub(super) fn from_cfg(cfg: &WarpOptimConfig, lambda: f64, band: Option<usize>) -> Self {
        Self {
            lambda,
            penalty: cfg.penalty,
            max_iter: cfg.max_iter,
            scl: cfg.scl,
            band,
            seed: cfg.dp_seed(),
            seed_lambda: lambda,
            pen_slope: 1.0,
            lam_euclid: None,
        }
    }
}

/// 1-D DP on the coarse grid `tc`, honouring the DP-expressible penalties
/// (fdasrsf `pen_start`) and the seed DP options / band. The classic options
/// without a band run the original seed code (bit-identical).
fn coarse_dp_1d(
    q1c: &[f64],
    q2c: &[f64],
    tc: &[f64],
    p: &RbfgsParams,
    band: Option<usize>,
) -> Vec<f64> {
    let lam = p.seed_lambda;
    if p.seed.is_classic() && band.is_none() {
        return if lam == 0.0 {
            dp_alignment_core_banded(q1c, q2c, tc, 0.0, None)
        } else if p.penalty == WarpObjectivePenalty::L2Gam {
            dp_alignment_core_banded(q1c, q2c, tc, lam, None)
        } else {
            let prune = dp_prune_is_exact(tc, lam);
            dp_alignment_core_with_penalty(q1c, q2c, tc, None, prune, |sc, tc_, sr, tr| {
                dp_l2psi_penalty(tc, sc, tc_, sr, tr, lam)
            })
        };
    }
    let nbhd = coprime_nbhd(p.seed.nbhd_dim);
    let opts = DpEngineOptions {
        moves: DpMoves {
            nbhd: &nbhd,
            prune: p.seed.dp_prune && dp_prune_is_exact(tc, lam),
        },
        scl: p.seed.dp_scl.max(1),
        cost: p.seed.dp_cost,
    };
    let l2psi = lam > 0.0 && p.penalty != WarpObjectivePenalty::L2Gam;
    dp_alignment_core_opts(q1c, q2c, tc, band, &opts, |sc, tc_, sr, tr| {
        if l2psi {
            dp_l2psi_penalty(tc, sc, tc_, sr, tr, lam)
        } else {
            dp_lambda_penalty(tc, sc, tc_, sr, tr, lam)
        }
    })
}

/// Coarse grid of ≤ 51 uniform points of `[0, 1]` and the SRSF channels
/// resampled onto it by natural cubic splines.
fn coarse_resample(q: &[&[f64]], t1: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let nc = t1.len().min(51);
    let tc: Vec<f64> = (0..nc).map(|i| i as f64 / (nc - 1) as f64).collect();
    let qc = q
        .iter()
        .map(|f| {
            let s = NaturalSpline::new(t1, f);
            tc.iter().map(|&t| s.eval(t).0).collect()
        })
        .collect();
    (tc, qc)
}

/// Refine `t1` (sorted, in `[0,1]`) `scl` times: `scl − 1` equispaced points
/// are inserted into every interval.
fn refine_grid(t1: &[f64], scl: usize) -> Vec<f64> {
    let m = t1.len();
    let mut tf = Vec::with_capacity((m - 1) * scl + 1);
    for k in 0..m - 1 {
        let (a, b) = (t1[k], t1[k + 1]);
        for s in 0..scl {
            tf.push(a + (b - a) * s as f64 / scl as f64);
        }
    }
    tf.push(t1[m - 1]);
    tf
}

/// RBFGS optimal warp aligning SRSF `q2` to `q1` on `argvals`
/// (port of fdasrsf `_rbfgs_reparam` + the final rescaling of
/// `optimum_reparam`). Returns γ in `argvals` units, boundary-exact and
/// monotone.
pub(super) fn rbfgs_warp(q1: &[f64], q2: &[f64], argvals: &[f64], p: &RbfgsParams) -> Vec<f64> {
    rbfgs_warp_multi(&[q1], &[q2], argvals, p, |q1c, q2c, tc, band| {
        coarse_dp_1d(&q1c[0], &q2c[0], tc, p, band)
    })
}

/// Pointwise projection of a warp on `t` onto the band `|γ(t) − t| ≤ b`.
fn clamp_to_band(gam: &[f64], t: &[f64], b: f64) -> Vec<f64> {
    gam.iter()
        .zip(t)
        .map(|(&g, &s)| g.max(s - b).min(s + b))
        .collect()
}

/// Band projection of the RBFGS result `gam` (see
/// [`WarpOptimConfig::band_frac`]): the better (RBFGS objective) of its
/// pointwise projection onto the band and the projected banded-DP seed.
fn project_to_band(
    prob: &RbfgsProblem,
    gam: Vec<f64>,
    seed: Option<&[f64]>,
    t: &[f64],
    b: f64,
) -> Vec<f64> {
    let projected = clamp_to_band(&gam, t, b);
    let start = match seed {
        Some(g) => clamp_to_band(g, t, b),
        None => t.to_vec(),
    };
    if prob.warp_cost(&projected) <= prob.warp_cost(&start) {
        projected
    } else {
        start
    }
}

/// Multi-channel RBFGS warp (the data term summed over the channels of an
/// R^d SRVF). `coarse_dp(q1c, q2c, tc, band)` returns the DP starting warp on
/// the coarse grid `tc ⊂ [0, 1]` for the spline-resampled channels; with a
/// band (`p.band`) `tc` is the normalised input grid itself, the channels are
/// the input SRSFs and `band` the input-grid radius. One channel with
/// [`coarse_dp_1d`] is [`rbfgs_warp`].
pub(super) fn rbfgs_warp_multi<F>(
    q1: &[&[f64]],
    q2: &[&[f64]],
    argvals: &[f64],
    p: &RbfgsParams,
    coarse_dp: F,
) -> Vec<f64>
where
    F: FnOnce(&[Vec<f64>], &[Vec<f64>], &[f64], Option<usize>) -> Vec<f64>,
{
    let m = argvals.len();
    if m < 3 {
        return argvals.to_vec();
    }
    let (a0, a1) = (argvals[0], argvals[m - 1]);
    let span = a1 - a0;
    let t1: Vec<f64> = argvals.iter().map(|&a| (a - a0) / span).collect();
    // Band half-width in normalised time (R grid steps of the input grid).
    let band_w = p
        .band
        .filter(|&r| r < m - 1)
        .map(|r| r as f64 / (m - 1) as f64);
    // With a band the seed is the banded DP on the input grid itself (cheap,
    // O(m·R)), so the banded result is compared with it exactly; otherwise
    // fdasrsf's coarse (≤ 51 point) start.
    let (tc, q1c, q2c, cband) = if band_w.is_some() {
        let own = |q: &[&[f64]]| -> Vec<Vec<f64>> { q.iter().map(|f| f.to_vec()).collect() };
        (t1.clone(), own(q1), own(q2), p.band)
    } else {
        let (tc, q1c) = coarse_resample(q1, &t1);
        let (_, q2c) = coarse_resample(q2, &t1);
        (tc, q1c, q2c, None)
    };
    let gamc = coarse_dp(&q1c, &q2c, &tc, cband);
    let identity_start = gamc.iter().zip(&tc).all(|(g, t)| (g - t).abs() < 1e-12);

    let scl = p.scl.max(1);
    let tf = refine_grid(&t1, scl);
    let gam0: Option<Vec<f64>> = if identity_start {
        None
    } else {
        let g: Vec<f64> = tf.iter().map(|&t| linear_interp(&tc, &gamc, t)).collect();
        Some(match band_w {
            Some(b) => clamp_to_band(&g, &tf, b),
            None => g,
        })
    };
    let fine = |q: &[&[f64]]| -> Vec<Vec<f64>> {
        q.iter()
            .map(|f| tf.iter().map(|&t| linear_interp(&t1, f, t)).collect())
            .collect()
    };
    let (q1f, q2f) = (fine(q1), fine(q2));
    let q1r: Vec<&[f64]> = q1f.iter().map(Vec::as_slice).collect();
    let q2r: Vec<&[f64]> = q2f.iter().map(Vec::as_slice).collect();
    // lam is divided by scl to weigh the penalty as it would on t1 (fdasrsf).
    let lamf = p.lambda / scl as f64;
    let mut prob = RbfgsProblem::new_multi(&q1r, &q2r, &tf, lamf, p.penalty, p.lam_euclid);
    prob.pen_slope = p.pen_slope;
    let (mut gamf, cost) = prob.solve(gam0.as_deref(), p.max_iter);
    if lamf > 0.0 {
        let (gami, costi) = prob.solve(None, p.max_iter);
        if costi < cost {
            gamf = gami;
        }
    }
    if let Some(b) = band_w {
        gamf = project_to_band(&prob, gamf, gam0.as_deref(), &tf, b);
    }
    let sub: Vec<f64> = gamf.iter().step_by(scl).copied().collect();
    let (gmin, gmax) = sub
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &g| {
            (lo.min(g), hi.max(g))
        });
    let mut gamma: Vec<f64> = sub
        .iter()
        .map(|g| a0 + (g - gmin) / (gmax - gmin) * span)
        .collect();
    crate::warping::normalize_warp(&mut gamma, argvals);
    gamma
}

/// Discretised RBFGS data cost `‖q₁ − (q₂∘γ)·√γ̇‖²` (both SRSFs `L²`-normalised
/// on the refined grid) of a warp `gamma` on `argvals` — the quantity RBFGS
/// minimises, used to compare optimisers on equal terms. Test-only.
#[cfg(test)]
pub(super) fn rbfgs_data_cost(q1: &[f64], q2: &[f64], argvals: &[f64], gamma: &[f64]) -> f64 {
    rbfgs_objective(
        &[q1],
        &[q2],
        argvals,
        gamma,
        0.0,
        WarpObjectivePenalty::L2Gam,
        3,
    )
}

/// Full discretised RBFGS objective (multi-channel data term + `λ`·penalty,
/// fdasrsf `lamScale` and `1/scl` included, no barrier) of `gamma` on
/// `argvals`, on the grid refined `scl` times — exactly the functional
/// [`rbfgs_warp_multi`] minimises. Test-only.
#[cfg(test)]
pub(super) fn rbfgs_objective(
    q1: &[&[f64]],
    q2: &[&[f64]],
    argvals: &[f64],
    gamma: &[f64],
    lambda: f64,
    penalty: WarpObjectivePenalty,
    scl: usize,
) -> f64 {
    let m = argvals.len();
    let (a0, a1) = (argvals[0], argvals[m - 1]);
    let span = a1 - a0;
    let t1: Vec<f64> = argvals.iter().map(|&a| (a - a0) / span).collect();
    let g1: Vec<f64> = gamma.iter().map(|&g| (g - a0) / span).collect();
    let tf = refine_grid(&t1, scl);
    let fine = |q: &[&[f64]]| -> Vec<Vec<f64>> {
        q.iter()
            .map(|f| tf.iter().map(|&t| linear_interp(&t1, f, t)).collect())
            .collect()
    };
    let (q1f, q2f) = (fine(q1), fine(q2));
    let q1r: Vec<&[f64]> = q1f.iter().map(Vec::as_slice).collect();
    let q2r: Vec<&[f64]> = q2f.iter().map(Vec::as_slice).collect();
    let gf: Vec<f64> = tf.iter().map(|&t| linear_interp(&t1, &g1, t)).collect();
    let prob = RbfgsProblem::new_multi(&q1r, &q2r, &tf, lambda / scl as f64, penalty, None);
    prob.warp_cost(&gf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
    }

    #[test]
    fn natural_spline_interpolates_and_has_zero_end_curvature() {
        let x = vec![0.0, 0.3, 0.5, 0.9, 1.0];
        let y = vec![1.0, -0.5, 2.0, 0.3, 0.7];
        let s = NaturalSpline::new(&x, &y);
        for (xi, yi) in x.iter().zip(&y) {
            assert!((s.eval(*xi).0 - yi).abs() < 1e-12);
        }
        assert_eq!(s.m2[0], 0.0);
        assert_eq!(s.m2[4], 0.0);
        // A cubic spline reproduces straight lines exactly.
        let yl: Vec<f64> = x.iter().map(|v| 2.0 * v - 1.0).collect();
        let sl = NaturalSpline::new(&x, &yl);
        let (v, d) = sl.eval(0.42);
        assert!((v - (2.0 * 0.42 - 1.0)).abs() < 1e-12);
        assert!((d - 2.0).abs() < 1e-12);
        // Derivative matches a central finite difference.
        let h = 1e-6;
        let fd = (s.eval(0.61 + h).0 - s.eval(0.61 - h).0) / (2.0 * h);
        assert!((fd - s.eval(0.61).1).abs() < 1e-6);
    }

    #[test]
    fn natural_spline_matches_scipy() {
        // scipy 1.18.1: CubicSpline([0,.3,.5,.9,1],[1,-.5,2,.3,.7],bc_type="natural")
        // s(0.4) = 0.7701, s'(0.4) = 14.788, s(0.95) = 0.4337375, s'(0.95) = 4.44175
        let s = NaturalSpline::new(&[0.0, 0.3, 0.5, 0.9, 1.0], &[1.0, -0.5, 2.0, 0.3, 0.7]);
        let (v, d) = s.eval(0.4);
        assert!((v - 0.7701).abs() < 1e-12, "{v}");
        assert!((d - 14.788).abs() < 1e-11, "{d}");
        let (v, d) = s.eval(0.95);
        assert!((v - 0.4337375).abs() < 1e-12, "{v}");
        assert!((d - 4.44175).abs() < 1e-11, "{d}");
    }

    fn fd_check(prob: &RbfgsProblem, l: &[f64], use_barrier: bool) {
        let c = prob.costgrad(l);
        let h = 1e-6;
        for k in [0, 3, l.len() / 2, l.len() - 1] {
            let mut lp = l.to_vec();
            let mut lm = l.to_vec();
            lp[k] += h;
            lm[k] -= h;
            let (cp, cm) = (prob.costgrad(&lp), prob.costgrad(&lm));
            let (fd, an) = if use_barrier {
                ((cp.barrier - cm.barrier) / (2.0 * h), c.barrier_grad[k])
            } else {
                ((cp.fixed - cm.fixed) / (2.0 * h), c.fixed_grad[k])
            };
            assert!(
                (fd - an).abs() < 1e-5 * (1.0 + an.abs()),
                "k={k}: fd {fd} vs analytic {an}"
            );
        }
    }

    fn test_problem(lam: f64, pen: WarpObjectivePenalty) -> (RbfgsProblem, Vec<f64>) {
        let t = grid(41);
        let q1: Vec<f64> = t.iter().map(|&s| (6.0 * s).sin() + 0.5).collect();
        let q2: Vec<f64> = t.iter().map(|&s| (6.0 * s + 0.7).sin() + 0.4 * s).collect();
        let prob = RbfgsProblem::new(&q1, &q2, &t, lam, pen);
        // A non-trivial positive l (not on the sphere — the partials do not care).
        let l: Vec<f64> = t.iter().map(|&s| 1.0 + 0.3 * (4.0 * s).cos()).collect();
        (prob, l)
    }

    #[test]
    fn costgrad_matches_finite_differences_for_all_penalties() {
        for pen in [
            WarpObjectivePenalty::L2Gam,
            WarpObjectivePenalty::L2Psi,
            WarpObjectivePenalty::Roughness,
            WarpObjectivePenalty::Geodesic,
        ] {
            let (prob, l) = test_problem(0.7, pen);
            fd_check(&prob, &l, false);
        }
        let (prob, l) = test_problem(0.0, WarpObjectivePenalty::L2Gam);
        fd_check(&prob, &l, true);
        // Landmark-segment penalty slope κ ≠ 1.
        for pen in [WarpObjectivePenalty::L2Gam, WarpObjectivePenalty::L2Psi] {
            let (mut prob, l) = test_problem(0.7, pen);
            prob.pen_slope = 1.7;
            fd_check(&prob, &l, false);
            // Value: Σ w (κ l² − 1)² / Σ w (√κ l − 1)².
            let want: f64 = match pen {
                WarpObjectivePenalty::L2Gam => l
                    .iter()
                    .zip(&prob.w)
                    .map(|(v, w)| w * (1.7 * v * v - 1.0).powi(2))
                    .sum(),
                _ => l
                    .iter()
                    .zip(&prob.w)
                    .map(|(v, w)| w * (1.7f64.sqrt() * v - 1.0).powi(2))
                    .sum(),
            };
            assert!((prob.penalty_cost(&l) - want).abs() < 1e-14);
        }
    }

    #[test]
    fn identity_is_stationary_for_identical_srsfs() {
        let t = grid(31);
        let q: Vec<f64> = t.iter().map(|&s| (5.0 * s).cos() + 1.2).collect();
        let prob = RbfgsProblem::new(&q, &q, &t, 0.0, WarpObjectivePenalty::L2Gam);
        let (gam, cost) = prob.solve(None, 30);
        for (g, s) in gam.iter().zip(&t) {
            assert!((g - s).abs() < 1e-9, "{g} vs {s}");
        }
        assert!(cost < 1e-20, "cost {cost}");
    }

    #[test]
    fn transport_preserves_inner_products_and_tangency() {
        let x = normalized(vec![1.0, 2.0, 0.5, 1.5]);
        let mut v = vec![0.3, -0.2, 0.9, 0.1];
        let ip = lbfgs_dot(&v, &x);
        for (vi, xi) in v.iter_mut().zip(&x) {
            *vi -= ip * xi;
        }
        let mut u = vec![-0.4, 0.5, 0.2, 0.3];
        let ip = lbfgs_dot(&u, &x);
        for (ui, xi) in u.iter_mut().zip(&x) {
            *ui -= ip * xi;
        }
        let xnew = sphere_exp(&x, &v, 0.7);
        assert!((euclid_norm(&xnew) - 1.0).abs() < 1e-12);
        let tv = transp(&x, &xnew, &v);
        let tu = transp(&x, &xnew, &u);
        assert!(lbfgs_dot(&tv, &xnew).abs() < 1e-12);
        assert!(lbfgs_dot(&tu, &xnew).abs() < 1e-12);
        assert!((lbfgs_dot(&tv, &tu) - lbfgs_dot(&v, &u)).abs() < 1e-12);
    }
}
