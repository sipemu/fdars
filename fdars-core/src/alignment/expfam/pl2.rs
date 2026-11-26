//! Parametric two-knot piecewise-linear inverse warps (registr `warping =
//! "piecewise_linear2"`, `piecewise_linear2_hinv`, `constraints`, and the
//! normal priors of `loss_h`).
//!
//! The inverse warp on the scaled domain `[0, 1]` is the polygon through
//! `(0, 0)`, `(x₁, y₁)`, `(x₂, y₂)`, `(1, 1)` with registr's parameter order
//! `β = (x₁, y₁, x₂, y₂)` and constraints `x₁ ≥ 0.1`, `x₂ − x₁ ≥ 0.1`,
//! `x₂ ≤ 0.9` (same for `y`).

use super::basis::{dot, TemplateBasis};
use super::warp::{lbfgs_minimize, CurveFit, ObsLoss, WarpOptim};

/// registr's start value `initial_params(warping = "piecewise_linear2")`.
pub(super) const PL2_START: [f64; 4] = [0.25, 0.3, 0.75, 0.8];

/// Lower bound of a knot coordinate and of the gap between the two knots.
const GAP: f64 = 0.1;
/// Free room `0.9 − 0.1 − 0.1` shared by the two gaps.
const ROOM: f64 = 0.7;
/// Relative distance from the constraint boundary of an encoded start.
const FRAC_EPS: f64 = 1e-3;

/// `h⁻¹(t)` and its gradient with respect to `β = (x₁, y₁, x₂, y₂)` (registr
/// `piecewise_linear2_hinv`, which assumes the domain end 1).
pub(super) fn hinv(t: f64, b: &[f64]) -> (f64, [f64; 4]) {
    let (x1, y1, x2, y2) = (b[0], b[1], b[2], b[3]);
    if t < x1 {
        (t * y1 / x1, [-t * y1 / (x1 * x1), t / x1, 0.0, 0.0])
    } else if t < x2 {
        let w = x2 - x1;
        let r = (t - x1) / w;
        let dy = y2 - y1;
        (y1 + r * dy, [dy * (r - 1.0) / w, 1.0 - r, -dy * r / w, r])
    } else {
        let w = 1.0 - x2;
        let q = (t - x2) / w;
        let dy = 1.0 - y2;
        (y2 + q * dy, [0.0, 0.0, dy * (q - 1.0) / w, 1.0 - q])
    }
}

/// One coordinate pair `(c₁, c₂)` of the constraint triangle from two free
/// parameters: `c₁ = 0.1 + 0.7 p₀`, `c₂ = c₁ + 0.1 + 0.7 p₁`, `p = softmax(u, v, 0)`.
/// Returns the pair and the Jacobian `∂(c₁, c₂)/∂(u, v)`.
fn decode_pair(u: f64, v: f64) -> ([f64; 2], [[f64; 2]; 2]) {
    let m = u.max(v).max(0.0);
    let (eu, ev, e0) = ((u - m).exp(), (v - m).exp(), (-m).exp());
    let tot = eu + ev + e0;
    let (p0, p1) = (eu / tot, ev / tot);
    let c1 = GAP + ROOM * p0;
    let c2 = c1 + GAP + ROOM * p1;
    // ∂p_i/∂z_j = p_i(δ_ij − p_j)
    let dp0 = [p0 * (1.0 - p0), -p0 * p1];
    let dp1 = [-p1 * p0, p1 * (1.0 - p1)];
    let j1 = [ROOM * dp0[0], ROOM * dp0[1]];
    let j2 = [j1[0] + ROOM * dp1[0], j1[1] + ROOM * dp1[1]];
    ([c1, c2], [j1, j2])
}

fn encode_pair(c1: f64, c2: f64) -> [f64; 2] {
    let a = ((c1 - GAP) / ROOM).max(0.0);
    let b = ((c2 - c1 - GAP) / ROOM).max(0.0);
    let rest = (1.0 - a - b).max(0.0);
    let mix = |p: f64| (1.0 - 3.0 * FRAC_EPS) * p + FRAC_EPS;
    let (a, b, rest) = (mix(a), mix(b), mix(rest));
    [(a / rest).ln(), (b / rest).ln()]
}

/// `β` (registr order) and `∂β/∂θ` (4 × 4) from the free parameters
/// `θ = (u_x, v_x, u_y, v_y)`.
fn decode(theta: &[f64]) -> ([f64; 4], [[f64; 4]; 4]) {
    let (x, jx) = decode_pair(theta[0], theta[1]);
    let (y, jy) = decode_pair(theta[2], theta[3]);
    let beta = [x[0], y[0], x[1], y[1]];
    let mut jac = [[0.0; 4]; 4];
    jac[0][0] = jx[0][0];
    jac[0][1] = jx[0][1];
    jac[2][0] = jx[1][0];
    jac[2][1] = jx[1][1];
    jac[1][2] = jy[0][0];
    jac[1][3] = jy[0][1];
    jac[3][2] = jy[1][0];
    jac[3][3] = jy[1][1];
    (beta, jac)
}

fn encode(beta: &[f64]) -> Vec<f64> {
    let x = encode_pair(beta[0], beta[2]);
    let y = encode_pair(beta[1], beta[3]);
    vec![x[0], x[1], y[0], y[1]]
}

/// One curve's piecewise-linear registration problem on the scaled domain.
pub(super) struct Pl2Problem<'a> {
    /// Scaled observation times in `[0, 1]`.
    pub s: &'a [f64],
    pub y: &'a [f64],
    pub template_basis: &'a TemplateBasis,
    pub template_coefs: &'a [f64],
    pub loss: ObsLoss,
    /// Standard deviation of registr's normal priors (`priors = TRUE`), if any.
    pub prior_sd: Option<f64>,
}

impl Pl2Problem<'_> {
    /// Negative log-likelihood (plus prior), its gradient in `β` and in the
    /// gamma scale `sc`.
    pub(super) fn loss_grad(&self, beta: &[f64], sc: f64) -> (f64, [f64; 4], f64) {
        let mut total = 0.0;
        let mut gb = [0.0; 4];
        let mut gs = 0.0;
        for (&t, &y) in self.s.iter().zip(self.y) {
            let (h, dh) = hinv(t, beta);
            let g = dot(&self.template_basis.eval(h), self.template_coefs);
            let (l, dl, ds) = self.loss.eval(y, g, sc);
            let dg = dot(&self.template_basis.deriv(h), self.template_coefs);
            total += l;
            gs += ds;
            for k in 0..4 {
                gb[k] += dl * dg * dh[k];
            }
        }
        if let Some(sd) = self.prior_sd {
            const MEANS: [f64; 4] = [0.25, 0.25, 0.75, 0.75];
            let c = 0.5 * (2.0 * std::f64::consts::PI).ln() + sd.ln();
            for k in 0..4 {
                let z = (beta[k] - MEANS[k]) / sd;
                total += c + 0.5 * z * z;
                gb[k] += z / sd;
            }
        }
        (total, gb, gs)
    }
}

/// Knot-coordinate pairs of the coarse start grid: `(c₁, c₂)` on
/// `{0.1, 0.3, 0.5, 0.7, 0.9}` with `c₂ − c₁ ≥ 0.2`.
fn grid_pairs() -> Vec<[f64; 2]> {
    let lv = [0.1, 0.3, 0.5, 0.7, 0.9];
    let mut out = Vec::new();
    for (a, &c1) in lv.iter().enumerate() {
        for &c2 in &lv[a + 1..] {
            out.push([c1, c2]);
        }
    }
    out
}

/// Best point of the coarse grid of feasible knot configurations (the
/// piecewise-linear loss is multi-modal for binary data).
fn grid_start(prob: &Pl2Problem<'_>, scale: f64) -> [f64; 4] {
    let pairs = grid_pairs();
    let mut best = (PL2_START, f64::INFINITY);
    for x in &pairs {
        for y in &pairs {
            let b = [x[0], y[0], x[1], y[1]];
            let f = prob.loss_grad(&b, scale).0;
            if f < best.1 {
                best = (b, f);
            }
        }
    }
    best.0
}

/// Local minimisation from `x0`: L-BFGS, then a Nelder–Mead polish (the loss
/// has kinks in β where observations cross a knot; Nelder–Mead is registr's
/// optimiser for this warping). Returns `(x, f, converged)`.
fn local_min<F>(objective: &F, x0: Vec<f64>, opt: WarpOptim) -> (Vec<f64>, f64, bool)
where
    F: Fn(&[f64]) -> (f64, Vec<f64>),
{
    let min = lbfgs_minimize(objective, x0, opt.max_iter, opt.tol);
    let value = |x: &[f64]| objective(x).0;
    let (x, f) = nelder_mead(&value, &min.x, 0.1, 4000);
    if f < min.f {
        (x, f, min.converged)
    } else {
        (min.x, min.f, min.converged)
    }
}

/// Register one curve with a piecewise-linear inverse warp from `beta0`; with
/// `multistart` the best point of a coarse grid of knot configurations is
/// tried as a second start.
pub(super) fn register_curve_pl2(
    prob: &Pl2Problem<'_>,
    beta0: &[f64],
    opt: WarpOptim,
    multistart: bool,
) -> CurveFit {
    let gamma = prob.loss.family == super::ExpFamily::Gamma;
    let objective = |x: &[f64]| {
        let (beta, jac) = decode(&x[..4]);
        let sc = if gamma { x[4].exp() } else { 1.0 };
        let (l, gb, gs) = prob.loss_grad(&beta, sc);
        let mut gx: Vec<f64> = (0..4)
            .map(|j| (0..4).map(|k| gb[k] * jac[k][j]).sum())
            .collect();
        if gamma {
            gx.push(gs * sc);
        }
        (l, gx)
    };
    let start = |b: &[f64]| {
        let mut x0 = encode(b);
        if gamma {
            x0.push(opt.gamma_scale_init.ln());
        }
        x0
    };
    let mut best = local_min(&objective, start(beta0), opt);
    if multistart {
        let g = grid_start(prob, opt.gamma_scale_init);
        let alt = local_min(&objective, start(&g), opt);
        if alt.1 < best.1 {
            best = alt;
        }
    }
    let (x, f, converged) = best;
    let beta = decode(&x[..4]).0.to_vec();
    let t_hat = prob.s.iter().map(|&t| hinv(t, &beta).0).collect();
    CurveFit {
        beta,
        t_hat,
        loss: f,
        scale: gamma.then(|| x[4].exp()),
        converged,
    }
}

/// Nelder–Mead (standard coefficients 1, 2, ½, ½) from `x0` with initial steps
/// `step`; stops when the simplex's function values agree to `1e-13`
/// relatively or after `max_eval` evaluations.
pub(super) fn nelder_mead(
    f: &dyn Fn(&[f64]) -> f64,
    x0: &[f64],
    step: f64,
    max_eval: usize,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let mut simplex: Vec<Vec<f64>> = vec![x0.to_vec()];
    for k in 0..n {
        let mut v = x0.to_vec();
        v[k] += step;
        simplex.push(v);
    }
    let mut fv: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut evals = n + 1;
    while evals < max_eval {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
        simplex = order.iter().map(|&k| simplex[k].clone()).collect();
        fv = order.iter().map(|&k| fv[k]).collect();
        if (fv[n] - fv[0]).abs() <= 1e-13 * (fv[0].abs() + 1e-300) {
            break;
        }
        evals += nm_step(f, &mut simplex, &mut fv);
    }
    let best = (0..=n)
        .min_by(|&a, &b| fv[a].total_cmp(&fv[b]))
        .expect("non-empty simplex");
    (simplex[best].clone(), fv[best])
}

fn nm_point(c: &[f64], w: &[f64], t: f64) -> Vec<f64> {
    c.iter().zip(w).map(|(a, b)| a + t * (b - a)).collect()
}

/// One Nelder–Mead step on a simplex sorted by `fv`; returns the number of
/// function evaluations.
fn nm_step(f: &dyn Fn(&[f64]) -> f64, simplex: &mut [Vec<f64>], fv: &mut [f64]) -> usize {
    let n = simplex.len() - 1;
    let c: Vec<f64> = (0..n)
        .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
        .collect();
    let xr = nm_point(&c, &simplex[n], -1.0);
    let fr = f(&xr);
    if fr < fv[0] {
        let xe = nm_point(&c, &simplex[n], -2.0);
        let fe = f(&xe);
        (simplex[n], fv[n]) = if fe < fr { (xe, fe) } else { (xr, fr) };
        return 2;
    }
    if fr < fv[n - 1] {
        (simplex[n], fv[n]) = (xr, fr);
        return 1;
    }
    let xc = if fr < fv[n] {
        nm_point(&c, &xr, 0.5)
    } else {
        nm_point(&c, &simplex[n], 0.5)
    };
    let fc = f(&xc);
    if fc < fr.min(fv[n]) {
        (simplex[n], fv[n]) = (xc, fc);
        return 2;
    }
    for k in 1..=n {
        simplex[k] = nm_point(&simplex[0], &simplex[k], 0.5);
        fv[k] = f(&simplex[k]);
    }
    2 + n
}
