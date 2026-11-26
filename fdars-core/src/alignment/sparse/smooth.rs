//! Alignment of a piecewise-constant SRV to a *linear* SRV spline (elasdics
//! `find_optimal_t`, `type = "smooth"`).
//!
//! The SRV mean `p(u) = Σ_j c_j B_j(u)` uses the hat basis on `knots`. On each
//! knot interval `⟨q_k, p(u)⟩` is linear in `u`, so the warped cross products
//! `H_k = Δs_k ∫_{t_k}^{t_{k+1}} ⟨q_k, p⟩_+²` and `‖p‖²` are integrated exactly
//! (elasdics uses numerical `integrate(rel.tol = 0.1)`). The interior warp
//! points are maximised coordinate-wise (even/odd sweeps as in the discrete
//! aligner) by a bracketed 1-D search on every knot sub-interval, instead of
//! elasdics' `constrOptim` (BFGS in a log-barrier); both are local ascent from
//! the same start.

use super::polygon::{add_norm_jacobian, dot, sq, SrvPoly};

/// Linear SRV spline: value `c[j*d..(j+1)*d]` at `knots[j]`, linear in between.
pub(super) struct LinSpline<'a> {
    pub d: usize,
    pub knots: &'a [f64],
    pub c: &'a [f64],
}

/// `∫_0^len g(u)_+² du` for `g` linear from `g0` to `g1`.
#[inline]
fn lin_pos_sq(g0: f64, g1: f64, len: f64) -> f64 {
    if g0 >= 0.0 && g1 >= 0.0 {
        len * (g0 * g0 + g0 * g1 + g1 * g1) / 3.0
    } else if g0 <= 0.0 && g1 <= 0.0 {
        0.0
    } else {
        let (gp, gn) = if g0 > 0.0 { (g0, g1) } else { (g1, g0) };
        len * gp / (gp - gn) * gp * gp / 3.0
    }
}

impl LinSpline<'_> {
    fn cj(&self, j: usize) -> &[f64] {
        &self.c[j * self.d..(j + 1) * self.d]
    }

    fn nint(&self) -> usize {
        self.knots.len() - 1
    }

    /// Knot interval containing `x` (clamped).
    fn interval(&self, x: f64) -> usize {
        self.knots
            .partition_point(|&v| v <= x)
            .saturating_sub(1)
            .min(self.nint() - 1)
    }

    /// Evaluate `p(x)` into `out`.
    pub fn eval(&self, x: f64, out: &mut [f64]) {
        let j = self.interval(x);
        let (a, b) = (self.knots[j], self.knots[j + 1]);
        let w1 = (x - a) / (b - a);
        let (c0, c1) = (self.cj(j), self.cj(j + 1));
        for c in 0..self.d {
            out[c] = (1.0 - w1) * c0[c] + w1 * c1[c];
        }
    }

    /// `⟨v, p(x)⟩` on knot interval `j` (linear in `x`).
    fn inner_on(&self, v: &[f64], j: usize, x: f64) -> f64 {
        let (a, b) = (self.knots[j], self.knots[j + 1]);
        let w1 = (x - a) / (b - a);
        (1.0 - w1) * dot(v, self.cj(j)) + w1 * dot(v, self.cj(j + 1))
    }

    /// `∫_a^b ⟨v, p(u)⟩_+² du` (exact; 0 when `b <= a`).
    pub fn pos_sq_integral(&self, v: &[f64], a: f64, b: f64) -> f64 {
        if b <= a {
            return 0.0;
        }
        let last = self.nint() - 1;
        let mut total = 0.0;
        for j in self.interval(a)..=last {
            let lo = a.max(self.knots[j]);
            // the last interval also absorbs anything beyond the final knot
            let hi = if j == last {
                b
            } else {
                b.min(self.knots[j + 1])
            };
            if hi > lo {
                total += lin_pos_sq(self.inner_on(v, j, lo), self.inner_on(v, j, hi), hi - lo);
            }
            if hi >= b {
                break;
            }
        }
        total
    }

    /// `∫_0^1 |p|²` (exact for the hat basis).
    pub fn norm_sq(&self) -> f64 {
        (0..self.nint())
            .map(|j| {
                let (u, v) = (self.cj(j), self.cj(j + 1));
                (self.knots[j + 1] - self.knots[j]) * (sq(u) + dot(u, v) + sq(v)) / 3.0
            })
            .sum()
    }
}

/// Objective of one coordinate: `sqrt(H_{i-1}(x)) + sqrt(H_i(x))`.
fn coord_objective(p: &LinSpline<'_>, q: &SrvPoly, t: &[f64], i: usize, x: f64) -> f64 {
    let hl = (q.t[i] - q.t[i - 1]) * p.pos_sq_integral(q.qk(i - 1), t[i - 1], x);
    let hr = (q.t[i + 1] - q.t[i]) * p.pos_sq_integral(q.qk(i), x, t[i + 1]);
    hl.max(0.0).sqrt() + hr.max(0.0).sqrt()
}

/// Golden-section maximisation of `f` on `[a, b]`.
fn golden_max(f: &impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> (f64, f64) {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut x1 = b - INV_PHI * (b - a);
    let mut x2 = a + INV_PHI * (b - a);
    let (mut f1, mut f2) = (f(x1), f(x2));
    for _ in 0..80 {
        if (b - a) <= 1e-13 * (1.0 + a.abs().max(b.abs())) {
            break;
        }
        if f1 < f2 {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + INV_PHI * (b - a);
            f2 = f(x2);
        } else {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - INV_PHI * (b - a);
            f1 = f(x1);
        }
    }
    if f1 >= f2 {
        (x1, f1)
    } else {
        (x2, f2)
    }
}

/// Best `t_i` on `[t_{i-1}, t_{i+1}]` with the other coordinates fixed; never
/// worse than the current value.
fn optimise_one_coord_smooth(p: &LinSpline<'_>, q: &SrvPoly, t: &[f64], i: usize) -> f64 {
    const N_GRID: usize = 8;
    let (lo, hi) = (t[i - 1], t[i + 1]);
    if hi <= lo {
        return t[i];
    }
    let f = |x: f64| coord_objective(p, q, t, i, x);
    let mut best_x = t[i];
    let mut best_f = f(t[i]);
    let mut breaks = vec![lo];
    breaks.extend(p.knots.iter().copied().filter(|&k| k > lo && k < hi));
    breaks.push(hi);
    for w in breaks.windows(2) {
        let (a, b) = (w[0], w[1]);
        let h = (b - a) / N_GRID as f64;
        let mut gbest = (a, f(a));
        for g in 1..=N_GRID {
            let x = if g == N_GRID { b } else { a + g as f64 * h };
            let fx = f(x);
            if fx > gbest.1 {
                gbest = (x, fx);
            }
        }
        let (xa, xb) = ((gbest.0 - h).max(a), (gbest.0 + h).min(b));
        let (x, fx) = golden_max(&f, xa, xb);
        let cand = if fx > gbest.1 { (x, fx) } else { gbest };
        if cand.1 > best_f {
            best_f = cand.1;
            best_x = cand.0;
        }
    }
    best_x
}

/// Maximum number of coordinate sweeps (safety cap; normally reached far earlier).
const MAX_SWEEPS: usize = 10_000;

/// Optimal parametrisation of `q` against the spline `p` starting from `initial_t`.
pub(super) fn find_optimal_t_smooth(
    p: &LinSpline<'_>,
    q: &SrvPoly,
    initial_t: &[f64],
    eps: f64,
) -> Vec<f64> {
    // elasdics: move initial_t off the boundary (zero increments -> machine eps).
    let mut diffs: Vec<f64> = initial_t
        .windows(2)
        .map(|w| {
            if w[1] - w[0] == 0.0 {
                f64::EPSILON
            } else {
                w[1] - w[0]
            }
        })
        .collect();
    let total: f64 = diffs.iter().sum();
    let mut t = Vec::with_capacity(initial_t.len());
    t.push(0.0);
    let mut acc = 0.0;
    for v in &mut diffs {
        acc += *v;
        t.push(acc / total);
    }
    let len = t.len();
    if len < 3 {
        return t;
    }
    let tol = eps / len as f64;
    for _ in 0..MAX_SWEEPS {
        let t_old = t.clone();
        for i in (1..len - 1).step_by(2) {
            t[i] = optimise_one_coord_smooth(p, q, &t, i);
        }
        for i in (2..len - 1).step_by(2) {
            t[i] = optimise_one_coord_smooth(p, q, &t, i);
        }
        let delta = t_old
            .iter()
            .zip(&t)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        if delta < tol {
            break;
        }
    }
    t
}

/// Elastic distance of `q` warped by `t` to the spline `p`.
pub(super) fn smooth_distance(p: &LinSpline<'_>, q: &SrvPoly, t: &[f64]) -> f64 {
    let cross: f64 = (0..q.nseg())
        .map(|k| {
            ((q.t[k + 1] - q.t[k]) * p.pos_sq_integral(q.qk(k), t[k], t[k + 1]))
                .max(0.0)
                .sqrt()
        })
        .sum();
    let d2 = p.norm_sq() + q.norm_sq() - 2.0 * cross;
    if d2 <= 0.0 {
        0.0
    } else {
        d2.sqrt()
    }
}

/// 8-point Gauss–Legendre nodes / weights on `[-1, 1]`.
const GL_X: [f64; 8] = [
    -0.960_289_856_497_536_3,
    -0.796_666_477_413_626_7,
    -0.525_532_409_916_329,
    -0.183_434_642_495_649_8,
    0.183_434_642_495_649_8,
    0.525_532_409_916_329,
    0.796_666_477_413_626_7,
    0.960_289_856_497_536_3,
];
const GL_W: [f64; 8] = [
    0.101_228_536_290_376_26,
    0.222_381_034_453_374_47,
    0.313_706_645_877_887_3,
    0.362_683_783_378_362,
    0.362_683_783_378_362,
    0.313_706_645_877_887_3,
    0.222_381_034_453_374_47,
    0.101_228_536_290_376_26,
];

/// `β(x) = ∫_0^x p|p|` at sorted `xs`, by 8-point Gauss–Legendre on every piece
/// between consecutive evaluation points and knots. Point-major output.
pub(super) fn integrate_curve(p: &LinSpline<'_>, xs: &[f64]) -> Vec<f64> {
    let d = p.d;
    let mut out = vec![0.0; xs.len() * d];
    let mut acc = vec![0.0; d];
    let mut buf = vec![0.0; d];
    let mut pos = 0.0;
    for (n, &x) in xs.iter().enumerate() {
        let mut breaks = vec![pos];
        breaks.extend(p.knots.iter().copied().filter(|&k| k > pos && k < x));
        breaks.push(x);
        for w in breaks.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b <= a {
                continue;
            }
            let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
            for (gx, gw) in GL_X.iter().zip(&GL_W) {
                p.eval(mid + half * gx, &mut buf);
                let nrm = sq(&buf).sqrt();
                for c in 0..d {
                    acc[c] += half * gw * buf[c] * nrm;
                }
            }
        }
        pos = pos.max(x);
        out[n * d..(n + 1) * d].copy_from_slice(&acc);
    }
    out
}

/// Jacobian of the closure residual `e = ∫_0^1 p|p|` with respect to the
/// point-major coefficients of `p` (same quadrature as [`integrate_curve`], so
/// it is the exact Jacobian of the discretised residual); row-major `d × ncoef`.
pub(super) fn closure_jacobian(p: &LinSpline<'_>) -> Vec<f64> {
    let d = p.d;
    let nc = p.c.len();
    let mut jac = vec![0.0; d * nc];
    let mut buf = vec![0.0; d];
    for j in 0..p.nint() {
        let (a, b) = (p.knots[j], p.knots[j + 1]);
        let (mid, half) = (0.5 * (a + b), 0.5 * (b - a));
        for (gx, gw) in GL_X.iter().zip(&GL_W) {
            let u = mid + half * gx;
            p.eval(u, &mut buf);
            let nrm = sq(&buf).sqrt();
            if nrm == 0.0 {
                continue;
            }
            // d(p|p|)/dp = |p| I + p pᵀ/|p|
            let w1 = (u - a) / (b - a);
            for (jj, phi) in [(j, 1.0 - w1), (j + 1, w1)] {
                add_norm_jacobian(&mut jac, nc, jj * d, half * gw * phi, &buf, nrm);
            }
        }
    }
    jac
}
