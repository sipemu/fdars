//! Elastic means of sparsely observed **closed** curves (elasdics 1.1.3
//! `fit_mean_closed`, `pre_align_closed_srv_curves`, `one_step_grad`,
//! `get_loss_pen`, `get_grad_penalty_fun`), generalised from `d = 2` to any `d`.

use super::discrete::compute_distance;
use super::mean::{
    align_all, design_row, distances_to_mean, expand_coefs, fit_coefs, model_data, nbasis,
    polygon_endpoint, SparseElasticMeanConfig, SparseElasticMeanResult, SparseMeanType,
};
use super::polygon::{add_norm_jacobian, Polygon, SrvPoly};
use super::smooth::{closure_jacobian, integrate_curve, LinSpline};
use super::to_points_matrix;
use crate::error::FdarError;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Warp tolerance of the polygon aligner inside `fit_mean_closed` (elasdics
/// calls `find_optimal_t_discrete` with its default `eps = 10^-3`).
const POLYGON_ALIGN_EPS: f64 = 1e-3;

/// Best starting segment of closed `srv2` against closed `srv1` and its distance
/// (elasdics `pre_align_closed_srv_curves`: the arc-length / given
/// parametrisation shifted to start at every observed point).
fn pre_align(srv1: &SrvPoly, srv2: &SrvPoly) -> (f64, usize) {
    if srv1.t == srv2.t && srv1.q == srv2.q {
        return (0.0, 0);
    }
    let mut best = (f64::INFINITY, 0);
    for i in 0..srv2.nseg() {
        let shifted: Vec<f64> = srv2.t.iter().map(|v| v - srv2.t[i]).collect();
        let dist = compute_distance(srv1, srv2, &shifted, true);
        if dist < best.0 {
            best = (dist, i);
        }
    }
    best
}

/// Starting segments of all curves: the pre-alignment to the curve minimising
/// the sum of squared pre-aligned distances to all others (elasdics).
fn pre_align_all(srvs: &[SrvPoly]) -> Vec<usize> {
    let n = srvs.len();
    let rows: Vec<(f64, Vec<usize>)> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let mut sum = 0.0;
            let mut starts = Vec::with_capacity(n);
            for srv in srvs {
                let (dist, s) = pre_align(&srvs[i], srv);
                sum += dist * dist;
                starts.push(s);
            }
            (sum, starts)
        })
        .collect();
    let mut best = 0;
    for (i, row) in rows.iter().enumerate() {
        if row.0 < rows[best].0 {
            best = i;
        }
    }
    rows.into_iter().nth(best).map(|r| r.1).unwrap_or_default()
}

/// Closed SRV rotated to start at segment `s0` (parametrisation shifted by
/// `t[s0]` and wrapped into `[0, 1)`).
pub(super) fn rotate_srv(srv: &SrvPoly, s0: usize) -> SrvPoly {
    let n = srv.nseg();
    let off = srv.t[s0];
    let mut t = Vec::with_capacity(n + 1);
    let mut q = Vec::with_capacity(srv.q.len());
    for k in 0..n {
        let kk = (s0 + k) % n;
        let v = srv.t[kk] - off;
        t.push(if v < 0.0 { v + 1.0 } else { v });
        q.extend_from_slice(srv.qk(kk));
    }
    t.push(1.0);
    SrvPoly { d: srv.d, t, q }
}

/// Least-squares data of the penalised loss: sparse closed design rows (at most
/// two non-zeros) and responses, rows with any non-finite value dropped
/// (elasdics `na.omit(model_data)`).
struct LsData {
    d: usize,
    rows: Vec<[(usize, f64); 2]>,
    y: Vec<f64>,
}

impl LsData {
    fn new(m: &[f64], y: &[f64], d: usize, knots: &[f64], mean_type: SparseMeanType) -> Self {
        let nb = nbasis(knots, mean_type, true);
        let mut row = vec![0.0; nb];
        let mut rows = Vec::with_capacity(m.len());
        let mut yy = Vec::with_capacity(y.len());
        for (n, &mn) in m.iter().enumerate() {
            let yn = &y[n * d..(n + 1) * d];
            if !mn.is_finite() || yn.iter().any(|v| !v.is_finite()) {
                continue;
            }
            design_row(knots, mean_type, true, mn, &mut row);
            let mut nz = [(0usize, 0.0); 2];
            let mut k = 0;
            for (a, &w) in row.iter().enumerate() {
                if w != 0.0 && k < 2 {
                    nz[k] = (a, w);
                    k += 1;
                }
            }
            rows.push(nz);
            yy.extend_from_slice(yn);
        }
        Self { d, rows, y: yy }
    }

    /// Residual `X c − y` of row `r`, coordinate `a`.
    fn residual(&self, c: &[f64], r: usize, a: usize) -> f64 {
        let d = self.d;
        self.rows[r]
            .iter()
            .map(|&(j, w)| w * c[j * d + a])
            .sum::<f64>()
            - self.y[r * d + a]
    }

    /// `‖X c − y‖²` (elasdics `L2_loss`).
    fn loss(&self, c: &[f64]) -> f64 {
        (0..self.rows.len())
            .map(|r| {
                (0..self.d)
                    .map(|a| self.residual(c, r, a).powi(2))
                    .sum::<f64>()
            })
            .sum()
    }

    /// `Xᵀ X c − Xᵀ y` (elasdics `grad_part_linear`, half the loss gradient).
    fn grad(&self, c: &[f64]) -> Vec<f64> {
        let d = self.d;
        let mut g = vec![0.0; c.len()];
        for (r, nz) in self.rows.iter().enumerate() {
            for a in 0..d {
                let res = self.residual(c, r, a);
                for &(j, w) in nz {
                    g[j * d + a] += w * res;
                }
            }
        }
        g
    }
}

/// Jacobian columns of the non-periodic basis (`d × (nc + d)`, row-major)
/// folded onto the periodic one (`d × nc`): the duplicated knot-`K` value
/// belongs to basis function 0.
fn fold_periodic(jf: &[f64], d: usize, nc: usize) -> Vec<f64> {
    let nf = nc + d;
    let mut jac = vec![0.0; d * nc];
    for a in 0..d {
        let row = &jf[a * nf..(a + 1) * nf];
        jac[a * nc..(a + 1) * nc].copy_from_slice(&row[..nc]);
        for b in 0..d {
            jac[a * nc + b] += row[nc + b];
        }
    }
    jac
}

/// The spline basis of a closed mean.
struct ClosedBasis<'a> {
    d: usize,
    knots: &'a [f64],
    mean_type: SparseMeanType,
}

impl ClosedBasis<'_> {
    fn full(&self, c: &[f64]) -> Vec<f64> {
        expand_coefs(c, self.d, self.mean_type, true)
    }

    /// Closure residual `∫_0^1 p|p|` of the periodic coefficients `c`.
    fn closure(&self, c: &[f64]) -> Vec<f64> {
        let full = self.full(c);
        match self.mean_type {
            SparseMeanType::Smooth => integrate_curve(
                &LinSpline {
                    d: self.d,
                    knots: self.knots,
                    c: &full,
                },
                &[1.0],
            ),
            SparseMeanType::Polygon => polygon_endpoint(self.d, self.knots, &full),
        }
    }

    /// Jacobian of the closure residual w.r.t. the periodic coefficients `c`;
    /// row-major `d × c.len()`.
    fn closure_jacobian(&self, c: &[f64]) -> Vec<f64> {
        let d = self.d;
        let nc = c.len();
        match self.mean_type {
            SparseMeanType::Smooth => {
                let full = self.full(c);
                let sp = LinSpline {
                    d,
                    knots: self.knots,
                    c: &full,
                };
                fold_periodic(&closure_jacobian(&sp), d, nc)
            }
            SparseMeanType::Polygon => {
                let mut jac = vec![0.0; d * nc];
                for (j, w) in self.knots.windows(2).enumerate() {
                    let cj = &c[j * d..(j + 1) * d];
                    let nrm = cj.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if nrm > 0.0 {
                        add_norm_jacobian(&mut jac, nc, j * d, w[1] - w[0], cj, nrm);
                    }
                }
                jac
            }
        }
    }

    /// Gradient `2 Jᵀ e` of the squared closure residual (elasdics
    /// differentiates it numerically with `numDeriv::grad`; analytic here).
    fn penalty_grad(&self, c: &[f64]) -> Vec<f64> {
        let e = self.closure(c);
        let jac = self.closure_jacobian(c);
        let nc = c.len();
        (0..nc)
            .map(|k| 2.0 * (0..self.d).map(|a| jac[a * nc + k] * e[a]).sum::<f64>())
            .collect()
    }

    /// One Gauss–Newton projection step towards `e(c) = 0`: the minimum-norm
    /// correction `δ = −Jᵀ(JJᵀ)⁻¹ e` with step halving until `|e|` decreases.
    /// `None` when no decrease is possible.
    fn newton_closure_step(&self, c: &[f64]) -> Option<Vec<f64>> {
        let (d, nc) = (self.d, c.len());
        let e = self.closure(c);
        let e0 = e.iter().map(|v| v * v).sum::<f64>();
        if e0 == 0.0 {
            return None;
        }
        let jac = DMatrix::from_row_slice(d, nc, &self.closure_jacobian(c));
        let y = (&jac * jac.transpose())
            .lu()
            .solve(&DVector::from_column_slice(&e))?;
        let delta = jac.transpose() * y;
        let mut step = 1.0;
        for _ in 0..40 {
            let cand: Vec<f64> = c
                .iter()
                .zip(delta.iter())
                .map(|(a, b)| a - step * b)
                .collect();
            if self.closure(&cand).iter().map(|v| v * v).sum::<f64>() < e0 {
                return Some(cand);
            }
            step *= 0.5;
        }
        None
    }
}

/// One penalised gradient step (elasdics `one_step_grad`): direction
/// `g = (XᵀXc − Xᵀy)/pen + ∇|e(c)|²` (`pen = None`: closure penalty only),
/// step `x ∈ [-1, 1]` minimising `‖X(c + xg) − y‖²/pen + |e(c + xg)|²` by
/// Brent's method (R `optimise`).
fn one_step_grad(c: &[f64], pen: Option<f64>, ls: &LsData, basis: &ClosedBasis<'_>) -> Vec<f64> {
    let mut g = basis.penalty_grad(c);
    if let Some(pen) = pen {
        for (gv, lv) in g.iter_mut().zip(ls.grad(c)) {
            *gv += lv / pen;
        }
    }
    let at = |x: f64| -> Vec<f64> { c.iter().zip(&g).map(|(a, b)| a + x * b).collect() };
    let f = |x: f64| {
        let cx = at(x);
        let p: f64 = basis.closure(&cx).iter().map(|v| v * v).sum();
        match pen {
            Some(pen) => ls.loss(&cx) / pen + p,
            None => p,
        }
    };
    let x = brent_fmin(&f, -1.0, 1.0, f64::EPSILON.powf(0.25));
    at(x)
}

/// State of Brent's minimiser (R `Brent_fmin`): bracket `[a, b]`, best point
/// `x`, second best `w`, previous `w` in `v`, last two steps `d`, `e`.
struct Brent {
    a: f64,
    b: f64,
    v: f64,
    w: f64,
    x: f64,
    fv: f64,
    fw: f64,
    fx: f64,
    d: f64,
    e: f64,
}

/// Squared inverse of the golden ratio.
const BRENT_C: f64 = 0.381_966_011_250_105_1;

impl Brent {
    /// Parabolic-interpolation numerator / denominator through `x`, `w`, `v`.
    fn parabola(&self) -> (f64, f64) {
        let r = (self.x - self.w) * (self.fx - self.fv);
        let q = (self.x - self.v) * (self.fx - self.fw);
        let p = (self.x - self.v) * q - (self.x - self.w) * r;
        let q = (q - r) * 2.0;
        if q > 0.0 {
            (-p, q)
        } else {
            (p, -q)
        }
    }

    /// Next evaluation point.
    fn propose(&mut self, xm: f64, tol1: f64, t2: f64) -> f64 {
        let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
        if self.e.abs() > tol1 {
            (p, q) = self.parabola();
            r = self.e;
            self.e = self.d;
        }
        let (a, b, x) = (self.a, self.b, self.x);
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            // golden-section step
            self.e = if x < xm { b - x } else { a - x };
            self.d = BRENT_C * self.e;
        } else {
            // parabolic-interpolation step, not too close to the bracket ends
            self.d = p / q;
            let u = x + self.d;
            if u - a < t2 || b - u < t2 {
                self.d = if x >= xm { -tol1 } else { tol1 };
            }
        }
        // never evaluate closer than tol1 to x
        if self.d.abs() >= tol1 {
            x + self.d
        } else if self.d > 0.0 {
            x + tol1
        } else {
            x - tol1
        }
    }

    /// Update the bracket and the best points with `f(u) = fu`.
    fn update(&mut self, u: f64, fu: f64) {
        if fu <= self.fx {
            if u < self.x {
                self.b = self.x;
            } else {
                self.a = self.x;
            }
            (self.v, self.fv) = (self.w, self.fw);
            (self.w, self.fw) = (self.x, self.fx);
            (self.x, self.fx) = (u, fu);
            return;
        }
        if u < self.x {
            self.a = u;
        } else {
            self.b = u;
        }
        if fu <= self.fw || self.w == self.x {
            (self.v, self.fv) = (self.w, self.fw);
            (self.w, self.fw) = (u, fu);
        } else if fu <= self.fv || self.v == self.x || self.v == self.w {
            (self.v, self.fv) = (u, fu);
        }
    }
}

/// Brent's derivative-free minimiser on `[ax, bx]` (port of R's `Brent_fmin`,
/// used by `optimise`; same iteration path).
fn brent_fmin(f: &impl Fn(f64) -> f64, ax: f64, bx: f64, tol: f64) -> f64 {
    let eps = f64::EPSILON.sqrt();
    let x = ax + BRENT_C * (bx - ax);
    let fx = f(x);
    let mut st = Brent {
        a: ax,
        b: bx,
        v: x,
        w: x,
        x,
        fv: fx,
        fw: fx,
        fx,
        d: 0.0,
        e: 0.0,
    };
    let tol3 = tol / 3.0;
    loop {
        let xm = (st.a + st.b) * 0.5;
        let tol1 = eps * st.x.abs() + tol3;
        let t2 = tol1 * 2.0;
        if (st.x - xm).abs() <= t2 - (st.b - st.a) * 0.5 {
            return st.x;
        }
        let u = st.propose(xm, tol1, t2);
        st.update(u, f(u));
    }
}

/// Final closure of the mean: elasdics' pure-penalty gradient step on
/// convergence (`closure_steps >= 1`), then up to `closure_steps − 1`
/// Gauss–Newton projection steps (stopping once `|e|` no longer decreases).
fn close_mean(
    mut coefs: Vec<f64>,
    converged: bool,
    closure_steps: usize,
    ls: &LsData,
    basis: &ClosedBasis<'_>,
) -> Vec<f64> {
    if converged && closure_steps >= 1 {
        coefs = one_step_grad(&coefs, None, ls, basis);
    }
    for _ in 1..closure_steps {
        match basis.newton_closure_step(&coefs) {
            Some(next) => coefs = next,
            None => break,
        }
    }
    coefs
}

/// Parametrisation of the original (closed) points from the rotated optimum
/// (elasdics `compute_elastic_mean` un-rotation; the closing point repeats the
/// first point's value).
fn unrotate(topt: &[f64], s0: usize) -> Vec<f64> {
    let n = topt.len() - 1;
    let mut out: Vec<f64> = (0..n).map(|j| topt[(j + n - s0) % n]).collect();
    out.push(out[0]);
    out
}

/// Relative squared coefficient change (elasdics `stop_crit`).
fn stop_crit(new: &[f64], old: &[f64]) -> f64 {
    let num: f64 = new.iter().zip(old).map(|(a, b)| (a - b).powi(2)).sum();
    num / new.iter().map(|a| a * a).sum::<f64>()
}

/// Closed elastic mean (elasdics `fit_mean_closed`) of prepared, closed curves.
pub(super) fn fit_mean_closed(
    polys: &[Polygon],
    srvs: &[SrvPoly],
    config: &SparseElasticMeanConfig,
) -> Result<SparseElasticMeanResult, FdarError> {
    let d = polys[0].d;
    let (knots, mean_type) = (&config.knots[..], config.mean_type);
    let basis = ClosedBasis {
        d,
        knots,
        mean_type,
    };
    let starts = pre_align_all(srvs);
    let rot: Vec<SrvPoly> = srvs
        .iter()
        .zip(&starts)
        .map(|(s, &s0)| rotate_srv(s, s0))
        .collect();
    let mut t_optims: Vec<Vec<f64>> = rot.iter().map(|s| s.t.clone()).collect();
    let (m, y) = model_data(&rot, &t_optims, knots, mean_type);
    let mut coefs = fit_coefs(&m, &y, d, knots, mean_type, true);
    let mut converged = false;
    let mut n_iter = 0;
    let mut last_ls = None;
    for it in 1..=config.max_iter {
        n_iter = it;
        let eps_align = match mean_type {
            SparseMeanType::Smooth => config.eps * 100.0 / it as f64,
            SparseMeanType::Polygon => POLYGON_ALIGN_EPS,
        };
        let full = basis.full(&coefs);
        t_optims = align_all(&rot, &t_optims, knots, &full, mean_type, eps_align);
        let (m, y) = model_data(&rot, &t_optims, knots, mean_type);
        let fitted = fit_coefs(&m, &y, d, knots, mean_type, true);
        let crit = stop_crit(&fitted, &coefs);
        let ls = LsData::new(&m, &y, d, knots, mean_type);
        coefs = one_step_grad(&fitted, Some(config.pen_factor * it as f64), &ls, &basis);
        last_ls = Some(ls);
        if crit < config.eps {
            converged = true;
            break;
        }
    }
    if let Some(ls) = &last_ls {
        coefs = close_mean(coefs, converged, config.closure_steps, ls, &basis);
    }
    let full = basis.full(&coefs);
    let distances = distances_to_mean(&rot, &t_optims, knots, &full, mean_type);
    let nb = coefs.len() / d;
    Ok(SparseElasticMeanResult {
        mean_type,
        knots: knots.to_vec(),
        coefs: FdMatrix::from_column_major(coefs, d, nb)?,
        curves: polys
            .iter()
            .map(|p| to_points_matrix(d, &p.pts))
            .collect::<Result<Vec<_>, _>>()?,
        params: polys.iter().map(|p| p.t.clone()).collect(),
        params_optim: t_optims
            .iter()
            .zip(&starts)
            .map(|(t, &s0)| unrotate(t, s0))
            .collect(),
        distances,
        n_iter,
        converged,
        closed: true,
        start_indices: starts,
    })
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn brent_matches_r_optimise() {
        // R 4.x: optimise(function(x) (x - 0.3)^2 + 0.1*sin(5*x), c(-1, 1))$minimum
        let f = |x: f64| (x - 0.3).powi(2) + 0.1 * (5.0 * x).sin();
        let x = brent_fmin(&f, -1.0, 1.0, f64::EPSILON.powf(0.25));
        assert!((x - R_OPTIMISE_MIN).abs() < 1e-12, "{x}");
    }

    // R 4.6.1: sprintf("%.17g", optimise(...)$minimum) -- tolerance-limited, so this
    // checks that the iteration path of R's Brent_fmin is reproduced.
    const R_OPTIMISE_MIN: f64 = 0.061_876_015_903_980_76;

    #[test]
    fn penalty_gradients_match_finite_differences() {
        let knots = [0.0, 0.2, 0.45, 0.7, 1.0];
        let c = [1.0, 0.3, -0.4, 0.9, -1.1, -0.2, 0.2, -1.3, 0.5, 0.1];
        for (mean_type, nb) in [(SparseMeanType::Smooth, 4), (SparseMeanType::Polygon, 4)] {
            let basis = ClosedBasis {
                d: 2,
                knots: &knots,
                mean_type,
            };
            let c = &c[..nb * 2];
            let g = basis.penalty_grad(c);
            let jac = basis.closure_jacobian(c);
            for k in 0..c.len() {
                let h = 1e-6;
                let mut cp = c.to_vec();
                let mut cm = c.to_vec();
                cp[k] += h;
                cm[k] -= h;
                let (ep, em) = (basis.closure(&cp), basis.closure(&cm));
                for a in 0..2 {
                    let fd = (ep[a] - em[a]) / (2.0 * h);
                    assert!(
                        (fd - jac[a * c.len() + k]).abs() < 1e-8,
                        "{mean_type:?} J[{a}][{k}]"
                    );
                }
            }
            let pen = |c: &[f64]| basis.closure(c).iter().map(|v| v * v).sum::<f64>();
            for k in 0..c.len() {
                let h = 1e-6;
                let mut cp = c.to_vec();
                let mut cm = c.to_vec();
                cp[k] += h;
                cm[k] -= h;
                let fd = (pen(&cp) - pen(&cm)) / (2.0 * h);
                assert!(
                    (fd - g[k]).abs() < 1e-7,
                    "{mean_type:?} {k}: {fd} vs {}",
                    g[k]
                );
            }
        }
    }

    #[test]
    fn ls_gradient_is_half_loss_gradient() {
        let knots = [0.0, 0.3, 0.6, 1.0];
        let m = [0.05, 0.2, 0.31, 0.5, 0.66, 0.9, 0.99];
        let y: Vec<f64> = (0..14).map(|k| ((k as f64) * 0.7).sin()).collect();
        let ls = LsData::new(&m, &y, 2, &knots, SparseMeanType::Smooth);
        let c = [0.3, -0.2, 1.0, 0.5, -0.7, 0.1];
        let g = ls.grad(&c);
        for k in 0..c.len() {
            let h = 1e-6;
            let mut cp = c.to_vec();
            let mut cm = c.to_vec();
            cp[k] += h;
            cm[k] -= h;
            let fd = (ls.loss(&cp) - ls.loss(&cm)) / (2.0 * h);
            assert!((fd - 2.0 * g[k]).abs() < 1e-6);
        }
    }

    #[test]
    fn rotation_and_unrotation_are_consistent() {
        let srv = SrvPoly {
            d: 1,
            t: vec![0.0, 0.1, 0.4, 0.8, 1.0],
            q: vec![1.0, 2.0, 3.0, 4.0],
        };
        let r = rotate_srv(&srv, 2);
        assert_eq!(r.q, vec![3.0, 4.0, 1.0, 2.0]);
        let expect = [0.0, 0.4, 0.6, 0.7, 1.0];
        for (a, b) in r.t.iter().zip(expect) {
            assert!((a - b).abs() < 1e-15);
        }
        // original point j gets the rotated parameter of position (j - 2) mod 4
        assert_eq!(
            unrotate(&[0.0, 0.4, 0.6, 0.7, 1.0], 2),
            vec![0.6, 0.7, 0.0, 0.4, 0.6]
        );
    }
}
