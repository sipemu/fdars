//! Native replacement for the `gamm4` mixed model of registr's two-step GFPCA:
//!
//! ```text
//! g(E[yᵢⱼ]) = f(tᵢⱼ) + Σₖ φₖ(tᵢⱼ) sᵢₖ,   sᵢₖ ~ N(0, σₖ²) independent,
//! ```
//!
//! with `f` a P-spline (`s(index, bs = "ps", k = Kt)`) represented, as
//! `gamm4` / mgcv `smooth2random` do, by fixed intercept and linear effects plus
//! i.i.d. random effects for the penalised part. Gaussian data are fitted by
//! REML (lme4 `lmer` profiled REML criterion), other families by the Laplace
//! approximation (lme4 `glmer`, `nAGQ = 1`: variance parameters first with the
//! fixed effects in the penalised IRLS, then jointly with the fixed effects).

use std::cell::RefCell;

use nalgebra::{DMatrix, DVector};

use crate::distributions::ln_gamma;
use crate::error::FdarError;

use super::basis::{glm_deviance, glm_mean, glm_variance};
use super::mgcv::{diff_penalty, fd_gradient, sym_eigen_desc, PsBasis};
use super::warp::lbfgs_minimize;
use super::ExpFamily;

/// Number of fixed effects (intercept and the linear null-space direction of
/// the second-order difference penalty).
const NF: usize = 2;

/// Rows of the mixed model, grouped by curve.
pub(super) struct GlmmData {
    pub y: Vec<f64>,
    pub t: Vec<f64>,
    /// Eigenfunction values `φₖ(t)` per row (`npc` each).
    pub psi: Vec<Vec<f64>>,
    /// Row range of every curve.
    pub curves: Vec<std::ops::Range<usize>>,
    pub family: ExpFamily,
}

/// Mixed-model design: fixed columns and the random columns of the smooth.
struct Design {
    basis: PsBasis,
    /// Maps spline coefficients to `[fixed (2) | random smooth (k − 2)]`
    /// columns: `X_f = B·tf`, `Z_s = B·tr`.
    tf: DMatrix<f64>,
    tr: DMatrix<f64>,
    xf: Vec<[f64; NF]>,
    zs: Vec<Vec<f64>>,
}

/// Orthonormal basis of the orthogonal complement of `c` (Householder).
fn complement_basis(c: &DVector<f64>) -> DMatrix<f64> {
    let k = c.len();
    let norm = c.norm();
    let mut v = c.clone();
    v[0] += if c[0] >= 0.0 { norm } else { -norm };
    let vv = v.dot(&v);
    let h = DMatrix::<f64>::identity(k, k) - (&v * v.transpose()) * (2.0 / vv);
    h.columns(1, k - 1).into_owned()
}

impl Design {
    fn new(t: &[f64], k: usize) -> Self {
        let (lo, hi) = t
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
                (a.min(v), b.max(v))
            });
        let basis = PsBasis::new(lo, hi, k);
        let rows: Vec<Vec<f64>> = t.iter().map(|&x| basis.eval(x)).collect();
        // mgcv: sum-to-zero constraint absorbed (Z ⟂ colSums(B)), then
        // smooth2random splits ZᵀSZ into its null space (fixed) and the
        // penalised range with unit-variance random effects.
        let csum = DVector::from_fn(k, |c, _| rows.iter().map(|r| r[c]).sum());
        let z = complement_basis(&csum);
        let sz = z.transpose() * diff_penalty(k) * &z;
        let (d, u) = sym_eigen_desc(&sz);
        let mut tf = DMatrix::<f64>::zeros(k, NF);
        tf.column_mut(0).fill(1.0);
        tf.set_column(1, &(&z * u.column(k - 2)));
        let tr = DMatrix::from_fn(k, k - 2, |r, c| {
            (0..k - 1).map(|j| z[(r, j)] * u[(j, c)]).sum::<f64>() / d[c].sqrt()
        });
        let proj = |m: &DMatrix<f64>, row: &[f64]| -> Vec<f64> {
            (0..m.ncols())
                .map(|c| (0..k).map(|j| row[j] * m[(j, c)]).sum())
                .collect()
        };
        let xf = rows
            .iter()
            .map(|r| {
                let v = proj(&tf, r);
                [v[0], v[1]]
            })
            .collect();
        let zs = rows.iter().map(|r| proj(&tr, r)).collect();
        Self {
            basis,
            tf,
            tr,
            xf,
            zs,
        }
    }
}

/// Unknowns of the penalised IRLS.
#[derive(Clone)]
struct Modes {
    beta: [f64; NF],
    us: Vec<f64>,
    ui: Vec<Vec<f64>>,
}

/// Penalised IRLS outcome.
struct Pirls {
    modes: Modes,
    /// `Σ devres + ‖u‖²` (gaussian: penalised RSS).
    pdev: f64,
    /// `log|L|²` (random-effect block of the system).
    ld_random: f64,
    /// `log|L|² + log|R_X|²` (whole system; meaningful when `β` is free).
    ld_full: f64,
    mu: Vec<f64>,
}

/// Variance parameters: per-FPC standard deviations and that of the smooth.
#[derive(Clone, Copy)]
struct Theta<'a> {
    psi: &'a [f64],
    smooth: f64,
}

struct Model<'a> {
    data: &'a GlmmData,
    design: Design,
    npc: usize,
}

/// One block-structured penalised WLS solve.
struct Solved {
    xg: DVector<f64>,
    xi: Vec<DVector<f64>>,
    ld_a: f64,
    ld_s_full: f64,
    ld_s_random: f64,
}

fn logdet_chol(m: &DMatrix<f64>) -> Option<f64> {
    let ch = m.clone().cholesky()?;
    Some(2.0 * ch.l().diagonal().iter().map(|v| v.ln()).sum::<f64>())
}

impl Model<'_> {
    fn eta(&self, r: usize, m: &Modes, th: Theta<'_>, i: usize) -> f64 {
        let d = &self.design;
        let mut e = d.xf[r][0] * m.beta[0] + d.xf[r][1] * m.beta[1];
        e += th.smooth * d.zs[r].iter().zip(&m.us).map(|(a, b)| a * b).sum::<f64>();
        for k in 0..self.npc {
            e += th.psi[k] * self.data.psi[r][k] * m.ui[i][k];
        }
        e
    }

    /// Global design row `[x_f (if β free) | θ_s z_s]` into `out`.
    fn global_row(&self, r: usize, th: Theta<'_>, beta_free: bool, out: &mut Vec<f64>) {
        let d = &self.design;
        out.clear();
        if beta_free {
            out.extend_from_slice(&d.xf[r]);
        }
        out.extend(d.zs[r].iter().map(|v| v * th.smooth));
    }

    /// Accumulate one curve's blocks `(A_i, C_i, b_i)` and add its rows to the
    /// global `G` (upper triangle, row-major `pg × pg`) and `b_g`.
    fn accumulate_curve(
        &self,
        range: std::ops::Range<usize>,
        th: Theta<'_>,
        (w, z): (&[f64], &[f64]),
        beta_free: bool,
        (g, bg): (&mut [f64], &mut [f64]),
        bufs: &mut (Vec<f64>, Vec<f64>),
    ) -> (DMatrix<f64>, DMatrix<f64>, DVector<f64>) {
        let npc = self.npc;
        let pg = bg.len();
        let mut a = vec![0.0; npc * npc];
        let mut c = vec![0.0; pg * npc];
        let mut bi = vec![0.0; npc];
        for r in range {
            self.global_row(r, th, beta_free, &mut bufs.0);
            bufs.1.clear();
            bufs.1
                .extend(th.psi.iter().zip(&self.data.psi[r]).map(|(t, p)| t * p));
            let (gr, lr) = (&bufs.0, &bufs.1);
            let (wr, zr) = (w[r], z[r]);
            for p in 0..pg {
                let gp = wr * gr[p];
                bg[p] += gp * zr;
                let grow = &mut g[p * pg..(p + 1) * pg];
                for q in p..pg {
                    grow[q] += gp * gr[q];
                }
                let crow = &mut c[p * npc..(p + 1) * npc];
                for (ck, lk) in crow.iter_mut().zip(lr) {
                    *ck += gp * lk;
                }
            }
            for k in 0..npc {
                let lk = wr * lr[k];
                bi[k] += lk * zr;
                for l in k..npc {
                    a[k * npc + l] += lk * lr[l];
                }
            }
        }
        let a = DMatrix::from_fn(npc, npc, |k, l| {
            let v = if k <= l {
                a[k * npc + l]
            } else {
                a[l * npc + k]
            };
            v + if k == l { 1.0 } else { 0.0 }
        });
        let c = DMatrix::from_row_slice(pg, npc, &c);
        (a, c, DVector::from_vec(bi))
    }

    /// Minimise `Σ w (z − D x)² + ‖u‖²` over `x = (β?, u_s, u_i)`.
    fn solve(&self, th: Theta<'_>, w: &[f64], z: &[f64], beta_free: bool) -> Option<Solved> {
        let nf = if beta_free { NF } else { 0 };
        let pg = nf + self.design.zs[0].len();
        let mut gbuf = vec![0.0; pg * pg];
        for k in nf..pg {
            gbuf[k * pg + k] = 1.0;
        }
        let mut bbuf = vec![0.0; pg];
        let mut s_corr = DMatrix::<f64>::zeros(pg, pg);
        let mut b_corr = DVector::<f64>::zeros(pg);
        let mut ld_a = 0.0;
        let mut blocks = Vec::with_capacity(self.data.curves.len());
        let mut bufs = (Vec::new(), Vec::new());
        for range in &self.data.curves {
            let (a, c, bi) = self.accumulate_curve(
                range.clone(),
                th,
                (w, z),
                beta_free,
                (&mut gbuf, &mut bbuf),
                &mut bufs,
            );
            let ch = a.cholesky()?;
            ld_a += 2.0 * ch.l().diagonal().iter().map(|v| v.ln()).sum::<f64>();
            let ainv_ct = ch.solve(&c.transpose());
            let ainv_b = ch.solve(&bi);
            s_corr += &c * &ainv_ct;
            b_corr += &c * &ainv_b;
            blocks.push((ch, c, bi));
        }
        let g = DMatrix::from_fn(pg, pg, |p, q| gbuf[p.min(q) * pg + p.max(q)]);
        let bg = DVector::from_vec(bbuf);
        let s = &g - s_corr;
        let xg = s.clone().cholesky()?.solve(&(bg - b_corr));
        let xi = blocks
            .iter()
            .map(|(ch, c, bi)| ch.solve(&(bi - c.transpose() * &xg)))
            .collect();
        let ld_s_full = logdet_chol(&s)?;
        let ld_s_random = if beta_free {
            logdet_chol(&s.view((nf, nf), (pg - nf, pg - nf)).into_owned())?
        } else {
            ld_s_full
        };
        Some(Solved {
            xg,
            xi,
            ld_a,
            ld_s_full,
            ld_s_random,
        })
    }

    fn penalised_dev(&self, m: &Modes, th: Theta<'_>) -> (f64, Vec<f64>) {
        let fam = self.data.family;
        let mut dev = m.us.iter().map(|v| v * v).sum::<f64>()
            + m.ui.iter().flatten().map(|v| v * v).sum::<f64>();
        let mut mu = vec![0.0; self.data.y.len()];
        for (i, range) in self.data.curves.iter().enumerate() {
            for r in range.clone() {
                mu[r] = glm_mean(fam, self.eta(r, m, th, i)).0;
                dev += glm_deviance(fam, self.data.y[r], mu[r]);
            }
        }
        (dev, mu)
    }

    fn unpack(&self, sol: &Solved, beta_free: bool, beta: [f64; NF]) -> Modes {
        let nf = if beta_free { NF } else { 0 };
        let beta = if beta_free {
            [sol.xg[0], sol.xg[1]]
        } else {
            beta
        };
        Modes {
            beta,
            us: sol.xg.iter().skip(nf).copied().collect(),
            ui: sol.xi.iter().map(|v| v.iter().copied().collect()).collect(),
        }
    }

    /// Penalised IRLS from `start` (β held fixed unless `beta_free`), with
    /// step halving on the penalised deviance.
    fn pirls(&self, th: Theta<'_>, start: &Modes, beta_free: bool) -> Option<Pirls> {
        let fam = self.data.family;
        let mut cur = start.clone();
        let (mut pdev, _) = self.penalised_dev(&cur, th);
        let max_iter = if fam == ExpFamily::Gaussian { 1 } else { 100 };
        let mut last = None;
        for _ in 0..max_iter {
            let (w, z) = self.working(&cur, th, beta_free);
            let sol = self.solve(th, &w, &z, beta_free)?;
            let target = self.unpack(&sol, beta_free, cur.beta);
            let (next, pnext) = self.halve_steps(&cur, target, pdev, th);
            let change = mode_change(&cur, &next);
            cur = next;
            pdev = pnext;
            last = Some((sol.ld_a + sol.ld_s_random, sol.ld_a + sol.ld_s_full));
            if change < 1e-10 {
                break;
            }
        }
        // Log-determinants at the converged weights.
        let (ld_random, ld_full) = if fam == ExpFamily::Gaussian {
            last?
        } else {
            self.logdets_at(&cur, th, beta_free)?
        };
        let (pdev, mu) = self.penalised_dev(&cur, th);
        Some(Pirls {
            modes: cur,
            pdev,
            ld_random,
            ld_full,
            mu,
        })
    }

    /// Newton working weights and responses (minus the fixed-effect offset when
    /// `β` is held fixed) at the modes `cur`.
    fn working(&self, cur: &Modes, th: Theta<'_>, beta_free: bool) -> (Vec<f64>, Vec<f64>) {
        let fam = self.data.family;
        let n = self.data.y.len();
        let mut w = vec![0.0; n];
        let mut z = vec![0.0; n];
        for (i, range) in self.data.curves.iter().enumerate() {
            for r in range.clone() {
                let (wr, zr) = newton_working(fam, self.data.y[r], self.eta(r, cur, th, i));
                w[r] = wr;
                let xf = self.design.xf[r];
                let off = if beta_free {
                    0.0
                } else {
                    xf[0] * cur.beta[0] + xf[1] * cur.beta[1]
                };
                z[r] = zr - off;
            }
        }
        (w, z)
    }

    /// Step halving from `cur` towards `target` until the penalised deviance
    /// does not increase (none for gaussian data).
    fn halve_steps(&self, cur: &Modes, target: Modes, pdev: f64, th: Theta<'_>) -> (Modes, f64) {
        let mut pnext = self.penalised_dev(&target, th).0;
        if self.data.family == ExpFamily::Gaussian {
            return (target, pnext);
        }
        let (mut step, mut next) = (1.0, target.clone());
        while (pnext.is_nan() || pnext > pdev * (1.0 + 1e-12)) && step > 1e-6 {
            step *= 0.5;
            next = blend(cur, &target, step);
            pnext = self.penalised_dev(&next, th).0;
        }
        (next, pnext)
    }

    fn logdets_at(&self, m: &Modes, th: Theta<'_>, beta_free: bool) -> Option<(f64, f64)> {
        let fam = self.data.family;
        let n = self.data.y.len();
        let mut w = vec![0.0; n];
        for (i, range) in self.data.curves.iter().enumerate() {
            for r in range.clone() {
                let (mu, dmu) = glm_mean(fam, self.eta(r, m, th, i));
                w[r] = dmu * dmu / glm_variance(fam, mu).max(1e-300);
            }
        }
        let sol = self.solve(th, &w, &vec![0.0; n], beta_free)?;
        Some((sol.ld_a + sol.ld_s_random, sol.ld_a + sol.ld_s_full))
    }

    /// `−2 log L(y; μ)` as lme4's `aic()` (gamma: Tweedie p = 2 with the
    /// dispersion `deviance / n` plugged in, `+ 2`).
    fn minus2_loglik(&self, mu: &[f64]) -> f64 {
        let y = &self.data.y;
        match self.data.family {
            ExpFamily::Gaussian | ExpFamily::Binomial => y
                .iter()
                .zip(mu)
                .map(|(&a, &b)| glm_deviance(self.data.family, a, b))
                .sum(),
            ExpFamily::Poisson => {
                -2.0 * y
                    .iter()
                    .zip(mu)
                    .map(|(&a, &m)| a * m.ln() - m - ln_gamma(a + 1.0))
                    .sum::<f64>()
            }
            ExpFamily::Gamma => {
                let dev: f64 = y
                    .iter()
                    .zip(mu)
                    .map(|(&a, &b)| glm_deviance(ExpFamily::Gamma, a, b))
                    .sum();
                let phi = dev / y.len() as f64;
                let shape = 1.0 / phi;
                let ll: f64 = y
                    .iter()
                    .zip(mu)
                    .map(|(&a, &m)| {
                        let scale = m * phi;
                        (shape - 1.0) * a.ln() - a / scale - ln_gamma(shape) - shape * scale.ln()
                    })
                    .sum();
                -2.0 * ll + 2.0
            }
        }
    }

    /// lme4 profiled REML criterion (gaussian).
    fn reml(&self, p: &Pirls) -> f64 {
        let df = (self.data.y.len() - NF) as f64;
        p.ld_full + df * (1.0 + (2.0 * std::f64::consts::PI * p.pdev / df).ln())
    }

    /// Laplace deviance (non-gaussian).
    fn laplace(&self, p: &Pirls) -> f64 {
        let u2 = p.modes.us.iter().map(|v| v * v).sum::<f64>()
            + p.modes.ui.iter().flatten().map(|v| v * v).sum::<f64>();
        self.minus2_loglik(&p.mu) + u2 + p.ld_random
    }

    fn zero_modes(&self, beta: [f64; NF]) -> Modes {
        Modes {
            beta,
            us: vec![0.0; self.design.zs[0].len()],
            ui: vec![vec![0.0; self.npc]; self.data.curves.len()],
        }
    }
}

/// Working weight and response of a Newton step on the deviance: Fisher
/// scoring for the canonical links, observed information for the gamma log
/// link (quadratic convergence; the mode is the same).
fn newton_working(fam: ExpFamily, y: f64, eta: f64) -> (f64, f64) {
    let (mu, dmu) = glm_mean(fam, eta);
    if fam == ExpFamily::Gamma {
        (y / mu, eta + (y - mu) / y)
    } else {
        (
            dmu * dmu / glm_variance(fam, mu).max(1e-300),
            eta + (y - mu) / dmu,
        )
    }
}

/// Largest change between two mode vectors, relative to `1 + max |x|`.
fn mode_change(a: &Modes, b: &Modes) -> f64 {
    let pairs = a
        .beta
        .iter()
        .zip(&b.beta)
        .chain(a.us.iter().zip(&b.us))
        .chain(a.ui.iter().flatten().zip(b.ui.iter().flatten()));
    let (mut d, mut m) = (0.0_f64, 0.0_f64);
    for (x, y) in pairs {
        d = d.max((x - y).abs());
        m = m.max(y.abs());
    }
    d / (1.0 + m)
}

fn blend(a: &Modes, b: &Modes, s: f64) -> Modes {
    let mix = |x: f64, y: f64| x + s * (y - x);
    Modes {
        beta: [mix(a.beta[0], b.beta[0]), mix(a.beta[1], b.beta[1])],
        us: a.us.iter().zip(&b.us).map(|(&x, &y)| mix(x, y)).collect(),
        ui: a
            .ui
            .iter()
            .zip(&b.ui)
            .map(|(u, v)| u.iter().zip(v).map(|(&x, &y)| mix(x, y)).collect())
            .collect(),
    }
}

/// Fitted mixed model.
pub(super) struct GlmmFit {
    /// Conditional modes of the FPC scores (`n × npc`).
    pub scores: Vec<Vec<f64>>,
    /// Spline coefficients of the mean `f` in the P-spline basis.
    pub mean_coef: Vec<f64>,
    pub mean_basis: PsBasis,
    /// `[θ_1, …, θ_npc, θ_smooth]` (gaussian: relative to `σ`).
    pub theta: Vec<f64>,
    /// Residual variance (gaussian only).
    pub sigma2: Option<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

impl GlmmFit {
    /// Mean `f(t)` (link scale).
    pub(super) fn mean_at(&self, t: f64) -> f64 {
        self.mean_basis
            .eval(t)
            .iter()
            .zip(&self.mean_coef)
            .map(|(a, b)| a * b)
            .sum()
    }
}

fn split_theta(x: &[f64], npc: usize) -> (Vec<f64>, f64) {
    (x[..npc].iter().map(|v| v.abs()).collect(), x[npc].abs())
}

fn glmm_failed() -> FdarError {
    FdarError::ComputationFailed {
        operation: "gfpca_two_step mixed model",
        detail: "penalised IRLS failed (singular system)".into(),
    }
}

/// Fit the two-step GFPCA mixed model with a P-spline mean of `k` basis
/// functions, from the variance parameters `start` (`npc + 1`, default all 1
/// as lme4).
pub(super) fn fit_glmm(
    data: &GlmmData,
    k: usize,
    npc: usize,
    start: Option<&[f64]>,
) -> Result<GlmmFit, FdarError> {
    let model = Model {
        data,
        design: Design::new(&data.t, k),
        npc,
    };
    let gaussian = data.family == ExpFamily::Gaussian;
    let x0: Vec<f64> = start.map_or_else(|| vec![1.0; npc + 1], <[f64]>::to_vec);
    // Warm start of the penalised IRLS from the previous evaluation's modes.
    let cache = RefCell::new(model.zero_modes([0.0; NF]));
    // Stage 1 (gaussian: the REML fit; otherwise nAGQ = 0).
    let stage1 = |x: &[f64]| model.stage_value(x, &cache, true);
    let obj1 = |x: &[f64]| {
        let f = stage1(x);
        (f, fd_gradient(&stage1, x, f))
    };
    let m1 = lbfgs_minimize(obj1, x0, 500, 1e-12);
    let fit1 = model.stage_fit(&m1.x, &cache, true)?;
    if gaussian {
        let sigma2 = fit1.pdev / (data.y.len() - NF) as f64;
        return Ok(assemble(
            &model,
            &m1.x,
            &fit1.modes,
            Some(sigma2),
            m1.iters,
            m1.converged,
        ));
    }
    // Stage 2: variance parameters and fixed effects jointly, modes of u only.
    let stage2 = |x: &[f64]| model.stage_value(x, &cache, false);
    let obj2 = |x: &[f64]| {
        let f = stage2(x);
        (f, fd_gradient(&stage2, x, f))
    };
    let mut x2 = m1.x.clone();
    x2.extend_from_slice(&fit1.modes.beta);
    let m2 = lbfgs_minimize(obj2, x2, 500, 1e-12);
    let fit2 = model.stage_fit(&m2.x, &cache, false)?;
    Ok(assemble(
        &model,
        &m2.x[..=npc],
        &fit2.modes,
        None,
        m1.iters + m2.iters,
        m2.converged,
    ))
}

impl Model<'_> {
    /// Penalised IRLS at the parameters `x` (`θ`, then `β` when held fixed),
    /// warm-started from and updating `cache`.
    fn stage_fit(
        &self,
        x: &[f64],
        cache: &RefCell<Modes>,
        beta_free: bool,
    ) -> Result<Pirls, FdarError> {
        let npc = self.npc;
        let (psi, smooth) = split_theta(x, npc);
        let th = Theta { psi: &psi, smooth };
        let mut start = cache.borrow().clone();
        if !beta_free {
            start.beta = [x[npc + 1], x[npc + 2]];
        }
        let fit = self.pirls(th, &start, beta_free).ok_or_else(glmm_failed)?;
        *cache.borrow_mut() = fit.modes.clone();
        Ok(fit)
    }

    /// Objective of stage 1 (`beta_free`: REML or Laplace with `β` in the IRLS)
    /// or stage 2 (Laplace at fixed `β`).
    fn stage_value(&self, x: &[f64], cache: &RefCell<Modes>, beta_free: bool) -> f64 {
        match self.stage_fit(x, cache, beta_free) {
            Ok(p) if self.data.family == ExpFamily::Gaussian => self.reml(&p),
            Ok(p) => self.laplace(&p),
            Err(_) => f64::INFINITY,
        }
    }
}

fn assemble(
    model: &Model<'_>,
    x: &[f64],
    modes: &Modes,
    sigma2: Option<f64>,
    n_iter: usize,
    converged: bool,
) -> GlmmFit {
    let npc = model.npc;
    let (psi, smooth) = split_theta(x, npc);
    let d = &model.design;
    let b_s = DVector::from_iterator(modes.us.len(), modes.us.iter().map(|v| v * smooth));
    let coef = &d.tf * DVector::from_row_slice(&modes.beta) + &d.tr * b_s;
    let mut theta = psi.clone();
    theta.push(smooth);
    GlmmFit {
        scores: modes
            .ui
            .iter()
            .map(|u| u.iter().zip(&psi).map(|(a, b)| a * b).collect())
            .collect(),
        mean_coef: coef.iter().copied().collect(),
        mean_basis: d.basis.clone(),
        theta,
        sigma2,
        n_iter,
        converged,
    }
}
