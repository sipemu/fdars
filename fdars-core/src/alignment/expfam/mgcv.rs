//! Native replacements for the mgcv / Matrix pieces of registr's two-step GFPCA
//! (`cov_hall`): P-spline bases (`bs = "ps"`), penalised GLMs with GCV / UBRE
//! smoothing-parameter selection, the tensor-product P-spline smoother
//! `te(i1, i2, bs = "ps")` and `Matrix::nearPD`.

use nalgebra::{DMatrix, DVector, SymmetricEigen};

use crate::alignment::continuous::clamped_bspline_eval;
use crate::error::FdarError;

use super::basis::{glm_deviance, glm_eta_start, glm_mean, glm_variance};
use super::warp::lbfgs_minimize;
use super::ExpFamily;

/// Cubic B-spline order of mgcv's `ps` basis (`m = c(2, 2)`).
const ORDER: usize = 4;

/// mgcv `smooth.construct.ps.smooth.spec` basis: `k` cubic B-splines on evenly
/// spaced knots covering the data range widened by 0.1 % (second-order
/// difference penalty).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PsBasis {
    knots: Vec<f64>,
}

impl PsBasis {
    pub(super) fn new(xmin: f64, xmax: f64, k: usize) -> Self {
        let nk = k - 2;
        let xr = xmax - xmin;
        let (xl, xu) = (xmin - xr * 0.001, xmax + xr * 0.001);
        let dx = (xu - xl) / (nk - 1) as f64;
        let (lo, hi) = (xl - dx * 3.0, xu + dx * 3.0);
        let n = nk + 6;
        let knots = (0..n)
            .map(|i| lo + (hi - lo) * i as f64 / (n - 1) as f64)
            .collect();
        Self { knots }
    }

    pub(super) fn eval(&self, x: f64) -> Vec<f64> {
        clamped_bspline_eval(x, &self.knots, ORDER, 0)
    }
}

/// Second-order difference penalty `DᵀD` (`k × k`).
pub(super) fn diff_penalty(k: usize) -> DMatrix<f64> {
    let mut d = DMatrix::<f64>::zeros(k - 2, k);
    for r in 0..k - 2 {
        d[(r, r)] = 1.0;
        d[(r, r + 1)] = -2.0;
        d[(r, r + 2)] = 1.0;
    }
    d.transpose() * d
}

/// Sorted (decreasing) symmetric eigen-decomposition `(values, vectors)`.
pub(super) fn sym_eigen_desc(m: &DMatrix<f64>) -> (Vec<f64>, DMatrix<f64>) {
    let sym = (m + m.transpose()) * 0.5;
    let eig = SymmetricEigen::new(sym);
    let mut order: Vec<usize> = (0..eig.eigenvalues.len()).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].total_cmp(&eig.eigenvalues[a]));
    let values = order.iter().map(|&k| eig.eigenvalues[k]).collect();
    let n = m.nrows();
    let vectors = DMatrix::from_fn(n, order.len(), |r, c| eig.eigenvectors[(r, order[c])]);
    (values, vectors)
}

fn singular(operation: &'static str) -> FdarError {
    FdarError::ComputationFailed {
        operation,
        detail: "singular penalised normal equations".into(),
    }
}

/// Solve the SPD system `a x = b` by Cholesky; `None` if not positive definite.
fn chol_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<(DVector<f64>, DMatrix<f64>)> {
    let ch = a.clone().cholesky()?;
    Some((ch.solve(b), ch.inverse()))
}

/// Minimise a 1-D function on `[lo, hi]`: grid of `n` points, then golden-section
/// refinement around the best one.
pub(super) fn minimise_1d(f: &dyn Fn(f64) -> f64, lo: f64, hi: f64, n: usize) -> f64 {
    let h = (hi - lo) / (n - 1) as f64;
    let (mut best, mut fbest) = (lo, f64::INFINITY);
    for k in 0..n {
        let x = lo + h * k as f64;
        let v = f(x);
        if v < fbest {
            (best, fbest) = (x, v);
        }
    }
    let (mut a, mut b) = ((best - h).max(lo), (best + h).min(hi));
    let g = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    while b - a > 1e-9 {
        if fc < fd {
            (b, d, fd) = (d, c, fc);
            c = b - g * (b - a);
            fc = f(c);
        } else {
            (a, c, fc) = (c, d, fd);
            d = a + g * (b - a);
            fd = f(d);
        }
    }
    let x = 0.5 * (a + b);
    if f(x) <= fbest {
        x
    } else {
        best
    }
}

/// A converged penalised GLM fit.
pub(super) struct PenFit {
    pub coef: Vec<f64>,
    pub eta: Vec<f64>,
    /// Sum of deviance residuals.
    pub dev: f64,
    /// Effective degrees of freedom `tr((XᵀWX + λS)⁻¹ XᵀWX)`.
    pub edf: f64,
}

/// Penalised IRLS for `g(E[y]) = Xc` with penalty `λ cᵀSc` (mgcv `gam.fit3`
/// inner loop, family as registr's: identity / logit / log / log (Tweedie p = 2
/// ≡ gamma working weights)).
pub(super) fn pirls(
    x: &DMatrix<f64>,
    y: &[f64],
    family: ExpFamily,
    s: &DMatrix<f64>,
    lambda: f64,
) -> Result<PenFit, FdarError> {
    let n = y.len();
    let mut eta: Vec<f64> = y.iter().map(|&v| glm_eta_start(family, v)).collect();
    let mut dev_old = f64::INFINITY;
    let max_iter = if family == ExpFamily::Gaussian {
        1
    } else {
        200
    };
    let mut out = None;
    for _ in 0..max_iter {
        let mut xw = x.clone();
        let mut z = DVector::<f64>::zeros(n);
        let mut w = vec![0.0; n];
        for r in 0..n {
            let (mu, dmu) = glm_mean(family, eta[r]);
            w[r] = dmu * dmu / glm_variance(family, mu).max(1e-300);
            z[r] = eta[r] + (y[r] - mu) / dmu;
            let sw = w[r].sqrt();
            for c in 0..x.ncols() {
                xw[(r, c)] *= sw;
            }
            z[r] *= sw;
        }
        let xtwx = xw.transpose() * &xw;
        let rhs = xw.transpose() * &z;
        let a = &xtwx + s * lambda;
        let (coef, ainv) = chol_solve(&a, &rhs).ok_or_else(|| singular("gfpca_two_step smooth"))?;
        eta = (x * &coef).iter().copied().collect();
        let dev: f64 = eta
            .iter()
            .zip(y)
            .map(|(&e, &yi)| glm_deviance(family, yi, glm_mean(family, e).0))
            .sum();
        let edf = ainv.component_mul(&xtwx).sum();
        out = Some(PenFit {
            coef: coef.iter().copied().collect(),
            eta: eta.clone(),
            dev,
            edf,
        });
        if (dev - dev_old).abs() / (dev.abs() + 0.1) < 1e-12 {
            break;
        }
        dev_old = dev;
    }
    out.ok_or_else(|| singular("gfpca_two_step smooth"))
}

/// mgcv's default (`method = "GCV.Cp"`) smoothness score: UBRE for the
/// known-scale families (binomial, poisson), GCV otherwise.
fn gcv_ubre(family: ExpFamily, fit: &PenFit, n: usize) -> f64 {
    let nf = n as f64;
    match family {
        ExpFamily::Binomial | ExpFamily::Poisson => fit.dev / nf - 1.0 + 2.0 * fit.edf / nf,
        ExpFamily::Gaussian | ExpFamily::Gamma => nf * fit.dev / (nf - fit.edf).powi(2),
    }
}

/// `gam(y ~ s(x, bs = "ps", k), family)` with the smoothing parameter chosen
/// by GCV / UBRE. Returns the basis and the fit.
pub(super) fn gam_ps(
    xs: &[f64],
    y: &[f64],
    k: usize,
    family: ExpFamily,
) -> Result<(PsBasis, PenFit), FdarError> {
    let (lo, hi) = xs
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| {
            (a.min(v), b.max(v))
        });
    let basis = PsBasis::new(lo, hi, k);
    let x = DMatrix::from_fn(xs.len(), k, |r, c| basis.eval(xs[r])[c]);
    let s = diff_penalty(k);
    let off = ((x.transpose() * &x).trace() / s.trace()).ln();
    let score = |rho: f64| {
        pirls(&x, y, family, &s, (off + rho).exp())
            .map_or(f64::INFINITY, |f| gcv_ubre(family, &f, y.len()))
    };
    let rho = minimise_1d(&score, -15.0, 15.0, 31);
    let fit = pirls(&x, y, family, &s, (off + rho).exp())?;
    Ok((basis, fit))
}

/// One margin of mgcv's `te(..., bs = "ps")` after the `np = TRUE`
/// reparameterisation (coefficients are function values at `k` evenly spaced
/// points) with the penalty scaled to unit largest eigenvalue.
struct TeMargin {
    basis: PsBasis,
    xp: DMatrix<f64>,
    s: DMatrix<f64>,
}

impl TeMargin {
    fn new(lo: f64, hi: f64, k: usize) -> Result<Self, FdarError> {
        let basis = PsBasis::new(lo, hi, k);
        let pts: Vec<f64> = (0..k)
            .map(|i| lo + (hi - lo) * i as f64 / (k - 1) as f64)
            .collect();
        let xk = DMatrix::from_fn(k, k, |r, c| basis.eval(pts[r])[c]);
        let xp = xk
            .try_inverse()
            .ok_or_else(|| singular("gfpca_two_step te"))?;
        let s = xp.transpose() * diff_penalty(k) * &xp;
        let top = sym_eigen_desc(&s).0[0];
        Ok(Self {
            basis,
            xp,
            s: s / top,
        })
    }

    fn eval(&self, x: f64) -> DVector<f64> {
        self.xp.transpose() * DVector::from_vec(self.basis.eval(x))
    }
}

/// Fitted tensor-product P-spline surface.
pub(super) struct TeFit {
    m1: TeMargin,
    m2: TeMargin,
    coef: DMatrix<f64>,
}

impl TeFit {
    /// Predictions on the grid `g × g` (`out[(a, b)] = f(g[a], g[b])`).
    pub(super) fn predict_grid(&self, g: &[f64]) -> DMatrix<f64> {
        let k1 = self.coef.nrows();
        let k2 = self.coef.ncols();
        let x1 = DMatrix::from_fn(g.len(), k1, |r, c| self.m1.eval(g[r])[c]);
        let x2 = DMatrix::from_fn(g.len(), k2, |r, c| self.m2.eval(g[r])[c]);
        x1 * &self.coef * x2.transpose()
    }
}

/// Accumulated least-squares system of a tensor smooth.
struct TeSystem {
    g: DMatrix<f64>,
    b: DVector<f64>,
    yy: f64,
    n: usize,
    s1: DMatrix<f64>,
    s2: DMatrix<f64>,
}

impl TeSystem {
    /// `(coef, gcv)` at log smoothing parameters `rho`.
    fn solve(&self, rho: [f64; 2]) -> Option<(DVector<f64>, f64)> {
        let a = &self.g + &self.s1 * rho[0].exp() + &self.s2 * rho[1].exp();
        let (coef, ainv) = chol_solve(&a, &self.b)?;
        let rss = (self.yy - 2.0 * coef.dot(&self.b) + coef.dot(&(&self.g * &coef))).max(0.0);
        let edf = ainv.component_mul(&self.g).sum();
        let nf = self.n as f64;
        Some((coef, nf * rss / (nf - edf).powi(2)))
    }
}

/// Gaussian `gam(z ~ te(x1, x2, k = c(k, k), bs = "ps"))` with both smoothing
/// parameters chosen by GCV (grid search, then L-BFGS on `log λ`).
pub(super) fn te_ps_gcv(rows: &[(f64, f64, f64)], k: usize) -> Result<TeFit, FdarError> {
    let range = |f: fn(&(f64, f64, f64)) -> f64| {
        rows.iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), r| {
                (a.min(f(r)), b.max(f(r)))
            })
    };
    let (l1, h1) = range(|r| r.0);
    let (l2, h2) = range(|r| r.1);
    let m1 = TeMargin::new(l1, h1, k)?;
    let m2 = TeMargin::new(l2, h2, k)?;
    let p = k * k;
    let mut g = DMatrix::<f64>::zeros(p, p);
    let mut b = DVector::<f64>::zeros(p);
    let mut yy = 0.0;
    for &(x1, x2, z) in rows {
        let (e1, e2) = (m1.eval(x1), m2.eval(x2));
        let row = DVector::from_fn(p, |c, _| e1[c / k] * e2[c % k]);
        g.ger(1.0, &row, &row, 1.0);
        b.axpy(z, &row, 1.0);
        yy += z * z;
    }
    let id = DMatrix::<f64>::identity(k, k);
    let sys = TeSystem {
        s1: m1.s.kronecker(&id),
        s2: id.kronecker(&m2.s),
        g,
        b,
        yy,
        n: rows.len(),
    };
    let off = (sys.g.trace() / sys.s1.trace()).ln();
    let gcv = |r: [f64; 2]| {
        sys.solve([off + r[0], off + r[1]])
            .map_or(f64::INFINITY, |s| s.1)
    };
    let mut best = ([0.0, 0.0], f64::INFINITY);
    for i in 0..5 {
        for j in 0..5 {
            let r = [-12.0 + 6.0 * i as f64, -12.0 + 6.0 * j as f64];
            let v = gcv(r);
            if v < best.1 {
                best = (r, v);
            }
        }
    }
    let objective = |x: &[f64]| {
        let f = gcv([x[0], x[1]]);
        (f, fd_gradient(&|y: &[f64]| gcv([y[0], y[1]]), x, f))
    };
    let min = lbfgs_minimize(objective, best.0.to_vec(), 200, 1e-12);
    let rho = if min.f < best.1 {
        [min.x[0], min.x[1]]
    } else {
        best.0
    };
    let (coef, _) = sys
        .solve([off + rho[0], off + rho[1]])
        .ok_or_else(|| singular("gfpca_two_step te"))?;
    let coef = DMatrix::from_fn(k, k, |a, c| coef[a * k + c]);
    Ok(TeFit { m1, m2, coef })
}

/// Central finite-difference gradient of `f` at `x` (`fx = f(x)` unused but kept
/// for a uniform objective signature).
pub(super) fn fd_gradient(f: &dyn Fn(&[f64]) -> f64, x: &[f64], _fx: f64) -> Vec<f64> {
    let mut xx = x.to_vec();
    (0..x.len())
        .map(|k| {
            let h = 1e-5 * x[k].abs().max(1.0);
            xx[k] = x[k] + h;
            let fp = f(&xx);
            xx[k] = x[k] - h;
            let fm = f(&xx);
            xx[k] = x[k];
            if fp.is_finite() && fm.is_finite() {
                (fp - fm) / (2.0 * h)
            } else {
                0.0
            }
        })
        .collect()
}

/// Infinity norm (maximum absolute row sum).
fn norm_inf(m: &DMatrix<f64>) -> f64 {
    (0..m.nrows())
        .map(|r| m.row(r).iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// `Matrix::nearPD(x, do2eigen = TRUE)` (Matrix 1.7; Higham 2002 alternating
/// projections with Dykstra's correction, `keepDiag = FALSE`, `corr = FALSE`,
/// `eig.tol = 1e-6`, `conv.tol = 1e-7`, `posd.tol = 1e-8`, `maxit = 100`).
pub(super) fn near_pd(x: &DMatrix<f64>) -> DMatrix<f64> {
    let n = x.nrows();
    let mut xk = x.clone();
    let mut ds = DMatrix::<f64>::zeros(n, n);
    for _ in 0..100 {
        let y = xk.clone();
        let r = &y - &ds;
        let (d, q) = sym_eigen_desc(&r);
        let keep: Vec<usize> = (0..n).filter(|&k| d[k] > 1e-6 * d[0]).collect();
        let mut nx = DMatrix::<f64>::zeros(n, n);
        for &k in &keep {
            let v = q.column(k);
            nx.ger(d[k], &v, &v, 1.0);
        }
        ds = &nx - &r;
        xk = nx;
        let conv = norm_inf(&(&y - &xk)) / norm_inf(&y);
        if conv <= 1e-7 {
            break;
        }
    }
    let (mut d, q) = sym_eigen_desc(&xk);
    let eps = 1e-8 * d[0].abs();
    if d[n - 1] < eps {
        d.iter_mut().for_each(|v| *v = v.max(eps));
        let o_diag: Vec<f64> = (0..n).map(|i| xk[(i, i)]).collect();
        let mut nx = DMatrix::<f64>::zeros(n, n);
        for k in 0..n {
            let v = q.column(k);
            nx.ger(d[k], &v, &v, 1.0);
        }
        let dd: Vec<f64> = (0..n)
            .map(|i| (o_diag[i].max(eps) / nx[(i, i)]).sqrt())
            .collect();
        for i in 0..n {
            for j in 0..n {
                nx[(i, j)] *= dd[i] * dd[j];
            }
        }
        xk = nx;
    }
    xk
}
