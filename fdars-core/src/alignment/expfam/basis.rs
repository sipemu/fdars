//! Spline bases, quantile knots and the template GLM of [`super::register_fpca`]
//! (R `splines::bs`, `stats::quantile`, `stats::glm` as used by registr 2.2.1).

use nalgebra::{DMatrix, DVector};

use crate::alignment::continuous::clamped_bspline_eval;
use crate::error::FdarError;

use super::ExpFamily;

/// Cubic B-spline order (registr always uses `degree = 3`).
const ORDER: usize = 4;

/// R `quantile(x, probs, type = 7)` of an ascending-sorted sample.
pub(super) fn quantile7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    let index = (n.saturating_sub(1)) as f64 * p;
    let lo = index.floor() as usize;
    let hi = (index.ceil() as usize).min(n - 1);
    let h = index - lo as f64;
    if h > 0.0 && sorted[hi] != sorted[lo] {
        (1.0 - h) * sorted[lo] + h * sorted[hi]
    } else {
        sorted[lo]
    }
}

/// `n_inner` interior knots at the quantiles `seq(0, 1, length = n_inner + 2)[-ends]`
/// of an ascending-sorted sample (registr's rule for both the template basis,
/// `n_inner = Kt − 4`, and the inverse-warp basis, `n_inner = Kh − 4`).
pub(super) fn quantile_knots(sorted: &[f64], n_inner: usize) -> Vec<f64> {
    let by = 1.0 / (n_inner + 1) as f64;
    (1..=n_inner)
        .map(|k| quantile7(sorted, k as f64 * by))
        .collect()
}

/// Cubic B-spline basis with intercept on `[a, b]` with the given interior
/// knots — R `splines::bs(x, knots, Boundary.knots = c(a, b), intercept = TRUE)`.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct CubicBs {
    knots: Vec<f64>,
}

impl CubicBs {
    pub(super) fn new(a: f64, b: f64, interior: &[f64]) -> Self {
        let mut knots = vec![a; ORDER];
        knots.extend_from_slice(interior);
        knots.extend(std::iter::repeat(b).take(ORDER));
        Self { knots }
    }

    /// Number of basis functions (`interior + 4`).
    pub(super) fn nbasis(&self) -> usize {
        self.knots.len() - ORDER
    }

    /// Basis values at `x`.
    pub(super) fn eval(&self, x: f64) -> Vec<f64> {
        clamped_bspline_eval(x, &self.knots, ORDER, 0)
    }

    /// First derivatives of the basis at `x` (registr `bs_deriv`).
    pub(super) fn deriv(&self, x: f64) -> Vec<f64> {
        clamped_bspline_eval(x, &self.knots, ORDER, 1)
    }

    /// Greville abscissae: the coefficients that reproduce the identity
    /// (equal to registr's `initial_params`, `ginv(Θ_h) t*`, since the identity
    /// lies in the spline space).
    pub(super) fn greville(&self) -> Vec<f64> {
        (0..self.nbasis())
            .map(|k| (self.knots[k + 1] + self.knots[k + 2] + self.knots[k + 3]) / 3.0)
            .collect()
    }
}

#[inline]
pub(super) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub(super) fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Canonical-family working quantities `(mu, dmu/deta)` of the template GLM
/// (identity / logit / log / log links — registr uses `Gamma(link = "log")`).
fn glm_mean(family: ExpFamily, eta: f64) -> (f64, f64) {
    match family {
        ExpFamily::Gaussian => (eta, 1.0),
        ExpFamily::Binomial => {
            let mu = sigmoid(eta).clamp(1e-12, 1.0 - 1e-12);
            (mu, mu * (1.0 - mu))
        }
        ExpFamily::Poisson | ExpFamily::Gamma => {
            let mu = eta.min(700.0).exp().max(1e-300);
            (mu, mu)
        }
    }
}

/// Variance function `V(μ)` (dispersion dropped).
fn glm_variance(family: ExpFamily, mu: f64) -> f64 {
    match family {
        ExpFamily::Gaussian => 1.0,
        ExpFamily::Binomial => mu * (1.0 - mu),
        ExpFamily::Poisson => mu,
        ExpFamily::Gamma => mu * mu,
    }
}

/// Deviance contribution of one observation.
fn glm_deviance(family: ExpFamily, y: f64, mu: f64) -> f64 {
    let xlogx = |a: f64, b: f64| if a > 0.0 { a * (a / b).ln() } else { 0.0 };
    match family {
        ExpFamily::Gaussian => (y - mu).powi(2),
        ExpFamily::Binomial => 2.0 * (xlogx(y, mu) + xlogx(1.0 - y, 1.0 - mu)),
        ExpFamily::Poisson => 2.0 * (xlogx(y, mu) - (y - mu)),
        ExpFamily::Gamma => -2.0 * ((y / mu).ln() - (y - mu) / mu),
    }
}

/// R `glm`'s `mustart` → initial linear predictor.
fn glm_eta_start(family: ExpFamily, y: f64) -> f64 {
    match family {
        ExpFamily::Gaussian => y,
        ExpFamily::Binomial => {
            let mu = (y + 0.5) / 2.0;
            (mu / (1.0 - mu)).ln()
        }
        ExpFamily::Poisson => (y + 0.1).ln(),
        ExpFamily::Gamma => y.ln(),
    }
}

/// Weighted least-squares solve `(XᵀWX) b = XᵀWz` (rows of `x` are design rows).
fn wls_solve(x: &[Vec<f64>], w: &[f64], z: &[f64]) -> Result<Vec<f64>, FdarError> {
    let p = x[0].len();
    let mut xtwx = DMatrix::<f64>::zeros(p, p);
    let mut xtwz = DVector::<f64>::zeros(p);
    for ((row, &wi), &zi) in x.iter().zip(w).zip(z) {
        for a in 0..p {
            let ra = row[a] * wi;
            xtwz[a] += ra * zi;
            for b in a..p {
                xtwx[(a, b)] += ra * row[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            xtwx[(a, b)] = xtwx[(b, a)];
        }
    }
    let sol = match xtwx.clone().cholesky() {
        Some(ch) => ch.solve(&xtwz),
        None => xtwx
            .lu()
            .solve(&xtwz)
            .ok_or_else(|| FdarError::ComputationFailed {
                operation: "register_fpca template GLM",
                detail: "singular weighted normal equations (too few distinct time points for Kt?)"
                    .into(),
            })?,
    };
    Ok(sol.iter().copied().collect())
}

/// Fit the template GLM `y ~ 0 + B` by IRLS (Fisher scoring), as R
/// `glm(y ~ 0 + B, family)` with registr's families (identity, logit, log, log).
///
/// Converges when the relative deviance change is `< 1e-12` (R: `1e-8`).
pub(super) fn glm_fit(x: &[Vec<f64>], y: &[f64], family: ExpFamily) -> Result<Vec<f64>, FdarError> {
    let mut eta: Vec<f64> = y.iter().map(|&v| glm_eta_start(family, v)).collect();
    let mut coef = Vec::new();
    let mut dev_old = f64::INFINITY;
    let max_iter = if family == ExpFamily::Gaussian {
        1
    } else {
        100
    };
    for _ in 0..max_iter {
        let (w, z): (Vec<f64>, Vec<f64>) = eta
            .iter()
            .zip(y)
            .map(|(&e, &yi)| {
                let (mu, dmu) = glm_mean(family, e);
                let w = dmu * dmu / glm_variance(family, mu).max(1e-300);
                (w, e + (yi - mu) / dmu)
            })
            .unzip();
        coef = wls_solve(x, &w, &z)?;
        eta = x.iter().map(|row| dot(row, &coef)).collect();
        let dev: f64 = eta
            .iter()
            .zip(y)
            .map(|(&e, &yi)| glm_deviance(family, yi, glm_mean(family, e).0))
            .sum();
        if (dev - dev_old).abs() / (dev.abs() + 0.1) < 1e-12 {
            break;
        }
        dev_old = dev;
    }
    Ok(coef)
}
