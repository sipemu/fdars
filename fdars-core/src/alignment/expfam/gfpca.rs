//! Variational-EM generalized FPCA (registr `fpca_gauss` and `bfpca`,
//! Wrobel et al. 2019 / Jaakkola & Jordan 1997 bound for binary data).

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::FdarError;
use crate::irreg_fdata::IrregFdata;
use crate::matrix::FdMatrix;

use super::basis::{glm_fit, quantile_knots, sigmoid, CubicBs};
use super::ExpFamily;

/// Number of points of the output grid (registr: `length.out = 100`).
const N_GRID: usize = 100;

/// Configuration of [`gfpca_variational`] (and the FPCA step of
/// [`super::register_fpca`]).
///
/// Construct with `GfpcaConfig::default()` and assign fields (the struct is
/// `#[non_exhaustive]`). Defaults are registr 2.2.1's.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GfpcaConfig {
    /// Number of functional principal components (default 1, `1 ≤ npc ≤ kt`).
    pub npc: usize,
    /// Number of cubic B-spline basis functions of the mean and the FPCs
    /// (registr `Kt`, default 8, `≥ 4`).
    pub kt: usize,
    /// Maximum number of EM iterations (registr `maxiter` / `fpca_maxiter`,
    /// default 50; at most `max_iter − 1` updates are performed, as in R).
    pub max_iter: usize,
    /// Stop when the squared change of all coefficients (and `σ²`) is
    /// `≤ error_thresh` (default `1e-4`).
    pub error_thresh: f64,
    /// Seed of the random initial FPC coefficients `0.5·N(0, 1)` (and, for binary
    /// data, of the initial variational parameters). Default 1988 (registr
    /// `fpca_seed`; the R and Rust random streams differ).
    pub seed: u64,
    /// Explicit initial FPC spline coefficients (`kt × npc`), overriding the
    /// random initialisation (e.g. to reproduce an R run).
    pub psi_init: Option<FdMatrix>,
    /// Constant initial variational parameter ξ for binary data, overriding the
    /// random `0.5·N(0, 1)` initialisation.
    pub xi_init: Option<f64>,
}

impl Default for GfpcaConfig {
    fn default() -> Self {
        Self {
            npc: 1,
            kt: 8,
            max_iter: 50,
            error_thresh: 1e-4,
            seed: 1988,
            psi_init: None,
            xi_init: None,
        }
    }
}

impl GfpcaConfig {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `kt < 4`, `npc` is 0 or exceeds `kt`,
    /// `max_iter == 0`, `error_thresh` is negative or not finite, or `psi_init`
    /// is not `kt × npc` / not finite.
    pub fn validate(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        if self.kt < 4 {
            return bad("kt", format!("must be >= 4, got {}", self.kt));
        }
        if self.npc == 0 || self.npc > self.kt {
            return bad(
                "npc",
                format!("must be in 1..=kt ({}), got {}", self.kt, self.npc),
            );
        }
        if self.max_iter == 0 {
            return bad("max_iter", "must be >= 1".into());
        }
        if !(self.error_thresh.is_finite() && self.error_thresh >= 0.0) {
            return bad(
                "error_thresh",
                format!("must be finite and >= 0, got {}", self.error_thresh),
            );
        }
        self.validate_init()
    }

    fn validate_init(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        if let Some(p) = &self.psi_init {
            if p.shape() != (self.kt, self.npc) || p.as_slice().iter().any(|v| !v.is_finite()) {
                return bad(
                    "psi_init",
                    format!(
                        "must be finite kt x npc = {} x {}, got {:?}",
                        self.kt,
                        self.npc,
                        p.shape()
                    ),
                );
            }
        }
        if matches!(self.xi_init, Some(x) if !x.is_finite()) {
            return bad("xi_init", "must be finite".into());
        }
        Ok(())
    }
}

/// Result of [`gfpca_variational`] (registr `fpca` object).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GfpcaResult {
    /// Family of the variational EM actually used: [`ExpFamily::Binomial`]
    /// (`bfpca`) or [`ExpFamily::Gaussian`] (`fpca_gauss`, also used for gamma /
    /// poisson data as in registr's `fpca_type = "variationalEM"`).
    pub em_family: ExpFamily,
    /// Interior knots of the cubic B-spline basis (`kt − 4` quantiles of the
    /// pooled time points).
    pub knots: Vec<f64>,
    /// Basis domain `[min t, max t]` of the pooled time points.
    pub domain: [f64; 2],
    /// Spline coefficients of the mean (latent scale; length `kt`).
    pub alpha_coefs: Vec<f64>,
    /// Spline coefficients of the (non-orthogonal) FPC basis (`kt × npc`).
    pub psi_coefs: FdMatrix,
    /// Output grid: 100 equally spaced points on `domain`.
    pub t_vec: Vec<f64>,
    /// Mean function on `t_vec` (latent scale; registr `mu`).
    pub mean: Vec<f64>,
    /// Orthonormal eigenvectors on `t_vec` (`100 × npc`, left singular vectors
    /// of the evaluated FPC basis; sign is arbitrary).
    pub efunctions: FdMatrix,
    /// Squared singular values (registr `evalues`).
    pub evalues: Vec<f64>,
    /// Orthogonalised scores (`n × npc`).
    pub scores: FdMatrix,
    /// Per-curve latent spline coefficients `α + ψ mᵢ` (`n × kt`).
    pub subject_coefs: FdMatrix,
    /// Per-curve fitted latent values at the input time points.
    pub fitted: Vec<Vec<f64>>,
    /// Residual variance (gaussian EM only).
    pub sigma2: Option<f64>,
    /// Number of EM updates performed.
    pub n_iter: usize,
    /// Whether the EM stopped by the `error_thresh` criterion.
    pub converged: bool,
}

/// Per-curve design quantities.
struct Subject {
    theta: DMatrix<f64>,
    y: DVector<f64>,
}

/// Kronecker accumulation of the M-step system (registr `phi_a`, `phi_b`).
struct MStep {
    a: DMatrix<f64>,
    b: DVector<f64>,
    p: usize,
}

impl MStep {
    fn new(kt: usize, npc: usize) -> Self {
        let p = npc + 1;
        Self {
            a: DMatrix::zeros(kt * p, kt * p),
            b: DVector::zeros(kt * p),
            p,
        }
    }

    /// `A += w · kron(Q, ss)`, `b += kron(Θᵀr, s)` with `s = (m, 1)`.
    fn add(
        &mut self,
        q: &DMatrix<f64>,
        w: f64,
        ci: &DMatrix<f64>,
        mi: &DVector<f64>,
        tr: &DVector<f64>,
    ) {
        let (p, kt) = (self.p, q.nrows());
        let npc = p - 1;
        let mut ss = DMatrix::<f64>::zeros(p, p);
        for a in 0..npc {
            for b in 0..npc {
                ss[(a, b)] = ci[(a, b)] + mi[a] * mi[b];
            }
            ss[(a, npc)] = mi[a];
            ss[(npc, a)] = mi[a];
        }
        ss[(npc, npc)] = 1.0;
        for k in 0..kt {
            for k2 in 0..kt {
                let qk = w * q[(k, k2)];
                for l in 0..p {
                    for l2 in 0..p {
                        self.a[(k * p + l, k2 * p + l2)] += qk * ss[(l, l2)];
                    }
                }
            }
            for l in 0..p {
                let sl = if l < npc { mi[l] } else { 1.0 };
                self.b[k * p + l] += tr[k] * sl;
            }
        }
    }

    /// Solve `A φ = sign · b`; returns `(α, ψ)` from `matrix(φ, kt, p, byrow = TRUE)`.
    fn solve(self, sign: f64) -> Result<(DVector<f64>, DMatrix<f64>), FdarError> {
        let p = self.p;
        let kt = self.b.len() / p;
        let phi =
            self.a
                .lu()
                .solve(&(self.b * sign))
                .ok_or_else(|| FdarError::ComputationFailed {
                    operation: "gfpca_variational M-step",
                    detail:
                        "singular Kronecker system (too few curves or time points for kt / npc?)"
                            .into(),
                })?;
        let alpha = DVector::from_fn(kt, |k, _| phi[k * p + p - 1]);
        let psi = DMatrix::from_fn(kt, p - 1, |k, l| phi[k * p + l]);
        Ok((alpha, psi))
    }
}

fn invert(m: DMatrix<f64>) -> Result<DMatrix<f64>, FdarError> {
    m.try_inverse().ok_or_else(|| FdarError::ComputationFailed {
        operation: "gfpca_variational E-step",
        detail: "singular score covariance".into(),
    })
}

/// Jaakkola–Jordan λ(ξ) = (½ − σ(ξ)) / (2ξ) (registr `lambdaF`), with its limit
/// −1/8 at 0.
fn lambda_jj(x: f64) -> f64 {
    if x.abs() < 1e-8 {
        -0.125
    } else {
        (0.5 - sigmoid(x)) / (2.0 * x)
    }
}

/// `Θᵀ diag(λ(ξ)) Θ` (registr `squareTheta`).
fn square_theta(theta: &DMatrix<f64>, xi: &[f64]) -> DMatrix<f64> {
    let mut w = theta.clone();
    for (j, &x) in xi.iter().enumerate() {
        let l = lambda_jj(x);
        for k in 0..w.ncols() {
            w[(j, k)] *= l;
        }
    }
    theta.transpose() * w
}

/// Mutable EM state.
struct EmState {
    alpha: DVector<f64>,
    psi: DMatrix<f64>,
    sigma2: f64,
    xi: Vec<Vec<f64>>,
    scores: Vec<DVector<f64>>,
}

/// One gaussian EM iteration (registr `fpca_gauss_optimization` loop body).
fn gauss_iteration(
    subjects: &[Subject],
    st: &mut EmState,
    n_total: usize,
) -> Result<(), FdarError> {
    let (kt, npc) = st.psi.shape();
    let mut ms = MStep::new(kt, npc);
    let mut sigma_sum = 0.0;
    for (i, sub) in subjects.iter().enumerate() {
        let q = sub.theta.transpose() * &sub.theta;
        let ty = sub.theta.transpose() * &sub.y;
        let m = st.psi.transpose() * &q * &st.psi;
        let ci = invert(&m / st.sigma2 + DMatrix::identity(npc, npc))?;
        let r = &ty - &q * &st.alpha;
        let pr = st.psi.transpose() * &r;
        let mi = &ci * &pr / st.sigma2;
        let e = &sub.y - &sub.theta * &st.alpha;
        sigma_sum += -2.0 * mi.dot(&pr) + e.dot(&e) + (&m * &ci).trace() + mi.dot(&(&m * &mi));
        ms.add(&q, 1.0, &ci, &mi, &ty);
        st.scores[i] = mi;
    }
    st.sigma2 = sigma_sum / n_total as f64;
    let (alpha, psi) = ms.solve(1.0)?;
    st.alpha = alpha;
    st.psi = psi;
    Ok(())
}

/// Updated variational parameters `ξ_t = √E[(θ_t(α + ψ s))²]` (registr `expectedXi`).
fn expected_xi(sub: &Subject, st: &EmState, ci: &DMatrix<f64>, mi: &DVector<f64>) -> Vec<f64> {
    let ta = &sub.theta * &st.alpha;
    let tp = &sub.theta * &st.psi;
    (0..sub.theta.nrows())
        .map(|t| {
            let b = tp.row(t).transpose();
            let bm = b.dot(mi);
            let a = ta[t];
            let v = a * a + 2.0 * a * bm + b.dot(&(ci * &b)) + bm * bm;
            v.max(0.0).sqrt()
        })
        .collect()
}

/// One binary EM iteration (registr `bfpca_optimization` loop body).
fn binary_iteration(subjects: &[Subject], st: &mut EmState) -> Result<(), FdarError> {
    let (kt, npc) = st.psi.shape();
    let mut ms = MStep::new(kt, npc);
    for (i, sub) in subjects.iter().enumerate() {
        let q = square_theta(&sub.theta, &st.xi[i]);
        let ci = invert(DMatrix::identity(npc, npc) - 2.0 * st.psi.transpose() * &q * &st.psi)?;
        let yc = sub.y.map(|v| v - 0.5);
        let ty = sub.theta.transpose() * &yc;
        let inner = st.psi.transpose() * &ty + 2.0 * st.psi.transpose() * &q * &st.alpha;
        let mi = &ci * inner;
        st.xi[i] = expected_xi(sub, st, &ci, &mi);
        let q = square_theta(&sub.theta, &st.xi[i]);
        ms.add(&q, 2.0, &ci, &mi, &ty);
        st.scores[i] = mi;
    }
    let (alpha, psi) = ms.solve(-1.0)?;
    st.alpha = alpha;
    st.psi = psi;
    Ok(())
}

fn sq_change(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Validated per-curve time points and values (sorted, finite).
pub(super) struct CurveData {
    pub t: Vec<Vec<f64>>,
    pub y: Vec<Vec<f64>>,
}

impl CurveData {
    pub(super) fn pooled_sorted_t(&self) -> Vec<f64> {
        let mut all: Vec<f64> = self.t.iter().flatten().copied().collect();
        all.sort_by(f64::total_cmp);
        all
    }

    pub(super) fn n_total(&self) -> usize {
        self.t.iter().map(Vec::len).sum()
    }
}

/// Initial `ψ` (`kt × npc`) and ξ.
fn initial_state(cfg: &GfpcaConfig, alpha: DVector<f64>, data: &CurveData) -> EmState {
    let (kt, npc) = (cfg.kt, cfg.npc);
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let psi = match &cfg.psi_init {
        Some(p) => DMatrix::from_fn(kt, npc, |k, l| p[(k, l)]),
        None => {
            let draws: Vec<f64> = (0..kt * npc)
                .map(|_| 0.5 * Distribution::<f64>::sample(&StandardNormal, &mut rng))
                .collect();
            DMatrix::from_column_slice(kt, npc, &draws)
        }
    };
    let mut xi_rng = StdRng::seed_from_u64(cfg.seed.wrapping_add(1));
    let xi = data
        .t
        .iter()
        .map(|t| {
            t.iter()
                .map(|_| match cfg.xi_init {
                    Some(x) => x,
                    None => 0.5 * Distribution::<f64>::sample(&StandardNormal, &mut xi_rng),
                })
                .collect()
        })
        .collect();
    EmState {
        alpha,
        psi,
        sigma2: 1.0,
        xi,
        scores: vec![DVector::zeros(npc); data.t.len()],
    }
}

/// Run the EM loop; returns `(n_updates, converged)`.
fn run_em(
    subjects: &[Subject],
    st: &mut EmState,
    binary: bool,
    cfg: &GfpcaConfig,
    n_total: usize,
) -> Result<(usize, bool), FdarError> {
    let mut iter = 1;
    let mut error = 100.0;
    while iter < cfg.max_iter && error > cfg.error_thresh {
        let (a0, p0, s0) = (st.alpha.clone(), st.psi.clone(), st.sigma2);
        if binary {
            binary_iteration(subjects, st)?;
        } else {
            gauss_iteration(subjects, st, n_total)?;
        }
        iter += 1;
        error = sq_change(st.psi.as_slice(), p0.as_slice())
            + sq_change(st.alpha.as_slice(), a0.as_slice());
        if !binary {
            error += (st.sigma2 - s0).powi(2);
        }
    }
    Ok((iter - 1, iter < cfg.max_iter))
}

/// SVD of the FPC basis on the output grid, sorted by decreasing singular value.
fn grid_svd(
    theta_grid: &DMatrix<f64>,
    psi: &DMatrix<f64>,
) -> (DMatrix<f64>, Vec<f64>, DMatrix<f64>) {
    let svd = (theta_grid * psi).svd(true, true);
    let (u, vt) = (svd.u.expect("u requested"), svd.v_t.expect("v requested"));
    let mut order: Vec<usize> = (0..svd.singular_values.len()).collect();
    order.sort_by(|&a, &b| svd.singular_values[b].total_cmp(&svd.singular_values[a]));
    let d: Vec<f64> = order.iter().map(|&k| svd.singular_values[k]).collect();
    let u_s = DMatrix::from_fn(u.nrows(), order.len(), |r, c| u[(r, order[c])]);
    let v_s = DMatrix::from_fn(vt.ncols(), order.len(), |r, c| vt[(order[c], r)]);
    (u_s, d, v_s)
}

fn to_fd(m: &DMatrix<f64>) -> FdMatrix {
    FdMatrix::from_column_major(m.as_slice().to_vec(), m.nrows(), m.ncols())
        .expect("dimensions match by construction")
}

/// Assemble the result from the converged EM state.
fn assemble(
    st: EmState,
    basis: &CubicBs,
    knots: Vec<f64>,
    domain: [f64; 2],
    subjects: &[Subject],
    binary: bool,
    (n_iter, converged): (usize, bool),
) -> GfpcaResult {
    let (kt, npc) = st.psi.shape();
    let n = subjects.len();
    let t_vec: Vec<f64> = (0..N_GRID)
        .map(|k| domain[0] + (domain[1] - domain[0]) * k as f64 / (N_GRID - 1) as f64)
        .collect();
    let theta_grid = DMatrix::from_fn(N_GRID, kt, |r, c| basis.eval(t_vec[r])[c]);
    let mean = (&theta_grid * &st.alpha).iter().copied().collect();
    let (u, d, v) = grid_svd(&theta_grid, &st.psi);
    let vd = &v * DMatrix::from_diagonal(&DVector::from_vec(d.clone()));
    let mut scores = FdMatrix::zeros(n, npc);
    let mut subject_coefs = FdMatrix::zeros(n, kt);
    let mut fitted = Vec::with_capacity(n);
    for (i, sub) in subjects.iter().enumerate() {
        let si = vd.transpose() * &st.scores[i];
        for l in 0..npc {
            scores[(i, l)] = si[l];
        }
        let coef = &st.alpha + &st.psi * &st.scores[i];
        for k in 0..kt {
            subject_coefs[(i, k)] = coef[k];
        }
        fitted.push((&sub.theta * &coef).iter().copied().collect());
    }
    GfpcaResult {
        em_family: if binary {
            ExpFamily::Binomial
        } else {
            ExpFamily::Gaussian
        },
        knots,
        domain,
        alpha_coefs: st.alpha.iter().copied().collect(),
        psi_coefs: to_fd(&st.psi),
        t_vec,
        mean,
        efunctions: to_fd(&u),
        evalues: d.iter().map(|x| x * x).collect(),
        scores,
        subject_coefs,
        fitted,
        sigma2: (!binary).then_some(st.sigma2),
        n_iter,
        converged,
    }
}

/// Core variational EM on validated curve data.
pub(super) fn gfpca_core(
    data: &CurveData,
    family: ExpFamily,
    cfg: &GfpcaConfig,
) -> Result<GfpcaResult, FdarError> {
    let binary = family == ExpFamily::Binomial;
    let pooled = data.pooled_sorted_t();
    let domain = [pooled[0], pooled[pooled.len() - 1]];
    let knots = quantile_knots(&pooled, cfg.kt - 4);
    let basis = CubicBs::new(domain[0], domain[1], &knots);
    let subjects: Vec<Subject> = data
        .t
        .iter()
        .zip(&data.y)
        .map(|(t, y)| Subject {
            theta: DMatrix::from_fn(t.len(), cfg.kt, |r, c| basis.eval(t[r])[c]),
            y: DVector::from_column_slice(y),
        })
        .collect();
    let rows: Vec<Vec<f64>> = data.t.iter().flatten().map(|&x| basis.eval(x)).collect();
    let ys: Vec<f64> = data.y.iter().flatten().copied().collect();
    let em_glm_family = if binary {
        ExpFamily::Binomial
    } else {
        ExpFamily::Gaussian
    };
    let alpha0 = DVector::from_vec(glm_fit(&rows, &ys, em_glm_family)?);
    let mut st = initial_state(cfg, alpha0, data);
    let iters = run_em(&subjects, &mut st, binary, cfg, data.n_total())?;
    Ok(assemble(
        st, &basis, knots, domain, &subjects, binary, iters,
    ))
}

/// Validate irregular data and copy it into per-curve vectors.
pub(super) fn curve_data(data: &IrregFdata, what: &'static str) -> Result<CurveData, FdarError> {
    let n = data.n_obs();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: what,
            expected: "at least 2 curves".into(),
            actual: format!("{n}"),
        });
    }
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let (ti, yi) = data.get_obs(i);
        let ok_len = ti.len() >= 2;
        let sorted = ti.windows(2).all(|w| w[1] > w[0]);
        let finite = ti.iter().chain(yi).all(|v| v.is_finite());
        if !(ok_len && sorted && finite) {
            return Err(FdarError::InvalidParameter {
                parameter: what,
                message: format!(
                    "curve {i} must have >= 2 finite points with strictly increasing argvals"
                ),
            });
        }
        t.push(ti.to_vec());
        y.push(yi.to_vec());
    }
    Ok(CurveData { t, y })
}

/// Check the response range of a family (binomial in `[0, 1]`, poisson `≥ 0`,
/// gamma `> 0`).
pub(super) fn check_family_values(data: &CurveData, family: ExpFamily) -> Result<(), FdarError> {
    let ok = |v: f64| match family {
        ExpFamily::Gaussian => true,
        ExpFamily::Binomial => (0.0..=1.0).contains(&v),
        ExpFamily::Poisson => v >= 0.0,
        ExpFamily::Gamma => v > 0.0,
    };
    if data.y.iter().flatten().all(|&v| ok(v)) {
        Ok(())
    } else {
        Err(FdarError::InvalidParameter {
            parameter: "data",
            message: format!("values out of range for family {family:?} (binomial: [0, 1], poisson: >= 0, gamma: > 0)"),
        })
    }
}

/// Variational-EM generalized functional PCA for exponential-family functional
/// data (R registr 2.2.1 `bfpca` for binary data, `fpca_gauss` otherwise).
///
/// Model: `g(E[Yᵢ(t)]) = Θ(t)ᵀ(α + ψ sᵢ)`, `sᵢ ~ N(0, I)`, with `Θ` a cubic
/// B-spline basis of `kt` functions (interior knots at quantiles of the pooled
/// time points). For gaussian data the EM is exact (`Yᵢ(t) ~ N(·, σ²)`); for
/// binary data the logistic likelihood is replaced by the Jaakkola–Jordan
/// variational lower bound with one parameter ξ per observation. As in registr's
/// `fpca_type = "variationalEM"`, gamma and poisson data are decomposed by the
/// gaussian EM on the data scale.
///
/// The mean, the orthonormal eigenvectors (`efunctions`, from the SVD of the
/// evaluated FPC basis on a 100-point grid) and the orthogonalised scores follow
/// registr's conventions exactly.
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] / [`FdarError::InvalidParameter`] for fewer
/// than 2 curves, curves with < 2 points or non-increasing argvals, values out
/// of the family range or an invalid `config`;
/// [`FdarError::ComputationFailed`] if an EM linear system is singular.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{gfpca_variational, ExpFamily, GfpcaConfig};
/// use fdars_core::irreg_fdata::IrregFdata;
///
/// let t: Vec<f64> = (0..30).map(|j| j as f64 / 29.0).collect();
/// let argvals: Vec<Vec<f64>> = (0..12).map(|_| t.clone()).collect();
/// let values: Vec<Vec<f64>> = (0..12)
///     .map(|i| {
///         let a = (i as f64 - 5.5) / 4.0;
///         t.iter().map(|&s| (2.0 * std::f64::consts::PI * s).sin() + a * s).collect()
///     })
///     .collect();
/// let data = IrregFdata::from_lists(&argvals, &values);
/// let mut cfg = GfpcaConfig::default();
/// cfg.npc = 1;
/// let fit = gfpca_variational(&data, ExpFamily::Gaussian, &cfg).unwrap();
/// assert_eq!(fit.mean.len(), 100);
/// assert_eq!(fit.efunctions.shape(), (100, 1));
/// assert!(fit.sigma2.unwrap() < 0.01);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn gfpca_variational(
    data: &IrregFdata,
    family: ExpFamily,
    config: &GfpcaConfig,
) -> Result<GfpcaResult, FdarError> {
    config.validate()?;
    let cd = curve_data(data, "data")?;
    check_family_values(&cd, family)?;
    gfpca_core(&cd, family, config)
}
