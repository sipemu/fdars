//! Variational-EM generalized FPCA (registr `fpca_gauss` and `bfpca`,
//! Wrobel et al. 2019 / Jaakkola & Jordan 1997 bound for binary data).

use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::FdarError;
use crate::irreg_fdata::IrregFdata;
use crate::matrix::FdMatrix;

use super::basis::{glm_fit, sigmoid, template_knots, TemplateBasis};
use super::ExpFamily;

/// Number of points of the output grid (registr: `length.out = 100`).
const N_GRID: usize = 100;

/// Basis size used by the variational EM when the number of components is
/// chosen by [`GfpcaConfig::npc_criterion`] (registr hard-codes `Kt = 20`).
pub(super) const KT_NPC_CRITERION: usize = 20;

/// GFPCA algorithm of [`super::register_fpca`] (registr `fpca_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FpcaType {
    /// Variational EM ([`gfpca_variational`], registr `"variationalEM"`; default).
    #[default]
    VariationalEm,
    /// Two-step GFPCA ([`super::gfpca_two_step`], registr `"two-step"`,
    /// Gertheiss, Goldsmith & Staicu 2017): family-specific for all four families.
    TwoStep,
}

/// Choice of the number of FPCs by the share of explained variance (registr
/// `npc_criterion`).
///
/// With `var_explained = c₁` and `min_component_share = Some(c₂)`, the number of
/// components is `min(npc₁, npc₂)` where `npc₁` is the smallest `k` whose
/// cumulative share reaches `c₁` and `npc₂` the number of leading components
/// that each explain at least `c₂` (registr `determine_npc`). The variational EM
/// uses only `c₁` (cumulative share strictly above `c₁`) and, as registr, fixes
/// the spline basis to 20 functions.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct NpcCriterion {
    /// Cumulative share of the (approximated) total variance to explain, in `[0, 1]`.
    pub var_explained: f64,
    /// Optional minimum share of every retained component, in `[0, 1]`
    /// (two-step GFPCA only; registr's second `npc_criterion` element).
    pub min_component_share: Option<f64>,
}

impl NpcCriterion {
    /// Criterion with cumulative share `var_explained` and no per-component bound.
    #[must_use]
    pub fn new(var_explained: f64) -> Self {
        Self {
            var_explained,
            min_component_share: None,
        }
    }

    /// Criterion with both elements (registr `npc_criterion = c(c1, c2)`).
    #[must_use]
    pub fn with_min_share(var_explained: f64, min_component_share: f64) -> Self {
        Self {
            var_explained,
            min_component_share: Some(min_component_share),
        }
    }

    fn validate(&self) -> Result<(), FdarError> {
        let ok = |v: f64| (0.0..=1.0).contains(&v);
        if ok(self.var_explained) && self.min_component_share.map_or(true, ok) {
            Ok(())
        } else {
            Err(FdarError::InvalidParameter {
                parameter: "npc_criterion",
                message: format!("elements must lie in [0, 1], got {self:?}"),
            })
        }
    }

    /// registr `determine_npc` on decreasing positive eigenvalues (at least 1).
    pub(super) fn choose(&self, evalues: &[f64]) -> usize {
        let total: f64 = evalues.iter().sum();
        let share: Vec<f64> = evalues.iter().map(|v| v / total).collect();
        let mut cum = 0.0;
        let mut npc1 = share.len();
        for (k, s) in share.iter().enumerate() {
            cum += s;
            if cum >= self.var_explained {
                npc1 = k + 1;
                break;
            }
        }
        let npc2 = self.min_component_share.map_or(share.len(), |c2| {
            share.iter().position(|&s| s < c2).unwrap_or(share.len())
        });
        npc1.min(npc2).max(1)
    }

    /// registr `fpca_gauss` / `bfpca`: first `k` whose cumulative share
    /// exceeds `var_explained` (strictly).
    pub(super) fn choose_variational(&self, evalues: &[f64]) -> usize {
        let total: f64 = evalues.iter().sum();
        let mut cum = 0.0;
        for (k, v) in evalues.iter().enumerate() {
            cum += v;
            if cum / total > self.var_explained {
                return k + 1;
            }
        }
        evalues.len()
    }
}

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
    /// Choose the number of components by explained variance instead of `npc`
    /// (registr `npc_criterion`; default `None`). For the variational EM this
    /// sets the basis size to 20 (as registr) and `psi_init`, if given, must be
    /// `20 × 20` (the chosen-`npc` run starts from its first `npc` columns).
    pub npc_criterion: Option<NpcCriterion>,
    /// Periodic B-spline basis (registr `periodic = TRUE`, `pbs::pbs`, `kt − 1`
    /// interior knots) for the mean / FPCs and, in [`super::register_fpca`], the
    /// registration template (default `false`).
    pub periodic: bool,
    /// Two-step GFPCA only: round the time points to this many significant
    /// digits to form the common output grid (registr
    /// `fpca_index_significantDigits`, default `Some(4)`; `None` disables).
    pub index_significant_digits: Option<u32>,
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
            npc_criterion: None,
            periodic: false,
            index_significant_digits: Some(4),
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
        if self.npc_criterion.is_none() && (self.npc == 0 || self.npc > self.kt) {
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
        if matches!(self.index_significant_digits, Some(d) if d < 2) {
            return bad("index_significant_digits", "must be >= 2".into());
        }
        if let Some(c) = &self.npc_criterion {
            c.validate()?;
        }
        self.validate_init()
    }

    fn validate_init(&self) -> Result<(), FdarError> {
        let bad = |parameter: &'static str, message: String| {
            Err(FdarError::InvalidParameter { parameter, message })
        };
        let want = if self.npc_criterion.is_some() {
            (KT_NPC_CRITERION, KT_NPC_CRITERION)
        } else {
            (self.kt, self.npc)
        };
        if let Some(p) = &self.psi_init {
            if p.shape() != want || p.as_slice().iter().any(|v| !v.is_finite()) {
                return bad(
                    "psi_init",
                    format!(
                        "must be finite kt x npc = {} x {}, got {:?}",
                        want.0,
                        want.1,
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
    /// Algorithm that produced this result.
    pub fpca_type: FpcaType,
    /// Family of the model actually fitted. Variational EM:
    /// [`ExpFamily::Binomial`] (`bfpca`) or [`ExpFamily::Gaussian`] (`fpca_gauss`,
    /// also used for gamma / poisson data as in registr's `fpca_type =
    /// "variationalEM"`); two-step: the data family.
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
    /// Output grid: 100 equally spaced points on `domain` (two-step: the sorted
    /// unique rounded time points).
    pub t_vec: Vec<f64>,
    /// Mean function on `t_vec` (latent scale; registr `mu`).
    pub mean: Vec<f64>,
    /// Orthonormal eigenvectors on `t_vec` (`t_vec.len() × npc`; variational EM:
    /// left singular vectors of the evaluated FPC basis, two-step: eigenvectors
    /// of the latent covariance; sign is arbitrary).
    pub efunctions: FdMatrix,
    /// Squared singular values (two-step: covariance eigenvalues; registr
    /// `evalues`).
    pub evalues: Vec<f64>,
    /// Orthogonalised scores (`n × npc`).
    pub scores: FdMatrix,
    /// Per-curve latent spline coefficients `α + ψ mᵢ` (`n × kt`).
    pub subject_coefs: FdMatrix,
    /// Per-curve fitted latent values at the input time points.
    pub fitted: Vec<Vec<f64>>,
    /// Residual variance (gaussian EM only).
    pub sigma2: Option<f64>,
    /// Number of EM updates performed (two-step: iterations of the GLMM
    /// variance-parameter optimiser).
    pub n_iter: usize,
    /// Whether the EM stopped by the `error_thresh` criterion (two-step: whether
    /// the GLMM optimiser converged).
    pub converged: bool,
    /// Sum of all (positive) eigenvalues when `npc` was chosen by
    /// [`GfpcaConfig::npc_criterion`] (variational EM) or always (two-step)
    /// (registr `evalues_sum`).
    pub evalues_sum: Option<f64>,
    /// Two-step only: GLMM random-effect standard deviations, one per FPC score
    /// (relative to the residual standard deviation for gaussian data, as lme4's
    /// `theta`), followed by the one of the penalised mean smooth (in fdars'
    /// parameterisation of the P-spline penalty).
    pub variance_components: Option<Vec<f64>>,
    /// Two-step only: fitted latent curves `α(t) + Σₖ sᵢₖ φₖ(t)` on `t_vec`
    /// (`n × t_vec.len()`, registr `Yhat`).
    pub yhat: Option<FdMatrix>,
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
    basis: &TemplateBasis,
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
        fpca_type: FpcaType::VariationalEm,
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
        evalues_sum: None,
        variance_components: None,
        yhat: None,
    }
}

/// Core variational EM on validated curve data.
pub(super) fn gfpca_core(
    data: &CurveData,
    family: ExpFamily,
    cfg: &GfpcaConfig,
) -> Result<GfpcaResult, FdarError> {
    let Some(crit) = cfg.npc_criterion else {
        return gfpca_fixed(data, family, cfg);
    };
    // registr `fpca_gauss` / `bfpca` with `npc_varExplained`: Kt = 20, a run with
    // npc = 20 chooses npc, then the EM is rerun with the chosen npc.
    let mut full = cfg.clone();
    full.kt = KT_NPC_CRITERION;
    full.npc = KT_NPC_CRITERION;
    full.npc_criterion = None;
    let choice = gfpca_fixed(data, family, &full)?;
    let npc = crit.choose_variational(&choice.evalues);
    let mut sub = full.clone();
    sub.npc = npc;
    sub.psi_init = cfg.psi_init.as_ref().map(|p| {
        FdMatrix::from_column_major(
            p.as_slice()[..KT_NPC_CRITERION * npc].to_vec(),
            KT_NPC_CRITERION,
            npc,
        )
        .expect("dimensions match by construction")
    });
    let mut res = gfpca_fixed(data, family, &sub)?;
    res.evalues = choice.evalues[..npc].to_vec();
    res.evalues_sum = Some(choice.evalues.iter().sum());
    Ok(res)
}

/// Variational EM with a fixed number of components.
fn gfpca_fixed(
    data: &CurveData,
    family: ExpFamily,
    cfg: &GfpcaConfig,
) -> Result<GfpcaResult, FdarError> {
    let binary = family == ExpFamily::Binomial;
    let pooled = data.pooled_sorted_t();
    let domain = [pooled[0], pooled[pooled.len() - 1]];
    let knots = template_knots(&pooled, cfg.kt, cfg.periodic);
    let basis = TemplateBasis::new(domain[0], domain[1], &knots, cfg.periodic);
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
/// With [`GfpcaConfig::npc_criterion`] the number of components is the first
/// whose cumulative share of the squared singular values of a 20-component,
/// 20-basis-function fit exceeds `var_explained`, and the EM is rerun with that
/// many components (registr `npc_varExplained`; `kt` is set to 20 and
/// `evalues` / `evalues_sum` come from the 20-component fit, as in R). With
/// [`GfpcaConfig::periodic`] the basis is the periodic cubic B-spline of
/// `pbs::pbs` with `kt − 1` interior knots.
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
