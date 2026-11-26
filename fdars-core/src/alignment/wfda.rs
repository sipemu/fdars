//! Tang–Müller pairwise curve synchronisation (R fdapace `WFDA`).
//!
//! Tang & Müller (2008, *Biometrika* 95, "Pairwise curve synchronization for
//! functional data") estimate each curve's time warp from *pairwise* warps
//! instead of warping every curve to a template. For curve `i`:
//!
//! 1. pick `K = round(subset_prop · (n − 1))` partner curves at random
//!    (deterministically seeded with `seed + i`);
//! 2. for every partner `k`, fit a monotone pairwise warp `h_ik` with
//!    `ỹ_k ∘ h_ik ≈ ỹ_i` by minimising
//!    `Σⱼ (ỹ_k(h_ik(wⱼ)) − ỹ_i(wⱼ))² + λ Σⱼ (h_ik(wⱼ) − wⱼ)²`,
//!    where `ỹ = y / max|y|` are amplitude-normalised curves and `w` is the grid
//!    mapped to `[0, 1]`;
//! 3. average the pairwise warps into the inverse warp `h_i⁻¹` — a *truncated*
//!    mean that drops the partners whose pairwise cost is above the 0.9
//!    quantile, or a mean *weighted* by the inverse costs ([`WfdaChoice`]);
//! 4. invert it to the warp `h_i` and register `y_i ∘ h_i`.
//!
//! Since `h_ik ≈ h_k⁻¹ ∘ h_i` for curves `y_i = x ∘ h_i⁻¹` of a common shape `x`,
//! the average over partners estimates `h_i` composed with the mean inverse
//! warp: the common time scale is the one in which the average warp is the
//! identity, so no template is needed.
//!
//! The pairwise warps are piecewise linear with `nknots` equally spaced
//! interior knots whose (sorted) heights are optimised inside
//! `[1e-6, 1 − 1e-6]` ([`WfdaConfig::is_pwl`]), or are drawn by a random search
//! over `K²` sorted-uniform monotone candidates (`is_pwl = false`).
//!
//! # Correspondence with fdapace 0.6.0
//!
//! The defaults are those of `WFDA(Ly, Lt, optns)`: `choice = "truncated"`,
//! `subsetProp = 0.5`, `nknots = 2`, `isPWL = TRUE`, `seed = 666` and
//! `λ = 10⁻⁴ · √(Σᵢ ∫(ỹᵢ − μ̃)² / (n − 1))` with the cross-sectional mean `μ̃`
//! and trapezoidal integration over `argvals`. Differences:
//!
//! - **Pairwise cost.** fdapace 0.6.0 computes the pairwise cost used by the
//!   truncation / weighting as `mean((ỹ_i(h_ik) − Ỹ[c_k])²)`, where `Ỹ[c_k]` is
//!   a single matrix element (linear indexing: the partner's value at the first
//!   grid point) rather than the partner curve. fdars defaults to the intended
//!   pairwise data-fit cost `mean((ỹ_k(h_ik) − ỹ_i)²)`
//!   ([`WfdaCostMeasure::PairwiseFit`]); [`WfdaCostMeasure::FdapaceLegacy`]
//!   reproduces fdapace.
//! - **Optimiser.** fdapace uses BOBYQA (`minqa`); fdars uses a box-projected
//!   Nelder–Mead with restarts from the same identity start, which reaches the
//!   same local minimum to optimiser tolerance.
//! - **Random numbers.** Partner subsets (`subset_prop < 1`) and random-search
//!   candidates come from Rust's `StdRng`, not R's Mersenne Twister, so they
//!   differ from fdapace for the same seed. With `subset_prop = 1` every other
//!   curve is a partner and the result does not depend on the generator.
//! - **Grids.** The piecewise-linear warp is evaluated on the work grid `w`
//!   (fdapace evaluates it at `M` equally spaced points); both coincide for
//!   equally spaced `argvals`.
//! - Zero pairwise costs under `Weighted` (infinite weights, `NaN` in R) give
//!   the plain mean over the zero-cost partners (the limit of the weighting).
//!
//! All warps are returned on the `argvals` scale, `t₀ + (T − t₀)·h(w)`.

use crate::error::FdarError;
use crate::helpers::{linear_interp, trapz};
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Box of the piecewise-linear knot heights (fdapace `lower`/`upper`).
const KNOT_LO: f64 = 1e-6;
const KNOT_HI: f64 = 1.0 - 1e-6;
/// Maximum number of Nelder–Mead restarts from the best point.
const MAX_RESTARTS: usize = 8;
/// Exponents `p` of the additional power-warp starts `u^p` of the knot search.
const START_POWERS: [f64; 4] = [0.5, 0.75, 1.5, 2.0];

/// How the pairwise warps of a curve are averaged into its inverse warp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WfdaChoice {
    /// Mean over the partners whose pairwise cost is at most the
    /// [`WfdaConfig::truncation_quantile`] (type-7) quantile of the costs
    /// (fdapace `"truncated"`, default).
    #[default]
    Truncated,
    /// Mean weighted by the inverse pairwise costs (fdapace `"weighted"`).
    Weighted,
}

/// Pairwise cost used to truncate or weight the pairwise warps (see the
/// module documentation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WfdaCostMeasure {
    /// `mean_j (ỹ_k(h_ik(w_j)) − ỹ_i(w_j))²`, the data-fit term of the pairwise
    /// objective divided by `m` (default).
    #[default]
    PairwiseFit,
    /// fdapace 0.6.0: `mean_j (ỹ_i(h_ik(w_j)) − ỹ_{k}(w_1))²` (the partner's
    /// value at the first grid point, from R's linear matrix indexing).
    FdapaceLegacy,
}

/// Configuration of [`pairwise_synchronization`] (fdapace `WFDA` `optns`).
///
/// Construct with `WfdaConfig::default()` and assign fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WfdaConfig {
    /// Averaging of the pairwise warps (default [`WfdaChoice::Truncated`]).
    pub choice: WfdaChoice,
    /// Proportion of the other `n − 1` curves used as partners, in `(0, 1]`;
    /// `K = round(subset_prop · (n − 1))` with round-half-even, as R (default 0.5).
    pub subset_prop: f64,
    /// Number of interior knots of the piecewise-linear warps, `1..=7`
    /// (default 2; ignored when `is_pwl` is `false`).
    pub nknots: usize,
    /// Piecewise-linear warps fitted by optimisation (`true`, default) or a
    /// random search over `K²` monotone candidates (`false`).
    pub is_pwl: bool,
    /// Seed: curve `i` draws its partners with `seed + i`; random-search
    /// candidate `u` uses `seed + n + u` (default 666).
    pub seed: u64,
    /// Warp penalty `λ ≥ 0`; `None` (default) uses fdapace's data-driven value.
    pub lambda: Option<f64>,
    /// Pairwise cost for truncation / weighting (default
    /// [`WfdaCostMeasure::PairwiseFit`]).
    pub cost_measure: WfdaCostMeasure,
    /// Quantile level of the truncation, in `(0, 1]` (default 0.9, as fdapace).
    pub truncation_quantile: f64,
    /// Maximum objective evaluations per pairwise fit (default 4000).
    pub max_evals: usize,
    /// Relative objective tolerance of the knot optimisation (default 1e-12).
    pub optim_tol: f64,
}

impl Default for WfdaConfig {
    fn default() -> Self {
        Self {
            choice: WfdaChoice::Truncated,
            subset_prop: 0.5,
            nknots: 2,
            is_pwl: true,
            seed: 666,
            lambda: None,
            cost_measure: WfdaCostMeasure::PairwiseFit,
            truncation_quantile: 0.9,
            max_evals: 4000,
            optim_tol: 1e-12,
        }
    }
}

impl WfdaConfig {
    /// Validate the parameters.
    ///
    /// # Errors
    ///
    /// [`FdarError::InvalidParameter`] if `subset_prop ∉ (0, 1]`,
    /// `nknots ∉ 1..=7` with `is_pwl`, `lambda` is negative or not finite,
    /// `truncation_quantile ∉ (0, 1]`, `max_evals == 0`, or `optim_tol` is
    /// negative or not finite.
    pub fn validate(&self) -> Result<(), FdarError> {
        let in_unit = |v: f64| v > 0.0 && v <= 1.0;
        let lambda_ok = self.lambda.map_or(true, |l| l.is_finite() && l >= 0.0);
        let checks: [(&'static str, bool, &str); 6] = [
            (
                "subset_prop",
                in_unit(self.subset_prop),
                "must be in (0, 1]",
            ),
            (
                "nknots",
                !self.is_pwl || (1..=7).contains(&self.nknots),
                "must be between 1 and 7",
            ),
            ("lambda", lambda_ok, "must be finite and non-negative"),
            (
                "truncation_quantile",
                in_unit(self.truncation_quantile),
                "must be in (0, 1]",
            ),
            ("max_evals", self.max_evals > 0, "must be positive"),
            (
                "optim_tol",
                self.optim_tol.is_finite() && self.optim_tol >= 0.0,
                "must be finite and non-negative",
            ),
        ];
        match checks.iter().find(|c| !c.1) {
            Some(&(parameter, _, message)) => Err(FdarError::InvalidParameter {
                parameter,
                message: message.to_string(),
            }),
            None => Ok(()),
        }
    }
}

/// Result of [`pairwise_synchronization`]. All warps are on the `argvals` scale.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WfdaResult {
    /// Warps `h_i` (n × m, row `i` = `h_i(argvals)`; fdapace `h`).
    pub warps: FdMatrix,
    /// Inverse warps `h_i⁻¹`, the averaged pairwise warps (fdapace `hInv`).
    pub inverse_warps: FdMatrix,
    /// Registered curves `y_i ∘ h_i` (fdapace `aligned`).
    pub aligned: FdMatrix,
    /// Mean pairwise cost per curve (fdapace `costs`).
    pub costs: Vec<f64>,
    /// Warp penalty used.
    pub lambda: f64,
    /// Partner curves of each curve, in draw order.
    pub partners: Vec<Vec<usize>>,
    /// Pairwise warps `h_ik` of each curve (`K × m`, rows in `partners` order).
    pub pairwise_warps: Vec<FdMatrix>,
    /// Pairwise costs of each curve, in `partners` order.
    pub pairwise_costs: Vec<Vec<f64>>,
}

/// Data shared by all curves.
struct Prepared<'a> {
    /// Work grid `w ∈ [0, 1]`.
    w: Vec<f64>,
    /// Amplitude-normalised curves `ỹ_i = y_i / max|y_i|`.
    ynorm: Vec<Vec<f64>>,
    /// Raw curves.
    raw: Vec<Vec<f64>>,
    argvals: &'a [f64],
    lambda: f64,
    /// Number of partners per curve.
    k: usize,
    /// Knot positions of the piecewise-linear warps.
    knots: Vec<f64>,
    /// Random-search candidate warps (empty for piecewise-linear warps).
    candidates: Vec<Vec<f64>>,
}

/// Per-curve output.
struct CurveOut {
    h: Vec<f64>,
    hinv: Vec<f64>,
    aligned: Vec<f64>,
    partners: Vec<usize>,
    pairwise: Vec<Vec<f64>>,
    costs: Vec<f64>,
}

fn validate_inputs(data: &FdMatrix, argvals: &[f64]) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 curves".to_string(),
            actual: format!("{n} curves"),
        });
    }
    if m < 3 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 3 evaluation points".to_string(),
            actual: format!("{m} points"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} values (data columns)"),
            actual: format!("{} values", argvals.len()),
        });
    }
    if argvals.iter().any(|v| !v.is_finite()) || argvals.windows(2).any(|p| p[1] <= p[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be finite and strictly increasing".to_string(),
        });
    }
    if data.as_slice().iter().any(|v| !v.is_finite()) {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: "must be finite (dense data without missing values)".to_string(),
        });
    }
    Ok(())
}

/// Divide every curve by its maximum absolute value.
fn normalise_rows(raw: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FdarError> {
    raw.iter()
        .enumerate()
        .map(|(i, row)| {
            let mx = row.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
            if mx > 0.0 {
                Ok(row.iter().map(|v| v / mx).collect())
            } else {
                Err(FdarError::InvalidParameter {
                    parameter: "data",
                    message: format!("curve {i} is identically zero"),
                })
            }
        })
        .collect()
}

/// fdapace's default penalty `10⁻⁴ · √(Σᵢ ∫(ỹᵢ − μ̃)² / (n − 1))`.
fn default_lambda(ynorm: &[Vec<f64>], argvals: &[f64]) -> f64 {
    let n = ynorm.len();
    let m = argvals.len();
    let mu: Vec<f64> = (0..m)
        .map(|j| ynorm.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let total: f64 = ynorm
        .iter()
        .map(|r| {
            let sq: Vec<f64> = r.iter().zip(&mu).map(|(a, b)| (a - b) * (a - b)).collect();
            trapz(&sq, argvals)
        })
        .sum();
    (total / (n - 1) as f64).sqrt() * 1e-4
}

/// Piecewise-linear warp with knot heights `(0, sort(x), 1)` at `knots`,
/// evaluated on `w`.
fn pwl_warp(knots: &[f64], x: &[f64], w: &[f64]) -> Vec<f64> {
    let mut heights = Vec::with_capacity(x.len() + 2);
    heights.push(0.0);
    heights.extend_from_slice(x);
    heights[1..].sort_by(f64::total_cmp);
    heights.push(1.0);
    w.iter()
        .map(|&t| linear_interp(knots, &heights, t))
        .collect()
}

/// Pairwise objective `Σ (ỹ_k(h) − ỹ_i)² + λ Σ (h − w)²`.
fn warp_cost(yi: &[f64], yk: &[f64], w: &[f64], h: &[f64], lambda: f64) -> f64 {
    let mut fit = 0.0;
    let mut pen = 0.0;
    for j in 0..w.len() {
        let r = linear_interp(w, yk, h[j]) - yi[j];
        fit += r * r;
        let d = h[j] - w[j];
        pen += d * d;
    }
    fit + lambda * pen
}

/// Box-projected Nelder–Mead. Returns the best point and value.
struct NelderMead<'f> {
    f: &'f dyn Fn(&[f64]) -> f64,
    lo: f64,
    hi: f64,
    tol: f64,
    evals: usize,
    max_evals: usize,
}

impl NelderMead<'_> {
    fn eval(&mut self, x: &mut [f64]) -> f64 {
        for v in x.iter_mut() {
            *v = v.clamp(self.lo, self.hi);
        }
        self.evals += 1;
        (self.f)(x)
    }

    fn initial_simplex(&mut self, x0: &[f64], step: f64) -> (Vec<Vec<f64>>, Vec<f64>) {
        let d = x0.len();
        let mut pts = vec![x0.to_vec()];
        for i in 0..d {
            let mut p = x0.to_vec();
            // Step inwards when the outward step would leave the box.
            p[i] = if p[i] + step <= self.hi {
                p[i] + step
            } else {
                p[i] - step
            };
            pts.push(p);
        }
        let fv = pts.iter_mut().map(|p| self.eval(p)).collect();
        (pts, fv)
    }

    fn converged(&self, pts: &[Vec<f64>], fv: &[f64]) -> bool {
        let spread = fv[fv.len() - 1] - fv[0];
        let size = pts[1..]
            .iter()
            .flat_map(|p| p.iter().zip(&pts[0]).map(|(a, b)| (a - b).abs()))
            .fold(0.0_f64, f64::max);
        spread <= self.tol * (fv[0].abs() + 1e-300) && size <= 1e-10
    }

    /// One run from `x0`; returns the best point and value.
    fn run(&mut self, x0: &[f64], step: f64) -> (Vec<f64>, f64) {
        let d = x0.len();
        let (mut pts, mut fv) = self.initial_simplex(x0, step);
        loop {
            sort_simplex(&mut pts, &mut fv);
            if self.converged(&pts, &fv) || self.evals >= self.max_evals {
                return (pts[0].clone(), fv[0]);
            }
            let centroid: Vec<f64> = (0..d)
                .map(|c| pts[..d].iter().map(|p| p[c]).sum::<f64>() / d as f64)
                .collect();
            self.step(&mut pts, &mut fv, &centroid);
        }
    }

    fn along(centroid: &[f64], worst: &[f64], coef: f64) -> Vec<f64> {
        centroid
            .iter()
            .zip(worst)
            .map(|(c, w)| c + coef * (c - w))
            .collect()
    }

    fn step(&mut self, pts: &mut [Vec<f64>], fv: &mut [f64], centroid: &[f64]) {
        let d = centroid.len();
        let mut xr = Self::along(centroid, &pts[d], 1.0);
        let fr = self.eval(&mut xr);
        if fr < fv[0] {
            let mut xe = Self::along(centroid, &pts[d], 2.0);
            let fe = self.eval(&mut xe);
            (pts[d], fv[d]) = if fe < fr { (xe, fe) } else { (xr, fr) };
            return;
        }
        if fr < fv[d - 1] {
            (pts[d], fv[d]) = (xr, fr);
            return;
        }
        let coef = if fr < fv[d] { 0.5 } else { -0.5 };
        let mut xc = Self::along(centroid, &pts[d], coef);
        let fc = self.eval(&mut xc);
        if fc < fv[d].min(fr) {
            (pts[d], fv[d]) = (xc, fc);
            return;
        }
        self.shrink(pts, fv);
    }

    fn shrink(&mut self, pts: &mut [Vec<f64>], fv: &mut [f64]) {
        let best = pts[0].clone();
        for (p, f) in pts.iter_mut().zip(fv.iter_mut()).skip(1) {
            for (v, b) in p.iter_mut().zip(&best) {
                *v = b + 0.5 * (*v - b);
            }
            *f = self.eval(p);
        }
    }
}

fn sort_simplex(pts: &mut [Vec<f64>], fv: &mut [f64]) {
    let mut idx: Vec<usize> = (0..fv.len()).collect();
    idx.sort_by(|&a, &b| fv[a].total_cmp(&fv[b]));
    let p: Vec<Vec<f64>> = idx.iter().map(|&i| pts[i].clone()).collect();
    let f: Vec<f64> = idx.iter().map(|&i| fv[i]).collect();
    pts.clone_from_slice(&p);
    fv.copy_from_slice(&f);
}

/// Nelder–Mead from `x0`, restarted from the best point until it stops
/// improving.
fn minimise_from(nm: &mut NelderMead, x0: &[f64]) -> (Vec<f64>, f64) {
    nm.evals = 0;
    let (mut best, mut fbest) = nm.run(x0, 0.1);
    for _ in 0..MAX_RESTARTS {
        if nm.evals >= nm.max_evals {
            break;
        }
        let (x, f) = nm.run(&best, 0.02);
        let improved = f < fbest - nm.tol * fbest.abs();
        if f < fbest {
            (best, fbest) = (x, f);
        }
        if !improved {
            break;
        }
    }
    (best, fbest)
}

/// Piecewise-linear pairwise warp minimising [`warp_cost`].
///
/// Multi-start: the identity knot heights (fdapace's start) and the power
/// warps `u^p`, `p ∈ START_POWERS`, at the knots; the lowest objective wins and
/// ties keep the identity start.
fn fit_pwl_pair(prep: &Prepared, i: usize, c: usize, config: &WfdaConfig) -> Vec<f64> {
    let (yi, yk) = (&prep.ynorm[i], &prep.ynorm[c]);
    let objective = |x: &[f64]| {
        let h = pwl_warp(&prep.knots, x, &prep.w);
        warp_cost(yi, yk, &prep.w, &h, prep.lambda)
    };
    let mut nm = NelderMead {
        f: &objective,
        lo: KNOT_LO,
        hi: KNOT_HI,
        tol: config.optim_tol,
        evals: 0,
        max_evals: config.max_evals,
    };
    let interior = &prep.knots[1..prep.knots.len() - 1];
    let (mut best, mut fbest) = minimise_from(&mut nm, interior);
    for p in START_POWERS {
        let x0: Vec<f64> = interior.iter().map(|u| u.powf(p)).collect();
        let (x, f) = minimise_from(&mut nm, &x0);
        if f < fbest - config.optim_tol * fbest.abs() {
            (best, fbest) = (x, f);
        }
    }
    pwl_warp(&prep.knots, &best, &prep.w)
}

/// `count` random monotone candidate warps `(0, sort(U(0,1)^{m−2}), 1)`.
fn random_candidates(m: usize, count: usize, seed: u64) -> Vec<Vec<f64>> {
    (0..count)
        .map(|u| {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(u as u64));
            let mut h: Vec<f64> = (0..m - 2).map(|_| rng.gen::<f64>()).collect();
            h.sort_by(f64::total_cmp);
            h.insert(0, 0.0);
            h.push(1.0);
            h
        })
        .collect()
}

/// Random-search pairwise warp: the candidate of minimum [`warp_cost`].
fn fit_random_pair(prep: &Prepared, i: usize, c: usize) -> Vec<f64> {
    let (yi, yk) = (&prep.ynorm[i], &prep.ynorm[c]);
    prep.candidates
        .iter()
        .map(|h| (warp_cost(yi, yk, &prep.w, h, prep.lambda), h))
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, h)| h.clone())
        .unwrap_or_else(|| prep.w.clone())
}

/// Pairwise cost of `h` (warp of curve `i` towards partner `c`).
fn pair_distance(measure: WfdaCostMeasure, prep: &Prepared, i: usize, c: usize, h: &[f64]) -> f64 {
    let w = &prep.w;
    let m = w.len() as f64;
    match measure {
        WfdaCostMeasure::PairwiseFit => {
            let (yi, yk) = (&prep.ynorm[i], &prep.ynorm[c]);
            h.iter()
                .zip(yi)
                .map(|(&hj, &v)| (linear_interp(w, yk, hj) - v).powi(2))
                .sum::<f64>()
                / m
        }
        WfdaCostMeasure::FdapaceLegacy => {
            let target = prep.ynorm[c][0];
            let yi = &prep.ynorm[i];
            h.iter()
                .map(|&hj| (linear_interp(w, yi, hj) - target).powi(2))
                .sum::<f64>()
                / m
        }
    }
}

/// R `quantile(x, p, type = 7)`.
fn quantile_type7(values: &[f64], p: f64) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let h = (v.len() - 1) as f64 * p;
    let lo = (h + 4.0 * f64::EPSILON).floor() as usize;
    let hi = (lo + 1).min(v.len() - 1);
    let frac = h - lo as f64;
    if frac <= 0.0 || v[hi] == v[lo] {
        v[lo]
    } else {
        (1.0 - frac) * v[lo] + frac * v[hi]
    }
}

/// Average of the pairwise warps with the given (non-negative) weights.
fn weighted_average(warps: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    let m = warps[0].len();
    (0..m)
        .map(|j| {
            warps
                .iter()
                .zip(weights)
                .map(|(h, &wt)| h[j] * wt)
                .sum::<f64>()
                / total
        })
        .collect()
}

/// Inverse warp: truncated or inverse-cost-weighted mean of the pairwise warps.
fn average_warps(warps: &[Vec<f64>], costs: &[f64], config: &WfdaConfig) -> Vec<f64> {
    let weights: Vec<f64> = match config.choice {
        WfdaChoice::Truncated => {
            let q = quantile_type7(costs, config.truncation_quantile);
            costs.iter().map(|&d| f64::from(u8::from(d <= q))).collect()
        }
        WfdaChoice::Weighted if costs.contains(&0.0) => costs
            .iter()
            .map(|&d| f64::from(u8::from(d == 0.0)))
            .collect(),
        WfdaChoice::Weighted => costs.iter().map(|&d| 1.0 / d).collect(),
    };
    weighted_average(warps, &weights)
}

/// Invert a monotone warp on the grid: R `approx(x = hinv, y = w, xout = w)`
/// with `ties = mean`.
fn invert_on_grid(w: &[f64], hinv: &[f64]) -> Vec<f64> {
    let mut xs: Vec<f64> = Vec::with_capacity(w.len());
    let mut ys: Vec<f64> = Vec::with_capacity(w.len());
    let mut j = 0;
    while j < w.len() {
        let mut k = j;
        while k + 1 < w.len() && hinv[k + 1] == hinv[j] {
            k += 1;
        }
        xs.push(hinv[j]);
        ys.push(w[j..=k].iter().sum::<f64>() / (k - j + 1) as f64);
        j = k + 1;
    }
    let mut h: Vec<f64> = w.iter().map(|&t| linear_interp(&xs, &ys, t)).collect();
    let last = w.len() - 1;
    h[0] = w[0];
    h[last] = w[last];
    h
}

/// Draw `k` distinct partners of curve `i` among the other `n − 1` curves.
fn sample_partners(n: usize, i: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(i as u64));
    rand::seq::index::sample(&mut rng, n - 1, k)
        .into_iter()
        .map(|j| if j >= i { j + 1 } else { j })
        .collect()
}

fn synchronise_curve(i: usize, prep: &Prepared, config: &WfdaConfig) -> CurveOut {
    let n = prep.ynorm.len();
    let partners = sample_partners(n, i, prep.k, config.seed);
    let pairwise: Vec<Vec<f64>> = partners
        .iter()
        .map(|&c| {
            if config.is_pwl {
                fit_pwl_pair(prep, i, c, config)
            } else {
                fit_random_pair(prep, i, c)
            }
        })
        .collect();
    let costs: Vec<f64> = partners
        .iter()
        .zip(&pairwise)
        .map(|(&c, h)| pair_distance(config.cost_measure, prep, i, c, h))
        .collect();
    let hinv = average_warps(&pairwise, &costs, config);
    let h = invert_on_grid(&prep.w, &hinv);
    let aligned = h
        .iter()
        .map(|&hj| linear_interp(&prep.w, &prep.raw[i], hj))
        .collect();
    CurveOut {
        h,
        hinv,
        aligned,
        partners,
        pairwise,
        costs,
    }
}

fn prepare<'a>(
    data: &FdMatrix,
    argvals: &'a [f64],
    config: &WfdaConfig,
) -> Result<Prepared<'a>, FdarError> {
    let (n, m) = data.shape();
    let k = (config.subset_prop * (n - 1) as f64).round_ties_even() as usize;
    if k == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "subset_prop",
            message: format!("round(subset_prop * (n - 1)) is 0 for n = {n}; use a larger value"),
        });
    }
    let (t0, t1) = (argvals[0], argvals[m - 1]);
    let w: Vec<f64> = argvals.iter().map(|&t| (t - t0) / (t1 - t0)).collect();
    let raw = data.rows();
    let ynorm = normalise_rows(&raw)?;
    let lambda = config
        .lambda
        .unwrap_or_else(|| default_lambda(&ynorm, argvals));
    let knots: Vec<f64> = (0..config.nknots + 2)
        .map(|j| j as f64 / (config.nknots + 1) as f64)
        .collect();
    let candidates = if config.is_pwl {
        Vec::new()
    } else {
        random_candidates(m, k * k, config.seed.wrapping_add(n as u64))
    };
    Ok(Prepared {
        w,
        ynorm,
        raw,
        argvals,
        lambda,
        k,
        knots,
        candidates,
    })
}

fn rows_to_matrix<'r>(rows: impl Iterator<Item = &'r Vec<f64>>, m: usize) -> FdMatrix {
    let rows: Vec<&Vec<f64>> = rows.collect();
    let mut out = FdMatrix::zeros(rows.len(), m);
    for (i, row) in rows.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[(i, j)] = v;
        }
    }
    out
}

/// Map a `[0, 1]` work-scale warp to the `argvals` scale.
fn to_argvals_scale(h: &[f64], argvals: &[f64]) -> Vec<f64> {
    let (t0, t1) = (argvals[0], argvals[argvals.len() - 1]);
    h.iter().map(|&v| t0 + (t1 - t0) * v).collect()
}

/// Tang–Müller pairwise curve synchronisation (R fdapace `WFDA`).
///
/// For every curve, fits monotone pairwise warps to `K = round(subset_prop ·
/// (n − 1))` partner curves under the penalty `λ`, averages them into the
/// curve's inverse warp to the common time scale (truncated or weighted mean,
/// [`WfdaChoice`]), inverts it and registers the curve. See the
/// module documentation for the algorithm and its correspondence with
/// fdapace 0.6.0.
///
/// # Arguments
///
/// * `data` — n × m dense curves on the common grid (rows = curves).
/// * `argvals` — the m strictly increasing evaluation points.
/// * `config` — options ([`WfdaConfig`]).
///
/// # Errors
///
/// [`FdarError::InvalidDimension`] for fewer than 2 curves or 3 points or a
/// mismatched `argvals`; [`FdarError::InvalidParameter`] for invalid options,
/// non-finite data, non-increasing `argvals`, an identically zero curve, or a
/// `subset_prop` that selects no partner.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{pairwise_synchronization, WfdaConfig};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 41;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// let shifts = [-0.3, -0.1, 0.0, 0.1, 0.3];
/// let mut data = FdMatrix::zeros(shifts.len(), m);
/// for (i, a) in shifts.iter().enumerate() {
///     for (j, &tj) in t.iter().enumerate() {
///         let s = tj + a * tj * (1.0 - tj); // warped time
///         data[(i, j)] = 0.2 + (-((s - 0.5) / 0.1_f64).powi(2)).exp();
///     }
/// }
/// let mut config = WfdaConfig::default();
/// config.subset_prop = 1.0;
/// let res = pairwise_synchronization(&data, &t, &config).unwrap();
/// assert_eq!(res.warps.shape(), (5, m));
/// // Warps are monotone and keep the end points.
/// for i in 0..5 {
///     let h = res.warps.row(i);
///     assert!(h.windows(2).all(|p| p[1] >= p[0]));
///     assert_eq!((h[0], h[m - 1]), (0.0, 1.0));
/// }
/// ```
#[must_use = "pairwise synchronisation is expensive; use the returned warps"]
pub fn pairwise_synchronization(
    data: &FdMatrix,
    argvals: &[f64],
    config: &WfdaConfig,
) -> Result<WfdaResult, FdarError> {
    config.validate()?;
    validate_inputs(data, argvals)?;
    let prep = prepare(data, argvals, config)?;
    let n = data.nrows();
    let m = argvals.len();
    let outs: Vec<CurveOut> = iter_maybe_parallel!(0..n)
        .map(|i| synchronise_curve(i, &prep, config))
        .collect();
    let scale = |h: &Vec<f64>| to_argvals_scale(h, prep.argvals);
    let warps: Vec<Vec<f64>> = outs.iter().map(|o| scale(&o.h)).collect();
    let inverse: Vec<Vec<f64>> = outs.iter().map(|o| scale(&o.hinv)).collect();
    let pairwise_warps = outs
        .iter()
        .map(|o| {
            let rows: Vec<Vec<f64>> = o.pairwise.iter().map(scale).collect();
            rows_to_matrix(rows.iter(), m)
        })
        .collect();
    Ok(WfdaResult {
        warps: rows_to_matrix(warps.iter(), m),
        inverse_warps: rows_to_matrix(inverse.iter(), m),
        aligned: rows_to_matrix(outs.iter().map(|o| &o.aligned), m),
        costs: outs
            .iter()
            .map(|o| o.costs.iter().sum::<f64>() / o.costs.len() as f64)
            .collect(),
        lambda: prep.lambda,
        partners: outs.iter().map(|o| o.partners.clone()).collect(),
        pairwise_warps,
        pairwise_costs: outs.into_iter().map(|o| o.costs).collect(),
    })
}

#[cfg(test)]
#[path = "wfda_golden.rs"]
mod golden;

#[cfg(test)]
#[path = "wfda_tests.rs"]
mod tests;
