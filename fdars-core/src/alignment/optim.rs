//! Warp-optimiser selection (DP / Riemannian L-BFGS) and in-objective warp
//! penalties for elastic alignment.
//!
//! [`WarpOptimConfig`] is the option bag behind the `*_with_config` entry points
//! ([`elastic_align_pair_with_config`](super::elastic_align_pair_with_config),
//! [`elastic_distance_with_config`](super::elastic_distance_with_config),
//! [`elastic_self_distance_matrix_with_config`](super::elastic_self_distance_matrix_with_config),
//! [`elastic_cross_distance_matrix_with_config`](super::elastic_cross_distance_matrix_with_config),
//! [`karcher_mean_with_config`](super::karcher_mean_with_config), the closed,
//! R^d, landmark-constrained, robust-Karcher and `align_to_target` variants
//! such as [`elastic_align_pair_nd_with_config`](super::elastic_align_pair_nd_with_config)
//! and [`elastic_align_pair_constrained_with_config`](super::elastic_align_pair_constrained_with_config)).
//! Its default (`Dp` + `L2Gam`, no band, native λ) reproduces the classic
//! functions bit-for-bit.

use super::rbfgs::{rbfgs_warp, RbfgsParams};
use super::{
    band_radius, coprime_nbhd, dp_alignment_core_opts, dp_l2psi_penalty, dp_lambda_penalty,
    dp_prune_is_exact, refine_srsf_linear, DpEngineOptions, DpMoves,
};
use crate::error::FdarError;

/// Which optimiser computes the optimal warp γ (fdasrvf `optim_method` /
/// fdasrsf `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarpOptimMethod {
    /// Exact dynamic programming over a coprime move neighbourhood (by default
    /// the classic 35-move `nbhd_dim = 7` engine; default and exactness
    /// reference). Tunable via [`WarpOptimConfig::nbhd_dim`],
    /// [`dp_scl`](WarpOptimConfig::dp_scl), [`dp_cost`](WarpOptimConfig::dp_cost)
    /// and [`dp_prune`](WarpOptimConfig::dp_prune).
    #[default]
    Dp,
    /// Riemannian limited-memory BFGS over `l = √γ̇` on the unit Hilbert
    /// sphere, started from a DP warp on a coarse (≤ 51 point) grid and refined
    /// on the SRSFs resampled `scl` times finer — port of fdasrsf 2.7.5
    /// `optimum_reparam(method="RBFGS")` (Huang, Gallivan, Srivastava & Absil).
    ///
    /// A local optimiser: it typically lowers the alignment cost below DP's
    /// piecewise-linear warp and yields smooth warps, but it may settle in a
    /// different local minimum than the global DP.
    Rbfgs,
}

/// Penalty on the warp applied **inside** the alignment objective, weighted by
/// [`WarpOptimConfig::lambda`] (fdasrvf/fdasrsf `penalty` /
/// `penalty_method`). With `l = √γ̇`:
///
/// | variant | penalty | DP | RBFGS |
/// |---|---|---|---|
/// | `L2Gam` | `∫ (γ̇ − 1)²` | yes (classic first-order) | yes |
/// | `L2Psi` | `∫ (√γ̇ − 1)²` | yes | yes |
/// | `Roughness` | `∫ γ̈²` | no | yes |
/// | `Geodesic` | `arccos(∫ √γ̇)²` | no | yes |
///
/// A DP can only express penalties that are integrals of a pointwise function
/// of γ̇ (they are summed edge by edge along a piecewise-linear path);
/// `Roughness` concentrates on the knots and `Geodesic` is a nonlinear function
/// of a global integral, so with `lambda > 0` they require
/// [`WarpOptimMethod::Rbfgs`] (as in fdasrsf, where DP raises a `ValueError`).
///
/// Scale of `lambda`: for DP, `L2Gam` is exactly the classic
/// `λ·(slope − 1)²·dt` edge penalty of [`elastic_align_pair`](super::elastic_align_pair)
/// (in `argvals` units, on unit-Euclidean-norm SRSFs) and `L2Psi` uses the same
/// convention; for RBFGS `lambda` has fdasrsf's `optimum_reparam(method="RBFGS",
/// lam=…)` semantics. For 1-D curves the two differ by a data-dependent factor
/// `φ = 1 + O(1/m)` (see [`WarpLambdaScale`] for the derivation); for R^d curve
/// shapes by `≈ 1/(m·‖q1‖‖q2‖)`. Set [`WarpOptimConfig::lambda_scale`] to make
/// one `λ` mean the same penalty strength under both optimisers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarpObjectivePenalty {
    /// `∫ (γ̇ − 1)²` — the classic fdars first-order penalty (default).
    #[default]
    L2Gam,
    /// `∫ (√γ̇ − 1)²` — fdasrsf's default penalty.
    L2Psi,
    /// `∫ γ̈²` — in-objective curvature (second-order) penalty; RBFGS only.
    Roughness,
    /// Squared geodesic (Fisher–Rao phase) distance of γ from the identity,
    /// `arccos(∫ √γ̇)²`; RBFGS only.
    Geodesic,
}

/// Edge-cost model of the dynamic-programming optimiser
/// ([`WarpOptimConfig::dp_cost`]).
///
/// Both integrate `(q1 − √γ'·q2∘γ)²` along each piecewise-linear DP edge by
/// walking the merged breakpoints of the two SRSF grids; they differ in how
/// the SRSFs are taken between their samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DpEdgeCost {
    /// SRSFs constant on each grid interval (left sample) — the classic fdars /
    /// pre-2.5 fdasrvf `dp_edge_weight` (default).
    #[default]
    PiecewiseConstant,
    /// SRSFs linear between samples, squared residual integrated exactly
    /// (`dt·(f0² + f0·f1 + f1²)/3` per sub-interval) — fdasrsf ≥ 2.7.4 `DP2`
    /// (`DynamicProgrammingQ2`). Exact on uniform grids; on a non-uniform grid
    /// the sub-interval split uses the same index-space approximation as the
    /// piecewise-constant model.
    PiecewiseLinear,
}

/// Which optimiser's convention [`WarpOptimConfig::lambda`] is expressed in
/// ([`WarpOptimConfig::lambda_scale`]).
///
/// # The two conventions
///
/// With `l = √γ̇`, `s ∈ [0, 1]` the normalised time and `P(γ) = ∫₀¹ pen(γ̇) ds`
/// the `L2Gam` (`(γ̇ − 1)²`) or `L2Psi` (`(√γ̇ − 1)²`) penalty:
///
/// * **DP** (fdars) aligns the SRSFs divided by their Euclidean norms
///   `E_i = ‖q_i‖₂` over the `m` input samples, with the edge costs integrated
///   in `argvals` units (span `S`):
///   `E_dp(γ) = S·[∫₀¹ (q1/E1 − (q2∘γ)·l/E2)² ds + λ_dp·P(γ)]`.
/// * **RBFGS** (fdasrsf 2.7.5) resamples the SRSFs `scl` times finer
///   (Euclidean norms `e_i` over the refined samples, `L²` norms `n_i`),
///   divides them by `n_i` and weighs the penalty by fdasrsf's
///   `lamScale = e1·e2/(n1·n2)` and `1/scl`:
///   `E_rb(γ) = ∫₀¹ (q1/n1 − (q2∘γ)·l/n2)² ds + (λ_rb/scl)·e1e2/(n1n2)·P(γ)`.
///
/// Because `γ ↦ (q2∘γ)·l` is an isometry, `∫(a·x − b·y)² = (a − b)² + a·b·∫(x − y)²`
/// for unit-norm `x`, `y`; with `a = n1/E1`, `b = n2/E2`:
///
/// `E_dp(γ) = S·[(a − b)² + a·b·(D_rb(γ) + λ_dp/(a·b)·P(γ))]`,
///
/// so DP and RBFGS minimise the same functional exactly when
///
/// `λ_rb = φ·λ_dp`, `φ = scl·E1·E2 / (e1·e2)`
///
/// ([`WarpOptimConfig::rbfgs_lambda_factor`]). As `e_i² ≈ scl·E_i²`, `φ = 1 +
/// O(1/m)`: the conventions nearly coincide for 1-D curves, and `Native`
/// differs from the converted scales only at the discretisation level. The
/// R^d curve-shape DP works on the raw (unnormalised) SRVFs instead, giving
/// `φ = scl/(e1·e2)`; the R^d shared-warp DP normalises every coordinate
/// separately, for which `φ = scl·E1·E2/(d·e1·e2)` (total norms) is exact when
/// the coordinates have equal norms. `Roughness` / `Geodesic` exist under
/// RBFGS only and are never converted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarpLambdaScale {
    /// Each optimiser reads `lambda` in its own convention (default; the
    /// behaviour of every earlier release).
    #[default]
    Native,
    /// `lambda` is in the DP (fdars) convention; RBFGS uses `φ·lambda`.
    Dp,
    /// `lambda` is in fdasrsf's RBFGS convention; DP uses `lambda/φ`.
    Fdasrsf,
}

/// Options for warp optimisation in elastic alignment.
///
/// Sealed (`#[non_exhaustive]`): construct with [`WarpOptimConfig::default`]
/// and assign fields. The default (`Dp`, `lambda = 0`, `L2Gam`) reproduces
/// [`elastic_align_pair`](super::elastic_align_pair) and friends bit-for-bit;
/// `Dp` + `L2Gam` with any `lambda` equals the classic functions called with
/// that `lambda`.
///
/// # Examples
///
/// ```
/// use fdars_core::alignment::{WarpObjectivePenalty, WarpOptimConfig, WarpOptimMethod};
///
/// let mut cfg = WarpOptimConfig::default();
/// cfg.method = WarpOptimMethod::Rbfgs;
/// cfg.penalty = WarpObjectivePenalty::Roughness;
/// cfg.lambda = 0.01;
/// assert!(cfg.validate().is_ok());
/// ```
///
/// A more accurate dynamic program — fdasrsf `DP2`'s 63-move `grid_dim = 10`
/// neighbourhood with the exact piecewise-linear cost (exact pruning is on by
/// default):
///
/// ```
/// use fdars_core::alignment::{elastic_align_pair_with_config, DpEdgeCost, WarpOptimConfig};
///
/// let t: Vec<f64> = (0..61).map(|i| i as f64 / 60.0).collect();
/// let f1: Vec<f64> = t.iter().map(|&s| (-(s - 0.4).powi(2) / 0.01).exp()).collect();
/// let f2: Vec<f64> = t.iter().map(|&s| (-(s - 0.55).powi(2) / 0.01).exp()).collect();
///
/// let mut cfg = WarpOptimConfig::default();
/// cfg.nbhd_dim = 10;
/// cfg.dp_cost = DpEdgeCost::PiecewiseLinear;
/// assert!(cfg.dp_prune); // exact pruning is on by default
/// let fine = elastic_align_pair_with_config(&f1, &f2, &t, &cfg).unwrap();
/// let classic = elastic_align_pair_with_config(&f1, &f2, &t, &WarpOptimConfig::default()).unwrap();
/// assert!(fine.gamma.windows(2).all(|w| w[1] >= w[0]));
/// assert!(fine.distance.is_finite() && classic.distance.is_finite());
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct WarpOptimConfig {
    /// Warp optimiser (default [`WarpOptimMethod::Dp`]).
    pub method: WarpOptimMethod,
    /// Penalty weight `λ ≥ 0` (default `0.0`, no penalty).
    pub lambda: f64,
    /// In-objective penalty (default [`WarpObjectivePenalty::L2Gam`]).
    pub penalty: WarpObjectivePenalty,
    /// RBFGS iteration cap (default `30`, fdasrsf's value). Ignored by DP.
    pub max_iter: usize,
    /// RBFGS fine-grid refinement factor: the SRSFs are linearly resampled
    /// `scl` times finer to evaluate the cost (default `3`, fdasrsf's value;
    /// must be ≥ 1). Ignored by DP (see [`dp_scl`](Self::dp_scl)).
    pub scl: usize,
    /// DP neighbourhood size (fdasrvf `nbhd_dim`, fdasrsf `grid_dim`): the DP
    /// moves are all `(dr, dc)` with `1 ≤ dr, dc ≤ nbhd_dim`, `gcd(dr, dc) = 1`,
    /// so warp slopes range over `{dr/dc}`. Default `7` (35 moves, the classic
    /// engine); fdasrsf `DP2` uses `10` (63 moves). Cost grows with the number
    /// of moves (`≈ 0.61·nbhd_dim²`) and the edge length; a larger
    /// neighbourhood contains the smaller one's moves, so the DP optimum of
    /// its own objective can only decrease. Must be in `1..=64`; `1` allows
    /// only the identity warp. DP only.
    pub nbhd_dim: usize,
    /// DP cost refinement (fdasrsf ≥ 2.7.4 `DP` `scl`): the SRSFs are linearly
    /// resampled `dp_scl` times finer to evaluate every edge cost, while the
    /// warp path stays on the input grid. Default `1` (off, classic). Improves
    /// the piecewise-constant cost towards the exact integral of the linearly
    /// interpolated SRSFs at `O(dp_scl)` cost per edge; the cost keeps its
    /// integral scale, so `lambda` means the same for every `dp_scl`. Has no
    /// effect beyond rounding with [`DpEdgeCost::PiecewiseLinear`] (linear
    /// resampling of a linear interpolant is exact). Must be in `1..=64`. DP only.
    pub dp_scl: usize,
    /// DP edge-cost model (default [`DpEdgeCost::PiecewiseConstant`]). DP only.
    pub dp_cost: DpEdgeCost,
    /// DP candidate pruning (fdasrsf ≥ 2.7.4): skip a predecessor whose
    /// accumulated cost already reaches the target cell's best, without
    /// evaluating its edge. Exact — bit-identical warps — because every edge
    /// cost is non-negative under a validated config (`lambda ≥ 0`, strictly
    /// increasing `argvals`); the engine re-checks both and never prunes
    /// otherwise. Default `true` (≈ 1.6× faster DP; the classic functions
    /// such as [`elastic_align_pair`](super::elastic_align_pair) and
    /// [`karcher_mean`](super::karcher_mean) prune under the same condition).
    /// Set `false` to run the unpruned reference fill. DP only (and the RBFGS
    /// coarse-DP seed).
    pub dp_prune: bool,
    /// Sakoe–Chiba band as a fraction of the grid (default `0.0` = off; values
    /// `≥ 1` are also off), applied by every `*_with_config` entry point —
    /// the same convention as [`karcher_mean_banded`](super::karcher_mean_banded):
    /// the index radius is `R = ⌈band_frac·m⌉` (≥ 1), i.e. `|γ(t) − t| ≤ R·h`
    /// on a uniform grid of spacing `h`.
    ///
    /// * **DP** confines the path to the corridor `|row − col| ≤ R` (exact
    ///   within the band, ≈ `O(m·R)` instead of `O(m²)`).
    /// * **RBFGS** starts from the banded DP on the input grid (with the DP
    ///   options; `O(m·R)`, replacing the unbanded coarse start), runs the
    ///   unconstrained Riemannian L-BFGS, then *projects* the warp onto the
    ///   band by the pointwise clamp `γ ← min(max(γ, t − b), t + b)`,
    ///   `b = R/(m − 1)` of the normalised domain (a clamp between two
    ///   increasing bounds keeps γ monotone and boundary-exact), and returns
    ///   the better — in the RBFGS objective — of the projected RBFGS warp and
    ///   the (projected) DP seed. The result therefore always satisfies the
    ///   band and is never worse than the banded DP. Iterates are not
    ///   projected (the sphere geometry has no cheap feasible-set
    ///   projection); on a non-uniform grid `b` is in normalised time while
    ///   the DP band is in grid indices.
    ///
    /// Not supported by the landmark-constrained or curve-shape entry points
    /// (they reject `band_frac > 0` with [`FdarError::InvalidParameter`]).
    pub band_frac: f64,
    /// Convention of [`lambda`](Self::lambda) (default
    /// [`WarpLambdaScale::Native`], unchanged behaviour); see
    /// [`WarpLambdaScale`] for the exact DP ↔ RBFGS conversion.
    pub lambda_scale: WarpLambdaScale,
}

impl Default for WarpOptimConfig {
    fn default() -> Self {
        Self {
            method: WarpOptimMethod::Dp,
            lambda: 0.0,
            penalty: WarpObjectivePenalty::L2Gam,
            max_iter: 30,
            scl: 3,
            nbhd_dim: 7,
            dp_scl: 1,
            dp_cost: DpEdgeCost::PiecewiseConstant,
            dp_prune: true,
            band_frac: 0.0,
            lambda_scale: WarpLambdaScale::Native,
        }
    }
}

impl WarpOptimConfig {
    /// Check the configuration.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `lambda` or `band_frac` is negative
    /// or non-finite, `scl == 0`, `nbhd_dim` or `dp_scl` is outside `1..=64`,
    /// or a `Roughness`/`Geodesic` penalty with `lambda > 0` is requested from
    /// the DP optimiser (it cannot express them).
    pub fn validate(&self) -> Result<(), FdarError> {
        if !self.lambda.is_finite() || self.lambda < 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "lambda",
                message: format!("must be finite and >= 0, got {}", self.lambda),
            });
        }
        if self.scl == 0 {
            return Err(FdarError::InvalidParameter {
                parameter: "scl",
                message: "must be >= 1".to_string(),
            });
        }
        if !self.band_frac.is_finite() || self.band_frac < 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "band_frac",
                message: format!("must be finite and >= 0, got {}", self.band_frac),
            });
        }
        for (parameter, v) in [("nbhd_dim", self.nbhd_dim), ("dp_scl", self.dp_scl)] {
            if !(1..=64).contains(&v) {
                return Err(FdarError::InvalidParameter {
                    parameter,
                    message: format!("must be in 1..=64, got {v}"),
                });
            }
        }
        let dp_inexpressible = matches!(
            self.penalty,
            WarpObjectivePenalty::Roughness | WarpObjectivePenalty::Geodesic
        );
        if self.method == WarpOptimMethod::Dp && dp_inexpressible && self.lambda > 0.0 {
            return Err(FdarError::InvalidParameter {
                parameter: "penalty",
                message: format!(
                    "{:?} is not additive along a piecewise-linear warp and has no DP \
                     counterpart; use WarpOptimMethod::Rbfgs, or L2Gam/L2Psi with DP",
                    self.penalty
                ),
            });
        }
        Ok(())
    }

    /// Config equal to the classic first-order DP with penalty weight `lambda`.
    pub(super) fn classic_dp(lambda: f64) -> Self {
        Self {
            lambda,
            ..Self::default()
        }
    }

    /// The exact DP → RBFGS penalty-weight factor `φ` for the 1-D SRSFs `q1`,
    /// `q2` (sampled on the alignment grid): RBFGS with `λ_rb = φ·λ_dp`
    /// minimises the same functional as DP with `λ_dp` (up to a positive
    /// factor and an additive constant), for the `L2Gam` and `L2Psi`
    /// penalties. `φ = scl·‖q1‖₂‖q2‖₂ / (‖q1ᶠ‖₂‖q2ᶠ‖₂)` with `qᶠ` the SRSF
    /// linearly resampled `scl` times finer (as RBFGS does); `φ → 1` as the
    /// grid is refined. See [`WarpLambdaScale`] for the derivation.
    ///
    /// # Examples
    ///
    /// ```
    /// use fdars_core::alignment::WarpOptimConfig;
    ///
    /// let q: Vec<f64> = (0..101).map(|i| 1.0 + (i as f64 * 0.07).sin()).collect();
    /// let phi = WarpOptimConfig::default().rbfgs_lambda_factor(&q, &q);
    /// assert!((phi - 1.0).abs() < 0.03); // 1 + O(1/m)
    /// ```
    #[must_use]
    pub fn rbfgs_lambda_factor(&self, q1: &[f64], q2: &[f64]) -> f64 {
        lambda_factor(&[q1], &[q2], self.scl, 1.0)
    }

    /// Penalty weight in `method`'s own convention, converting
    /// [`lambda`](Self::lambda) per [`lambda_scale`](Self::lambda_scale);
    /// `phi` returns the DP → RBFGS factor (evaluated only when needed).
    pub(super) fn native_lambda(&self, method: WarpOptimMethod, phi: impl FnOnce() -> f64) -> f64 {
        let lam = self.lambda;
        if lam == 0.0 {
            return lam;
        }
        match (self.lambda_scale, method) {
            (WarpLambdaScale::Dp, WarpOptimMethod::Rbfgs) => lam * phi(),
            (WarpLambdaScale::Fdasrsf, WarpOptimMethod::Dp) => lam / phi(),
            _ => lam,
        }
    }

    /// The DP options of the RBFGS coarse-DP seed.
    pub(super) fn dp_seed(&self) -> DpSeed {
        DpSeed {
            nbhd_dim: self.nbhd_dim,
            dp_scl: self.dp_scl,
            dp_cost: self.dp_cost,
            dp_prune: self.dp_prune,
        }
    }
}

/// DP options of the RBFGS coarse-DP seed (the DP fields of
/// [`WarpOptimConfig`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct DpSeed {
    pub nbhd_dim: usize,
    pub dp_scl: usize,
    pub dp_cost: DpEdgeCost,
    pub dp_prune: bool,
}

impl DpSeed {
    /// The classic 35-move, unrefined, piecewise-constant DP.
    pub(super) fn is_classic(&self) -> bool {
        self.nbhd_dim == 7 && self.dp_scl == 1 && self.dp_cost == DpEdgeCost::PiecewiseConstant
    }
}

/// `scl·E1·E2 / (div·e1·e2)` for multi-channel SRSFs: `E_i` the total
/// Euclidean norm over all channels and samples, `e_i` the same after linear
/// resampling `scl` times finer (`div = 1` for 1-D, `d` for the per-channel
/// normalised R^d DP).
pub(super) fn lambda_factor(q1: &[&[f64]], q2: &[&[f64]], scl: usize, div: f64) -> f64 {
    let scl = scl.max(1);
    let euclid = |chs: &[&[f64]]| -> f64 {
        chs.iter()
            .map(|f| f.iter().map(|v| v * v).sum::<f64>())
            .sum::<f64>()
            .sqrt()
    };
    let (big1, big2) = (euclid(q1), euclid(q2));
    let (f1, f2) = (refined_euclid(q1, scl), refined_euclid(q2, scl));
    let den = div * f1 * f2;
    if den > 0.0 && big1 > 0.0 && big2 > 0.0 {
        scl as f64 * big1 * big2 / den
    } else {
        1.0
    }
}

/// Total Euclidean norm of the channels linearly resampled `scl` times finer
/// (the `e_i` of [`WarpLambdaScale`]).
pub(super) fn refined_euclid(q: &[&[f64]], scl: usize) -> f64 {
    q.iter()
        .map(|f| {
            if f.len() < 2 || scl <= 1 {
                f.iter().map(|v| v * v).sum::<f64>()
            } else {
                refine_srsf_linear(f, scl)
                    .iter()
                    .map(|v| v * v)
                    .sum::<f64>()
            }
        })
        .sum::<f64>()
        .sqrt()
}

/// Optimal warp aligning SRSF `q2` to `q1` under `cfg` (assumed validated).
///
/// `band` is an explicit Sakoe–Chiba index radius (the classic banded
/// functions); `None` falls back to [`WarpOptimConfig::band_frac`]. The
/// penalty weight is converted per [`WarpOptimConfig::lambda_scale`].
/// `Dp` + `L2Gam` with the default DP options runs exactly the arithmetic of
/// [`dp_alignment_core_banded`](super::dp_alignment_core_banded) the classic
/// functions use, so it is bit-identical to them.
pub(super) fn optimal_warp(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    cfg: &WarpOptimConfig,
    band: Option<usize>,
) -> Vec<f64> {
    let band = band.or_else(|| band_radius(cfg.band_frac, argvals.len()));
    let lambda = cfg.native_lambda(cfg.method, || cfg.rbfgs_lambda_factor(q1, q2));
    match cfg.method {
        WarpOptimMethod::Dp => dp_warp(q1, q2, argvals, cfg, lambda, band),
        WarpOptimMethod::Rbfgs => {
            rbfgs_warp(q1, q2, argvals, &RbfgsParams::from_cfg(cfg, lambda, band))
        }
    }
}

/// DP branch of [`optimal_warp`]: builds the engine options from `cfg`, with
/// penalty weight `lambda` (already in the DP convention).
pub(super) fn dp_warp(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    cfg: &WarpOptimConfig,
    lambda: f64,
    band: Option<usize>,
) -> Vec<f64> {
    let nbhd = coprime_nbhd(cfg.nbhd_dim);
    let opts = DpEngineOptions {
        moves: DpMoves {
            nbhd: &nbhd,
            // Exact (bit-identical) only for non-negative edge costs.
            prune: cfg.dp_prune && dp_prune_is_exact(argvals, lambda),
        },
        scl: cfg.dp_scl.max(1),
        cost: cfg.dp_cost,
    };
    match cfg.penalty {
        WarpObjectivePenalty::L2Psi if lambda > 0.0 => {
            dp_alignment_core_opts(q1, q2, argvals, band, &opts, |sc, tc, sr, tr| {
                dp_l2psi_penalty(argvals, sc, tc, sr, tr, lambda)
            })
        }
        WarpObjectivePenalty::L2Gam => {
            dp_alignment_core_opts(q1, q2, argvals, band, &opts, |sc, tc, sr, tr| {
                dp_lambda_penalty(argvals, sc, tc, sr, tr, lambda)
            })
        }
        // lambda == 0 (validated): no penalty.
        _ => dp_alignment_core_opts(q1, q2, argvals, band, &opts, |sc, tc, sr, tr| {
            dp_lambda_penalty(argvals, sc, tc, sr, tr, 0.0)
        }),
    }
}

/// Validate `argvals` / curve lengths shared by the `*_with_config` entry points.
pub(super) fn check_grid(argvals: &[f64], m: usize, what: &'static str) -> Result<(), FdarError> {
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: what,
            expected: format!("length {}", argvals.len()),
            actual: format!("length {m}"),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: "at least 2 points".to_string(),
            actual: format!("{m}"),
        });
    }
    if argvals
        .windows(2)
        .any(|w| w[1].partial_cmp(&w[0]) != Some(std::cmp::Ordering::Greater))
    {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "must be strictly increasing".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
#[path = "optim_golden.rs"]
mod golden;

#[cfg(test)]
#[path = "optim_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "dp_options_golden.rs"]
mod dp_golden;

#[cfg(test)]
#[path = "dp_options_tests.rs"]
mod dp_tests;

#[cfg(test)]
#[path = "optim_coverage_tests.rs"]
mod coverage_tests;
