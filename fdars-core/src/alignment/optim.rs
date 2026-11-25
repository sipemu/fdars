//! Warp-optimiser selection (DP / Riemannian L-BFGS) and in-objective warp
//! penalties for elastic alignment.
//!
//! [`WarpOptimConfig`] is the option bag behind the `*_with_config` entry points
//! ([`elastic_align_pair_with_config`](super::elastic_align_pair_with_config),
//! [`elastic_distance_with_config`](super::elastic_distance_with_config),
//! [`elastic_self_distance_matrix_with_config`](super::elastic_self_distance_matrix_with_config),
//! [`elastic_cross_distance_matrix_with_config`](super::elastic_cross_distance_matrix_with_config),
//! [`karcher_mean_with_config`](super::karcher_mean_with_config)). Its default
//! (`Dp` + `L2Gam`) reproduces the classic functions bit-for-bit.

use super::rbfgs::{rbfgs_warp, RbfgsParams};
use super::{dp_alignment_core_banded, dp_alignment_core_with_penalty, dp_l2psi_penalty};
use crate::error::FdarError;

/// Which optimiser computes the optimal warp γ (fdasrvf `optim_method` /
/// fdasrsf `method`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum WarpOptimMethod {
    /// Exact dynamic programming over fdars's 35-move coprime neighbourhood
    /// (the classic engine; default and exactness reference).
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
/// lam=…)` semantics. The two scales are not interchangeable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
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
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
    /// must be ≥ 1). Ignored by DP.
    pub scl: usize,
}

impl Default for WarpOptimConfig {
    fn default() -> Self {
        Self {
            method: WarpOptimMethod::Dp,
            lambda: 0.0,
            penalty: WarpObjectivePenalty::L2Gam,
            max_iter: 30,
            scl: 3,
        }
    }
}

impl WarpOptimConfig {
    /// Check the configuration.
    ///
    /// # Errors
    /// [`FdarError::InvalidParameter`] if `lambda` is negative or non-finite,
    /// `scl == 0`, or a `Roughness`/`Geodesic` penalty with `lambda > 0` is
    /// requested from the DP optimiser (it cannot express them).
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
}

/// Optimal warp aligning SRSF `q2` to `q1` under `cfg` (assumed validated).
///
/// `band` is the Sakoe–Chiba radius for the DP optimiser (ignored by RBFGS).
/// `Dp` + `L2Gam` calls [`dp_alignment_core_banded`] exactly as the classic
/// functions do, so it is bit-identical to them.
pub(super) fn optimal_warp(
    q1: &[f64],
    q2: &[f64],
    argvals: &[f64],
    cfg: &WarpOptimConfig,
    band: Option<usize>,
) -> Vec<f64> {
    match cfg.method {
        WarpOptimMethod::Dp => match cfg.penalty {
            WarpObjectivePenalty::L2Psi if cfg.lambda > 0.0 => {
                dp_alignment_core_with_penalty(q1, q2, argvals, band, |sc, tc, sr, tr| {
                    dp_l2psi_penalty(argvals, sc, tc, sr, tr, cfg.lambda)
                })
            }
            WarpObjectivePenalty::L2Gam => {
                dp_alignment_core_banded(q1, q2, argvals, cfg.lambda, band)
            }
            // lambda == 0 (validated): no penalty.
            _ => dp_alignment_core_banded(q1, q2, argvals, 0.0, band),
        },
        WarpOptimMethod::Rbfgs => rbfgs_warp(
            q1,
            q2,
            argvals,
            &RbfgsParams {
                lambda: cfg.lambda,
                penalty: cfg.penalty,
                max_iter: cfg.max_iter,
                scl: cfg.scl,
            },
        ),
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
