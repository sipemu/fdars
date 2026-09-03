//! Optimal experimental design criteria for sparse functional data (FOptDes).
//!
//! This module scores a caller-supplied set of design points (grid indices)
//! against a fitted [`PaceFpcaResult`], computing one of two criteria dispatched
//! through the [`DesignCriterion`] / [`OptimalityKind`] enum pair:
//!
//! - **Trajectory** ([`DesignCriterion::Trajectory`], FOD-01): the integrated,
//!   Simpson-weighted conditional BLUP mean-squared reconstruction error of the
//!   latent trajectory `x(t)` given noisy observations at the design points.
//! - **Score** ([`DesignCriterion::Score`], FOD-02): an A- or D-optimal summary
//!   of the posterior FPC-score covariance `Cov(ξ | Y_S)` — trace for A, log-det
//!   for D.
//!
//! Both criteria share the private [`build_sigma_design`] helper, which assembles
//! the `p×p` covariance `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p` of the observations at the
//! `p = |selected|` design points (mirroring the `Σ_yi` assembly in
//! `pace_fpca.rs`). All criteria are *minimized* and are monotone non-increasing as
//! design points are added, so the (future) greedy selector minimizes uncertainty.
//!
//! The mathematics follows Ji & Müller (2017) and the Yao–Müller–Wang (2005) PACE
//! formulation already implemented in [`crate::pace_fpca`]. [`design_criterion`]
//! is the pure numerical core; [`optimal_design`] wraps it in a deterministic
//! greedy sequential forward-selection loop.
//!
//! # End-to-end example
//!
//! Fit a sparse PACE FPCA model, then greedily select informative design points:
//!
//! ```rust
//! use fdars_core::irreg_fdata::IrregFdata;
//! use fdars_core::pace_fpca::{pace_fpca, PaceFpcaConfig};
//! use fdars_core::{optimal_design, DesignCriterion, OptDesConfig};
//!
//! // A handful of sparsely-sampled curves on [0, 1].
//! let argvals_list = vec![
//!     vec![0.1, 0.4, 0.7],
//!     vec![0.0, 0.3, 0.6, 0.9],
//!     vec![0.2, 0.5, 0.8],
//!     vec![0.0, 0.25, 0.5, 0.75, 1.0],
//!     vec![0.1, 0.5, 0.9],
//!     vec![0.0, 0.4, 0.8],
//! ];
//! let values_list: Vec<Vec<f64>> = argvals_list
//!     .iter()
//!     .enumerate()
//!     .map(|(i, ts)| ts.iter().map(|&t: &f64| (i as f64 + 1.0) * t.sin()).collect())
//!     .collect();
//! let data = IrregFdata::from_lists(&argvals_list, &values_list);
//!
//! // Fit PACE on a small work grid.
//! let m = 21_usize;
//! let pace_cfg = PaceFpcaConfig {
//!     ncomp: 2,
//!     bandwidth: 0.2,
//!     sigma2: 0.01,
//!     work_grid: (0..m).map(|i| i as f64 / (m - 1) as f64).collect(),
//!     alpha: 0.05,
//! };
//! let model = pace_fpca(&data, &pace_cfg).unwrap();
//!
//! // Greedily select 2 design points over the fitted model (read-only).
//! let config = OptDesConfig {
//!     candidate_grid: model.argvals.clone(),
//!     budget: 2,
//!     criterion: DesignCriterion::Trajectory,
//! };
//! let result = optimal_design(&model, &config).unwrap();
//!
//! assert_eq!(result.selected_indices.len(), 2);
//! assert_eq!(result.criterion_trace.len(), 2);
//! let chosen: &[f64] = &result.selected_argvals;
//! assert_eq!(chosen.len(), 2);
//! ```

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::iter_maybe_parallel;
// Import the factor/forward-back pair directly rather than `cholesky_solve`:
// the trajectory criterion factors Σ_d once (O(p³)) and then solves the m grid-point
// right-hand-sides via `cholesky_forward_back` (O(p²) each), amortizing the single
// factorization instead of re-factoring O(m) times as `cholesky_solve` would.
use crate::linalg::{cholesky_factor, cholesky_forward_back, log_det_from_cholesky};
use crate::pace_fpca::PaceFpcaResult;

/// Which design criterion to evaluate.
///
/// Dispatched by [`design_criterion`]. `Trajectory` scores reconstruction of the
/// latent curve; `Score` scores recovery of the FPC scores under an A- or
/// D-optimality summary.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DesignCriterion {
    /// Integrated Simpson-weighted conditional BLUP trajectory-reconstruction MSE
    /// (FOD-01). Empty design returns the prior integrated variance `Σ_k λ_k`.
    Trajectory,
    /// FPC-score posterior-covariance summary (FOD-02); see [`OptimalityKind`].
    Score(OptimalityKind),
}

/// Optimality kind for the [`DesignCriterion::Score`] criterion.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimalityKind {
    /// A-optimality: trace of the posterior score covariance `Cov(ξ | Y_S)`.
    /// Empty design returns `Σ_k λ_k`.
    A,
    /// D-optimality: log-determinant of the posterior score covariance,
    /// `Σ_k log(posterior eigenvalues)`, returned un-negated. Adding design points
    /// shrinks the posterior covariance, so this value is monotone NON-INCREASING:
    /// `log det Cov(ξ | Y_S) ≤ log det Λ = Σ_k log λ_k`. Its SIGN is not fixed — it
    /// depends on the eigenvalue scale (e.g. `λ = [2, 1]` gives an empty-design value
    /// of `ln 2 ≈ +0.693`, positive). Do NOT assume it is negative, and do NOT negate
    /// it. Empty design returns `Σ_k log λ_k`.
    D,
}

/// Score a design point index set against a fitted PACE FPCA model.
///
/// `selected` holds indices into `model.argvals` (0-based). Every index must be
/// `< model.argvals.len()`. An empty `selected` returns the prior baseline:
/// `Σ_k λ_k` for [`DesignCriterion::Trajectory`] and [`OptimalityKind::A`], and
/// `Σ_k log λ_k` for [`OptimalityKind::D`].
///
/// Duplicate indices are *tolerated* (the resulting `Σ_d` is singular in the
/// duplicated rows but the ridge-retry keeps the solve stable); callers that
/// require distinct design points must dedupe upstream.
///
/// All criteria are minimized and are monotone non-increasing as design points
/// are added: `criterion(S ∪ {t}) ≤ criterion(S) + 1e-12`.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `model.ncomp == 0`,
/// `model.sigma2 <= 0.0`, or any index in `selected` is out of range. Returns
/// [`FdarError::ComputationFailed`] only if a Cholesky factorization fails even
/// after the `1e-8` ridge-retry (never panics).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn design_criterion(
    model: &PaceFpcaResult,
    selected: &[usize],
    criterion: DesignCriterion,
) -> Result<f64, FdarError> {
    // --- Validation (ASVS V5 input validation) ---
    let m = model.argvals.len();
    if model.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "model.ncomp",
            message: "ncomp must be > 0; the model has no FPC components".into(),
        });
    }
    if model.eigenvalues.len() < model.ncomp {
        return Err(FdarError::InvalidParameter {
            parameter: "model.eigenvalues",
            message: format!(
                "eigenvalues length {} is smaller than ncomp {}",
                model.eigenvalues.len(),
                model.ncomp
            ),
        });
    }
    if model.sigma2 <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "model.sigma2",
            message: format!("sigma2 must be > 0; got {}", model.sigma2),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "model.argvals",
            message: format!(
                "argvals must have length >= 2 (a trajectory integral / Simpson quadrature is undefined for m < 2); got {m}"
            ),
        });
    }
    for &idx in selected {
        if idx >= m {
            return Err(FdarError::InvalidParameter {
                parameter: "selected",
                message: format!("index {idx} is out of range for argvals of length {m}"),
            });
        }
    }

    // --- Dispatch ---
    match criterion {
        DesignCriterion::Trajectory => trajectory_criterion(model, selected),
        DesignCriterion::Score(kind) => score_criterion(model, selected, kind),
    }
}

// ---------------------------------------------------------------------------
// Greedy selection wrapper (FOD-04 / FOD-05)
// ---------------------------------------------------------------------------

/// Configuration for the greedy [`optimal_design`] selector.
///
/// Carries a *single* [`DesignCriterion`] field — `Score(OptimalityKind)` already
/// wraps the optimality kind, and `Trajectory` needs none, so no separate
/// optimality field is required. NOT `#[non_exhaustive]`, so callers can build it
/// with a struct literal (mirrors [`crate::pace_fpca::PaceFpcaConfig`]).
///
/// The empty-grid [`Default`] is a safe minimal placeholder; the empty grid is
/// rejected at [`optimal_design`] call time, not at construction.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesConfig {
    /// Candidate design points. Every value must appear (within `1e-9`) in
    /// `model.argvals`; each is mapped to its grid index before selection.
    pub candidate_grid: Vec<f64>,
    /// Number of design points to select (`p`). Must be `> 0` and
    /// `<= candidate_grid.len()`.
    pub budget: usize,
    /// Criterion evaluated at every greedy step via [`design_criterion`].
    pub criterion: DesignCriterion,
}

impl Default for OptDesConfig {
    fn default() -> Self {
        Self {
            candidate_grid: vec![],
            budget: 1,
            criterion: DesignCriterion::Trajectory,
        }
    }
}

/// Result of greedy [`optimal_design`] selection.
///
/// `#[non_exhaustive]` for forward compatibility (mirrors
/// [`crate::pace_fpca::PaceFpcaResult`]).
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OptDesResult {
    /// Grid indices (into `model.argvals`) of the selected design points, in
    /// selection order. Length `== config.budget`; duplicate-free.
    pub selected_indices: Vec<usize>,
    /// `model.argvals` values at the selected indices, in selection order.
    /// Length `== config.budget`.
    pub selected_argvals: Vec<f64>,
    /// Achieved criterion value after each greedy step. Length `== config.budget`;
    /// monotone non-increasing (`trace[i+1] <= trace[i] + 1e-12`).
    pub criterion_trace: Vec<f64>,
}

/// Map each `candidate_grid` value to its `model.argvals` grid index.
///
/// Uses an FP-tolerant position search (`|t - cand| < 1e-9`) so grid values
/// computed as `i as f64 / (m-1) as f64` match a caller-supplied equivalent that
/// may differ by a few ULPs. Preserves `candidate_grid` order.
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if any candidate is not found in
/// `argvals` within `1e-9`.
fn map_candidates_to_indices(
    candidate_grid: &[f64],
    argvals: &[f64],
) -> Result<Vec<usize>, FdarError> {
    candidate_grid
        .iter()
        .map(|&cand| {
            argvals
                .iter()
                .position(|&t| (t - cand).abs() < 1e-9)
                .ok_or_else(|| FdarError::InvalidParameter {
                    parameter: "config.candidate_grid",
                    message: format!(
                        "candidate {cand:.6} not found in model.argvals within tolerance 1e-9"
                    ),
                })
        })
        .collect()
}

/// Greedy sequential forward-selection of design points over a fitted PACE model.
///
/// Starting from the empty design, at each of `config.budget` steps this adds the
/// not-yet-selected candidate index that most reduces `config.criterion` (evaluated
/// through the Phase-64 [`design_criterion`]), until the budget is reached. The
/// supplied [`PaceFpcaResult`] is consumed **read-only** — no re-estimation of the
/// eigenstructure or `σ²` (the two-stage FOptDes contract, FOD-05).
///
/// # Determinism
///
/// Candidate *evaluation* is parallelized (`iter_maybe_parallel!`), but the argmin
/// is a **sequential** fold over the collected `(index, value)` pairs with a
/// smallest-index tie-break (never rayon `min_by`, which is not stable under ties).
/// Two identical calls produce byte-identical `selected_indices` and
/// `criterion_trace`, and the result is identical with and without the `parallel`
/// feature.
///
/// # Guarantees
///
/// - `selected_indices.len() == config.budget`, duplicate-free.
/// - `criterion_trace` is monotone non-increasing (inherited from
///   [`design_criterion`]).
///
/// # Errors
///
/// Returns [`FdarError::InvalidParameter`] if `config.budget == 0`,
/// `config.budget > config.candidate_grid.len()`, any candidate is not in
/// `model.argvals` (within `1e-9`), `model.ncomp == 0`, or `model.sigma2 <= 0.0`.
/// Propagates any [`FdarError`] raised by [`design_criterion`] during evaluation.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn optimal_design(
    model: &PaceFpcaResult,
    config: &OptDesConfig,
) -> Result<OptDesResult, FdarError> {
    // --- Validation (ASVS V5 input validation) — fail fast before any candidate work ---
    if config.budget == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "config.budget",
            message: "budget must be > 0".into(),
        });
    }
    if config.budget > config.candidate_grid.len() {
        return Err(FdarError::InvalidParameter {
            parameter: "config.budget",
            message: format!(
                "budget {} exceeds the number of candidate points {}",
                config.budget,
                config.candidate_grid.len()
            ),
        });
    }
    if model.ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "model.ncomp",
            message: "ncomp must be > 0; the model has no FPC components".into(),
        });
    }
    if model.sigma2 <= 0.0 {
        return Err(FdarError::InvalidParameter {
            parameter: "model.sigma2",
            message: format!("sigma2 must be > 0; got {}", model.sigma2),
        });
    }

    // Map candidate_grid → argvals indices once (preserves candidate_grid order).
    let candidate_indices = map_candidates_to_indices(&config.candidate_grid, &model.argvals)?;

    let mut selected: Vec<usize> = Vec::with_capacity(config.budget);
    let mut trace: Vec<f64> = Vec::with_capacity(config.budget);

    for _step in 0..config.budget {
        // Not-yet-selected candidates, in candidate_indices (== candidate_grid) order.
        let remaining: Vec<usize> = candidate_indices
            .iter()
            .copied()
            .filter(|idx| !selected.contains(idx))
            .collect();

        // PARALLEL evaluate: each closure captures only immutable refs and allocates
        // its own `trial`. `PaceFpcaResult` is Send + Sync (all Vec<f64>/FdMatrix/
        // usize/f64 fields), so this compiles under `--features parallel`.
        #[cfg(feature = "parallel")]
        use rayon::iter::ParallelIterator;
        let scores: Vec<(usize, f64)> = iter_maybe_parallel!(remaining)
            .map(|idx| {
                let mut trial = selected.clone();
                trial.push(idx);
                let val = design_criterion(model, &trial, config.criterion.clone())?;
                Ok::<(usize, f64), FdarError>((idx, val))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // SEQUENTIAL argmin over the collected, fixed-order `scores`. Strict `<`
        // keeps the FIRST minimum → smallest-index tie-break (rayon `min_by` is NOT
        // stable under ties, so it must not be used here).
        let (best_idx, best_val) = scores
            .into_iter()
            .fold(None::<(usize, f64)>, |acc, (idx, val)| {
                Some(match acc {
                    None => (idx, val),
                    Some((bi, bv)) => {
                        if val < bv {
                            (idx, val)
                        } else {
                            (bi, bv)
                        }
                    }
                })
            })
            .expect("remaining is non-empty — guaranteed by budget <= candidate count");

        selected.push(best_idx);
        trace.push(best_val);
    }

    let selected_argvals = selected.iter().map(|&i| model.argvals[i]).collect();
    Ok(OptDesResult {
        selected_indices: selected,
        selected_argvals,
        criterion_trace: trace,
    })
}

/// Assemble the `p×p` design covariance `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p`
/// (row-major), where `p = selected.len()`.
///
/// Mirrors the `Σ_yi` assembly in `pace_fpca.rs`, substituting design-point grid
/// indices for per-curve observation indices. Shape is `|S|×|S|`, NOT `K×K`.
fn build_sigma_design(model: &PaceFpcaResult, selected: &[usize]) -> Vec<f64> {
    let p = selected.len();
    let ncomp = model.ncomp;
    let mut sigma_d = vec![0.0_f64; p * p];
    for row in 0..p {
        let j_row = selected[row];
        for col in 0..p {
            let j_col = selected[col];
            let mut s = 0.0_f64;
            for k in 0..ncomp {
                s += model.eigenfunctions[(j_row, k)]
                    * model.eigenvalues[k]
                    * model.eigenfunctions[(j_col, k)];
            }
            sigma_d[row * p + col] = s;
        }
        sigma_d[row * p + row] += model.sigma2; // σ²I_p diagonal
    }
    sigma_d
}

/// Cholesky-factor `Σ_d` with a single `1e-8` diagonal ridge-retry on failure.
///
/// Mirrors the ridge-retry in `pace_fpca.rs:480–490`. Never panics; returns the
/// lower-triangular factor `L` on success.
fn factor_sigma_design_with_retry(mut sigma_d: Vec<f64>, p: usize) -> Result<Vec<f64>, FdarError> {
    match cholesky_factor(&sigma_d, p) {
        Ok(l) => Ok(l),
        Err(_) => {
            for i in 0..p {
                sigma_d[i * p + i] += 1e-8;
            }
            cholesky_factor(&sigma_d, p).map_err(|_| FdarError::ComputationFailed {
                operation: "optimal_design Sigma_d Cholesky",
                detail: "Cholesky failed after 1e-8 ridge; sigma2 may be too small".into(),
            })
        }
    }
}

/// Cholesky-factor the `K×K` posterior covariance `Cov` with a single ridge-retry
/// on failure, mirroring [`factor_sigma_design_with_retry`].
///
/// The Schur-complement `Cov = Λ − A_mat` is positive-definite in exact arithmetic,
/// but FP cancellation (especially after a ridge-adjusted `Σ_d`) can make it fail the
/// Cholesky diagonal test. On failure we add a tiny `1e-8`-scaled diagonal ridge and
/// retry once, keeping D-opt as robust as A-opt. Never panics.
fn factor_posterior_cov_with_retry(mut cov: Vec<f64>, ncomp: usize) -> Result<Vec<f64>, FdarError> {
    match cholesky_factor(&cov, ncomp) {
        Ok(l) => Ok(l),
        Err(_) => {
            // Ridge scaled to the covariance magnitude, matching the `1e-8` convention
            // used for the Σ_d retry (there the scale is implicitly ~O(1)).
            let scale: f64 = (0..ncomp)
                .map(|k| cov[k * ncomp + k].abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let ridge = 1e-8 * scale;
            for i in 0..ncomp {
                cov[i * ncomp + i] += ridge;
            }
            cholesky_factor(&cov, ncomp).map_err(|_| FdarError::ComputationFailed {
                operation: "optimal_design D-optimality log-det",
                detail: "posterior covariance Cholesky failed after ridge; \
                         model may be near-degenerate"
                    .into(),
            })
        }
    }
}

/// Extract the `p×ncomp` design-point eigenfunction sub-matrix `Φ_d` (row-major),
/// where `phi_d[i * ncomp + k] = eigenfunctions[(selected[i], k)]`.
fn build_phi_d(model: &PaceFpcaResult, selected: &[usize]) -> Vec<f64> {
    let p = selected.len();
    let ncomp = model.ncomp;
    let mut phi_d = vec![0.0_f64; p * ncomp];
    for (i, &j) in selected.iter().enumerate() {
        for k in 0..ncomp {
            phi_d[i * ncomp + k] = model.eigenfunctions[(j, k)];
        }
    }
    phi_d
}

/// Trajectory criterion (FOD-01): integrated Simpson-weighted conditional
/// BLUP-MSE `Σ_j w_j (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))`.
fn trajectory_criterion(model: &PaceFpcaResult, selected: &[usize]) -> Result<f64, FdarError> {
    let m = model.argvals.len();
    let ncomp = model.ncomp;
    let p = selected.len();
    let weights = simpsons_weights(&model.argvals);

    // Empty-set fast path: no design points → no reduction → prior variance only.
    if p == 0 {
        let mut mse = 0.0_f64;
        for j in 0..m {
            let prior_var: f64 = (0..ncomp)
                .map(|k| model.eigenvalues[k] * model.eigenfunctions[(j, k)].powi(2))
                .sum();
            mse += weights[j] * prior_var;
        }
        return Ok(mse);
    }

    // Factor Σ_d once (O(p³)); each grid point is then an O(p²) forward/back solve.
    let l = factor_sigma_design_with_retry(build_sigma_design(model, selected), p)?;
    let phi_d = build_phi_d(model, selected); // p × ncomp, row-major

    let mut mse = 0.0_f64;
    let mut rhs = vec![0.0_f64; p];
    for j in 0..m {
        // Prior variance at grid point j: Σ_k λ_k φ_k(t_j)².
        let prior_var: f64 = (0..ncomp)
            .map(|k| model.eigenvalues[k] * model.eigenfunctions[(j, k)].powi(2))
            .sum();

        // Cross-covariance p-vector: rhs[i] = Σ_k λ_k φ_k(t_j) φ_k(argvals[selected[i]]).
        for (i, r) in rhs.iter_mut().enumerate() {
            let mut s = 0.0_f64;
            for k in 0..ncomp {
                s += model.eigenvalues[k] * model.eigenfunctions[(j, k)] * phi_d[i * ncomp + k];
            }
            *r = s;
        }

        // reduction = rhsᵀ Σ_d⁻¹ rhs, via the pre-factored Cholesky.
        let v = cholesky_forward_back(&l, &rhs, p);
        let reduction: f64 = rhs.iter().zip(v.iter()).map(|(&a, &b)| a * b).sum();

        mse += weights[j] * (prior_var - reduction);
    }
    Ok(mse)
}

/// Score criterion (FOD-02): A- or D-optimal summary of the K×K posterior FPC
/// score covariance `Cov(ξ | Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ`.
fn score_criterion(
    model: &PaceFpcaResult,
    selected: &[usize],
    kind: OptimalityKind,
) -> Result<f64, FdarError> {
    let ncomp = model.ncomp;
    let p = selected.len();

    // Empty-set fast path: no information → posterior = prior = diag(λ).
    if p == 0 {
        return match kind {
            OptimalityKind::A => Ok(model.eigenvalues.iter().take(ncomp).sum()),
            OptimalityKind::D => {
                let mut s = 0.0_f64;
                for &lam in model.eigenvalues.iter().take(ncomp) {
                    if lam <= 0.0 {
                        return Err(FdarError::ComputationFailed {
                            operation: "optimal_design D-optimality",
                            detail: "non-positive eigenvalue in prior".into(),
                        });
                    }
                    s += lam.ln();
                }
                Ok(s)
            }
        };
    }

    // Factor Σ_d once, then solve Σ_d x_k = Φ_d[:,k] per component (forward/back).
    let l = factor_sigma_design_with_retry(build_sigma_design(model, selected), p)?;
    let phi_d = build_phi_d(model, selected); // p × ncomp, row-major

    // sigma_inv_phi_lam[j,k] = λ_k · (Σ_d⁻¹ Φ_d[:,k])[j]  (mirror pace_fpca.rs:525–545).
    let mut sigma_inv_phi_lam = vec![0.0_f64; p * ncomp];
    let mut phi_col = vec![0.0_f64; p];
    for k in 0..ncomp {
        for (i, c) in phi_col.iter_mut().enumerate() {
            *c = phi_d[i * ncomp + k];
        }
        let sol = cholesky_forward_back(&l, &phi_col, p);
        for j in 0..p {
            sigma_inv_phi_lam[j * ncomp + k] = model.eigenvalues[k] * sol[j];
        }
    }

    // A_mat[k,l] = λ_k · Σ_j Φ_d[j,k] · sigma_inv_phi_lam[j,l]  (pace_fpca.rs:547–558).
    let mut a_mat = vec![0.0_f64; ncomp * ncomp];
    for k in 0..ncomp {
        for l in 0..ncomp {
            let mut s = 0.0_f64;
            for j in 0..p {
                s += phi_d[j * ncomp + k] * sigma_inv_phi_lam[j * ncomp + l];
            }
            a_mat[k * ncomp + l] = model.eigenvalues[k] * s;
        }
    }

    // Posterior covariance Cov[k,l] = (k==l ? λ_k : 0) − A_mat[k,l].
    let mut cov = vec![0.0_f64; ncomp * ncomp];
    for k in 0..ncomp {
        for l in 0..ncomp {
            let prior = if k == l { model.eigenvalues[k] } else { 0.0 };
            cov[k * ncomp + l] = prior - a_mat[k * ncomp + l];
        }
    }

    match kind {
        OptimalityKind::A => {
            // trace(Cov) = Σ_k Cov[k,k].
            let tr: f64 = (0..ncomp).map(|k| cov[k * ncomp + k]).sum();
            Ok(tr)
        }
        OptimalityKind::D => {
            // log det(Cov) via Cholesky. Returned un-negated and monotone
            // non-increasing (its sign depends on the eigenvalue scale). Do NOT negate.
            //
            // Mirror the Σ_d ridge-retry: the Schur complement Cov is PD in exact
            // arithmetic, but when Σ_d was itself ridge-adjusted (tiny sigma2), FP
            // cancellation in `(λ_k : 0) − A_mat` can push a diagonal entry to
            // (near-)zero and make the Cholesky fail. A tiny ridge rescues it so D-opt
            // succeeds wherever A-opt does. Never panics.
            let l_cov = factor_posterior_cov_with_retry(cov, ncomp)?;
            Ok(log_det_from_cholesky(&l_cov, ncomp))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::FdMatrix;

    /// Build a synthetic [`PaceFpcaResult`] with exactly-orthonormal eigenfunctions
    /// under the grid's Simpson weights.
    ///
    /// Two eigenfunctions (scaled Fourier cosines) on a uniform `[0, 1]` grid of
    /// length `m`, each normalized so `Σ_j w_j φ_k(t_j)² = 1`. `λ = [2.0, 1.0]`,
    /// `σ² = 0.5`, `ncomp = 2`. Unused result fields are valid-shape placeholders.
    fn synthetic_model(m: usize) -> PaceFpcaResult {
        synthetic_model_params(m, vec![2.0, 1.0], 0.5)
    }

    fn synthetic_model_params(m: usize, eigenvalues: Vec<f64>, sigma2: f64) -> PaceFpcaResult {
        let ncomp = eigenvalues.len();
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let weights = simpsons_weights(&argvals);

        // Raw eigenfunctions: cos(k·π·t) for k = 1..=ncomp (orthogonal under the grid),
        // normalized to unit Simpson-weighted L² norm.
        let mut ef = vec![0.0_f64; m * ncomp];
        for k in 0..ncomp {
            let freq = (k + 1) as f64 * std::f64::consts::PI;
            let raw: Vec<f64> = argvals.iter().map(|&t| (freq * t).cos()).collect();
            let norm_sq: f64 = (0..m).map(|j| weights[j] * raw[j] * raw[j]).sum();
            let norm = norm_sq.sqrt();
            for j in 0..m {
                // column-major: element (row=j, col=k) at index j + k*m
                ef[j + k * m] = raw[j] / norm;
            }
        }
        let eigenfunctions = FdMatrix::from_column_major(ef, m, ncomp).unwrap();

        PaceFpcaResult {
            mean: vec![0.0; m],
            eigenvalues,
            eigenfunctions,
            scores: FdMatrix::zeros(1, ncomp),
            fitted: FdMatrix::zeros(1, m),
            fitted_lower: FdMatrix::zeros(1, m),
            fitted_upper: FdMatrix::zeros(1, m),
            argvals,
            sigma2,
            ncomp,
        }
    }

    // ---- Trajectory branch (FOD-01) ----

    #[test]
    fn test_trajectory_empty_set() {
        let model = synthetic_model(51);
        let mse = design_criterion(&model, &[], DesignCriterion::Trajectory).unwrap();
        // MSE(∅) = Σ_k λ_k = 2.0 + 1.0 = 3.0
        assert!((mse - 3.0).abs() < 1e-10, "MSE(∅) = {mse}, expected 3.0");
    }

    #[test]
    fn test_trajectory_grid_invariance() {
        let m21 = design_criterion(&synthetic_model(21), &[], DesignCriterion::Trajectory).unwrap();
        let m51 = design_criterion(&synthetic_model(51), &[], DesignCriterion::Trajectory).unwrap();
        let m101 =
            design_criterion(&synthetic_model(101), &[], DesignCriterion::Trajectory).unwrap();
        assert!((m21 - m51).abs() < 1e-10, "m21={m21} m51={m51}");
        assert!((m51 - m101).abs() < 1e-10, "m51={m51} m101={m101}");
    }

    #[test]
    fn test_trajectory_reduces_on_point() {
        let model = synthetic_model(51);
        let mse_empty = design_criterion(&model, &[], DesignCriterion::Trajectory).unwrap();
        let mse_one = design_criterion(&model, &[25], DesignCriterion::Trajectory).unwrap();
        assert!(
            mse_one <= mse_empty + 1e-12,
            "mse_one={mse_one} mse_empty={mse_empty}"
        );
    }

    #[test]
    fn test_monotonicity_trajectory() {
        let model = synthetic_model(51);
        let s0 = design_criterion(&model, &[10], DesignCriterion::Trajectory).unwrap();
        let s1 = design_criterion(&model, &[10, 30], DesignCriterion::Trajectory).unwrap();
        assert!(s1 <= s0 + 1e-12, "s1={s1} s0={s0}");
    }

    #[test]
    fn test_validation_index_range() {
        let model = synthetic_model(51);
        let res = design_criterion(&model, &[51], DesignCriterion::Trajectory);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_sigma2() {
        let model = synthetic_model_params(51, vec![2.0, 1.0], 0.0);
        let res = design_criterion(&model, &[0], DesignCriterion::Trajectory);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_ncomp() {
        // ncomp == 0 with empty eigenvalues.
        let model = synthetic_model_params(51, vec![], 0.5);
        let res = design_criterion(&model, &[0], DesignCriterion::Trajectory);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_ridge_retry() {
        // Force Σ_d genuinely non-PD so the FIRST cholesky_factor fails and only the
        // 1e-8 ridge-retry rescues it. Duplicating a design index makes two rows of Σ_d
        // identical → Σ_d is rank-1 + σ²I. Its second Cholesky pivot is
        // ((a+σ²)² − a²)/(a+σ²) ≈ 2·σ² for small σ². With σ² = 1e-13 this pivot is
        // ~2e-13 ≤ the 1e-12 cholesky_factor threshold, so the first factorization
        // FAILS; the 1e-8 ridge lifts the pivot to ~1e-8 and the retry succeeds.
        // (Duplicate indices are explicitly tolerated per `design_criterion` docs.)
        let model = synthetic_model_params(51, vec![2.0, 1.0], 1e-13);
        let res = design_criterion(&model, &[10, 10], DesignCriterion::Trajectory);
        assert!(
            res.is_ok(),
            "ridge-retry should rescue near-singular Σ_d: {res:?}"
        );
        // Sanity: without the retry this input is non-PD. Confirm the raw factorization
        // does fail, so the test genuinely exercises the retry branch (it would fail
        // to reach Ok if the retry were removed).
        let sigma_d = build_sigma_design(&model, &[10, 10]);
        assert!(
            crate::linalg::cholesky_factor(&sigma_d, 2).is_err(),
            "test precondition: raw Σ_d must be non-PD so the retry branch is exercised"
        );
    }

    #[test]
    fn test_validation_grid_too_small() {
        // m = 1: a Simpson quadrature / trajectory integral is undefined. Construct the
        // model directly (synthetic_model_params divides by m-1, so it can't build m=1).
        let model = PaceFpcaResult {
            mean: vec![0.0; 1],
            eigenvalues: vec![2.0, 1.0],
            eigenfunctions: FdMatrix::from_column_major(vec![1.0, 0.5], 1, 2).unwrap(),
            scores: FdMatrix::zeros(1, 2),
            fitted: FdMatrix::zeros(1, 1),
            fitted_lower: FdMatrix::zeros(1, 1),
            fitted_upper: FdMatrix::zeros(1, 1),
            argvals: vec![0.0],
            sigma2: 0.5,
            ncomp: 2,
        };
        let res = design_criterion(&model, &[], DesignCriterion::Trajectory);
        assert!(
            matches!(res, Err(FdarError::InvalidParameter { parameter, .. }) if parameter == "model.argvals"),
            "m<2 must be rejected with InvalidParameter(model.argvals), got {res:?}"
        );
    }

    // ---- Score branch (FOD-02) ----

    #[test]
    fn test_score_a_empty_set() {
        let model = synthetic_model(51);
        let a = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::A)).unwrap();
        // A(∅) = Σ_k λ_k = 3.0
        assert!((a - 3.0).abs() < 1e-10, "A(∅) = {a}, expected 3.0");
    }

    #[test]
    fn test_score_d_empty_set() {
        let model = synthetic_model(51);
        let d = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::D)).unwrap();
        // D(∅) = ln(2.0) + ln(1.0) = ln 2
        let expected = 2.0_f64.ln();
        assert!(
            (d - expected).abs() < 1e-10,
            "D(∅) = {d}, expected {expected}"
        );
    }

    #[test]
    fn test_score_prior_recovery() {
        let model = synthetic_model(51);
        let a = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::A)).unwrap();
        let expected_a: f64 = model.eigenvalues.iter().sum();
        assert!(
            (a - expected_a).abs() < 1e-10,
            "a={a} expected_a={expected_a}"
        );

        let d = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::D)).unwrap();
        let expected_d: f64 = model.eigenvalues.iter().map(|&lam| lam.ln()).sum();
        assert!(
            (d - expected_d).abs() < 1e-10,
            "d={d} expected_d={expected_d}"
        );
    }

    #[test]
    fn test_monotonicity_a_opt() {
        let model = synthetic_model(51);
        let s0 =
            design_criterion(&model, &[10], DesignCriterion::Score(OptimalityKind::A)).unwrap();
        let s1 =
            design_criterion(&model, &[10, 30], DesignCriterion::Score(OptimalityKind::A)).unwrap();
        assert!(s1 <= s0 + 1e-12, "s1={s1} s0={s0}");
    }

    #[test]
    fn test_monotonicity_d_opt() {
        let model = synthetic_model(51);
        let s0 =
            design_criterion(&model, &[10], DesignCriterion::Score(OptimalityKind::D)).unwrap();
        let s1 =
            design_criterion(&model, &[10, 30], DesignCriterion::Score(OptimalityKind::D)).unwrap();
        assert!(s1 <= s0 + 1e-12, "s1={s1} s0={s0}");
    }

    #[test]
    fn test_enum_dispatch() {
        let model = synthetic_model(51);
        let traj = design_criterion(&model, &[10], DesignCriterion::Trajectory).unwrap();
        let a = design_criterion(&model, &[10], DesignCriterion::Score(OptimalityKind::A)).unwrap();
        let d = design_criterion(&model, &[10], DesignCriterion::Score(OptimalityKind::D)).unwrap();
        assert!(
            traj.is_finite() && a.is_finite() && d.is_finite(),
            "traj={traj} a={a} d={d}"
        );
        // Route-correctness. NOTE: when eigenfunctions are orthonormal w.r.t. the
        // integration weights, the integrated trajectory MSE equals trace(Cov(ξ)),
        // so Trajectory ≡ A-optimality is an exact algebraic identity — not a
        // dispatch bug. We assert that identity (proving Trajectory runs the real
        // integral, not a stub) AND that D (log-det, a distinct code path) yields a
        // value distinct from both, confirming all three variants route separately.
        assert!(
            (traj - a).abs() < 1e-9,
            "orthonormal identity broken: traj={traj} a={a}"
        );
        assert!(
            (d - a).abs() > 1e-9,
            "D failed to route separately: d={d} a={a}"
        );
        assert!(
            d < a,
            "D-opt (log-det) should be below A-opt (trace) here: d={d} a={a}"
        );
    }

    // ---- Greedy selection wrapper (FOD-04 / FOD-05) ----

    #[test]
    fn test_optimal_design_basic() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 3,
            criterion: DesignCriterion::Trajectory,
        };
        let r = optimal_design(&model, &config).unwrap();
        assert_eq!(r.selected_indices.len(), 3);
        assert_eq!(r.selected_argvals.len(), 3);
        assert_eq!(r.criterion_trace.len(), 3);
    }

    #[test]
    fn test_determinism_two_calls() {
        // Doubles as the seq==parallel gate: run under BOTH default and
        // `--features parallel`; the selection must be byte-identical either way.
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 3,
            criterion: DesignCriterion::Trajectory,
        };
        let r1 = optimal_design(&model, &config).expect("first call");
        let r2 = optimal_design(&model, &config).expect("second call");
        assert_eq!(
            r1.selected_indices, r2.selected_indices,
            "selection must be deterministic"
        );
        assert_eq!(
            r1.criterion_trace, r2.criterion_trace,
            "trace must be deterministic"
        );
    }

    #[test]
    fn test_duplicate_free() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 5,
            criterion: DesignCriterion::Trajectory,
        };
        let r = optimal_design(&model, &config).unwrap();
        let mut sorted = r.selected_indices.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            r.selected_indices.len(),
            "no index may appear twice: {:?}",
            r.selected_indices
        );
    }

    #[test]
    fn test_monotone_trace() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 5,
            criterion: DesignCriterion::Trajectory,
        };
        let r = optimal_design(&model, &config).unwrap();
        for w in r.criterion_trace.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-12,
                "trace not monotone non-increasing: {:?}",
                r.criterion_trace
            );
        }
    }

    #[test]
    fn test_validation_budget_zero() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 0,
            criterion: DesignCriterion::Trajectory,
        };
        let res = optimal_design(&model, &config);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_budget_exceeds_grid() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: vec![model.argvals[0], model.argvals[1]],
            budget: 3,
            criterion: DesignCriterion::Trajectory,
        };
        let res = optimal_design(&model, &config);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_off_grid_candidate() {
        let model = synthetic_model(51);
        // A value strictly between two grid points, well outside the 1e-9 tolerance.
        let off_grid = model.argvals[0] + 0.5 / (51.0 - 1.0);
        let config = OptDesConfig {
            candidate_grid: vec![off_grid],
            budget: 1,
            criterion: DesignCriterion::Trajectory,
        };
        let res = optimal_design(&model, &config);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_ncomp_zero() {
        // ncomp == 0 (empty eigenvalues). May be caught at entry or delegated to
        // design_criterion — either way an InvalidParameter must surface.
        let model = synthetic_model_params(51, vec![], 0.5);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 1,
            criterion: DesignCriterion::Trajectory,
        };
        let res = optimal_design(&model, &config);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_validation_sigma2_nonpositive() {
        let model = synthetic_model_params(51, vec![2.0, 1.0], 0.0);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 1,
            criterion: DesignCriterion::Trajectory,
        };
        let res = optimal_design(&model, &config);
        assert!(matches!(res, Err(FdarError::InvalidParameter { .. })));
    }

    #[test]
    fn test_trajectory_selects_informative_point() {
        let model = synthetic_model(51);
        let m = model.argvals.len();
        // Compute the expected first index numerically: sequential smallest-index
        // argmin of the single-point Trajectory criterion over ALL candidates.
        let mut best: Option<(usize, f64)> = None;
        for idx in 0..m {
            let val = design_criterion(&model, &[idx], DesignCriterion::Trajectory).unwrap();
            best = Some(match best {
                None => (idx, val),
                Some((bi, bv)) => {
                    if val < bv {
                        (idx, val)
                    } else {
                        (bi, bv)
                    }
                }
            });
        }
        let expected_first = best.unwrap().0;

        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 2,
            criterion: DesignCriterion::Trajectory,
        };
        let r = optimal_design(&model, &config).unwrap();
        assert_eq!(
            r.selected_indices[0], expected_first,
            "first greedy pick must equal the numerically-computed argmin"
        );
    }

    #[test]
    fn test_score_a_selects() {
        let model = synthetic_model(51);
        let config = OptDesConfig {
            candidate_grid: model.argvals.clone(),
            budget: 2,
            criterion: DesignCriterion::Score(OptimalityKind::A),
        };
        let r = optimal_design(&model, &config).unwrap();
        assert_eq!(r.selected_indices.len(), 2);
        assert_eq!(r.criterion_trace.len(), 2);
        for w in r.criterion_trace.windows(2) {
            assert!(w[1] <= w[0] + 1e-12, "Score(A) trace not non-increasing");
        }
    }

    #[test]
    fn test_config_default() {
        // Default constructs (empty grid, budget 1, Trajectory); the empty grid is
        // caught at call time, NOT at construction.
        let config = OptDesConfig::default();
        assert_eq!(config.budget, 1);
        assert!(config.candidate_grid.is_empty());
        assert_eq!(config.criterion, DesignCriterion::Trajectory);
        let model = synthetic_model(51);
        let res = optimal_design(&model, &config);
        assert!(
            matches!(res, Err(FdarError::InvalidParameter { .. })),
            "empty grid + budget 1 must fail at call time (budget > grid.len())"
        );
    }

    #[test]
    fn test_prelude_reexport() {
        // In-crate reachability placeholder. The external prelude/crate-root
        // reachability is verified as a doctest in plan 65-02.
        assert_eq!(OptDesConfig::default().budget, 1);
    }
}
