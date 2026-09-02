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
//! formulation already implemented in [`crate::pace_fpca`]. This module is the pure
//! numerical core; the greedy selection wrapper is a separate concern.

use crate::error::FdarError;
use crate::helpers::simpsons_weights;
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
    /// D-optimality: log-determinant of the posterior score covariance. The value
    /// is NEGATIVE for an informative design (posterior eigenvalues ≤ prior λ_k)
    /// and is returned un-negated. Empty design returns `Σ_k log λ_k`.
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
            // log det(Cov) via Cholesky; NEGATIVE for an informative design. Do NOT negate.
            let l_cov = cholesky_factor(&cov, ncomp).map_err(|_| FdarError::ComputationFailed {
                operation: "optimal_design D-optimality log-det",
                detail: "posterior covariance Cholesky failed; matrix not positive-definite".into(),
            })?;
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
        // Near-singular regime: sigma2 = 1e-12 → Cholesky of Σ_d may fail → ridge-retry.
        let model = synthetic_model_params(51, vec![2.0, 1.0], 1e-12);
        let res = design_criterion(&model, &[10, 20, 30], DesignCriterion::Trajectory);
        assert!(res.is_ok(), "expected Ok after ridge-retry, got {res:?}");
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
}
