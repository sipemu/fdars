# Phase 78 — CORR-02 Backward-Pass Audit

**Phase:** 78-gradient-correctness-soft-dtw-fix-backward-pass-audit
**Requirement:** CORR-02
**Date:** 2026-09-06
**Audited:** Every hand-written backward/gradient pass in `fdars-core` that could
contain a boundary-seed overwrite bug analogous to the `soft_dtw_backward`
endpoint-seed defect (CORR-01).

---

## Disposition Table

| # | Module | Audited Functions | Gradient Mechanism | Overwriteable Boundary Seed? | Disposition | Rationale |
|---|--------|-------------------|--------------------|------------------------------|-------------|-----------|
| 1 | `fdars-core/src/metric/soft_dtw.rs` | `soft_dtw_backward` (lines 262–306) | DP backward pass with pre-set endpoint seed `E[n][m] = 1.0` | **YES** (was overwritten at line 302 by `e[i][j] = a + b + c` at endpoint) | **fixed** | CORR-01 / Plan 01: inserted `if i == n && j == m { continue; }` guard; see commit `d99569c7` |
| 2 | `fdars-core/src/alignment/differentiable.rs` | `generic_linear_interp` (33–55), `generic_srsf_central_diff` (64–87), `generic_l2_srsf_distance` (93–100), `amplitude_distance_at_warp_generic` (125–138) | Forward-mode `Dual` pipeline (no backward pass) | No — boundary conditions are value-level clamps for interpolation/differentiation, not gradient seeds | **clean** | Pure forward-mode Dual chain rule; boundary handling at endpoints is clamping (lines 38–42) and one-sided finite-difference stencils (lines 77–81), not overwriteable seeds |
| 3 | `fdars-core/src/autodiff.rs` | `diff` (484), `grad` (511), `jacobian` (559), `directional_derivative` (611); `Dual::seed` (239), `Dual::constant` (248) | Forward-mode AD substrate; no backward (reverse) pass exists | No — `Dual::seed` / `Dual::constant` are constructors with no boundary logic | **clean** | This module IS the forward-mode foundation; reverse-mode AD is explicitly deferred (DIF-F1). No DP recurrence, no endpoint seed pattern |
| 4 | `fdars-core/src/boosting_regression/gamlss.rs` | `mu_neg_gradient` (64–79), `sigma_neg_gradient` (86–101) | Closed-form pointwise Gaussian log-likelihood analytical gradients | No — only a `sigma_floor` guard and `NUMERICAL_EPS` clip for numerical stability | **clean** | Analytical formulas `(Y−μ)/σ²` and `−1 + (Y−μ)²/σ²` computed pointwise; no DP structure, no recurrence, no endpoint seed |
| 5 | `fdars-core/src/elastic_regression/logistic.rs` | `logistic_gradients` (466–496), `armijo_line_search_logistic` (499–531) | Closed-form binary cross-entropy + L2 penalty analytical gradient | No — no DP, no recurrence; `prob = sigmoid(η)` is element-wise | **clean** | Standard logistic gradient `mean(prob − target)` and `mean((prob − target) * q * w) + λβ` with Armijo line search; no backward-pass seed pattern |
| 6 | `fdars-core/src/explain_generic/counterfactual.rs` | `compute_gradient_finite_diff` (7–22), `counterfactual_gd_search` (58–83) | Finite-difference numerical gradient (forward-difference, eps = 1e-5) | No — no DP, no recurrence; each perturbation is independent | **clean** | Forward-difference approximation over FPC score components; no recurrence structure means no boundary seed to overwrite |
| 7 | `fdars-core/src/regression.rs` | `project_scores_generic` (232–251) | Forward-mode `Dual` linear projection | No — pure linear map with no boundary conditions | **clean** | `score_k = Σ_j (curve[j] − mean[j]) * rotation[(j,k)] * weights[j]`; analytic gradient of a linear function; no recurrence, no seed |
| 8 | `fdars-core/src/seasonal/mod.rs` | `refine_period_gradient` (1112–1135) | Three-point stencil scalar hill-climber (not a gradient backward pass) | No — only a floor guard `period.max(dt)` | **clean** | Evaluates ACF score at `period ± step_size` and moves toward the highest; no DP, no matrix propagation, no endpoint seed |
| 9 | `fdars-core/src/smooth_basis.rs` | `differentiate_basis_columns` (984–1005); `gradient_uniform` in `helpers.rs` (747–784) | Numerical differentiation via `gradient_uniform` (central differences + boundary stencils) | No — boundary stencils are standard one-sided difference formulas, not gradient seeds | **clean** | `differentiate_basis_columns` iteratively applies `gradient_uniform` to build a penalty matrix; no recurrence, no endpoint value that is pre-set and then recomputed |

---

## Fix Reference

- **Row 1 (`metric/soft_dtw`)**: `fixed` by CORR-01 / Plan 01, commit `d99569c7`.
  Regression test `soft_dtw_backward_nonzero_and_matches_oracle` asserts
  non-zero E matrix + oracle/Dual gradient match within 1e-6.

---

## Summary

The CORR-02 sweep found **no additional boundary-seed bugs** beyond
`metric/soft_dtw`.  All 8 sibling modules use closed-form analytical
gradients, forward-mode `Dual` AD, finite-difference approximations, or a
scalar hill-climber — none has a DP backward pass with a pre-set endpoint
seed that could be overwritten by a general update rule.  The endpoint-seed
overwrite pattern is unique to the DTW backward algorithm in this codebase.
