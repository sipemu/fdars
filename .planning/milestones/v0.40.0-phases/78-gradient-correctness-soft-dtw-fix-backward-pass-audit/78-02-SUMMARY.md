---
phase: 78-gradient-correctness-soft-dtw-fix-backward-pass-audit
plan: 02
status: complete
provides: [CORR-02]
key-files:
  - .planning/phases/78-gradient-correctness-soft-dtw-fix-backward-pass-audit/78-AUDIT.md
completed: 2026-09-06
---

## Accomplishments

### Task 1: CORR-02 Module Sweep (read-and-verify)

Each of the 9 CORR-02 target modules was opened at its cited functions and
confirmed against current source:

1. **`metric/soft_dtw.rs`** — `soft_dtw_backward` — DP backward pass with
   overwriteable endpoint seed: **FIXED** (CORR-01 / Plan 01, commit `d99569c7`).

2. **`alignment/differentiable.rs`** — `generic_linear_interp`, `generic_srsf_central_diff`,
   `generic_l2_srsf_distance`, `amplitude_distance_at_warp_generic` — forward-mode Dual
   pipeline; boundary conditions are value-level clamps/one-sided stencils, not
   gradient seeds: **clean**.

3. **`autodiff.rs`** — `diff`, `grad`, `jacobian`, `directional_derivative`, `Dual::seed` —
   the forward-mode AD substrate itself; no reverse pass exists; `Dual::seed` is a
   constructor with no boundary logic: **clean**.

4. **`boosting_regression/gamlss.rs`** — `mu_neg_gradient`, `sigma_neg_gradient` —
   pointwise closed-form Gaussian log-likelihood gradients with `sigma_floor`/
   `NUMERICAL_EPS` stability guards; no DP, no recurrence: **clean**.

5. **`elastic_regression/logistic.rs`** — `logistic_gradients`, `armijo_line_search_logistic` —
   standard logistic gradient + Armijo line search; `prob = sigmoid(η)` is element-wise;
   no backward pass: **clean**.

6. **`explain_generic/counterfactual.rs`** — `compute_gradient_finite_diff`,
   `counterfactual_gd_search` — forward-difference numerical gradient over FPC score
   components; no recurrence or boundary seed: **clean**.

7. **`regression.rs`** — `project_scores_generic` — linear projection generic over
   `Scalar`; exact forward-mode Dual gradient of a linear map; no boundary conditions: **clean**.

8. **`seasonal/mod.rs`** — `refine_period_gradient` — three-point stencil scalar
   hill-climber; not a gradient backward pass; only a floor guard `period.max(dt)`: **clean**.

9. **`smooth_basis.rs`** — `differentiate_basis_columns`; `gradient_uniform` in `helpers.rs` —
   numerical differentiation for penalty-matrix construction; standard boundary stencils
   (one-sided difference formulas); no endpoint seed to overwrite: **clean**.

No new localized boundary-seed bug was found in any sibling module.

### Task 2: 78-AUDIT.md Written

`.planning/phases/78-gradient-correctness-soft-dtw-fix-backward-pass-audit/78-AUDIT.md`
was written with a 9-row disposition table (module path, audited functions,
mechanism, boundary seed yes/no, disposition, one-line rationale).  The
`metric/soft_dtw` row is `fixed` and cross-references CORR-01/Plan 01
commit `d99569c7`.  Closes with a summary statement that no additional
boundary-seed bugs were found.

## Task Commits

| Task | Hash | Description |
|------|------|-------------|
| Task 2 (artifact) | `b8936ff0` | docs(78-02): write CORR-02 audit disposition table (78-AUDIT.md) |

(Task 1 produced no code change — read-only verification.)

## Files Modified

- `.planning/phases/78-gradient-correctness-soft-dtw-fix-backward-pass-audit/78-AUDIT.md` — new artifact

## Verification

### Audit source files present
```
test -f fdars-core/src/alignment/differentiable.rs && ...  && echo AUDIT_SOURCES_PRESENT
# AUDIT_SOURCES_PRESENT
```

### AUDIT_OK check
```
test -f .planning/.../78-AUDIT.md && grep -qi 'soft_dtw' ... && [ row-count -ge 9 ] && echo AUDIT_OK
# AUDIT_OK
```

### No code gates needed
Plan 02 made no code changes; cargo gates are not required.

## CORR-02 Disposition Summary

| Disposition | Count | Modules |
|-------------|-------|---------|
| fixed | 1 | `metric/soft_dtw` (CORR-01) |
| clean | 8 | `alignment/differentiable`, `autodiff`, `boosting_regression/gamlss`, `elastic_regression/logistic`, `explain_generic/counterfactual`, `regression`, `seasonal/mod`, `smooth_basis` |
| deferred | 0 | — |

The endpoint-seed overwrite pattern is unique to the DTW backward algorithm
in this codebase. All 8 sibling gradient passes use mechanism types (closed-form
analytical, forward-mode Dual, finite-difference, scalar hill-climber) that have
no analogous boundary-seed overwrite vulnerability.

## Reference

Audit trail: see `78-AUDIT.md` in this phase directory.
