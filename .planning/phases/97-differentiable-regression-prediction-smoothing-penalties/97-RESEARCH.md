# Phase 97: Differentiable Regression Prediction & Smoothing Penalties — Research

**Researched:** 2026-09-11
**Domain:** Rust autodiff — FPCR prediction generalization (DOP-02) + roughness-penalty value evaluation (DOP-03)
**Confidence:** HIGH (all claims verified from source files read this session)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **DOP-02:** new per-curve generic prediction `predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S`:
  1. `let scores = project_scores_generic::<S>(curve, &fit.fpca.mean, &fit.fpca.rotation, &fit.fpca.weights, fit.ncomp);`
  2. `let mut yhat = S::from_f64(fit.intercept); for k in 0..fit.ncomp { yhat += S::from_f64(fit.coefficients[1 + k]) * scores[k]; }`
  3. return `yhat`.
- **`predict_fregre_lm` public signature UNCHANGED.** May delegate its inner per-curve loop to `predict_curve_generic::<f64>`. The existing `test_predict_fregre_lm_on_training_data` (tol 1e-6) is the DOP-02 non-breaking guard.
- **Scalar covariates stay f64** — not part of the generic per-curve fn.
- **DOP-03:** new standalone `penalty_value_generic<S: Scalar>(coef: &[S], penalty: &FdMatrix, lambda: f64) -> S` computing `λ·cᵀRc`, differentiable w.r.t. `coef`. R matrix + lambda stay f64. Penalty-matrix constructors unchanged.
- **Plain `<S: Scalar>` (no `= f64` default)** — Rust 1.97 rejects `invalid_type_param_default` on free fns.
- Additive new functions only; existing public signatures frozen.
- No new crate dependency.

### Claude's Discretion

- Exact module placement of `predict_curve_generic` (scalar_on_function/fregre_lm.rs beside `predict_fregre_lm`) and `penalty_value_generic` (smooth_basis.rs or a small penalty module).
- Penalty-matrix carrier type (`&FdMatrix` vs `&[f64]`+dims) — must interoperate with `bspline_penalty_matrix`/`fourier_penalty_matrix` (return `Vec<f64>` column-major) and `difference_matrix` (returns `DMatrix<f64>`; `DᵀD` is the penalty for P-splines). Provide a small adapter if needed.
- Whether to expose both new functions at crate root/prelude (prefer yes, additive).

### Deferred Ideas (OUT OF SCOPE)

- Differentiating the penalty w.r.t. raw curve values through the smoothing solve, or w.r.t. λ.
- Making scalar covariates a differentiable input.
- Depth + curve distances → Phase 98 (DOP-04).
- Unified gradient API + composition demo → Phase 99 (GEN-02/API-01).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DOP-02 | Scalar-on-function regression prediction (`fregre_lm` / FPCR path) generic over `Scalar`, differentiable w.r.t. inputs; f64 parity preserved, FD-checked. | `predict_curve_generic<S>` recipe in §Implementation Recipes (P-1). Accumulation-order divergence from batch kernel documented in §Risk P-1a — parity is within 1e-6, NOT bit-identical. |
| DOP-03 | Smoothing / roughness-penalty evaluation generic over `Scalar`, differentiable w.r.t. coefficients `coef`; f64 parity preserved, FD-checked. | `penalty_value_generic<S>` recipe in §Implementation Recipes (Q-1). f64 parity is bit-identical (new function, no pre-existing reference). Gradient `2λRc` for symmetric R (confirmed symmetric by construction — §Risk Q-1b). |
</phase_requirements>

---

## Summary

Phase 97 adds two new generic-over-`Scalar` functions to the fdars-core autodiff substrate: `predict_curve_generic<S>` (DOP-02) and `penalty_value_generic<S>` (DOP-03). Both are purely additive — no public signature changes, no new crate dependency, no alteration of existing fits or solves.

The implementation is straightforward in structure but carries two important precision risks that must be documented explicitly to prevent the planner from writing incorrect assertions:

1. **Accumulation-order divergence (DOP-02, Risk P-1a):** `predict_curve_generic::<f64>` is NOT bit-identical to the current `predict_fregre_lm` inner kernel. The batch kernel accumulates `(x-mean) * rotation * weights` as `((x-mean) * rotation) * weights` (two sequential f64 multiplies). `project_scores_generic` (which `predict_curve_generic` composes) pre-multiplies `rotation[(j,k)] * weights[j]` as a single f64, then multiplies `(x-mean) * (rot*w)`. These differ in floating-point rounding. The existing test tolerance of 1e-6 covers this gap; no assertion of bit-identity is valid. If `predict_fregre_lm` is refactored to call `predict_curve_generic::<f64>` in its inner loop, the batch function output WILL CHANGE at the sub-1e-12 level — the test remains green, but callers that snapshot exact outputs (e.g., integration tests comparing to stored values) may be affected.

2. **DOP-03 parity is bit-identical (Risk Q-1b):** `penalty_value_generic::<f64>` is a brand-new standalone function with no pre-existing counterpart. The f64 parity target is a direct inline reference computation `Σ coef[i]*R[i,j]*coef[j]*lambda` — bit-identical because `f64::from_f64(v) == v` is the identity. The penalty matrices built by `integrate_symmetric_penalty` are exactly symmetric by construction (entries mirrored explicitly: `penalty[j+l*k] = val; penalty[l+j*k] = val`), so the gradient `2λRc` is exact.

**Primary recommendation:** Place `predict_curve_generic` in `scalar_on_function/fregre_lm.rs` alongside the existing `predict_fregre_lm`. Place `penalty_value_generic` in `smooth_basis.rs` (the module that already owns `bspline_penalty_matrix` and `fourier_penalty_matrix`). Accept `penalty: &FdMatrix` as the penalty-matrix carrier — wrap `Vec<f64>` penalty outputs via `FdMatrix::from_column_major` at the call site (no change to constructors). Export both from `lib.rs` and `prelude.rs` additively.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `predict_curve_generic<S>` implementation | `scalar_on_function/fregre_lm.rs` | — | Lives beside `predict_fregre_lm`; shares `FregreLmResult` and imports `project_scores_generic` |
| `penalty_value_generic<S>` implementation | `smooth_basis.rs` | — | Owned by the module that builds the penalty matrices; natural co-location |
| Autodiff gradient propagation | `autodiff/` (Phase 94/95 substrate) | both new fns | `Scalar` trait impls for `Dual`/`Var` are already in `autodiff/`; new fns just use `S` arithmetic |
| Public API surface | `lib.rs` + `prelude.rs` | `scalar_on_function/mod.rs` | Additive re-exports; existing paths frozen |
| Non-breaking proof | clippy `--all-targets` + full test suite | — | `predict_fregre_lm` batch signature unchanged; penalty constructors unchanged |

---

## Standard Stack

### Core (no new dependencies)

| Component | Version | Purpose | Status |
|-----------|---------|---------|--------|
| `crate::autodiff::Scalar` | in-crate | Trait bounding S | `[VERIFIED: fdars-core/src/autodiff/mod.rs:38-85]` |
| `crate::autodiff::Dual` | in-crate | Forward-mode S | `[VERIFIED: fdars-core/src/autodiff/forward.rs:106-109]` — `f64::from_f64` is identity |
| `crate::autodiff::Var` + `vjp` | in-crate | Reverse-mode S | `[VERIFIED: fdars-core/src/autodiff/reverse.rs]` |
| `project_scores_generic<S>` | in-crate Phase 94 | FPCR score projection — reused unchanged | `[VERIFIED: fdars-core/src/regression.rs:232-251]` |
| `FpcaResult` | in-crate | Carries `mean`, `rotation`, `weights`, `ncomp` | `[VERIFIED: fdars-core/src/regression.rs:130-147]` |
| `FregreLmResult` | in-crate | Carries `intercept`, `fpca`, `coefficients`, `ncomp`, `gamma` | `[VERIFIED: fdars-core/src/scalar_on_function/mod.rs:62-98]` |
| `FdMatrix` | in-crate | Column-major matrix carrier for penalty R | `[VERIFIED: fdars-core/src/matrix.rs]` |

**Installation:** No new dependencies.

---

## Package Legitimacy Audit

No external packages are installed in this phase. The entire implementation is in-crate. Not applicable.

---

## Architecture Patterns

### System Architecture Diagram

```
curve: &[S]  (differentiable input — DOP-02)
    │
    ▼
predict_curve_generic<S>(curve, fit: &FregreLmResult)   [scalar_on_function/fregre_lm.rs]
    │
    ├─► project_scores_generic<S>(curve, &fit.fpca.mean, &fit.fpca.rotation, &fit.fpca.weights, fit.ncomp)
    │       [REUSE — regression.rs:232, Phase 94; not re-touched]
    │       per j: centered = curve[j] - S::from_f64(mean[j])
    │              w_rot    = S::from_f64(rotation[(j,k)] * weights[j])   ← f64 pre-multiply
    │              score_k += centered * w_rot
    │
    └─► yhat = S::from_f64(intercept)
            + Σ_k S::from_f64(coefficients[1+k]) * scores[k]
    returns S

predict_fregre_lm(fit, new_data, new_scalar) → Vec<f64>   [unchanged public signature]
    │  (may refactor inner loop to call predict_curve_generic::<f64> + scalar additive term)
    └─► returns Vec<f64>  [parity: within 1e-6 of current, NOT bit-identical — see Risk P-1a]

coef: &[S]  (differentiable input — DOP-03)
    │
    ▼
penalty_value_generic<S>(coef, penalty: &FdMatrix, lambda: f64)   [smooth_basis.rs]
    │  R: &FdMatrix  (column-major, stays f64)
    │  lambda: f64   (stays f64)
    │  sum = S::zero()
    │  for i { for j { sum += coef[i] * S::from_f64(R[i,j]) * coef[j] } }
    └─► returns S::from_f64(lambda) * sum    [= λ·cᵀRc]
```

### Recommended Project Structure

```
fdars-core/src/
├── scalar_on_function/
│   └── fregre_lm.rs        # Add: predict_curve_generic<S>; optionally refactor
│                            #      predict_fregre_lm inner loop to call it at S=f64
├── smooth_basis.rs          # Add: penalty_value_generic<S>
├── lib.rs                   # Add: re-export predict_curve_generic, penalty_value_generic
└── prelude.rs               # Add: re-export predict_curve_generic, penalty_value_generic
```

No new files required. Tests are added inline in the existing `#[cfg(test)]` modules.

---

## Detailed Implementation Recipes

### P-1: `predict_curve_generic<S>` — Exact Form

**Source:** Derived from `project_scores_generic` at `[VERIFIED: fdars-core/src/regression.rs:232-251]` and the batch kernel at `[VERIFIED: fdars-core/src/scalar_on_function/fregre_lm.rs:454-471]`.

```rust
// In: scalar_on_function/fregre_lm.rs
use crate::autodiff::Scalar;
use crate::regression::project_scores_generic;

/// Generic-over-[`Scalar`] per-curve prediction for a fitted functional linear model.
///
/// Computes `intercept + Σ_k coefficients[1+k] · score_k(curve)` where scores
/// are projected via [`project_scores_generic`]. Differentiable w.r.t. `curve`;
/// all model parameters (`intercept`, `coefficients`, fpca `mean`/`rotation`/`weights`)
/// stay `f64`, lifted via `S::from_f64` only at the point of mixing.
///
/// Instantiated at `S = f64` this reproduces `predict_fregre_lm`'s per-curve result
/// within ~1e-12 (not bit-identical — see accumulation-order note in crate docs).
/// Scalar covariates (`gamma`) are NOT included; they remain an additive f64 term
/// in the batch [`predict_fregre_lm`].
#[must_use]
pub fn predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S {
    let scores = project_scores_generic(
        curve,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        &fit.fpca.weights,
        fit.ncomp,
    );
    let mut yhat = S::from_f64(fit.intercept);
    for k in 0..fit.ncomp {
        yhat += S::from_f64(fit.coefficients[1 + k]) * scores[k];
    }
    yhat
}
```

**Required import in fregre_lm.rs:** `use crate::autodiff::Scalar;` and `use crate::regression::project_scores_generic;`. The second import may require checking if `project_scores_generic` is already in scope (it is in `regression.rs`, re-exported from `lib.rs` but the submodule must import directly).

### P-1a: Risk — Accumulation-Order Divergence (critical)

**The issue:** `project_scores_generic` pre-multiplies `rotation[(j,k)] * weights[j]` as a single f64:
```rust
// regression.rs:245
let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);  // f64 multiply first
sum += centered * w_rot;                                   // then T multiply
```
`[VERIFIED: fdars-core/src/regression.rs:243-247]`

The current batch kernel does:
```rust
// fregre_lm.rs:459-461
s += (new_data[(i, j)] - fit.fpca.mean[j])
    * fit.fpca.rotation[(j, k)]
    * fit.fpca.weights[j];
```
`[VERIFIED: fdars-core/src/scalar_on_function/fregre_lm.rs:458-462]`

This is `((x-mean) * rotation) * weights` — three sequential f64 multiplies. The `project_scores_generic` form is `(x-mean) * (rotation * weights)` — the product `rotation*weights` is folded first.

**Consequence:** `predict_curve_generic::<f64>` ≠ batch kernel result in general. Difference is sub-1e-12 (typically 0–2 ULP per accumulation step), but accumulates across the `j` loop. Over `m=50` evaluation points and `ncomp=3` components, the worst-case absolute error is on the order of `m * 2 * eps ≈ 50 * 2 * 2.2e-16 ≈ 2e-14` — well within 1e-6.

**What this means for the plan:**
- The parity test for DOP-02 MUST use tolerance 1e-6 (matching the existing `test_predict_fregre_lm_on_training_data`), NOT `assert_eq!` or 1e-12.
- If `predict_fregre_lm` is refactored to call `predict_curve_generic::<f64>` in its inner loop, the `fitted_values` stored in the `FregreLmResult` (computed during fitting using the same 3-mul kernel) will no longer match exactly the predictions from `predict_fregre_lm` on training data — but the existing test at 1e-6 still passes.
- Do NOT state "bit-identical" anywhere in DOP-02 task descriptions. Use "within 1e-6" instead.

**Decision:** The CONTEXT.md says "refactor its inner per-curve loop to call `predict_curve_generic::<f64>` so the batch result is identical (bit-for-bit)". This is INCORRECT — the batch result will NOT be bit-for-bit identical after the refactor. The planner must note this discrepancy and use "within 1e-6" as the parity standard, consistent with the existing test.

**Mitigation options (Claude's discretion):**
1. Do NOT refactor `predict_fregre_lm`'s inner loop — keep the original 3-mul accumulation in the batch path. `predict_curve_generic::<f64>` exists as an independent generic entry point, not a rewrite of the batch. This is the lowest-risk option: the existing test stays green trivially.
2. Refactor `predict_fregre_lm` to call `predict_curve_generic::<f64>`. The existing 1e-6-tolerance test still passes; the output changes at the ~1e-14 level. This is clean but requires documenting the sub-1e-12 output change.

**Recommendation:** Option 1. Do not refactor `predict_fregre_lm`'s inner loop. The batch function is the f64 performance path; the generic fn is the autodiff entry point. Both are correct; they produce results within 1e-6 of each other, and that is the stated non-breaking guarantee.

### Q-1: `penalty_value_generic<S>` — Exact Form

**Source:** Derived from the penalty-matrix layout at `[VERIFIED: fdars-core/src/smooth_basis.rs:1010-1028]` and the CONTEXT.md §DOP-03 recipe.

```rust
// In: smooth_basis.rs
use crate::autodiff::Scalar;

/// Generic-over-[`Scalar`] roughness-penalty value `λ · cᵀRc`.
///
/// Evaluates the scalar quadratic form `lambda * Σ_i Σ_j coef[i] * R[i,j] * coef[j]`
/// where `R` (the penalty matrix, e.g. from [`bspline_penalty_matrix`] or
/// [`fourier_penalty_matrix`]) stays `f64` (column-major, `K × K`) and `lambda`
/// stays `f64`. Only `coef` (the spline coefficients) is the differentiable `S` input.
///
/// Analytic gradient: `d(λ cᵀRc)/d(coef_i) = λ·((R + Rᵀ)c)_i = 2λ(Rc)_i`
/// for symmetric R. The B-spline and Fourier penalty matrices are exactly symmetric
/// by construction ([`integrate_symmetric_penalty`] mirrors entries explicitly).
///
/// # Arguments
/// * `coef`    — Spline coefficient vector `c` (length K), differentiable.
/// * `penalty` — Penalty matrix R in column-major layout (K × K), stays `f64`.
/// * `lambda`  — Smoothing parameter (non-negative), stays `f64`.
#[must_use]
pub fn penalty_value_generic<S: Scalar>(
    coef: &[S],
    penalty: &FdMatrix,
    lambda: f64,
) -> S {
    let k = coef.len();
    debug_assert_eq!(
        penalty.nrows(), k,
        "penalty must be k×k; got {}×{}", penalty.nrows(), penalty.ncols()
    );
    let mut sum = S::zero();
    for i in 0..k {
        for j in 0..k {
            sum += coef[i] * S::from_f64(penalty[(i, j)]) * coef[j];
        }
    }
    S::from_f64(lambda) * sum
}
```

**Required import in smooth_basis.rs:** `use crate::autodiff::Scalar;`. The file already imports `use crate::matrix::FdMatrix;` `[VERIFIED: fdars-core/src/smooth_basis.rs:14]`.

### Q-1a: Penalty-Matrix Carrier — `&FdMatrix` vs `&[f64]`

**Decision:** Use `penalty: &FdMatrix`. Rationale:

- `FdMatrix` is the crate's canonical column-major matrix type. `[VERIFIED: fdars-core/src/matrix.rs]`
- `bspline_penalty_matrix` returns `Vec<f64>` column-major `[VERIFIED: fdars-core/src/smooth_basis.rs:112-146]`. Call sites wrap it: `FdMatrix::from_column_major(penalty_vec, k, k).unwrap()`.
- `fourier_penalty_matrix` returns `Vec<f64>` column-major `[VERIFIED: fdars-core/src/smooth_basis.rs:159-188]`. Same adapter.
- `difference_matrix` returns `DMatrix<f64>` `[VERIFIED: fdars-core/src/basis/pspline.rs:9-37]`. For `DᵀD`, call sites compute `d.transpose() * &d` (nalgebra), get a `DMatrix<f64>`, then wrap via `FdMatrix::from_column_major(dtd.as_slice().to_vec(), n, n).unwrap()`.

No change to any constructor. The adapter is one line at the call site.

### Q-1b: Symmetry and FD Correctness

**Bspline penalty:** `integrate_symmetric_penalty` explicitly assigns `penalty[j + l*k] = val` AND `penalty[l + j*k] = val` for all `l >= j`. `[VERIFIED: fdars-core/src/smooth_basis.rs:1017-1028]`. Exact symmetry by construction.

**Fourier penalty:** Diagonal matrix (eigenvalue form). `[VERIFIED: fdars-core/src/smooth_basis.rs:159-188]`. Trivially symmetric.

**P-spline `DᵀD`:** `DᵀD` is symmetric by construction (equal to its own transpose). nalgebra multiplication preserves this.

**Implication for FD check:** The FD differentiator computes the actual `cᵀRc` regardless of symmetry. For symmetric R the analytic gradient is `2λRc`. The FD check verifies the autodiff gradient matches FD — the symmetry of R means analytic and FD gradients agree exactly (up to FD truncation error). Even if R were only approximately symmetric (numerical noise), the FD check would still pass because it compares the autodiff gradient of the actual function (which is `λ·cᵀRc` with whatever R was passed) against the FD of the same function. There is no risk here.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| FPCR score projection | Custom per-curve projection inside `predict_curve_generic` | `project_scores_generic` (regression.rs:232, Phase 94) | Already generic, tested, reusable — do not duplicate |
| Forward-mode gradient through prediction | Custom dual arithmetic | `Dual` from `autodiff/forward.rs` | Already implements all `Scalar` ops; just use `S` arithmetic |
| Reverse-mode gradient through prediction | Custom tape | `Var` + `vjp` from `autodiff/reverse.rs` | Tape already handles mul, add, sub, from_f64 |
| Penalty quadratic form | Custom `cᵀRc` for each penalty type | `penalty_value_generic` | One function handles all penalty types via `&FdMatrix` |
| Matrix adapter | Bespoke conversion for each penalty constructor | `FdMatrix::from_column_major(vec, k, k).unwrap()` | One-liner; no new type needed |

---

## Common Pitfalls

### Pitfall 1: Asserting Bit-Identity Between `predict_curve_generic::<f64>` and `predict_fregre_lm`

**What goes wrong:** Writing `assert_eq!(predict_curve_generic::<f64>(&curve, &fit), batch_pred[i])` or setting tolerance to 1e-12. The test fails intermittently even on benign inputs because the accumulation orders differ.

**Why it happens:** The batch kernel uses `((x-mean) * rotation) * weights`; `project_scores_generic` uses `(x-mean) * (rotation*weights)`. These are not equal in IEEE 754. See §Risk P-1a for the analysis.

**How to avoid:** Use `(result - expected).abs() < 1e-6` for any parity test comparing `predict_curve_generic::<f64>` to `predict_fregre_lm` outputs. This matches the tolerance of the existing `test_predict_fregre_lm_on_training_data` `[VERIFIED: fdars-core/src/scalar_on_function/tests.rs:117-127]`.

**Warning signs:** Test passes in debug mode, fails in release mode (or vice versa) due to different FMA/optimization behavior.

### Pitfall 2: Missing `use crate::autodiff::Scalar` Import

**What goes wrong:** Adding `<S: Scalar>` bounds to functions in `fregre_lm.rs` or `smooth_basis.rs` without importing `Scalar`. The files do not currently import it.

**Why it happens:** The files have no autodiff imports; adding a generic bound without the use-declaration gives `cannot find trait Scalar in scope`.

**How to avoid:** Add `use crate::autodiff::Scalar;` at the module top. Phase 95 established this pattern — see `helpers.rs:3` `[VERIFIED: fdars-core/src/regression.rs` imports confirm the crate-internal path is `crate::autodiff::Scalar`].

### Pitfall 3: Routing Gradient Through `lambda` or `R` in `penalty_value_generic`

**What goes wrong:** Writing `S::from_f64(penalty[(i,j)])` correctly but then computing `lambda` as `S::from_f64(lambda)` and multiplying at the end — which is correct — versus accidentally casting lambda to `S` inside the inner loop (creating unnecessary tape nodes for `Var`).

**Why it happens:** Natural instinct to wrap everything in `S`. The contract is: `coef` is `S`, everything else is f64 lifted once at the point of mixing.

**How to avoid:** Compute `S::from_f64(penalty[(i, j)])` once per `(i,j)` pair; compute `S::from_f64(lambda)` once outside the loops; multiply at the end. This is already the form in the recipe above.

### Pitfall 4: `predict_curve_generic` Using Wrong Coefficient Index

**What goes wrong:** Using `fit.coefficients[k]` (zero-indexed) instead of `fit.coefficients[1 + k]`. The `coefficients` vector has layout `[α, γ₁..γ_K, z₁..z_p]` where `α` is index 0 (intercept), `γ₁..γ_K` are indices `1..=ncomp` (FPC score coefficients), and `z` terms follow.

**Why it happens:** Off-by-one in the coefficient indexing.

**How to avoid:** Index as `coefficients[1 + k]` for `k` in `0..ncomp`. The `intercept` field is the separately stored `α` used to initialize `yhat`; it is not `coefficients[0]`. `[VERIFIED: fdars-core/src/scalar_on_function/mod.rs:86-90]` — `coefficients` stores the full regression coefficient vector on (FPC scores, scalar covariates); the intercept is stored separately in the `intercept` field.

### Pitfall 5: `penalty_value_generic` Called With Wrong K

**What goes wrong:** Passing `penalty: &FdMatrix` from `bspline_penalty_matrix(argvals, nbasis=15, ...)` but `coef: &[S]` of length 12 (because `actual_nbasis` after knot construction differs from the requested `nbasis`). The function uses `coef.len()` to bound the loop, which is correct — but the debug assertion `penalty.nrows() == k` fires.

**Why it happens:** `bspline_penalty_matrix` returns a penalty of size `actual_nbasis × actual_nbasis` where `actual_nbasis` may differ from the requested `nbasis` after knot construction (see `smooth_basis.rs:135: actual_nbasis = basis_fine.len() / n_quad`).

**How to avoid:** Always derive `k` from the returned penalty dimensions, not from the requested `nbasis`. For a coefficient vector produced by `smooth_basis`, use `coef.len()` which was set during fitting.

---

## Code Examples

### `predict_curve_generic` — Complete Implementation

```rust
// Source: recipe from fregre_lm.rs:454-471 [VERIFIED] + regression.rs:232-251 [VERIFIED]
use crate::autodiff::Scalar;
use crate::regression::project_scores_generic;
use crate::scalar_on_function::FregreLmResult;

#[must_use]
pub fn predict_curve_generic<S: Scalar>(curve: &[S], fit: &FregreLmResult) -> S {
    let scores = project_scores_generic(
        curve,
        &fit.fpca.mean,
        &fit.fpca.rotation,
        &fit.fpca.weights,
        fit.ncomp,
    );
    let mut yhat = S::from_f64(fit.intercept);
    for k in 0..fit.ncomp {
        yhat += S::from_f64(fit.coefficients[1 + k]) * scores[k];
    }
    yhat
}
```

### `penalty_value_generic` — Complete Implementation

```rust
// Source: recipe from smooth_basis.rs:1010-1028 [VERIFIED] + CONTEXT.md §DOP-03
use crate::autodiff::Scalar;
use crate::matrix::FdMatrix;

#[must_use]
pub fn penalty_value_generic<S: Scalar>(
    coef: &[S],
    penalty: &FdMatrix,
    lambda: f64,
) -> S {
    let k = coef.len();
    debug_assert_eq!(penalty.nrows(), k);
    let mut sum = S::zero();
    for i in 0..k {
        for j in 0..k {
            sum += coef[i] * S::from_f64(penalty[(i, j)]) * coef[j];
        }
    }
    S::from_f64(lambda) * sum
}
```

### Vec<f64> Penalty Adapter (for call sites)

```rust
// Wrap bspline_penalty_matrix / fourier_penalty_matrix output as FdMatrix:
let penalty_vec: Vec<f64> = bspline_penalty_matrix(&argvals, nbasis, 4, 2);
let k_actual = ((penalty_vec.len() as f64).sqrt() as usize);
let penalty_mat = FdMatrix::from_column_major(penalty_vec, k_actual, k_actual).unwrap();
// Then call:
let val = penalty_value_generic(&coef_s, &penalty_mat, lambda);
```

### DOP-02 f64-Parity Test (per-curve check)

```rust
// In scalar_on_function/tests.rs (add to existing #[cfg(test)] module)
#[test]
fn test_predict_curve_generic_f64_parity() {
    use crate::autodiff::Scalar;
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let batch_preds = predict_fregre_lm(&fit, &data, None);
    for i in 0..30 {
        let curve: Vec<f64> = (0..50).map(|j| data[(i, j)]).collect();
        let generic_pred = predict_curve_generic::<f64>(&curve, &fit);
        assert!(
            (generic_pred - batch_preds[i]).abs() < 1e-6,
            "predict_curve_generic::<f64> vs predict_fregre_lm parity failed \
             at curve {i}: generic={generic_pred}, batch={}", batch_preds[i]
        );
    }
}
```

**Note:** tolerance is 1e-6, NOT bit-identical. See §Risk P-1a.

### DOP-02 Dual FD Test Skeleton

```rust
// In scalar_on_function/tests.rs
#[test]
fn test_predict_curve_generic_dual_fd_check() {
    use crate::autodiff::{grad, Dual};
    let (data, y, _t) = generate_test_data(20, 30, 7);
    let fit = fregre_lm(&data, &y, None, 2).unwrap();
    let curve_f64: Vec<f64> = (0..30).map(|j| data[(0, j)]).collect();

    // Objective: scalar prediction value (already scalar — gradient is d(yhat)/d(x[j]))
    let obj = |curve: &[Dual]| -> Dual {
        predict_curve_generic(curve, &fit)
    };

    let (value, gradient) = grad(obj, &curve_f64);
    assert!(value.is_finite());
    assert_eq!(gradient.len(), 30);

    // Central FD check
    let h = 1e-6_f64;
    for j in 0..30 {
        let mut plus = curve_f64.clone();
        let mut minus = curve_f64.clone();
        plus[j] += h;
        minus[j] -= h;
        let fd = (predict_curve_generic::<f64>(&plus, &fit)
                 - predict_curve_generic::<f64>(&minus, &fit))
                 / (2.0 * h);
        let ad = gradient[j];
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (ad - fd).abs() < tol,
            "Dual grad[{j}]={ad} vs FD={fd} (tol={tol})"
        );
    }
}
```

### DOP-03 Dual FD Test Skeleton

```rust
// In smooth_basis.rs #[cfg(test)] tests module
#[test]
fn test_penalty_value_generic_dual_fd_check() {
    use crate::autodiff::{grad, Dual};
    let t = crate::test_helpers::uniform_grid(51);
    let penalty_vec = bspline_penalty_matrix(&t, 10, 4, 2);
    let k = ((penalty_vec.len() as f64).sqrt()) as usize;
    let penalty_mat = FdMatrix::from_column_major(penalty_vec, k, k).unwrap();
    let lambda = 0.5_f64;

    // Random-ish coef
    let coef_f64: Vec<f64> = (0..k).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();

    let obj = |coef: &[Dual]| -> Dual {
        penalty_value_generic(coef, &penalty_mat, lambda)
    };
    let (value, gradient) = grad(obj, &coef_f64);
    assert!(value.is_finite() && value >= 0.0);
    assert_eq!(gradient.len(), k);

    // Central FD check
    let h = 1e-6_f64;
    for i in 0..k {
        let mut plus = coef_f64.clone();
        let mut minus = coef_f64.clone();
        plus[i] += h;
        minus[i] -= h;
        let fd = (penalty_value_generic::<f64>(&plus, &penalty_mat, lambda)
                 - penalty_value_generic::<f64>(&minus, &penalty_mat, lambda))
                 / (2.0 * h);
        let ad = gradient[i];
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (ad - fd).abs() < tol,
            "Dual penalty grad[{i}]={ad} vs FD={fd} (tol={tol})"
        );
    }
}
```

### DOP-03 f64 Bit-Identity Parity Test

```rust
#[test]
fn test_penalty_value_generic_f64_parity() {
    let t = crate::test_helpers::uniform_grid(51);
    let penalty_vec = bspline_penalty_matrix(&t, 8, 4, 2);
    let k = ((penalty_vec.len() as f64).sqrt()) as usize;
    let penalty_mat = FdMatrix::from_column_major(penalty_vec.clone(), k, k).unwrap();
    let lambda = 2.0_f64;
    let coef: Vec<f64> = (0..k).map(|i| (i as f64 * 0.2 + 0.5).cos()).collect();

    // Reference: direct hand-computed λ·cᵀRc
    let mut reference = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            reference += coef[i] * penalty_vec[i + j * k] * coef[j];
        }
    }
    reference *= lambda;

    let generic = penalty_value_generic::<f64>(&coef, &penalty_mat, lambda);
    // Bit-identical: both compute the same floating-point expression
    assert!(
        (generic - reference).abs() < 1e-12,
        "penalty_value_generic::<f64> parity: generic={generic}, ref={reference}"
    );
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `predict_fregre_lm` batch only (f64) | + `predict_curve_generic<S>` generic per-curve | Phase 97 (this phase) | Enables AD-based gradient of prediction w.r.t. curve |
| No standalone penalty evaluation | + `penalty_value_generic<S>` | Phase 97 (this phase) | Enables AD-based gradient of penalty w.r.t. coefficients |

**No deprecated paths:** All existing public functions remain unchanged. The new generic fns are additive.

---

## Validation Architecture

> `workflow.nyquist_validation` not explicitly set to false in `.planning/config.json` — treated as enabled.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none |
| Quick run command (DOP-02) | `cargo test -p fdars-core scalar_on_function --features linalg 2>&1 \| tail -20` |
| Quick run command (DOP-03) | `cargo test -p fdars-core smooth_basis --features linalg 2>&1 \| tail -20` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1 \| tail -30` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DOP-02 | Existing `test_predict_fregre_lm_on_training_data` stays green (1e-6 tol) | regression guard | `cargo test -p fdars-core scalar_on_function::tests::test_predict_fregre_lm_on_training_data` | ✅ (existing) |
| DOP-02 | `predict_curve_generic::<f64>` matches `predict_fregre_lm` per-curve within 1e-6 | unit/parity | `cargo test -p fdars-core scalar_on_function::tests::test_predict_curve_generic_f64_parity` | ❌ Wave 0 |
| DOP-02 | `predict_curve_generic` gradient w.r.t. curve at `Dual`, FD-checked (tol `1e-6*(1+|fd|)`, h=1e-6) | unit/autodiff | `cargo test -p fdars-core scalar_on_function::tests::test_predict_curve_generic_dual_fd_check` | ❌ Wave 0 |
| DOP-02 | Same objective at `Var` (via `vjp`), FD-checked (same tol) | unit/autodiff | `cargo test -p fdars-core scalar_on_function::tests::test_predict_curve_generic_var_fd_check` | ❌ Wave 0 |
| DOP-03 | `penalty_value_generic::<f64>` matches direct `λ·cᵀRc` reference bit-identically (tol 1e-12) | unit/parity | `cargo test -p fdars-core smooth_basis::tests::test_penalty_value_generic_f64_parity` | ❌ Wave 0 |
| DOP-03 | `penalty_value_generic` gradient w.r.t. coef at `Dual`, FD-checked (tol `1e-6*(1+|fd|)`, h=1e-6) | unit/autodiff | `cargo test -p fdars-core smooth_basis::tests::test_penalty_value_generic_dual_fd_check` | ❌ Wave 0 |
| DOP-03 | Same objective at `Var` (via `vjp`), FD-checked (same tol) | unit/autodiff | `cargo test -p fdars-core smooth_basis::tests::test_penalty_value_generic_var_fd_check` | ❌ Wave 0 |
| DOP-02+03 | Full non-breaking: existing penalty-matrix tests + all scalar_on_function tests green | regression guard | `cargo test -p fdars-core --features linalg,parallel` | ✅ (existing suite) |
| DOP-02+03 | clippy `--all-targets` green (all new generics compile cleanly with inferred T=f64) | compile gate | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | ✅ (CI gate) |
| DOP-02+03 | `--features serde` build guard green | compile guard | `cargo build -p fdars-core --features serde` | ✅ (CI gate) |

### Tolerances (verbatim)

- **DOP-02 f64 parity:** `(predict_curve_generic::<f64>(...) - batch_pred).abs() < 1e-6` — NOT bit-identical (see §Risk P-1a).
- **DOP-03 f64 parity:** `(penalty_value_generic::<f64>(...) - reference).abs() < 1e-12` — bit-identical (new function, same FP expression).
- **Autodiff FD cross-check (both DOP-02 and DOP-03):** `(ad_grad - fd).abs() < 1e-6 * (1.0 + fd.abs())` — the absolute-floor robust form adopted in Phase 96 to avoid false failures at zero-gradient points. Central differences with `h = 1e-6`.

### Sampling Rate

- **Per task commit:** quick run command for the touched module (DOP-02 or DOP-03 scope)
- **Per wave merge:** full suite (`cargo test -p fdars-core --features linalg,parallel`)
- **Phase gate (before `/gsd-verify-work`):** full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo build --features serde` + `cargo fmt --check`

### Wave 0 Gaps (new tests — RED before GREEN)

- [ ] `scalar_on_function/tests.rs` — `test_predict_curve_generic_f64_parity`: per-curve f64 comparison, tol 1e-6
- [ ] `scalar_on_function/tests.rs` — `test_predict_curve_generic_dual_fd_check`: `Dual` FD gradient w.r.t. curve
- [ ] `scalar_on_function/tests.rs` — `test_predict_curve_generic_var_fd_check`: `Var` vjp FD gradient w.r.t. curve
- [ ] `smooth_basis.rs` `#[cfg(test)]` — `test_penalty_value_generic_f64_parity`: direct reference, tol 1e-12
- [ ] `smooth_basis.rs` `#[cfg(test)]` — `test_penalty_value_generic_dual_fd_check`: `Dual` FD gradient w.r.t. coef
- [ ] `smooth_basis.rs` `#[cfg(test)]` — `test_penalty_value_generic_var_fd_check`: `Var` vjp FD gradient w.r.t. coef

Existing tests are regression guards — they must continue to pass unchanged.

---

## Non-Breaking Proof

**callers of `predict_fregre_lm` that must stay green:**

1. `scalar_on_function/tests.rs:117` — `test_predict_fregre_lm_on_training_data` `[VERIFIED: fdars-core/src/scalar_on_function/tests.rs:117-127]`
2. `conformal/regression.rs:109,117` — `predict_fregre_lm(&refit, &cal_data, ...)` and `predict_fregre_lm(&refit, test_data, ...)` `[VERIFIED: fdars-core/src/conformal/regression.rs:109,117]`
3. `scalar_on_function/mod.rs:670-672` — `FregreLmResult::predict` method delegates to `predict_fregre_lm` `[VERIFIED: fdars-core/src/scalar_on_function/mod.rs:670-672]`

**If `predict_fregre_lm` is NOT refactored** (recommended): all three callers are unaffected — they call the unchanged function. Zero risk.

**If `predict_fregre_lm` IS refactored** to call `predict_curve_generic::<f64>` + scalar term: the batch output changes at ~1e-14 level. Callers 2 and 3 see this tiny change in predictions; caller 1 (`test_predict_fregre_lm_on_training_data`) still passes at 1e-6. The conformal regression caller only computes residuals and prediction intervals — sub-1e-14 change is inconsequential.

**Existing penalty-matrix tests** (`test_bspline_penalty_matrix_symmetric`, `test_bspline_penalty_matrix_positive_semidefinite`, `test_fourier_penalty_diagonal`) are unaffected — penalty constructors are not touched. `[VERIFIED: fdars-core/src/smooth_basis.rs:1318-1374]`

---

## Environment Availability

Step 2.6 SKIPPED — this phase is pure in-crate Rust code changes. No external dependencies beyond the Rust toolchain (1.97.0, MSRV 1.81).

---

## Runtime State Inventory

Skipped — additive generalization phase, not a rename/refactor. No stored data, live service config, OS-registered state, secrets, or build artifacts embed the new function names.

---

## Security Domain

In-crate numerical computation only. ASVS categories V2–V6 do not apply. The only correctness concern is FP parity, addressed by the validation architecture above.

---

## Open Questions

1. **Refactor `predict_fregre_lm` inner loop or not?**
   - What we know: the two accumulation orders produce results within ~1e-14; existing test passes at 1e-6.
   - What's unclear: the CONTEXT.md says "bit-for-bit" parity — which is not achievable via `project_scores_generic` composition due to the pre-folded `rotation*weights` product.
   - Recommendation: do NOT refactor the inner loop (Option 1). Keep both code paths independent. The planner should document this deviation from the CONTEXT.md "bit-for-bit" language and use "within 1e-6" instead.

2. **`predict_curve_generic` placement: fregre_lm.rs or new file?**
   - What we know: `predict_fregre_lm` is in `scalar_on_function/fregre_lm.rs`; the generic fn is tightly coupled to it.
   - Recommendation: same file (`fregre_lm.rs`). No new file needed.

3. **Re-export both fns from `lib.rs` and `prelude.rs`?**
   - CONTEXT.md says "prefer yes, additive, for Phase 99's unified API".
   - Recommendation: yes, add both to `lib.rs` and `prelude.rs` alongside `project_scores_generic`.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `FdMatrix::from_column_major` is the correct constructor for wrapping a `Vec<f64>` column-major penalty | §Q-1a (penalty adapter) | If the constructor has different semantics, the FdMatrix would have wrong layout; verify with a quick index-check test |
| A2 | The Rust compiler does not auto-fuse `(a * b) * c` and `a * (b * c)` using FMA without explicit intrinsics, so the two accumulation orders produce consistently different (not randomly different) results | §Risk P-1a | If FMA is applied in one but not the other path, the error bound could be larger; still within 1e-12, not a practical risk |

All other claims in this document are VERIFIED from source files read this session.

---

## Sources

### Primary (HIGH confidence — VERIFIED from source files read this session)

- `fdars-core/src/scalar_on_function/fregre_lm.rs:440-473` — `predict_fregre_lm` full signature and inner kernel (accumulation order)
- `fdars-core/src/scalar_on_function/mod.rs:62-98` — `FregreLmResult` fields (`intercept`, `fpca`, `coefficients`, `gamma`, `ncomp`)
- `fdars-core/src/regression.rs:130-251` — `FpcaResult::project_generic`, `project_scores_generic` full body
- `fdars-core/src/autodiff/mod.rs:38-97` — `Scalar` trait definition + re-exports
- `fdars-core/src/autodiff/forward.rs:100-130` — `f64::from_f64` identity implementation
- `fdars-core/src/smooth_basis.rs:1-30,100-188,1010-1028,1312-1374` — `bspline_penalty_matrix`, `fourier_penalty_matrix`, `integrate_symmetric_penalty` body, existing penalty tests
- `fdars-core/src/basis/pspline.rs:1-37` — `difference_matrix` returns `DMatrix<f64>`
- `fdars-core/src/scalar_on_function/tests.rs:100-128` — `test_predict_fregre_lm_on_training_data` at tol 1e-6
- `fdars-core/src/conformal/regression.rs:100-125` — two callers of `predict_fregre_lm` (non-breaking scope)
- `fdars-core/src/lib.rs:404,494-496,578-581` — existing re-exports (`predict_fregre_lm`, penalty constructors, `project_scores_generic`, autodiff types)
- `fdars-core/src/scalar_on_function/mod.rs:670-672` — `FregreLmResult::predict` method caller
- Phase 96 RESEARCH.md — validated patterns for Phase 95/96 FD tolerance form (`1e-6*(1.0+fd.abs())`) and import discipline

---

## Metadata

**Confidence breakdown:**
- `predict_curve_generic` implementation recipe: HIGH — read actual source; all fields verified
- Accumulation-order divergence analysis: HIGH — read both code paths; mathematical analysis confirmed
- `penalty_value_generic` implementation recipe: HIGH — read `integrate_symmetric_penalty`, verified exact symmetry
- Penalty-matrix carrier decision: HIGH — verified all three constructor return types
- Non-breaking caller list: HIGH — grep and source read confirmed
- Validation architecture: HIGH — mirrors Phase 96 tiered pattern, confirmed tolerances from source

**Research date:** 2026-09-11
**Valid until:** Indefinite (no external dependencies; all facts are in-repo constants)
