# Phase 97: Differentiable Regression Prediction & Smoothing Penalties — Pattern Map

**Mapped:** 2026-09-11
**Files analyzed:** 4 (2 impl files + 2 test locations)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/scalar_on_function/fregre_lm.rs` | service / utility | transform (generic lift) | `fdars-core/src/regression.rs` — `project_scores_generic` | exact (same Scalar pattern, same f64-lift idiom) |
| `fdars-core/src/smooth_basis.rs` | service / utility | transform (generic lift) | `fdars-core/src/regression.rs` — `project_scores_generic` | exact (same S::zero accumulator, S::from_f64 lift) |
| `fdars-core/src/scalar_on_function/tests.rs` | test | request-response | `fdars-core/src/basis/tests.rs` — Dual/Var FD tests (lines 663–781) | exact (same h=1e-6, same tol form, same vjp call) |
| `fdars-core/src/lib.rs` + `fdars-core/src/prelude.rs` | config / API surface | — | existing `project_scores_generic` / `predict_fregre_lm` re-exports | role-match (additive only) |

---

## Pattern Assignments

---

### `fdars-core/src/scalar_on_function/fregre_lm.rs` — Add `predict_curve_generic<S>`

**Analog:** `fdars-core/src/regression.rs` lines 232–251 (`project_scores_generic`)

**Role:** additive generic function beside `predict_fregre_lm`

#### Imports to add (at top of file, beside existing imports)

Current imports (lines 1–10 — do NOT remove any):
```rust
use super::{
    build_design_matrix, cholesky_factor, compute_beta_se, compute_fitted, compute_ols_std_errors,
    compute_r_squared, compute_xtx, ols_solve, recover_beta_t, resolve_ncomp,
    validate_fregre_inputs, FregreCvResult, FregreLmResult, ModelSelectionResult,
    SelectionCriterion,
};
use crate::cv::create_folds;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::fdata_to_pc;
```

Add these two lines (neither file currently imports them):
```rust
use crate::autodiff::Scalar;
use crate::regression::project_scores_generic;
```

**Pitfall:** `project_scores_generic` is re-exported from `lib.rs` at `fdars_core::project_scores_generic`, but the in-crate path is `crate::regression::project_scores_generic`. Use the `crate::regression::` path.

#### Core pattern — `project_scores_generic` (analog, regression.rs:232–251)

This is the PRIMARY template. The new `predict_curve_generic` wraps this function and adds the coefficient combination on top:

```rust
// fdars-core/src/regression.rs:228–251 — THE ANALOG (do not modify)
#[must_use]
pub fn project_scores_generic<S: Scalar>(
    curve: &[S],
    mean: &[f64],
    rotation: &FdMatrix,
    weights: &[f64],
    ncomp: usize,
) -> Vec<S> {
    let m = curve.len();
    let mut scores = vec![S::zero(); ncomp];
    for (k, score) in scores.iter_mut().enumerate() {
        let mut sum = S::zero();
        for j in 0..m {
            let centered = curve[j] - S::from_f64(mean[j]);
            let w_rot = S::from_f64(rotation[(j, k)] * weights[j]);  // f64 pre-multiply
            sum += centered * w_rot;
        }
        *score = sum;
    }
    scores
}
```

Key idiom: `S::from_f64(rotation[(j, k)] * weights[j])` — the two f64 factors are multiplied first in f64, then lifted once via `from_f64`. This is NOT the same as `((x-mean) * rotation) * weights` (three sequential multiplies). This pre-fold difference causes ~1e-14 divergence from the batch kernel — tolerated by the 1e-6 parity test.

#### New function to add (fregre_lm.rs, after `predict_fregre_lm` line 473)

```rust
/// Generic-over-[`Scalar`] per-curve prediction for a fitted functional linear model.
///
/// Computes `intercept + Σ_k coefficients[1+k] · score_k(curve)` where scores are
/// projected via [`project_scores_generic`]. Differentiable w.r.t. `curve`; all model
/// parameters (`intercept`, `coefficients`, fpca `mean`/`rotation`/`weights`) stay `f64`,
/// lifted via `S::from_f64` only at the point of mixing.
///
/// Instantiated at `S = f64` this reproduces [`predict_fregre_lm`]'s per-curve result
/// within ~1e-6 (NOT bit-identical — accumulation order differs; see module notes).
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

**Critical:** coefficient index is `fit.coefficients[1 + k]`, NOT `[k]`. Index 0 is the intercept stored separately in `fit.intercept` (`FregreLmResult.coefficients` layout: `[α, γ₁..γ_K, z₁..z_p]` — verified at `scalar_on_function/mod.rs:86–90`).

**`predict_fregre_lm` inner kernel reference** (lines 444–473, stays UNCHANGED — do NOT refactor):
```rust
// fregre_lm.rs:453–472 — f64 batch path, frozen
for i in 0..n_new {
    let mut yhat = fit.intercept;
    for k in 0..ncomp {
        let mut s = 0.0;
        for j in 0..m {
            s += (new_data[(i, j)] - fit.fpca.mean[j])
                * fit.fpca.rotation[(j, k)]
                * fit.fpca.weights[j];   // ((x-mean)*rot)*weights — 3 sequential f64 muls
        }
        yhat += fit.coefficients[1 + k] * s;
    }
    // ... scalar covariate additive terms ...
    predictions[i] = yhat;
}
```

---

### `fdars-core/src/smooth_basis.rs` — Add `penalty_value_generic<S>`

**Analog:** `fdars-core/src/regression.rs:240–248` (accumulator idiom in `project_scores_generic`)

**Role:** additive standalone generic function; placed after existing penalty-matrix constructors

#### Import to add (beside existing imports at top of smooth_basis.rs)

Current imports (lines 12–16):
```rust
use crate::basis::{bspline_basis, fourier_basis_with_period};
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use nalgebra::DMatrix;
use std::f64::consts::PI;
```

Add one line:
```rust
use crate::autodiff::Scalar;
```

#### Symmetry context — `integrate_symmetric_penalty` (smooth_basis.rs:1010–1028, stay f64)

The existing penalty-matrix builder that produces R:
```rust
// smooth_basis.rs:1017–1028 (do not modify)
let mut penalty = vec![0.0; k * k];
for j in 0..k {
    for l in j..k {
        let mut val = 0.0;
        for i in 0..n_quad {
            val += deriv_basis[i + j * n_quad] * deriv_basis[i + l * n_quad] * weights[i];
        }
        penalty[j + l * k] = val;
        penalty[l + j * k] = val;   // explicit mirror — exact symmetry
    }
}
```

This explicit mirroring guarantees exact symmetry. The analytic gradient `d(λ cᵀRc)/d(c_i) = 2λ(Rc)_i` is therefore exact.

#### New function to add (smooth_basis.rs, after `fourier_penalty_matrix` ~line 188)

```rust
/// Generic-over-[`Scalar`] roughness-penalty value `λ · cᵀRc`.
///
/// Evaluates `lambda * Σ_i Σ_j coef[i] * R[i,j] * coef[j]` where `R` (the
/// penalty matrix, e.g. from [`bspline_penalty_matrix`] or
/// [`fourier_penalty_matrix`]) stays `f64` (column-major K×K) and `lambda`
/// stays `f64`. Only `coef` (the spline coefficients) is the differentiable `S`
/// input.
///
/// Analytic gradient: `d(λ cᵀRc)/d(coef_i) = λ·((R + Rᵀ)c)_i = 2λ(Rc)_i`
/// for symmetric R. B-spline and Fourier penalty matrices are exactly symmetric
/// by construction ([`integrate_symmetric_penalty`] mirrors entries explicitly).
///
/// # Penalty-matrix adapter
/// Wrap [`bspline_penalty_matrix`] / [`fourier_penalty_matrix`] output (`Vec<f64>`) as:
/// `FdMatrix::from_column_major(penalty_vec, k, k).unwrap()`
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

**Accumulation idiom** (matches `project_scores_generic`):
- `S::zero()` as accumulator — same as regression.rs:240
- `S::from_f64(penalty[(i, j)])` — lift f64 R-value exactly once per (i,j) pair
- `S::from_f64(lambda)` computed ONCE outside the loops, multiplied at the very end
- The double loop is ordered `i` outer / `j` inner — stable for f64 parity

**FdMatrix column-major indexing:** `penalty[(i, j)]` accesses element at `i + j * nrows` — the crate's canonical `FdMatrix` operator, verified at `src/matrix.rs`.

**Adapter pattern for call sites** (no change to constructors):
```rust
let penalty_vec: Vec<f64> = bspline_penalty_matrix(&argvals, nbasis, 4, 2);
let k_actual = ((penalty_vec.len() as f64).sqrt()) as usize;
let penalty_mat = FdMatrix::from_column_major(penalty_vec, k_actual, k_actual).unwrap();
let val = penalty_value_generic(&coef_s, &penalty_mat, lambda);
```

---

### Tests — `fdars-core/src/scalar_on_function/tests.rs`

**Analog:** `fdars-core/src/basis/tests.rs:663–781` (Phase 96 Dual/Var FD test pair)

#### Test module imports (DOP-02 tests add to existing `#[cfg(test)]`)

```rust
use super::{predict_curve_generic, predict_fregre_lm};
use crate::autodiff::Dual;           // Dual FD test
use crate::autodiff::{vjp, Scalar, Var};  // Var FD test
```

#### Existing regression guard (line 117 — must stay green, do NOT touch)

```rust
// scalar_on_function/tests.rs:117–127 — frozen
#[test]
fn test_predict_fregre_lm_on_training_data() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let preds = predict_fregre_lm(&fit, &data, None);
    for i in 0..30 {
        assert!(
            (preds[i] - fit.fitted_values[i]).abs() < 1e-6,
            "Prediction on training data should match fitted values"
        );
    }
}
```

#### f64-parity test (DOP-02 — new, tolerance 1e-6, NOT bit-identical)

```rust
#[test]
fn test_predict_curve_generic_f64_parity() {
    let (data, y, _t) = generate_test_data(30, 50, 42);
    let fit = fregre_lm(&data, &y, None, 3).unwrap();
    let batch_preds = predict_fregre_lm(&fit, &data, None);
    for i in 0..30 {
        let curve: Vec<f64> = (0..50).map(|j| data[(i, j)]).collect();
        let generic_pred = predict_curve_generic::<f64>(&curve, &fit);
        assert!(
            (generic_pred - batch_preds[i]).abs() < 1e-6,
            "predict_curve_generic::<f64> vs predict_fregre_lm at curve {i}: \
             generic={generic_pred}, batch={}", batch_preds[i]
        );
    }
}
```

**CRITICAL:** tolerance is 1e-6 (NOT assert_eq!, NOT 1e-12). Accumulation order divergence explained in RESEARCH.md §Risk P-1a.

#### Dual FD test (DOP-02 — new, mirrors basis/tests.rs:663–721)

```rust
#[test]
fn test_predict_curve_generic_dual_fd_check() {
    use crate::autodiff::Dual;
    let (data, y, _t) = generate_test_data(20, 30, 7);
    let fit = fregre_lm(&data, &y, None, 2).unwrap();
    let curve_f64: Vec<f64> = (0..30).map(|j| data[(0, j)]).collect();
    let h = 1e-6_f64;

    // Compute gradient via Dual: one seed per coordinate
    for seed_idx in 0..30 {
        let curve_dual: Vec<Dual> = curve_f64
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == seed_idx { Dual::seed(v) } else { Dual::constant(v) })
            .collect();
        let result = predict_curve_generic::<Dual>(&curve_dual, &fit);
        let (_, tangent) = result.extract();

        let mut plus = curve_f64.clone();
        let mut minus = curve_f64.clone();
        plus[seed_idx] += h;
        minus[seed_idx] -= h;
        let fd = (predict_curve_generic::<f64>(&plus, &fit)
                 - predict_curve_generic::<f64>(&minus, &fit))
                 / (2.0 * h);
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (tangent - fd).abs() <= tol,
            "Dual grad[{seed_idx}]={tangent} vs FD={fd} (tol={tol})"
        );
    }
}
```

**Dual idiom (from basis/tests.rs:688–706):** `Dual::seed(v)` for the active coordinate, `Dual::constant(v)` for all others, then `.extract()` → `(value, tangent)`.

#### Var FD test (DOP-02 — new, mirrors basis/tests.rs:727–781)

```rust
#[test]
fn test_predict_curve_generic_var_fd_check() {
    use crate::autodiff::{vjp, Scalar, Var};
    let (data, y, _t) = generate_test_data(20, 30, 7);
    let fit = fregre_lm(&data, &y, None, 2).unwrap();
    let curve_f64: Vec<f64> = (0..30).map(|j| data[(0, j)]).collect();
    let h = 1e-6_f64;

    let (_, grad) = vjp(
        |curve: &[Var]| predict_curve_generic::<Var>(curve, &fit),
        &curve_f64,
    );
    for j in 0..30 {
        let mut plus = curve_f64.clone();
        let mut minus = curve_f64.clone();
        plus[j] += h;
        minus[j] -= h;
        let fd = (predict_curve_generic::<f64>(&plus, &fit)
                 - predict_curve_generic::<f64>(&minus, &fit))
                 / (2.0 * h);
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (grad[j] - fd).abs() <= tol,
            "Var grad[{j}]={} vs FD={fd} (tol={tol})", grad[j]
        );
    }
}
```

**Var/vjp idiom (from basis/tests.rs:752–763):** `vjp(|x: &[Var]| { ... scalar output ... }, &x_f64)` returns `(value, grad_vec)` directly.

---

### Tests — `fdars-core/src/smooth_basis.rs` `#[cfg(test)]` module

**Analog:** `fdars-core/src/basis/tests.rs:663–781` + `smooth_basis.rs:1318–1374` (existing penalty-matrix tests)

#### Existing penalty-matrix tests (lines 1318+ — frozen, do NOT touch)

These test symmetry and positive-semidefiniteness of `bspline_penalty_matrix` / `fourier_penalty_matrix`. They do not involve `penalty_value_generic`.

#### f64 parity test (DOP-03 — new, bit-identical / tol 1e-12)

```rust
#[test]
fn test_penalty_value_generic_f64_parity() {
    let t = crate::test_helpers::uniform_grid(51);
    let penalty_vec = bspline_penalty_matrix(&t, 8, 4, 2);
    let k = ((penalty_vec.len() as f64).sqrt()) as usize;
    let penalty_mat = FdMatrix::from_column_major(penalty_vec.clone(), k, k).unwrap();
    let lambda = 2.0_f64;
    let coef: Vec<f64> = (0..k).map(|i| (i as f64 * 0.2 + 0.5).cos()).collect();

    // Reference: direct hand-computed λ·cᵀRc (column-major indexing: R[i+j*k])
    let mut reference = 0.0_f64;
    for i in 0..k {
        for j in 0..k {
            reference += coef[i] * penalty_vec[i + j * k] * coef[j];
        }
    }
    reference *= lambda;

    let generic = penalty_value_generic::<f64>(&coef, &penalty_mat, lambda);
    assert!(
        (generic - reference).abs() < 1e-12,
        "penalty_value_generic::<f64> parity: got={generic}, ref={reference}"
    );
}
```

**Why bit-identical:** `f64::from_f64(v) == v` is the identity; `FdMatrix::from_column_major` stores the same data column-major; both loops iterate in the same `i`-outer/`j`-inner order. The two expressions are the same FP computation.

#### Dual FD test (DOP-03 — new)

```rust
#[test]
fn test_penalty_value_generic_dual_fd_check() {
    use crate::autodiff::Dual;
    let t = crate::test_helpers::uniform_grid(51);
    let penalty_vec = bspline_penalty_matrix(&t, 10, 4, 2);
    let k = ((penalty_vec.len() as f64).sqrt()) as usize;
    let penalty_mat = FdMatrix::from_column_major(penalty_vec, k, k).unwrap();
    let lambda = 0.5_f64;
    let coef_f64: Vec<f64> = (0..k).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
    let h = 1e-6_f64;

    for seed_idx in 0..k {
        let coef_dual: Vec<Dual> = coef_f64
            .iter()
            .enumerate()
            .map(|(i, &v)| if i == seed_idx { Dual::seed(v) } else { Dual::constant(v) })
            .collect();
        let result = penalty_value_generic::<Dual>(&coef_dual, &penalty_mat, lambda);
        let (_, tangent) = result.extract();

        let mut plus = coef_f64.clone();
        let mut minus = coef_f64.clone();
        plus[seed_idx] += h;
        minus[seed_idx] -= h;
        let fd = (penalty_value_generic::<f64>(&plus, &penalty_mat, lambda)
                 - penalty_value_generic::<f64>(&minus, &penalty_mat, lambda))
                 / (2.0 * h);
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (tangent - fd).abs() <= tol,
            "Dual penalty grad[{seed_idx}]={tangent} vs FD={fd} (tol={tol})"
        );
    }
}
```

#### Var FD test (DOP-03 — new)

```rust
#[test]
fn test_penalty_value_generic_var_fd_check() {
    use crate::autodiff::{vjp, Var};
    let t = crate::test_helpers::uniform_grid(51);
    let penalty_vec = bspline_penalty_matrix(&t, 10, 4, 2);
    let k = ((penalty_vec.len() as f64).sqrt()) as usize;
    let penalty_mat = FdMatrix::from_column_major(penalty_vec, k, k).unwrap();
    let lambda = 0.5_f64;
    let coef_f64: Vec<f64> = (0..k).map(|i| (i as f64 * 0.3 + 0.1).sin()).collect();
    let h = 1e-6_f64;

    let pm_clone = penalty_mat.clone();
    let (_, grad) = vjp(
        |coef: &[Var]| penalty_value_generic::<Var>(coef, &pm_clone, lambda),
        &coef_f64,
    );
    for i in 0..k {
        let mut plus = coef_f64.clone();
        let mut minus = coef_f64.clone();
        plus[i] += h;
        minus[i] -= h;
        let fd = (penalty_value_generic::<f64>(&plus, &penalty_mat, lambda)
                 - penalty_value_generic::<f64>(&minus, &penalty_mat, lambda))
                 / (2.0 * h);
        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (grad[i] - fd).abs() <= tol,
            "Var penalty grad[{i}]={} vs FD={fd} (tol={tol})", grad[i]
        );
    }
}
```

---

### `fdars-core/src/lib.rs` — Additive re-exports

**Analog:** line 578 (`project_scores_generic` re-export) and lines 404–413 (`predict_fregre_lm` re-export block)

Add `predict_curve_generic` alongside `predict_fregre_lm` at line ~404:
```rust
// Current (lib.rs:404):
    permutation_test_fam, predict_fregre_lm, predict_fregre_lm_multi, predict_fregre_np,
// New (add predict_curve_generic to this list):
    permutation_test_fam, predict_curve_generic, predict_fregre_lm, predict_fregre_lm_multi,
```

Add `penalty_value_generic` alongside `bspline_penalty_matrix` at line ~495:
```rust
// Current (lib.rs:495):
    basis_nbasis_cv, basis_nbasis_cv_with_config, bspline_penalty_matrix, fourier_penalty_matrix,
// New:
    basis_nbasis_cv, basis_nbasis_cv_with_config, bspline_penalty_matrix, fourier_penalty_matrix,
    penalty_value_generic,
```

(Or on the same line alphabetically — the planner may choose the exact position.)

### `fdars-core/src/prelude.rs` — Additive re-exports

**Analog:** line 18 (`project_scores_generic` re-export)

```rust
// Current (prelude.rs:18):
pub use crate::regression::{project_scores_generic, FpcaResult, PlsResult};
// Add alongside scalar_on_function and smooth_basis re-exports:
pub use crate::scalar_on_function::predict_curve_generic;
pub use crate::smooth_basis::penalty_value_generic;
```

---

## Shared Patterns

### Scalar Trait Import (applies to both new fns)

**Source:** `fdars-core/src/regression.rs:1` (implicit via `crate::autodiff::Scalar`)
**Apply to:** `fregre_lm.rs` (add), `smooth_basis.rs` (add)

```rust
use crate::autodiff::Scalar;
```

### S::zero Accumulator + S::from_f64 Lift (applies to both new fns)

**Source:** `fdars-core/src/regression.rs:240–246`
**Apply to:** `predict_curve_generic` (scores already computed by `project_scores_generic`; the coefficient loop uses `S::from_f64`), `penalty_value_generic` (double loop)

Pattern:
```rust
let mut acc = S::zero();
// ... inner loop ...
acc += lhs * S::from_f64(f64_constant) * rhs;
// Multiply f64 scalar outside all loops:
S::from_f64(scalar_f64) * acc
```

### Dual Seed/Constant + .extract() Pattern

**Source:** `fdars-core/src/basis/tests.rs:688–706`
**Apply to:** All Dual FD tests (DOP-02 and DOP-03)

```rust
let dual_input: Vec<Dual> = x_f64
    .iter()
    .enumerate()
    .map(|(i, &v)| if i == seed_idx { Dual::seed(v) } else { Dual::constant(v) })
    .collect();
let result = fn_under_test(&dual_input, ...);
let (_, tangent) = result.extract();
```

### vjp Call Pattern

**Source:** `fdars-core/src/basis/tests.rs:752–763`
**Apply to:** All Var FD tests (DOP-02 and DOP-03)

```rust
let (_, grad) = vjp(
    |x: &[Var]| fn_under_test::<Var>(x, &other_captured_args),
    &x_f64,
);
```

Note: any non-`S` captures inside the closure must be cloned/owned (e.g. `pm_clone = penalty_mat.clone()`) if they need to satisfy `'static` or ownership requirements.

### Robust FD Tolerance

**Source:** `fdars-core/src/basis/tests.rs:715`
**Apply to:** ALL autodiff FD assertions in this phase

```rust
let tol = 1e-6 * (1.0 + fd.abs());
assert!((ad - fd).abs() <= tol, "...");
```

Never use `assert_eq!` or fixed absolute tolerances for FD checks.

---

## FregreLmResult Field Reference

**Source:** `fdars-core/src/scalar_on_function/mod.rs:62–98`

| Field | Type | Use in DOP-02 |
|-------|------|---------------|
| `intercept` | `f64` | `S::from_f64(fit.intercept)` → initialize `yhat` |
| `fpca.mean` | `Vec<f64>` | passed to `project_scores_generic` |
| `fpca.rotation` | `FdMatrix` | passed to `project_scores_generic` |
| `fpca.weights` | `Vec<f64>` | passed to `project_scores_generic` |
| `ncomp` | `usize` | bounds the `k` loop |
| `coefficients` | `Vec<f64>` | indexed as `coefficients[1 + k]` for k in `0..ncomp` |
| `gamma` | `Vec<f64>` | NOT used in `predict_curve_generic` (scalar covariates excluded) |

---

## No Analog Found

None. All new functions have direct structural analogs.

---

## Metadata

**Analog search scope:** `fdars-core/src/` — `regression.rs`, `scalar_on_function/`, `smooth_basis.rs`, `basis/tests.rs`, `autodiff/`
**Files scanned:** 8
**Pattern extraction date:** 2026-09-11
