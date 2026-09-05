---
phase: 73-veesa-explainability-pipeline-integration
reviewed: 2026-09-05T00:00:00Z
depth: standard
files_reviewed: 4
files_reviewed_list:
  - fdars-core/src/elastic_pfi.rs
  - fdars-core/src/jfpca_model.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 0
  warning: 4
  info: 2
  total: 6
status: issues_found
---

# Phase 73: Code Review Report

**Reviewed:** 2026-09-05
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

## Summary

Phase 73 adds the VEESA explainability layer: model-agnostic PFI (`elastic_pfi`), principal-direction reconstruction (`JfpcaModel::principal_directions`), and the `veesa_pipeline` convenience wrapper. The additive-only mandate is met — no existing public signatures are changed.

The core algorithmic choices are correct: single advancing `StdRng`, reuse of `shuffle_global`/`clone_scores_matrix`, `sigma_j = eigenvalues[j].sqrt()`, and the SUMMARY's auto-fix of using `mean_srsf` (not `mean_q[0..m]`) as the reconstruction base, which achieves the c=0 gate at < 1e-10. Validation at entry (dimension checks, `n_repeats >= 1`, `pc_index < ncomp`, `c_values` non-empty) is present.

Four warnings are raised. No correctness blockers were found, but the MSE sign-convention documentation is actively misleading (WR-02) and the missing derives on `PfiMetric` break the "all public types derive Debug/Clone/PartialEq" project invariant (WR-01).

## Structural Findings (fallow)

No structural pre-pass was provided.

## Narrative Findings (AI reviewer)

## Critical Issues

None.

## Warnings

### WR-01: `PfiMetric` missing `Debug`, `Clone`, and `PartialEq` derives

**File:** `fdars-core/src/elastic_pfi.rs:55-73`
**Issue:** The project convention (CLAUDE.md, Code Style section) mandates `#[derive(Debug, Clone, PartialEq)]` on all public types. `PfiMetric` is public but carries none of these derives because the `Custom(Box<dyn Fn>)` variant prevents auto-derive. No manual impls are provided either. This means downstream code (tests, user code, other crate modules) cannot print, clone, or compare `PfiMetric` values. Of the three: `Clone` is the most operationally painful (cannot clone a `PfiMetric::Mse` reference to pass to a helper without moving it). `PartialEq` absence blocks `assert_eq!` on enum values in tests. `Debug` absence blocks format strings and `{:?}` printing of any struct that embeds `PfiMetric`.

The doc comment notes only the `serde` omission; the derive omission is undocumented.

**Fix:** Add manual partial impls for the variants that support them:

```rust
impl std::fmt::Debug for PfiMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PfiMetric::Mse => write!(f, "PfiMetric::Mse"),
            PfiMetric::Mae => write!(f, "PfiMetric::Mae"),
            PfiMetric::Accuracy => write!(f, "PfiMetric::Accuracy"),
            PfiMetric::Custom(_) => write!(f, "PfiMetric::Custom(<fn>)"),
        }
    }
}

impl Clone for PfiMetric {
    fn clone(&self) -> Self {
        match self {
            PfiMetric::Mse => PfiMetric::Mse,
            PfiMetric::Mae => PfiMetric::Mae,
            PfiMetric::Accuracy => PfiMetric::Accuracy,
            PfiMetric::Custom(_) => {
                // Box<dyn Fn> is not Clone; panic or omit as needed.
                // Document that Custom cannot be cloned.
                unimplemented!("PfiMetric::Custom cannot be cloned (Box<dyn Fn> is not Clone)")
            }
        }
    }
}

impl PartialEq for PfiMetric {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (PfiMetric::Mse, PfiMetric::Mse)
                | (PfiMetric::Mae, PfiMetric::Mae)
                | (PfiMetric::Accuracy, PfiMetric::Accuracy)
        )
        // Custom variants are never equal (closures cannot be compared)
    }
}
```

Update the doc note from mentioning only serde to also mention derive limitations:
```rust
/// Note: `PfiMetric` does NOT derive `Debug`, `Clone`, `PartialEq`, or `serde` because
/// the `Custom` variant contains a `Box<dyn Fn>`. Manual impls are provided for `Debug`
/// and `PartialEq` (Mse/Mae/Accuracy variants only). `Custom` cannot be cloned.
```

---

### WR-02: `PfiMetric::Mse` doc comment has wrong sign direction — will actively mislead users

**File:** `fdars-core/src/elastic_pfi.rs:59-65`
**Issue:** The `PfiMetric::Mse` variant doc states:

> "Importance is computed as `baseline_mse - permuted_mse`, so a **positive importance** means permuting the component **increases MSE** (i.e., the component matters)."

This is mathematically backwards. With MSE (lower = better):
- For an informative PC: permuting raises MSE, so `permuted_mse > baseline_mse`
- `importance = baseline_mse - permuted_mse < 0` (NEGATIVE)

A positive importance with MSE means `baseline_mse > permuted_mse`, i.e., permuting **decreases** MSE — i.e., the PC was hurting the model. An informative PC always yields **negative** importance with `PfiMetric::Mse`.

This is compounded by the `ElasticPfiResult.importance` field doc saying "Positive = informative", which is correct for `Accuracy`/`Custom(higher-is-better)` but wrong for `Mse`.

The SUMMARY's own description of the known-signal test failure confirms this: "For `PfiMetric::Mse`, `importance = baseline - permuted_mean < 0` for informative PCs." The implementation is correct; the doc is wrong.

**Fix:** Correct the `Mse` variant doc:

```rust
/// Mean squared error. **Lower values = better prediction.**
///
/// Because MSE is a loss (not a score), `importance = baseline_mse - permuted_mse` will
/// be **negative** for an informative component (permuting a useful PC raises MSE, so
/// `permuted_mse > baseline_mse`).
///
/// For intuitive positive-importance interpretation, either negate MSE inside a
/// `Custom` closure or use `Accuracy` for classification tasks.
Mse,
```

Also fix `ElasticPfiResult.importance` to note the sign depends on the metric:

```rust
/// Metric change per PC (length ncomp).
/// For higher-is-better metrics (Accuracy, Custom): positive = informative PC.
/// For lower-is-better metrics (Mse, Mae): negative = informative PC.
pub importance: Vec<f64>,
```

---

### WR-03: `JfpcaModel::principal_directions` missing `#[must_use]`

**File:** `fdars-core/src/jfpca_model.rs:428`
**Issue:** The project convention marks `#[must_use]` on all expensive computations. `principal_directions` runs `n_c` SRSF inversions and `n_c` sphere exponentiations in a loop — clearly expensive. All three other public methods on `JfpcaModel` (`jfpca_fit`, `transform`, `score_training`) are annotated with `#[must_use]`. `principal_directions` is the only one missing it.

**Fix:**

```rust
#[must_use = "expensive computation: principal_directions returns amplitude and phase curves; use the result"]
pub fn principal_directions(
    &self,
    pc_index: usize,
    c_values: &[f64],
) -> Result<PrincipalDirections, FdarError> {
```

---

### WR-04: `compute_metric` silently truncates if `predict()` returns wrong-length vector

**File:** `fdars-core/src/elastic_pfi.rs:109-138`
**Issue:** `compute_metric` uses `y.iter().zip(pred.iter())` and divides by `n = y.len()`. If the caller's `predict` closure returns a `Vec<f64>` shorter than `n`, the `zip` silently iterates only `pred.len()` pairs but divides by the full `n`. This produces a wrong (deflated) metric — systematically smaller than the true value — without any error.

The bug path: `elastic_pfi` validates `scores.nrows() == y.len()`, but does not validate that `predict(scores).len() == n`. A caller whose closure computes `(0..8).map(...)` but is passed to a pipeline where `n=10` will silently get wrong importance values.

The same closure is used for both the baseline and all permuted evaluations, so the error is consistent in direction but the magnitude of `importance` values is incorrect.

**Fix:** After computing `baseline_pred` and inside the repeat loop, validate prediction length:

```rust
let baseline_pred = predict(scores);
if baseline_pred.len() != n {
    return Err(FdarError::InvalidDimension {
        parameter: "predict(scores)",
        expected: format!("length {} (== scores.nrows())", n),
        actual: format!("length {}", baseline_pred.len()),
    });
}
let baseline_metric = compute_metric(y, &baseline_pred, metric);

// ... inside repeat loop:
let pred = predict(&perm);
if pred.len() != n {
    return Err(FdarError::InvalidDimension {
        parameter: "predict(perm)",
        expected: format!("length {} (== scores.nrows())", n),
        actual: format!("length {}", pred.len()),
    });
}
```

Alternatively, validate only the baseline (closure contract), adding a `debug_assert!` inside the loop to avoid overhead in the hot path.

---

## Info

### IN-01: `principal_directions_sigma_sqrt_scaling` test uses augmented-dim index in `max_vert`

**File:** `fdars-core/src/jfpca_model.rs:867-870`
**Issue:** The sigma-scaling gate computes:

```rust
let max_vert = (0..=m)   // includes index m (augmented dimension)
    .map(|l| model.vert_component[(pc_index, l)].abs())
    .fold(0.0f64, f64::max);
```

But the amplitude perturbation in `principal_directions` uses only `l in 0..m` (not the augmented element at `m`). Including the augmented element in `max_vert` can make `expected_sqrt_scale` larger than the actual perturbation, giving the test more slack and potentially allowing the raw-eigenvalue bug to slip through in cases where the augmented element is the column maximum. The test still provides useful signal, but uses `0..m` (matching the actual perturbation range) for the tightest possible gate.

**Fix:**

```rust
let max_vert = (0..m)   // match principal_directions perturbation range (not augmented)
    .map(|l| model.vert_component[(pc_index, l)].abs())
    .fold(0.0f64, f64::max);
```

---

### IN-02: `eigenvalues[pc_index].sqrt()` not guarded against numerically negative eigenvalues

**File:** `fdars-core/src/jfpca_model.rs:449`
**Issue:** Eigenvalues come from `svd_scores_and_eigenvalues` as `sv^2 / (n-1)`, so they are theoretically non-negative. However, SVD intermediate floating-point rounding can produce values like `-1e-16`. `(-1e-16_f64).sqrt()` is `NaN`, which then silently propagates into all amplitude and phase curves, producing `NaN`-valued output with no error.

This is low probability but undetectable since `principal_directions` does not validate eigenvalue non-negativity.

**Fix:** Clamp before sqrt:

```rust
// Clamp to 0.0 to guard against floating-point rounding near zero.
let sigma_j = self.eigenvalues[pc_index].max(0.0).sqrt();
```

---

_Reviewed: 2026-09-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
