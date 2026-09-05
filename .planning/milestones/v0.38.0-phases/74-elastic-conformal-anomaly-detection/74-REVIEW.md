---
phase: 74-elastic-conformal-anomaly-detection
reviewed: 2026-09-05T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - fdars-core/src/tolerance/conformal_anomaly.rs
  - fdars-core/src/tolerance/types.rs
  - fdars-core/src/tolerance/conformal.rs
  - fdars-core/src/tolerance/mod.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
findings:
  critical: 1
  warning: 1
  info: 2
  total: 4
status: issues_found
---

# Phase 74: Code Review Report

**Reviewed:** 2026-09-05
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 74 adds inductive conformal anomaly detection via elastic distances to the `tolerance`
module. The algorithmic core is sound: the p-value formula is correct, the threshold order
statistic matches the existing `conformal_quantile` convention, the `conformal_prediction_band`
guard is correctly placed, both `unreachable!()` arms are genuinely unreachable, and the
`CombinedElastic = sqrt(amp² + phase²)` formula is a genuine combination rather than an alias.
Re-exports in `mod.rs`, `lib.rs`, and `prelude.rs` are additive and non-breaking.

One CRITICAL defect was found: the caller-supplied template path in
`elastic_conformal_anomaly` does not validate the template length against `argvals.len()`. A
mismatched template silently propagates through `srsf_transform` (which returns a zero matrix
on length mismatch instead of an error), producing garbage scores with no `FdarError` raised.

One WARNING was found: the `unwrap_or(f64::NAN)` fallback for calibration and test scores —
while guarded at the variant-validation level — leaves a silent NaN-propagation path open if
any elastic distance call fails. NaN in calibration scores produces a NaN threshold, which
causes all test curves to be silently classified as non-anomalous regardless of their actual
scores, violating the marginal validity guarantee with no error signal.

---

## Critical Issues

### CR-01: Caller-Supplied Template Length Not Validated

**File:** `fdars-core/src/tolerance/conformal_anomaly.rs:279-291`

**Issue:** When `config.template = Some(t)` is provided, the template vector is used
directly without checking that `t.len() == argvals.len()`. The downstream elastic distance
functions call `srsf_single(template, argvals)`, which calls `srsf_transform`. That function
(in `alignment/srsf.rs:38`) silently returns `FdMatrix::zeros(n, m)` when
`argvals.len() != template.len()` — no panic, no `FdarError`. The DP alignment then runs
against a zero SRSF, producing scores on a completely wrong scale that are not caught by any
error handler. This violates the crate's dimension-check-at-entry-point invariant and the
documented behavior that `elastic_conformal_anomaly` returns `FdarError::InvalidDimension`
for dimension mismatches.

```rust
// current (conformal_anomaly.rs:279-291)
let template: Vec<f64> = match config.template.clone() {
    Some(t) => t,                  // ← no length check; t.len() may != m
    None => {
        let km = crate::alignment::karcher_mean(
            calibration, argvals, config.max_iter, config.tol, config.lambda,
        );
        km.mean
    }
};
```

**Fix:** Add a length guard in the `Some(t)` arm:

```rust
let template: Vec<f64> = match config.template.clone() {
    Some(t) => {
        if t.len() != m {
            return Err(FdarError::InvalidDimension {
                parameter: "config.template",
                expected: format!("{m} (argvals.len())"),
                actual: format!("{}", t.len()),
            });
        }
        t
    }
    None => {
        let km = crate::alignment::karcher_mean(
            calibration, argvals, config.max_iter, config.tol, config.lambda,
        );
        km.mean
    }
};
```

---

## Warnings

### WR-01: NaN Fallback Silently Suppresses Anomaly Flags

**File:** `fdars-core/src/tolerance/conformal_anomaly.rs:296-299, 314-316`

**Issue:** Calibration and test scores use `.unwrap_or(f64::NAN)` as the fallback for
`elastic_nonconformity`. Because `config.variant` is validated to be an elastic variant before
this point (lines 264-276), the `Err` path of `elastic_nonconformity` is unreachable in
practice. However, the NaN fallback disguises the true issue and, more importantly, creates a
fragile contract: if the fallback is ever reached (e.g., after a future refactor that removes
or changes the upstream variant guard), NaN values silently enter `calib_scores`.

When NaN appears in `calib_scores`:

1. `sort_nan_safe` places NaN at an undefined position (it treats NaN as `Equal` to neighbors,
   giving implementation-defined sort order).
2. `calibrated_threshold` may return `NaN` if the k-th element is NaN.
3. All subsequent comparisons `a_star >= NaN` and `a_star > NaN` are `false` (IEEE 754), so
   every test curve gets `flag = false` — silent miss of all anomalies with no error signal.

The variant-validation check at lines 264-276 makes this unreachable today, but the defense
should be explicit. Use the `?` operator to propagate errors instead of silently swallowing
them into NaN:

```rust
// current (calibration loop, line 296-299)
let calib_scores: Vec<f64> = (0..n_calib)
    .map(|i| {
        let curve = calibration.row(i);
        elastic_nonconformity(&curve, &template, argvals, config.lambda, config.variant)
            .unwrap_or(f64::NAN)   // ← NaN silently contaminates threshold
    })
    .collect();
```

```rust
// recommended: propagate errors via collect::<Result<Vec<_>, _>>()
let calib_scores: Result<Vec<f64>, FdarError> = (0..n_calib)
    .map(|i| {
        let curve = calibration.row(i);
        elastic_nonconformity(&curve, &template, argvals, config.lambda, config.variant)
    })
    .collect();
let calib_scores = calib_scores?;

// and similarly for the test loop:
let a_star =
    elastic_nonconformity(&curve, &template, argvals, config.lambda, config.variant)?;
```

This makes any future failure visible rather than silently corrupting the anomaly flags.

---

## Info

### IN-01: `elastic_nonconformity` Missing `#[must_use]`

**File:** `fdars-core/src/tolerance/conformal_anomaly.rs:157`

**Issue:** `elastic_nonconformity` wraps the three elastic distance functions
(`amplitude_distance`, `phase_distance_pair`, `elastic_distance`), all of which are marked
`#[must_use = "expensive computation whose result should not be discarded"]` in
`alignment/pairwise.rs`. The wrapper is itself an expensive computation (it performs a full
DP elastic alignment) and returns a `Result<f64>`. Per crate convention (74+ functions marked
`#[must_use]`), it should carry the attribute for consistency and to prevent accidentally
discarding the result.

**Fix:**

```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_nonconformity(
    curve: &[f64],
    template: &[f64],
    argvals: &[f64],
    lambda: f64,
    variant: NonConformityScore,
) -> Result<f64, FdarError> {
```

### IN-02: Test Module Defines `uniform_grid` Locally Instead of Using `crate::test_helpers`

**File:** `fdars-core/src/tolerance/conformal_anomaly.rs:361-363`

**Issue:** The inline test module defines a local `uniform_grid` helper (identical to the
crate-shared one in `src/test_helpers.rs`). The sibling file `tolerance/tests.rs:5` already
uses `crate::test_helpers::uniform_grid`. Defining a private duplicate is inconsistent with
the tolerance module's own convention.

**Fix:** Remove the local definition and import from `test_helpers`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;    // ← add this, remove local fn
    // ... remove the local `fn uniform_grid` definition ...
}
```

---

_Reviewed: 2026-09-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
